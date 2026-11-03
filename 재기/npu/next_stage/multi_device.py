"""Independent GL tiles distributed over 1/2/4 NPU compute workers.

The memory tile splits adjacent input frames and joins outputs in the same order.
The total host buffer size and arithmetic are identical for all worker counts.
"""
import time
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from native_post import HERE, ROOT
from gl_aie.runtime import Device


def make_parallel_kernel(kind):
    obj = ROOT / f"빌드/npu/gl-aie/{kind}.o"
    output_size = 32 if kind == "gl_row1" else 128

    @iron.jit
    @iron.compileconfig(object_files=[obj])
    def design(a: In, b: In, c: Out, *, frames: CompileTime[int], workers: CompileTime[int]):
        at = np.ndarray[(256,), np.dtype[np.int8]]
        bt = np.ndarray[(1024,), np.dtype[np.int8]]
        ct = np.ndarray[(output_size,), np.dtype[np.int32]]
        af = np.ndarray[(frames * 256,), np.dtype[np.int8]]
        bf = np.ndarray[(frames * 1024,), np.dtype[np.int8]]
        cf = np.ndarray[(frames * output_size,), np.dtype[np.int32]]
        fa = ObjectFifo(np.ndarray[(workers * 256,), np.dtype[np.int8]], name="a")
        fb = ObjectFifo(np.ndarray[(workers * 1024,), np.dtype[np.int8]], name="b")
        fc = ObjectFifo(np.ndarray[(workers * output_size,), np.dtype[np.int32]], name="c")
        if workers == 1:
            inputs_a, inputs_b, outputs = [fa], [fb], [fc]
        else:
            inputs_a = fa.cons().split([i * 256 for i in range(workers)],
                                      obj_types=[at] * workers, names=[f"a{i}" for i in range(workers)])
            inputs_b = fb.cons().split([i * 1024 for i in range(workers)],
                                      obj_types=[bt] * workers, names=[f"b{i}" for i in range(workers)])
            outputs = fc.prod().join([i * output_size for i in range(workers)],
                                    obj_types=[ct] * workers, names=[f"c{i}" for i in range(workers)])
        kernel = Kernel("geul_tile", str(obj), [at, bt, ct])

        def body(a, b, c, kernel):
            for _ in range_(frames // workers):
                aa, bb, cc = a.acquire(1), b.acquire(1), c.acquire(1)
                kernel(aa, bb, cc)
                a.release(1)
                b.release(1)
                c.release(1)

        cores = [Worker(body, fn_args=[inputs_a[i].cons(), inputs_b[i].cons(),
                                      outputs[i].prod(), kernel]) for i in range(workers)]
        runtime = Runtime()
        with runtime.sequence(af, bf, cf) as (a, b, c):
            runtime.start(*cores)
            runtime.fill(fa.prod(), a)
            runtime.fill(fb.prod(), b)
            runtime.drain(fc.cons(), c, wait=True)
        module = Program(iron.get_current_device(), runtime).resolve_program()
        (HERE / f"results/multi-{kind}-{workers}-{frames}.mlir").write_text(str(module), encoding="utf8")
        return module

    return design


class MultiDevice(Device):
    def __init__(self, workers=1, frames=2048, kind="gl"):
        if workers not in (1, 2, 4) or frames % workers:
            raise ValueError("workers must be 1/2/4 and divide the frame capacity")
        super().__init__(kind, frames)
        self.workers = workers
        self.kernel = make_parallel_kernel(kind)
        self.profile = {}

    def __call__(self, a, b):
        frames = len(a)
        if a.dtype != np.int8 or b.dtype != np.int8 or a.shape != (frames, 8, 32) or b.shape != (frames, 32, 32) or not 0 < frames <= self.frames:
            raise ValueError("invalid input tiles")
        start = time.perf_counter()
        self.ah.fill(0)
        self.bh.fill(0)
        self.ah[:a.size] = a.ravel()
        self.bh[:b.size] = b.ravel()
        copied = time.perf_counter()
        self.a._sync_to_device()
        self.b._sync_to_device()
        uploaded = time.perf_counter()
        self.kernel(self.a, self.b, self.c, frames=self.frames, workers=self.workers)
        executed = time.perf_counter()
        result = self.c.numpy().reshape(self.frames, self.rows, 32)[:frames].copy()
        finished = time.perf_counter()
        self.profile = dict(host_copy=copied-start, upload=uploaded-copied,
                            runtime_call=executed-uploaded, readback=finished-executed)
        self.calls += 1
        return result
