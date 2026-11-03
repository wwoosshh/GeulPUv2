"""XDNA2 signed int8 dot32; no floating point and no CPU fallback."""
import numpy as np
import aie.iron as iron
from aie.iron import ExternalFunction, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.config import cxx_header_path

FRAME = 64
FRAMES = 256
CAPACITY = FRAME * FRAMES
SOURCE = r'''
#define NOCPP
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void geul_dot32(int8_t *a, int8_t *b, int32_t *c) {
  for (int t=0; t<64; ++t) {
    auto x = aie::load_v<32>(a + t*32);
    auto y = aie::load_v<32>(b + t*32);
    auto p = aie::mul(x, y);
    c[t] = aie::reduce_add(p.to_vector<int32_t>(0));
  }
}
'''

@iron.jit
def dot_kernel(a: In, b: In, c: Out):
    tile = np.ndarray[(FRAME * 32,), np.dtype[np.int8]]
    result = np.ndarray[(FRAME,), np.dtype[np.int32]]
    full = np.ndarray[(CAPACITY * 32,), np.dtype[np.int8]]
    out = np.ndarray[(CAPACITY,), np.dtype[np.int32]]
    fa, fb, fc = ObjectFifo(tile, name="a"), ObjectFifo(tile, name="b"), ObjectFifo(result, name="c")
    kernel = ExternalFunction("geul_dot32", source_string=SOURCE,
                              arg_types=[tile, tile, result], include_dirs=[cxx_header_path()])
    def body(a, b, c, kernel):
        for _ in range_(FRAMES):
            aa, bb, cc = a.acquire(1), b.acquire(1), c.acquire(1)
            kernel(aa, bb, cc)
            a.release(1)
            b.release(1)
            c.release(1)
    worker = Worker(body, fn_args=[fa.cons(), fb.cons(), fc.prod(), kernel])
    runtime = Runtime()
    with runtime.sequence(full, full, out) as (a, b, c):
        runtime.start(worker)
        runtime.fill(fa.prod(), a)
        runtime.fill(fb.prod(), b)
        runtime.drain(fc.cons(), c, wait=True)
    return Program(iron.get_current_device(), runtime).resolve_program()

class NpuDot32:
    def __init__(self):
        self.a = iron.zeros((CAPACITY * 32,), dtype=np.int8, device="npu")
        self.b = iron.zeros((CAPACITY * 32,), dtype=np.int8, device="npu")
        self.c = iron.zeros((CAPACITY,), dtype=np.int32, device="npu")
        self.calls = 0

    def __call__(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != np.int8 or b.dtype != np.int8 or a.shape != b.shape or a.ndim != 2 or a.shape[1] != 32:
            raise ValueError("expected matching int8 [tasks,32] inputs")
        n = len(a)
        if not 0 < n <= CAPACITY:
            raise ValueError("task count outside kernel capacity")
        self.a.numpy()[:] = 0
        self.b.numpy()[:] = 0
        self.a.numpy()[:n*32] = a.ravel()
        self.b.numpy()[:n*32] = b.ravel()
        self.a._sync_to_device()
        self.b._sync_to_device()
        dot_kernel(self.a, self.b, self.c)
        self.calls += 1
        return self.c.numpy()[:n].copy()

if __name__ == "__main__":
    rng = np.random.default_rng(17)
    a = rng.integers(-128, 128, (CAPACITY, 32), dtype=np.int8)
    b = rng.integers(-128, 128, a.shape, dtype=np.int8)
    a[:4] = np.array([-128, 127, -128, 127], np.int8)[:, None]
    b[:4] = np.array([-128, 127, 127, -128], np.int8)[:, None]
    expected = (a.astype(np.int32) * b.astype(np.int32)).sum(axis=1)
    device = NpuDot32()
    for repeat in range(3):
        actual = device(a, b)
        np.testing.assert_array_equal(actual, expected)
        print(f"NPU dot32 repeat={repeat} tasks={len(a)} exact=True", flush=True)
