"""Fixed-work multiworker GL experiment, followed by exact linear comparisons."""
import json
import time
import traceback
from contextlib import closing
import numpy as np
from multi_device import MultiDevice
from native_post import HERE, ROOT, NativePost
from joined_prefill import JoinedPrefill
from cpu_linear import CpuLinear
from cached_qwen import R


def check(a, b):
    if not np.array_equal(a.view(np.uint32), b.view(np.uint32)):
        raise AssertionError("output bits differ")


def main():
    report = {"complete": False, "tiles": [], "linear": [], "checks": 0}
    path = HERE / "results/multiworker.json"

    def save():
        path.write_text(json.dumps(report, indent=2), encoding="utf8")

    try:
        rng = np.random.default_rng(154)
        devices = {w: MultiDevice(w) for w in (1, 2, 4)}
        for size in (1, 7, 128, 2047, 2048):
            a = rng.integers(-128, 128, (size, 8, 32), dtype=np.int8)
            b = rng.integers(-128, 128, (size, 32, 32), dtype=np.int8)
            if size == 7:
                a[:3] = -128
                b[:3] = -128
            expected = np.einsum("frk,fnk->frn", a[:, :4].astype(np.int32) + 256 * a[:, 4:].astype(np.int32), b.astype(np.int32))
            for workers, device in devices.items():
                actual = device(a, b)
                check(expected, actual)
                report["checks"] += 1
                save()
                print("correct", size, workers, flush=True)
        for repeat in range(8):
            order = (1,2,4) if repeat % 2 == 0 else (4,2,1)
            for workers in order:
                device = devices[workers]
                start = time.perf_counter()
                actual = device(a,b)
                elapsed = time.perf_counter()-start
                check(expected,actual)
                row = dict(repeat=repeat,workers=workers,seconds=elapsed,profile=device.profile.copy())
                report["tiles"].append(row)
                report["checks"] += 1
                save()
        print("tile timing complete", flush=True)
        g = R.GR.GGUF(str(ROOT / "빌드/npu/models/Qwen3-0.6B-Q8_0.gguf"))
        weight = R.가중(g, "blk.0.ffn_up.weight")
        q, scales = weight[1:3]
        bias = np.zeros(len(q), np.float32)
        backends = {w: JoinedPrefill(device, finalizer=NativePost()) for w,device in devices.items()}
        with closing(CpuLinear()) as cpu:
            for rows in (16,64,256):
                x = rng.standard_normal((rows,q.shape[1])).astype(np.float32)
                reference = cpu(x,q,scales,bias)
                for backend in backends.values():
                    check(reference,backend(x,q,scales,bias))
                    report["checks"] += 1
                for repeat in range(4):
                    order = (1,2,4) if repeat % 2 == 0 else (4,2,1)
                    for workers in order:
                        start = time.perf_counter()
                        actual = backends[workers](x,q,scales,bias)
                        elapsed = time.perf_counter()-start
                        check(reference,actual)
                        row = dict(rows=rows,repeat=repeat,workers=workers,seconds=elapsed,
                                   profile=backends[workers].profile.copy())
                        report["linear"].append(row)
                        report["checks"] += 1
                        save()
                        print("linear",row,flush=True)
        report["complete"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
