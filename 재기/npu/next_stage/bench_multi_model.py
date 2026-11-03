"""End-to-end prefill comparison. CPU lm_head and decode are shared."""
import hashlib
import json
import time
import traceback
from contextlib import closing
import numpy as np
from multi_device import MultiDevice
from native_post import HERE, ROOT, NativePost
from joined_prefill import JoinedPrefill
from cpu_linear import CpuLinear
from cached_qwen import CachedQwen, R


def main():
    report = {"complete": False, "runs": []}
    path = HERE / "results/multiworker-model.json"
    try:
        g = R.GR.GGUF(str(ROOT / "빌드/npu/models/Qwen3-0.6B-Q8_0.gguf"))
        prompt = np.random.default_rng(407).integers(100,30000,256).tolist()[:16]
        prior = json.loads((HERE / "results/phase-bench.json").read_text())
        reference = next(r for r in prior["runs"] if r["length"] == 16 and r["backend"] == "cpu")
        with closing(CpuLinear()) as cpu:
            backends = {w: JoinedPrefill(MultiDevice(w), finalizer=NativePost()) for w in (1,4)}
            model = CachedQwen(g, cpu, quant_cache=True)
            def route(x, *args, **kwargs):
                backend = cpu if selected == "cpu" or len(x) == 1 else backends[selected]
                return backend(x,*args,**kwargs)
            model.backend = route
            for selected in (1,4):
                model.reset()
                model.step(prompt[:3])
            for selected in ("cpu",1,4,4,1,"cpu"):
                model.reset()
                start = time.perf_counter()
                logits = model.step(prompt)
                token = int(logits.argmax())
                prefill_end = time.perf_counter()
                decoded = model.step([token])
                end = time.perf_counter()
                first_hash = hashlib.sha256(logits.tobytes()).hexdigest()
                next_hash = hashlib.sha256(decoded.tobytes()).hexdigest()
                if (first_hash,next_hash) != (reference["prefill_hash"],reference["decode_hash"]):
                    raise AssertionError("model logits differ from CPU reference")
                row = dict(backend=selected,prefill_seconds=prefill_end-start,
                           decode_seconds=end-prefill_end,first_hash=first_hash,next_hash=next_hash)
                report["runs"].append(row)
                path.write_text(json.dumps(report,indent=2),encoding="utf8")
                print(row,flush=True)
        report["complete"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        path.write_text(json.dumps(report,indent=2),encoding="utf8")


if __name__ == "__main__":
    main()
