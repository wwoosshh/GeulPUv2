"""Compare identical Qwen weights and host KV across a prompt/decode backend switch."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import numpy as np
from cached_qwen import CachedQwen, R
from cpu_linear import CpuLinear
from gl_aie.runtime import Device, PlannedLinear
from native_post import HERE, ROOT, NativePost


def digest(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def checked(reference, actual):
    if not np.array_equal(reference.view(np.uint32), actual.view(np.uint32)):
        raise AssertionError("FP32 output bits differ")


def main():
    g = R.GR.GGUF(str(ROOT / "빌드/npu/models/Qwen3-0.6B-Q8_0.gguf"))
    output = HERE / "results/phase-bench.json"
    report = json.loads(output.read_text()) if output.exists() else {"complete": False, "linear": [], "runs": []}

    def record(section, row):
        report[section].append(row)
        output.write_text(json.dumps(report, indent=2), encoding="utf8")
        print(section, row, flush=True)

    with closing(CpuLinear()) as cpu, ThreadPoolExecutor(1) as pool:
        device = Device("gl", 2048)
        npu = PlannedLinear(device, finalizer=NativePost(), finalizer_executor=pool)
        model = CachedQwen(g, cpu, quant_cache=True, context_capacity=257)
        rng = np.random.default_rng(407)
        prompt = rng.integers(100, 30000, 256).tolist()
        # The actual up projection from the first layer, same input on both paths.
        w = model.layers[0][1]["ffn_up"]
        bias = np.zeros(w[1].shape[0], dtype=np.float32)
        for length in (16,64,256):
            x = rng.standard_normal((length,w[1].shape[1])).astype(np.float32)
            if sum(r['length']==length for r in report['linear'])==4:
                continue
            expected = cpu(x,w[1],w[2],bias)
            checked(expected,npu(x,w[1],w[2],bias))
            for name,backend in (("cpu",cpu),("npu",npu),("npu",npu),("cpu",cpu)):
                start=time.perf_counter()
                actual=backend(x,w[1],w[2],bias)
                elapsed=time.perf_counter()-start
                checked(expected,actual)
                record("linear",dict(length=length,backend=name,seconds=elapsed))
        # Warm the entire NPU host weight plan before comparing steady-state TTFT.
        model.backend=npu
        model.step(prompt[:3])
        for length in (16,64,256):
            reference = {}
            for name in (("phase","cpu") if length==64 else ("cpu","phase")):
                existing=next((r for r in report['runs'] if r['length']==length and r['backend']==name),None)
                if existing is not None:
                    continue
                model.reset()
                model.backend = npu if name=="phase" else cpu
                start=time.perf_counter()
                prefill=model.step(prompt[:length])
                token=int(prefill.argmax())
                first=time.perf_counter()
                profile=model.last_profile.copy()
                model.backend=cpu
                decode=model.step([token])
                second=time.perf_counter()
                results=(prefill,decode)
                if reference:
                    for a,b in zip(reference["arrays"],results):
                        checked(a,b)
                else:
                    reference["arrays"]=results
                record("runs",dict(length=length,backend=name,prefill=first-start,
                     next_token=second-first,first_token=token,second_token=int(decode.argmax()),
                     prefill_hash=digest(prefill),decode_hash=digest(decode),
                     prefill_profile=profile))
        report["complete"]=True
        output.write_text(json.dumps(report, indent=2), encoding="utf8")

if __name__=="__main__":
    main()
