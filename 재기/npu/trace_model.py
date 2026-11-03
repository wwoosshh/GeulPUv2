"""Retest GGUF with every Q8/Q4 linear routed through the actual NPU.

No fallback: any device error or bit mismatch stops the run. JSONL survives failure.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from exact_linear import ExactLinear, C, ROOT, difference

sys.path.insert(0, str(ROOT / "시험"))
import 참조7B as R

def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", default="1,2,3")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--backend", choices=["dot", "tiled", "planned", "fused", "gl-aie", "gl-native-post", "native-c", "native-c-unrolled", "native-aie"], default="dot")
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be positive")
    if args.backend == "gl-native-post":
        from gl_aie.runtime import Device, PlannedLinear
        from next_stage.native_post import NativePost
        dot=Device('gl',frames=2048)
        linear=PlannedLinear(dot,finalizer=NativePost())
    elif args.backend in ("gl-aie", "native-c", "native-c-unrolled", "native-aie"):
        from gl_aie.runtime import Device, PlannedLinear
        kind={"gl-aie":"gl","native-c":"c","native-c-unrolled":"c_unrolled","native-aie":"aie_cpp"}[args.backend]
        dot=Device(kind,frames=2048)
        linear=PlannedLinear(dot)
    elif args.backend == "fused":
        from fused_kernel import NpuFusedTiles
        from planned_linear import PlannedLinear
        dot = NpuFusedTiles(frames=2048)
        linear = PlannedLinear(dot)
    elif args.backend == "planned":
        from tiled_kernel import NpuTiles
        from planned_linear import PlannedLinear
        dot = NpuTiles(frames=2048)
        linear = PlannedLinear(dot)
    elif args.backend == "tiled":
        from tiled_kernel import NpuTiles
        from tiled_linear import TiledLinear
        dot = NpuTiles()
        linear = TiledLinear(dot)
    else:
        from exact_kernel import NpuDot32
        dot = NpuDot32()
        linear = ExactLinear(dot)
    original = C.행렬곱블라스
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokens = [int(x) for x in args.tokens.split(",")]
    with args.output.open("w", encoding="utf-8") as stream:
        def emit(obj):
            stream.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps(obj, ensure_ascii=False), flush=True)
        emit({"event":"start", "model":str(args.model), "tokens":tokens, "backend":args.backend,
              "scope":"full reference forward; Q8/Q4 linear integer dots on NPU; remaining operations CPU",
              "model_sha256":hashlib.file_digest(args.model.open("rb"), "sha256").hexdigest()})
        g = R.GR.GGUF(str(args.model))
        layer_count = R.설정(g)["층"]
        call = 0
        repeat = 0
        def traced(x, q, d, bias, **kwargs):
            nonlocal call
            call += 1
            start = time.perf_counter()
            ref = original(x,q,d,bias,**kwargs)
            ref_seconds = time.perf_counter()-start
            start = time.perf_counter()
            actual = linear(x,q,d,bias,**kwargs)
            hybrid_seconds = time.perf_counter()-start
            delta = difference(ref,actual)
            name = (f"blk.{(call-1)//5}." + ["qkv", "attn_output", "ffn_gate", "ffn_up", "ffn_down"][(call-1)%5]
                    if call <= layer_count*5 else "output")
            emit({"event":"linear", "repeat":repeat, "call":call,
                  "name":name, "min":float(actual.min()), "max":float(actual.max()),
                  "sample":actual.ravel()[:4].tolist(),
                  "input_shape":list(x.shape), "weight_shape":list(q.shape),
                  "cpu_seconds":ref_seconds, "hybrid_seconds":hybrid_seconds,
                  "profile":getattr(linear,"profile",{}),
                  "sha256":hashlib.sha256(actual.tobytes()).hexdigest(), **delta})
            if delta["different_bits_elements"]:
                np.savez(args.output.with_suffix(".mismatch.npz"), reference=ref, actual=actual, input=x)
                raise AssertionError("linear contract mismatch")
            return actual
        try:
            baseline, _, base_residual = R.앞으로(g, [tokens], 말=False, 렌즈행=range(len(tokens)), 잔차=True)
            np.save(args.output.with_suffix(".baseline-logits.npy"), baseline)
            previous = None
            for repeat in range(args.repeats):
                call = 0
                C.행렬곱블라스 = traced
                actual, _, residual = R.앞으로(g,[tokens],말=False, 렌즈행=range(len(tokens)), 잔차=True)
                np.save(args.output.with_suffix(f".repeat-{repeat}-logits.npy"), actual)
                for layer in range(layer_count):
                    residual_delta = difference(base_residual[:,layer], residual[:,layer])
                    emit({"event":"residual", "repeat":repeat, "layer":layer,
                          **residual_delta})
                    if residual_delta["different_bits_elements"]:
                        raise AssertionError("residual contract mismatch")
                delta = difference(baseline,actual)
                emit({"event":"logits", "repeat":repeat,
                      "sha256":hashlib.sha256(actual.tobytes()).hexdigest(),
                      "greedy_tokens":np.argmax(actual,axis=1).tolist(),
                      "repeat_difference":None if previous is None else difference(previous,actual), **delta})
                if delta["different_bits_elements"]:
                    raise AssertionError("logit contract mismatch")
                previous = actual.copy()
            emit({"event":"complete", "npu_dispatches":dot.calls, "passed":True,
                  "plan_cache_hits":getattr(linear,"hits",None), "plan_cache_misses":getattr(linear,"misses",None),
                  "host_plan_bytes":getattr(linear,"bytes",None)})
        except Exception as exc:
            emit({"event":"failed", "error":repr(exc), "npu_dispatches":dot.calls})
            raise
        finally:
            C.행렬곱블라스 = original

if __name__ == "__main__":
    main()
