#!/usr/bin/env python3
"""22a (1) — 7B 모듈의 줄 판(정수줄자리 — 행 넷까지, 활성값의 자리 둘 · 정보를 전역에서)을 CPU 참조(계약/정수계약.행렬곱블라스)와 비트까지.
7B 층 0 의 실제 가중치(엔진과 같은 배치 — 값 [KB][N][32] · 척도 [KB][Np] 반실수, gate · up 은 32 열마다 번갈아)와 무작위 활성값(행 1 … 4):
  ""            qkv (+ 치우침), 로짓(output.weight)
  "_잔차"       o · down — 출력 = 잔차 + 행렬곱
  "_SwiGLU정수" gate·up — h = silu(g) · u 를 down 의 자리 둘 · 정보로

  python -I 시험/줄128.py <qwen2Q8_7b.ptx> <7B gguf> <출력.json>
"""
import ctypes
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
for _d in ("계약",):
    sys.path.insert(0, os.path.join(ROOT, _d))
sys.path.insert(0, HERE)
import 쿠다드라이버 as G          # noqa: E402
import 시험장치               # noqa: E402
import gguf읽기 as GR       # noqa: E402
import 정수계약 as C        # noqa: E402
from 참조7B import q8       # noqa: E402
from 머리128 import 자리바이트    # noqa: E402

f32 = np.float32
u64, i64 = G.c_u64, G.c_i64


def 깔기(q, d):
    """GGUF 배치 (q [N][K] int8, d [N][KB]) → 엔진 배치의 바이트: 값 [KB][N][32] + 척도 [KB][Np] 반실수 (남는 칸 0)."""
    N, K = q.shape
    KB = K // 32
    Np = (N + 63) // 64 * 64
    값 = np.ascontiguousarray(q.reshape(N, KB, 32).transpose(1, 0, 2))
    척 = np.zeros((KB, Np), np.float16)
    척[:, :N] = d.T.astype(np.float16)
    return 값.tobytes(), 척.tobytes()


def 다른칸(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.dtype == f32:
        a, b = a.view(np.int32), np.asarray(b, f32).view(np.int32)
    return int(np.count_nonzero(a != b))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ptx, gguf, 출력 = sys.argv[1:4]
    dr = 시험장치.드라이버()
    mod = dr.모듈(open(ptx, "rb").read())
    fn = {e: dr.함수(mod, f"정수줄자리{e}") for e in ("", "_잔차", "_SwiGLU정수")}
    dr.cu.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
    속성 = {}
    for e, f in fn.items():
        v = []
        for a in (4, 3, 1):
            x = ctypes.c_int()
            dr.확인(dr.cu.cuFuncGetAttribute(ctypes.byref(x), a, f))
            v.append(x.value)
        속성[f"정수줄자리{e}"] = {"레지스터": v[0], "지역(넘침) 바이트": v[1], "공유 바이트": v[2]}
    print(json.dumps(속성, ensure_ascii=False), flush=True)

    def 띄우기(e, grid, params):
        ps = [u64(p) if 참조 else i64(p) for 참조, p in params]
        args = (ctypes.c_void_p * len(ps))(*[ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in ps])
        r = dr.cu.cuLaunchKernel(fn[e], grid, 1, 1, 256, 1, 1, 0, None, args, None)
        assert r == 0, (e, r)

    def up(b):
        p = dr.할당(len(b))
        dr.확인(dr.cu.cuMemcpyHtoD_v2(p, b, len(b)))
        return p

    g = GR.GGUF(gguf)
    r = np.random.default_rng(22)
    가중 = {}
    qs, ds = zip(*(q8(g, f"blk.0.attn_{a}.weight") for a in ("q", "k", "v")))
    가중["qkv"] = (np.concatenate(qs), np.concatenate(ds), np.concatenate([g.풀기(f"blk.0.attn_{a}.bias") for a in ("q", "k", "v")]))
    가중["o"] = (*q8(g, "blk.0.attn_output.weight"), None)
    가중["down"] = (*q8(g, "blk.0.ffn_down.weight"), None)
    wg, dg = q8(g, "blk.0.ffn_gate.weight")
    wu, du = q8(g, "blk.0.ffn_up.weight")
    F = wg.shape[0]
    순 = np.empty(2 * F, np.int64)
    for j in range(F // 32):
        순[64 * j:64 * j + 32] = np.arange(32 * j, 32 * j + 32)
        순[64 * j + 32:64 * j + 64] = F + np.arange(32 * j, 32 * j + 32)
    가중["gu"] = (np.concatenate([wg, wu])[순], np.concatenate([dg, du])[순], None)
    가중["로짓"] = (*q8(g, "output.weight"), None)
    결과 = {"커널 속성": 속성, "경우": []}
    for 이름, (q, d, b) in 가중.items():
        N, K = q.shape
        값, 척 = 깔기(q, d)
        p값, p척 = up(값), up(척)
        e = {"qkv": "", "로짓": "", "o": "_잔차", "down": "_잔차", "gu": "_SwiGLU정수"}[이름]
        W = 64 if e == "_SwiGLU정수" else 32
        치 = b if b is not None else np.zeros(N, f32)
        p치 = up(np.ascontiguousarray(치, f32).tobytes())
        for m in (1, 2, 3, 4):
            X = (r.standard_normal((m, K)) * 10.0 ** r.uniform(-1, 1, (m, 1))).astype(f32)
            폭 = (m + 63) // 64 * 64
            바이트, M0 = 자리바이트(X)
            정보 = np.zeros((K // 32, 폭), np.int32)
            정보[:, :m] = M0.T
            p자, p정 = up(바이트.tobytes()), up(정보.tobytes())
            한 = {"가중치": 이름, "끝손질": e or "(없음)", "K": K, "N": N, "행": m}
            if e == "_SwiGLU정수":
                p다자, p다정 = dr.할당(2 * m * (N // 2)), dr.할당((N // 64) * 폭 * 4)
                띄우기(e, (N + W - 1) // W, [(1, p자), (1, p정), (1, p값), (1, p척), (1, p치), (1, p다정), (1, p다자), (0, m), (0, K), (0, N)])
                dr.맞추기()
                글자리 = dr.내리기(p다자, np.empty((2, m, N // 2), np.int8))
                글정보 = dr.내리기(p다정, np.empty((N // 64, 폭), np.int32))[:, :m].T
                y = C.행렬곱블라스(X, q, d, np.zeros(N, f32))
                gcol = np.concatenate([np.arange(64 * j, 64 * j + 32) for j in range(N // 64)])
                ucol = gcol + 32
                h = C.swiglu(y[:, gcol], y[:, ucol])
                참자리, 참정보 = 자리바이트(h)
                한["다른 칸"] = {"자리": 다른칸(글자리, 참자리), "정보": 다른칸(글정보, 참정보)}
            else:
                p출 = dr.할당(m * N * 4)
                잔 = (r.standard_normal((m, N)) * 3.0).astype(f32)
                if e == "_잔차":
                    p잔 = up(잔.tobytes())
                    띄우기(e, (N + W - 1) // W, [(1, p자), (1, p정), (1, p값), (1, p척), (1, p치), (1, p잔), (1, p출), (0, m), (0, K), (0, N)])
                else:
                    띄우기(e, (N + W - 1) // W, [(1, p자), (1, p정), (1, p값), (1, p척), (1, p치), (1, p출), (0, m), (0, K), (0, N)])
                dr.맞추기()
                글 = dr.내리기(p출, np.empty((m, N), f32))
                참 = C.행렬곱블라스(X, q, d, 치)
                if e == "_잔차":
                    참 = (잔 + 참).astype(f32)
                한["다른 칸"] = {"출력": 다른칸(글, 참)}
            print(한, flush=True)
            결과["경우"].append(한)
        for p in dr.할당들:
            dr.확인(dr.cu.cuMemFree_v2(u64(p)))
        dr.할당들 = []
    결과["모두 0"] = all(v == 0 for c in 결과["경우"] for v in c["다른 칸"].values())
    print("모두 0:", 결과["모두 0"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
