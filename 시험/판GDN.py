#!/usr/bin/env python3
"""0.12.0 12-3e-2 — Qwen3.5 의 선형 어텐션 GDN 의 판(커널생성 … --GDN — 커널/생성GDN.gl)을 CPU 참조(계약/정수계약.py — 계약 4.10)와 비트까지.
판 다섯을 차례로(합성곱 → 합성곱 상태 → L2 정규화 → 게이트 값 → 순환) 무작위 값으로 돌려 단계마다 견주고, 같은 토큰열을 덩이 둘로 나눠
(합성곱 상태 · S 를 넘겨) 돌린 끝의 o · S 도 한 번에 돌린 CPU 참조와 견준다. 끝에 순환 판의 속도(토큰 · 층 하나의 시간)를 잰다.
2026-10-25 속도 S3a — 순환 판은 GDN순환둘(블록 = (v 헤드, 열 64) — 띄움이 v머리 · 2 블록), 덩이 셋(끝은 토큰 하나)의 경우.

  python -I 시험/판GDN.py <ptx> <output.json> [--토큰 300] [--k머리 4] [--v머리 8] [--재기 8192]
마지막 줄 "모두 0: True|False".
"""
import ctypes
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "계약"))
sys.path.insert(0, HERE)
import 쿠다드라이버 as G          # noqa: E402
import 시험장치               # noqa: E402
import 정수계약 as C        # noqa: E402

f32 = np.float32
u64, i64 = G.c_u64, G.c_i64
EPS = 1e-6                      # 모듈의 RMS_eps(qwen2큰 의 기본 — Qwen3.5 와 같다)


def 다른칸(a, b):
    a, b = np.ascontiguousarray(a, f32), np.ascontiguousarray(b, f32)
    assert a.shape == b.shape, (a.shape, b.shape)
    return int(np.count_nonzero(a.view(np.int32) != b.view(np.int32)))


def 인자값(n, a):
    v = int(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else a
    return v


def 자료(r, T, Hk, Hv):
    """무작위 입력 — qkv(행렬곱 출력 꼴), 합성곱 가중치 · 처음 상태, b · a, dt · A(음수), 처음 S. 몇 헤드는 0(L2 의 eps 자리)."""
    Cn = 2 * Hk * 128 + Hv * 128
    qkv = (r.standard_normal((T, Cn)) * 10.0 ** r.uniform(-1, 1, (1, Cn))).astype(f32)
    qkv[r.integers(0, T, 3), :128] = 0.0                              # q 헤드 0 이 0 인 행(합성곱 뒤에도 0 이 되지는 않는다)
    w = (r.standard_normal((Cn, 4)) * 0.5).astype(f32)
    w[:128] = 0.0                                                     # q 헤드 0 의 채널: 합성곱이 0 → L2 의 S = 0(eps 자리)
    상태0 = (r.standard_normal((3, Cn))).astype(f32)
    b = (r.standard_normal((T, Hv)) * 3).astype(f32)
    a = (r.standard_normal((T, Hv)) * 4).astype(f32)
    dt = (r.standard_normal(Hv) * 2).astype(f32)
    A = (-np.exp(r.standard_normal(Hv))).astype(f32)
    S0 = (r.standard_normal((Hv, 128, 128)) * 0.05).astype(f32)
    return qkv, w, 상태0, b, a, dt, A, S0


def 참조(qkv, w, 상태0, b, a, dt, A, S0, Hk, Hv):
    """CPU 참조 — 단계마다: 합성곱 + SiLU, 새 합성곱 상태, 정규화한 y, 베타 · 감마, o · S."""
    T = qkv.shape[0]
    kd = Hk * 128
    y0, 새상태 = C.합성곱4(qkv, w, 상태0)
    y0 = C.실루(y0)
    y = y0.copy()
    y[:, :kd] = C.곱(C.L2정규화(y0[:, :kd], EPS, 128), C.눈금128)
    y[:, kd:2 * kd] = C.L2정규화(y0[:, kd:2 * kd], EPS, 128)
    베타 = C.시그모이드(b)
    감마 = C.적힌지수(C.곱(C.소프트플러스(C.더(a, dt[None, :])), A[None, :]))
    짝 = np.arange(Hv) % Hk
    qh = np.ascontiguousarray(y[:, :kd].reshape(T, Hk, 128)[:, 짝])
    kh = np.ascontiguousarray(y[:, kd:2 * kd].reshape(T, Hk, 128)[:, 짝])
    vh = np.ascontiguousarray(y[:, 2 * kd:]).reshape(T, Hv, 128)
    o, S = C.GDN(qh, kh, vh, 베타, 감마, S0)
    return y0, 새상태, y, 베타, 감마, o, S


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ptx, 출력 = [a for a in sys.argv[1:] if not a.startswith("--")][:2]
    T, Hk, Hv = 인자값("--토큰", 300), 인자값("--k머리", 4), 인자값("--v머리", 8)
    재기 = 인자값("--재기", 8192)
    dr = 시험장치.드라이버()
    mod = dr.모듈(open(ptx, "rb").read())
    fn = {}

    def 띄우기(이름, grid, block, params):
        if 이름 not in fn:
            fn[이름] = dr.함수(mod, 이름)
        ps = [u64(p) if 참조_ else i64(p) for 참조_, p in params]
        args = (ctypes.c_void_p * len(ps))(*[ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in ps])
        rr = dr.cu.cuLaunchKernel(fn[이름], grid[0], grid[1], 1, block, 1, 1, 0, None, args, None)
        assert rr == 0, (이름, rr)

    def up(x):
        b = np.ascontiguousarray(x).tobytes()
        p = dr.할당(len(b))
        dr.확인(dr.cu.cuMemcpyHtoD_v2(p, b, len(b)))
        return p

    def down(p, 모양):
        return dr.내리기(p, np.empty(모양, f32))

    def 덩이(pqkv, pw, p상태, py, pb, pa, pdt, pA, p베, p감, pS, po, T_):
        """판 다섯을 차례로 — 덩이 하나(T_ 행)."""
        Cn = 2 * Hk * 128 + Hv * 128
        띄우기("GDN합성곱", ((T_ * Cn + 255) // 256, 1), 256, [(1, pqkv), (1, pw), (1, p상태), (1, py), (0, T_), (0, Cn)])
        y0 = down(py, (T_, Cn))
        띄우기("GDN합성곱상태", ((Cn + 255) // 256, 1), 256, [(1, pqkv), (1, p상태), (0, T_), (0, Cn)])
        띄우기("GDN정규화", ((T_ * Hk * 2 + 3) // 4, 1), 128, [(1, py), (0, T_), (0, Hk), (0, Cn)])
        띄우기("GDN게이트값", ((T_ * Hv + 255) // 256, 1), 256, [(1, pb), (1, pa), (1, pdt), (1, pA), (1, p베), (1, p감), (0, T_), (0, Hv)])
        띄우기("GDN순환둘", (Hv * 2, 1), 128, [(1, py), (1, p베), (1, p감), (1, pS), (1, po), (0, T_), (0, Hk), (0, Hv), (0, Cn)])
        dr.맞추기()
        return y0

    r = np.random.default_rng(35)
    결과 = {"토큰": T, "k머리": Hk, "v머리": Hv, "경우": []}
    qkv, w, 상태0, b, a, dt, A, S0 = 자료(r, T, Hk, Hv)
    t0 = time.time()
    y0r, 새상태r, yr, 베r, 감r, or_, Sr = 참조(qkv, w, 상태0, b, a, dt, A, S0, Hk, Hv)
    print(f"CPU 참조 {time.time() - t0:.1f} 초", flush=True)
    Cn = qkv.shape[1]
    # 한 번에
    pqkv, pw, p상태, py = up(qkv), up(w), up(상태0), dr.할당(T * Cn * 4)
    pb, pa, pdt, pA = up(b), up(a), up(dt), up(A)
    p베, p감, pS, po = dr.할당(T * Hv * 4), dr.할당(T * Hv * 4), up(S0), dr.할당(T * Hv * 128 * 4)
    y0 = 덩이(pqkv, pw, p상태, py, pb, pa, pdt, pA, p베, p감, pS, po, T)
    다 = {"합성곱 + SiLU": 다른칸(y0, y0r), "합성곱 상태": 다른칸(down(p상태, (3, Cn)), 새상태r), "L2 정규화": 다른칸(down(py, (T, Cn)), yr),
         "β": 다른칸(down(p베, (T, Hv)), 베r), "γ": 다른칸(down(p감, (T, Hv)), 감r),
         "o": 다른칸(down(po, (T, Hv, 128)), or_), "S": 다른칸(down(pS, (Hv, 128, 128)), Sr)}
    결과["경우"].append({"경우": f"한 번에({T} 행)", "다른 칸": 다})
    print("한 번에:", 다, flush=True)
    # 덩이 둘(상태를 넘겨) — 끝의 o · S · 합성곱 상태가 한 번에 한 CPU 참조와 같아야 한다
    T1 = T // 3
    p상태2, pS2 = up(상태0), up(S0)
    os_ = []
    for s0, s1 in ((0, T1), (T1, T)):
        n = s1 - s0
        q2, py2 = up(qkv[s0:s1]), dr.할당(n * Cn * 4)
        b2, a2 = up(b[s0:s1]), up(a[s0:s1])
        베2, 감2, po2 = dr.할당(n * Hv * 4), dr.할당(n * Hv * 4), dr.할당(n * Hv * 128 * 4)
        덩이(q2, pw, p상태2, py2, b2, a2, pdt, pA, 베2, 감2, pS2, po2, n)
        os_.append(down(po2, (n, Hv, 128)))
    다2 = {"o": 다른칸(np.concatenate(os_), or_), "S": 다른칸(down(pS2, (Hv, 128, 128)), Sr), "합성곱 상태": 다른칸(down(p상태2, (3, Cn)), 새상태r)}
    결과["경우"].append({"경우": f"덩이 둘({T1} · {T - T1} 행 — 상태를 넘겨)", "다른 칸": 다2})
    print("덩이 둘:", 다2, flush=True)
    # 2026-10-25 속도 S3a — 덩이 셋(끝은 토큰 하나 — 생성 걸음의 꼴: 순환 판이 덩이(토큰 8)를 판에 올리는 길의 끝)
    p상태3, pS3a = up(상태0), up(S0)
    os3 = []
    for s0, s1 in ((0, T1), (T1, T - 1), (T - 1, T)):
        n = s1 - s0
        q2, py2 = up(qkv[s0:s1]), dr.할당(n * Cn * 4)
        b2, a2 = up(b[s0:s1]), up(a[s0:s1])
        베2, 감2, po2 = dr.할당(n * Hv * 4), dr.할당(n * Hv * 4), dr.할당(n * Hv * 128 * 4)
        덩이(q2, pw, p상태3, py2, b2, a2, pdt, pA, 베2, 감2, pS3a, po2, n)
        os3.append(down(po2, (n, Hv, 128)))
    다3 = {"o": 다른칸(np.concatenate(os3), or_), "S": 다른칸(down(pS3a, (Hv, 128, 128)), Sr), "합성곱 상태": 다른칸(down(p상태3, (3, Cn)), 새상태r)}
    결과["경우"].append({"경우": f"덩이 셋({T1} · {T - 1 - T1} · 1 행 — 끝은 토큰 하나)", "다른 칸": 다3})
    print("덩이 셋:", 다3, flush=True)
    # 속도 — 순환 판(4B · 9B 꼴: k머리 16 · v머리 32), 토큰 재기 개
    Hk2, Hv2 = 16, 32
    C2 = 2 * Hk2 * 128 + Hv2 * 128
    py3 = up((r.standard_normal((재기, C2)) * 0.05).astype(f32))
    p베3, p감3 = up(np.full((재기, Hv2), 0.5, f32)), up(np.full((재기, Hv2), 0.9, f32))
    pS3, po3 = up(np.zeros((Hv2, 128, 128), f32)), dr.할당(재기 * Hv2 * 128 * 4)
    인자3 = [(1, py3), (1, p베3), (1, p감3), (1, pS3), (1, po3), (0, 재기), (0, Hk2), (0, Hv2), (0, C2)]
    t1 = time.perf_counter()
    while time.perf_counter() - t1 < 0.5:          # 2026-10-25 S3a — CPU 참조 뒤 GPU 클럭이 낮다(데우기)
        띄우기("GDN순환둘", (Hv2 * 2, 1), 128, 인자3)
        dr.맞추기()
    시간 = []
    for _ in range(3):
        dr.맞추기()
        t1 = time.perf_counter()
        띄우기("GDN순환둘", (Hv2 * 2, 1), 128, 인자3)
        dr.맞추기()
        시간.append(time.perf_counter() - t1)
    결과["속도"] = {"토큰": 재기, "k머리": Hk2, "v머리": Hv2, "ms": round(1000 * min(시간), 2), "µs/토큰": round(1e6 * min(시간) / 재기, 3)}
    print("순환 판 속도:", 결과["속도"], flush=True)
    결과["모두 0"] = all(v == 0 for c in 결과["경우"] for v in c["다른 칸"].values())
    print("모두 0:", 결과["모두 0"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
