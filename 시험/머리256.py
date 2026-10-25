#!/usr/bin/env python3
"""0.12.0 12-3e-3 — Qwen3.5 의 헤드 256 어텐션 · 게이트 · GDN 출력의 판(커널생성 … --GDN — 커널/생성머리256.gl)을 CPU 참조(계약/정수계약.py)와
비트까지: 머리정규화256(q · k) → RoPEKV256(앞 64 차원, θ 10⁷ — q 자리 · 키 · 값 캐시) → 어텐션256묶음 · 접기256(O / L) → 게이트256 → 실수자리로
(o 투영의 입력), 그리고 GDN출력(헤드 128 의 RMSNorm × SiLU(z)). 같은 행들을 덩이 둘로(앞 덩이의 키 · 값을 캐시에 두고) 돌린 어텐션 출력도
한 번에 셈한 CPU 참조와 견준다. 2026-10-25 속도 S1 — 생성 걸음(행 하나)의 판(조각어텐션256 · 조각묶음256 → 접기256): 한 번에 의 캐시에서
위치마다 행 하나를 CPU 참조와, 긴 캐시(4,300 행 — 묶음 다섯 · 조각 80)에서 위치마다 옛 판(어텐션256묶음 · 접기256 의 행 하나)과.

  python -I 시험/머리256.py <ptx> <output.json> [--행 300] [--덩이 0] [--빠른]
--행: 토큰 수(1024 를 넘으면 묶음이 둘 — 두 층 병합). --덩이 n: 앞 덩이의 행 수(0 이면 반). --빠른: 모듈이 --빠른(실수자리로 가 자리 하나).
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
from 머리128 import 자리바이트    # noqa: E402

f32 = np.float32
u64, i64 = G.c_u64, G.c_i64
H, HKV, HD, ROT = 8, 2, 256, 64            # Qwen3.5-0.8B 의 전체 어텐션 층
HV = 16                                     # GDN 출력의 v 헤드
EPS = 1e-6
θ = 1e7


def 다른칸(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.dtype == f32 or b.dtype == f32:
        a, b = a.astype(f32).view(np.int32), b.astype(f32).view(np.int32)
    return int(np.count_nonzero(a != b))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ptx, 출력 = [a for a in sys.argv[1:] if not a.startswith("--")][:2]
    T = int(sys.argv[sys.argv.index("--행") + 1]) if "--행" in sys.argv else 300
    T1 = int(sys.argv[sys.argv.index("--덩이") + 1]) if "--덩이" in sys.argv else 0
    T1 = T1 or T // 2
    자리수 = 1 if "--빠른" in sys.argv else 2
    최대길이 = (T + 1023) // 1024 * 1024
    묶음수 = 최대길이 // 1024
    폭 = H * 512 + HKV * 512
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

    def down(p, 모양, dt=f32):
        return dr.내리기(p, np.empty(모양, dt))

    r = np.random.default_rng(256)
    qkv = (r.standard_normal((T, 폭)) * 10.0 ** r.uniform(-0.5, 0.5, (1, 폭))).astype(f32)
    qw = (1.0 + 0.1 * r.standard_normal(HD)).astype(f32)
    kw = (1.0 + 0.1 * r.standard_normal(HD)).astype(f32)
    cos, sin = C.rope표(최대길이, θ, hd=ROT)
    결과 = {"행": T, "덩이": [T1, T - T1], "자리수": 자리수, "경우": []}

    # ---- CPU 참조 ----
    t0 = time.time()
    qn = qkv.copy()
    for a in range(H):
        qn[:, a * 512:a * 512 + HD] = C.머리정규화(qkv[:, a * 512:a * 512 + HD], qw, EPS, HD)
    kb = H * 512
    qn[:, kb:kb + HKV * HD] = C.머리정규화(qkv[:, kb:kb + HKV * HD], kw, EPS, HD)
    위치 = np.arange(T)

    def 돌(x):
        return np.concatenate([C.rope(x[:, :ROT], cos, sin, 위치), x[:, ROT:]], 1)
    qh = [돌(qn[:, a * 512:a * 512 + HD]) for a in range(H)]
    kh = [돌(qn[:, kb + a * HD:kb + a * HD + HD]) for a in range(HKV)]
    vh = [np.ascontiguousarray(qn[:, kb + HKV * HD + a * HD:kb + HKV * HD + a * HD + HD]) for a in range(HKV)]
    o참 = np.concatenate([C.어텐션(qh[a], kh[a // (H // HKV)], vh[a // (H // HKV)], 눈금=f32(0.0625)) for a in range(H)], 1)
    gate = np.concatenate([qn[:, a * 512 + HD:a * 512 + 2 * HD] for a in range(H)], 1)
    og참 = C.게이트(o참, gate)
    자참, 정참 = 자리바이트(og참, 자리수)
    print(f"CPU 참조 {time.time() - t0:.1f} 초", flush=True)

    # ---- GPU: 한 번에 ----
    def 앞단(pq, n, 시작, pqd, pqE, pkd, pkE, pvd, pvE, 길이=None, pc=None, ps=None):
        길이 = 길이 or 최대길이
        띄우기("머리정규화256", ((n * (H + HKV) + 3) // 4, 1), 128, [(1, pq), (1, pqw), (1, pkw), (0, n), (0, H), (0, HKV), (0, 폭)])
        띄우기("RoPEKV정수로256", (((H + 2 * HKV) * 8 + 31) // 32, n), 256,
              [(1, pq), (1, pqd), (1, pqE), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (1, pc or pcos), (1, ps or psin), (0, n), (0, H), (0, HKV), (0, 폭), (0, 시작), (0, 길이)])

    def 어텐션(n, 시작, pqd, pqE, pkd, pkE, pvd, pvE, po, 길이=None):
        길이 = 길이 or 최대길이
        묶음수_ = (길이 + 1023) // 1024
        p묶 = dr.할당(n * H * 묶음수_ * 260 * 4)
        띄우기("어텐션256묶음", (H * 묶음수_, n), 256,
              [(1, pqd), (1, pqE), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (1, p묶), (0, n), (0, H), (0, HKV), (0, 시작), (0, 길이)])
        띄우기("접기256", (H, n), 256, [(1, p묶), (1, po), (0, n), (0, H), (0, 시작), (0, 길이)])

    def 조각어텐션(시작, pqd, pqE, pkd, pkE, pvd, pvE, po, 길이=None):
        """2026-10-25 속도 S1 — 생성 걸음(행 하나): 조각어텐션256 → 조각묶음256 → 접기256(엔진의 큰층띄움들35 와 같은 띄움)"""
        길이 = 길이 or 최대길이
        조각수_, 묶음수_ = (길이 + 63) // 64, (길이 + 1023) // 1024
        p조, p묶 = dr.할당(H * 조각수_ * 260 * 4), dr.할당(H * 묶음수_ * 260 * 4)
        띄우기("조각어텐션256", (H * 64, 1), 256,
              [(1, pqd), (1, pqE), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (1, p조), (0, 1), (0, H), (0, HKV), (0, 시작), (0, 길이)])
        띄우기("조각묶음256", (H * 묶음수_, 1), 256, [(1, p조), (1, p묶), (0, 1), (0, H), (0, 시작), (0, 길이)])
        띄우기("접기256", (H, 1), 256, [(1, p묶), (1, po), (0, 1), (0, H), (0, 시작), (0, 길이)])

    def 행하나q(qd, qE, p):
        """q 자리 [2][행][머리 · 256] · 지수 [행][머리 · 8] 에서 행 p 하나(생성 걸음의 q)"""
        return up(np.ascontiguousarray(qd[:, p:p + 1])), up(np.ascontiguousarray(qE[p:p + 1]))

    pqw, pkw, pcos, psin = up(qw), up(kw), up(cos), up(sin)
    pq = up(qkv)
    pqd, pqE = dr.할당(2 * T * H * HD), dr.할당(T * H * 8 * 8)
    pkd, pkE = dr.할당(2 * HKV * 최대길이 * HD), dr.할당(HKV * 최대길이 * 8 * 8)
    pvd, pvE = dr.할당(2 * HKV * 최대길이 * HD), dr.할당(HKV * 최대길이 * 8 * 8)
    앞단(pq, T, 0, pqd, pqE, pkd, pkE, pvd, pvE)
    dr.맞추기()
    qnG = down(pq, (T, 폭))
    qD, qE_ = C.자리로(np.concatenate(qh, 1), 2)
    kD, kE_ = zip(*[C.자리로(x, 2) for x in kh])
    vD, vE_ = zip(*[C.자리로(x, 2) for x in vh])
    qdG = down(pqd, (2, T, H * HD), np.int8)
    kdG = down(pkd, (2, HKV, 최대길이, HD), np.int8)[:, :, :T]
    vdG = down(pvd, (2, HKV, 최대길이, HD), np.int8)[:, :, :T]
    다 = {"q · k 정규화": 다른칸(qnG, qn),
         "q 자리": 다른칸(qdG, qD.astype(np.int8)), "q 지수": 다른칸(down(pqE, (T, H * 8), np.int64), qE_),
         "키 자리": 다른칸(kdG, np.stack([np.stack([d[j] for d in kD]) for j in range(2)]).astype(np.int8)),
         "키 지수": 다른칸(down(pkE, (HKV, 최대길이, 8), np.int64)[:, :T], np.stack(kE_)),
         "값 자리": 다른칸(vdG, np.stack([np.stack([d[j] for d in vD]) for j in range(2)]).astype(np.int8)),
         "값 지수": 다른칸(down(pvE, (HKV, 최대길이, 8), np.int64)[:, :T], np.stack(vE_))}
    po = dr.할당(T * H * HD * 4)
    어텐션(T, 0, pqd, pqE, pkd, pkE, pvd, pvE, po)
    pog = dr.할당(T * H * HD * 4)
    띄우기("게이트256", ((T * H * HD + 255) // 256, 1), 256, [(1, po), (1, pq), (1, pog), (0, T), (0, H), (0, 폭)])
    깊이 = H * HD
    정보폭 = (T + 63) // 64 * 64
    p자, p정 = dr.할당(자리수 * T * 깊이), dr.할당((깊이 // 32) * 정보폭 * 4)
    띄우기("실수자리로", ((깊이 // 32 + 31) // 32, T), 256, [(1, pog), (1, p자), (1, p정), (0, T), (0, 깊이)])
    dr.맞추기()
    다["어텐션 O / L"] = 다른칸(down(po, (T, H * HD)), o참)
    다["게이트"] = 다른칸(down(pog, (T, H * HD)), og참)
    다["o 투영의 자리"] = 다른칸(down(p자, (자리수, T, 깊이), np.int8), 자참)
    다["o 투영의 정보"] = 다른칸(down(p정, (깊이 // 32, 정보폭), np.int32)[:, :T].T, 정참)
    결과["경우"].append({"경우": f"한 번에({T} 행)", "다른 칸": 다})
    print("한 번에:", 다, flush=True)

    # ---- GPU: 덩이 둘(앞 덩이의 키 · 값을 캐시에) ----
    pkd2, pkE2 = dr.할당(2 * HKV * 최대길이 * HD), dr.할당(HKV * 최대길이 * 8 * 8)
    pvd2, pvE2 = dr.할당(2 * HKV * 최대길이 * HD), dr.할당(HKV * 최대길이 * 8 * 8)
    os_ = []
    for s0, s1 in ((0, T1), (T1, T)):
        n = s1 - s0
        pq2 = up(qkv[s0:s1])
        pqd2, pqE2 = dr.할당(2 * n * H * HD), dr.할당(n * H * 8 * 8)
        앞단(pq2, n, s0, pqd2, pqE2, pkd2, pkE2, pvd2, pvE2)
        po2 = dr.할당(n * H * HD * 4)
        어텐션(n, s0, pqd2, pqE2, pkd2, pkE2, pvd2, pvE2, po2)
        dr.맞추기()
        os_.append(down(po2, (n, H * HD)))
    다2 = {"어텐션 O / L": 다른칸(np.concatenate(os_), o참)}
    결과["경우"].append({"경우": f"덩이 둘({T1} · {T - T1} 행)", "다른 칸": 다2})
    print("덩이 둘:", 다2, flush=True)

    # ---- GPU: 생성 걸음(행 하나 — 조각어텐션256) — 한 번에 의 키 · 값 캐시에서 위치 p 의 행을 CPU 참조와 ----
    qEG = down(pqE, (T, H * 8), np.int64)
    po1 = dr.할당(H * HD * 4)
    다4 = {}
    for p in sorted({x for x in (0, 1, 31, 63, 64, 65, 127, 128, 200, 1023, 1024, 1025, 1087, 1088, T - 1) if 0 <= x < T}):
        pq1, pE1 = 행하나q(qdG, qEG, p)
        조각어텐션(p, pq1, pE1, pkd, pkE, pvd, pvE, po1)
        dr.맞추기()
        다4[f"위치 {p}"] = 다른칸(down(po1, (H * HD,)), o참[p])
    결과["경우"].append({"경우": "생성 걸음(행 하나 — 조각어텐션256)", "다른 칸": 다4})
    print("생성 걸음:", 다4, flush=True)

    # ---- GPU: 생성 걸음의 긴 위치 — 묶음 다섯 · 조각 80(블록이 조각을 64 개 걸러 둘째 조각까지) — 옛 판(어텐션256묶음 · 접기256, 행 하나)과 ----
    T2 = 4300
    길이2 = (T2 + 1023) // 1024 * 1024
    r2 = np.random.default_rng(257)                 # 앞의 경우들(GDN 출력)의 입력은 그대로
    qkv2 = (r2.standard_normal((T2, 폭)) * 10.0 ** r2.uniform(-0.5, 0.5, (1, 폭))).astype(f32)
    cos2, sin2 = C.rope표(길이2, θ, hd=ROT)
    pq2 = up(qkv2)
    pqd2, pqE2 = dr.할당(2 * T2 * H * HD), dr.할당(T2 * H * 8 * 8)
    pkd3, pkE3 = dr.할당(2 * HKV * 길이2 * HD), dr.할당(HKV * 길이2 * 8 * 8)
    pvd3, pvE3 = dr.할당(2 * HKV * 길이2 * HD), dr.할당(HKV * 길이2 * 8 * 8)
    앞단(pq2, T2, 0, pqd2, pqE2, pkd3, pkE3, pvd3, pvE3, 길이2, up(cos2), up(sin2))
    dr.맞추기()
    qd2G, qE2G = down(pqd2, (2, T2, H * HD), np.int8), down(pqE2, (T2, H * 8), np.int64)
    po옛 = dr.할당(H * HD * 4)
    다5 = {}
    for p in (0, 63, 64, 1023, 1024, 1025, 2048, 4095, 4096, 4097, 4159, 4160, T2 - 1):
        pq1, pE1 = 행하나q(qd2G, qE2G, p)
        어텐션(1, p, pq1, pE1, pkd3, pkE3, pvd3, pvE3, po옛, 길이2)
        조각어텐션(p, pq1, pE1, pkd3, pkE3, pvd3, pvE3, po1, 길이2)
        dr.맞추기()
        다5[f"위치 {p}"] = 다른칸(down(po1, (H * HD,)), down(po옛, (H * HD,)))
    결과["경우"].append({"경우": f"생성 걸음의 긴 위치(최대길이 {길이2} — 옛 판과)", "다른 칸": 다5})
    print("생성 걸음 · 긴 위치:", 다5, flush=True)

    # ---- GDN 출력 ----
    o = (r.standard_normal((T, HV, 128)) * 0.3).astype(f32)
    z = (r.standard_normal((T, HV * 128)) * 2).astype(f32)
    nw = (1.0 + 0.1 * r.standard_normal(128)).astype(f32)
    참 = C.곱(C.머리정규화(o.reshape(T, HV * 128), nw, EPS, 128), C.실루(z))
    pgo = dr.할당(T * HV * 128 * 4)
    띄우기("GDN출력", ((T * HV + 3) // 4, 1), 128, [(1, up(o)), (1, up(z)), (1, up(nw)), (1, pgo), (0, T), (0, HV)])
    dr.맞추기()
    다3 = {"GDN 출력": 다른칸(down(pgo, (T, HV * 128)), 참)}
    결과["경우"].append({"경우": "GDN 출력", "다른 칸": 다3})
    print("GDN 출력:", 다3, flush=True)
    결과["모두 0"] = all(v == 0 for c in 결과["경우"] for v in c["다른 칸"].values())
    print("모두 0:", 결과["모두 0"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
