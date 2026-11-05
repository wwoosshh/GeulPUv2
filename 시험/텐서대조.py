#!/usr/bin/env python3
"""llama.cpp 실행의 비트 재현 판정 3-2(문서/라마재현.md) — 실수 텐서 코어 흉내(백엔드/SPIRV/실수텐서.gl)를 4070 Ti 의 mma 와 견준다.

같은 판(커널/텐서시험.gl)을 PTX 로 CUDA 에서(텐서 코어) · SPIR-V 로 Vulkan 에서(흉내) 띄운다. 형식 셋(f16 → f32 · f16 → f16 · bf16 → f32)
마다 워프 2^17 개 × 걸음 8 = mma 1,048,576 번(출력 1억 3천만 개). 피연산자는 조각 묶음(조각 16,384 개씩 — 조각마다 보통 · 넓은 지수 ·
지움(같은 지수) · 특별한 값(0 · 비정규수 · 끝값 · 무한 · NaN)을 섞은 방식)에서 판이 고른다. 덩이(스레드 256 개) 해시를 견주고, 어긋난
덩이는 출력을 통째로 받아 그 mma 를 MMA-Sim 의 numpy 판(계약 밖 — 진단용)과도 견준다. 첫 덩이의 CUDA 출력은 모델과도 견준다(헛통과 막기).

  python -I -X utf8 시험/텐서대조.py <판실행.exe> <텐서시험.ptx> <판실행V.exe> <텐서시험.spvs> <작업 폴더>
        [--장치 <Vulkan 장치>] [--덩이 <띄움마다 스레드 2^k — 기본 18>] [--형식 반실수,반누적,BF16] [--워프 <2^k — 기본 17>]
        [--모델 <mma_np.py 가 있는 폴더>]
마지막 줄 "모두 같음: True|False".
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from 판실행 import 판실행    # noqa: E402

NF = 16384
U64 = (1 << 64) - 1


def 반쪽들(rng, n, 방식, bf):
    """방식(조각마다 0 … 15)에 따라 f16 · bf16 비트 n 개씩 — [조각, n]"""
    nf = len(방식)
    eb, ew, 가수비트 = (127, 254, 7) if bf else (15, 30, 10)
    부 = rng.integers(0, 2, (nf, n)) << 15
    가 = rng.integers(0, 1 << 가수비트, (nf, n))
    보통E = rng.integers(eb - 6, eb + 7, (nf, n))
    넓은E = rng.integers(0, ew + 1, (nf, n))
    섞 = rng.integers(0, 100, (nf, n))
    E = np.where(섞 < 85, 보통E, np.where(섞 < 95, 넓은E, 0))
    m = 방식[:, None]
    E = np.where((m >= 12) & (m <= 13), 넓은E, E)
    지움E = rng.integers(eb - 3, eb + 4, (nf, 1))
    E = np.where(m == 14, 지움E, E)
    v = 부 | (E << 가수비트) | 가
    특별 = np.array([0x7BFF, 0x7C00, 0x7E00, 0x0400, 0x0001, 0x8000] if not bf else [0x7F7F, 0x7F80, 0x7FC0, 0x0080, 0x0001, 0x8000])
    특 = 특별[rng.integers(0, len(특별), (nf, n))] | (부 & 0x8000)
    v = np.where((m == 15) & (rng.integers(0, 10, (nf, n)) == 0), 특, v)
    return v.astype(np.uint32)


def 짧은실수들(rng, n, 방식):
    nf = len(방식)
    부 = rng.integers(0, 2, (nf, n)).astype(np.uint64) << 31
    가 = rng.integers(0, 1 << 23, (nf, n)).astype(np.uint64)
    E = np.where(rng.integers(0, 100, (nf, n)) < 85, rng.integers(115, 140, (nf, n)), rng.integers(0, 255, (nf, n))).astype(np.uint64)
    E = np.where(rng.integers(0, 100, (nf, n)) < 5, 0, E)
    v = 부 | (E << 23) | 가
    특별 = np.array([0x7F7FFFFF, 0x7F800000, 0x7FC00000, 0x00800000, 0x00000001, 0x80000000], np.uint64)
    특 = 특별[rng.integers(0, len(특별), (nf, n))] | (부 & 0x80000000)
    v = np.where((방식[:, None] == 15) & (rng.integers(0, 10, (nf, n)) == 0), 특, v)
    return v.astype(np.uint32)


def 묶음(rng, 형식):
    """조각 묶음 A[NF, 32, 4] · B[NF, 32, 2] · C[NF, 32, 4 또는 2] (u32 낱말)"""
    bf = 형식 == "BF16"
    방식 = [rng.integers(0, 16, NF) for _ in range(3)]
    h = 반쪽들(rng, 32 * 8, 방식[0], bf).reshape(NF, 32, 4, 2)
    A = (h[..., 0] | (h[..., 1] << 16)).astype(np.uint32)
    h = 반쪽들(rng, 32 * 4, 방식[1], bf).reshape(NF, 32, 2, 2)
    B = (h[..., 0] | (h[..., 1] << 16)).astype(np.uint32)
    if 형식 == "반누적":
        h = 반쪽들(rng, 32 * 4, 방식[2], False).reshape(NF, 32, 2, 2)
        C = (h[..., 0] | (h[..., 1] << 16)).astype(np.uint32)
    else:
        C = 짧은실수들(rng, 32 * 4, 방식[2]).reshape(NF, 32, 4)
    return A, B, C


def 고른조각(i):
    z = ((i + 1) * 0x9E3779B97F4A7C15) & U64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & U64
    z = z ^ (z >> 31)
    return z & 16383, (z >> 14) & 16383, (z >> 28) & 16383


def 행렬(A, B, C, fa, fb, fc, 형식):
    """조각 → mma_np 의 A[16,16] · B[16,8] · C[16,8](float64)"""
    kind = "bf16" if 형식 == "BF16" else "f16"
    from mma_np import to_f64
    Am = np.zeros((16, 16), np.uint32)
    Bm = np.zeros((16, 8), np.uint32)
    Cm = np.zeros((16, 8), np.uint32)
    for l in range(32):
        g, t = l >> 2, l & 3
        a = A[fa, l]
        for j, (r, c) in enumerate([(g, 2 * t), (g + 8, 2 * t), (g, 2 * t + 8), (g + 8, 2 * t + 8)]):
            Am[r, c], Am[r, c + 1] = a[j] & 0xFFFF, a[j] >> 16
        b = B[fb, l]
        Bm[2 * t, g], Bm[2 * t + 1, g] = b[0] & 0xFFFF, b[0] >> 16
        Bm[2 * t + 8, g], Bm[2 * t + 9, g] = b[1] & 0xFFFF, b[1] >> 16
        c = C[fc, l]
        if 형식 == "반누적":
            Cm[g, 2 * t], Cm[g, 2 * t + 1] = c[0] & 0xFFFF, c[0] >> 16
            Cm[g + 8, 2 * t], Cm[g + 8, 2 * t + 1] = c[1] & 0xFFFF, c[1] >> 16
        else:
            Cm[g, 2 * t], Cm[g, 2 * t + 1], Cm[g + 8, 2 * t], Cm[g + 8, 2 * t + 1] = c
    return to_f64(Am, kind), to_f64(Bm, kind), to_f64(Cm, "f16" if 형식 == "반누적" else "f32")


def 레인출력(D, 형식):
    """mma_np 의 D[16,8] → 레인마다 낱말 넷(반누적은 둘 + 0 둘) — 판의 원값 차례"""
    out = np.zeros((32, 4), np.uint32)
    bits = D.view(np.uint16).astype(np.uint32) if 형식 == "반누적" else D.view(np.uint32)
    for l in range(32):
        g, t = l >> 2, l & 3
        if 형식 == "반누적":
            out[l, 0] = bits[g, 2 * t] | (bits[g, 2 * t + 1] << 16)
            out[l, 1] = bits[g + 8, 2 * t] | (bits[g + 8, 2 * t + 1] << 16)
        else:
            out[l] = [bits[g, 2 * t], bits[g, 2 * t + 1], bits[g + 8, 2 * t], bits[g + 8, 2 * t + 1]]
    return out


def 모델(A, B, C, w, it, 형식):
    from mma_np import mma_m16n8k16
    fa, fb, fc = 고른조각(w * 8 + it)
    Am, Bm, Cm = 행렬(A, B, C, fa, fb, fc, 형식)
    with np.errstate(all="ignore"):
        D = mma_m16n8k16(Am, Bm, Cm, "bf16" if 형식 == "BF16" else "f16", "f16" if 형식 == "반누적" else "f32")
    return 레인출력(D, 형식)


class 판들:
    def __init__(self, d, A, B, C):
        self.d = d
        self.pA, self.pB, self.pC = d.할당(A.nbytes), d.할당(B.nbytes), d.할당(C.nbytes)
        d.올리기(self.pA, A)
        d.올리기(self.pB, B)
        d.올리기(self.pC, C)

    def 덩이해시(self, 형식, 워프로그2, 덩이로그2):
        d = self.d
        T = 32 << 워프로그2
        th = d.할당(T * 8)
        bh = d.할당((T >> 8) * 8)
        헛 = d.할당(4)
        n = 1 << min(덩이로그2, 워프로그2 + 5)
        for s in range(0, T, n):
            d.띄우기(f"텐서판_{형식}", (n // 256,), (256,), [self.pA, self.pB, self.pC, 헛, th + s * 8, s, n, 0])
        d.띄우기("해시묶기", ((T >> 8) // 256,), (256,), [th, bh, T >> 8])
        out = np.zeros(T >> 8, np.uint64)
        d.내리기(bh, out)
        d.맞추기()
        return out

    def 원값(self, 형식, 덩이):
        d = self.d
        raw = d.할당(256 * 8 * 4 * 4)
        th = d.할당(256 * 8)
        d.띄우기(f"텐서판_{형식}", (1,), (256,), [self.pA, self.pB, self.pC, raw, th, 덩이 * 256, 256, 1])
        out = np.zeros((256, 8, 4), np.uint32)
        d.내리기(raw, out)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("쿠다exe")
    ap.add_argument("ptx")
    ap.add_argument("벌칸exe")
    ap.add_argument("spvs")
    ap.add_argument("폴더")
    ap.add_argument("--장치", default="NVIDIA")
    ap.add_argument("--덩이", type=int, default=18)
    ap.add_argument("--워프", type=int, default=17)
    ap.add_argument("--형식", default="반실수,반누적,BF16")
    ap.add_argument("--모델", default=r"E:\geulpu-work\xreplay\scripts")
    args = ap.parse_args()
    sys.path.insert(0, args.모델)
    rng = np.random.default_rng(20261105)
    모두 = True
    기록 = []
    for 형식 in args.형식.split(","):
        A, B, C = 묶음(rng, 형식)
        dC = 판실행(args.쿠다exe, args.ptx, args.폴더)
        dV = 판실행(args.벌칸exe, args.spvs, args.폴더, 장치=args.장치)
        pc, pv = 판들(dC, A, B, C), 판들(dV, A, B, C)
        t0 = time.time()
        hc = pc.덩이해시(형식, args.워프, args.덩이)
        t1 = time.time()
        hv = pv.덩이해시(형식, args.워프, args.덩이)
        t2 = time.time()
        다른 = np.nonzero(hc != hv)[0]
        # 헛통과 막기: 첫 덩이의 CUDA 출력이 MMA-Sim 과 같은가(mma 64 번)
        rc0 = pc.원값(형식, 0)
        모델틀림 = 0
        for w in range(8):
            for it in range(8):
                m = 모델(A, B, C, w, it, 형식)
                got = rc0[32 * w:32 * w + 32, it, :]
                모델틀림 += int(np.count_nonzero(got[:, :2] != m[:, :2]) if 형식 == "반누적" else np.count_nonzero(got != m))
        출력수 = (32 << args.워프) * 8 * 4            # 레인마다 mma 하나에 출력 넷(반누적은 f16 둘씩 낱말 둘)
        같음 = len(다른) == 0 and len(np.unique(hc)) > len(hc) // 2 and 모델틀림 == 0
        모두 = 모두 and 같음
        print(f"{형식}: mma {(1 << args.워프) * 8} 번 · 출력 {출력수:,} 개 · 덩이 {len(hc)} 중 어긋남 {len(다른)} · 첫 덩이의 CUDA 대 MMA-Sim 틀림 "
              f"{모델틀림} · CUDA {t1 - t0:.1f} 초 · Vulkan {t2 - t1:.1f} 초", flush=True)
        보기 = []
        for b in 다른[:3]:
            rc, rv = pc.원값(형식, int(b)), pv.원값(형식, int(b))
            틀린 = np.argwhere(rc != rv)
            for th_, it, j in 틀린[:4]:
                t = int(b) * 256 + int(th_)
                w = t >> 5
                m = 모델(A, B, C, w, int(it), 형식)[t & 31, j]
                보기.append(f"워프 {w} 걸음 {int(it)} 레인 {t & 31} 칸 {int(j)}: CUDA {int(rc[th_, it, j]):08x} · Vulkan {int(rv[th_, it, j]):08x} · "
                          f"MMA-Sim {int(m):08x}")
            print(f"  덩이 {int(b)}: 어긋난 출력 낱말 {len(틀린)} 개 — " + " | ".join(보기[-4:]), flush=True)
        기록.append({"형식": 형식, "어긋난덩이": int(len(다른)), "모델틀림": 모델틀림, "출력": 출력수, "초": [round(t1 - t0, 2), round(t2 - t1, 2)], "보기": 보기})
        dC.닫기()
        dV.닫기()
    with open(os.path.join(args.폴더, "텐서대조.json"), "w", encoding="utf-8") as f:
        json.dump({"판": 기록}, f, ensure_ascii=False, indent=1)
    print(f"모두 같음: {모두}")


if __name__ == "__main__":
    main()
