#!/usr/bin/env python3
"""llama.cpp 실행의 비트 재현 판정 3-1(문서/라마재현.md) — 근사 명령 흉내(백엔드/SPIRV/근사.gl)를 하드웨어 명령과 견준다.

같은 판(커널/근사시험.gl)을 PTX 로 CUDA 에서(sm_89 의 MUFU · add.rz · cvt) · SPIR-V 로 Vulkan 에서(정수 셈의 흉내) 띄운다. 판마다 입력
2^32 개(영쪽더하기 는 쌍 2^30 개)의 결과를 스레드마다(입력 256 개) · 덩이마다(스레드 256 개 = 입력 65,536 개) 해시로 묶어 덩이 해시를
견주고, 어긋난 덩이는 결과를 통째로 받아 입력을 찾는다. 오류 칸(반실수로 넘침 — 비트 4)도 견준다.

  python -I -X utf8 시험/근사대조.py <판실행.exe> <근사시험.ptx> <판실행V.exe> <근사시험.spvs> <작업 폴더>
        [--장치 <Vulkan 장치 이름의 일부>] [--덩이 <띄움마다 입력 2^k — 기본 26>] [--판 ex2,rcp,…] [--쪽 둘|쿠다|벌칸]
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

판들 = [("ex2", 32), ("rcp", 32), ("rsqrt", 32), ("lg2", 32), ("sin", 32), ("cos", 32), ("지수", 32), ("f16", 32), ("bf16", 32),
       ("반실수아래", 32), ("반실수위", 32), ("영쪽더하기", 30), ("반실수왕복", 32)]
U64 = (1 << 64) - 1


def 덩이해시(d, 판, 로그2, 덩이로그2):
    """판의 입력 2^로그2 개 → 덩이 해시(입력 65,536 개마다 하나) · 오류 칸"""
    N = 1 << 로그2
    T = N >> 8
    th = d.할당(T * 8)
    bh = d.할당((T >> 8) * 8)
    헛 = d.할당(4)
    덧 = []
    if 판 == "영쪽더하기":
        덧 = [d.할당(특별.nbytes)]
        d.올리기(덧[0], 특별)
    C = 1 << min(덩이로그2, 로그2)
    for s in range(0, N, C):
        n = C >> 8
        d.띄우기(f"근사판_{판}", (max(n // 256, 1),), (min(n, 256),), [헛, th + (s >> 8) * 8, s, n, 0] + 덧)
    d.띄우기("해시묶기", ((T >> 8) // 256,), (256,), [th, bh, T >> 8])
    out = np.zeros(T >> 8, np.uint64)
    d.내리기(bh, out)
    칸 = d.맞추기()
    d.놓기()
    return out, 칸


def 원값(d, 판, 덩이):
    raw = d.할당(65536 * 4)
    th = d.할당(256 * 8)
    덧 = []
    if 판 == "영쪽더하기":
        덧 = [d.할당(특별.nbytes)]
        d.올리기(덧[0], 특별)
    d.띄우기(f"근사판_{판}", (1,), (256,), [raw, th, 덩이 * 65536, 256, 1] + 덧)
    out = np.zeros(65536, np.uint32)
    d.내리기(raw, out)
    d.놓기()
    return out


def 값확인(판, raw, 덩이):
    """헛통과를 막는다: CUDA(하드웨어)가 낸 덩이 하나의 값이 그 함수의 값과 가까운가 — 변환은 numpy 와 비트까지. 틀린 입력 수"""
    u = (np.arange(65536, dtype=np.uint64) + 덩이 * 65536).astype(np.uint32)
    x = u.view(np.float32).astype(np.float64)
    y = raw.view(np.float32).astype(np.float64)
    if 판 in ("f16", "반실수왕복"):
        h = u.view(np.float32).astype(np.float16)
        기대 = h.view(np.uint16).astype(np.uint32) * 65537 if 판 == "f16" else h.astype(np.float32).view(np.uint32)
        return int(np.count_nonzero(raw != 기대))
    if 판 == "bf16":
        b = ((u.astype(np.uint64) + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32)
        return int(np.count_nonzero(raw != b * 65537))
    if 판 in ("반실수아래", "반실수위"):
        h = ((u & 0xFFFF) if 판 == "반실수아래" else (u >> 16)).astype(np.uint16)
        기대 = h.view(np.float16).astype(np.float32).view(np.uint32)
        기대 = np.where(((h & 0x7C00) == 0x7C00) & ((h & 0x3FF) != 0), np.uint32(0x7FFFFFFF), 기대)   # sm_89: NaN → 0x7FFFFFFF
        return int(np.count_nonzero(raw != 기대))
    f = {"ex2": np.exp2, "rcp": lambda v: 1 / v, "rsqrt": lambda v: 1 / np.sqrt(v), "lg2": np.log2, "sin": np.sin, "cos": np.cos,
         "지수": np.exp}[판]
    with np.errstate(all="ignore"):
        참 = f(x)
    쓸 = np.isfinite(참) & np.isfinite(x) & (np.abs(참) > 1e-30) & (np.abs(참) < 1e30)
    return int(np.count_nonzero(쓸 & (np.abs(y - 참) > 1e-5 * np.abs(참) + 1e-6)))


특별 = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 1, 0x807FFFFF, 0x7F7FFFFF, 0x00800000, 0x3F800000, 0xFF7FFFFF,
                0x80800000, 0x7FFFFFFF, 0xFFC00001, 0x80000001, 0x00400000], np.uint32)   # 판의 표(특별) — 0 · 무한 · NaN · 비정규수 · 끝값


def 쌍(i):
    """커널의 영쪽더하기 입력 (a, b) — 번호 i(정수)에서"""
    z = ((i + 1) * 0x9E3779B97F4A7C15) & U64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & U64
    z = z ^ (z >> 31)
    a, hi, m = z & 0xFFFFFFFF, (z >> 32) & 0xFFFFFFFF, i & 3
    if m == 0:
        return a, hi
    if m == 1:
        return a, (a ^ 0x80000000) ^ ((z >> 40) & 32767)
    if m == 2:
        e = max(((a >> 23) & 255) - (hi & 31), 0)
        return a, (e << 23) | ((z >> 37) & 0x7FFFFF) | (hi & 0x80000000)
    return a, int(특별[hi & 15])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("쿠다exe")
    ap.add_argument("ptx")
    ap.add_argument("벌칸exe")
    ap.add_argument("spvs")
    ap.add_argument("폴더")
    ap.add_argument("--장치", default="NVIDIA")
    ap.add_argument("--덩이", type=int, default=26)
    ap.add_argument("--판", default="")
    args = ap.parse_args()
    고른 = [x for x in 판들 if not args.판 or x[0] in args.판.split(",")]
    dC = 판실행(args.쿠다exe, args.ptx, args.폴더)
    dV = 판실행(args.벌칸exe, args.spvs, args.폴더, 장치=args.장치)
    print(f"CUDA: {dC.장치} · Vulkan: {dV.장치} · 띄움마다 입력 2^{args.덩이}", flush=True)
    모두 = True
    기록 = []
    for 판, 로그2 in 고른:
        t0 = time.time()
        hc, ec = 덩이해시(dC, 판, 로그2, args.덩이)
        t1 = time.time()
        hv, ev = 덩이해시(dV, 판, 로그2, args.덩이)
        t2 = time.time()
        다른 = np.nonzero(hc != hv)[0]
        여럿 = len(np.unique(hc))
        확인덩이 = 0x3F80 if 로그2 == 32 else 0
        틀린값 = 값확인(판, 원값(dC, 판, 확인덩이), 확인덩이) if 판 != "영쪽더하기" else 0
        최소여럿 = 1 if 판 == "반실수아래" else len(hc) // 16       # 반실수아래 는 덩이마다 같은 값들(아래 16 비트가 모두 한 번씩)
        같음 = len(다른) == 0 and ec == ev and 여럿 >= 최소여럿 and 틀린값 == 0
        모두 = 모두 and 같음
        줄 = (f"{판}: 입력 2^{로그2} · 덩이 {len(hc)} 중 어긋남 {len(다른)} · 덩이 해시 {여럿} 가지 · 값 확인(덩이 {확인덩이:#x}) 틀림 {틀린값} · "
             f"오류 칸 CUDA {ec} · Vulkan {ev} · CUDA {t1 - t0:.1f} 초 · Vulkan {t2 - t1:.1f} 초")
        print(줄, flush=True)
        보기 = []
        for b in 다른[:4]:
            rc, rv = 원값(dC, 판, int(b)), 원값(dV, 판, int(b))
            for k in np.nonzero(rc != rv)[0][:6]:
                i = int(b) * 65536 + int(k)
                입력 = f"{쌍(i)[0]:08x} {쌍(i)[1]:08x}" if 판 == "영쪽더하기" else f"{i:08x}"
                보기.append(f"입력 {입력} → CUDA {int(rc[k]):08x} · Vulkan {int(rv[k]):08x}")
            print(f"  덩이 {int(b)}: 어긋난 입력 {int(np.count_nonzero(rc != rv))} 개 — " + " | ".join(보기[-6:]), flush=True)
        기록.append({"판": 판, "입력": 1 << 로그2, "어긋난덩이": int(len(다른)), "오류칸": [ec, ev], "초": [round(t1 - t0, 2), round(t2 - t1, 2)],
                   "보기": 보기})
    dC.닫기()
    dV.닫기()
    with open(os.path.join(args.폴더, "근사대조.json"), "w", encoding="utf-8") as f:
        json.dump({"쿠다": dC.장치, "벌칸": dV.장치, "판": 기록}, f, ensure_ascii=False, indent=1)
    print(f"모두 같음: {모두}")


if __name__ == "__main__":
    main()
