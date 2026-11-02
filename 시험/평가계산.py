#!/usr/bin/env python3
"""모델 평가 1-2(문서/모델평가.md 3.3 · 3.7) — 지표 계산(앱/평가계산.gl)의 파이썬 참조와 GPU 없는 대조(엔진/시험_평가계산.gl).

    python -I -X utf8 시험/평가계산.py check <시험_평가계산.exe>      (빠른 점검 · CI)

사례(씨앗 고정): 지수(−700 … 10 과 경계) · 로그(10⁻³⁰⁰ … 1) · KL(K 1 · 20 · 64 · 128 — 무작위 분포, 꼬리가 아주 작은 분포, 평가 쪽 합이
1 을 넘는 분포) · 꼬리 · 백분위수(무작위 · 0 과 음수가 섞임) · 부호 검정(b · c 0 … 60). 모두 비트까지 같아야 한다.
이 파일의 함수(지수 · 로그 · KL · 꼬리 · 백분위 · 부호검정)는 평가 결과를 다시 계산하는 시험도 쓴다.
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "계약"))
import 로그확률 as LP          # noqa: E402

바닥 = LP.비트실수(0x3D719799812DEA11)          # 10⁻¹²


def 지수(a):
    return 0.0 if a < -700.0 else LP.적힌지수하나(a)


def 로그(x):
    return LP.적힌로그(x)


def KL(lr, lq):
    머리 = ps = qs = 0.0
    for a, b in zip(lr, lq):
        p = 지수(a)
        q = 지수(b)
        머리 = 머리 + p * (a - b)
        ps = ps + p
        qs = qs + q
    pt = 1.0 - ps
    qt = 1.0 - qs
    if pt < 바닥:
        return 머리
    if qt < 바닥:
        qt = 바닥
    return 머리 + pt * (로그(pt) - 로그(qt))


def 꼬리(lr):
    ps = 0.0
    for a in lr:
        ps = ps + 지수(a)
    pt = 1.0 - ps
    return 0.0 if pt < 0.0 else pt


def 백분위(x, q):
    n = len(x)
    if n < 1:
        return 0.0
    a = sorted(LP.실수비트(float(v)) if v > 0.0 else 0 for v in x)
    순 = min(max((q * n + 99) // 100, 1), n)
    return LP.비트실수(a[순 - 1])


def 부호검정(b, c):
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    t = 1.0
    for _ in range(n):
        t = t * 0.5
    s = 0.0
    for i in range(m + 1):
        s = s + t
        t = t * float(n - i) / float(i + 1)
    p = 2.0 * s
    return 1.0 if p > 1.0 else p


def _로그소프트(z):
    z = np.asarray(z, np.float64)
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def 사례들():
    r = np.random.default_rng(20261102)
    out = []                                                     # (종류, 인자들, 참값)
    for a in list(np.linspace(-700.0, 10.0, 211)) + [0.0, -0.0, -40.0, -1e-12, 1e-12, -699.9999, -700.0, -700.0000001]:
        out.append((3, [float(a)], 지수(float(a))))
    for x in list(10.0 ** r.uniform(-300, 0, 200)) + [1.0, 0.5, 1e-12, 0.999999999, 2.2250738585072014e-308]:
        out.append((4, [float(x)], 로그(float(x))))
    for K in (1, 20, 64, 128):
        for 몇째 in range(40):
            V = 2000
            if 몇째 % 10 == 0:
                z = r.normal(0, 1, V)
                z[r.integers(V)] += 60.0                        # 꼬리가 아주 작은 분포(pt < 10⁻¹²)
            else:
                z = r.normal(0, 1 + 몇째 % 4, V)
            lp = _로그소프트(z)
            idx = np.lexsort((np.arange(V), -lp))[:K]
            lr = [float(v) for v in lp[idx]]
            if 몇째 % 10 == 5:
                lq = [v + 1e-3 for v in lr]                     # 평가 쪽 합이 1 을 넘는다(qt < 0 → 10⁻¹²)
            else:
                lq = [float(v) for v in _로그소프트(z + r.normal(0, 0.3, V))[idx]]
            out.append((0, [lr, lq], KL(lr, lq)))
            out.append((5, [lr], 꼬리(lr)))
    for n in (1, 2, 3, 10, 101, 1000):
        for q in (50, 99, 100):
            x = [float(v) for v in r.exponential(0.1, n)]
            if n > 3:
                x[0] = 0.0
                x[1] = -1e-15
                x[2] = -0.0
            out.append((1, [x, q], 백분위(x, q)))
    for b in range(0, 61, 3):
        for c in range(0, 61, 7):
            out.append((2, [b, c], 부호검정(b, c)))
    return out


def 묶기(사례):
    w = [len(사례)]
    for 종류, 인자, _ in 사례:
        w.append(종류)
        if 종류 == 0:
            lr, lq = 인자
            w += [len(lr)] + [LP.실수비트(v) for v in lr] + [LP.실수비트(v) for v in lq]
        elif 종류 == 5:
            w += [len(인자[0])] + [LP.실수비트(v) for v in 인자[0]]
        elif 종류 == 1:
            x, q = 인자
            w += [len(x)] + [LP.실수비트(v) for v in x] + [q]
        elif 종류 == 2:
            w += 인자
        else:
            w.append(LP.실수비트(인자[0]))
    return struct.pack(f"<{len(w)}Q", *[v & ((1 << 64) - 1) for v in w])


def check(exe):
    사례 = 사례들()
    with tempfile.TemporaryDirectory() as d:
        i, o = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
        open(i, "wb").write(묶기(사례))
        r = subprocess.run([exe, i, o], capture_output=True, stdin=subprocess.DEVNULL, timeout=600)
        if r.returncode != 0 or not os.path.exists(o):
            print(f"시험_평가계산 실패({r.returncode}): {(r.stdout + r.stderr).decode('utf-8', 'replace')[-300:]}")
            return 1
        got = struct.unpack(f"<{len(사례)}Q", open(o, "rb").read())
    이름 = {0: "KL", 1: "백분위", 2: "부호검정", 3: "지수", 4: "로그", 5: "꼬리"}
    다름 = {}
    for (종류, _, 참), g in zip(사례, got):
        if g != LP.실수비트(참):
            다름[이름[종류]] = 다름.get(이름[종류], 0) + 1
    print(f"평가 지표의 계산: 사례 {len(사례)} · 다름 {sum(다름.values())}" + (f" — {다름}" if 다름 else ""))
    return 0 if not 다름 else 1


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 3 or sys.argv[1] != "check":
        print(__doc__)
        return 2
    return check(os.path.abspath(sys.argv[2]))


if __name__ == "__main__":
    sys.exit(main())
