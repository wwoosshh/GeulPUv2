#!/usr/bin/env python3
"""38단계 — 로그확률 계약(문서/계약.md 5.2)의 CPU 참조(numpy). 서버(호스트/로그확률.gl)와 f64 비트까지 같아야 한다(38a).

로짓 행 z(엔진이 낸 짧은실수 V 개)에서, 모든 셈은 f64 로 적은 차례대로(곱해더하기 없음):

  1. j₁ = z 가 가장 큰 번호 중 가장 작은 것, m = f64(z_{j₁}). a_j = f64(z_j) − m.
  2. w_j = exp_w(a_j)(5절의 적힌 exp) — a_j ≥ −40 인 것만, 아니면 0(밝힌 손실 — e⁻⁴⁰ 아래).
  3. 어휘를 16 구간으로: B = ⌈V / 16⌉, 구간 r = [rB, min(V, (r + 1)B)). 구간마다 0.0 에서 시작해 번호 차례로
     S_r = S_r + w_j, T_r = T_r + w_j · a_j(곱은 반올림 하나). S = (…((S₀ + S₁) + S₂) + …) + S₁₅, T 도 같게.
  4. L = log_w(S)(적힌 log — 아래), logprob_j = a_j − L, 엔트로피 H = L − T / S.
  5. 상위 k: 번호를 (z 내림차순, 같으면 번호 오름차순 — −0 과 +0 은 같은 값)으로 늘어놓은 앞 k 개.

분포는 모델의 것(온도 1 · 거르기 없음)이다 — 뽑기(5절)의 분포가 아니다.

적힌 log(log_w) — fdlibm 의 e_log.c 와 같은 차례 · 상수(비트로 정함). x > 0 이고 정규수(2⁻¹⁰²² 이상 · 유한)만. S 는 1 이상
(가장 큰 번호의 w = exp_w(0) = 1) V 이하다.

  python -I -X utf8 계약/로그확률.py      (자체 시험 — 시험/계약자체.py 가 같은 것을 부른다)
"""
import math
import struct

import numpy as np

M64 = (1 << 64) - 1


def 비트실수(b):
    return struct.unpack("<d", struct.pack("<Q", b & M64))[0]


def 실수비트(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


# 5절(뽑기)의 적힌 exp — 상수는 비트로
INV_LN2 = 비트실수(0x3FF71547652B82FE)
LN2_HI = 비트실수(0x3FE62E42FEE00000)
LN2_LO = 비트실수(0x3DEA39EF35793C76)
역계승 = [1.0 / float(math.factorial(n)) for n in range(14)]      # f64(1) / f64(n!) — n! 은 f64 로 정확(13! < 2⁵³)

# 적힌 log — fdlibm 의 Lg1 … Lg7 과 1/3(그 글자 0.33333333333333333 의 f64)
LG1 = 비트실수(0x3FE5555555555593)
LG2 = 비트실수(0x3FD999999997FA04)
LG3 = 비트실수(0x3FD2492494229359)
LG4 = 비트실수(0x3FCC71C51D8E78AF)
LG5 = 비트실수(0x3FC7466496CB03DE)
LG6 = 비트실수(0x3FC39A09D078C69F)
LG7 = 비트실수(0x3FC2F112DF3E5244)
셋째 = 비트실수(0x3FD5555555555555)

구간수 = 16
바닥 = -40.0


def 적힌지수하나(a):
    """exp_w(a) — a ∈ [−40, 0] 의 f64 하나(5절 그대로: k = ⌊a·(1/ln2) + 0.5⌋, r = (a − k·ln2높) − k·ln2낮, 호너, × 2^k)."""
    y = a * INV_LN2 + 0.5
    k = math.floor(y)
    kf = float(k)
    r = (a - kf * LN2_HI) - kf * LN2_LO
    p = 역계승[13]
    for n in range(12, -1, -1):
        p = p * r + 역계승[n]
    return p * math.ldexp(1.0, k)


def 적힌지수(a):
    """exp_w 를 배열에 — 연산 하나마다 반올림 하나(numpy 는 곱 · 더를 합치지 않는다). [−40, 0] 밖의 값은 쓰지 않는다(가려진다)."""
    a = np.asarray(a, np.float64)
    with np.errstate(all="ignore"):
        y = a * INV_LN2 + 0.5
        k = np.floor(y)
        r = (a - k * LN2_HI) - k * LN2_LO
        p = np.full(a.shape, 역계승[13])
        for n in range(12, -1, -1):
            p = p * r + 역계승[n]
        kk = np.clip(np.nan_to_num(k), -1000, 1000).astype(np.int64)
        return np.ldexp(p, kk)                    # p · 2^k (정확 — 2 의 거듭제곱 곱)


def 적힌로그(x):
    """log_w(x) — fdlibm __ieee754_log 의 차례 그대로(정규수 x > 0)."""
    b = 실수비트(x)
    hx = (b >> 32) & 0xFFFFFFFF
    lx = b & 0xFFFFFFFF
    if hx < 0x00100000 or hx >= 0x7FF00000 or (hx & 0x80000000):
        raise ValueError(f"적힌로그: 정규수 x > 0 만 — {x!r}")
    k = (hx >> 20) - 1023
    hx &= 0x000FFFFF
    i = (hx + 0x95F64) & 0x100000
    xn = 비트실수(((hx | (i ^ 0x3FF00000)) << 32) | lx)       # x 또는 x/2 를 [√2/2, √2) 로
    k += i >> 20
    f = xn - 1.0
    dk = float(k)
    if (0x000FFFFF & (2 + hx)) < 3:                            # |f| < 2⁻²⁰
        if f == 0.0:
            if k == 0:
                return 0.0
            return dk * LN2_HI + dk * LN2_LO
        R = f * f * (0.5 - 셋째 * f)
        if k == 0:
            return f - R
        return dk * LN2_HI - ((R - dk * LN2_LO) - f)
    s = f / (2.0 + f)
    z = s * s
    i = hx - 0x6147A
    w = z * z
    j = 0x6B851 - hx
    t1 = w * (LG2 + w * (LG4 + w * LG6))
    t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)))
    i |= j
    R = t2 + t1
    if i > 0:
        hfsq = 0.5 * f * f
        if k == 0:
            return f - (hfsq - s * (hfsq + R))
        return dk * LN2_HI - ((hfsq - (s * (hfsq + R) + dk * LN2_LO)) - f)
    if k == 0:
        return f - s * (f - R)
    return dk * LN2_HI - ((s * (f - R) - dk * LN2_LO) - f)


def 구간들(V):
    B = (V + 구간수 - 1) // 구간수
    return [(min(V, r * B), min(V, (r + 1) * B)) for r in range(구간수)]


def _차례합(v):
    """0.0 에서 시작해 차례로 더한 값(add.accumulate 는 한 칸씩 차례로 더한다 — 쌍 합이 아니다)."""
    if len(v) == 0:
        return 0.0
    return float(np.add.accumulate(np.concatenate(([0.0], v)))[-1])


def 로그확률행(z, 위수=0):
    """로짓 행 하나의 로그확률. 사전: 큰(j₁) · S · T · L · H · 로그확률(f64 [V]) · 위([(번호, 로그확률)] k 개)."""
    z = np.ascontiguousarray(np.asarray(z, np.float32))
    V = int(z.shape[0])
    큰 = int(np.argmax(z))                     # 같은 값이면 앞의 것
    m = float(z[큰])
    a = z.astype(np.float64) - m
    가 = a >= 바닥
    w = np.where(가, 적힌지수(np.where(가, a, 0.0)), 0.0)
    wa = np.where(가, w * a, 0.0)
    S = T = None
    for lo, hi in 구간들(V):
        s, t = _차례합(w[lo:hi]), _차례합(wa[lo:hi])
        if S is None:
            S, T = s, t
        else:
            S, T = S + s, T + t
    L = 적힌로그(S)
    H = L - T / S
    lp = a - L
    위 = []
    if 위수 > 0:
        차례 = np.lexsort((np.arange(V), -z.astype(np.float64)))[:위수]
        위 = [(int(j), float(lp[j])) for j in 차례]
    return {"큰": 큰, "S": S, "T": T, "L": L, "H": H, "로그확률": lp, "위": 위}


def 로그확률행느린(z, 위수=0):
    """같은 계약을 한 칸씩(파이썬 실수) — 자체 시험에서 numpy 판과 비트까지 견준다(작은 V)."""
    zf = [float(x) for x in np.asarray(z, np.float32)]
    V = len(zf)
    큰 = 0
    for j in range(1, V):
        if zf[j] > zf[큰]:
            큰 = j
    m = zf[큰]
    a = [x - m for x in zf]
    Sr, Tr = [], []
    for lo, hi in 구간들(V):
        s = 0.0
        t = 0.0
        for j in range(lo, hi):
            if a[j] >= 바닥:
                w = 적힌지수하나(a[j])
                s = s + w
                t = t + w * a[j]
        Sr.append(s)
        Tr.append(t)
    S, T = Sr[0], Tr[0]
    for r in range(1, 구간수):
        S = S + Sr[r]
        T = T + Tr[r]
    L = 적힌로그(S)
    H = L - T / S
    lp = [x - L for x in a]
    차례 = sorted(range(V), key=lambda j: (-(zf[j] + 0.0), j))[:위수]
    return {"큰": 큰, "S": S, "T": T, "L": L, "H": H, "로그확률": lp, "위": [(j, lp[j]) for j in 차례]}


# ── 자체 시험(GPU · 모델 없이) ─────────────────────────────────────────────────────────────────────────────────────

def _바른로그(x):
    """ln x 를 바르게 반올림한 f64(십진 60 자리로 셈한 뒤 가까운 f64)."""
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    return float(Decimal(x).ln())


def _ulp차(a, b):
    if a == b:
        return 0
    return abs(실수비트(a) - 실수비트(b)) if (a > 0) == (b > 0) else 1 << 62


def 자체시험(r=None, 알림=print):
    """(1) 적힌지수 — 배열판 대 하나판 비트, (2) 적힌로그 — 바르게 반올림한 ln 과 1 ulp 안(경계값 · 넓은 범위),
    (3) 로그확률행(numpy) 대 한 칸씩(파이썬) — S · T · L · H · 로그확률 · 상위 k 의 비트, (4) Σ exp(로그확률) ≈ 1."""
    if r is None:
        r = np.random.default_rng(38)
    결과 = []

    a = np.concatenate([-r.uniform(0, 40, 200000), [0.0, -0.0, -40.0, -39.999999999999, -1e-300, -0.34657359027997264,
                                                   -0.3465735902799727, -0.6931471805599453, -20.0, -1e-17]])
    v = 적힌지수(a)
    다름 = sum(1 for x, y in zip(a.tolist(), v.tolist()) if 실수비트(적힌지수하나(x)) != 실수비트(y))
    결과.append(("적힌지수 — 배열판 대 하나판(비트)", 다름 == 0, f"{len(a)} 값 중 다름 {다름}"))

    xs = [1.0, 1.0 + 2.0 ** -52, 1.0 + 2.0 ** -30, 1.0 + 2.0 ** -21, 1.0 + 2.0 ** -20, 1.0 + 2.0 ** -19, 2.0 - 2.0 ** -52, 2.0,
          math.sqrt(2.0), math.sqrt(2.0) * (1 + 2.0 ** -52), math.sqrt(0.5) * 2, 3.0, 151936.0, 152064.0, 1.5, 0.75, 0.5,
          2.0 ** -1000, 1e300, 0.7071067811865476, 1.4142135623730951]
    xs += [float(x) for x in r.uniform(1.0, 152064.0, 3000)] + [float(1.0 + x) for x in r.uniform(0, 2.0 ** -18, 1000)]
    xs += [float(x) for x in np.exp(r.uniform(-700, 700, 2000))]
    최대 = 0
    나쁜 = None
    for x in xs:
        d = _ulp차(적힌로그(x), _바른로그(x))
        if d > 최대:
            최대, 나쁜 = d, x
    결과.append(("적힌로그 — 바르게 반올림한 ln 과의 차 ≤ 1 ulp", 최대 <= 1, f"{len(xs)} 값 · 최대 {최대} ulp" + (f" (x = {나쁜!r})" if 나쁜 else "")))

    같음 = True
    덧 = []
    for V, 꼴 in ((5000, 0), (4097, 1), (37, 2), (16, 0), (3, 1)):
        if 꼴 == 0:
            z = (r.standard_normal(V) * 4).astype(np.float32)
        elif 꼴 == 1:
            z = np.full(V, -3.0, np.float32)
            z[r.integers(0, V, 5)] = 7.0                     # 같은 가장 큰 값 여럿
            z[r.integers(0, V, 5)] = -60.0                   # 바닥 아래
            z[0] = -0.0
        else:
            z = np.array([0.0, -0.0] * (V // 2) + [0.0] * (V % 2), np.float32)   # 모두 0(±0) — 같은 값만
        빠 = 로그확률행(z, 위수=8)
        느 = 로그확률행느린(z, 위수=8)
        같다 = (빠["큰"] == 느["큰"] and all(실수비트(빠[k]) == 실수비트(느[k]) for k in ("S", "T", "L", "H"))
                and all(실수비트(x) == 실수비트(y) for x, y in zip(빠["로그확률"].tolist(), 느["로그확률"]))
                and [(j, 실수비트(x)) for j, x in 빠["위"]] == [(j, 실수비트(x)) for j, x in 느["위"]])
        같음 = 같음 and 같다
        덧.append(f"V {V}{'' if 같다 else ' 다름'}")
    결과.append(("로그확률행 — numpy 판 대 한 칸씩(S · T · L · H · 로그확률 · 상위 8 의 비트)", 같음, ", ".join(덧)))

    z = (r.standard_normal(152064) * 3 + r.uniform(-5, 5)).astype(np.float32)
    빠 = 로그확률행(z, 위수=5)
    합 = math.fsum(math.exp(x) for x in 빠["로그확률"].tolist())
    p = np.exp(빠["로그확률"])
    H참 = -math.fsum((p * 빠["로그확률"]).tolist())
    결과.append(("로그확률 — Σ exp ≈ 1 · 엔트로피 ≈ −Σ p log p (V 152064)", abs(합 - 1.0) < 1e-12 and abs(빠["H"] - H참) < 1e-10,
                 f"Σ − 1 = {합 - 1.0:.2e}, H − H참 = {빠['H'] - H참:.2e}"))
    for 이름, 됨, 덧글 in 결과:
        알림(f"{'통과' if 됨 else '못 미침'}  {이름} — {덧글}")
    return all(됨 for _, 됨, _ in 결과)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(0 if 자체시험() else 1)
