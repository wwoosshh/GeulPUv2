#!/usr/bin/env python3
"""계약의 CPU 참조(계약/정수계약.py)가 스스로 맞는지 — GPU · 모델 파일 없이(빠른 점검 · CI). 같은 계약의 서로 다른 구현을 무작위 값으로
견주고, 바탕 연산은 정확한 유리수 셈과 견준다:
  1. fma32(반올림 한 번의 곱해더하기) 대 정확한 유리수의 반올림(가까운 쪽, 같으면 짝수)
  2. 행렬곱(느린 — 자리 둘) 대 행렬곱빠른 대 행렬곱블라스 — Q8_0 · Q4_0 의 정수와 반실수 척도
  3. 행렬곱K(느린) 대 행렬곱K블라스 — Q4_K · Q5_K · Q6_K 의 날 블록(GGUF 꼴)을 만들어 풀기K 로
  4. 임베딩K 대 날 블록에서 값 하나씩 푼 것(llama.cpp 의 dequantize_row_q*_K 와 같은 식) — 풀기K 의 배치도 함께 시험한다
  5. 규칙 R(31단계 — 비정규수 버림)의 곱 · 더 · 빼기 · fma32 · 나눔 · 제곱근 대 정확한 유리수 셈 — 막 정규수(2⁻¹²⁶) 둘레의 경계값과
     비정규수 입력. 스위치를 끈 IEEE 쪽(비정규수버림 = False — GPT-2 판)도 같은 자료로 IEEE 반올림과 견준다
활성값은 블록마다 크기가 다르게(2^−20 … 2^20) 하고, 0 인 블록 · 최댓값이 2 의 거듭제곱인 블록을 섞는다.

  python -I -X utf8 시험/계약자체.py
마지막 줄 "계약 자체 시험: 모두 통과"(종료 0) 또는 "못 미침"(종료 1).
"""
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "계약"))
import 정수계약 as C        # noqa: E402

f32 = np.float32
실패 = []


def 확인(이름, 같음, 덧=""):
    print(f"{'통과' if 같음 else '못 미침'}  {이름}{(' — ' + 덧) if 덧 else ''}", flush=True)
    if not 같음:
        실패.append(이름)


def 다른칸(a, b):
    a = np.asarray(a, f32).view(np.int32)
    b = np.asarray(b, f32).view(np.int32)
    assert a.shape == b.shape, (a.shape, b.shape)
    return int(np.count_nonzero(a != b))


def 반올림32(r):
    """정확한 유리수 r → 가장 가까운 짧은실수(같으면 짝수) — 정규 범위(지수 −126 … 127)와 버금정규."""
    if r == 0:
        return f32(0.0)
    부호 = -1 if r < 0 else 1
    r = abs(r)
    e = r.numerator.bit_length() - r.denominator.bit_length()
    if Fraction(2) ** e > r:
        e -= 1
    밑 = max(e, -126) - 23                                          # 이 자리의 ulp = 2^밑
    q = r / (Fraction(2) ** 밑)
    n = q.numerator // q.denominator
    나머지 = q - n
    if 나머지 > Fraction(1, 2) or (나머지 == Fraction(1, 2) and n % 2 == 1):
        n += 1
    return f32(부호 * n * 2.0 ** 밑)                                 # fp64 로 정확(24 비트 · 2 의 거듭제곱) → 짧은실수로 정확


def 반올림R(r, 영음수=False):
    """정확한 유리수 r → 규칙 R 의 짧은실수: 가수 24 비트로(지수 제한 없이) 반올림, 크기가 2⁻¹²⁶ 보다 작으면 부호를 지킨 0.
    r 이 0 이면 영음수 가 부호(IEEE 의 정확한 0 의 부호)."""
    if r == 0:
        return f32(-0.0) if 영음수 else f32(0.0)
    부호 = -1 if r < 0 else 1
    a = abs(r)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    밑 = e - 23
    q = a / (Fraction(2) ** 밑)
    n = q.numerator // q.denominator
    나머지 = q - n
    if 나머지 > Fraction(1, 2) or (나머지 == Fraction(1, 2) and n % 2 == 1):
        n += 1
    v = n * Fraction(2) ** 밑
    if v < Fraction(1, 2 ** 126):
        return f32(-0.0) if 부호 < 0 else f32(0.0)
    if v >= Fraction(2) ** 128:
        return f32(부호 * np.inf)
    return f32(부호 * float(v))


def 반올림I(r, 영음수=False):
    """정확한 유리수 r → IEEE 짧은실수(비정규수 그대로) — 반올림32 에 0 의 부호를 더한 것."""
    if r == 0:
        return f32(-0.0) if 영음수 else f32(0.0)
    v = 반올림32(r)
    return (f32(-0.0) if r < 0 else f32(0.0)) if v == 0 else v       # 0 으로 반올림되면 부호를 지킨다(반올림32 는 부호를 잃는다)


def 활성값(r, M, K):
    """[M][K] 짧은실수 — 블록(32 칸)마다 크기를 2^−20 … 2^20 로, 0 인 블록과 최댓값이 2 의 거듭제곱인 블록을 섞는다."""
    x = r.standard_normal((M, K)).astype(np.float64)
    크기 = 2.0 ** r.integers(-20, 21, (M, K // 32))
    x = (x.reshape(M, K // 32, 32) * 크기[:, :, None]).reshape(M, K)
    x[0, :32] = 0.0                                                 # 0 인 블록
    x[1, 32:64] = 0.0
    x[1, 40] = 2.0 ** 7                                             # 최댓값이 2 의 거듭제곱
    x[2, 64:96] = -(2.0 ** -3)
    return x.astype(f32)


def 반실수(r, 모양, 아래, 위):
    """[아래, 위) 의 크기 · 부호 섞인 반실수 값(짧은실수로)."""
    v = r.uniform(아래, 위, 모양) * np.where(r.random(모양) < 0.2, -1.0, 1.0)
    return v.astype(np.float16).astype(f32)


def 시험_fma32(r):
    n = 4000
    a = (r.standard_normal(n) * 2.0 ** r.integers(-30, 30, n)).astype(f32)
    b = (r.standard_normal(n) * 2.0 ** r.integers(-30, 30, n)).astype(f32)
    c = (r.standard_normal(n) * 2.0 ** r.integers(-60, 60, n)).astype(f32)
    c[:500] = (-(a[:500].astype(np.float64) * b[:500])).astype(f32)  # 거의 지워지는 경우
    c[500:1000] = 0.0
    got = C.fma32(a, b, c)
    want = np.array([반올림R(Fraction(float(a[i])) * Fraction(float(b[i])) + Fraction(float(c[i]))) for i in range(n)], f32)
    d = 다른칸(got, want)
    확인("fma32 — 정확한 유리수의 반올림", d == 0, f"{n} 개 중 다른 칸 {d}")


def 시험_행렬곱(r):
    M, K, N = 9, 256, 40
    x = 활성값(r, M, K)
    치 = r.standard_normal(N).astype(f32)
    for 이름, 아래, 위 in (("Q8_0", -127, 128), ("Q4_0", -8, 8)):
        q = r.integers(아래, 위, (K, N)).astype(np.int64)
        d = 반실수(r, (K // 32, N), 1e-4, 0.05)
        느린 = C.행렬곱(x, q, d, 치, 자리수=2)
        빠른 = C.행렬곱빠른(x, q, d, 치)
        블라스 = C.행렬곱블라스(x, q.T.astype(np.int8), d.T, 치, 일꾼=2)
        확인(f"행렬곱 {이름} — 느린 대 빠른", 다른칸(느린, 빠른) == 0, f"다른 칸 {다른칸(느린, 빠른)} / {M * N}")
        확인(f"행렬곱 {이름} — 느린 대 블라스", 다른칸(느린, 블라스) == 0, f"다른 칸 {다른칸(느린, 블라스)} / {M * N}")


def 날블록(r, t, N, KS):
    """GGUF k-양자화의 날 바이트 [N][KS][블록 바이트] — d · m 은 유한한 반실수, 나머지는 무작위 바이트."""
    bb = {12: 144, 13: 176, 14: 210}[t]
    raw = r.integers(0, 256, (N, KS, bb)).astype(np.uint8)
    if t in (12, 13):
        raw[:, :, 0:2] = 반실수(r, (N, KS), 1e-4, 0.02).astype(np.float16)[..., None].view(np.uint8)
        raw[:, :, 2:4] = 반실수(r, (N, KS), 1e-4, 0.02).astype(np.float16)[..., None].view(np.uint8)
    else:
        raw[:, :, 208:210] = 반실수(r, (N, KS), 1e-5, 0.002).astype(np.float16)[..., None].view(np.uint8)
    return raw.reshape(-1)


def 값하나씩(t, raw, N, K):
    """날 블록 → 값 [N][K] (짧은실수) — llama.cpp 의 dequantize_row_q4_K · q5_K · q6_K 와 같은 식을 칸마다(풀기K 와 따로 쓴 것)."""
    bb = {12: 144, 13: 176, 14: 210}[t]
    bl = np.asarray(raw).reshape(N, K // 256, bb)
    y = np.zeros((N, K), f32)
    for n in range(N):
        for s in range(K // 256):
            b = bl[n, s]
            o = 256 * s
            if t in (12, 13):
                d = f32(b[0:2].view(np.float16)[0])
                m = f32(b[2:4].view(np.float16)[0])
                sc = [int(v) for v in b[4:16]]

                def 척(j):
                    if j < 4:
                        return sc[j] & 63, sc[j + 4] & 63
                    return (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4), (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)
                qs = b[16:144] if t == 12 else b[48:176]
                qh = b[16:48] if t == 13 else None
                for c in range(4):                                  # 값 64 개 = 아래블록 2c · 2c + 1
                    for h in range(2):
                        sv, mv = 척(2 * c + h)
                        a_ = f32(d * f32(sv))
                        b_ = f32(m * f32(mv))
                        for l in range(32):
                            q = (int(qs[32 * c + l]) >> (4 * h)) & 15
                            if t == 13:
                                q |= ((int(qh[l]) >> (2 * c + h)) & 1) << 4
                            y[n, o + 64 * c + 32 * h + l] = f32(f32(a_ * f32(q)) - b_)
            else:
                ql, qh = b[0:128], b[128:192]
                sc = b[192:208].view(np.int8)
                d = f32(b[208:210].view(np.float16)[0])
                for h in range(2):                                  # 값 128 개마다
                    for l in range(32):
                        i = l // 16
                        q1 = ((int(ql[64 * h + l]) & 15) | (((int(qh[32 * h + l]) >> 0) & 3) << 4)) - 32
                        q2 = ((int(ql[64 * h + l + 32]) & 15) | (((int(qh[32 * h + l]) >> 2) & 3) << 4)) - 32
                        q3 = ((int(ql[64 * h + l]) >> 4) | (((int(qh[32 * h + l]) >> 4) & 3) << 4)) - 32
                        q4 = ((int(ql[64 * h + l + 32]) >> 4) | (((int(qh[32 * h + l]) >> 6) & 3) << 4)) - 32
                        for k, q in ((0, q1), (2, q2), (4, q3), (6, q4)):
                            a_ = f32(d * f32(int(sc[8 * h + i + k])))
                            y[n, o + 128 * h + 16 * k + l] = f32(a_ * f32(q))
    return y


def 시험_K(r):
    M, K, N = 7, 512, 24
    x = 활성값(r, M, K)
    치 = r.standard_normal(N).astype(f32)
    for t, 이름 in ((12, "Q4_K"), (13, "Q5_K"), (14, "Q6_K")):
        raw = 날블록(r, t, N, K // 256)
        w = C.풀기K(t, raw, N, K)
        느린 = C.행렬곱K(x, w, 치)
        블라스 = C.행렬곱K블라스(x, w, 치, 일꾼=2)
        확인(f"행렬곱K {이름} — 느린 대 블라스", 다른칸(느린, 블라스) == 0, f"다른 칸 {다른칸(느린, 블라스)} / {M * N}")
        ids = np.arange(N)
        풀린 = C.임베딩K(w, ids)
        하나씩 = 값하나씩(t, raw, N, K)
        확인(f"임베딩K {이름} — 날 블록에서 값 하나씩", 다른칸(풀린, 하나씩) == 0, f"다른 칸 {다른칸(풀린, 하나씩)} / {N * K}")


def _경계자료(r):
    """막 정규수 둘레의 짧은실수 a, b, c (탐침31_셈.py 의 경계 꼴을 줄인 것)."""
    작 = 2.0 ** -126
    부 = lambda x: -x if r.random() < 0.5 else x          # noqa: E731
    비정규 = lambda: 부(float(r.integers(1, 1 << 23)) * 2.0 ** -149)        # noqa: E731
    곱, 더, fma, 나 = [], [], [], []
    for k in list(range(1, 41)) + list(range(1440, 1461)) + list(range(2040, 2061)):   # a·b = 2⁻¹²⁶ − k²·2⁻¹⁷²
        a, b = 부(((1 << 23) + k) * 2.0 ** -24), 부(((1 << 24) - 2 * k) * 2.0 ** -149)
        곱.append((a, b, 0.0))
        fma.append((a, b, 부(0.0)))
    for j in range(1, 17):                                  # 2⁻¹²⁶ − j·2⁻¹⁵⁰
        곱.append((부(((1 << 24) - j) * 2.0 ** -24), 부(작), 0.0))
        나.append((부(((1 << 24) - j) * 2.0 ** -24), 부(2.0 ** 126), 0.0))
    for _ in range(200):
        곱.append((비정규(), 부(2.0 ** float(r.integers(0, 40))), 0.0))                     # 비정규수 입력
        a = 부(float(r.integers(1 << 23, 1 << 24)) * 2.0 ** -149)
        b = -np.sign(a) * float(r.integers(1 << 23, 1 << 24)) * 2.0 ** -149
        더.append((a, float(b), 0.0))                                                        # 상쇄로 비정규수
        더.append((비정규(), 부(float(r.integers(1 << 23, 1 << 26)) * 2.0 ** -149), 0.0))       # 비정규수 + 작은 정규수
        x = 부(float(r.integers(1 << 23, 1 << 24)) * 2.0 ** -86)
        y = 부(float(r.integers(1 << 23, 1 << 24)) * 2.0 ** -86)                               # x·y ≈ 2⁻¹²⁴, c 로 2⁻¹²⁶ 근처까지 지운다
        p = Fraction(x) * Fraction(y)
        q = Fraction(float(반올림R(p)))
        c = float(np.sign(float(p)) * float(r.choice([(1 << 23) - 4, 1 << 23, 1 << 23, (1 << 23) + 4])) * 2.0 ** -149 - float(q))
        if f32(c) == c:
            fma.append((x, y, c))
        n = int(r.integers((1 << 24) + 1, 1 << 25)) | 1                                     # 곱이 정확히 한가운데 — c 의 부호로 갈린다
        pf = next((t for t in range(3, 4096, 2) if n % t == 0), None)
        if pf is not None and n // pf < (1 << 24):
            fma.append((부(pf * 2.0 ** -12), (n // pf) * 2.0 ** -12, 비정규()))
        나.append((부(float(r.integers(1 << 23, 1 << 24)) * 2.0 ** -149 * 2.0 ** float(r.integers(0, 16))),
                  부(float(r.integers(1 << 23, 1 << 24)) * 2.0 ** float(r.integers(-22, 8))), 0.0))  # 몫이 비정규수 둘레
        나.append((비정규(), 부(float(r.integers(1 << 23, 1 << 24)) * 2.0 ** -40), 0.0))            # 비정규수 나뉨수
    근 = [(abs(비정규()), 0.0, 0.0) for _ in range(100)] + [(0.0, 0.0, 0.0), (-0.0, 0.0, 0.0)]
    return {"곱": 곱, "더": 더, "fma32": fma, "나눔": 나, "제곱근": 근}


def _정확(연산, a, b, c, 규칙R):
    """한 칸의 기대값 — 규칙R 이면 입력의 비정규수를 부호를 지킨 0 으로 읽고 반올림R, 아니면 반올림I."""
    작 = 2.0 ** -126
    if 규칙R:
        a, b, c = ((-0.0 if np.signbit(v) else 0.0) if (v != 0 and abs(v) < 작) else v for v in (a, b, c))
    음 = lambda v: bool(np.signbit(v))                     # noqa: E731
    반 = 반올림R if 규칙R else 반올림I
    A, B, C_ = Fraction(a), Fraction(b), Fraction(c)
    if 연산 == "곱":
        return 반(A * B, 음(a) != 음(b))
    if 연산 == "더":
        return 반(A + B, 음(a) and 음(b))
    if 연산 == "fma32":
        return 반(A * B + C_, (음(a) != 음(b)) and 음(c))
    if 연산 == "나눔":
        return 반(A / B, 음(a) != 음(b))
    return f32(np.sqrt(np.float64(a))) if a != 0 else f32(a)     # 제곱근: 입력이 정규수면 fp64 의 √ 를 짧은실수로 — 바른 반올림(53 ≥ 2·24 + 2)


def 시험_규칙R(r):
    자료 = _경계자료(r)
    for 규칙R in (True, False):
        C.비정규수버림 = 규칙R
        for 연산, 칸 in 자료.items():
            a, b, c = (np.array([t[i] for t in 칸], f32) for i in range(3))
            if 연산 == "곱":
                got = C.곱(a, b)
            elif 연산 == "더":
                got = C.더(a, b)
            elif 연산 == "fma32":
                got = C.fma32(a, b, c)
            elif 연산 == "나눔":
                got = C.나눔(a, b)
            else:
                got = C.제곱근(a)
            want = np.array([_정확(연산, float(a[i]), float(b[i]), float(c[i]), 규칙R) for i in range(len(칸))], f32)
            d = 다른칸(got, want)
            확인(f"규칙 {'R' if 규칙R else 'IEEE'} — {연산} 경계값", d == 0, f"{len(칸)} 개 중 다른 칸 {d}")
    C.비정규수버림 = True
    # 빼기는 더(a, −b)
    a = np.array([t[0] for t in 자료["더"]], f32)
    b = np.array([t[1] for t in 자료["더"]], f32)
    d = 다른칸(C.빼기(a, -b), C.더(a, b))
    확인("규칙 R — 빼기 = 더(a, −b)", d == 0, f"다른 칸 {d}")


def 시험_머리정규화(r):
    """37단계 — 머리마다 Q · K 정규화(계약 4.6): (1) 판의 차례(워프 하나 = 머리 하나, 레인 l 이 d = l + 32·i — 64 접기를 둘로 나눠
    a = p_l + p_{l+64} · b = p_{l+32} + p_{l+96}, 32 접기 a + b, 셔플 xor 16 · 8 · 4 · 2 · 1)가 계약의 접기(rms정규화 를 D = 128 로)와
    비트까지 같고 모든 레인이 같은 S 를 갖는지, (2) fp64 의 정규화와의 상대 오차(공식이 맞는지 — 크기를 넓게)."""
    hd, eps = 128, 1e-6
    m = 4096
    x = (r.standard_normal((m, hd)) * 10.0 ** r.uniform(-25, 17, (m, 1))).astype(f32)
    x[:8] = 0                                                         # 0 인 머리
    x[8:16, ::3] = 0                                                  # 칸 몇이 0
    x[16:24] = (r.standard_normal((8, hd)) * 2.0 ** -140).astype(f32)  # 비정규수 입력
    x[24:32, :5] = (2.0 ** r.integers(-126, 60, (8, 5))).astype(f32)  # 2 의 거듭제곱
    w = (r.standard_normal(hd) * 2).astype(f32)
    w[:4] = (1.0, -1.0, 0.0, 2.0 ** -20)
    계약 = C.머리정규화(x.reshape(m // 8, 8 * hd), w, eps).reshape(m, hd)
    p = C.곱(x, x)
    s = C.더(C.더(p[:, 0:32], p[:, 64:96]), C.더(p[:, 32:64], p[:, 96:128]))
    for k in (16, 8, 4, 2, 1):
        s = C.더(s, s[:, np.arange(32) ^ k])
    레인같음 = int(np.count_nonzero((s.view(np.int32) != s[:, :1].view(np.int32)).any(1)))
    rr = C.나눔(f32(1), C.제곱근(C.더(C.나눔(s[:, 0], f32(hd)), f32(eps))))
    판 = C.곱(C.곱(x, rr[:, None]), w[None, :])
    d = 다른칸(계약, 판)
    확인("머리 정규화 — 판의 차례(64 접기를 둘로 · 셔플 xor) 대 계약의 접기", d == 0 and 레인같음 == 0,
         f"{m} 머리 중 다른 칸 {d}, 레인끼리 S 가 다른 머리 {레인같음}")
    x64 = x.astype(np.float64)
    y64 = x64 / np.sqrt((x64 * x64).mean(1, keepdims=True) + float(f32(eps))) * w.astype(np.float64)[None, :]
    정상 = np.abs(x64).max(1) > 1e-15                                  # x · x 가 버려지지 않는 머리(규칙 R 의 경계 밖)
    상대 = np.abs(계약[정상].astype(np.float64) - y64[정상]) / np.maximum(np.abs(y64[정상]), 1e-30)
    상대 = np.where(np.abs(y64[정상]) > 1e-30, 상대, 0)
    확인("머리 정규화 — fp64 의 정규화와의 상대 오차 ≤ 2⁻¹⁹", float(상대.max()) <= 2.0 ** -19,
         f"머리 {int(정상.sum())} · 최대 {float(상대.max()):.2e} · 가운데 {float(np.median(상대)):.2e}")


def 시험_로그확률(r):
    """38단계 — 로그확률(계약 5.2): 적힌 exp 의 배열판 대 하나판, 적힌 log 대 바르게 반올림한 ln(1 ulp 안), numpy 판 대 한 칸씩
    (16 구간의 차례 · 상위 k), Σ exp ≈ 1 — 계약/로그확률.py 의 자체시험."""
    import 로그확률 as LP          # noqa: E402 — 계약/ 은 위에서 sys.path 에 넣었다

    def 알림(줄):
        됨 = 줄.startswith("통과")
        이름, _, 덧 = 줄.partition("  ")[2].partition(" — ")
        확인(f"로그확률 — {이름}", 됨, 덧)
    LP.자체시험(r, 알림)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    r = np.random.default_rng(27)
    시험_fma32(r)
    시험_행렬곱(r)
    시험_K(r)
    시험_규칙R(r)
    시험_머리정규화(r)
    시험_로그확률(r)
    print("계약 자체 시험:", "모두 통과" if not 실패 else f"못 미침 {실패}", flush=True)
    return 0 if not 실패 else 1


if __name__ == "__main__":
    sys.exit(main())
