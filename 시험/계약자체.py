#!/usr/bin/env python3
"""계약의 CPU 참조(계약/정수계약.py)가 스스로 맞는지 — GPU · 모델 파일 없이(빠른 점검 · CI). 같은 계약의 서로 다른 구현을 무작위 값으로
견주고, 바탕 연산은 정확한 유리수 셈과 견준다:
  1. fma32(반올림 한 번의 곱해더하기) 대 정확한 유리수의 반올림(가까운 쪽, 같으면 짝수)
  2. 행렬곱(느린 — 자리 둘) 대 행렬곱빠른 대 행렬곱블라스 — Q8_0 · Q4_0 의 정수와 반실수 척도. 빠른 계약 F1(자리 하나)도 느린 대 블라스(#27)
  3. 행렬곱K(느린) 대 행렬곱K블라스 — Q4_K · Q5_K · Q6_K · IQ4_XS 의 날 블록(GGUF 꼴)을 만들어 풀기K 로(자리 둘 · 하나)
  4. 임베딩K 대 날 블록에서 값 하나씩 푼 것(llama.cpp 의 dequantize_row_q*_K · iq4_xs 와 같은 식) — 풀기K 의 배치도 함께 시험한다
  5. 규칙 R(31단계 — 비정규수 버림)의 곱 · 더 · 빼기 · fma32 · 나눔 · 제곱근 대 정확한 유리수 셈 — 막 정규수(2⁻¹²⁶) 둘레의 경계값과
     비정규수 입력. 스위치를 끈 IEEE 쪽(비정규수버림 = False — GPT-2 판)도 같은 자료로 IEEE 반올림과 견준다
  6. 8비트 KV(GeulPUv2 ③b-2 — 계약 F1A′, 계약.md 2.1): 저장(팔비트KV) 대 정확한 유리수 계산, 스케일이 2 의 거듭제곱일 때 어텐션 대
     ③b-1 의 에뮬레이션, 어텐션(numpy) 대 계약 문장을 행 하나씩 옮긴 계산
  7. 빠른 어텐션 계약 F2(계약.md 2.2): 지수2 · q 의 자리 둘(지수를 헤드마다) · 값의 저장(스케일을 헤드마다) 대 정확한 유리수 계산,
     어텐션F2(numpy) 대 계약 문장을 행 하나씩 옮긴 계산(그룹 둘)
  8. Qwen3.5 의 계약 4.9 · 4.10(0.12.0 12-3e): 로그1p · σ · softplus 의 오차, 열 합의 차례, GDN · 합성곱을 덩이로 나눠도 같은 비트
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
        # #27 — 빠른 계약 F1(선형층의 자리 하나 — 계약.md 1.2): 느린 대 블라스(repro_check · 시험/참조7B.py --빠른 이 쓰는 길)
        느린1 = C.행렬곱(x, q, d, 치, 자리수=1)
        블라스1 = C.행렬곱블라스(x, q.T.astype(np.int8), d.T, 치, 일꾼=2, 자리수=1)
        확인(f"행렬곱 {이름} 자리 하나 — 느린 대 블라스", 다른칸(느린1, 블라스1) == 0, f"다른 칸 {다른칸(느린1, 블라스1)} / {M * N}")
        확인(f"행렬곱 {이름} — 자리 하나와 둘이 다르다(자리수가 실제로 쓰인다)", 다른칸(느린1, 느린) > 0, f"다른 칸 {다른칸(느린1, 느린)} / {M * N}")


def 날블록(r, t, N, KS):
    """GGUF k-양자화의 날 바이트 [N][KS][블록 바이트] — d · m 은 유한한 반실수, 나머지는 무작위 바이트(1.0.0 1-6c — IQ4_XS 23 도)."""
    bb = {12: 144, 13: 176, 14: 210, 23: 136}[t]
    raw = r.integers(0, 256, (N, KS, bb)).astype(np.uint8)
    if t == 23:
        raw[:, :, 0:2] = 반실수(r, (N, KS), 1e-5, 0.002).astype(np.float16)[..., None].view(np.uint8)
    elif t in (12, 13):
        raw[:, :, 0:2] = 반실수(r, (N, KS), 1e-4, 0.02).astype(np.float16)[..., None].view(np.uint8)
        raw[:, :, 2:4] = 반실수(r, (N, KS), 1e-4, 0.02).astype(np.float16)[..., None].view(np.uint8)
    else:
        raw[:, :, 208:210] = 반실수(r, (N, KS), 1e-5, 0.002).astype(np.float16)[..., None].view(np.uint8)
    return raw.reshape(-1)


def 값하나씩(t, raw, N, K):
    """날 블록 → 값 [N][K] (짧은실수) — llama.cpp 의 dequantize_row_q4_K · q5_K · q6_K · iq4_xs 와 같은 식을 칸마다(풀기K 와 따로 쓴 것)."""
    bb = {12: 144, 13: 176, 14: 210, 23: 136}[t]
    bl = np.asarray(raw).reshape(N, K // 256, bb)
    y = np.zeros((N, K), f32)
    코드북 = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]    # llama.cpp 의 kvalues_iq4nl
    for n in range(N):
        for s in range(K // 256):
            b = bl[n, s]
            o = 256 * s
            if t == 23:                                             # 1.0.0 1-6c — IQ4_XS(dequantize_row_iq4_xs)
                d = f32(b[0:2].view(np.float16)[0])
                sh = int(b[2]) | (int(b[3]) << 8)
                for ib in range(8):
                    ls = ((int(b[4 + ib // 2]) >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4)
                    dl = f32(d * f32(ls - 32))
                    for j in range(16):
                        q = int(b[8 + 16 * ib + j])
                        y[n, o + 32 * ib + j] = f32(dl * f32(코드북[q & 15]))
                        y[n, o + 32 * ib + 16 + j] = f32(dl * f32(코드북[q >> 4]))
            elif t in (12, 13):
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
    for t, 이름 in ((12, "Q4_K"), (13, "Q5_K"), (14, "Q6_K"), (23, "IQ4_XS")):
        raw = 날블록(r, t, N, K // 256)
        w = C.풀기K(t, raw, N, K)
        느린 = C.행렬곱K(x, w, 치)
        블라스 = C.행렬곱K블라스(x, w, 치, 일꾼=2)
        확인(f"행렬곱K {이름} — 느린 대 블라스", 다른칸(느린, 블라스) == 0, f"다른 칸 {다른칸(느린, 블라스)} / {M * N}")
        느린1 = C.행렬곱K(x, w, 치, 자리수=1)                                                     # #27 — 빠른 계약 F1
        블라스1 = C.행렬곱K블라스(x, w, 치, 일꾼=2, 자리수=1)
        확인(f"행렬곱K {이름} 자리 하나 — 느린 대 블라스", 다른칸(느린1, 블라스1) == 0, f"다른 칸 {다른칸(느린1, 블라스1)} / {M * N}")
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


def _가까운정수(r):
    """정확한 유리수 r → 가장 가까운 정수(같으면 짝수)."""
    n = r.numerator // r.denominator
    나머지 = r - n
    return n + 1 if 나머지 > Fraction(1, 2) or (나머지 == Fraction(1, 2) and n % 2 == 1) else n


def _스케일2(r, n, hd):
    """[n][hd] fp32 — 블록(값 32 개)마다 정수 m(|m| ≤ 127, 한 값은 ±127) × 2^k(k −10 … −5) 라 8비트 KV 의 스케일 s = 2^k 이고 n · s = x 다.
    블록 몇은 0."""
    m = r.integers(-127, 128, (n, hd // 32, 32))
    m[np.arange(n)[:, None], np.arange(hd // 32)[None, :], r.integers(0, 32, (n, hd // 32))] = np.where(r.random((n, hd // 32)) < 0.5, -127, 127)
    m[r.random((n, hd // 32)) < 0.03] = 0
    return (m * 2.0 ** r.integers(-10, -4, (n, hd // 32, 1))).reshape(n, hd).astype(f32)


def 시험_KV8(r):
    """GeulPUv2 ③b-2 — 8비트 KV(계약 F1A′, 계약.md 2.1)."""
    # 1. 저장 — 팔비트KV 대 정확한 유리수 계산(규칙 R 의 반올림), n · s 대 에뮬레이션(팔비트)
    x = 활성값(r, 40, 128)
    x[3, :32] = (r.standard_normal(32) * 2.0 ** -140).astype(f32)              # 비정규수만 — 0 으로 읽는다
    x[4, :32] = (r.standard_normal(32) * 2.0 ** -124).astype(f32)              # m / 127 이 2⁻¹²⁶ 아래 — s = 0
    x[4, 32:64] = (r.standard_normal(32) * 2.0 ** -117).astype(f32)            # s 가 막 정규수 둘레
    x[5, :32] = (r.standard_normal(32) * 2.0 ** 120).astype(f32)
    x[6, :32] = _스케일2(r, 1, 32)[0]                                         # 스케일이 2 의 거듭제곱
    n, s = C.팔비트KV(x)
    nw = np.zeros_like(n)
    sw = np.zeros_like(s)
    for i in range(x.shape[0]):
        for b in range(x.shape[1] // 32):
            블록 = [Fraction(float(v)) if abs(float(v)) >= 2.0 ** -126 else Fraction(0) for v in x[i, 32 * b:32 * b + 32]]
            sw[i, b] = 반올림R(max(abs(v) for v in 블록) / 127)
            if sw[i, b] > 0:
                nw[i, 32 * b:32 * b + 32] = [_가까운정수(Fraction(float(반올림R(v / Fraction(float(sw[i, b])))))) for v in 블록]
    dn, ds = int(np.count_nonzero(n != nw)), 다른칸(s, sw)
    확인("8비트 KV — 저장(팔비트KV) 대 정확한 유리수 계산", dn == 0 and ds == 0 and np.abs(n).max() <= 127,
         f"n 불일치 {dn} / {n.size}, s 불일치 {ds} / {s.size}")
    y = C.곱(n.astype(f32), np.repeat(s, 32, 1))
    확인("8비트 KV — n · s 대 에뮬레이션(팔비트)", bool(np.array_equal(y, C.팔비트(x))), "값 비교(0 의 부호는 보지 않음)")
    # 2. 스케일이 2 의 거듭제곱이고 값의 크기가 보통이면 어텐션이 ③b-1 의 에뮬레이션과 비트까지 같다 — 타일 · 그룹 병합까지(위치 1100 =
    #    그룹 둘). 점수 폭이 아주 넓으면 e′ 가 16비트 형식의 지수 −98 아래로 내려가 둘이 다를 수 있다(계약.md 2.1).
    for hd, 길이 in ((64, 150), (128, 1100)):
        q = (r.standard_normal((길이, hd)) * 2.0 ** r.integers(-8, -2, (길이, 1))).astype(f32)
        k, v = _스케일2(r, 길이, hd), _스케일2(r, 길이, hd)
        C.KV8 = True
        얻음 = C.어텐션(q, k, v)
        C.KV8, C.어텐션한자리 = False, "kv8"
        기대 = C.어텐션(q, k, v)
        C.어텐션한자리 = ""
        d = 다른칸(얻음, 기대)
        확인(f"8비트 KV — 스케일이 2 의 거듭제곱일 때 어텐션 대 에뮬레이션(헤드 크기 {hd}, 위치 {길이})", d == 0, f"불일치 {d} / {얻음.size}")
    # 3. 어텐션(numpy) 대 계약 문장(2.1 · 2 절 3 · 4 절 1)을 행 하나씩 옮긴 계산 — 타일 하나(위치 64), 스케일은 2 의 거듭제곱이 아니다
    hd, 길이 = 128, 64
    q, k, v = 활성값(r, 길이, hd), 활성값(r, 길이, hd), 활성값(r, 길이, hd)
    C.KV8 = True
    얻음 = C.어텐션(q, k, v)
    C.KV8 = False
    nk, sk = C.팔비트KV(k)
    nv, sv = C.팔비트KV(v)
    D, Eq = C.자리로(q, 2)
    nq, Pq = D[1] * 256 + D[0], C.거듭제곱(Eq)
    c = C.점수눈금(hd)
    기대 = np.zeros((길이, hd), f32)
    for p in range(길이):
        st = np.zeros(p + 1, f32)
        for t in range(p + 1):
            z = None
            for b in range(hd // 32):
                u = f32(int(np.dot(nq[p, 32 * b:32 * b + 32], nk[t, 32 * b:32 * b + 32])))        # 정확한 정수 합 → 반올림 하나
                a = C.곱(u, Pq[p, b])
                z = C.곱(a, sk[t, b]) if b == 0 else C.fma32(a, sk[t, b], z)
            st[t] = C.곱(z, c)
        e = C.적힌지수(C.빼기(st, st.max()))
        l = f32(sum(int(np.rint(C.곱(ei, f32(8388608.0)))) for ei in e)) * f32(2.0 ** -23)
        o = np.zeros(hd, f32)
        for b in range(hd // 32):
            ep = np.zeros(64, f32)
            ep[:p + 1] = C.곱(e, sv[:p + 1, b])
            uj = []
            for j in (0, 32):
                ne, Ee = C.e자리(ep[None, j:j + 32])
                u = [f32(int(np.dot(ne[0], nv[j:j + 32, d]))) for d in range(32 * b, 32 * b + 32)]
                uj.append((np.array(u, f32), C.거듭제곱(Ee[0, 0])))
            (u0, P0), (u1, P1) = uj
            o[32 * b:32 * b + 32] = C.fma32(u1, P1, C.곱(u0, P0))
        기대[p] = C.나눔(o, l)
    d = 다른칸(얻음, 기대)
    확인("8비트 KV — 어텐션 대 계약 문장을 행 하나씩 옮긴 계산(헤드 크기 128, 위치 64)", d == 0, f"불일치 {d} / {얻음.size}")


def _바닥로그2(m):
    """양의 유리수 m 의 ⌊log₂ m⌋."""
    e = m.numerator.bit_length() - m.denominator.bit_length()
    if Fraction(2) ** e > m:
        e -= 1
    elif Fraction(2) ** (e + 1) <= m:
        e += 1
    return e


def _지수2정확(x):
    """계약.md 2.2 의 4(지수2)를 정확한 유리수 셈과 규칙 R 의 반올림으로 — 연산 하나마다 반올림 하나(곱해더하기는 곱을 반올림하지 않는다)."""
    F = lambda v: Fraction(float(v))                                       # noqa: E731 — 짧은실수 → 정확한 유리수
    xc = max(F(x), Fraction(-126))
    t = 반올림R(xc + 12582912)
    r = 반올림R(xc - F(반올림R(F(t) - 12582912)))
    rf = F(r)
    c = [Fraction(float(v)) for v in C._k2]                                  # c₅ … c₁ (fp32)
    p = Fraction(float(반올림R(c[0] * rf + c[1])))
    for ck in c[2:]:
        p = Fraction(float(반올림R(p * rf + ck)))
    p = 반올림R(p * rf + 1)
    bits = (int(np.float32(p).view(np.int32)) + (int(np.float32(t).view(np.int32)) << 23)) & 0xFFFFFFFF
    return np.uint32(bits).view(f32)


def 시험_F2(r):
    """GeulPUv2 빠른 어텐션 계약 F2(계약.md 2.2)."""
    # 1. 지수2 대 정확한 유리수 셈 — 넓은 범위, 반올림 경계(n ± ½), 자르기(−126) 둘레, 0
    xs = np.concatenate([-(r.random(3000) * 30).astype(f32), -(r.random(500) * 0.6).astype(f32),
                         np.array([0.0, -0.5, -1.5, -2.5, -125.6, -126.0, -126.4, -300.0, -3.0e38], f32),
                         (np.arange(1, 200) * -0.5 + r.standard_normal(199) * 1e-6).astype(f32)]).astype(f32)
    얻음 = C.지수2(xs)
    기대 = np.array([_지수2정확(x) for x in xs], f32)
    d = 다른칸(얻음, 기대)
    확인("F2 — 지수2 대 정확한 유리수 셈", d == 0, f"불일치 {d} / {xs.size}")
    # 2. 저장 — 자리로머리(q) · 팔비트KV머리(값) 대 정확한 유리수 셈
    x = 활성값(r, 30, 128)
    x[2] = 0
    x[3, 5] = (2.0 ** -140)
    D, E = C.자리로머리(x, 2)
    dw = 0
    for i in range(x.shape[0]):
        m = max(abs(Fraction(float(v))) for v in x[i])
        Ew = -98 if m == 0 else max(-98, min(100, _바닥로그2(m)))
        nw = [_가까운정수(Fraction(float(v)) * Fraction(2) ** (13 - Ew)) for v in x[i]]
        n = D[1, i] * 256 + D[0, i]
        dw += int(Ew != E[i]) + int(np.count_nonzero(n != np.array(nw)))
    확인("F2 — q 의 자리 둘(지수를 헤드마다 하나) 대 정확한 유리수 셈", dw == 0, f"불일치 {dw}")
    n, s = C.팔비트KV머리(x)
    nw = np.zeros_like(n)
    sw = np.zeros_like(s)
    for i in range(x.shape[0]):
        행 = [Fraction(float(v)) if abs(float(v)) >= 2.0 ** -126 else Fraction(0) for v in x[i]]
        sw[i] = 반올림R(max(abs(v) for v in 행) / 127)
        if sw[i] > 0:
            nw[i] = [_가까운정수(Fraction(float(반올림R(v / Fraction(float(sw[i])))))) for v in 행]
    dn, ds = int(np.count_nonzero(n != nw)), 다른칸(s, sw)
    확인("F2 — 값의 저장(헤드마다 스케일 하나) 대 정확한 유리수 셈", dn == 0 and ds == 0, f"n 불일치 {dn}, s 불일치 {ds}")
    # 3. 어텐션F2(numpy) 대 계약 문장(2.2 의 1 ~ 6, 3 절)을 행 하나씩 옮긴 계산 — 위치 1100(타일 18 · 그룹 둘, 끝 타일은 반쯤)
    hd, 길이 = 128, 1100
    q = (r.standard_normal((길이, hd)) * 2.0 ** r.integers(-6, 0, (길이, 1))).astype(f32)
    k, v = 활성값(r, 길이, hd), 활성값(r, 길이, hd)
    for 블록q in (False, True):                                     # 1.0.0 1-4 — F2B(qwen3): q 의 지수를 블록마다
        시험_F2어텐션(q, k, v, hd, 길이, 블록q)


def 시험_F2어텐션(q, k, v, hd, 길이, 블록q):
    """F2(블록q 거짓) · F2B(참)의 어텐션(numpy) 대 계약 문장을 행 하나씩 옮긴 계산."""
    C.F2, C.F2B = True, 블록q
    얻음 = C.어텐션(q, k, v)
    C.F2, C.F2B = False, False
    Dq, Eq = C.자리로(q, 2) if 블록q else C.자리로머리(q, 2)
    nq = Dq[1] * 256 + Dq[0]
    nk, sk = C.팔비트KV(k)
    nv, sv = C.팔비트KV머리(v)
    기대 = np.zeros((길이, hd), f32)
    for p in range(길이):
        qc = C.점수눈금2 if 블록q else C.곱(C.거듭제곱(Eq[p]), C.점수눈금2)
        그룹들 = []
        for g0 in range(0, p + 1, 1024):
            M = L = O = None
            for t0 in range(g0, min(g0 + 1024, p + 1), 64):
                ks = np.arange(t0, min(t0 + 64, p + 1))                         # 이 행에 보이는 타일의 키(가린 키는 지수가 0 이라 빼도 같다)
                z = None
                for b in range(hd // 32):
                    u = (nk[ks, 32 * b:32 * b + 32] @ nq[p, 32 * b:32 * b + 32]).astype(f32)
                    if 블록q:                                           # F2B — 블록마다 u · 2^(E_q,b − 13)(정확)
                        u = C.곱(u, C.거듭제곱(Eq[p, b]))
                    z = C.곱(u, sk[ks, b]) if b == 0 else C.fma32(u, sk[ks, b], z)
                st = C.곱(z, qc)
                m = st.max()
                e = C.지수2(C.빼기(st, m))
                l = f32(int(np.rint((e * f32(8388608.0)).astype(f32)).astype(np.int64).sum())) * f32(2.0 ** -23)
                ep = C.곱(e, sv[ks])
                mm = float(ep.max())
                E = -98 if mm == 0 else max(-98, min(100, int(np.frexp(mm)[1]) - 1))
                ne = np.rint(ep.astype(np.float64) * 2.0 ** (13 - E)).astype(np.int64)
                U = (ne @ nv[ks]).astype(f32)
                P = C.거듭제곱(E)
                if M is None:
                    M, L, O = m, l, C.fma32(U, P, f32(0))
                else:
                    M2 = max(M, m)
                    가, 나 = C.지수2(C.빼기(M, M2)), C.지수2(C.빼기(m, M2))
                    L = C.fma32(l, 나, C.곱(L, 가))
                    O = C.fma32(U, C.곱(P, 나), C.곱(O, 가))
                    M = M2
            그룹들.append((M, L, O))
        M, L, O = 그룹들[0]
        for m, l, o in 그룹들[1:]:
            M2 = max(M, m)
            가, 나 = C.지수2(C.빼기(M, M2)), C.지수2(C.빼기(m, M2))
            L = C.fma32(l, 나, C.곱(L, 가))
            O = C.fma32(o, 나, C.곱(O, 가))
            M = M2
        기대[p] = C.나눔(O, L)
    d = 다른칸(얻음, 기대)
    이름 = "F2B" if 블록q else "F2"
    확인(f"{이름} — 어텐션 대 계약 문장을 행 하나씩 옮긴 계산(헤드 크기 128, 위치 1100 — 그룹 둘)", d == 0, f"불일치 {d} / {얻음.size}")


def 시험_상수():
    """#27 — 커널 생성기의 RMSNorm eps 글(생성바탕.gl 의 기본 · 커널생성.gl 의 llama큰 · qwen2e5큰)이 계약(계약.md 4 절)의 짧은실수 비트와 같은가."""
    import re
    import struct
    뿌리 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    비트 = lambda 글: struct.unpack("<I", struct.pack("<f", float(글)))[0]          # noqa: E731 — 파이썬 float → 짧은실수(가까운 쪽)
    바탕 = re.findall(r'문자열 RMS_eps = "([0-9.]+)"', open(os.path.join(뿌리, "커널", "생성바탕.gl"), encoding="utf-8").read())
    바꿈 = re.findall(r'RMS_eps = "([0-9.]+)"', open(os.path.join(뿌리, "커널", "커널생성.gl"), encoding="utf-8").read())
    계약글 = open(os.path.join(뿌리, "계약", "계약.md"), encoding="utf-8").read()
    확인("상수 — 생성기의 기본 eps 가 짧은실수(10⁻⁶) = 0x358637BD", len(바탕) == 1 and 비트(바탕[0]) == 0x358637BD, f"{바탕}")
    확인("상수 — 생성기의 eps 10⁻⁵(llama큰 · qwen2e5큰)가 0x3727C5AC", len(바꿈) >= 1 and all(비트(x) == 0x3727C5AC for x in 바꿈), f"{바꿈}")
    확인("상수 — 계약.md 가 두 비트를 적는다", "0x358637BD" in 계약글 and "0x3727C5AC" in 계약글)


def 시험_가중치16(r):
    """0.12.0 12-3b — 계약 1.3(16비트 블록 가중치 — F16 · BF16 · F32): 바꿈 대 정확한 유리수 계산, 행렬곱16 대 계약 문장을 그대로 옮긴
    유리수 계산(정밀 · 빠른 모드, 깊이가 32 의 배수가 아닌 경우의 0 채우기), 헤드에 0 을 채워도 어텐션의 비트가 같은가."""
    N, K = 7, 100                                                    # K 100 — 0 을 채워 128
    w = 반실수(r, (N, K), 1e-6, 4.0) * (2.0 ** r.integers(-12, 4, (N, 1))).astype(f32)
    w[0, :32] = 0.0                                                  # 0 인 블록
    w[1, 32:64] = 0.0
    w[1, 40] = 2.0 ** -5                                             # 최댓값이 2 의 거듭제곱
    w[2] = (w[2].astype(np.float32).view(np.uint32) & 0xFFFF0000).view(f32)   # BF16 값
    n, E = C.가중치16(w)
    다름 = 0
    정확 = 0
    for i in range(N):
        for b in range(n.shape[1] // 32):
            칸 = [Fraction(float(v)) for v in w[i, 32 * b:32 * b + 32]] + [Fraction(0)] * max(0, 32 * b + 32 - K)
            m = max(abs(v) for v in 칸)
            if m == 0:
                e = -98
            else:
                e = m.numerator.bit_length() - m.denominator.bit_length()
                if Fraction(2) ** e > m:
                    e -= 1
                e = min(max(e, -98), 100)
            다름 += int(E[i, b] != e)
            for j, v in enumerate(칸):
                q = v * Fraction(2) ** (13 - e)
                f = q.numerator // q.denominator
                나머지 = q - f
                if 나머지 > Fraction(1, 2) or (나머지 == Fraction(1, 2) and f % 2 == 1):
                    f += 1
                다름 += int(n[i, 32 * b + j] != f)
                if m and abs(v) * 8 >= m and abs(v) >= Fraction(2) ** -14 and 32 * b + j < K and i != 2:    # F16 값: 최댓값의 1/8 이상은 그대로
                    정확 += int(Fraction(int(n[i, 32 * b + j])) * Fraction(2) ** (e - 13) != v)
    확인("가중치16 — 정확한 유리수 계산(블록 지수 · 가까운 정수, 0 채우기)", 다름 == 0, f"다른 값 {다름}")
    확인("가중치16 — F16 값 가운데 블록 최댓값의 1/8 이상은 바뀌지 않는다", 정확 == 0, f"바뀐 값 {정확}")
    M = 5
    x = 활성값(r, M, 128)[:, :K].copy()
    치 = r.standard_normal(N).astype(f32)
    for 자리수 in (2, 1):
        got = C.행렬곱16(x, n, E, 치, 자리수=자리수)
        xp = np.concatenate([x, np.zeros((M, 28), f32)], 1)
        D, Ex = C.자리로(xp, 자리수)
        nx = (D[1] * 256 + D[0]) if 자리수 == 2 else D[0]
        P = 8 * 자리수 - 3
        want = np.zeros((M, N), f32)
        for a in range(M):
            for c in range(N):
                y = Fraction(0)
                for b in range(4):
                    T = int(sum(int(nx[a, 32 * b + k]) * int(n[c, 32 * b + k]) for k in range(32)))
                    u = Fraction(float(반올림R(Fraction(T))))
                    부분 = Fraction(float(반올림R(u * Fraction(2) ** (int(Ex[a, b]) - P))))
                    y = Fraction(float(반올림R(부분 * Fraction(2) ** (int(E[c, b]) - 13) + y)))
                want[a, c] = 반올림R(y + Fraction(float(치[c])))
        d = 다른칸(got, want)
        확인(f"행렬곱16(자리 {'둘' if 자리수 == 2 else '하나'}) — 계약 문장을 그대로 옮긴 유리수 계산", d == 0, f"불일치 {d} / {M * N}")
    w2 = 반실수(r, (300, K), 1e-6, 4.0)
    n2, E2 = C.가중치16(w2)
    치2 = r.standard_normal(300).astype(f32)
    한청크 = C.행렬곱16(x, n2, E2, 치2, 칸한도=1 << 30)
    여러청크 = C.행렬곱16(x, n2, E2, 치2, 칸한도=1 << 12, 일꾼=3)
    확인("행렬곱16 — 열을 청크로 나눠도 비트가 같다", 다른칸(한청크, 여러청크) == 0, f"불일치 {다른칸(한청크, 여러청크)} / {M * 300}")
    q, k, v = (활성값(r, 70, 64) * f32(0.01) for _ in range(3))
    채움 = lambda a: np.concatenate([a, np.zeros((a.shape[0], 32), f32)], 1)          # noqa: E731
    a = C.어텐션(q, k, v)
    b = C.어텐션(채움(q), 채움(k), 채움(v), 눈금=f32(0.125))[:, :64]
    확인("어텐션 — 헤드에 0 을 채워도(64 → 96, 같은 스케일) 비트가 같다", 다른칸(a, b) == 0, f"불일치 {다른칸(a, b)}")


def 시험_GDN(r):
    """0.12.0 12-3e — 계약 4.9 · 4.10(Qwen3.5): 로그1p 의 오차(정확한 log1p 대비 1 ulp 미만), σ · softplus 의 상대오차, _열합 대 계약 문장을
    그대로 옮긴 차례(스칼라 곱해더하기), GDN 순환 · 합성곱4 를 토큰 덩이로 나눠도(상태를 넘겨) 한 번에 한 것과 비트가 같은가(실행 경로의 약속)."""
    x = np.concatenate([r.uniform(0, 1, 200000), 10.0 ** r.uniform(-30, 0, 50000),
                        [0.0, 1.0, 0.4142135, 0.41421354, 0.41421357, 2.0 ** -24, 2.0 ** -25]]).astype(f32)
    ref = np.log1p(x.astype(np.float64))
    ulp = np.abs(C.로그1p(x).astype(np.float64) - ref) / np.spacing(ref.astype(f32)).astype(np.float64)
    확인("계약 4.10 — 로그1p 의 오차 < 1 ulp([0, 1] 의 25 만 점)", ulp.max() < 1.0, f"최대 {ulp.max():.3f} ulp")
    u = (r.standard_normal(100000) * 8).astype(f32)
    ud = u.astype(np.float64)
    오차σ = np.abs(C.시그모이드(u) - 1 / (1 + np.exp(-ud))) * (1 + np.exp(-ud))
    오차sp = np.abs(C.소프트플러스(u) - np.logaddexp(0, ud)) / np.logaddexp(0, ud)
    확인("계약 4.9 · 4.10 — σ · softplus 의 상대오차 < 4 · 2⁻²⁴", 오차σ.max() < 4 * 2.0 ** -24 and 오차sp.max() < 4 * 2.0 ** -24,
         f"σ {오차σ.max():.2e} · softplus {오차sp.max():.2e}")
    H, T = 2, 29
    S = (r.standard_normal((H, 128, 128)) * 0.1).astype(f32)
    xk = r.standard_normal((H, 128)).astype(f32)
    문장 = np.empty((H, 128), f32)
    for h in range(H):
        for j in range(128):
            P = []
            for b in range(4):
                p = C.곱(S[h, 32 * b, j], xk[h, 32 * b])
                for i in range(32 * b + 1, 32 * b + 32):
                    p = C.fma32(S[h, i, j], xk[h, i], p)
                P.append(p)
            문장[h, j] = C.더(C.더(P[0], P[1]), C.더(P[2], P[3]))
    확인("계약 4.10 — 열 합(kv · o) 대 계약 문장의 차례(스칼라)", 다른칸(C._열합(S, xk), 문장) == 0)
    q, k, v = ((r.standard_normal((T, H, 128)) * 0.3).astype(f32) for _ in range(3))
    베타 = C.시그모이드(r.standard_normal((T, H)).astype(f32))
    감마 = C.적힌지수(C.곱(C.소프트플러스(r.standard_normal((T, H)).astype(f32)), f32(-0.7)))
    o1, S1 = C.GDN(q, k, v, 베타, 감마)
    oa, Sa = C.GDN(q[:11], k[:11], v[:11], 베타[:11], 감마[:11])
    ob, Sb = C.GDN(q[11:], k[11:], v[11:], 베타[11:], 감마[11:], Sa)
    확인("계약 4.10 — GDN 을 토큰 덩이로 나눠도(상태를 넘겨) 비트가 같다", 다른칸(o1, np.concatenate([oa, ob])) == 0 and 다른칸(S1, Sb) == 0)
    xc = r.standard_normal((T, 96)).astype(f32)
    wc = r.standard_normal((96, 4)).astype(f32)
    y1, c1 = C.합성곱4(xc, wc)
    ya, ca = C.합성곱4(xc[:7], wc)
    yb, cb = C.합성곱4(xc[7:], wc, ca)
    확인("계약 4.10 — 합성곱을 토큰 덩이로 나눠도(상태를 넘겨) 비트가 같다", 다른칸(y1, np.concatenate([ya, yb])) == 0 and 다른칸(c1, cb) == 0)


def 시험_MoE(r):
    """0.12.0 12-3g — 계약 4.11(MoE FFN): 상위 k 는 같은 값이면 작은 번호 먼저(−0 과 +0 도 같은 값), 무게 · 합 · softmax 대 계약 문장을 그대로
    옮긴 차례(스칼라), 라우터 FMA 체인 대 스칼라 체인, 한 행의 셈이 다른 행과 상관없다(행을 묶는 길과 상관없이 같은 비트)."""
    l = np.array([[1.0, 2.0, 2.0, 0.5, 2.0], [0.0, -0.0, 0.0, -1.0, 3.0]], f32)
    확인("계약 4.11 — 상위 k: 같은 값이면 작은 번호 먼저", C.상위k(l, 3).tolist() == [[1, 2, 4], [4, 0, 1]], str(C.상위k(l, 3).tolist()))
    T, E, k, D = 7, 64, 8, 48
    lg = (r.standard_normal((T, E)) * 3).astype(f32)
    lg[2, 5] = lg[2, 9]                                            # 같은 값 하나
    p = C.소프트맥스전문가(lg)
    문장p = np.empty_like(p)
    for t in range(T):
        m = lg[t].max()
        e = [C.적힌지수(C.빼기(lg[t, i], m)) for i in range(E)]
        S = e[0]
        for i in range(1, E):
            S = C.더(S, e[i])
        문장p[t] = [C.나눔(e[i], S) for i in range(E)]
    고른 = C.상위k(lg, k)
    w = C.전문가무게(p, 고른)
    문장w = np.empty_like(w)
    for t in range(T):
        pr = [p[t, 고른[t, j]] for j in range(k)]
        s_ = pr[0]
        for j in range(1, k):
            s_ = C.더(s_, pr[j])
        문장w[t] = [C.나눔(pr[j], max(s_, f32(2.0 ** -14))) for j in range(k)]
    ys = (r.standard_normal((k, T, D))).astype(f32)
    o = C.전문가합(w, ys)
    문장o = np.empty_like(o)
    for t in range(T):
        for d in range(D):
            a = C.곱(w[t, 0], ys[0, t, d])
            for j in range(1, k):
                a = C.fma32(w[t, j], ys[j, t, d], a)
            문장o[t, d] = a
    확인("계약 4.11 — softmax · 무게 · 합 대 계약 문장의 차례(스칼라)", 다른칸(p, 문장p) == 0 and 다른칸(w, 문장w) == 0 and 다른칸(o, 문장o) == 0)
    h = r.standard_normal((T, 96)).astype(f32)
    W = (r.standard_normal((E, 96)) * 0.05).astype(f32)
    lf = C.라우터FMA(h, W)
    xq = C.활성값16(h)
    문장l = np.empty_like(lf)
    for t in range(T):
        for e_ in range(E):
            a = C.곱(xq[t, 0], W[e_, 0])
            for kk in range(1, 96):
                a = C.fma32(xq[t, kk], W[e_, kk], a)
            문장l[t, e_] = a
    확인("계약 4.11 — 라우터 FMA 체인 대 계약 문장의 차례(스칼라)", 다른칸(lf, 문장l) == 0)
    둘 = C.전문가합(w[[4, 1]], ys[:, [4, 1]])
    확인("계약 4.11 — 한 행의 셈은 다른 행과 상관없다(행을 고르고 묶어도 같은 비트)", 다른칸(둘, o[[4, 1]]) == 0 and
         다른칸(C.라우터FMA(h[[6, 0]], W), lf[[6, 0]]) == 0)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    r = np.random.default_rng(27)
    시험_상수()
    시험_fma32(r)
    시험_행렬곱(r)
    시험_K(r)
    시험_규칙R(r)
    시험_머리정규화(r)
    시험_로그확률(r)
    시험_KV8(r)
    시험_F2(r)
    시험_가중치16(r)
    시험_GDN(r)
    시험_MoE(r)
    print("계약 자체 시험:", "모두 통과" if not 실패 else f"못 미침 {실패}", flush=True)
    return 0 if not 실패 else 1


if __name__ == "__main__":
    sys.exit(main())
