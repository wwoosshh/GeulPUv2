#!/usr/bin/env python3
"""llama.cpp 재현 5단계(문서/라마재현.md) — GPU 없는 시험: llama-server(b12008)의 CPU 고르기 사슬 흉내(엔진/라마고르기.gl)가 파이썬 참조와
같은 토큰을 고르는가. 참조는 서버 코드(common/sampling.cpp · src/llama-sampler.cpp)를 옮긴 것 — top_k(MSVC std::partial_sort) · top_p ·
min_p · temperature → dist(seed): std::mt19937 과 MSVC uniform_real_distribution<double> 의 두 판(VS 2022 17.12 부터의 P0952R2 · 그 전),
expf · logf 는 서버와 같은 C 런타임(윈도 ucrtbase · 리눅스 libm, ctypes).

  python -I -X utf8 시험/라마고르기.py <시험_라마고르기 실행 파일>
      정한 무작위 로짓(맨 위가 뚜렷한 줄 · 같은 logit 이 많은 줄)으로 온도 0 · 0.6 · 0.8 · 1 · 1.5, top_k 20 · 40 · 128, min_keep 0 · 3,
      판 둘을 견준다(빠른 점검 4v). 36단계 — 반복 벌점(repeat · frequency · presence, repeat_last_n — 프롬프트 토큰의 끝을 받아들인 뒤
      걸음마다 고른 토큰을 받아들인다)과 logit_bias(−inf 포함)를 준 사례. 마지막 줄 "모두 같음: True|False".
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 라마영수증 import _c, 부분정렬  # noqa: E402  (MSVC partial_sort · 같은 C 런타임의 expf · logf)

F32 = np.float32


class MT19937:
    """std::mt19937(seed) — init_genrand"""

    def __init__(self, seed):
        self.mt = [0] * 624
        self.mt[0] = seed & 0xFFFFFFFF
        for i in range(1, 624):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self.i = 624

    def __call__(self):
        if self.i >= 624:
            for k in range(624):
                y = (self.mt[k] & 0x80000000) | (self.mt[(k + 1) % 624] & 0x7FFFFFFF)
                self.mt[k] = self.mt[(k + 397) % 624] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
            self.i = 0
        y = self.mt[self.i]
        self.i += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF


def 고름수(g, 판):
    """uniform_real_distribution<double>(0, 1) — 판 1: (g2 << 21) + (g1 >> 11) 을 2⁻⁵³ 배, 판 0: double 로 g1 + g2·2³² 를 2⁶⁴ 로"""
    g1 = g()
    g2 = g()
    if 판 == 1:
        return float((g2 << 21) + (g1 >> 11)) * 2.0 ** -53
    return (float(g1) + float(g2) * 4294967296.0) / 18446744073709551616.0


def 고르기(로짓, g, 판, top_k, top_p, min_p, 온도, min_keep=0):
    cur = [(F32(v), i) for i, v in enumerate(로짓)]
    k = min(top_k, len(cur))
    부분정렬(cur, k)
    cur = cur[:k]
    if top_p < 1.0:
        mx = cur[0][0]
        ps = [F32(_c.expf(float(F32(l - mx)))) for l, _ in cur]
        s = F32(0)
        for p in ps:
            s = F32(s + p)
        ps = [F32(p / s) for p in ps]
        acc, last = F32(0), len(cur)
        for i, p in enumerate(ps):
            acc = F32(acc + p)
            if acc >= F32(top_p) and i + 1 >= min_keep:
                last = i + 1
                break
        cur = cur[:last]
    if min_p > 0.0 and cur:
        lo = F32(cur[0][0] + F32(_c.logf(float(F32(min_p)))))
        i = 1
        while i < len(cur) and not (cur[i][0] < lo and i >= min_keep):
            i += 1
        cur = cur[:i]
    if 온도 <= 0.0:
        out, mi = list(cur), 0
        for i in range(1, len(cur)):
            if cur[i][0] > out[mi][0]:
                out[mi] = (F32(-np.inf), out[mi][1])
                mi = i
            else:
                out[i] = (F32(-np.inf), out[i][1])
        cur = out
    else:
        cur = [(F32(l / F32(온도)), t) for l, t in cur]
    if len(cur) == 1:
        고름수(g, 판)
        return cur[0][1]
    mx = cur[0][0]
    ps = [F32(_c.expf(float(F32(l - mx)))) for l, _ in cur]
    s = 0.0
    for p in ps:
        s += float(p)
    tgt = s * 고름수(g, 판)
    run = 0.0
    for i, p in enumerate(ps):
        run += float(p)
        if run >= tgt:
            return cur[i][1]
    return cur[-1][1]


class 벌점:
    """36단계 — llama_sampler_penalties(accept · apply)와 logit_bias(사슬의 맨 앞)"""

    def __init__(self, n, rep, freq, pres, 편향):
        self.n, self.rep, self.freq, self.pres, self.편향 = n, F32(rep), F32(freq), F32(pres), 편향
        self.cnt, self.ring = {}, []

    def 받기(self, t):
        if self.n == 0:
            return
        self.cnt[t] = self.cnt.get(t, 0) + 1
        if len(self.ring) >= self.n:
            old = self.ring.pop(0)
            self.cnt[old] -= 1
            if self.cnt[old] == 0:
                del self.cnt[old]
        self.ring.append(t)

    def 손질(self, 로짓):
        로짓 = 로짓.copy()
        for t, b in self.편향:
            로짓[t] = F32(로짓[t] + F32(b))
        if self.n == 0 or (self.rep == F32(1) and self.freq == F32(0) and self.pres == F32(0)):
            return 로짓
        for t, c in self.cnt.items():
            l = 로짓[t]
            l = F32(l * self.rep) if l <= F32(0) else F32(l / self.rep)
            로짓[t] = F32(l - F32(F32(F32(c) * self.freq) + self.pres))
        return 로짓


def 비트(x):
    return struct.unpack("<I", struct.pack("<f", float(F32(x))))[0]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    exe = os.path.abspath(sys.argv[1])
    g0 = MT19937(5489)                                            # 표준이 정한 값: 10000 번째 출력 4123659995
    for _ in range(9999):
        g0()
    나쁨 = int(g0() != 4123659995)
    전체 = 1
    사례 = [(2000, 40, 1, 0.8, 40, 0.95, 0.05, 0, 1, None), (2000, 40, 2, 0.8, 40, 0.95, 0.05, 0, 0, None),
            (5000, 30, 7, 1.5, 40, 0.9, 0.02, 0, 1, None), (5000, 30, 9, 0.0, 40, 0.95, 0.05, 0, 1, None),
            (3000, 30, 11, 0.6, 20, 1.0, 0.0, 0, 0, None), (3000, 30, 12, 1.0, 128, 0.99, 0.1, 3, 1, None)]
    # 36단계 — (repeat_last_n, repeat, frequency, presence, logit_bias [(토큰, 편향)], 이력 길이)
    사례 += [(2000, 40, 21, 0.8, 40, 0.95, 0.05, 0, 1, (64, 1.1, 0.0, 0.0, [], 80)),
             (2000, 40, 22, 0.0, 40, 0.95, 0.05, 0, 1, (64, 1.3, 0.0, 0.0, [], 50)),
             (3000, 40, 23, 0.7, 40, 0.9, 0.05, 0, 1, (32, 1.05, 0.4, 0.3, [(7, -2.5), (11, 1.5), (7, 0.25)], 10)),
             (3000, 40, 24, 1.0, 40, 0.95, 0.0, 0, 1, (0, 1.2, 0.0, 0.0, [(7, float("-inf")), (100, 3.0)], 5)),
             (2000, 40, 25, 0.8, 128, 1.0, 0.0, 0, 1, (16, 1.0, 0.8, 0.0, [], 0))]
    for (V, R, seed, temp, k, tp, mp, keep, 판, 덧) in 사례:
        g = np.random.default_rng(seed)
        rows = g.normal(0, 3, (R, V)).astype(F32)
        rows[:, 7] = 9.0                                          # 맨 위가 뚜렷한 줄
        rows[5, :50] = 4.0                                        # 같은 logit 이 많은 줄
        이력 = [int(x) for x in g.integers(0, 60, 덧[5])] if 덧 else []        # 프롬프트 토큰의 끝(작은 번호 — 맨 위와 겹치게)
        덧글 = b""
        if 덧:
            n_, r_, f_, p_, 편, _ = 덧
            덧글 = np.array([n_, 비트(r_), 비트(f_), 비트(p_), len(편)] + [v for t, b in 편 for v in (t, 비트(b))] + [len(이력)] + 이력,
                           np.int64).tobytes()
        with tempfile.TemporaryDirectory() as td:
            i, o = os.path.join(td, "in.bin"), os.path.join(td, "out.bin")
            with open(i, "wb") as f:
                f.write(np.array([V, R, k, 비트(tp), 비트(mp), 비트(temp), seed, 판, keep], np.int64).tobytes())
                f.write(rows.tobytes())
                f.write(덧글)
            r = subprocess.run([exe, i, o], capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"시험_라마고르기 실패({r.returncode}): {r.stdout}{r.stderr}")
            got = np.fromfile(o, np.int64)
        rng = MT19937(seed)
        if 덧:
            bp = 벌점(덧[0], 덧[1], 덧[2], 덧[3], 덧[4])
            for t in 이력:
                bp.받기(t)
            want = []
            for q in range(R):
                w = 고르기(bp.손질(rows[q]), rng, 판, k, tp, mp, temp, keep)
                want.append(w)
                bp.받기(w)
        else:
            want = [고르기(rows[q], rng, 판, k, tp, mp, temp, keep) for q in range(R)]
        d = sum(1 for a, b in zip(got, want) if int(a) != b)
        나쁨 += d
        전체 += R
        if d:
            print(f"V {V} 씨앗 {seed} 온도 {temp} k {k} p {tp} min_p {mp} keep {keep} 판 {판}: 다름 {d}/{R}")
    print(f"고르기 사슬: 걸음 {전체 - 1} · mt19937 표준 값 · 다름 {나쁨}")
    print(f"모두 같음: {나쁨 == 0}")
    return 0 if 나쁨 == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
