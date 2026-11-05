#!/usr/bin/env python3
"""llama.cpp 재현 7단계(문서/라마재현.md) — GPU 없는 시험: llama-server(b12008) 한 슬롯의 프롬프트 캐시 셈 흉내(앱/라마사슬.gl)가 서버
코드(tools/server/server-context.cpp — get_available_slot 의 LCP 비율 · 남기는 몫, n_past, 체크포인트 되살리기 · 지우기, 프롬프트의
덩어리 나눔 · create_checkpoint, 생성 걸음)를 옮긴 파이썬 참조와 같은 cache_n · 되살림 · 슬롯의 토큰 수 · ubatch 사슬을 내는가.

  python -I -X utf8 시험/라마사슬.py <시험_라마사슬 실행 파일>
      정한 무작위 대화(잇기 · 같은 프롬프트 · 생성 잇기 · 앞을 고침 · 다른 프롬프트)를 설정 여섯으로 견준다(빠른 점검 4w).
      마지막 줄 "모두 같음: True|False".
  python -I -X utf8 시험/라마사슬.py --영수증 <폴더>
      폴더의 r1.json, r2.json, …(한 슬롯의 이어지는 요청 — prompt_tokens · response)로 참조의 cache_n · tokens_cached 를 서버와 견준다.
"""
import json
import os
import random
import struct
import subprocess
import sys
import tempfile

import numpy as np

F32 = np.float32


class 슬롯:
    def __init__(self, nb, nu, cp, ms, sp, ram):
        self.nb, self.nu, self.cp, self.ms, self.sp, self.ram = nb, nu, cp, ms, F32(sp), ram
        self.tokens, self.cps, self.hist = [], [], []                 # cps: [n_tokens, 작업], hist: [(p0, T)]
        self.되살림 = -1

    def 체크만들기(self, b0, task):
        last, i = -1, 0
        while len(self.cps) + 1 >= self.cp and i < len(self.cps):
            cn, ct = self.cps[i]
            if ct != task and last >= 0 and cn <= last + self.ms:
                del self.cps[i]
            else:
                last = cn
                i += 1
        while len(self.cps) >= self.cp and self.cps:
            del self.cps[0]
        self.cps = [c for c in self.cps if c[0] != b0]
        self.cps.append([b0, task])

    def 요청(self, prompt, gen, task):
        self.되살림 = -1
        n, m = len(prompt), len(self.tokens)
        if n < 1:
            return -1, []
        lcp = 0
        while lcp < m and lcp < n and self.tokens[lcp] == prompt[lcp]:
            lcp += 1
        if m > 0:
            if self.sp <= 0:
                return -1, []
            f_sim = F32(lcp) / F32(n)
            if not f_sim > self.sp:
                return -1, []
            f_keep = (f_sim * F32(n)) / F32(m)
            if f_keep < F32(0.5) and self.ram != 0:
                return -1, []
        n_past = lcp
        pos_next = n_prefix = n_past
        pos_eval = pos_next - (0 if n_past < n else 1)
        thold = max(0, pos_eval)
        if 0 < n_past <= m:
            pos_min = m - 1
            if pos_min >= thold:
                found = None
                for cn, _ in reversed(self.cps):
                    pm = cn - 1
                    if pm > pos_next:
                        continue
                    if pm < thold or pm == 0:
                        found = cn
                        break
                if found is not None:
                    pos_next = min(pos_next, found)
                    n_past = min(pos_next, found)
                    self.되살림 = found
                else:
                    pos_next = n_past = 0
                    self.되살림 = 0
        self.cps = [c for c in self.cps if not (c[0] - 1 > pos_next or c[0] > n_prefix)]
        if n_past == n and n_past > 0:
            return -1, []
        self.tokens = self.tokens[:n_past]
        h = [(p0, T) for p0, T in self.hist if p0 + T <= n_past]
        if (h[-1][0] + h[-1][1] if h else 0) != n_past:
            return -1, []
        self.hist = h
        cache_n = n_past
        o1, o2 = min(self.nb, 4 + self.nu), min(self.nb, 4)
        cur = n_past
        while cur < n:
            b0 = cur
            while cur < n and cur - b0 < self.nb:
                cur += 1
                if self.cp > 0 and (n == cur + o1 or n == cur + o2):
                    break
            b1 = cur
            if self.cp > 0:
                do = True
                near = n < b1 + self.nu
                if b1 < n and not near:
                    do = False
                if b0 <= 0:
                    do = False
                if do and self.cps and not near and b0 <= self.cps[-1][0] + self.ms:
                    do = False
                if do:
                    self.체크만들기(b0, task)
            for u0 in range(b0, b1, self.nu):
                self.hist.append((u0, min(self.nu, b1 - u0)))
        self.tokens = list(prompt)
        사슬 = [T for _, T in self.hist]
        for i, t in enumerate(gen[:-1]):
            self.hist.append((n + i, 1))
            self.tokens.append(t)
        return cache_n, 사슬


설정들 = [  # n_batch, n_ubatch, 체크수, 최소걸음, 유사, 램 — 서버 기본 · 작은 덩어리 · 체크포인트 적음 · 끔 · 최소걸음 · LCP 비율 끔
    (2048, 512, 32, 8192, 0.1, 8192),
    (512, 128, 32, 8192, 0.1, 8192),
    (64, 16, 3, 0, 0.1, 0),
    (2048, 512, 0, 8192, 0.1, 8192),
    (100, 30, 2, 50, 0.5, 0),
    (2048, 512, 32, 8192, 0.0, 8192),
]


def 사례들(씨앗=7, 수=12):
    rnd = random.Random(씨앗)
    out = []
    for 설정 in 설정들:
        for _ in range(수):
            길이 = rnd.choice([5, 40, 300, 700, 1500, 2600])
            prompt = [rnd.randrange(1000) for _ in range(길이)]
            reqs = []
            prev = None
            for k in range(rnd.randint(2, 8)):
                if prev is None:
                    p = prompt
                else:
                    pp, pg = prev
                    갈래 = rnd.choice(["잇기", "잇기", "잇기", "같음", "생성잇기", "고침", "다른것", "일부"])
                    새 = [rnd.randrange(1000) for _ in range(rnd.choice([1, 3, 20, 90, 600]))]
                    if 갈래 == "잇기":
                        p = pp + pg + 새
                    elif 갈래 == "일부":                          # 다시 토큰화가 달라진 응답 — 생성의 앞만 같다
                        p = pp + pg[:rnd.randrange(len(pg) + 1)] + 새
                    elif 갈래 == "같음":
                        p = list(pp)
                    elif 갈래 == "생성잇기":
                        p = pp + pg
                    elif 갈래 == "고침":
                        p = pp[:rnd.randrange(1, len(pp) + 1)] + 새
                    else:
                        p = 새 + [rnd.randrange(1000) for _ in range(30)]
                g = [rnd.randrange(1000) for _ in range(rnd.choice([1, 2, 8, 64]))]
                reqs.append((p, g))
                prev = (p, g)
            out.append((설정, reqs))
    return out


def 참조출력(cases):
    out = []
    for (nb, nu, cp, ms, sp, ram), reqs in cases:
        s = 슬롯(nb, nu, cp, ms, sp, ram)
        멈춤 = False
        for r, (p, g) in enumerate(reqs):
            if 멈춤:
                continue
            c, 사슬 = s.요청(p, g, r + 1)
            out.append(c)
            if c < 0:
                멈춤 = True
                out += [0, 0, 0]
            else:
                out += [s.되살림, len(s.tokens), len(사슬), *사슬]
    return out


def 사례파일(cases, 경로):
    v = [len(cases)]
    for (nb, nu, cp, ms, sp, ram), reqs in cases:
        v += [nb, nu, cp, ms, struct.unpack("<I", struct.pack("<f", sp))[0], ram, len(reqs)]
        for p, g in reqs:
            v += [len(p), *p, len(g), *g]
    np.array(v, np.int64).tofile(경로)


def 시험(exe):
    cases = 사례들()
    want = 참조출력(cases)
    with tempfile.TemporaryDirectory() as d:
        inp, outp = os.path.join(d, "사례.bin"), os.path.join(d, "출력.bin")
        사례파일(cases, inp)
        r = subprocess.run([exe, inp, outp], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"시험_라마사슬 실패({r.returncode}): {r.stdout}{r.stderr}")
        got = np.fromfile(outp, np.int64).tolist()
    같음 = got == want
    요청 = sum(1 for (_, reqs) in cases for _ in reqs)
    # 참조의 갈래 세기(되살림 · 셈하지 않음)
    되살림 = 셈안함 = 이음 = 0
    for (nb, nu, cp, ms, sp, ram), reqs in cases:
        s = 슬롯(nb, nu, cp, ms, sp, ram)
        for r, (p, g) in enumerate(reqs):
            c, _ = s.요청(p, g, r + 1)
            if c < 0:
                셈안함 += 1
                break
            if s.되살림 > 0:
                되살림 += 1
            elif s.되살림 < 0 and c > 0:
                이음 += 1
    if not 같음:
        k = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        print(f"처음 다른 자리 {k}: 글 {got[k:k + 8]} · 참조 {want[k:k + 8]} (길이 {len(got)} · {len(want)})")
    print(f"사례 {len(cases)} · 요청 {요청} · 체크포인트 되살림 {되살림} · 되살리지 않고 이음 {이음} · 셈하지 않음 {셈안함}")
    print(f"모두 같음: {같음}")
    return 0 if 같음 else 1


def 영수증(폴더):
    s = None
    k, 나쁨 = 1, 0
    while os.path.exists(os.path.join(폴더, f"r{k}.json")):
        d = json.load(open(os.path.join(폴더, f"r{k}.json"), encoding="utf-8"))
        if s is None:
            s = 슬롯(2048, 512, 32, 8192, 0.1, 8192)
        r = d["response"]
        c, 사슬 = s.요청(d["prompt_tokens"], r["tokens"], k)
        서c, 서t = r["timings"]["cache_n"], r.get("tokens_cached")
        같음 = c == 서c and len(s.tokens) == 서t
        나쁨 += not 같음
        print(f"r{k}: cache_n 참조 {c} · 서버 {서c} · tokens_cached 참조 {len(s.tokens)} · 서버 {서t} · 되살림 {s.되살림} · ubatch {len(사슬)}")
        k += 1
    print(f"모두 같음: {나쁨 == 0}")
    return 0 if 나쁨 == 0 else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) == 3 and sys.argv[1] == "--영수증":
        sys.exit(영수증(sys.argv[2]))
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(시험(sys.argv[1]))
