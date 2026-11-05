#!/usr/bin/env python3
"""llama.cpp 재현 7단계(문서/라마재현.md) — GPU 없는 시험: llama-server(b12008) 한 슬롯의 프롬프트 캐시 셈 흉내(앱/라마사슬.gl)가 서버
코드(tools/server/server-context.cpp — get_available_slot 의 LCP 비율 · 남기는 몫, n_past, 체크포인트 되살리기 · 지우기, 프롬프트의
덩어리 나눔 · create_checkpoint, 생성 걸음, 9단계 — 채팅 API 의 사용자 메시지 자리에서 끊음, 10단계 — 되돌릴 수 있는 메모리(dense):
체크포인트 없이 n_past = 겹치는 앞, 프롬프트 전부가 캐시면 n_past − 1, 16단계 — 캐시 끝이 ubatch 가운데면 그 ubatch 를 남기고
n_past 부터 이어 셈: 시작이 n_past 앞인 이력을 차례대로 남김)를 옮긴 파이썬 참조와 같은 cache_n · 되살림 · 슬롯의 토큰 수 · ubatch 사슬
(위치 · 크기 · 토큰의 합 셈)을 내는가.

  python -I -X utf8 시험/라마사슬.py <시험_라마사슬 실행 파일>
      정한 무작위 대화(잇기 · 같은 프롬프트 · 생성 잇기 · 앞을 고침 · 다른 프롬프트)를 설정 여섯으로 견준다(빠른 점검 4w).
      마지막 줄 "모두 같음: True|False".
  python -I -X utf8 시험/라마사슬.py --영수증 <폴더>
      폴더의 r1.json, r2.json, …(한 슬롯의 이어지는 요청 — prompt_tokens · response)로 참조의 cache_n · tokens_cached 를 서버와 견준다.
      폴더에 구분.json(user_start — 채팅 API 의 사용자 구분 토큰열)이 있으면 프롬프트에서 사용자 메시지 자리를 찾는다.
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
    def __init__(self, nb, nu, cp, ms, sp, ram, 슬롯수=4, 되돌림=0):
        self.nb, self.nu, self.cp, self.ms, self.sp, self.ram, self.슬롯수 = nb, nu, cp, ms, F32(sp), ram, 슬롯수
        self.되돌림 = 되돌림                                           # 10단계 — 1: dense(PART — 체크포인트 없음)
        self.tokens, self.cps, self.hist = [], [], []                 # cps: [n_tokens, 작업], hist: [(p0, T, 토큰들)]
        self.램 = []                                                  # 이 사슬에서 RAM 캐시에 저장했을 슬롯의 토큰들(위 집합)
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

    def 요청(self, prompt, gen, task, 사용자=()):
        self.되살림 = -1
        n, m = len(prompt), len(self.tokens)
        if n < 1:
            return -1, []
        lcp = 0
        while lcp < m and lcp < n and self.tokens[lcp] == prompt[lcp]:
            lcp += 1
        if m > 0:                                                  # get_available_slot · server_prompt_cache::load
            f_sim = F32(lcp) / F32(n)
            if self.sp > 0 and f_sim > self.sp:
                고침 = (f_sim * F32(n)) / F32(m) < F32(0.5)
            else:
                if self.슬롯수 > 1:                                      # LRU — 쓰지 않은 빈 슬롯이 먼저
                    return -1, []
                고침 = True
            if 고침 and self.ram != 0:
                self.램.append(list(self.tokens))
                기남김 = F32(lcp) / F32(m)
                for e in self.램:
                    l2 = 0
                    while l2 < len(e) and l2 < n and e[l2] == prompt[l2]:
                        l2 += 1
                    e남김, e유사 = F32(l2) / F32(len(e)), F32(l2) / F32(n)
                    if e남김 >= F32(0.25) and 기남김 < e남김 and f_sim < e유사:
                        return -1, []
        체크 = self.cp > 0 and not self.되돌림                         # do_checkpoint(server-context.cpp 4391)
        n_past = lcp
        pos_next = n_prefix = n_past
        pos_eval = pos_next - (0 if n_past < n else 1)
        thold = max(0, pos_eval)
        if 0 < n_past <= m:
            pos_min = 0 if self.되돌림 else m - 1
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
        if n_past == n and n_past > 0:                             # [TAG_PROMPT_LOGITS]
            if not self.되돌림:
                return -1, []
            n_past -= 1
        self.tokens = self.tokens[:n_past]
        if self.되돌림:
            # 16단계 — dense: 캐시 [0, n_past) 를 쓴 이력은 시작이 n_past 앞인 ubatch 들(차례대로). 가운데에 걸린 ubatch 는 다 셈한 뒤
            # n_past 부터 덮어 쓴다(서버: 그 ubatch 를 다 셈했고 seq_rm 으로 n_past 뒤를 지웠다)
            self.hist = [e for e in self.hist if e[0] < n_past]
        else:
            h = [e for e in self.hist if e[0] + e[1] <= n_past]
            if (h[-1][0] + h[-1][1] if h else 0) != n_past:
                return -1, []
            self.hist = h
        cache_n = n_past
        o1, o2 = min(self.nb, 4 + self.nu), min(self.nb, 4)
        last_user = 사용자[-1] if 사용자 else -1
        cur = n_past
        while cur < n:
            b0 = cur
            while cur < n and cur - b0 < self.nb:
                cur += 1
                if 체크 and cur in 사용자:                         # 사용자 메시지 앞에서 끊음(server-context.cpp 4502)
                    if cur == last_user or not self.cps or cur > self.cps[-1][0] + self.ms:
                        break
                if 체크 and (n == cur + o1 or n == cur + o2):
                    break
            b1 = cur
            if 체크:
                do = True
                near = n < b1 + self.nu
                if b1 < n and b0 not in 사용자 and not near:
                    do = False
                if b0 <= 0:
                    do = False
                if do and self.cps and b0 != last_user and not near and b0 <= self.cps[-1][0] + self.ms:
                    do = False
                if do:
                    self.체크만들기(b0, task)
            for u0 in range(b0, b1, self.nu):
                T = min(self.nu, b1 - u0)
                self.hist.append((u0, T, list(prompt[u0:u0 + T])))
        self.tokens = list(prompt)
        사슬 = list(self.hist)
        for i, t in enumerate(gen[:-1]):
            self.hist.append((n + i, 1, [t]))
            self.tokens.append(t)
        return cache_n, 사슬


설정들 = [  # n_batch, n_ubatch, 체크수, 최소걸음, 유사, 램, 슬롯 수 — 서버 기본 · 작은 덩어리 · 체크포인트 적음 · 끔 · 최소걸음 ·
    # LCP 비율 끔 · 슬롯 하나(LRU 가 이 슬롯)
    (2048, 512, 32, 8192, 0.1, 8192, 4),
    (512, 128, 32, 8192, 0.1, 8192, 4),
    (64, 16, 3, 0, 0.1, 0, 4),
    (2048, 512, 0, 8192, 0.1, 8192, 4),
    (100, 30, 2, 50, 0.5, 0, 4),
    (2048, 512, 32, 8192, 0.0, 8192, 4),
    (2048, 512, 32, 8192, 0.1, 8192, 1),
    (64, 16, 4, 0, 0.0, 8192, 1),
    # 10단계 — 되돌릴 수 있는 메모리(dense): 서버 기본 · 슬롯 하나 · 작은 덩어리
    (2048, 512, 32, 8192, 0.1, 8192, 4, 1),
    (2048, 512, 32, 8192, 0.1, 8192, 1, 1),
    (64, 16, 4, 0, 0.0, 8192, 1, 1),
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
                자리 = sorted(set(rnd.randrange(len(p)) for _ in range(rnd.choice([0, 0, 1, 3, 8])))) if rnd.random() < 0.6 else []
                reqs.append((p, g, 자리))
                prev = (p, g)
            out.append((설정, reqs))
    return out


def 설정풀기(설정):
    return 설정 if len(설정) == 8 else (*설정, 0)


def 토큰셈(사슬):
    # 16단계 — 사슬 토큰의 합 셈: Σ (i + 1)·(토큰 + 1) mod 1,000,000,007(차례대로 이은 토큰)
    k, v = 0, 0
    for _, _, tk in 사슬:
        for t in tk:
            k += 1
            v = (v + k * (t + 1)) % 1000000007
    return v


def 참조출력(cases):
    out = []
    for 설정, reqs in cases:
        s = 슬롯(*설정풀기(설정))
        멈춤 = False
        for r, (p, g, 자리) in enumerate(reqs):
            if 멈춤:
                continue
            c, 사슬 = s.요청(p, g, r + 1, 자리)
            out.append(c)
            if c < 0:
                멈춤 = True
                out += [0, 0, 0]
            else:
                out += [s.되살림, len(s.tokens), len(사슬)]
                for p0, T, _ in 사슬:
                    out += [p0, T]
                out.append(토큰셈(사슬))
    return out


def 사례파일(cases, 경로):
    v = [len(cases)]
    for 설정, reqs in cases:
        nb, nu, cp, ms, sp, ram, ns, dg = 설정풀기(설정)
        v += [nb, nu, cp, ms, struct.unpack("<I", struct.pack("<f", sp))[0], ram, ns, dg, len(reqs)]
        for p, g, 자리 in reqs:
            v += [len(p), *p, len(g), *g, len(자리), *자리]
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
    for 설정, reqs in cases:
        s = 슬롯(*설정풀기(설정))
        for r, (p, g, 자리) in enumerate(reqs):
            c, _ = s.요청(p, g, r + 1, 자리)
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


def 자리찾기(tokens, 구분):
    return [i for i in range(len(tokens)) if 구분 and tokens[i:i + len(구분)] == 구분]


def 영수증(폴더):
    s = None
    k, 나쁨 = 1, 0
    gp = os.path.join(폴더, "구분.json")
    구분 = json.load(open(gp))["user_start"] if os.path.exists(gp) else []
    while os.path.exists(os.path.join(폴더, f"r{k}.json")):
        d = json.load(open(os.path.join(폴더, f"r{k}.json"), encoding="utf-8"))
        if s is None:
            s = 슬롯(2048, 512, 32, 8192, 0.1, 8192, 4)
        r = d["response"]
        c, 사슬 = s.요청(d["prompt_tokens"], r["tokens"], k, 자리찾기(d["prompt_tokens"], 구분))
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
