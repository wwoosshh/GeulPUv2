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
  python -I -X utf8 시험/라마사슬.py --서버영수증 <자료> <out> [하이브리드]
      ('하이브리드' 를 주면 qwen35 의 체크포인트 규칙 — 없으면 dense) 24단계 —기록기의 영수증(자료\\영수증)을 out\\영수증.json 의 차례로 서버 단위 흉내에 넣어 cache_n · tokens_cached 를 견준다.
      25단계 — out\\llama-server.log 가 KV 디버그 기록(LLAMA_KV_CACHE_DEBUG=1 · -v)이면 칸 흉내의 find_slot 마다(used_max_p1 · used ·
      head)도 견준다.
25단계 — 서버 흉내는 통합 KV 의 칸(find_slot · apply_ubatch · seq_rm · 상태 되살리기 · get_n_kv — 칸판)도 흉내 내고, 시험은 칸 배치의 합
셈 · 칸이 위치와 다른가를 견주며, 칸판만의 무작위 시험(작은 KV — 끝에서 0 으로 돎 · 칸 모자람)을 더한다.
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


def 패드256(x):
    return (x + 255) // 256 * 256


class 칸판:
    """25단계 — 통합 KV 의 칸(src/llama-kv-cache.cpp — 흐름 하나): 칸마다 (시퀀스, 위치), 머리. find_slot(cont = false — 머리부터 빈 칸을
    하나씩, 머리 > 쓴 칸 + 2·n 이면 0 부터, 끝에 닿으면 0 으로) · apply_ubatch(머리 = 마지막 칸 + 1) · seq_rm(머리 = min(머리, 처음 비운
    칸)) · get_n_kv(min(크기, max(256, used_max_p1 을 256 의 배수로 올림)))"""

    def __init__(self, 크기):
        self.크기 = 크기
        self.seq = [-1] * 크기
        self.pos = [0] * 크기
        self.머리 = 0
        self.쓴수 = 0
        self.끝 = 0                                                   # used_max_p1
        self.자취 = []                                                # find_slot 마다 (used_max_p1, used, head) — 서버의 KV 디버그 기록과 견줌
        self.돎 = self.처음부터 = 0                                   # 시험의 갈래 세기 — 끝에서 0 으로 · head 가 멀어 0 부터

    def 지우기(self, seq, p0=0):                                     # seq_rm(seq, p0, −1)
        새머리 = None
        for i in range(self.끝):
            if self.seq[i] == seq and self.pos[i] >= p0:
                self.seq[i] = -1
                self.쓴수 -= 1
                if 새머리 is None:
                    새머리 = i
        if 새머리 is not None and 새머리 < self.머리:
            self.머리 = 새머리
        while self.끝 > 0 and self.seq[self.끝 - 1] < 0:
            self.끝 -= 1

    def 찾기(self, n):                                               # find_slot(ubatch, cont = false)
        self.자취.append((self.끝, self.쓴수, self.머리))
        h = self.머리
        if h > self.쓴수 + 2 * n:
            h = 0
            self.처음부터 += 1
        if n > self.크기:
            return None
        out, tested = [], 0
        while True:
            if h + 1 > self.크기:
                tested += self.크기 - h
                h = 0
                self.돎 += 1
                continue
            idx = h
            h += 1
            tested += 1
            if self.seq[idx] < 0:
                out.append(idx)
            if len(out) == n:
                return out
            if tested >= self.크기:
                return None

    def 놓기(self, cells, seq, poss):                                # apply_ubatch
        for c, p in zip(cells, poss):
            self.seq[c] = seq
            self.pos[c] = p
            self.쓴수 += 1
            self.끝 = max(self.끝, c + 1)
        self.머리 = cells[-1] + 1

    def nkv(self):                                                  # get_n_kv
        return min(self.크기, max(256, 패드256(self.끝)))


class 슬롯:
    def __init__(self, nb, nu, cp, ms, sp, ram, 슬롯수=4, 되돌림=0):
        self.nb, self.nu, self.cp, self.ms, self.sp, self.ram, self.슬롯수 = nb, nu, cp, ms, F32(sp), ram, 슬롯수
        self.되돌림 = 되돌림                                           # 10단계 — 1: dense(PART — 체크포인트 없음)
        self.tokens, self.cps, self.hist = [], [], []                 # cps: [n_tokens, 작업], hist: [(p0, T, 토큰들, 칸들, n_kv)]
        self.램 = []                                                  # 이 사슬에서 RAM 캐시에 저장했을 슬롯의 토큰들(위 집합)
        self.되살림 = -1
        self.서버흉내 = False                                         # 24단계 — 참이면 슬롯 고르기 · RAM 캐시는 서버(아래)가 한다
        # 25단계 — 서버 흉내의 KV 칸: 슬롯 번호(시퀀스) · 칸판(없으면 칸을 셈하지 않음) · 위치 → 칸. 이력의 항목은 ubatch(p0 ≥ 0 — 토큰마다
        # 칸, 놓은 뒤의 n_kv) 또는 옮김(p0 = −1, T = 위치 수 — RAM 캐시에서 불러와 위치 0 … T − 1 이 놓인 새 칸, 토큰 없음)
        self.번호 = -1
        self.칸판 = None
        self.칸 = {}
        self.까닭 = None

    def 항목(self, p0, T, toks):
        if self.칸판 is None:
            return (p0, T, toks, None, 0)
        cells = self.칸판.찾기(T)
        if cells is None:                                         # 서버도 이 ubatch 를 놓지 못한다(디코드 실패)
            self.까닭 = "칸 모자람"
            return (p0, T, toks, None, 0)
        self.칸판.놓기(cells, self.번호, range(p0, p0 + T))
        for t in range(T):
            self.칸[p0 + t] = cells[t]
        return (p0, T, toks, cells, self.칸판.nkv())

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
        if m > 0 and not self.서버흉내:                             # get_available_slot · server_prompt_cache::load
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
            # n_past 부터 덮어 쓴다(서버: 그 ubatch 를 다 셈했고 seq_rm 으로 n_past 뒤를 지웠다). 옮김(p0 = −1)은 남는다
            self.hist = [e for e in self.hist if e[0] < n_past]
        else:
            # 25단계 — 옮김은 남는다(n_past 앞 위치의 칸도 옮겼다)
            h = [e for e in self.hist if e[0] < 0 or e[0] + e[1] <= n_past]
            ub = [e for e in h if e[0] >= 0]
            if (ub[-1][0] + ub[-1][1] if ub else 0) != n_past:
                return -1, []
            self.hist = h
        if self.칸판 is not None:                                  # 25단계 — memory_seq_rm [n_past, end)
            self.칸판.지우기(self.번호, n_past)
            self.칸 = {p: c for p, c in self.칸.items() if p < n_past}
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
                self.hist.append(self.항목(u0, T, list(prompt[u0:u0 + T])))
        self.tokens = list(prompt)
        사슬 = list(self.hist)
        for i, t in enumerate(gen[:-1]):
            self.hist.append(self.항목(n + i, 1, [t]))
            self.tokens.append(t)
        if self.까닭 is not None:
            return -1, []
        return cache_n, 사슬


def 겹침(a, b):
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k


class 서버:
    """24단계 — 쉬는 슬롯을 비우는 서버(슬롯 여럿 · 통합 KV · RAM 캐시 > 0 · --cache-idle-slots): 새 작업을 띄울 때 쉬는 슬롯을 모두 RAM
    캐시에 저장하고 비우므로, KV 에는 마지막 작업의 슬롯(활성) 하나만 남는다. RAM 캐시 항목은 슬롯의 사본(토큰 · 이력 · 체크포인트) —
    server_prompt_cache::alloc · load(server-task.cpp), get_available_slot(server-context.cpp)"""

    def __init__(self, 설정):
        self.설정 = 설정풀기(설정)[:8]
        self.sp = F32(self.설정[4])
        self.활성 = None                                             # (슬롯 번호, 슬롯)
        self.항목 = []                                               # RAM 캐시 항목(오래된 것부터)
        self.칸판 = 칸판(설정풀기(설정)[9])                          # 25단계 — 통합 KV(크기 = n_ctx)

    def 새슬롯(self, 번호):
        s = 슬롯(*self.설정)
        s.서버흉내 = True
        s.번호, s.칸판 = 번호, self.칸판
        return s

    def 저장(self, s):                                             # prompt_save → alloc
        if not s.tokens:
            return
        if any(겹침(e.tokens, s.tokens) == len(s.tokens) for e in self.항목):
            return
        self.항목 = [e for e in self.항목 if 겹침(e.tokens, s.tokens) != len(e.tokens)]
        c = 슬롯(*self.설정)                                         # 항목은 KV 밖(칸판 없음) — 저장한 때의 위치 → 칸을 지닌다
        c.서버흉내 = True
        c.tokens, c.hist, c.cps, c.칸 = list(s.tokens), list(s.hist), [list(x) for x in s.cps], dict(s.칸)
        self.항목.append(c)

    def 불러오기(self, s, prompt):                                   # prompt_load → load
        n = len(prompt)
        l0 = 겹침(s.tokens, prompt)
        fk_best = F32(l0) / F32(len(s.tokens)) if s.tokens else F32(-1.0)
        fs_best = F32(l0) / F32(n)
        best = None
        for i, e in enumerate(self.항목):
            l = 겹침(e.tokens, prompt)
            fk, fs = F32(l) / F32(len(e.tokens)), F32(l) / F32(n)
            if fk < F32(0.25):
                continue
            if fk_best < fk and fs_best < fs:
                fk_best, fs_best, best = fk, fs, i
        if best is not None:
            e = self.항목.pop(best)
            # 25단계 — state_read_meta: seq_rm(슬롯, 모두) → find_slot(cont = false — 다른 슬롯의 칸이 있으면 그 뒤로) → apply_ubatch.
            # 저장한 차례는 칸 차례(state_write 가 칸을 앞에서부터 — 위치 차례가 아닐 수 있다)
            self.칸판.지우기(s.번호, 0)
            차례 = sorted(e.칸, key=lambda p: e.칸[p])
            assert sorted(차례) == list(range(len(e.tokens))), "항목의 칸이 토큰과 맞지 않음"
            cells = self.칸판.찾기(len(차례))
            if cells is None:
                s.까닭 = "칸 모자람"
                return
            self.칸판.놓기(cells, s.번호, 차례)
            s.칸 = dict(zip(차례, cells))
            s.tokens, s.cps = list(e.tokens), [list(x) for x in e.cps]
            s.hist = list(e.hist) + [(-1, len(차례), [], [s.칸[p] for p in range(len(차례))], 0)]

    def 요청(self, prompt, gen, task, 사용자, 슬롯번호):
        n = len(prompt)
        if n < 1:
            return -1, []
        if self.활성 is not None and self.활성[0] == 슬롯번호:
            s = self.활성[1]
            l = 겹침(s.tokens, prompt)
            f_sim = F32(l) / F32(n)
            if not (s.tokens and self.sp > 0 and f_sim > self.sp):
                return -1, []                                         # 서버는 LCP 비율로 이 슬롯을 고르지 않았을 것 — 흉내와 다르다
            if (f_sim * F32(n)) / F32(len(s.tokens)) < F32(0.5):      # 남기는 몫 < 0.5 → 캐시 고침
                self.저장(s)
                self.불러오기(s, prompt)
        else:
            s = self.새슬롯(슬롯번호)                                 # 빈 슬롯(다른 슬롯은 앞 작업을 띄울 때 비워졌다)
            self.불러오기(s, prompt)                                  # 쉬던 활성 슬롯의 칸이 아직 KV 에 있다
            if self.활성 is not None:
                self.저장(self.활성[1])                               # 쉬는 슬롯을 저장하고 비운다(띄운 뒤 — prompt_clear)
                self.칸판.지우기(self.활성[0], 0)
            self.활성 = (슬롯번호, s)
        if s.까닭 is not None:
            return -1, []
        return s.요청(prompt, gen, task, 사용자)


def 칸셈(사슬, 걸음):
    """25단계 — 칸 배치의 합 셈: 사슬의 항목(ubatch · 옮김)과 이 요청의 생성 걸음마다 칸들 · n_kv —
    Σ (i + 1)·(값 + 1) mod 1,000,000,007"""
    k, v = 0, 0
    for e in list(사슬) + list(걸음):
        for c in list(e[3] or []) + [e[4]]:
            k += 1
            v = (v + k * (c + 1)) % 1000000007
    return v


def 칸같음(사슬, 걸음):
    """25단계 — 칸이 위치와 같고 n_kv 가 위치로 정해지는가(재현 호스트의 방식 4 로 셈할 수 있다)"""
    for p0, T, _, cells, nkv in list(사슬) + list(걸음):
        if cells is None:
            continue
        if p0 < 0:
            if cells != list(range(T)):
                return False
        elif cells != list(range(p0, p0 + T)) or nkv != 패드256(p0 + T):
            return False
    return True


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
    # 24단계 — 쉬는 슬롯을 비우는 서버(여러 대화가 슬롯을 나눠 씀): dense 기본 · dense 작은 덩어리 · 하이브리드 기본 · LCP 비율 끔
    (2048, 512, 32, 8192, 0.1, 8192, 4, 1, 1),
    (64, 16, 4, 0, 0.1, 8192, 4, 1, 1),
    (2048, 512, 32, 8192, 0.1, 8192, 4, 0, 1),
    (512, 128, 32, 8192, 0.0, 8192, 4, 1, 1),
    # 25단계 — KV 가 작은 서버(n_ctx 4096 · 1024 · 512 — 칸 찾기가 끝에서 앞으로 돌고, head 가 멀면 0 부터, 모자라면 셈하지 않음 —
    # 4096 아래는 짧은 대화로)
    (2048, 512, 32, 8192, 0.1, 8192, 4, 1, 1, 4096),
    (64, 16, 4, 0, 0.1, 8192, 4, 0, 1, 4096),
    (2048, 512, 32, 8192, 0.1, 8192, 4, 1, 1, 1024),
    (64, 16, 4, 0, 0.1, 8192, 4, 1, 1, 512),
    (128, 32, 32, 8192, 0.1, 8192, 4, 0, 1, 512),
]


def 서버사례(rnd, 설정):
    """24단계 — 대화 서넛이 오가는 요청들. 슬롯 번호는 서버의 고르기(활성 슬롯과의 LCP 비율 > 유사 → 그 슬롯, 아니면 다른 빈 슬롯)를
    참조 서버로 정한다"""
    sv = 서버(설정)
    ns = 설정풀기(설정)[6]
    작음 = 설정풀기(설정)[9] < 4096                                # 25단계 — KV 가 작은 서버는 짧은 대화로
    대화 = []                                                      # (프롬프트, 생성)
    reqs = []
    for k in range(rnd.randint(3, 12 if 작음 else 9)):
        새 = [rnd.randrange(1000) for _ in range(rnd.choice([1, 3, 20, 60] if 작음 else [1, 3, 20, 90]))]
        if not 대화 or rnd.random() < 0.3:
            길이 = rnd.choice([10, 40, 120, 300] if 작음 else [40, 300, 700, 1500])
            p = [rnd.randrange(1000) for _ in range(길이)]        # 새 대화(앞이 겹치지 않음)
            i = len(대화)
            대화.append(None)
        else:
            i = rnd.randrange(len(대화))
            pp, pg = 대화[i]
            갈래 = rnd.choice(["잇기", "잇기", "같음", "생성잇기", "고침"])
            if 갈래 == "잇기":
                p = pp + pg + 새
            elif 갈래 == "같음":
                p = list(pp)
            elif 갈래 == "생성잇기":
                p = pp + pg
            else:
                p = pp[:rnd.randrange(1, len(pp) + 1)] + 새
        g = [rnd.randrange(1000) for _ in range(rnd.choice([1, 2, 8, 64]))]
        if sv.활성 is not None and sv.활성[1].tokens and sv.sp > 0 and F32(겹침(sv.활성[1].tokens, p)) / F32(len(p)) > sv.sp:
            번호 = sv.활성[0]
        else:
            번호 = ((sv.활성[0] if sv.활성 is not None else -1) + 1) % ns
        c, _ = sv.요청(p, g, k + 1, [], 번호)
        reqs.append((p, g, [], 번호))
        대화[i] = (p, g)
        if c < 0:
            break
    return reqs


def 사례들(씨앗=7, 수=12):
    rnd = random.Random(씨앗)
    out = []
    for 설정 in 설정들:
        for _ in range(수):
            if 설정풀기(설정)[8]:
                out.append((설정, 서버사례(rnd, 설정)))
                continue
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
                reqs.append((p, g, 자리, 0))
                prev = (p, g)
            out.append((설정, reqs))
    return out


def 설정풀기(설정):
    # n_batch, n_ubatch, 체크수, 최소걸음, 유사, 램, 슬롯 수, 되돌림(10단계), 서버(24단계), KV 크기(25단계 — n_ctx, 기본 32768)
    s = tuple(설정) + (0,) * (9 - len(설정))
    return s if len(s) >= 10 else s + (32768,)


def 토큰셈(사슬):
    # 16단계 — 사슬 토큰의 합 셈: Σ (i + 1)·(토큰 + 1) mod 1,000,000,007(차례대로 이은 토큰 — 옮김은 토큰이 없다)
    k, v = 0, 0
    for e in 사슬:
        for t in e[2]:
            k += 1
            v = (v + k * (t + 1)) % 1000000007
    return v


def 참조출력(cases):
    out = []
    for 설정, reqs in cases:
        서버흉내 = 설정풀기(설정)[8]
        s = 슬롯(*설정풀기(설정)[:8])
        sv = 서버(설정)
        멈춤 = False
        for r, (p, g, 자리, 번호) in enumerate(reqs):
            if 멈춤:
                continue
            if 서버흉내:
                c, 사슬 = sv.요청(p, g, r + 1, 자리, 번호)
                s = sv.활성[1] if sv.활성 is not None else s
            else:
                c, 사슬 = s.요청(p, g, r + 1, 자리)
            out.append(c)
            if c < 0:
                멈춤 = True
                out += [0, 0, 0, 0, 0]
            else:
                out += [s.되살림, len(s.tokens), len(사슬)]
                for e in 사슬:
                    out += [e[0], e[1]]
                out.append(토큰셈(사슬))
                걸음 = s.hist[len(사슬):]                             # 이 요청의 생성 걸음(마지막 토큰은 셈하지 않는다)
                if 서버흉내:
                    out += [칸셈(사슬, 걸음), 0 if 칸같음(사슬, 걸음) else 1]
                else:
                    out += [0, 0]
    return out


def 칸판사례들(씨앗=11, 수=24):
    """25단계 — 칸판만의 무작위 시험(작은 KV — 끝에서 0 으로 돎 · head 가 멀어 0 부터 · 모자람): [(크기, [(종류, a, b)])].
    종류 0 = 지우기(시퀀스 a 의 위치 b 부터), 1 = 찾기(n = a) · 찾으면 놓기(시퀀스 b — 그 시퀀스의 다음 위치부터)"""
    rnd = random.Random(씨앗)
    out = []
    for _ in range(수):
        크기 = rnd.choice([16, 40, 64, 100])
        ops, 끝위치 = [], [0] * 4
        for _ in range(rnd.randint(20, 120)):
            seq = rnd.randrange(4)
            if rnd.random() < 0.35:
                p0 = rnd.randrange(끝위치[seq] + 1) if rnd.random() < 0.6 else 0
                ops.append((0, seq, p0))
                끝위치[seq] = min(끝위치[seq], p0)
            else:
                n = rnd.choice([1, 1, 2, 5, 12, 30])
                ops.append((1, n, seq))
                끝위치[seq] += n
        out.append((크기, ops))
    return out


def 칸판참조(사례):
    """연산마다 [찾기: 찾은 칸들의 합 셈(못 찾으면 −1) · 지우기: −2] [끝] [쓴 칸] [머리] [n_kv]"""
    out = []
    세기 = [0, 0, 0]                                                # 끝에서 돎 · head 0 부터 · 모자람
    for 크기, ops in 사례:
        k = 칸판(크기)
        다음 = [0] * 4
        for 종류, a, b in ops:
            if 종류 == 0:
                k.지우기(a, b)
                다음[a] = min(다음[a], b)
                out.append(-2)
            else:
                cells = k.찾기(a)
                if cells is None:
                    out.append(-1)
                    세기[2] += 1
                else:
                    k.놓기(cells, b, range(다음[b], 다음[b] + a))
                    다음[b] += a
                    v = 0
                    for i, c in enumerate(cells):
                        v = (v + (i + 1) * (c + 1)) % 1000000007
                    out.append(v)
            out += [k.끝, k.쓴수, k.머리, k.nkv()]
        세기[0] += k.돎
        세기[1] += k.처음부터
    return out, 세기


def 사례파일(cases, 경로, 칸사례=()):
    v = [len(cases)]
    for 설정, reqs in cases:
        nb, nu, cp, ms, sp, ram, ns, dg, sv, kv = 설정풀기(설정)
        v += [nb, nu, cp, ms, struct.unpack("<I", struct.pack("<f", sp))[0], ram, ns, dg, sv, kv, len(reqs)]
        for p, g, 자리, 번호 in reqs:
            v += [len(p), *p, len(g), *g, len(자리), *자리, 번호]
    v.append(len(칸사례))                                          # 25단계 — 칸판만의 시험
    for 크기, ops in 칸사례:
        v += [크기, len(ops)]
        for op in ops:
            v += list(op)
    np.array(v, np.int64).tofile(경로)


def 시험(exe):
    cases = 사례들()
    칸사례 = 칸판사례들()
    칸want, 칸세기 = 칸판참조(칸사례)
    want = 참조출력(cases) + 칸want
    with tempfile.TemporaryDirectory() as d:
        inp, outp = os.path.join(d, "사례.bin"), os.path.join(d, "출력.bin")
        사례파일(cases, inp, 칸사례)
        r = subprocess.run([exe, inp, outp], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"시험_라마사슬 실패({r.returncode}): {r.stdout}{r.stderr}")
        got = np.fromfile(outp, np.int64).tolist()
    같음 = got == want
    요청 = sum(1 for (_, reqs) in cases for _ in reqs)
    # 참조의 갈래 세기(되살림 · 셈하지 않음)
    되살림 = 셈안함 = 이음 = 서버요청 = 불러옴 = 칸다름 = 옮김수 = 0
    for 설정, reqs in cases:
        서버흉내 = 설정풀기(설정)[8]
        s = 슬롯(*설정풀기(설정)[:8])
        sv = 서버(설정)
        for r, (p, g, 자리, 번호) in enumerate(reqs):
            if 서버흉내:
                앞항목 = len(sv.항목)
                c, 사슬 = sv.요청(p, g, r + 1, 자리, 번호)
                서버요청 += 1
                if c > 0 and len(sv.항목) < 앞항목 + (1 if r > 0 else 0) and sv.활성 is not None and r > 0:
                    불러옴 += 1
                s = sv.활성[1] if sv.활성 is not None else s
                if c >= 0:                                        # 25단계 — 칸이 위치와 다른 요청 · 사슬의 옮김
                    칸다름 += not 칸같음(사슬, s.hist[len(사슬):])
                    옮김수 += sum(1 for e in 사슬 if e[0] < 0)
            else:
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
    print(f"사례 {len(cases)} · 요청 {요청} · 체크포인트 되살림 {되살림} · 되살리지 않고 이음 {이음} · 셈하지 않음 {셈안함} · 서버 흉내 요청 {서버요청}"
          f" · 칸이 위치와 다른 요청 {칸다름} · 사슬의 옮김 {옮김수}")
    print(f"칸판 시험 {len(칸사례)} 개 · 연산 {sum(len(o) for _, o in 칸사례)} · 끝에서 0 으로 돎 {칸세기[0]} · head 가 멀어 0 부터 {칸세기[1]}"
          f" · 칸 모자람 {칸세기[2]}")
    print(f"모두 같음: {같음}")
    return 0 if 같음 else 1


def 자리찾기(tokens, 구분):
    return [i for i in range(len(tokens)) if 구분 and tokens[i:i + len(구분)] == 구분]


def 서버기록칸(경로):
    """25단계 — 서버 기록(LLAMA_KV_CACHE_DEBUG=1 · -v)의 find_slot 줄마다 (used_max_p1, used, head) — 첫 작업을 띄운 뒤부터(올릴 때의
    시험 디코드는 뺀다)"""
    import re
    fs = re.compile(r"find_slot: stream\[0\], n = +(\d+), used = +(\d+), head = +(\d+)")
    out, 띄움 = [], False
    for ln in open(경로, encoding="utf-8", errors="replace"):
        if "launch_slot_" in ln:
            띄움 = True
        m = fs.search(ln)
        if m and 띄움:
            out.append(tuple(int(x) for x in m.groups()))
    return out


def 서버영수증(자료, out, 되돌림=1):
    """24단계 — 기록기의 영수증 폴더(자료\\영수증)와 out\\영수증.json 의 차례로 서버 단위 흉내의 cache_n · tokens_cached 를 서버와 견준다.
    25단계 — out\\llama-server.log 가 KV 디버그 기록이면 흉내의 칸 찾기(find_slot 마다 used_max_p1 · used · head)도 견준다"""
    표 = json.load(open(os.path.join(out, "영수증.json"), encoding="utf-8"))
    sv = None
    나쁨 = 0
    for k, (이름, 파일) in enumerate(표.items()):
        d = json.load(open(os.path.join(자료, "영수증", 파일 + ".json"), encoding="utf-8"))
        r = d["response"]
        if sv is None:
            sv = 서버((2048, 512, 32, 8192, 0.1, 8192, 4, 되돌림, 1, d["server"].get("n_ctx", 32768)))
        c, 사슬 = sv.요청(d["prompt_tokens"], r["tokens"], k + 1, [], r["id_slot"])
        서c, 서t = r["timings"]["cache_n"], r.get("tokens_cached")
        s = sv.활성[1]
        같음 = c == 서c and len(s.tokens) == 서t
        나쁨 += not 같음
        걸음 = s.hist[len(사슬):]
        옮김 = sum(1 for e in 사슬 if e[0] < 0)
        nkv = max([e[4] for e in 사슬 + 걸음 if e[0] >= 0] or [0])
        print(f"{이름}: 슬롯 {r['id_slot']} · cache_n 참조 {c} · 서버 {서c} · tokens_cached 참조 {len(s.tokens)} · 서버 {서t} · RAM 항목 {len(sv.항목)}"
              f" · 사슬 항목 {len(사슬)}(옮김 {옮김}) · 칸 {'위치와 같음' if 칸같음(사슬, 걸음) else '위치와 다름'} · n_kv 최대 {nkv}")
    로그 = os.path.join(out, "llama-server.log")
    if sv is not None and os.path.exists(로그):
        서 = 서버기록칸(로그)
        흉 = sv.칸판.자취
        if 서:
            k = next((i for i, (a, b) in enumerate(zip(흉, 서)) if a != b), min(len(흉), len(서)))
            칸맞음 = len(흉) == len(서) and k == len(서)
            print(f"칸 찾기 {len(서)} 번(서버 기록) · 흉내 {len(흉)} 번 · 모두 같음 {칸맞음}"
                  + ("" if 칸맞음 else f" — 처음 다른 자리 {k}: 흉내 {흉[k:k + 3]} · 서버 {서[k:k + 3]}"))
            나쁨 += not 칸맞음
    print(f"모두 같음: {나쁨 == 0}")
    return 0 if 나쁨 == 0 else 1


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
    if len(sys.argv) in (4, 5) and sys.argv[1] == "--서버영수증":        # 다섯째 인자 '하이브리드' — qwen35(체크포인트)
        sys.exit(서버영수증(sys.argv[2], sys.argv[3], 0 if sys.argv[4:] == ["하이브리드"] else 1))
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(시험(sys.argv[1]))
