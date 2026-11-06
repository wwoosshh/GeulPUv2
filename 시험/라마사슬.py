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

    def __init__(self, 크기, 흐름=0, 자취=None):
        self.크기 = 크기
        self.seq = [-1] * 크기
        self.pos = [0] * 크기
        self.머리 = 0
        self.쓴수 = 0
        self.끝 = 0                                                   # used_max_p1
        self.흐름 = 흐름                                              # 27단계 — 흐름 번호(슬롯마다 KV 면 슬롯 번호)
        self.자취 = [] if 자취 is None else 자취                     # find_slot 마다 (흐름, used_max_p1, used, head) — 서버의 KV 디버그 기록과 견줌
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
        self.자취.append((self.흐름, self.끝, self.쓴수, self.머리))
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
    """24 · 26단계 — 슬롯이 여럿인 서버(통합 KV · 흐름 하나 — server-context.cpp 의 get_available_slot · get_free_slots · 작업 띄우기,
    server-task.cpp 의 server_prompt_cache::alloc · load). 슬롯마다 상태(토큰 · 이력 · 체크포인트 · 칸)와 마지막으로 쓴 때(작업이
    끝날 때 — t_last_used). 슬롯 고르기: LCP 비율이 기준보다 크고 가장 큰 슬롯(남기는 몫 < 0.5 면 캐시 고침), 없으면 빈 슬롯 먼저 ·
    그다음 마지막으로 쓴 때가 이른 슬롯(같으면 번호 차례 — 캐시 고침). 캐시 고침(RAM 캐시가 있을 때): 그 슬롯을 저장하고 더 나은 항목을
    불러온다. 쉬는 슬롯을 비우는 서버(--cache-idle-slots — 기본 켬, RAM 캐시가 있어야 함): 작업을 띄운 뒤 다른 슬롯을 번호 차례로
    저장하고 비운다. 비우지 않는 서버(26단계 — --no-cache-idle-slots · RAM 캐시 0)는 쉬는 슬롯의 칸이 KV 에 남는다"""

    def __init__(self, 설정):
        f = 설정풀기(설정)
        self.설정 = f[:8]
        self.sp = F32(f[4])
        self.램 = f[5]
        self.슬롯수 = f[6]
        self.통합 = f[11] != 0                                       # 27단계 — 통합 KV(아니면 슬롯마다 흐름 — -np N)
        쉬는저장 = f[10] != 0 and self.램 != 0                        # --cache-idle-slots(기본 켬 — RAM 캐시가 있어야 함)
        self.쉬는 = (2 if self.통합 else 1) if 쉬는저장 else 0       # 26 · 27단계 — 쉬는 슬롯: 0 그대로 · 1 저장만 · 2 저장하고 비움
        self.비움 = self.쉬는 == 2
        self.슬롯들 = {}                                             # 번호 → 슬롯(토큰이 있는 것)
        self.마지막 = {}                                             # 번호 → 마지막으로 쓴 차례(없으면 −1)
        self.차례 = 0
        self.활성 = None                                             # (슬롯 번호, 슬롯) — 마지막 요청의 슬롯
        self.항목 = []                                               # RAM 캐시 항목(오래된 것부터)
        self.자취 = []
        # 25단계 — 통합 KV(크기 = n_ctx), 27단계 — 슬롯마다 KV 면 슬롯마다 흐름(크기 = n_ctx_seq — 영수증의 n_ctx)
        self.칸판들 = [칸판(f[9], 0, self.자취)] if self.통합 else [칸판(f[9], i, self.자취) for i in range(self.슬롯수)]
        self.까닭 = None

    def 새슬롯(self, 번호):
        s = 슬롯(*self.설정)
        s.서버흉내 = True
        s.번호, s.칸판 = 번호, self.칸판들[0 if self.통합 else 번호]
        return s

    def 고르기(self, prompt):
        """get_available_slot 이 고를 슬롯 번호와 캐시 고침(남기는 몫 < 0.5 · LRU)"""
        n = len(prompt)
        if self.sp != 0:
            best, fbest = None, F32(0)
            for 번호 in range(self.슬롯수):
                s = self.슬롯들.get(번호)
                if s is None or not s.tokens:
                    continue
                f = F32(겹침(s.tokens, prompt)) / F32(n)
                if f > fbest and f > self.sp:
                    fbest, best = f, 번호
            if best is not None:
                return best, (fbest * F32(n)) / F32(len(self.슬롯들[best].tokens)) < F32(0.5)
        차례 = sorted(range(self.슬롯수), key=lambda i: (bool(self.슬롯들.get(i) and self.슬롯들[i].tokens), self.마지막.get(i, -1)))
        return 차례[0], True

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
            s.칸판.지우기(s.번호, 0)
            차례 = sorted(e.칸, key=lambda p: e.칸[p])
            assert sorted(차례) == list(range(len(e.tokens))), "항목의 칸이 토큰과 맞지 않음"
            cells = s.칸판.찾기(len(차례))
            if cells is None:
                s.까닭 = "칸 모자람"
                return
            s.칸판.놓기(cells, s.번호, 차례)
            s.칸 = dict(zip(차례, cells))
            s.tokens, s.cps = list(e.tokens), [list(x) for x in e.cps]
            s.hist = list(e.hist) + [(-1, len(차례), [], [s.칸[p] for p in range(len(차례))], 0)]

    def 요청(self, prompt, gen, task, 사용자, 슬롯번호):
        self.까닭 = None
        n = len(prompt)
        if n < 1:
            return -1, []
        번호, 고침 = self.고르기(prompt)
        if 슬롯번호 >= 0 and 번호 != 슬롯번호:                        # 서버는 다른 슬롯을 골랐을 것 — 기록되지 않은 요청 등(30단계 — −1 은 모름)
            self.까닭 = f"슬롯 {번호} 을 골랐을 것(영수증은 {슬롯번호})"
            return -1, []
        s = self.슬롯들.get(번호)
        if s is None:
            s = self.새슬롯(번호)
            self.슬롯들[번호] = s
        if 고침 and self.램 != 0:                                     # 캐시 고침 — 저장 → 불러오기(다른 슬롯의 칸이 KV 에 있을 수 있다)
            self.저장(s)
            self.불러오기(s, prompt)
        if self.쉬는:                                                 # 띄운 뒤 쉬는 슬롯을 저장(번호 차례) — 통합 KV 면 비운다(prompt_clear)
            for i in range(self.슬롯수):
                o = self.슬롯들.get(i)
                if i != 번호 and o is not None and o.tokens:
                    self.저장(o)
                    if self.쉬는 == 2:
                        o.칸판.지우기(i, 0)
                        del self.슬롯들[i]
        self.활성 = (번호, s)
        if s.까닭 is not None:
            self.까닭 = s.까닭
            return -1, []
        c, 사슬 = s.요청(prompt, gen, task, 사용자)
        self.마지막[번호] = self.차례                                 # release — t_last_used
        self.차례 += 1
        return c, 사슬


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
    # 26단계 — 쉬는 슬롯을 비우지 않는 서버(--no-cache-idle-slots · RAM 캐시 0): dense · 하이브리드 · 작은 덩어리 · LCP 비율 끔 · 작은 KV
    (2048, 512, 32, 8192, 0.1, 8192, 4, 1, 1, 32768, 0),
    (2048, 512, 32, 8192, 0.1, 0, 4, 1, 1, 32768, 0),
    (2048, 512, 32, 8192, 0.1, 8192, 4, 0, 1, 32768, 0),
    (2048, 512, 32, 8192, 0.1, 0, 4, 0, 1, 32768, 0),
    (64, 16, 4, 0, 0.1, 8192, 4, 1, 1, 4096, 0),
    (512, 128, 32, 8192, 0.0, 0, 4, 1, 1, 32768, 0),
    (128, 32, 32, 8192, 0.1, 8192, 4, 0, 1, 1024, 0),
    # 27단계 — 슬롯마다 KV 를 둔 서버(-np N — 흐름마다 n_ctx / N): 쉬는 슬롯을 저장만(기본) · dense · 하이브리드 · RAM 캐시 0 ·
    # --no-cache-idle-slots · 작은 흐름
    (2048, 512, 32, 8192, 0.1, 8192, 4, 1, 1, 8192, 1, 0),
    (2048, 512, 32, 8192, 0.1, 8192, 4, 0, 1, 8192, 1, 0),
    (2048, 512, 32, 8192, 0.1, 0, 4, 1, 1, 8192, 1, 0),
    (2048, 512, 32, 8192, 0.1, 8192, 4, 1, 1, 8192, 0, 0),
    (64, 16, 4, 0, 0.1, 8192, 4, 0, 1, 1024, 1, 0),
    (512, 128, 32, 8192, 0.0, 8192, 2, 1, 1, 2048, 1, 0),
]


def 서버사례(rnd, 설정):
    """24단계 — 대화 서넛이 오가는 요청들. 26단계 — 쉬는 슬롯을 비우지 않는 서버는 대화를 더(슬롯보다 많이 — 칸이 있는 슬롯을 LRU 로
    다시 씀), 새 대화의 절반은 같은 머리(채팅 틀의 시스템 머리처럼 — 다른 대화의 슬롯과 앞이 조금 겹친다). 슬롯 번호는 참조 서버의
    고르기로 정한다"""
    sv = 서버(설정)
    f = 설정풀기(설정)
    작음 = f[9] < 4096                                             # 25단계 — KV 가 작은 서버는 짧은 대화로
    남김 = f[10] == 0 or f[5] == 0 or f[11] == 0                   # 26 · 27단계 — 쉬는 슬롯의 칸이 남는 서버
    머리 = [999 - j for j in range(30)]
    대화 = []                                                      # (프롬프트, 생성)
    reqs = []
    for k in range(rnd.randint(3, 12 if (작음 or 남김) else 9)):
        새 = [rnd.randrange(1000) for _ in range(rnd.choice([1, 3, 20, 60] if 작음 else [1, 3, 20, 90]))]
        if not 대화 or rnd.random() < (0.45 if 남김 else 0.3):
            길이 = rnd.choice([10, 40, 120, 300] if 작음 else [40, 300, 700, 1500])
            p = [rnd.randrange(1000) for _ in range(길이)]        # 새 대화(앞이 겹치지 않음)
            if 남김 and rnd.random() < 0.5:
                p = 머리 + p
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
        번호, _ = sv.고르기(p)
        c, _ = sv.요청(p, g, k + 1, [], 번호)
        reqs.append((p, g, [], 번호))
        대화[i] = (p, g)
        if c < 0:
            break
    return reqs


def 사례들(씨앗=7, 수=12):
    rnd = random.Random(씨앗)
    out = []
    서버수 = 0
    for 설정 in 설정들:
        for _ in range(수):
            if 설정풀기(설정)[8]:
                reqs = 서버사례(rnd, 설정)
                out.append((설정, reqs))
                서버수 += 1
                if 서버수 % 4 == 0:                                   # 30단계 — 슬롯을 모르는 영수증(채팅 응답 — 슬롯 번호 −1)
                    out.append((설정, [(p, g, 자리, -1) for p, g, 자리, _ in reqs]))
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
    # n_batch, n_ubatch, 체크수, 최소걸음, 유사, 램, 슬롯 수, 되돌림(10단계), 서버(24단계), KV 크기(25단계 — n_ctx, 기본 32768 —
    # 슬롯마다 KV 면 흐름 하나의 크기), 쉬는 슬롯 저장(26단계 — --cache-idle-slots, 기본 1 · RAM 캐시가 0 이면 하지 않는다),
    # 통합 KV(27단계 — 기본 1, 0 이면 슬롯마다 흐름)
    s = tuple(설정) + (0,) * (9 - len(설정))
    s = s if len(s) >= 10 else s + (32768,)
    s = s if len(s) >= 11 else s + (1,)
    return s if len(s) >= 12 else s + (1,)


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


def 기록묶음(b, 슬롯, 작업, 프, 생, 빠짐, 칸수):
    """29단계 — 앱/라마묶음.gl(라마기록묶음)의 참조: llama-server 기록(바이트 — LLAMA_BATCH_DEBUG=1 · -v)에서 슬롯 · 작업(영수증)의 방식 6
    입력. 돌려줌: [까닭, 까1, 까2, 까3, KV 칸, 항목 수, 섞인 수, 출력 수, ubatch 수, 견준 수, 값 수, 값 …] — 시험_라마사슬 --기록 의 출력"""
    n = len(b)
    r = {"까닭": 0, "까": [0, 0, 0], "항목": 0, "섞인": 0, "ub": 0, "견준": 0}
    st = {"기대줄수": -1, "표적시작": 0, "첫줄봄": 0, "표적출력": 0, "최대칸": -1, "최대nkv": 256, "최대위치": -1, "기대읽음": 0}
    판 = 칸판(칸수)
    ut, up, us, uo = [], [], [], []
    기대, 표토, 머리l, 값l = [], [-1] * 칸수, [], []

    def 까닭(k, a=0, bb=0, c=0):
        if r["까닭"] == 0:
            r["까닭"], r["까"] = k, [a, bb, c]

    def 찾(a, e, 바늘):
        return b.find(바늘, a, e) if e - a >= len(바늘) else -1

    def 수(i, e):
        k = i
        while k < e and b[k] == 32:
            k += 1
        부호 = 1
        if k < e and b[k] == 45:
            부호, k = -1, k + 1
        k0, v = k, 0
        while k < e and 48 <= b[k] <= 57:
            v, k = v * 10 + b[k] - 48, k + 1
        return (0, -1) if k == k0 else (부호 * v, k)

    def 뒤수(a, e, 바늘):
        p = 찾(a, e, 바늘)
        return (0, -1) if p < 0 else 수(p + len(바늘), e)

    def 마감():
        m = len(ut)
        if m > 0:
            r["ub"] += 1
            if st["기대줄수"] >= 0 and st["기대줄수"] != m:
                까닭(14, m, st["기대줄수"])
            본, u, 출력, 표적수 = set(), 0, 0, 0
            for t in range(m):
                sq = us[t]
                if sq < 0 or sq >= 64:
                    까닭(14, t, sq)
                elif sq not in 본:
                    본.add(sq)
                    u += 1
                출력 += uo[t] != 0
                표적수 += sq == 슬롯
            for _ in range(u):
                기대.extend((판.끝, 판.쓴수, 판.머리))
            cells = 판.찾기(m)
            판.자취.clear()
            if cells is None:
                까닭(9, m)
            else:
                for t in range(m):
                    c = cells[t]
                    판.seq[c], 판.pos[c] = us[t], up[t]
                    판.쓴수 += 1
                    판.끝 = max(판.끝, c + 1)
                판.머리 = cells[m - 1] + 1
                nkv = 판.nkv()
                if 표적수 > 0:
                    if 출력 > 8:
                        까닭(6, 출력)
                    r["섞인"] += u > 1
                    r["항목"] += 1
                    머리l.extend((0, m, nkv))
                    st["최대nkv"] = max(st["최대nkv"], nkv)
                    np_ = len(프)
                    for t in range(m):
                        tok, pos, c3, 표시 = ut[t], up[t], cells[t], 0
                        st["최대칸"] = max(st["최대칸"], c3)
                        if us[t] == 슬롯:
                            표시 = 1
                            if pos < 0 or pos >= 칸수:
                                까닭(15, pos)
                            else:
                                st["최대위치"] = max(st["최대위치"], pos)
                                if st["표적시작"]:
                                    if not st["첫줄봄"]:
                                        st["첫줄봄"] = 1
                                        for q in range(pos):
                                            want = 프[q] if q < np_ else -1
                                            if 표토[q] != want:
                                                까닭(11, q, 표토[q], want)
                                    want2 = -2
                                    if pos < np_:
                                        want2 = 프[pos]
                                    elif 빠짐 == 0 and pos - np_ < len(생):
                                        want2 = 생[pos - np_]
                                    if want2 != -2 and want2 != tok:
                                        까닭(11, pos, tok, want2)
                                    if uo[t]:
                                        표시 = 3
                                        st["표적출력"] += 1
                                표토[pos] = tok
                        값l.extend((tok, pos, c3, 표시))
            ut.clear(), up.clear(), us.clear(), uo.clear()
        st["기대줄수"] = -1

    def 지움(sq, p):
        마감()
        p0 = max(p, 0)
        판.지우기(sq, p0)
        if sq == 슬롯:
            r["항목"] += 1
            머리l.extend((2, p0, 0))
            for q in range(p0, min(st["최대위치"] + 1, 칸수)):
                표토[q] = -1

    시작 = 끝남 = False
    모드, i = 0, 0
    while i < n and not 끝남 and r["까닭"] == 0:
        e = b.find(b"\n", i)
        e = n if e < 0 else e
        if not 시작 and 찾(i, e, b"launch_slot_") >= 0:
            시작 = True
        if 시작:
            up_ = 찾(i, e, b"ubatch_print:")
            if up_ >= 0:
                if 모드 == 2:
                    ip = 찾(up_, e, b": id =")
                    if ip >= 0:
                        한계 = min(ip + 65536, n)
                        행, 행끝 = 수(up_ + 13, ip)
                        tok, x = 수(ip + 6, e)
                        pp = 찾(x, 한계, b"), pos =") if x >= 0 else -1
                        if 행끝 < 0 or x < 0 or pp < 0:
                            까닭(14, len(ut), -1)
                        else:
                            pos, x = 수(pp + 8, 한계)
                            nsq = 0
                            if x >= 0:
                                nsq, x = 뒤수(x, 한계, b"n_seq_id =")
                            sp = 찾(x, 한계, b"seq_id = [") if x >= 0 else -1
                            sq = -1
                            if sp >= 0:
                                k2 = sp + 10
                                while k2 < 한계 and b[k2] == 46:
                                    k2 += 1
                                if k2 < n and 48 <= b[k2] <= 57:
                                    sq = k2 - (sp + 10)
                                x = k2
                            o = 0
                            if sq >= 0:
                                o, x = 뒤수(x, 한계, b"output =")
                            if sq < 0 or x < 0:
                                까닭(14, len(ut), -2)
                            elif nsq != 1:
                                까닭(5, nsq)
                            elif 행 != len(ut):
                                까닭(14, len(ut), 행)
                            else:
                                ut.append(tok), up.append(pos), us.append(sq), uo.append(o)
                            if x >= 0:
                                e = b.find(b"\n", x)
                                e = n if e < 0 else e
                    else:
                        k3 = up_ + 13
                        while k3 < e and b[k3] == 32:
                            k3 += 1
                        if k3 < e and b[k3] == 93:
                            마감()
                            모드 = 0
                        elif 찾(up_, e, b"n_tokens") >= 0:
                            v, x = 뒤수(up_, e, b"=")
                            st["기대줄수"] = v if x >= 0 else -1
            elif 찾(i, e, b"added ubatch to split") >= 0:
                마감()
                모드 = 2
            elif 찾(i, e, b"input batch info") >= 0:
                마감()
                모드 = 1
            elif 찾(i, e, b"find_slot: stream[") >= 0:
                fn, x = 뒤수(i, e, b", n =")
                if x >= 0:
                    마감()
                    st_, _ = 뒤수(i, e, b"stream[")
                    fu, _ = 뒤수(i, e, b"used =")
                    fh, _ = 뒤수(i, e, b"head =")
                    r["견준"] += 1
                    if st_ != 0:
                        까닭(7, st_)
                    elif st["기대읽음"] + 3 > len(기대):
                        까닭(8, r["견준"], fn, -1)
                    else:
                        en, eu, eh = 기대[st["기대읽음"]:st["기대읽음"] + 3]
                        st["기대읽음"] += 3
                        if (en, eu, eh) != (fn, fu, fh):
                            까닭(8, r["견준"], fn, en)
            elif 찾(i, e, b"memory_seq_rm [") >= 0:
                sl, x = 뒤수(i, e, b": id ")
                p = 0
                if x >= 0:
                    p, x = 뒤수(i, e, b"memory_seq_rm [")
                if x >= 0:
                    지움(sl, p)
            elif 찾(i, e, b"clearing prompt with") >= 0:
                sl2, x = 뒤수(i, e, b": id ")
                if x >= 0:
                    지움(sl2, 0)
            elif 찾(i, e, b"processing task") >= 0:
                sl3, x = 뒤수(i, e, b": id ")
                tk = 0
                if x >= 0:
                    tk, x = 뒤수(i, e, b"| task ")
                if x >= 0:
                    마감()
                    if sl3 == 슬롯:
                        if tk == 작업:
                            st["표적시작"] = 1
                        elif st["표적시작"]:
                            끝남 = True
            elif 찾(i, e, b"stop processing") >= 0:
                sl4, x = 뒤수(i, e, b": id ")
                tk4 = 0
                if x >= 0:
                    tk4, x = 뒤수(i, e, b"| task ")
                if x >= 0:
                    마감()
                    if sl4 == 슬롯 and tk4 == 작업:
                        끝남 = True
            elif 찾(i, e, b"found better prompt") >= 0:
                까닭(3)
            elif 찾(i, e, b"context shift") >= 0:
                까닭(4)
            elif 찾(i, e, b"failed to decode") >= 0 or 찾(i, e, b"failed to find a memory slot") >= 0:
                까닭(13)
        i = e + 1
    마감()
    if r["까닭"] == 0:
        if not 시작:
            까닭(1)
        elif r["ub"] == 0:
            까닭(2)
        elif not st["표적시작"]:
            까닭(10)
        elif st["표적출력"] != len(생) + 빠짐:
            까닭(12, st["표적출력"], len(생) + 빠짐)
        elif r["견준"] > 0 and st["기대읽음"] < len(기대):
            까닭(8, r["견준"] + 1, -1, 기대[st["기대읽음"]])
    kv = max(st["최대nkv"], (st["최대칸"] + 256) // 256 * 256, (st["최대위치"] + 256) // 256 * 256)
    값 = [r["항목"]] + 머리l + 값l
    return [r["까닭"], *r["까"], kv, r["항목"], r["섞인"], st["표적출력"], r["ub"], r["견준"], len(값), *값]


def 기록짓기(rnd, 슬롯수, 칸수, n_batch, n_ubatch, 요청들, kv디버그=True):
    """29단계 — llama-server(b12008)의 연속 배치를 흉내 낸 기록(LLAMA_BATCH_DEBUG=1 · -v 의 줄 꼴): 요청들 = [(도착 차례, 슬롯, 앞 대화를
    잇나, 프롬프트 더함, 생성 수)]. 차례마다 생성 중인 슬롯의 토큰(슬롯 차례) 뒤에 프롬프트를 n_batch 까지, ubatch 는 n_ubatch 로 나눔
    (split_simple), 칸은 칸판. 돌려줌: (기록 바이트, 영수증들 [(슬롯, 작업, 프롬프트, 생성)])"""
    줄 = []
    t0 = [0]

    def 쓰기(글, 수준="D"):
        t0[0] += rnd.randint(1, 40)
        ms = t0[0]
        줄.append(f"{ms // 60000}.{ms // 1000 % 60:02d}.{ms % 1000:03d}.{rnd.randint(0, 999):03d} {수준} {글}")

    def 잡음():
        if rnd.random() < 0.3:
            쓰기(rnd.choice(["srv  log_server_r: request: POST /completion 127.0.0.1 200", "que    process_new_: processing task, id = 17",
                           "srv  update_slots: posting NEXT_RESPONSE", "que  yield_to_que: yielding to queue",
                           "srv          load:  - looking for better prompt, base f_keep = -1.000, f_sim = 0.000"]))

    def 조각(tok):
        if tok % 97 == 5:
            return "\n\n"                                          # 줄바꿈이 든 토큰 글
        if tok % 89 == 3:
            return "), x ("
        return f"t{tok}"

    def 줄들(rows, 머리):
        쓰기(머리)
        for k, v in (("equal_seqs", 0), ("n_tokens", len(rows)), ("n_seq_tokens", 1), ("n_seqs", len(rows))):
            쓰기(f"ubatch_print:   {k:<12} = {v}")
        쓰기("ubatch_print:   seq_id_unq = [ " + " ".join(str(s) for s in sorted({x[2] for x in rows})) + " ]")
        쓰기("ubatch_print:   token     = [")
        for k, (tok, pos, sq, o) in enumerate(rows):
            표 = "".join(str(s % 10) if s == sq else "." for s in range(max(2, 슬롯수)))
            쓰기(f"ubatch_print:   {k:4d}: id = {tok:6d} ({조각(tok):>16}), pos = {pos:4d}, n_seq_id = {1:2d}, seq_id = [{표}], output = {o}")
            if rnd.random() < 0.02:
                잡음()
        쓰기("ubatch_print:   ]")

    # 올릴 때의 시험 디코드(처음 작업 앞 — 읽는 쪽이 건너뛴다)
    줄들([(1, 0, 0, 0), (2, 1, 0, 1)], "init: input batch info:")
    줄들([(1, 0, 0, 0), (2, 1, 0, 1)], "ubatch_add: added ubatch to split:")
    쓰기("find_slot: stream[0], n =     0, used =     0, head =     0, size = 99, n_swa =     0")
    쓰기("srv          init: idle slots will be saved to prompt cache and cleared upon starting a new task", "I")
    판 = 칸판(칸수)
    슬 = [{"토큰": [], "작업": -1, "때": "쉼", "프": [], "n_past": 0, "남음": 0, "생": [], "마지막": 0} for _ in range(슬롯수)]
    대기 = sorted(요청들, key=lambda q: q[0])
    영수증들 = []
    작업수 = 0
    차례 = 0
    while 대기 or any(s["때"] != "쉼" for s in 슬):
        # 도착 — 슬롯이 쉬면 띄운다(아니면 다음 차례로)
        남김 = []
        for q in 대기:
            도착, 번호, 잇기, 더함, 생성수 = q
            s = 슬[번호]
            if 도착 > 차례 or s["때"] != "쉼":
                남김.append(q)
                continue
            쓰기(f"slot launch_slot_: id  {번호} | task -1 | launching slot : {{\"id\":{번호}}}")
            작업 = 작업수
            작업수 += 1
            쓰기(f"slot launch_slot_: id  {번호} | task {작업} | processing task, is_child = 0", "I")
            잡음()
            앞 = s["토큰"] if 잇기 else []
            프 = list(앞) + [rnd.randrange(1, 1000) for _ in range(더함)]
            n_past = 0
            while n_past < min(len(s["토큰"]), len(프) - 1) and s["토큰"][n_past] == 프[n_past]:
                n_past += 1
            for j, o in enumerate(슬):                              # 쉬는 슬롯을 비운다(--cache-idle-slots)
                if j != 번호 and o["때"] == "쉼" and (o["토큰"] or rnd.random() < 0.3) and rnd.random() < 0.5:
                    쓰기(f"slot prompt_clear: id  {j} | task -1 | clearing prompt with {len(o['토큰'])} tokens", "I")
                    판.지우기(j, 0)
                    o["토큰"] = []
            쓰기(f"slot   operator(): id  {번호} | task {작업} | cached n_tokens = {n_past}, memory_seq_rm [{n_past}, end)", "I")
            판.지우기(번호, n_past)
            s.update({"토큰": s["토큰"][:n_past], "작업": 작업, "때": "프", "프": 프, "n_past": n_past, "남음": 생성수, "생": []})
        대기 = 남김
        rows = []
        for j, s in enumerate(슬):
            if s["때"] == "생":
                rows.append((s["마지막"], len(s["토큰"]), j, 1))
                s["토큰"].append(s["마지막"])
        for j, s in enumerate(슬):
            if s["때"] == "프":
                while s["n_past"] < len(s["프"]) and len(rows) < n_batch:
                    끝 = s["n_past"] == len(s["프"]) - 1
                    rows.append((s["프"][s["n_past"]], s["n_past"], j, 1 if 끝 else 0))
                    s["토큰"].append(s["프"][s["n_past"]])
                    s["n_past"] += 1
        if not rows:
            차례 += 1
            continue
        줄들(rows, "init: input batch info:")
        ubs = [rows[k:k + n_ubatch] for k in range(0, len(rows), n_ubatch)]
        for u in ubs:
            줄들(u, "ubatch_add: added ubatch to split:")
        for u in ubs:
            for _ in sorted({x[2] for x in u}):
                if kv디버그:
                    쓰기(f"find_slot: stream[0], n = {판.끝:5d}, used = {판.쓴수:5d}, head = {판.머리:5d}, size = {칸수:5d}, n_swa =     0")
                    for sq in range(슬롯수):
                        ps = [판.pos[c] for c in range(판.끝) if 판.seq[c] == sq]
                        if ps:
                            쓰기(f"find_slot: stream[0] min[{sq}] = {min(ps):5d}, max[{sq}] = {max(ps):5d}")
            cells = 판.찾기(len(u))
            판.자취.clear()
            if cells is None:
                raise RuntimeError("시험 기록의 KV 가 모자란다 — 칸 수를 늘린다")
            for (tok, pos, sq, o), c in zip(u, cells):
                판.seq[c], 판.pos[c] = sq, pos
                판.쓴수 += 1
                판.끝 = max(판.끝, c + 1)
            판.머리 = cells[-1] + 1
        잡음()
        for j, s in enumerate(슬):                                  # 고르기 — 출력이 있는 슬롯
            if any(x[2] == j and x[3] for x in rows):
                tok = rnd.randrange(1, 1000)
                s["생"].append(tok)
                s["남음"] -= 1
                if s["남음"] <= 0:
                    쓰기(f"slot      release: id  {j} | task {s['작업']} | stop processing: n_tokens = {len(s['토큰'])}, truncated = 0", "I")
                    영수증들.append((j, s["작업"], s["프"], s["생"]))
                    s["때"] = "쉼"
                else:
                    s["때"], s["마지막"] = "생", tok
        차례 += 1
    return ("\n".join(줄) + "\n").encode("utf-8"), 영수증들


def 기록사례들(씨앗=29, 수=20):
    """29단계 — 서버 기록 사례 [(기록, 슬롯, 작업, 칸 수, 프롬프트, 생성, 빠짐, 기대 까닭)]: 겹친 요청들의 기록마다 영수증 하나 + 꾸민 사례
    (RAM 캐시 불러옴 · find_slot 다름 · 작업 없음 · 토큰 다름 · 생성 수 다름 · 시퀀스 둘 · 출력 9 · KV 디버그 없음 · 스트리밍의 빠짐 · 앞뒤가 잘림)"""
    rnd = random.Random(씨앗)
    out = []
    for ci in range(수):
        슬롯수 = rnd.choice([2, 3, 4])
        칸수 = rnd.choice([512, 640, 1024])
        n_ubatch = rnd.choice([4, 8, 16, 64]) if ci % 10 != 9 else rnd.choice([16, 64])
        n_batch = rnd.choice([n_ubatch, 2 * n_ubatch, 64, 128])
        n_batch = max(n_batch, n_ubatch)
        요청들 = []
        for _ in range(rnd.randint(3, 7)):
            요청들.append((rnd.randint(0, 12), rnd.randrange(슬롯수), rnd.random() < 0.5, rnd.randint(1, 50), rnd.randint(1, 12)))
        kvd = ci % 5 != 3
        b, 영들 = 기록짓기(rnd, 슬롯수, 칸수, n_batch, n_ubatch, 요청들, kvd)
        sl, tk, p, g = rnd.choice(영들)
        out.append((b, sl, tk, 칸수, p, g, 0, 0))
        꾸밈 = ci % 10
        if 꾸밈 == 0:                                                 # RAM 캐시에서 불러옴 — 이 작업이 끝나기 앞
            k = b.find(f"| task {tk} | processing task".encode())
            out.append((b[:k] + b"srv          load:  - found better prompt with f_keep = 0.500, f_sim = 0.900\n" + b[k:], sl, tk, 칸수, p, g, 0, 3))
        elif 꾸밈 == 1 and kvd:                                      # find_slot 하나가 다르다
            k = b.find(b"find_slot: stream[0], n =", b.find(b"launch_slot_"))
            k = b.find(b"used =", k)
            out.append((b[:k] + b"used = 99999" + b[k + 12:], sl, tk, 칸수, p, g, 0, 8))
        elif 꾸밈 == 2:
            out.append((b, sl, tk + 1000, 칸수, p, g, 0, 10))
        elif 꾸밈 == 4:
            q = list(p)
            q[-1] = (q[-1] + 1) % 1000
            out.append((b, sl, tk, 칸수, q, g, 0, 11))
        elif 꾸밈 == 5:
            out.append((b, sl, tk, 칸수, p, g + [7], 0, 12))
        elif 꾸밈 == 6:                                              # 시퀀스 둘인 줄
            k = b.find(b"n_seq_id =  1", b.find(b"added ubatch to split", b.find(b"launch_slot_")))
            out.append((b[:k] + b"n_seq_id =  2" + b[k + 13:], sl, tk, 칸수, p, g, 0, 5))
        elif 꾸밈 == 7 and len(g) > 2:                               # 스트리밍 응답에서 빠진 생성 토큰 둘
            out.append((b, sl, tk, 칸수, p, g[:-2], 2, 0))
        elif 꾸밈 == 8:                                              # 이 작업이 끝나기 앞에서 잘린 기록
            k = b.find(f"release: id  {sl} | task {tk} |".encode())
            k = b.rfind(b"init: input batch info", 0, k)
            out.append((b[:k], sl, tk, 칸수, p, g, 0, 12))
        elif 꾸밈 == 9:                                              # 출력 9 줄인 ubatch(이 슬롯의 줄이 든)
            k = b.find(f"| task {tk} | processing task".encode())
            k = b.find(b"added ubatch to split", k)
            e = b.find(b"ubatch_print:   ]", k)
            blk = b[k:e].replace(b"output = 0", b"output = 1")
            if blk.count(b"output = 1") > 8 and f"seq_id = [{'.' * sl}{sl % 10}".encode() in blk:
                out.append((b[:k] + blk + b[e:], sl, tk, 칸수, p, g, 0, 6))
    return out


def 기록참조(사례):
    want = []
    for b, sl, tk, kn, p, g, 빠짐, _ in 사례:
        want += 기록묶음(b, sl, tk, p, g, 빠짐, kn)
    return want


def 기록사례파일(사례, 경로):
    with open(경로, "wb") as f:
        f.write(struct.pack("<q", len(사례)))
        for b, sl, tk, kn, p, g, 빠짐, _ in 사례:
            f.write(struct.pack("<q", len(b)) + b + b"\0" * ((-len(b)) % 8))
            f.write(struct.pack(f"<{4 + len(p)}q", sl, tk, kn, len(p), *p))
            f.write(struct.pack(f"<{2 + len(g)}q", len(g), *g, 빠짐))


def 사례파일(cases, 경로, 칸사례=()):
    v = [len(cases)]
    for 설정, reqs in cases:
        nb, nu, cp, ms, sp, ram, ns, dg, sv, kv, 비움, 통합 = 설정풀기(설정)
        v += [nb, nu, cp, ms, struct.unpack("<I", struct.pack("<f", sp))[0], ram, ns, dg, sv, kv, 비움, 통합, len(reqs)]
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
        # 29단계 — 서버 기록의 방식 6 입력(앱/라마묶음.gl)
        기록사례 = 기록사례들()
        기록want = 기록참조(기록사례)
        inp2, outp2 = os.path.join(d, "기록.bin"), os.path.join(d, "기록출력.bin")
        기록사례파일(기록사례, inp2)
        r2 = subprocess.run([exe, "--기록", inp2, outp2], capture_output=True, text=True)
        if r2.returncode != 0:
            raise SystemExit(f"시험_라마사슬 --기록 실패({r2.returncode}): {r2.stdout}{r2.stderr}")
        기록got = np.fromfile(outp2, np.int64).tolist()
    같음 = got == want
    기록같음 = 기록got == 기록want
    # 참조의 까닭이 꾸민 대로인가(참조 자체의 시험), 섞인 ubatch 수
    까닭들, 꾸밈맞음, 섞인합, o = {}, True, 0, 0
    for 사 in 기록사례:
        v = 기록want[o:o + 11]
        o += 11 + v[10]
        까닭들[v[0]] = 까닭들.get(v[0], 0) + 1
        꾸밈맞음 = 꾸밈맞음 and v[0] == 사[7]
        섞인합 += v[6]
    if not 기록같음:
        k = next((i for i, (a, b) in enumerate(zip(기록got, 기록want)) if a != b), min(len(기록got), len(기록want)))
        print(f"서버 기록: 처음 다른 자리 {k}: 글 {기록got[k:k + 8]} · 참조 {기록want[k:k + 8]} (길이 {len(기록got)} · {len(기록want)})")
    기록글 = f"서버 기록 사례 {len(기록사례)} · 까닭 {dict(sorted(까닭들.items()))} · 꾸민 대로 {꾸밈맞음} · 섞인 ubatch {섞인합}"
    같음 = 같음 and 기록같음 and 꾸밈맞음
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
    print(기록글)                                                 # 29단계 — 빠른 점검의 요약(끝 세 줄)에 보이도록 끝에서 둘째
    print(f"모두 같음: {같음}")
    return 0 if 같음 else 1


def 자리찾기(tokens, 구분):
    return [i for i in range(len(tokens)) if 구분 and tokens[i:i + len(구분)] == 구분]


def 서버기록칸(경로):
    """25단계 — 서버 기록(LLAMA_KV_CACHE_DEBUG=1 · -v)의 find_slot 줄마다 (used_max_p1, used, head) — 첫 작업을 띄운 뒤부터(올릴 때의
    시험 디코드는 뺀다)"""
    import re
    fs = re.compile(r"find_slot: stream\[(\d+)\], n = +(\d+), used = +(\d+), head = +(\d+)")
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
            sd = d["server"]
            비움 = 0 if "--no-cache-idle-slots" in sd.get("args", "") else 1
            sv = 서버((sd["n_batch"], sd["n_ubatch"], sd["ctx_checkpoints"], sd["checkpoint_min_step"], sd["slot_prompt_similarity"],
                      sd["cache_ram_mib"], sd["n_parallel"], 되돌림, 1, sd.get("n_ctx", 32768), 비움, 1 if sd.get("kv_unified") else 0))
        구분 = (d.get("chat") or {}).get("user_start_tokens") or []      # 25단계 — 채팅 API 의 사용자 메시지 자리(하이브리드의 덩어리 끊기)
        c, 사슬 = sv.요청(d["prompt_tokens"], r["tokens"], k + 1, 자리찾기(d["prompt_tokens"], 구분), r["id_slot"])
        서c, 서t = r["timings"]["cache_n"], r.get("tokens_cached")
        if c < 0 and sv.까닭:
            print(f"{이름}: 흉내가 멈춤 — {sv.까닭}")
            나쁨 += 1
            break
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
        흉 = sv.자취
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
