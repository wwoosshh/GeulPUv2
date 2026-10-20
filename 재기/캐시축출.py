#!/usr/bin/env python3
"""프롬프트 캐시 축출 정책 시뮬레이터(문서/캐시축출.md — 이슈 #1). GPU · 모델 없이, 표준 라이브러리만.

엔진(엔진/저장소.gl · 엔진/서버몸.gl)의 프롬프트 캐시를 요청마다 그대로 따라간다: 슬롯 고르기 → 덮일 청크 쓰기 → 다른 슬롯 복사 →
조회 · 쓴 때 바꾸기 · 복원 → 계산 → 쉴 때 쓰기(슬롯 0, 1 차례로, 쓰기 고리만큼 묶어) → 한도 지키기. 정책은 '어느 청크를 지우나'와
그 정책이 청크마다 두는 값만 다르다. 시간은 이미 잰 값으로 만든 비용 모델이다(새로 재지 않는다).

    python -X utf8 재기/캐시축출.py                   작업 부하 R · L7 · L8 · M · D × 정책 × 용량(시드 1 ~ 5) — 마크다운 표
    python -X utf8 재기/캐시축출.py --확인             V1(시험/저장소CPU.py 의 색인과 지우는 차례) · V2(W0 — 문맥저장소.md 4단계 ②)
    python -X utf8 재기/캐시축출.py --json 결과.json --시드 5 --일꾼 8 --시나리오 R,L7

사후 탐색(결과를 본 뒤 — 문서 6절, 3절의 판정에 쓰지 않음): 쓰기 방식 비울때 · 비울때 + 슬롯에도 있는 청크를 먼저 지움, 슬롯과 겹친 몫.
"""
import argparse
import bisect
import heapq
import importlib.util
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

덩이 = 256
때폭 = 1 << 20
머리말 = 4096
GB = 10 ** 9
SSD = 1.75e9                 # 차가운 SSD 에서 복원(문맥저장소.md 4단계 — 1.8 GB/s)
OS캐시 = 2.7e9               # OS 캐시에서 복원(같은 곳 — 2.7 GB/s). W0 확인만
고리바이트 = 268435456       # 엔진의 쓰기 고리 = 256 MB / 청크 바이트, 2 ~ 16 개(저장소.gl 저장소일최대)
없음 = 1 << 60

# 모델 — 프롬프트 누적 시간 T(L) = a·L + b·L² ms(빠른 모드 · RTX 4070 Ti · Q4_K_M — 문서/긴문맥재기.md 의 16k · 32k 와 깊이 + 1,024 에
# 맞춤), 토큰당 KV 바이트, 생성 스텝 c0 + c1·깊이 ms(같은 문서의 행 1 스텝) × 추측배(추측 디코딩 — W0 의 생성 157.3 초에 맞춤, --확인)
모델들 = {
    "7B": dict(a=0.1225, b=6.3e-6, kv=57344, c0=10.20, c1=1.465e-4),     # Qwen2.5-7B-Instruct
    "8B": dict(a=0.1339, b=8.2e-6, kv=131072, c0=11.35, c1=3.121e-4),    # Llama-3.1-8B-Instruct
}
추측배 = 0.62

# 녹화한 Qwen Code 작업 셋(E:\geulpu-work\moe\sess — Qwen3.6 토큰): 요청마다 (프롬프트, 앞 요청과 같은 앞부분 cache_n, 답).
# t1 의 20번째는 cache_n 이 0 이지만 글은 시스템 프롬프트가 같고 도구 목록부터 다르다(t2 · t3 의 11번째와 같은 꼴) — 5,383 로 둔다.
틀 = {
    "t1": [(15112, 0, 108), (15266, 14596, 1645), (17023, 15133, 2407), (19474, 16890, 108), (19682, 19341, 172),
           (20356, 19549, 169), (20680, 20223, 133), (20957, 20547, 229), (21814, 20824, 124), (22050, 21681, 243),
           (22450, 21917, 125), (22677, 22317, 157), (22995, 22544, 146), (23643, 22862, 142), (23929, 23510, 229),
           (24785, 23796, 714), (25809, 24652, 104), (27440, 25676, 491), (28107, 27307, 31), (18615, 5383, 570),
           (19387, 19184, 290), (19877, 19676, 993), (20985, 20869, 202), (21246, 21186, 249)],
    "t2": [(15276, 0, 381), (17300, 14760, 475), (17996, 17167, 535), (18853, 17863, 78), (19021, 18720, 1114),
           (20941, 18888, 110), (21160, 20808, 105), (21575, 21027, 110), (21819, 21442, 109), (22057, 21686, 294),
           (12993, 5383, 366), (13535, 13358, 228), (13797, 13762, 293), (14199, 14089, 601), (14862, 14799, 171)],
    "t3": [(15205, 0, 113), (16100, 14689, 1145), (17695, 15967, 105), (17912, 17562, 101), (18690, 17779, 315),
           (19161, 18557, 308), (19693, 19028, 150), (19996, 19560, 773), (21169, 19863, 96), (21519, 21036, 373),
           (12131, 5383, 399), (12706, 12529, 125), (12865, 12830, 533), (13465, 13397, 135), (13659, 13599, 130)],
}
배율 = 0.97                  # Qwen3.6 → Qwen2.5 토큰(t1 첫 요청 15,112 → 14,671)
모든폴더 = 3819              # Qwen Code 의 캐시가 끊긴 위치(문맥저장소.md 4단계): 다른 폴더는 여기까지 같다
같은폴더 = 13591             # 같은 폴더의 다음 날(날짜 줄)

정책들 = ("지금", "LRU", "LFU", "SLRU", "GDS", "Marconi", "Belady")
민감정책 = ("Marconi0.5", "Marconi2")
사후정책 = ("비울때", "슬롯먼저")      # 결과를 본 뒤의 탐색 — 쓰기 방식 비울때(엔진 그대로) · 비울때 + 슬롯에도 있는 청크를 먼저 지움
용량들 = (5, 10, 20, 40)
시나리오들 = ("R", "L7", "L8", "M", "D")


def T(m, L):
    d = 모델들[m]
    return d["a"] * L + d["b"] * L * L


def 청크바이트(m):
    return 머리말 + 덩이 * 모델들[m]["kv"]


def 생성ms(m, 깊이, n):
    d = 모델들[m]
    return 추측배 * n * (d["c0"] + d["c1"] * (깊이 + n / 2))


# ───── 토큰열: 조각 (번호, 시작, 끝)의 목록 — 같은 번호 · 같은 위치면 같은 토큰 ─────

def 길이(열):
    return sum(e - s for _, s, e in 열)


def 앞자르기(열, n):
    out = []
    for g, s, e in 열:
        if n <= 0:
            break
        if e - s <= n:
            out.append((g, s, e))
            n -= e - s
        else:
            out.append((g, s, s + n))
            n = 0
    return out


def 같은앞(x, y):
    """앞이 같은 토큰 수(엔진의 앞같은길이)"""
    if not x or not y:
        return 0
    i = j = n = 0
    gx, px, ex = x[0]
    gy, py, ey = y[0]
    while gx == gy and px == py:
        t = min(ex - px, ey - py)
        n += t
        px += t
        py += t
        if px == ex:
            i += 1
            if i == len(x):
                return n
            gx, px, ex = x[i]
        if py == ey:
            j += 1
            if j == len(y):
                return n
            gy, py, ey = y[j]
    return n


class 이름표:
    """청크 이름 — (앞 청크, 이 청크에 든 조각들)마다 번호 하나. 앞부분이 같으면 같은 번호(엔진의 해시 사슬과 같다)"""

    def __init__(self):
        self.열쇠 = {}
        self.부모 = []
        self.깊이 = []
        self.모델 = []
        self.뿌리 = {}

    def 이름들(self, 모델, 열, n):
        out = []
        if n <= 0:
            return out
        if 모델 not in self.뿌리:
            self.뿌리[모델] = -2 - len(self.뿌리)
        앞 = self.뿌리[모델]
        조각들 = []
        채운 = 0
        for g, s, e in 열:
            while s < e:
                t = min(e - s, 덩이 - 채운)
                if 조각들 and 조각들[-1][0] == g and 조각들[-1][2] == s:
                    조각들[-1] = (g, 조각들[-1][1], s + t)
                else:
                    조각들.append((g, s, s + t))
                s += t
                채운 += t
                if 채운 == 덩이:
                    열쇠 = (앞, tuple(조각들))
                    c = self.열쇠.get(열쇠)
                    if c is None:
                        c = len(self.부모)
                        self.열쇠[열쇠] = c
                        self.부모.append(앞 if 앞 >= 0 else -1)
                        self.깊이.append(len(out))
                        self.모델.append(모델)
                    out.append(c)
                    if len(out) == n:
                        return out
                    앞 = c
                    조각들 = []
                    채운 = 0
        return out


# ───── 작업 부하: 사건 = None(다시 켜기 · 모델 바꾸기 — 슬롯이 빈다) 또는 (모델, 프롬프트 열, 답 길이, 답 조각 번호) ─────

class 부하:
    def __init__(self):
        self.n = 0
        self.일 = []

    def 새(self):
        self.n += 1
        return self.n


def 작업(w, 이름, 시스템, 폴더조각, 날조각, 모델="7B", 남의답=False):
    """녹화한 틀 하나를 새 작업으로 — 요청들. 앞부분은 [모든 폴더][같은 폴더][같은 날](둘째 요청이 첫 요청과 같은 데까지) 뒤가 작업마다
    다르다. 길이는 배율로 줄이되 앞 요청과의 관계는 토큰 수 그대로: 앞 프롬프트 끝보다 g 앞에서 갈라지면 줄인 끝보다 g 앞, 앞 답까지
    이어지면(덧붙임) 줄인 끝 + 같은 답 토큰. 도구 목록이 바뀐 요청(5,383)만 자리를 줄인다. 남의답 = 다음 프롬프트의 답이 서버가 만든 답과
    다름(4단계 재생 — 녹화한 답을 보냄): 같은 앞부분은 앞 프롬프트 끝까지"""
    줄 = 틀[이름]
    P0 = round(줄[0][0] * 배율)
    C = P0 - (줄[0][0] - 줄[1][1])
    열 = [(시스템, 0, 모든폴더), (폴더조각, 0, 같은폴더 - 모든폴더), (날조각, 0, C - 같은폴더), (w.새(), 0, P0 - C)]
    out = []
    A, g = 줄[0][2], w.새()
    out.append((모델, 열, A, g))
    전체, 앞P, 앞Pr = 열 + [(g, 0, A)], P0, 줄[0][0]
    for Pr, Lr, A in 줄[1:]:
        P = round(Pr * 배율)
        if Lr == 5383:
            L = round(Lr * 배율)
        elif Lr <= 앞Pr:
            L = 앞P - (앞Pr - Lr)
        else:
            L = 앞P + (Lr - 앞Pr)
        if 남의답:
            L = min(L, 앞P)
        열 = 앞자르기(전체, L) + [(w.새(), 0, P - L)]
        g = w.새()
        out.append((모델, 열, A, g))
        전체, 앞P, 앞Pr = 열 + [(g, 0, A)], P, Pr
    return out


def 번갈아(줄기들):
    """에이전트 여럿 — 요청을 하나씩 번갈아"""
    out = []
    i = 0
    while True:
        더 = False
        for 줄기 in 줄기들:
            if i < len(줄기):
                out.append(줄기[i])
                더 = True
        if not 더:
            return out
        i += 1


def 부하W0():
    """녹화한 54 요청(t1 → t2 → t3, 폴더가 다름) — 문맥저장소.md 4단계 ②"""
    w = 부하()
    시스템 = w.새()
    for 이름 in ("t1", "t2", "t3"):
        w.일.extend(작업(w, 이름, 시스템, w.새(), w.새(), 남의답=True))
    return w.일


def 부하R(rng, 사이=None, 날수=5, 차례수=3):
    """코딩 에이전트의 한 주 — 날마다 다시 켜기, 하루 3 차례, 차례마다 새 작업 1 ~ 3 개를 번갈아. 30 %는 30 ~ 70 %에서 끊겨 다음 날 잇는다"""
    w = 부하()
    시스템 = w.새()
    폴더 = [w.새() for _ in range(3)]
    이어갈 = [[] for _ in range(날수 + 1)]
    for d in range(날수):
        w.일.append(None)
        날 = [w.새() for _ in range(3)]
        차례들 = []
        for _ in range(차례수):
            줄기들 = []
            for _ in range(rng.randint(1, 3)):
                f = rng.randrange(3)
                요청들 = 작업(w, rng.choice(("t1", "t2", "t3")), 시스템, 폴더[f], 날[f])
                if rng.random() < 0.3:
                    끊음 = min(len(요청들) - 1, max(1, round(len(요청들) * rng.uniform(0.3, 0.7))))
                    줄기들.append(요청들[:끊음])
                    이어갈[d + 1].append(요청들[끊음:])          # 마지막 날에 끊긴 것은 잇지 않는다
                else:
                    줄기들.append(요청들)
            차례들.append(줄기들)
        for 줄기 in 이어갈[d]:
            차례들[rng.randrange(차례수)].append(줄기)
        for i, 줄기들 in enumerate(차례들):
            w.일.extend(번갈아(줄기들))
            if 사이 is not None and i < 차례수 - 1:
                사이(w, rng)
    return w.일


def 부하L(rng, 모델, 날수=3, 방문수=10, 턴수=4):
    """긴 대화 100k+ — 큰 프로젝트 4 개([시스템][문맥 60k ~ 90k] 뒤에 턴이 쌓임), 날마다 다시 켜기, 하루 10 번 방문 · 방문마다 4 턴"""
    w = 부하()
    시스템 = w.새()
    대화 = [[(시스템, 0, 모든폴더), (w.새(), 0, rng.randint(60000, 90000))] for _ in range(4)]
    for _ in range(날수):
        w.일.append(None)
        for _ in range(방문수):
            j = rng.randrange(4)
            for _ in range(턴수):
                열 = 대화[j] + [(w.새(), 0, rng.randint(300, 1500))]
                A, g = rng.randint(100, 600), w.새()
                w.일.append((모델, 열, A, g))
                대화[j] = 열 + [(g, 0, A)]
    return w.일


def 부하M(rng):
    """두 모델 번갈아 — R 의 7B 작업 사이(하루 두 번)에 8B 대화 6 턴(새 대화 [시스템 500][문서 15k ~ 30k] 또는 앞서의 대화 잇기, 반반)"""
    대화들 = []
    시스템 = []

    def 사이(w, rng):
        if not 시스템:
            시스템.append(w.새())
        w.일.append(None)                                          # 8B 로 — 슬롯이 빈다
        if 대화들 and rng.random() < 0.5:
            대화 = rng.choice(대화들)
        else:
            대화 = [[(시스템[0], 0, 500), (w.새(), 0, rng.randint(15000, 30000))]]
            대화들.append(대화)
        for _ in range(6):
            열 = 대화[0] + [(w.새(), 0, rng.randint(100, 600))]
            A, g = rng.randint(200, 800), w.새()
            w.일.append(("8B", 열, A, g))
            대화[0] = 열 + [(g, 0, A)]
        w.일.append(None)                                          # 7B 로

    return 부하R(rng, 사이)


def 부하D(rng):
    """한 번 쓰고 마는 큰 문서 — R 의 차례 사이(하루 두 번)에 [시스템 300][새 문서 40k ~ 80k][질문 50] 한 요청(답 500)"""
    시스템 = []

    def 사이(w, rng):
        if not 시스템:
            시스템.append(w.새())
        열 = [(시스템[0], 0, 300), (w.새(), 0, rng.randint(40000, 80000)), (w.새(), 0, 50)]
        w.일.append(("7B", 열, 500, w.새()))

    return 부하R(rng, 사이)


def 만들기(시나리오, 시드):
    rng = random.Random(f"{시나리오}-{시드}")
    if 시나리오 == "R":
        return 부하R(rng)
    if 시나리오 == "L7":
        return 부하L(rng, "7B")
    if 시나리오 == "L8":
        return 부하L(rng, "8B")
    if 시나리오 == "M":
        return 부하M(rng)
    if 시나리오 == "D":
        return 부하D(rng)
    raise ValueError(시나리오)


# ───── 슬롯 쪽 — 정책과 관계없다(슬롯 내용 = 프롬프트 + 답). 요청마다 고른 슬롯 · 슬롯이 채운 길이 · 청크 이름 ─────

def 미리(일, 표):
    슬롯 = [[], []]
    슬롯길이 = [0, 0]
    모델 = None
    out = []
    for ev in 일:
        if ev is not None and 모델 is not None and ev[0] != 모델:
            out.append(None)                                      # 모델이 바뀌면 슬롯이 빈다
            슬롯, 슬롯길이 = [[], []], [0, 0]
        if ev is None:
            out.append(None)
            슬롯, 슬롯길이 = [[], []], [0, 0]
            continue
        m, 열, A, g = ev
        모델 = m
        P = 길이(열)
        한도 = P - 1                                              # 마지막 프롬프트 토큰은 늘 계산
        ps = [같은앞(슬롯[s], 열) for s in range(2)]
        고른, 고른길이, 고른잃음 = -1, -1, 0
        for s in range(2):                                        # 덮일 꽉 찬 청크가 적은 슬롯, 같으면 앞이 더 길게 같은 슬롯
            잃음 = (슬롯길이[s] - ps[s]) // 덩이
            if 고른 < 0 or 잃음 < 고른잃음 or (잃음 == 고른잃음 and ps[s] > 고른길이):
                고른, 고른길이, 고른잃음 = s, ps[s], 잃음
        s = 고른
        p = min(고른길이, 한도)
        q = min(ps[1 - s], 한도)                                  # 다른 슬롯(GPU 안 복사)
        a = max(p, q)
        이름토큰 = min(P, ((한도 + 덩이 - 1) // 덩이) * 덩이)
        m2 = 이름토큰 // 덩이
        내용 = 열 + [(g, 0, A - 1)] if A > 1 else 열
        n = P + A - 1
        out.append(dict(m=m, s=s, keep=p // 덩이, p=p, a=a, m2=m2, pids=표.이름들(m, 열, m2), P=P, A=A,
                        ids=표.이름들(m, 내용, n // 덩이)))
        슬롯[s], 슬롯길이[s] = 내용, n
    return out


def 필요표(미리들):
    """Belady 용 — 청크마다 슬롯 뒤로 그 청크가 필요한 요청 번호들"""
    필요 = {}
    for r, ev in enumerate(미리들):
        if ev is None:
            continue
        for c in ev["pids"][ev["a"] // 덩이:ev["m2"]]:
            필요.setdefault(c, []).append(r)
    return 필요


# ───── 한 실행 — 정책 하나 · 한도 하나 ─────

class 실행:
    def __init__(self, 표, 정책, 한도, 속도=SSD, 배=1.0, 필요=None, 켬=True):
        self.표 = 표
        self.쓰기방식 = 1                                         # 1 모두(기본) · 2 비울때(슬롯을 덮기 전 · 끄기 전에만 씀)
        if 정책 in 사후정책:                                      # 사후 탐색(3절의 판정에 쓰지 않음)
            self.쓰기방식 = 2
            정책 = "지금" if 정책 == "비울때" else 정책
        self.정책 = 정책
        self.한도 = 한도                                          # 바이트, None 이면 없음
        self.배 = 배
        self.필요 = 필요 or {}
        self.켬 = 켬
        self.α = 1.0
        if 정책.startswith("Marconi") and 정책 != "Marconi":
            self.α = float(정책[len("Marconi"):])
        self.복원ms = {m: 청크바이트(m) / 속도 * 1000.0 for m in 모델들}
        self.쓰기ms = {m: 청크바이트(m) / SSD * 1000.0 for m in 모델들}
        self.슬롯자리 = [{}, {}]                                  # 슬롯 내용의 청크 → 번호(슬롯에 있는 청크)
        self.쓰기대기 = 0.0
        self.고리 = {m: max(2, min(16, 고리바이트 // 청크바이트(m))) for m in 모델들}
        self.가치값 = {}
        self.바이트 = {}                                          # 있는 청크 → 바이트
        self.합 = 0
        self.자식 = {}                                            # 청크 → 있는 자식 수(그 청크가 없어도 센다)
        self.리프 = set()
        # 지금(엔진) — 쓴 때 · 색인 자리
        self.때 = {}
        self.순번 = {}
        self.시계 = 0
        self.힙 = []
        # 리프 정책
        self.틱 = 0
        self.마지막 = {}
        self.횟수 = {}
        self.구역 = {}
        self.보호합 = 0
        self.보호힙 = []
        self.H = {}
        self.L = 0.0
        self.문턱 = 0
        self.저장 = [0, 0]                                        # 슬롯저장 — 앞에서부터 프롬프트 캐시에 있다고 아는 청크 수
        self.슬롯ids = [[], []]
        self.ms = 0.0
        self.복원 = 0
        self.필요수 = 0
        self.쓴 = 0
        self.쓴바이트 = 0
        self.지운 = 0
        self.이슬롯 = 0
        self.다른슬롯 = 0
        self.저장소토큰 = 0
        self.생성 = 0.0
        self.요청수 = 0
        self.프롬프트 = 0
        self.중복합 = 0                                          # 사후 진단 — 요청이 올 때 프롬프트 캐시의 바이트 중 슬롯에도 있는 것
        self.저장합 = 0

    # 청크마다 아끼는 시간 / 바이트(ms/B) — 다시 계산 − 복원
    def 가치(self, c):
        v = self.가치값.get(c)
        if v is None:
            m, k = self.표.모델[c], self.표.깊이[c]
            아낌 = self.배 * (T(m, 덩이 * (k + 1)) - T(m, 덩이 * k)) - self.복원ms[m]
            v = self.가치값[c] = 아낌 / 청크바이트(m)
        return v

    def 넣기(self, c, k):
        m = self.표.모델[c]
        b = 청크바이트(m)
        self.바이트[c] = b
        self.합 += b
        부 = self.표.부모[c]
        if 부 >= 0:
            self.자식[부] = self.자식.get(부, 0) + 1
            self.리프.discard(부)
        if self.자식.get(c, 0) == 0:
            self.리프.add(c)
        if self.정책 == "지금":
            if k > 0 and 부 in self.바이트:                        # 저장소새때
                t = self.때[부] - 1
            else:
                self.시계 += 1
                t = self.시계 * 때폭 - k
            self.때[c] = t
            if c not in self.순번:
                self.순번[c] = len(self.순번)
            heapq.heappush(self.힙, (t, self.순번[c], c))
        else:
            self.마지막[c] = self.틱
            self.횟수[c] = 1
            self.구역[c] = 0
            self.H[c] = self.L + self.가치(c)
        self.쓴 += 1
        self.쓴바이트 += b

    def 빼기(self, c):
        b = self.바이트.pop(c)
        self.합 -= b
        부 = self.표.부모[c]
        if 부 >= 0:
            self.자식[부] -= 1
            if self.자식[부] == 0 and 부 in self.바이트:
                self.리프.add(부)
        self.리프.discard(c)
        if self.구역.get(c) == 1:
            self.구역[c] = 0
            self.보호합 -= b
        if self.정책 == "슬롯먼저":                               # 슬롯에 있는 채 지웠다 — 슬롯을 덮기 전에 다시 쓰게
            for s in (0, 1):
                k = self.슬롯자리[s].get(c)
                if k is not None and k < self.저장[s]:
                    self.저장[s] = k
        self.지운 += 1

    def 쓴때(self, pids, k):
        """조회 — 청크 0 … k−1 중 있는 것을 지금 쓴 것으로(저장소앞쪽쓴때)"""
        if k <= 0:
            return
        if self.정책 == "지금":
            self.시계 += 1
            지금 = self.시계 * 때폭
            for i in range(k):
                c = pids[i]
                if c in self.바이트:
                    self.때[c] = 지금 - i
                    heapq.heappush(self.힙, (지금 - i, self.순번[c], c))
            return
        self.틱 += 1
        for i in range(k):
            c = pids[i]
            if c in self.바이트:
                self.마지막[c] = self.틱
                self.횟수[c] += 1
                self.H[c] = self.L + self.가치(c)
                if self.정책 == "SLRU":
                    if self.구역[c] == 0:
                        self.구역[c] = 1
                        self.보호합 += self.바이트[c]
                    heapq.heappush(self.보호힙, (self.틱, -self.표.깊이[c], c))
        if self.정책 == "SLRU" and self.한도 is not None:
            while self.보호합 > 0.8 * self.한도 and self.보호힙:      # 보호 구역이 넘치면 그 LRU 를 시험 구역으로
                t, _, c = heapq.heappop(self.보호힙)
                if c in self.바이트 and self.구역.get(c) == 1 and self.마지막[c] == t:
                    self.구역[c] = 0
                    self.보호합 -= self.바이트[c]

    def 값(self, c, 잣대):
        깊 = -self.표.깊이[c]
        p = self.정책
        if p == "LRU":
            return (self.마지막[c], 깊)
        if p == "LFU":
            return (self.횟수[c], self.마지막[c], 깊)
        if p == "SLRU":
            return (self.구역[c], self.마지막[c], 깊)
        if p == "GDS":
            return (self.H[c], 깊)
        if p.startswith("Marconi"):
            r0, rs, e0, es = 잣대
            return ((self.마지막[c] - r0) / rs + self.α * (self.가치(c) - e0) / es, 깊)
        if p == "Belady":
            목록 = self.필요.get(c, ())
            i = bisect.bisect_left(목록, self.문턱)
            다음 = 목록[i] if i < len(목록) else 없음
            return (-다음, 깊)
        if p == "슬롯먼저":
            return (0 if c in self.슬롯자리[0] or c in self.슬롯자리[1] else 1, self.마지막[c], 깊)
        raise ValueError(p)

    def 지키기(self):
        """한도 지키기 — 지금은 쓴 때가 이른 청크부터(엔진), 나머지는 정책의 값이 가장 작은 리프부터"""
        if self.한도 is None or self.합 <= self.한도:
            return
        if self.정책 == "지금":
            while self.합 > self.한도 and self.힙:
                t, _, c = heapq.heappop(self.힙)
                if c in self.바이트 and self.때[c] == t:
                    self.빼기(c)
            return
        잣대 = None
        if self.정책.startswith("Marconi"):
            rs = [self.마지막[c] for c in self.리프]
            es = [self.가치(c) for c in self.리프]
            잣대 = (min(rs), (max(rs) - min(rs)) or 1, min(es), (max(es) - min(es)) or 1e-12)
        힙 = [(self.값(c, 잣대), c) for c in self.리프]
        heapq.heapify(힙)
        while self.합 > self.한도 and 힙:
            _, c = heapq.heappop(힙)
            if c not in self.바이트:
                continue
            if self.정책 == "GDS":
                self.L = self.H[c]
            부 = self.표.부모[c]
            self.빼기(c)
            if 부 >= 0 and 부 in self.리프:
                heapq.heappush(힙, (self.값(부, 잣대), 부))

    def 쓰기(self, ids, 시작):
        """슬롯 내용의 청크 [시작, 끝) 중 없는 것을 쓴다 — 고리만큼 담고, 거둘 때마다 한도를 지킨다(저장소쉬는때 · 저장소일거두기)"""
        k, n = 시작, len(ids)
        쓴 = 0
        while k < n:
            묶음 = []
            고리 = self.고리[self.표.모델[ids[k]]]
            while k < n and len(묶음) < 고리:
                if ids[k] not in self.바이트:
                    묶음.append(k)
                k += 1
            if 묶음:
                self.틱 += 1
                for kk in 묶음:
                    self.넣기(ids[kk], kk)
                쓴 += len(묶음)
                self.지키기()
        return 쓴

    def 다시켜기(self):
        if self.켬 and self.쓰기방식 == 2:                       # 서버를 멈추기 전의 비우기(앱이 부른다)
            self.슬롯자리 = [{}, {}]
            for s in (0, 1):
                self.쓰기(self.슬롯ids[s], self.저장[s])
        self.저장 = [0, 0]
        self.슬롯ids = [[], []]
        self.슬롯자리 = [{}, {}]
        if self.정책 == "지금" and self.바이트:                    # 색인.txt 를 다시 읽음 — 차례 그대로, 때 = (차례 + 1) × 때폭
            차례 = sorted(self.바이트, key=lambda c: (self.때[c], self.순번[c]))
            self.순번 = {c: i for i, c in enumerate(차례)}          # 색인을 새로 지음 — 지운 청크의 자리는 없어진다
            self.힙 = []
            for i, c in enumerate(차례):
                self.때[c] = (i + 1) * 때폭
                self.힙.append(((i + 1) * 때폭, i, c))
            self.시계 = len(차례) + 1

    def 요청(self, r, ev):
        m, s, a, P = ev["m"], ev["s"], ev["a"], ev["P"]
        self.요청수 += 1
        self.프롬프트 += P
        self.이슬롯 += ev["p"]
        self.다른슬롯 += a - ev["p"]
        복원 = 0
        if self.켬:
            슬롯에 = set(self.슬롯ids[0]) | set(self.슬롯ids[1])
            self.중복합 += sum(self.바이트[c] for c in 슬롯에 if c in self.바이트)
            self.저장합 += self.합
            self.문턱 = r
            옛 = self.슬롯ids[s]
            self.슬롯자리[s] = {c: k for k, c in enumerate(옛[:ev["keep"]])}
            n = self.쓰기(옛, max(self.저장[s], ev["keep"]))       # 덮일 청크 중 아직 쓰지 않은 것 — 고리가 차면 빌 때까지 기다린다
            if 옛 and n > self.고리[m]:
                대기 = (n - self.고리[m]) * self.쓰기ms[m]
                self.쓰기대기 += 대기
                self.ms += 대기
            self.저장[s] = min(self.저장[s], ev["keep"])
            pids, m2 = ev["pids"], ev["m2"]
            시작 = a // 덩이
            k0 = 0
            while k0 < m2 and pids[k0] in self.바이트:
                k0 += 1
            k = 시작
            while k < m2 and pids[k] in self.바이트:
                k += 1
            self.쓴때(pids, k)
            if k > 시작:
                복원 = k - 시작
                b = min(k * 덩이, P - 1)
                if b > a:
                    self.저장소토큰 += b - a
                    a = b
            앞쪽 = k if k0 >= 시작 else k0
            self.저장[s] = max(self.저장[s], min(앞쪽, a // 덩이))
            self.필요수 += max(0, m2 - 시작)
            self.복원 += 복원
        self.ms += 복원 * self.복원ms[m] + self.배 * (T(m, P) - T(m, a))
        self.생성 += 생성ms(m, P, ev["A"])
        self.슬롯ids[s] = ev["ids"]
        self.슬롯자리[s] = {c: k for k, c in enumerate(ev["ids"])}
        if self.켬 and self.쓰기방식 == 1:
            self.문턱 = r + 1
            for s2 in (0, 1):                                     # 쉴 때 쓰기 — 슬롯 0, 1 차례
                ids = self.슬롯ids[s2]
                if self.저장[s2] < len(ids):
                    self.쓰기(ids, self.저장[s2])
                    self.저장[s2] = len(ids)

    def 돌리기(self, 미리들):
        for r, ev in enumerate(미리들):
            if ev is None:
                self.다시켜기()
            else:
                self.요청(r, ev)
        return self

    def 결과(self):
        return dict(ttft=self.ms / 1000.0, 복원=self.복원, 필요=self.필요수, 적중률=self.복원 / self.필요수 if self.필요수 else 0.0,
                    쓴=self.쓴, 쓴GB=self.쓴바이트 / GB, 지운=self.지운, 생성=self.생성 / 1000.0, 요청=self.요청수,
                    프롬프트=self.프롬프트, 이슬롯=self.이슬롯, 다른슬롯=self.다른슬롯, 저장소토큰=self.저장소토큰,
                    슬롯중복=self.중복합 / self.저장합 if self.저장합 else 0.0, 쓰기대기=self.쓰기대기 / 1000.0)


# ───── 돌리기 ─────

def 한경우(인자):
    """(시나리오, 시드) 하나 — 부하를 한 번 만들고 모든 정책 · 용량을 돌린다"""
    시나리오, 시드 = 인자
    표 = 이름표()
    미리들 = 미리(만들기(시나리오, 시드), 표)
    필요 = 필요표(미리들)
    out = {}
    out["끔"] = 실행(표, "지금", None, 켬=False).돌리기(미리들).결과()
    out["한도 없음"] = 실행(표, "지금", None).돌리기(미리들).결과()
    for 용량 in 용량들:
        for 정책 in 정책들 + 민감정책 + 사후정책:
            out[f"{정책}|{용량}"] = 실행(표, 정책, 용량 * GB, 필요=필요).돌리기(미리들).결과()
    if 시나리오 == "R":                                            # 정밀 모드(프롬프트 × 1.3) — 민감도
        out["정밀|끔"] = 실행(표, "지금", None, 배=1.3, 켬=False).돌리기(미리들).결과()
        for 정책 in 정책들:
            out[f"정밀|{정책}|20"] = 실행(표, 정책, 20 * GB, 배=1.3, 필요=필요).돌리기(미리들).결과()
    return 시나리오, 시드, out


def 가운데(xs):
    return statistics.median(xs)


def 표들(모두, 시드들):
    """마크다운 표 — Σ TTFT(초, 시드 가운데값)와 지금 대비 줄어든 몫(%, 시드마다 셈의 가운데값 · 가장 작은 ~ 가장 큰)"""
    줄 = []
    판정 = []
    있는 = [sc for sc in 시나리오들 if (sc, 시드들[0]) in 모두]
    for 시나리오 in 있는:
        r = {s: 모두[(시나리오, s)] for s in 시드들}
        한 = r[시드들[0]]["끔"]
        줄.append(f"\n### {시나리오}\n")
        줄.append(f"요청 {가운데([r[s]['끔']['요청'] for s in 시드들]):.0f} · 프롬프트 {가운데([r[s]['끔']['프롬프트'] for s in 시드들]) / 1e6:.2f} M 토큰 · "
                 f"쓴 청크(한도 없음) {가운데([r[s]['한도 없음']['쓴GB'] for s in 시드들]):.0f} GB · "
                 f"생성 어림 {가운데([r[s]['끔']['생성'] for s in 시드들]):.0f} 초\n")
        줄.append("| 정책 | " + " | ".join(f"{c} GB" for c in 용량들) + " |")
        줄.append("|---|" + "---|" * len(용량들))
        끔 = 가운데([r[s]["끔"]["ttft"] for s in 시드들])
        없 = 가운데([r[s]["한도 없음"]["ttft"] for s in 시드들])
        줄.append(f"| 프롬프트 캐시 끔 | " + " | ".join(f"{끔:,.0f} 초" for _ in 용량들) + " |")
        for 정책 in 정책들 + 민감정책:
            칸 = []
            for 용량 in 용량들:
                v = 가운데([r[s][f"{정책}|{용량}"]["ttft"] for s in 시드들])
                if 정책 == "지금":
                    칸.append(f"**{v:,.0f} 초**")
                    continue
                g = [1 - r[s][f"{정책}|{용량}"]["ttft"] / r[s][f"지금|{용량}"]["ttft"] for s in 시드들]
                칸.append(f"{v:,.0f} 초 ({가운데(g) * 100:+.1f} % · {min(g) * 100:+.1f} ~ {max(g) * 100:+.1f})")
                판정.append((시나리오, 정책, 용량, 가운데(g)))
            줄.append(f"| {정책} | " + " | ".join(칸) + " |")
        줄.append(f"| 한도 없음 | " + " | ".join(f"{없:,.0f} 초" for _ in 용량들) + " |")
        줄.append("\n적중률(복원한 청크 / 슬롯 뒤로 필요한 청크):\n")
        줄.append("| 정책 | " + " | ".join(f"{c} GB" for c in 용량들) + " |")
        줄.append("|---|" + "---|" * len(용량들))
        for 정책 in 정책들:
            줄.append(f"| {정책} | " + " | ".join(
                f"{가운데([r[s][f'{정책}|{용량}']['적중률'] for s in 시드들]) * 100:.1f} %" for 용량 in 용량들) + " |")
        줄.append(f"| 한도 없음 | " + " | ".join(
            f"{가운데([r[s]['한도 없음']['적중률'] for s in 시드들]) * 100:.1f} %" for _ in 용량들) + " |")
    줄.append("\n### 사후 탐색 — 쓰기 방식 비울때 · 비울때 + 슬롯에도 있는 청크를 먼저 지움(지금 대비 Σ TTFT, 괄호는 슬롯을 덮을 때 쓰기를 기다린 시간)\n")
    줄.append("| 작업 부하 · 안 | " + " | ".join(f"{c} GB" for c in 용량들) + " |")
    줄.append("|---|" + "---|" * len(용량들))
    for 시나리오 in 있는:
        for 정책 in 사후정책:
            칸 = []
            for 용량 in 용량들:
                g = [1 - 모두[(시나리오, s)][f"{정책}|{용량}"]["ttft"] / 모두[(시나리오, s)][f"지금|{용량}"]["ttft"] for s in 시드들]
                w = 가운데([모두[(시나리오, s)][f"{정책}|{용량}"]["쓰기대기"] for s in 시드들])
                칸.append(f"{가운데(g) * 100:+.1f} % · {min(g) * 100:+.1f} ~ {max(g) * 100:+.1f} ({w:,.0f} 초)")
            줄.append(f"| {시나리오} · {정책} | " + " | ".join(칸) + " |")
    줄.append("\n### 사후 진단 — 지금: 요청이 올 때 프롬프트 캐시 바이트 중 슬롯에도 있는 몫(시간 평균)\n")
    줄.append("| 작업 부하 | " + " | ".join(f"{c} GB" for c in 용량들) + " |")
    줄.append("|---|" + "---|" * len(용량들))
    for 시나리오 in 있는:
        줄.append(f"| {시나리오} | " + " | ".join(
            f"{가운데([모두[(시나리오, s)][f'지금|{용량}']['슬롯중복'] for s in 시드들]) * 100:.0f} %" for 용량 in 용량들) + " |")
    if "R" in 있는 and "정밀|끔" in 모두[("R", 시드들[0])]:
        r = {s: 모두[("R", s)] for s in 시드들}
        줄.append("\n### R · 정밀 모드(프롬프트 계산 × 1.3) · 20 GB\n")
        줄.append("| 정책 | Σ TTFT | 지금 대비 |")
        줄.append("|---|---|---|")
        줄.append(f"| 프롬프트 캐시 끔 | {가운데([r[s]['정밀|끔']['ttft'] for s in 시드들]):,.0f} 초 | |")
        for 정책 in 정책들:
            v = 가운데([r[s][f"정밀|{정책}|20"]["ttft"] for s in 시드들])
            g = 가운데([1 - r[s][f"정밀|{정책}|20"]["ttft"] / r[s]["정밀|지금|20"]["ttft"] for s in 시드들])
            줄.append(f"| {정책} | {v:,.0f} 초 | {g * 100:+.1f} % |")
    return "\n".join(줄), 판정


def 판정하기(판정):
    """문서/캐시축출.md 3절 — 결과를 보기 전에 적은 기준"""
    줄 = ["\n### 판정(3절의 기준)\n"]
    후보 = [p for p in 정책들 if p not in ("지금", "Belady")]
    for p in 후보:
        이득 = [g for (sc, pp, c, g) in 판정 if pp == p and c <= 20]
        최악 = min(g for (sc, pp, c, g) in 판정 if pp == p)
        최고 = max(이득)
        if 최고 >= 0.10 and 최악 >= -0.02:
            등급 = "구현 제안"
        elif 최고 >= 0.03 and 최악 >= -0.02:
            등급 = "기록만"
        else:
            등급 = "지금 방식을 둠"
        줄.append(f"- {p}: 20 GB 이하에서 가장 큰 이득 {최고 * 100:+.1f} %, 모든 경우의 가장 나쁜 값 {최악 * 100:+.1f} % → {등급}")
    b = max(g for (sc, pp, c, g) in 판정 if pp == "Belady" and c <= 20)
    줄.append(f"- Belady(오프라인 기준): 20 GB 이하에서 가장 큰 여유 {b * 100:+.1f} %")
    return "\n".join(줄)


# ───── 확인 ─────

def 확인V1(번수=300):
    """지금의 때 규칙 · 차례가 시험/저장소CPU.py 의 색인(엔진과 빠른점검에서 대조되는 참조)과 같은가"""
    여기 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("저장소CPU", os.path.join(여기, "시험", "저장소CPU.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    틀린 = 0
    for 번 in range(번수):
        rng = random.Random(번)
        표 = 이름표()
        w = 부하()
        사슬들 = []
        for _ in range(4):                                         # 앞부분을 나눠 가진 토큰열 몇 개 → 청크 사슬
            앞 = 사슬들[rng.randrange(len(사슬들))][1] if 사슬들 and rng.random() < 0.7 else []
            열 = 앞자르기(앞, rng.randint(0, 길이(앞))) + [(w.새(), 0, rng.randint(300, 4000))]
            사슬들.append((표.이름들("7B", 열, 길이(열) // 덩이), 열))
        나 = 실행(표, "지금", None)
        참 = mod.색인()
        for _ in range(60):
            ids = 사슬들[rng.randrange(len(사슬들))][0]
            if not ids:
                continue
            일 = rng.random()
            if 일 < 0.45:                                          # 쓰기 — 청크 [a, b)
                a = rng.randrange(len(ids))
                b = rng.randint(a + 1, len(ids))
                for k in range(a, b):
                    if ids[k] not in 나.바이트:
                        나.넣기(ids[k], k)
                        참.때[str(ids[k])] = 참.새때(str(ids[k - 1]) if k > 0 else "", k)
            elif 일 < 0.9:                                         # 조회 — 0 … m−1
                m = rng.randint(1, len(ids))
                나.쓴때(ids, m)
                참.앞쪽쓴때([str(c) for c in ids], m)
            else:                                                  # 다시 켜기 — 색인.txt(이른 것부터)를 읽음
                줄 = [str(c) for c in sorted(나.바이트, key=lambda c: (나.때[c], 나.순번[c]))]
                나.다시켜기()
                참.읽기(list(줄), 줄)
            내차례 = [str(c) for c in sorted(나.바이트, key=lambda c: (나.때[c], 나.순번[c]))]
            if 내차례 != [n for n, _ in 참.차례()]:
                틀린 += 1
                break
    return 틀린, 번수


def 확인V2():
    """W0 — 녹화한 54 요청을 4단계 ②의 네 실행처럼(OS 캐시 복원)"""
    표 = 이름표()
    미리들 = 미리(부하W0(), 표)
    다시 = sum(T(ev["m"], ev["P"]) for ev in 미리들 if ev) / 1000.0
    끔 = 실행(표, "지금", None, 속도=OS캐시, 켬=False).돌리기(미리들)
    켬 = 실행(표, "지금", 20 * GB, 속도=OS캐시).돌리기(미리들)
    빈 = 켬.결과()
    for 이름 in ("ms", "생성"):
        setattr(켬, 이름, 0.0)
    for 이름 in ("복원", "필요수", "쓴", "쓴바이트", "지운", "이슬롯", "다른슬롯", "저장소토큰", "요청수", "프롬프트"):
        setattr(켬, 이름, 0)
    켬.다시켜기()
    뒤 = 켬.돌리기(미리들).결과()
    끔 = 끔.결과()
    생성1 = sum(생성ms(ev["m"], ev["P"], ev["A"]) for ev in 미리들 if ev) / 추측배 / 1000.0
    return dict(다시=다시, 끔=끔, 빈=빈, 뒤=뒤, 생성_추측없이=생성1, 추측배=157.3 / 생성1,
                프롬프트=sum(ev["P"] for ev in 미리들 if ev), 답=sum(ev["A"] for ev in 미리들 if ev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--확인", action="store_true")
    ap.add_argument("--시드", type=int, default=5)
    ap.add_argument("--일꾼", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--json")
    ap.add_argument("--시나리오", default=",".join(시나리오들))
    인자 = ap.parse_args()
    if 인자.확인:
        틀린, 번수 = 확인V1()
        print(f"V1 지금의 때 · 차례 = 시험/저장소CPU.py 의 색인: {번수 - 틀린}/{번수} 같음")
        v = 확인V2()
        print(f"V2 W0: 요청 54 · 프롬프트 {v['프롬프트']:,} 토큰 · 답 {v['답']:,} 토큰")
        print(f"  다시 계산 Σ TTFT {v['다시']:.1f} 초 (4단계 249.8)")
        for 이름, 잰, 기 in (("끔", "31.1", v["끔"]), ("켬 · 빈 폴더", "29.7", v["빈"]), ("켬 · 다시 켠 뒤", "5.4", v["뒤"])):
            print(f"  {이름}: Σ TTFT {기['ttft']:.1f} 초 (4단계 {잰}) · 이 슬롯 {기['이슬롯']:,} · 다른 슬롯 {기['다른슬롯']:,} · "
                  f"프롬프트 캐시 {기['저장소토큰']:,} · 쓴 청크 {기['쓴']}")
        print("  4단계: 이 슬롯 859,362 · 다른 슬롯 24,147 · 프롬프트 캐시 0 / 10,230 / 106,187 · 쓴 청크 478")
        print(f"  생성(행 1 스텝, 추측 없이) {v['생성_추측없이']:.1f} 초 → 4단계 157.3 초에 맞춘 추측배 {v['추측배']:.3f}")
        return
    시드들 = list(range(1, 인자.시드 + 1))
    고른 = 인자.시나리오.split(",")
    경우 = [(sc, s) for sc in 고른 for s in 시드들]
    t0 = time.time()
    모두 = {}
    with ProcessPoolExecutor(max_workers=인자.일꾼) as ex:
        for sc, s, out in ex.map(한경우, 경우):
            모두[(sc, s)] = out
            print(f"  {sc} 시드 {s} ({time.time() - t0:.0f} 초)", file=sys.stderr, flush=True)
    글, 판정 = 표들(모두, 시드들)
    print(글)
    print(판정하기(판정))
    if 인자.json:
        with open(인자.json, "w", encoding="utf-8") as f:
            json.dump({f"{sc}|{s}": out for (sc, s), out in 모두.items()}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
