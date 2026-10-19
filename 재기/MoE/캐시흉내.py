#!/usr/bin/env python3
"""MoE 재기 — 기록한 전문가 고르기로 VRAM 전문가 캐시를 흉내 낸다(정책마다 적중률 · 놓친 수).

  python 캐시흉내.py <슬롯 수> <대상.npz> [--프로필 a.npz,b.npz] [--대조 c.npz] [--ub 512] [--작은 32] [--json 출력]

걸음: 출력 토큰 하나 = 걸음 하나(층마다 고른 8 개). 프롬프트는 ub 덩이로 나눠 '작은'(llama.cpp 의 max_batch 32) 이하인 덩이만
캐시를 거친다(llama.cpp 와 같음 — 큰 덩이는 캐시를 바꾸지 않는다). reprocess 는 뺀다.
열쇠 = 층 · 256 + 전문가. 정책:
  LRU         llama.cpp 의 plan 과 같은 차례(층마다 적중을 먼저 갱신, 놓친 것을 번호 차례로 넣으며 가장 오래된 것을 뺌)
  Belady      앞날을 아는 최적(다음에 가장 늦게 쓸 것을 뺌) — 어떤 정책도 넘지 못하는 적중의 상한
  고정-제것    같은 기록의 빈도 상위 C 개를 처음부터 끝까지(정적 배치가 앞날을 알 때의 상한)
  고정-프로필  다른 기록(같은 사람의 다른 작업)의 빈도 상위 — 한 사람의 프로필
  LRU+프로필  프로필 상위로 채우고 시작하는 LRU
  고정-대조글  코딩이 아닌 글의 빈도 상위
"""
import heapq
import json
import sys
from collections import Counter, OrderedDict

import numpy as np

NE = 256


def 걸음들(npz, ub=512, 작은=32):
    d = np.load(npz)
    ids, kind = d["ids"].astype(np.int64), d["kind"]
    L = ids.shape[1]
    층키 = (np.arange(L, dtype=np.int64) * NE)[None, :, None]
    키 = ids + 층키                                     # [토큰, 층, 고름]
    출력, 묶음 = [], []
    n = len(kind)
    i = 0
    while i < n:
        k = kind[i]
        j = i
        while j < n and kind[j] == k:
            j += 1
        if k == 1:
            for t in range(i, j):
                출력.append(True)
                묶음.append(키[t])                      # [층, 고름] — 층 안에서 서로 다름
        elif k == 0:
            for a in range(i, j, ub):
                b = min(j, a + ub)
                if b - a <= 작은:
                    출력.append(False)
                    묶음.append([np.unique(키[a:b, l].ravel()) for l in range(L)])
        i = j
    return 출력, 묶음, L


def 빈도(묶음):
    c = Counter()
    for ls in 묶음:
        for es in ls:
            c.update(es.tolist())
    return c


def lru(출력, 묶음, C, 처음=()):
    od = OrderedDict((k, None) for k in list(처음)[:C])
    h = m = ho = mo = 0
    for o, ls in zip(출력, 묶음):
        for es in ls:
            es = es.tolist()
            hit = [k for k in es if k in od]
            for k in hit:
                od.move_to_end(k)
            miss = sorted(k for k in es if k not in od)
            for k in miss:
                if len(od) >= C:
                    od.popitem(last=False)
                od[k] = None
            h += len(hit)
            m += len(miss)
            if o:
                ho += len(hit)
                mo += len(miss)
    return h, m, ho, mo


def slru(출력, 묶음, C, 보호비=0.8):
    """두 칸 LRU — 처음 들어온 것은 시험 칸, 시험 칸에서 다시 쓰이면 보호 칸으로(넘치면 보호 칸의 가장 오래된 것을 시험 칸 앞으로).
    빼는 것은 시험 칸의 가장 오래된 것. 자주 쓰는 전문가가 한 번 쓰고 마는 전문가에게 밀려나지 않게 한다(실용 정책의 예)."""
    시험, 보호 = OrderedDict(), OrderedDict()
    P = int(C * 보호비)
    h = m = ho = mo = 0
    for o, ls in zip(출력, 묶음):
        for es in ls:
            es = es.tolist()
            a = b = 0
            for k in es:
                if k in 보호:
                    보호.move_to_end(k)
                    a += 1
                elif k in 시험:
                    del 시험[k]
                    보호[k] = None
                    if len(보호) > P:
                        kk, _ = 보호.popitem(last=False)
                        시험[kk] = None
                    a += 1
            for k in sorted(es):
                if k in 보호 or k in 시험:
                    continue
                if len(보호) + len(시험) >= C:
                    if 시험:
                        시험.popitem(last=False)
                    else:
                        보호.popitem(last=False)
                시험[k] = None
                b += 1
            h += a
            m += b
            if o:
                ho += a
                mo += b
    return h, m, ho, mo


def belady(출력, 묶음, C, 데움=False):
    """데움 = 참이면 처음 쓰는 차례로 앞의 C 개 열쇠를 미리 채우고 시작(앞날을 아는 미리 채우기 — 처음 놓침도 없앤 상한)."""
    키 = np.concatenate([np.concatenate(list(ls)) for ls in 묶음])
    출력표 = np.concatenate([np.full(sum(len(es) for es in ls), o, bool) for o, ls in zip(출력, 묶음)])
    n = len(키)
    차례 = np.argsort(키, kind="stable")
    다음 = np.full(n, 1 << 62, np.int64)
    같음 = 키[차례[1:]] == 키[차례[:-1]]
    다음[차례[:-1][같음]] = 차례[1:][같음]
    in_cache, heap = {}, []
    if 데움:
        _, 첫자리 = np.unique(키, return_index=True)
        for p in np.sort(첫자리)[:C].tolist():
            in_cache[int(키[p])] = p
            heap.append((-p, int(키[p])))
        heapq.heapify(heap)
    h = m = ho = mo = 0
    키l, 다음l, 출력l = 키.tolist(), 다음.tolist(), 출력표.tolist()
    for p in range(n):
        k, nx = 키l[p], 다음l[p]
        if k in in_cache:
            h += 1
            ho += 출력l[p]
        else:
            m += 1
            mo += 출력l[p]
            while len(in_cache) >= C:
                nn, kk = heapq.heappop(heap)
                if in_cache.get(kk) == -nn:
                    del in_cache[kk]
        in_cache[k] = nx
        heapq.heappush(heap, (-nx, k))
    return h, m, ho, mo


def 고정(출력, 묶음, 집합):
    h = m = ho = mo = 0
    for o, ls in zip(출력, 묶음):
        for es in ls:
            a = sum(1 for k in es.tolist() if k in 집합)
            h += a
            m += len(es) - a
            if o:
                ho += a
                mo += len(es) - a
    return h, m, ho, mo


def main():
    C = int(sys.argv[1])
    대상 = sys.argv[2]
    인자 = sys.argv[3:]

    def 값(이름, 기본=None):
        return 인자[인자.index(이름) + 1] if 이름 in 인자 else 기본

    ub, 작은 = int(값("--ub", 512)), int(값("--작은", 32))
    출력, 묶음, L = 걸음들(대상, ub, 작은)
    n출력 = sum(출력)
    결과 = {"target": 대상, "slots": C, "steps": len(묶음), "output_tokens": n출력, "layers": L, "policies": {}}

    def 적기(이름, r):
        h, m, ho, mo = r
        결과["policies"][이름] = {"hit": h / max(1, h + m), "hit_output": ho / max(1, ho + mo),
                                 "miss_per_output_token": mo / max(1, n출력)}
        print(f"{이름:10s} 적중 {100 * h / max(1, h + m):6.2f} %  출력 적중 {100 * ho / max(1, ho + mo):6.2f} %  "
              f"출력 토큰당 놓침 {mo / max(1, n출력):6.2f}", flush=True)

    print(f"{대상}: 걸음 {len(묶음)} (출력 {n출력}), 층 {L}, 슬롯 {C}", flush=True)
    적기("LRU", lru(출력, 묶음, C))
    적기("SLRU", slru(출력, 묶음, C))
    적기("Belady", belady(출력, 묶음, C))
    적기("Belady+채움", belady(출력, 묶음, C, True))
    적기("고정-제것", 고정(출력, 묶음, {k for k, _ in 빈도(묶음).most_common(C)}))
    if 값("--프로필"):
        프 = Counter()
        for f in 값("--프로필").split(","):
            프.update(빈도(걸음들(f, ub, 작은)[1]))
        상위 = [k for k, _ in 프.most_common(C)]
        적기("고정-프로필", 고정(출력, 묶음, set(상위)))
        적기("LRU+프로필", lru(출력, 묶음, C, 상위[::-1]))
    if 값("--대조"):
        g = Counter()
        for f in 값("--대조").split(","):
            g.update(빈도(걸음들(f, ub, 작은)[1]))
        적기("고정-대조글", 고정(출력, 묶음, {k for k, _ in g.most_common(C)}))
    if 값("--json"):
        json.dump(결과, open(값("--json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
