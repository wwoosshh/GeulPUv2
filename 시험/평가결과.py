#!/usr/bin/env python3
"""모델 평가 1-2(문서/모델평가.md 1단계 판정 3) — 앱이 쓴 결과 하나를 파이썬으로 다시 계산해 비트까지 견준다(GPU 없이).

    python -I -X utf8 시험/평가결과.py check <결과 폴더> <기준값 폴더>

결과는 덤프를 켜고(/api/eval/run 의 "dump": true) 낸 것이어야 한다(cand.bin — 위치마다 평가할 모델의 lq(K) · 토큰 로그확률 · 1 위 번호).
기준값 폴더의 reference.json · topk.bin(위치마다 기준의 상위 K 번호 · 로그확률 · 토큰 로그확률)과 함께 위치마다 KL · top-1 · Δp · 음의
로그확률 둘 · 꼬리를 시험/평가계산.py 의 함수로 계산해 positions.bin 과, 그 합 · 평균 · 99 백분위수 · PPL 비를 summary.json 의 bits 와 견준다.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "계약"))
import 평가계산 as E           # noqa: E402
import 로그확률 as LP          # noqa: E402


def 비트(x):
    v = LP.실수비트(float(x))
    return v - (1 << 64) if v >= (1 << 63) else v


def check(결과, 기준):
    ref = json.load(open(os.path.join(기준, "reference.json"), encoding="utf-8"))
    K = ref["top_k"]
    topk = np.fromfile(os.path.join(기준, "topk.bin"), dtype="<i8")
    cand = np.fromfile(os.path.join(결과, "cand.bin"), dtype="<i8")
    pos = np.fromfile(os.path.join(결과, "positions.bin"), dtype="<i8").reshape(-1, 6)
    items = [json.loads(x) for x in open(os.path.join(결과, "items.jsonl"), encoding="utf-8") if x.strip()]
    s = json.load(open(os.path.join(결과, "summary.json"), encoding="utf-8"))
    o_ref = 0
    o_cand = 0
    줄 = 0
    다름 = 0
    합 = dict(n=0, top1=0, kl=0.0, dp=0.0, dp2=0.0, nr=0.0, nc=0.0, tail=0.0, klmax=0.0)
    kls = []
    for ri, it in zip(ref["items"], items):
        m = len(ri["answer"]) if ri.get("scored") else 0
        refrows = topk[o_ref:o_ref + m * (2 * K + 1)].reshape(m, 2 * K + 1) if m else None
        o_ref += m * (2 * K + 1)
        if not it.get("scored"):
            continue
        n = it["positions"]
        rows = cand[o_cand:o_cand + n * (K + 2)].reshape(n, K + 2)
        o_cand += n * (K + 2)
        for i in range(n):
            lr = [LP.비트실수(int(v)) for v in refrows[i, K:2 * K]]
            lq = [LP.비트실수(int(v)) for v in rows[i, :K]]
            lt_ref = LP.비트실수(int(refrows[i, 2 * K]))
            lt = LP.비트실수(int(rows[i, K]))
            kl = E.KL(lr, lq)
            top1 = 1 if int(rows[i, K + 1]) == int(refrows[i, 0]) else 0
            dp = E.지수(lt_ref) - E.지수(lt)
            tail = E.꼬리(lr)
            want = [비트(kl), top1, 비트(dp), 비트(0.0 - lt_ref), 비트(0.0 - lt), 비트(tail)]
            if [int(v) for v in pos[줄]] != want:
                다름 += 1
            줄 += 1
            합["n"] += 1
            합["top1"] += top1
            합["kl"] = 합["kl"] + kl
            합["dp"] = 합["dp"] + dp
            합["dp2"] = 합["dp2"] + dp * dp
            합["nr"] = 합["nr"] + (0.0 - lt_ref)
            합["nc"] = 합["nc"] + (0.0 - lt)
            합["tail"] = 합["tail"] + tail
            if kl > 합["klmax"]:
                합["klmax"] = kl
            kls.append(kl)
    n = 합["n"]
    def 평균(a):
        return a / float(n) if n else 0.0
    want = {"kl_mean": 평균(합["kl"]), "kl_p99": E.백분위(kls, 99), "kl_max": 합["klmax"], "top1": 평균(float(합["top1"])),
            "dp_mean": 평균(합["dp"]), "dp_sq_mean": 평균(합["dp2"]), "ppl_ratio": E.지수(평균(합["nc"]) - 평균(합["nr"])),
            "tail_mean": 평균(합["tail"])}
    bits = s["metrics"]["bits"]
    합다름 = [k for k, v in want.items() if 비트(v) != bits.get(k)]
    print(f"평가 결과 다시 계산: 위치 {줄}(positions.bin {len(pos)}) · 위치의 다름 {다름} · 합의 다름 {len(합다름)}" + (f" — {합다름}" if 합다름 else ""))
    return 0 if 다름 == 0 and not 합다름 and 줄 == len(pos) and s["metrics"]["positions"] == 줄 else 1


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 4 or sys.argv[1] != "check":
        print(__doc__)
        return 2
    return check(sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    sys.exit(main())
