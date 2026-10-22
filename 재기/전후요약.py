#!/usr/bin/env python3
"""1.0.0 측정 묶음의 요약(문서/1.0.0.md · 재기/전후.sh 의 결과).

  python -I -X utf8 재기/전후요약.py <결과 폴더> [<견줄 결과 폴더>] [--json 출력.json]

결과 폴더(재기/전후.sh 의 $W/<이름표>): E/*.json(에이전트 재생) · M/*.txt(재기_긴문맥) · L/*.json(llama-bench).
견줄 폴더를 주면 같은 이름끼리 비(뒤 ÷ 앞 — 1 보다 작으면 뒤가 빠르다)를 더한다. 시간은 ms(생성은 ms/토큰), E 는 초.
"""
import glob
import json
import os
import re
import sys


def 엔진(p):
    값 = {}
    for 줄 in open(p, encoding="utf-8"):
        m = re.match(r"빈 (\d+) ms ([\d.]+)", 줄)
        if m:
            값[f"빈 {int(m[1])}"] = float(m[2])
        m = re.match(r"깊이 (\d+) 새 1024 ms ([\d.]+)", 줄)
        if m:
            값[f"깊이 {int(m[1])} 새 1024"] = float(m[2])
        m = re.match(r"깊이 (\d+) 생성 (\d+) ms ([\d.]+)", 줄)
        if m:
            값[f"깊이 {int(m[1])} 행 {m[2]}"] = float(m[3])
    return 값


def 라마(p):
    값 = {}
    try:
        js = json.load(open(p, encoding="utf-8"))
    except Exception:
        return 값
    for r in js:
        n, d, g = r.get("n_prompt", 0), r.get("n_depth", 0), r.get("n_gen", 0)
        ms = r["avg_ns"] / 1e6
        if g:
            열쇠, v = f"깊이 {d} 행 1", ms / g
        elif d:
            열쇠, v = f"깊이 {d} 새 {n}", ms
        else:
            열쇠, v = f"빈 {n}", ms
        if 열쇠 not in 값 or v < 값[열쇠]:
            값[열쇠] = v
    return 값


def 읽기(폴더):
    out = {"E": {}, "M": {}, "L": {}}
    for p in sorted(glob.glob(os.path.join(폴더, "E", "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        out["E"][os.path.basename(p)[:-5]] = d.get("summary", {})
    for p in sorted(glob.glob(os.path.join(폴더, "M", "*.txt"))):
        out["M"][os.path.basename(p)[:-4]] = 엔진(p)
    묶음 = {}
    for p in sorted(glob.glob(os.path.join(폴더, "L", "*.json"))):
        이름 = os.path.basename(p)[:-5].rsplit("_", 1)[0]
        묶음.setdefault(이름, {}).update(라마(p))
    out["L"] = 묶음
    return out


def 열쇠차례(k):
    m = re.match(r"(빈|깊이) (\d+)(?: (새|행) (\d+))?", k)
    if not m:
        return (9, 0, k)
    return ({"빈": 0, "깊이": 1}[m[1]], int(m[2]), k)


def 보이기(A, B=None):
    print("== E 에이전트 재생 (요청 54 — 초)")
    for 이름, s in A["E"].items():
        a = s.get("all", {})
        줄 = (f"  {이름:22s} 첫 토큰 합 {a.get('ttft_s', 0):8.1f}  생성 {a.get('gen_tokens', 0):6d} 토큰 {a.get('gen_s', 0):7.1f}"
              f" ({a.get('gen_ms_per_token') or 0:6.2f} ms/토큰)  끝까지 {a.get('wall_s', 0):8.1f}  오류 {a.get('errors', 0)}")
        if B and 이름 in B["E"]:
            b = B["E"][이름].get("all", {})
            줄 += f"  | 앞 {b.get('wall_s', 0):8.1f} → 비 {a.get('wall_s', 0) / b['wall_s']:.3f}" if b.get("wall_s") else ""
        print(줄)
    for 쪽 in ("M", "L"):
        print(f"== {쪽} {'엔진 측정(재기_긴문맥)' if 쪽 == 'M' else 'llama.cpp llama-bench(ub 가운데 가장 빠른 것)'} — ms(행 1 은 ms/토큰)")
        for 이름, 값 in A[쪽].items():
            print(f"  -- {이름}")
            for k in sorted(값, key=열쇠차례):
                줄 = f"    {k:22s} {값[k]:10.2f}"
                if B and 이름 in B[쪽] and k in B[쪽][이름]:
                    줄 += f"   앞 {B[쪽][이름][k]:10.2f}  비 {값[k] / B[쪽][이름][k]:.3f}"
                print(줄)


def main():
    a = sys.argv[1:]
    js = None
    if "--json" in a:
        i = a.index("--json")
        js = a[i + 1]
        a = a[:i] + a[i + 2:]
    A = 읽기(a[0])
    B = 읽기(a[1]) if len(a) > 1 else None
    보이기(A, B)
    if js:
        json.dump({"A": A, "B": B}, open(js, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
