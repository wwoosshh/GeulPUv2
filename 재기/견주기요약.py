#!/usr/bin/env python3
"""재기/견주기2.sh 의 결과 폴더를 표로: 모델 · 길이마다 글(정밀 · 빠른)과 llama.cpp(설정마다 표본의 가운데 값 → 가장 빠른 설정)의 ms 와 비.
여러 바퀴면 바퀴마다의 값의 가운데.

  python -X utf8 재기/견주기요약.py <결과 폴더> [<요약.json>]
"""
import glob
import json
import os
import re
import statistics
import sys


def 글값(경로):
    v = {}
    for l in open(경로, encoding="utf-8"):
        m = re.match(r"프롬프트 (\d+) ([\d.]+)", l)
        if m:
            v[int(m.group(1))] = float(m.group(2))
    return v


def 라마값(경로):
    """길이 → (가장 빠른 설정의 ms, 설정 글) — 설정마다 samples_ns 의 가운데."""
    txt = open(경로, encoding="utf-8").read().strip()
    try:
        rows = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]
    except ValueError:                               # 아직 쓰는 중(재는 동안의 중간 요약) — 이 바퀴는 뺀다
        print(f"(아직 덜 쓴 파일 — 뺌: {os.path.basename(경로)})", file=sys.stderr)
        return {}
    최선 = {}
    for r in rows:
        p = int(r["n_prompt"])
        if p == 0:
            continue
        ms = statistics.median(r["samples_ns"]) / 1e6
        설정 = f"fa{int(bool(r.get('flash_attn', 0)))}·ub{r['n_ubatch']}"
        if p not in 최선 or ms < 최선[p][0]:
            최선[p] = (ms, 설정)
    return 최선


def main():
    폴더 = sys.argv[1]
    이름들 = sorted({re.sub(r"^(글정밀|글빠른|라마)_(.+)_\d+\.(txt|json)$", r"\2", os.path.basename(f))
                  for f in glob.glob(os.path.join(폴더, "*_*_*.*")) if re.search(r"_\d+\.(txt|json)$", f)})
    요약 = {}
    for 이름 in 이름들:
        정 = [글값(f) for f in sorted(glob.glob(os.path.join(폴더, f"글정밀_{이름}_*.txt")))]
        빠 = [글값(f) for f in sorted(glob.glob(os.path.join(폴더, f"글빠른_{이름}_*.txt")))]
        라 = [라마값(f) for f in sorted(glob.glob(os.path.join(폴더, f"라마_{이름}_*.json")))]
        print(f"\n{이름} (바퀴: 정밀 {len(정)} · 빠른 {len(빠)} · llama.cpp {len(라)})")
        print(f"{'토큰':>6} {'정밀':>9} {'빠른':>9} {'llama.cpp':>10} {'설정':>11} {'정밀/라마':>9} {'빠른/라마':>9} {'정밀/빠른':>9}")
        요약[이름] = {}
        for p in (512, 2048, 4096, 8192):
            g = statistics.median([v[p] for v in 정 if p in v]) if 정 else None
            f = statistics.median([v[p] for v in 빠 if p in v]) if 빠 else None
            ls = [v[p] for v in 라 if p in v]
            l = statistics.median([x[0] for x in ls]) if ls else None
            설정 = ls[0][1] if ls else "-"
            fmt = lambda x: f"{x:9.1f}" if x is not None else f"{'-':>9}"
            비 = lambda a, b: f"{a / b:9.3f}" if a is not None and b is not None else f"{'-':>9}"
            print(f"{p:>6} {fmt(g)} {fmt(f)} {fmt(l):>10} {설정:>11} {비(g, l)} {비(f, l)} {비(g, f)}")
            요약[이름][p] = {"정밀": g, "빠른": f, "llama.cpp": l, "llama 설정": 설정}
    if len(sys.argv) > 2:
        json.dump(요약, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
