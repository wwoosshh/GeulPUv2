#!/usr/bin/env python3
"""재기/생성2.sh 의 결과 폴더를 표로: 모델마다 정밀 · 빠른 모드의 생성 ms/토큰(채팅 질문 · 문맥 2048 · 8192 — 엔진/재기_7B)과 빠른 / 정밀.
여러 바퀴면 바퀴마다의 값의 가운데. 판정(2단계 C 의 생성 쪽): 빠른이 정밀의 ±2 % 안.

  python -X utf8 재기/생성요약.py <결과 폴더> [<요약.json>]
"""
import glob
import json
import os
import re
import statistics
import sys

칸들 = (("채팅 질문", r"생성 채팅 질문 ([\d.]+)"), ("문맥 2048", r"생성 문맥 2048 ([\d.]+)"), ("문맥 8192", r"생성 문맥 8192 ([\d.]+)"))


def 값(경로):
    글 = open(경로, encoding="utf-8").read()
    v = {}
    for 이름, 꼴 in 칸들:
        m = re.search(꼴, 글)
        if m:
            v[이름] = float(m.group(1))
    return v


def main():
    폴더 = sys.argv[1]
    이름들 = sorted({re.sub(r"^생성(정밀|빠른)_(.+)_\d+\.txt$", r"\2", os.path.basename(f)) for f in glob.glob(os.path.join(폴더, "생성*_*_*.txt"))})
    요약 = {}
    통과 = True
    for 이름 in 이름들:
        정 = [값(f) for f in sorted(glob.glob(os.path.join(폴더, f"생성정밀_{이름}_*.txt")))]
        빠 = [값(f) for f in sorted(glob.glob(os.path.join(폴더, f"생성빠른_{이름}_*.txt")))]
        print(f"\n{이름} (바퀴: 정밀 {len(정)} · 빠른 {len(빠)}) — ms/토큰")
        print(f"{'칸':>10} {'정밀':>9} {'빠른':>9} {'빠른/정밀':>10}")
        요약[이름] = {}
        for 칸, _ in 칸들:
            g = statistics.median([v[칸] for v in 정 if 칸 in v]) if any(칸 in v for v in 정) else None
            f = statistics.median([v[칸] for v in 빠 if 칸 in v]) if any(칸 in v for v in 빠) else None
            비 = f / g if g and f else None
            if 비 is not None and 비 > 1.02:
                통과 = False
            fmt = lambda x: f"{x:9.3f}" if x is not None else f"{'-':>9}"
            print(f"{칸:>10} {fmt(g)} {fmt(f)} {f'{비:10.3f}' if 비 is not None else '-':>10}")
            요약[이름][칸] = {"정밀": g, "빠른": f, "빠른/정밀": 비}
    print("\n판정(빠른 ≤ 정밀 × 1.02):", "통과" if 통과 else "못 미침")
    if len(sys.argv) > 2:
        json.dump({"모델": 요약, "통과": 통과}, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
