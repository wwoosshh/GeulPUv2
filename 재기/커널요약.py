#!/usr/bin/env python3
"""재기/커널2c.sh 의 결과(엔진/재기_커널 — 커널 종류마다 ms)를 표로: 모델 · 모드 · 길이마다 합, 행렬곱 넷(qkv · o · gate·up · down)의 합과
int8 텐서 코어 바닥 대비, 어텐션 · 접기 · 그 밖. 바닥 = 토큰 × 층 × (D·QKV + D·D + D·2F + F·D) 곱셈 · 덧셈 ÷ 160 TMAC/s(RTX 4070 Ti — SM 60 ×
1,022 MAC/클록 × 2.61 GHz). 자리 하나(F1)든 둘(정밀)이든 텐서곱 한 벌의 바닥(정밀은 이 두 배가 일의 양).

  python -X utf8 재기/커널요약.py <결과 폴더> [<기준 폴더 — 견줄 때>]
"""
import glob
import os
import re
import sys

모양 = {"qwen7b-q4km": (3584, 4608, 18944, 28), "qwen7b-q8": (3584, 4608, 18944, 28), "llama8b-q4km": (4096, 6144, 14336, 32)}
MACS = 60 * 1022 * 2.61e9


def 읽기(경로):
    v = {}
    for l in open(경로, encoding="utf-8"):
        m = re.match(r"프롬프트 (\d+) 합 ([\d.]+) (.*)", l)
        if m:
            칸 = dict((a, float(b)) for a, b in re.findall(r"(\S+) ([\d.]+)", m.group(3)))
            칸["합"] = float(m.group(2))
            v[int(m.group(1))] = 칸
    return v


def 바닥(모델, n):
    D, Q, F, L = 모양[모델]
    return n * L * (D * Q + D * D + D * 2 * F + F * D) / MACS * 1e3


def main():
    폴더 = sys.argv[1]
    기준 = sys.argv[2] if len(sys.argv) > 2 else None
    for f in sorted(glob.glob(os.path.join(폴더, "커널*_*.txt"))):
        m = re.match(r"커널(정밀|빠른)_(.+)\.txt$", os.path.basename(f))
        if not m:
            continue
        모드, 모델 = m.groups()
        v = 읽기(f)
        b = 읽기(os.path.join(기준, os.path.basename(f))) if 기준 and os.path.exists(os.path.join(기준, os.path.basename(f))) else None
        print(f"\n{모델} · {모드} (ms)")
        print(f"{'토큰':>6} {'합':>8} {'행렬곱':>8} {'바닥':>7} {'바닥%':>6} {'qkv':>7} {'o':>7} {'gate·up':>8} {'down':>7} {'어텐션':>7} {'접기':>6}"
              + (f" {'행렬곱 전':>9} {'비':>6}" if b else ""))
        for n in sorted(v):
            x = v[n]
            행 = x.get("qkv", 0) + x.get("o", 0) + x.get("gate·up", 0) + x.get("down", 0)
            fl = 바닥(모델, n)
            줄 = (f"{n:>6} {x['합']:8.1f} {행:8.1f} {fl:7.1f} {100 * fl / 행:5.0f}% {x.get('qkv', 0):7.1f} {x.get('o', 0):7.1f} "
                 f"{x.get('gate·up', 0):8.1f} {x.get('down', 0):7.1f} {x.get('어텐션', 0):7.1f} {x.get('접기', 0):6.1f}")
            if b and n in b:
                행b = sum(b[n].get(k, 0) for k in ("qkv", "o", "gate·up", "down"))
                줄 += f" {행b:9.1f} {행 / 행b:6.3f}"
            print(줄)


if __name__ == "__main__":
    main()
