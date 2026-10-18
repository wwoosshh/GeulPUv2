#!/usr/bin/env python3
"""38단계 — 로그확률(계약 5.2)의 글 판(호스트/로그확률.gl)이 파이썬 참조(계약/로그확률.py)와 비트까지 같은가.

    python -I -X utf8 시험/로그확률38.py 대조 <시험_로그확률.exe> [--실제 로짓.bin ...] [--어휘 151936] [--출력 결과.json]
    python -I -X utf8 시험/로그확률38.py 시간 <시험_로그확률.exe> [--행 64]

대조: 합성 로짓 행(정규 · 뾰족한 것 · 같은 값이 많은 것 · 바닥 아래가 섞인 것)과 (주면) 실제 로짓 행을, 글 판이 한 번에(행 여럿 —
구간 합을 일꾼으로 나눔) · 하나씩(생성 걸음의 꼴) 셈한 것과 파이썬 참조를 견준다 — 표적의 로그확률 · 엔트로피 · 상위 k(번호 · 값).
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "계약"))
import 로그확률 as LP          # noqa: E402


def 합성행들(V, n, r):
    행들 = []
    for k in range(n):
        종류 = k % 4
        if 종류 == 0:
            z = r.standard_normal(V) * r.uniform(1, 6)
        elif 종류 == 1:
            z = r.standard_normal(V) * 2 - 5
            z[r.integers(0, V, 7)] = r.uniform(8, 25, 7)
        elif 종류 == 2:
            z = r.integers(-8, 3, V).astype(np.float64)
            z[r.integers(0, V, 3)] = -0.0
        else:
            z = r.standard_normal(V) * 15                     # 바닥(−40) 아래가 많은 것
        행들.append(z.astype(np.float32))
    return 행들


def 글돌리기(exe, 일, 행들, 표적, k):
    V = len(행들[0])
    d = tempfile.mkdtemp(prefix="geulpu_lp_")
    입, 출 = os.path.join(d, "입력.bin"), os.path.join(d, "출력.bin")
    with open(입, "wb") as f:
        f.write(struct.pack("<qqq", V, len(행들), k))
        for z in 행들:
            f.write(np.ascontiguousarray(z, np.float32).tobytes())
        f.write(struct.pack(f"<{len(행들)}q", *표적))
    subprocess.run([exe, 일, 입, 출], check=True)
    raw = open(출, "rb").read()
    out, o = [], 0
    for _ in 행들:
        tv, h = struct.unpack_from("<QQ", raw, o)
        o += 16
        위 = []
        for _ in range(k):
            j, v = struct.unpack_from("<qQ", raw, o)
            o += 16
            위.append((j, v))
        out.append((tv, h, 위))
    return out


def 대조(a):
    r = np.random.default_rng(38)
    행들 = 합성행들(a.어휘, a.합성, r)
    for 경로 in a.실제 or []:
        raw = open(경로, "rb").read()
        n = len(raw) // (4 * a.어휘)
        for i in range(min(n, a.실제행)):
            행들.append(np.frombuffer(raw, np.float32, a.어휘, i * 4 * a.어휘).copy())
    k = a.위수
    표적 = [int(r.integers(0, a.어휘)) if i % 3 else int(np.argmax(z)) for i, z in enumerate(행들)]
    참 = []
    for z, t in zip(행들, 표적):
        x = LP.로그확률행(z, 위수=k)
        참.append((LP.실수비트(float(x["로그확률"][t])), LP.실수비트(x["H"]), [(j, LP.실수비트(v)) for j, v in x["위"]]))
    결과 = {"행": len(행들), "어휘": a.어휘, "위수": k}
    for 일 in ("대조", "하나씩"):
        글 = 글돌리기(a.exe, 일, 행들, 표적, k)
        다름 = sum(1 for p, g in zip(참, 글) if p != g)
        결과[일] = 다름
        print(f"38a 글 판({일}): 행 {len(행들)} · 다름 {다름}")
        for i, (p, g) in enumerate(zip(참, 글)):
            if p != g and 다름 <= 3:
                print(f"  행 {i}: 참조 {p[:2]} {p[2][:3]} / 글 {g[:2]} {g[2][:3]}")
    return 결과


def 시간(a):
    r = np.random.default_rng(1)
    행들 = 합성행들(a.어휘, a.행, r)
    d = tempfile.mkdtemp(prefix="geulpu_lp_")
    입, 출 = os.path.join(d, "입력.bin"), os.path.join(d, "출력.txt")
    with open(입, "wb") as f:
        f.write(struct.pack("<qqq", a.어휘, len(행들), a.위수))
        for z in 행들:
            f.write(z.tobytes())
        f.write(struct.pack(f"<{len(행들)}q", *([0] * len(행들))))
    subprocess.run([a.exe, "시간", 입, 출], check=True)
    글 = open(출, encoding="utf-8").read()
    print(글, end="")
    return {"글": 글}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("일", choices=["대조", "시간"])
    ap.add_argument("exe")
    ap.add_argument("--실제", nargs="*")
    ap.add_argument("--실제행", type=int, default=200)
    ap.add_argument("--어휘", type=int, default=151936)
    ap.add_argument("--합성", type=int, default=24)
    ap.add_argument("--위수", type=int, default=20)
    ap.add_argument("--행", type=int, default=64)
    ap.add_argument("--출력")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    결과 = {"대조": 대조, "시간": 시간}[a.일](a)
    if a.출력:
        json.dump(결과, open(a.출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.일 == "대조":
        return 0 if 결과["대조"] == 0 and 결과["하나씩"] == 0 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
