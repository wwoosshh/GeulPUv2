#!/usr/bin/env python3
"""38단계 — 재현 캡슐을 GPU 없이 확인한다. 받은 사람이 GeulPU 를 쓰지 않아도 계약(적힌 명세)으로 비트까지 확인하는 길.

    python -I 도구/캡슐확인.py <캡슐.json> <모델.gguf> [--출력 결과.json] [--해시건너뜀]

캡슐(연구 패널 · 재현 → 캡슐 내보내기)의 프롬프트 + 결과 토큰을 계약의 CPU 참조 구현(시험/참조7B.py — 정수 계약의 Qwen2 · Qwen3
앞으로, 22 · 26 · 37단계에서 GPU 와 모든 칸이 같았던 것)으로 셈해, 걸음마다

  1. 로짓 해시 — 로짓 행(짧은실수 V 개)의 FNV-1a 64(서버와 같은 정의: 32비트 값마다 h = (h ^ w) × 1099511628211, 처음 h = cbf29ce484222325)
  2. 로그확률 · 엔트로피 · 상위 k(번호 · 값) — 계약 5.2(계약/로그확률.py)
  3. 고른 토큰 — 계약 5절의 뽑기(시험/뽑기.py): 온도 0 이면 가장 큰 번호, 아니면 씨앗 · 걸음 번호(step_offset + 걸음)의 난수와 top-k · top-p · min-p
  4. 지문 — SHA-256("geulpu-fp/1\\nmodel <모델 SHA-256>\\nprompt <번호,…>\\ntokens <번호,…>\\n") · SHA-256("<토큰 지문>\\n<로짓 해시,…>\\n")

이 캡슐의 값과 비트까지 같은지 본다. 모델 파일의 SHA-256 이 캡슐과 같아야 한다(--해시건너뜀 이면 셈하지 않는다). numpy 가 필요하다
(pip install numpy). 7B 는 CPU 로 몇 분 걸린다.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
for _d in ("계약", "시험"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import gguf읽기 as GR       # noqa: E402
import 로그확률 as LP       # noqa: E402
import 참조7B as REF        # noqa: E402
import 뽑기 as SM           # noqa: E402

M64 = (1 << 64) - 1


def 행해시(z):
    """로짓 행의 FNV-1a 64 — 서버(호스트/서버몸.gl 의 행해시)와 같은 정의, 16 자리 16진 글."""
    h = 0xCBF29CE484222325
    for w in np.ascontiguousarray(z, np.float32).view(np.uint32).tolist():
        h = ((h ^ w) * 0x100000001B3) & M64
    return f"{h:016x}"


def 파일해시(경로):
    h = hashlib.sha256()
    with open(경로, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                return h.hexdigest()
            h.update(b)


def 지문(sha, prompt, tokens, hashes):
    fp = hashlib.sha256(f"geulpu-fp/1\nmodel {sha}\nprompt {','.join(map(str, prompt))}\ntokens {','.join(map(str, tokens))}\n".encode()).hexdigest()
    return fp, hashlib.sha256(f"{fp}\n{','.join(hashes)}\n".encode()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="GeulPU 재현 캡슐을 CPU 참조 구현으로 확인한다")
    ap.add_argument("캡슐")
    ap.add_argument("모델")
    ap.add_argument("--출력")
    ap.add_argument("--해시건너뜀", action="store_true")
    a = ap.parse_args()
    raw = open(a.캡슐, encoding="utf-8").read()
    cap = json.loads(raw)                                   # 파이썬의 json 은 큰 정수(씨앗)를 그대로 읽는다
    if cap.get("format") != "geulpu-capsule/1":
        sys.exit("GeulPU 재현 캡슐이 아니다")
    rq, rs = cap["request"], cap["result"]
    if (cap.get("made_on") or {}).get("precision") == "fast":
        REF.선형자리수 = 1          # GeulPUv2 빠른 계약 F1(계약.md 1.2) — 선형층의 활성값 자리 하나
    결과 = {"캡슐": os.path.basename(a.캡슐), "모델": cap["model"], "만든 곳": cap.get("made_on"), "걸음": len(rs["tokens"]),
            "계약": "F1(선형 자리 하나)" if REF.선형자리수 == 1 else "정밀(자리 둘)"}
    if not a.해시건너뜀:
        t0 = time.time()
        h = 파일해시(a.모델)
        결과["모델 SHA-256"] = "같음" if h == cap["model"]["sha256"] else "다름"
        print(f"모델 파일 SHA-256 {h[:16]}… — {결과['모델 SHA-256']} ({time.time() - t0:.0f} 초)", flush=True)
        if h != cap["model"]["sha256"]:
            sys.exit("모델 파일이 캡슐의 것과 다르다")
    P, toks = list(rq["prompt_tokens"]), list(rs["tokens"])
    n, k = len(toks), len(rs["top_ids"][0]) if rs["top_ids"] and rs["top_ids"][0] else 0
    T = float(rq["temperature"])
    K, Pp, Q = int(rq.get("top_k", 40)), float(rq.get("top_p", 1.0)), float(rq.get("min_p", 0.0))
    seed, off = int(rq.get("seed", 0)), int(rq.get("step_offset", 0))
    g = GR.GGUF(a.모델)
    S = np.array(P + toks[:-1], np.int64)                   # 행 P−1 … P+n−2 가 결과 토큰마다의 분포
    print(f"CPU 참조로 행 {len(S)} 개 셈하는 중(프롬프트 {len(P)} + 결과 {n} 토큰) …", flush=True)
    t0 = time.time()
    로짓, _ = REF.앞으로(g, [S], 말=False)
    결과["CPU 초"] = round(time.time() - t0, 1)
    다름 = {"로짓 해시": [], "로그확률": [], "엔트로피": [], "상위 k": [], "고른 토큰": []}
    hs = []
    for t in range(n):
        z = 로짓[len(P) - 1 + t]
        hs.append(행해시(z))
        r = LP.로그확률행(z, 위수=k)
        if hs[-1] != rs["logit_hashes"][t]:
            다름["로짓 해시"].append(t)
        if float(r["로그확률"][toks[t]]) != rs["token_logprobs"][t]:
            다름["로그확률"].append(t)
        if float(r["H"]) != rs["entropy"][t]:
            다름["엔트로피"].append(t)
        if [j for j, _ in r["위"]] != rs["top_ids"][t] or [v for _, v in r["위"]] != rs["top_logprobs"][t]:
            다름["상위 k"].append(t)
        j, _ = SM.뽑기([float(x) for x in z], T, K, Pp, Q, seed, off + t)
        if j != toks[t]:
            다름["고른 토큰"].append(t)
    fp, fl = 지문(cap["model"]["sha256"], P, toks, hs)
    fpc = cap.get("fingerprint") or {}
    결과["지문"] = "같음" if (fp, fl) == (fpc.get("token"), fpc.get("logit")) else ("캡슐에 없음" if not fpc else "다름")
    for 이름, v in 다름.items():
        결과[이름] = "같음" if not v else f"다름 — 걸음 {[x + 1 for x in v[:8]]}{' …' if len(v) > 8 else ''} ({len(v)}/{n})"
        print(f"{이름}: {결과[이름]}", flush=True)
    print(f"지문: {결과['지문']}", flush=True)
    결과["모두 같음"] = all(not v for v in 다름.values()) and 결과["지문"] in ("같음", "캡슐에 없음")
    print("모두 같음 — 이 캡슐은 계약대로 셈한 값과 비트까지 같다" if 결과["모두 같음"] else "다른 곳이 있다", flush=True)
    if a.출력:
        json.dump(결과, open(a.출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if 결과["모두 같음"] else 1)


if __name__ == "__main__":
    main()
