#!/usr/bin/env python3
"""38단계 — 재현 파일(reproduction file)을 GPU 없이 확인한다. 받은 사람이 GeulPU 를 쓰지 않아도 정밀도 계약(적힌 명세)으로 비트까지
확인하는 방법.

    python -I 도구/repro_check.py <재현 파일.json> <모델.gguf> [--out 결과.json] [--skip-hash]

재현 파일(연구 패널 · 재현 → 재현 파일 내보내기)의 프롬프트 + 결과 토큰을 계약의 CPU 참조 구현(시험/참조7B.py — 정수 계약의 Qwen2 ·
Qwen3 순전파, 22 · 26 · 37단계에서 GPU 와 모든 값이 같았던 것)으로 계산해, 스텝마다

  1. 로짓 해시 — 로짓 행(짧은실수 V 개)의 FNV-1a 64(서버와 같은 정의: 32비트 값마다 h = (h ^ w) × 1099511628211, 처음 h = cbf29ce484222325)
  2. 로그확률 · 엔트로피 · 상위 k(번호 · 값) — 계약 5.2(계약/로그확률.py)
  3. 샘플링한 토큰 — 계약 5절의 샘플링(시험/뽑기.py): 온도 0 이면 가장 큰 번호, 아니면 시드 · 스텝 번호(step_offset + 스텝)의 난수와
     top-k · top-p · min-p
  4. 지문 — SHA-256("geulpu-fp/1\\nmodel <모델 SHA-256>\\nprompt <번호,…>\\ntokens <번호,…>\\n") · SHA-256("<토큰 지문>\\n<로짓 해시,…>\\n")

이 재현 파일의 값과 비트까지 같은지 본다. 모델 파일의 SHA-256 이 재현 파일과 같아야 한다(--skip-hash 면 계산하지 않는다). numpy 가
필요하다(pip install numpy). 7B 는 CPU 로 몇 분 걸린다. 0.6.0 까지의 옵션 이름(--출력 · --해시건너뜀)도 받는다. 파일 형식 표시
"geulpu-capsule/1" 은 이미 내보낸 파일과 맞도록 그대로다.

디버거 v1 3단계 — 실행 로그(run log, 도구/debugger.py · 앱의 디버거)도 받는다. 고른 요청 · 분기마다 같은 방법으로 스텝마다 로짓 해시 ·
샘플링한 토큰을, 분기의 echo 행(분기 앞의 스텝)은 로짓 해시를, 그 요청에 같은 서버의 점수 줄이 있으면 로그확률 · 엔트로피 · 상위 k 를
본다. 7B 는 요청마다 CPU 로 몇 분이라 확인할 것을 고른다:

    python -I 도구/repro_check.py <실행 로그.jsonl> <모델.gguf> --requests 0,25 [--branches 1] [--out 결과.json] [--skip-hash]

샘플링 설정이 요청에 없으면 서버의 기본값(온도 0.8 · top_k 40 · top_p 0.95 · min_p 0.05)이다.
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


서버기본 = {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05}     # 서버(엔진/서버몸.gl 의 일감받기)의 기본값


def 헤더인가(x):
    return isinstance(x, dict) and (str(x.get("format", "")).startswith("geulpu-run-log/") or x.get("geulrun") == 1)


def 실행확인(a, 줄들):
    """실행 로그의 고른 요청 · 분기를 CPU 참조로 — 항목마다 같은가"""
    헤더, 본 = 줄들[0], 줄들[1:]
    요청들 = {x.get("i"): x for x in 본 if "type" not in x}
    분기들 = {x.get("b"): x for x in 본 if x.get("type") == "branch"}
    점수들 = [x for x in 본 if x.get("type") == "score"]
    고른요청 = [int(v) for v in (a.requests or "").split(",") if v.strip()]
    고른분기 = [int(v) for v in (a.branches or "").split(",") if v.strip()]
    if not 고른요청 and not 고른분기:
        sys.exit("실행 로그는 확인할 요청을 --requests 0,3 (분기는 --branches 1) 으로 고른다 — 7B 는 요청마다 CPU 로 몇 분 걸린다")

    def 서버(x):
        if isinstance(x.get("server"), dict):
            return x["server"]
        hs = 헤더.get("server") or {}
        return {"sha256": (헤더.get("model") or {}).get("sha256"), "precision": hs.get("precision"), "kv8": bool(hs.get("kv8")),
                "fast_attention": bool(hs.get("fast_attention")), "layer": hs.get("layer"), "yarn": hs.get("yarn")}

    항목 = []
    for i in 고른요청:
        if i not in 요청들:
            sys.exit(f"요청 {i} 가 실행 로그에 없다")
        항목.append((f"요청 {i}", 요청들[i]))
    for b in 고른분기:
        if b not in 분기들:
            sys.exit(f"분기 b{b} 가 실행 로그에 없다")
        항목.append((f"분기 b{b}", 분기들[b]))
    shas = {서버(x).get("sha256") for _, x in 항목}
    if len(shas) != 1:
        sys.exit(f"고른 항목의 모델이 하나가 아니다: {sorted(map(str, shas))}")
    sha = shas.pop()
    결과 = {"file": os.path.basename(a.repro_file), "format": 헤더.get("format") or "geulrun", "model_sha256_expected": sha, "items": []}
    if not a.skip_hash:
        t0 = time.time()
        h = 파일해시(a.model)
        결과["model_sha256"] = "same" if h == sha else "different"
        print(f"모델 파일 SHA-256 {h[:16]}… — {'같음' if h == sha else '다름'} ({time.time() - t0:.0f} 초)", flush=True)
        if h != sha:
            sys.exit("모델 파일이 이 실행의 것과 다르다")
    g = GR.GGUF(a.model)
    모두같음 = True
    for 이름, x in 항목:
        sv = 서버(x)
        REF.선형자리수 = 1 if sv.get("precision") == "fast" else 2           # GeulPUv2 빠른 계약 F1(계약.md 1.2) — 선형층의 활성값 자리 하나
        REF.C.KV8 = REF.선형자리수 == 1 and bool(sv.get("kv8"))               # GeulPUv2 ③b-2 — 8비트 KV(계약 F1A′, 계약.md 2.1)
        REF.C.F2 = REF.C.KV8 and bool(sv.get("fast_attention"))               # GeulPUv2 빠른 어텐션(계약 F2, 계약.md 2.2 — 8비트 KV 위에서)
        rq = x.get("request") or {}
        gp = ((x.get("response") or {}).get("geulpu")) or {}
        if x.get("type") == "branch" and x.get("kind") == "token":
            P = [int(v) for v in rq["prompt"]]
            메아리부터 = int((rq.get("geulpu") or {}).get("echo_from", 0))
            메아리해시 = gp.get("prompt_logit_hashes") or []
        else:
            P = list(gp.get("prompt_tokens") or [])
            메아리부터, 메아리해시 = 0, []
        toks = list(gp.get("tokens") or [])
        if not P or not toks or "seed" not in gp:
            print(f"{이름}: 건너뜀 — 응답에 prompt_tokens · tokens · seed 가 있어야 한다", flush=True)
            결과["items"].append({"item": 이름, "skipped": True})
            continue
        T = float(rq.get("temperature", 서버기본["temperature"]))
        K, Pp, Q = int(rq.get("top_k", 서버기본["top_k"])), float(rq.get("top_p", 서버기본["top_p"])), float(rq.get("min_p", 서버기본["min_p"]))
        seed, off = int(gp["seed"]), int((rq.get("geulpu") or {}).get("step_offset", 0))
        점수 = next((s for s in 점수들 if x.get("type") != "branch" and s.get("of") == x.get("i") and (s.get("server") or {}).get("sha256") == sha
                     and (s.get("server") or {}).get("precision") == sv.get("precision")
                     and bool((s.get("server") or {}).get("kv8")) == bool(sv.get("kv8"))
                     and bool((s.get("server") or {}).get("fast_attention")) == bool(sv.get("fast_attention"))), None)
        n, k = len(toks), (len(점수["top_ids"][0]) if 점수 and 점수.get("top_ids") and 점수["top_ids"][0] else 0)
        S = np.array(P + toks[:-1], np.int64)
        계약이름 = (("빠른 · 8비트 KV · 빠른 어텐션" if REF.C.F2 else "빠른 · 8비트 KV") if REF.C.KV8 else "빠른") if REF.선형자리수 == 1 else "정밀"
        print(f"{이름}: CPU 참조로 행 {len(S)} 개 계산하는 중(프롬프트 {len(P)} + 답 {n} 토큰 · {계약이름} 계약) …", flush=True)
        t0 = time.time()
        로짓, _ = REF.앞으로(g, [S], 말=False)
        초 = round(time.time() - t0, 1)
        다름 = {"logit_hash": [], "sampled_token": []}
        if 점수:
            다름.update({"logprob": [], "entropy": [], "top_k": []})
        if 메아리해시:
            다름["echo_logit_hash"] = []
        hashes = gp.get("logit_hashes")
        for t in range(n):
            z = 로짓[len(P) - 1 + t]
            if hashes is not None and 행해시(z) != hashes[t]:
                다름["logit_hash"].append(t)
            j, _ = SM.뽑기([float(v) for v in z], T, K, Pp, Q, seed, off + t)
            if j != toks[t]:
                다름["sampled_token"].append(t)
            if 점수:
                r = LP.로그확률행(z, 위수=k)
                if float(r["로그확률"][toks[t]]) != 점수["logprobs"][t]:
                    다름["logprob"].append(t)
                if float(r["H"]) != 점수["entropy"][t]:
                    다름["entropy"].append(t)
                if [q for q, _ in r["위"]] != 점수["top_ids"][t] or [v for _, v in r["위"]] != 점수["top_logprobs"][t]:
                    다름["top_k"].append(t)
        for q, hx in enumerate(메아리해시):                     # 분기 앞의 스텝 — echo 행(위치 echo_from − 1 부터)
            if 행해시(로짓[메아리부터 - 1 + q]) != hx:
                다름["echo_logit_hash"].append(q)
        같음 = all(not v for v in 다름.values())
        모두같음 &= 같음
        보임 = " · ".join(f"{이름표[c]} {'같음' if not v else f'다름 {len(v)}'}" for c, v in 다름.items())
        print(f"{이름}: {보임} ({n} 스텝{f' · echo 행 {len(메아리해시)}' if 메아리해시 else ''} · {초} 초)", flush=True)
        결과["items"].append({"item": 이름, "steps": n, "cpu_seconds": 초, "same": 같음, "score_line": bool(점수),
                              **{c: {"same": not v, "different_steps": v[:50], "of": len(메아리해시) if c == "echo_logit_hash" else n} for c, v in 다름.items()}})
    결과["all_same"] = 모두같음
    print("모두 같음 — 고른 요청 · 분기가 계약대로 계산한 값과 비트까지 같다" if 모두같음 else "다른 곳이 있다", flush=True)
    if a.out:
        json.dump(결과, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if 모두같음 else 1)


이름표 = {"logit_hash": "로짓 해시", "sampled_token": "샘플링한 토큰", "logprob": "로그확률", "entropy": "엔트로피", "top_k": "상위 k",
        "echo_logit_hash": "echo 행 로짓 해시"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="GeulPU 재현 파일 · 실행 로그를 정밀도 계약의 CPU 참조 구현으로 확인한다")
    ap.add_argument("repro_file", metavar="repro_file.json|run.jsonl")
    ap.add_argument("model", metavar="model.gguf")
    ap.add_argument("--out", "--출력", dest="out", metavar="result.json", help="결과를 JSON 으로")
    ap.add_argument("--skip-hash", "--해시건너뜀", dest="skip_hash", action="store_true", help="모델 파일의 SHA-256 을 계산하지 않는다")
    ap.add_argument("--requests", help="실행 로그 — 확인할 요청 차례(i), 쉼표로")
    ap.add_argument("--branches", help="실행 로그 — 확인할 분기(b), 쉼표로")
    a = ap.parse_args()
    raw = open(a.repro_file, encoding="utf-8").read()
    첫 = raw.lstrip().split("\n", 1)[0]
    try:
        첫줄 = json.loads(첫)
    except ValueError:
        첫줄 = None
    if 헤더인가(첫줄):                                      # 실행 로그(JSONL)
        실행확인(a, [json.loads(l) for l in raw.splitlines() if l.strip()])
        return
    cap = json.loads(raw)                                   # 파이썬의 json 은 큰 정수(시드)를 그대로 읽는다
    if cap.get("format") != "geulpu-capsule/1":
        sys.exit("GeulPU 재현 파일이 아니다")
    rq, rs = cap["request"], cap["result"]
    if (cap.get("made_on") or {}).get("precision") == "fast":
        REF.선형자리수 = 1          # GeulPUv2 빠른 계약 F1(계약.md 1.2) — 선형층의 활성값 자리 하나
        REF.C.KV8 = bool((cap.get("made_on") or {}).get("kv8"))       # GeulPUv2 ③b-2 — 8비트 KV(계약 F1A′, 계약.md 2.1)
        REF.C.F2 = REF.C.KV8 and bool((cap.get("made_on") or {}).get("fast_attention"))   # GeulPUv2 빠른 어텐션(계약 F2, 계약.md 2.2)
    계약 = "exact (two digits)"
    if REF.선형자리수 == 1:
        계약 = "F1 (one digit in linear layers) + F1A′ (8-bit KV cache)" if REF.C.KV8 else "F1 (one digit in linear layers)"
        if REF.C.F2:
            계약 += " + F2 (fast attention)"
    결과 = {"file": os.path.basename(a.repro_file), "model": cap["model"], "made_on": cap.get("made_on"), "steps": len(rs["tokens"]),
            "contract": 계약}
    if not a.skip_hash:
        t0 = time.time()
        h = 파일해시(a.model)
        결과["model_sha256"] = "same" if h == cap["model"]["sha256"] else "different"
        print(f"모델 파일 SHA-256 {h[:16]}… — {'같음' if 결과['model_sha256'] == 'same' else '다름'} ({time.time() - t0:.0f} 초)", flush=True)
        if h != cap["model"]["sha256"]:
            sys.exit("모델 파일이 재현 파일의 것과 다르다")
    P, toks = list(rq["prompt_tokens"]), list(rs["tokens"])
    n, k = len(toks), len(rs["top_ids"][0]) if rs["top_ids"] and rs["top_ids"][0] else 0
    T = float(rq["temperature"])
    K, Pp, Q = int(rq.get("top_k", 40)), float(rq.get("top_p", 1.0)), float(rq.get("min_p", 0.0))
    seed, off = int(rq.get("seed", 0)), int(rq.get("step_offset", 0))
    g = GR.GGUF(a.model)
    S = np.array(P + toks[:-1], np.int64)                   # 행 P−1 … P+n−2 가 결과 토큰마다의 분포
    print(f"CPU 참조로 행 {len(S)} 개 계산하는 중(프롬프트 {len(P)} + 결과 {n} 토큰) …", flush=True)
    t0 = time.time()
    로짓, _ = REF.앞으로(g, [S], 말=False)
    결과["cpu_seconds"] = round(time.time() - t0, 1)
    다름 = {"logit_hash": [], "logprob": [], "entropy": [], "top_k": [], "sampled_token": []}
    보일이름 = {"logit_hash": "로짓 해시", "logprob": "로그확률", "entropy": "엔트로피", "top_k": "상위 k", "sampled_token": "샘플링한 토큰"}
    hs = []
    for t in range(n):
        z = 로짓[len(P) - 1 + t]
        hs.append(행해시(z))
        r = LP.로그확률행(z, 위수=k)
        if hs[-1] != rs["logit_hashes"][t]:
            다름["logit_hash"].append(t)
        if float(r["로그확률"][toks[t]]) != rs["token_logprobs"][t]:
            다름["logprob"].append(t)
        if float(r["H"]) != rs["entropy"][t]:
            다름["entropy"].append(t)
        if [j for j, _ in r["위"]] != rs["top_ids"][t] or [v for _, v in r["위"]] != rs["top_logprobs"][t]:
            다름["top_k"].append(t)
        j, _ = SM.뽑기([float(x) for x in z], T, K, Pp, Q, seed, off + t)
        if j != toks[t]:
            다름["sampled_token"].append(t)
    fp, fl = 지문(cap["model"]["sha256"], P, toks, hs)
    fpc = cap.get("fingerprint") or {}
    결과["fingerprint"] = "same" if (fp, fl) == (fpc.get("token"), fpc.get("logit")) else ("absent" if not fpc else "different")
    for 이름, v in 다름.items():
        결과[이름] = {"same": not v, "different_steps": [x + 1 for x in v], "of": n}
        보임 = "같음" if not v else f"다름 — 스텝 {[x + 1 for x in v[:8]]}{' …' if len(v) > 8 else ''} ({len(v)}/{n})"
        print(f"{보일이름[이름]}: {보임}", flush=True)
    지문보임 = {"same": "같음", "absent": "재현 파일에 없음", "different": "다름"}[결과["fingerprint"]]
    print(f"지문: {지문보임}", flush=True)
    결과["all_same"] = all(not v for v in 다름.values()) and 결과["fingerprint"] in ("same", "absent")
    print("모두 같음 — 이 재현 파일은 계약대로 계산한 값과 비트까지 같다" if 결과["all_same"] else "다른 곳이 있다", flush=True)
    if a.out:
        json.dump(결과, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if 결과["all_same"] else 1)


if __name__ == "__main__":
    main()
