#!/usr/bin/env python3
"""#96 — 층 렌즈(logit lens, 계약 5.3)를 서버에 바로 묻고 견준다. 서버(또는 앱의 /v1)가 떠 있어야 한다 — GPU 줄에 서서.

    python -I -X utf8 시험/렌즈.py <주소> <모델.gguf> [--빠른] [--KV8] [--F2] [--F2B] [--jlens J파일] [--CPU없이] [--출력 결과.json]
    python -I -X utf8 시험/렌즈.py CPU <결과.json> <모델.gguf> [--빠른] [--KV8] [--F2] [--F2B] [--jlens J파일]   (4 · 7 만 — GPU 줄 밖에서)
    python -I -X utf8 시험/렌즈.py 견주기 <결과 a.json> <결과 b.json>          (백엔드를 건너 — CUDA ↔ Vulkan, 정밀 모드)

모드 깃발은 서버가 도는 계약(시험/참조7B.py 와 같은 뜻 — --F2 는 --빠른 · --KV8 을, --KV8 은 --빠른 을 함께 뜻한다). 한 질문의 탐욕 답
(토큰 n 개) 가운데 스텝 셋(처음 · 가운데 · 끝)마다:
  1. 렌즈 요청(echo · max_tokens 0 · echo_from = 그 토큰)의 layer_lens 가 층 L 개 · 행(echo_from − 1) · 토큰이 맞는가.
  2. 마지막 층 = 같은 응답의 echo 값(로짓 해시 · 로그확률 · 엔트로피 · 상위 k) — 계약 5.3 의 2.
  3. 렌즈 요청의 echo 값 = 렌즈 없이 답 전체를 echo 로 매긴 값 — 덩이 · 길과 상관없이 같은 비트(렌즈 길은 넷 이하의 행도 계획 길).
  4. 층마다 CPU 참조(시험/참조7B.py 의 렌즈행 — 0.13.0 gemma4 는 시험/참조gemma4.py, 소프트캡까지)와 로짓 해시 · 5.2 의 값(로그확률 ·
     엔트로피 · 상위 k)이 비트까지 같은가(--CPU없이 면 건너뜀).
  5. 거절(400): echo 없음 · echo_from 이 프롬프트 끝 · top_k 21 · 틀린 형 · jacobian 이 참거짓이 아님. layer_lens 거짓이면 응답에 없다.
#96 2단계 — --jlens(서버를 같은 J 파일로 띄웠을 때 — Jacobian lens, 계약 5.4):
  6. layer_lens: {"jacobian": true} 의 layer_lens = 1 의 layer_lens(J 를 묻는 길도 같은 비트), jacobian_lens 의 꼴(층 · 행 · 토큰 · J 파일의
     SHA-256 · 원천 층 수)과 마지막 층 = logit lens 의 마지막 층.
  7. 층마다 CPU 참조(시험/참조7B.py 의 층로짓 — 같은 J)와 비트까지(--CPU없이 면 건너뜀).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

여기 = os.path.dirname(os.path.abspath(__file__))
for _d in ("계약", "시험", "도구"):
    sys.path.insert(0, os.path.join(여기, "..", _d))
import jlens읽기 as JL         # noqa: E402
import 로그확률 as LP          # noqa: E402

질문 = "질문: 대한민국의 수도는 어디인가? 답:"
위수 = 5


def 묻기(주소, 몸):
    q = urllib.request.Request(주소 + "/v1/completions", data=json.dumps(몸, ensure_ascii=False).encode("utf-8"),
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(q, timeout=900) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {}


def 받기(주소, 몸):
    s, j = 묻기(주소, 몸)
    if s != 200:
        sys.exit(f"HTTP {s}: {json.dumps(j, ensure_ascii=False)[:400]}")
    return j


def 돌리기(a):
    판정 = {}

    def 적기(이름, 맞음, 덧=""):
        판정[이름] = bool(맞음)
        print(f"  {'PASS' if 맞음 else 'FAIL'} {이름}{(' — ' + 덧) if 덧 else ''}", flush=True)

    t0 = time.time()
    생성 = 받기(a.주소, {"prompt": 질문, "max_tokens": 12, "temperature": 0, "logprobs": 위수,
                       "geulpu": {"prompt_tokens": True, "logit_hash": True}})
    P, T = list(생성["geulpu"]["prompt_tokens"]), list(생성["geulpu"]["tokens"])
    n = len(T)
    print(f"프롬프트 {len(P)} 토큰 · 답 {n} 토큰 {T} ({time.time() - t0:.1f} 초)", flush=True)
    전체 = 받기(a.주소, {"prompt": P + T, "echo": True, "max_tokens": 0, "logprobs": 위수, "temperature": 0,
                       "geulpu": {"echo_from": len(P), "logit_hash": True}})
    eg, elp = 전체["geulpu"], 전체["choices"][0]["logprobs"]
    걸음들 = sorted({0, n // 2, n - 1})
    결과 = {"model": os.path.basename(a.모델), "prompt_tokens": P, "tokens": T, "steps": []}
    렌즈들 = {}
    for i in 걸음들:
        ef = len(P) + i
        j = 받기(a.주소, {"prompt": P + T[:i + 1], "echo": True, "max_tokens": 0, "logprobs": 위수, "temperature": 0,
                          "geulpu": {"echo_from": ef, "logit_hash": True, "layer_lens": True}})
        g, lp = j["geulpu"], j["choices"][0]["logprobs"]
        ll = g.get("layer_lens") or {}
        L = int(ll.get("layers") or 0)
        print(f"스텝 {i + 1}(행 {ef - 1} · 토큰 #{T[i]}): 층 {L}", flush=True)
        적기(f"s{i + 1} 꼴", ll.get("row") == ef - 1 and ll.get("token") == T[i] and L > 0
             and all(len(ll[c]) == L for c in ("token_logprobs", "entropy", "top_ids", "top_logprobs", "logit_hashes", "ranks"))
             and ll["ranks"][L - 1] == (1 + g["prompt_top_ids"][ef].index(T[i]) if T[i] in g["prompt_top_ids"][ef] else ll["ranks"][L - 1]),
             f"row {ll.get('row')} · token {ll.get('token')}")
        echo = {"hash": g["prompt_logit_hashes"][0], "lp": lp["token_logprobs"][ef], "H": g["prompt_entropy"][ef],
                "ids": g["prompt_top_ids"][ef], "lps": g["prompt_top_logprobs"][ef]}
        끝 = {"hash": ll["logit_hashes"][L - 1], "lp": ll["token_logprobs"][L - 1], "H": ll["entropy"][L - 1],
              "ids": ll["top_ids"][L - 1], "lps": ll["top_logprobs"][L - 1]}
        적기(f"s{i + 1} 마지막 층 = echo", 끝 == echo, "" if 끝 == echo else f"{끝} ≠ {echo}")
        보통 = {"hash": eg["prompt_logit_hashes"][i], "lp": elp["token_logprobs"][ef], "H": eg["prompt_entropy"][ef],
                "ids": eg["prompt_top_ids"][ef], "lps": eg["prompt_top_logprobs"][ef]}
        적기(f"s{i + 1} 렌즈 길의 echo = 답 전체의 echo", echo == 보통, "" if echo == 보통 else f"{echo} ≠ {보통}")
        렌즈들[i] = ll
        x = {"step": i + 1, "row": ef - 1, "token": T[i], "layer_lens": ll, "echo": echo}
        if a.jlens:                                          # #96 2단계 — Jacobian lens(계약 5.4)
            jj = 받기(a.주소, {"prompt": P + T[:i + 1], "echo": True, "max_tokens": 0, "logprobs": 위수, "temperature": 0,
                               "geulpu": {"echo_from": ef, "logit_hash": True, "layer_lens": {"jacobian": True}}})
            jg = jj["geulpu"]
            jl = jg.get("jacobian_lens") or {}
            적기(f"s{i + 1} J 길의 layer_lens = 1 의 layer_lens", jg.get("layer_lens") == ll)
            적기(f"s{i + 1} jacobian_lens 꼴", jl.get("row") == ef - 1 and jl.get("token") == T[i] and jl.get("layers") == L
                 and jl.get("jlens_sha256") == a.J해시 and jl.get("source_layers") == a.J층수
                 and all(len(jl.get(c) or []) == L for c in ("token_logprobs", "entropy", "top_ids", "top_logprobs", "logit_hashes", "ranks")),
                 f"layers {jl.get('layers')} · sha {str(jl.get('jlens_sha256'))[:12]} · source {jl.get('source_layers')}")
            적기(f"s{i + 1} J 의 마지막 층 = logit lens 의 마지막 층", L > 0 and all((jl.get(c) or [None] * L)[L - 1] == ll[c][L - 1]
                 for c in ("logit_hashes", "token_logprobs", "entropy", "top_ids", "top_logprobs", "ranks")))
            x["jacobian_lens"] = jl
        결과["steps"].append(x)
    if not a.CPU없이:
        CPU확인(결과, a, 적기)
    # 거절
    짧은 = P + T[:1]
    for 이름, 몸 in (("echo 없음", {"prompt": 짧은, "max_tokens": 1, "logprobs": 위수, "geulpu": {"layer_lens": True}}),
                    ("echo_from 이 끝", {"prompt": 짧은, "echo": True, "max_tokens": 0, "logprobs": 위수,
                                         "geulpu": {"echo_from": len(짧은), "layer_lens": True}}),
                    ("top_k 21", {"prompt": 짧은, "echo": True, "max_tokens": 0, "logprobs": 위수,
                                  "geulpu": {"echo_from": len(P), "layer_lens": {"top_k": 21}}}),
                    ("틀린 형", {"prompt": 짧은, "echo": True, "max_tokens": 0, "logprobs": 위수,
                                "geulpu": {"echo_from": len(P), "layer_lens": 3}}),
                    ("jacobian 의 형", {"prompt": 짧은, "echo": True, "max_tokens": 0, "logprobs": 위수,
                                       "geulpu": {"echo_from": len(P), "layer_lens": {"jacobian": 1}}})):
        s, j = 묻기(a.주소, 몸)
        글 = ((j.get("error") or {}).get("message")) or ""
        적기(f"거절 — {이름}", s == 400 and "layer_lens" in 글, f"HTTP {s} {글[:80]}")
    s, j = 묻기(a.주소, {"prompt": 짧은, "echo": True, "max_tokens": 0, "logprobs": 위수,
                         "geulpu": {"echo_from": len(P), "layer_lens": False}})
    적기("layer_lens 거짓이면 없음", s == 200 and "layer_lens" not in (j.get("geulpu") or {}))
    결과["checks"] = 판정
    결과["all_pass"] = all(판정.values())
    print("모두 PASS" if 결과["all_pass"] else "FAIL 이 있다", flush=True)
    if a.출력:
        json.dump(결과, open(a.출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if 결과["all_pass"] else 1)


def CPU확인(결과, a, 적기):
    """4 — 층마다 CPU 참조(계약 5.3)와 로짓 해시 · 5.2 의 값이 비트까지 같은가. 결과의 프롬프트 · 답 · 스텝마다의 layer_lens 로"""
    import gguf읽기 as GR
    from repro_check import 행해시, 순위
    gg = GR.GGUF(a.모델)
    젬마 = gg.meta.get("general.architecture") == "gemma4"            # 0.13.0 — Gemma 4 의 계약 순전파(정밀만)
    if 젬마:
        import 참조gemma4 as REF
    else:
        import 참조7B as REF
    if (a.빠른 or a.KV8 or a.F2 or a.F2B) and not 젬마:
        REF.선형자리수 = 1
    REF.C.KV8 = bool(a.KV8 or a.F2 or a.F2B)
    REF.C.F2 = bool(a.F2 or a.F2B)
    REF.C.F2B = bool(a.F2B)
    P, T = 결과["prompt_tokens"], 결과["tokens"]
    행들 = [x["row"] for x in 결과["steps"]]
    t1 = time.time()
    print(f"CPU 참조로 행 {len(P) + len(T) - 1} 개 · 렌즈 행 {행들} 계산하는 중 …", flush=True)
    덧 = {"행": [(0, 1)]} if 젬마 else {}                              # gemma4 — 끝 로짓은 행 하나만(렌즈는 잔차로)
    _, _, 잔 = REF.앞으로(gg, [np.array(P + T[:-1], np.int64)], 말=False, 렌즈행=행들, 잔차=True, **덧)
    렌즈 = REF.층로짓(gg, 잔)
    렌즈J = REF.층로짓(gg, 잔, JL.읽기(a.jlens)[0]) if getattr(a, "jlens", None) and not 젬마 else None    # #96 2단계 — 같은 J 파일로(계약 5.4)
    print(f"  ({time.time() - t1:.0f} 초)", flush=True)

    def 다른층들(zz, ll, c):
        다른 = []
        for l in range(zz.shape[0]):
            z = zz[l]
            r = LP.로그확률행(z, 위수=len(ll["top_ids"][l]))
            같음 = (행해시(z) == ll["logit_hashes"][l] and float(r["로그확률"][c]) == ll["token_logprobs"][l]
                    and float(r["H"]) == ll["entropy"][l] and [int(v) for v, _ in r["위"]] == ll["top_ids"][l]
                    and [float(v) for _, v in r["위"]] == ll["top_logprobs"][l]
                    and ("ranks" not in ll or 순위(z, c) == ll["ranks"][l]))
            if not 같음:
                다른.append(l)
        return 다른
    for q, x in enumerate(결과["steps"]):
        ll = x["layer_lens"]
        다른층 = 다른층들(렌즈[q], ll, x["token"])
        적기(f"s{x['step']} 층마다 = CPU 참조", not 다른층 and 렌즈[q].shape[0] == ll["layers"], f"다른 층 {다른층[:8]}" if 다른층 else "")
        if 렌즈J is not None and x.get("jacobian_lens"):
            다른J = 다른층들(렌즈J[q], x["jacobian_lens"], x["token"])
            적기(f"s{x['step']} Jacobian lens 층마다 = CPU 참조", not 다른J and 렌즈J[q].shape[0] == x["jacobian_lens"]["layers"],
                 f"다른 층 {다른J[:8]}" if 다른J else "")


def CPU만(argv):
    ap = argparse.ArgumentParser(description="저장한 층 렌즈 결과를 CPU 참조로 확인한다")
    ap.add_argument("결과")
    ap.add_argument("모델")
    for f in ("--빠른", "--KV8", "--F2", "--F2B"):
        ap.add_argument(f, action="store_true")
    ap.add_argument("--jlens")
    a = ap.parse_args(argv)
    판정 = {}

    def 적기(이름, 맞음, 덧=""):
        판정[이름] = bool(맞음)
        print(f"  {'PASS' if 맞음 else 'FAIL'} {이름}{(' — ' + 덧) if 덧 else ''}", flush=True)

    CPU확인(json.load(open(a.결과, encoding="utf-8")), a, 적기)
    print("모두 PASS" if all(판정.values()) else "FAIL 이 있다", flush=True)
    sys.exit(0 if all(판정.values()) else 1)


def 견주기(pa, pb):
    a, b = (json.load(open(p, encoding="utf-8")) for p in (pa, pb))
    같음 = a["tokens"] == b["tokens"] and len(a["steps"]) == len(b["steps"])
    for x, y in zip(a["steps"], b["steps"]):
        h = x["layer_lens"]["logit_hashes"] == y["layer_lens"]["logit_hashes"] and x["echo"] == y["echo"]
        h = h and (x.get("jacobian_lens") or {}).get("logit_hashes") == (y.get("jacobian_lens") or {}).get("logit_hashes")    # #96 2단계
        print(f"스텝 {x['step']}: 층 {x['layer_lens']['layers']} 의 로짓 해시 · echo {'같음' if h else '다름'}", flush=True)
        같음 &= h
    print("모두 같음" if 같음 else "다른 곳이 있다", flush=True)
    sys.exit(0 if 같음 else 1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) >= 2 and sys.argv[1] == "견주기":
        견주기(sys.argv[2], sys.argv[3])
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "CPU":
        CPU만(sys.argv[2:])
        return
    ap = argparse.ArgumentParser(description="층 렌즈(계약 5.3)를 서버에 묻고 견준다")
    ap.add_argument("주소")
    ap.add_argument("모델")
    for f in ("--빠른", "--KV8", "--F2", "--F2B", "--CPU없이"):
        ap.add_argument(f, action="store_true")
    ap.add_argument("--jlens", help="서버를 띄운 J 파일(계약 5.4) — Jacobian lens 도 견준다")
    ap.add_argument("--출력")
    a = ap.parse_args()
    if a.jlens:
        J, _ = JL.읽기(a.jlens)
        a.J해시, a.J층수 = JL.해시(a.jlens), len(J)
    돌리기(a)


if __name__ == "__main__":
    main()
