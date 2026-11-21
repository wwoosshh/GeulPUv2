#!/usr/bin/env python3
"""로짓 행(계약 5.2 — geulpu.logits) — 서버가 응답에 싣는 로짓 행(logits · prompt_logits)을 서버에 바로 묻는다. 서버를 --기록 폴더로
띄워 둔다(GPU 줄에 서서). 표준 라이브러리만 쓴다.

    python -I -X utf8 시험/로짓행서버.py <주소> --log-dir <서버의 기록 폴더> [--max 16] [--name lg]

완성(/v1/completions) · 그리디로:
  1. 생성 + geulpu.logits: logits 가 스텝 수만큼이고, 행마다 base64 를 풀면 fp32 V 개(리틀 엔디언)이며 기록한 로짓 행(_로짓.bin)과
     바이트가 같고, 그 행의 FNV-1a 64 가 logit_hashes 와 같다.
  2. 같은 요청을 geulpu.logits 없이: 토큰 · 로짓 해시 · 로그확률이 1 과 같고(로짓을 내도 결과는 그대로) 응답에 logits 가 없다.
  3. echo(프롬프트 + 1 의 답, echo_from = 프롬프트 길이, max_tokens 0) + geulpu.logits: prompt_logits 가 점수를 매긴 행 수만큼이고
     기록(_프롬로짓.bin) · prompt_logit_hashes 와 같으며, 1 의 logits 와 행마다 같은 바이트다(echo = 생성). logits 는 빈 배열.
  4. stream + geulpu.logits(해시 · 로그확률 없이): 마지막 청크의 geulpu.logits 가 1 과 같다.
  5. 한도(한 요청에 64 행 — echo 가 점수를 매기는 행 + max_tokens): 꼭 맞는 64 행은 받고 행 수 · 해시가 맞다(끝 토큰 무시). 65 행이면
     400, max_tokens 를 주지 않아도 · max_tokens 65 도 400.
"""
import argparse
import base64
import json
import os
import struct
import sys
import urllib.error
import urllib.request

프롬프트글 = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
한도 = 64


def 묻기(주소, 길, 몸):
    q = urllib.request.Request(주소 + 길, data=json.dumps(몸).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(q, timeout=600) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def 받기(주소, 길, 몸):
    s, j = 묻기(주소, 길, 몸)
    if s != 200:
        raise SystemExit(f"{길} → HTTP {s}: {str(j)[:300]}")
    return j


def 스트리밍받기(주소, 길, 몸):
    """스트리밍 응답의 청크들 가운데 geulpu 가 든 것(마지막 청크)."""
    q = urllib.request.Request(주소 + 길, data=json.dumps(dict(몸, stream=True)).encode(), headers={"Content-Type": "application/json"})
    g = None
    with urllib.request.urlopen(q, timeout=600) as r:
        for 줄 in r:
            줄 = 줄.strip()
            if not 줄.startswith(b"data: ") or 줄 == b"data: [DONE]":
                continue
            사건 = json.loads(줄[6:].decode("utf-8"))
            if "geulpu" in 사건:
                g = 사건["geulpu"]
    return g


def fnv(행):
    h = 0xCBF29CE484222325
    for x in struct.unpack(f"<{len(행) // 4}I", 행):
        h = ((h ^ x) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def 풀기(목록):
    return [base64.b64decode(s, validate=True) for s in 목록]


def 기록행(경로, V):
    raw = open(경로, "rb").read()
    return [raw[i:i + 4 * V] for i in range(0, len(raw), 4 * V)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--max", type=int, default=16)
    ap.add_argument("--name", default="lg")
    a = ap.parse_args()
    with urllib.request.urlopen(a.url + "/tokenizer_info", timeout=60) as r:
        V = json.loads(r.read().decode("utf-8"))["vocab_size"]
    못미침 = []

    def 본다(이름, 참, 덧=""):
        print(("  맞음  " if 참 else "  다름  ") + 이름 + (f" — {덧}" if 덧 else ""), flush=True)
        if not 참:
            못미침.append(이름)

    # 1 — 생성
    몸 = {"prompt": 프롬프트글, "max_tokens": a.max, "temperature": 0.0, "logprobs": 5,
          "geulpu": {"logits": True, "logit_hash": True, "prompt_tokens": True, "record": a.name + "_g"}}
    r1 = 받기(a.url, "/v1/completions", 몸)
    g1 = r1["geulpu"]
    행1 = 풀기(g1["logits"])
    n = len(g1["tokens"])
    P = len(g1["prompt_tokens"])
    print(f"1. 생성: 프롬프트 {P} 토큰 · 스텝 {n} · 어휘 {V} · 응답의 로짓 행 {len(행1)}")
    본다("행 수 = 스텝 수 = 로짓 해시 수", len(행1) == n == len(g1["logit_hashes"]) and n > 0)
    본다("행마다 fp32 V 개", all(len(x) == 4 * V for x in 행1))
    기록1 = 기록행(os.path.join(a.log_dir, a.name + "_g_로짓.bin"), V)
    본다("기록한 로짓 행과 같은 바이트", 기록1 == 행1)
    본다("행의 FNV-1a 64 = logit_hashes", [fnv(x) for x in 행1] == g1["logit_hashes"])
    본다("prompt_logits 는 없다(echo 가 아님)", "prompt_logits" not in g1)

    # 2 — geulpu.logits 없이
    몸2 = dict(몸, geulpu={"logit_hash": True, "prompt_tokens": True})
    r2 = 받기(a.url, "/v1/completions", 몸2)
    g2 = r2["geulpu"]
    print("2. 같은 요청을 geulpu.logits 없이")
    본다("토큰 · 로짓 해시가 같다", g2["tokens"] == g1["tokens"] and g2["logit_hashes"] == g1["logit_hashes"])
    본다("로그확률 · 상위 후보가 같다", r2["choices"][0]["logprobs"] == r1["choices"][0]["logprobs"]
         and g2["top_logprobs"] == g1["top_logprobs"] and g2["entropy"] == g1["entropy"])
    본다("응답에 logits 가 없다", "logits" not in g2)

    # 3 — echo
    전체 = g1["prompt_tokens"] + g1["tokens"]
    몸3 = {"prompt": 전체, "echo": True, "max_tokens": 0, "temperature": 0.0, "logprobs": 5,
          "geulpu": {"echo_from": P, "logits": True, "logit_hash": True, "record": a.name + "_e"}}
    g3 = 받기(a.url, "/v1/completions", 몸3)["geulpu"]
    행3 = 풀기(g3["prompt_logits"])
    print(f"3. echo: 점수를 매긴 행 {len(행3)}")
    본다("행 수 = 프롬프트 길이 − echo_from = 해시 수", len(행3) == n == len(g3["prompt_logit_hashes"]))
    본다("기록한 프롬프트 로짓 행과 같은 바이트", 기록행(os.path.join(a.log_dir, a.name + "_e_프롬로짓.bin"), V) == 행3)
    본다("행의 FNV-1a 64 = prompt_logit_hashes", [fnv(x) for x in 행3] == g3["prompt_logit_hashes"])
    본다("생성의 행과 같은 바이트(echo = 생성)", 행3 == 행1)
    본다("logits 는 빈 배열(max_tokens 0)", g3["logits"] == [])

    # 4 — stream
    g4 = 스트리밍받기(a.url, "/v1/completions", {"prompt": 프롬프트글, "max_tokens": a.max, "temperature": 0.0, "geulpu": {"logits": True}})
    print("4. stream(해시 · 로그확률 없이)")
    본다("마지막 청크의 logits 가 1 과 같다", g4 is not None and 풀기(g4.get("logits", [])) == 행1)

    # 5 — 한도
    print(f"5. 한도 {한도} 행")
    매길 = len(전체) - P                                  # echo 가 점수를 매기는 행 = 프롬프트 길이 − echo_from
    몸5 = {"prompt": 전체, "echo": True, "max_tokens": 한도 - 매길, "temperature": 0.0, "logprobs": 1, "ignore_eos": True,
          "geulpu": {"echo_from": P, "logits": True, "logit_hash": True}}
    s5, r5 = 묻기(a.url, "/v1/completions", 몸5)
    if s5 != 200:
        본다(f"꼭 맞는 {한도} 행을 받는다", False, f"HTTP {s5}: {str(r5)[:200]}")
    else:
        g5 = r5["geulpu"]
        p행, s행 = 풀기(g5["prompt_logits"]), 풀기(g5["logits"])
        본다(f"꼭 맞는 {한도} 행을 받는다(echo {len(p행)} + 생성 {len(s행)})", len(p행) == 매길 and len(s행) == 한도 - 매길)
        본다("그 행들의 해시가 맞다", [fnv(x) for x in p행] == g5["prompt_logit_hashes"] and [fnv(x) for x in s행] == g5["logit_hashes"])
    몸5["max_tokens"] = 한도 - 매길 + 1
    s6, r6 = 묻기(a.url, "/v1/completions", 몸5)
    본다(f"echo 의 행 + max_tokens 가 {한도 + 1} 이면 400", s6 == 400 and "geulpu.logits" in str(r6), f"HTTP {s6}")
    for 이름, 덧 in (("max_tokens 를 주지 않으면 400", {}), (f"max_tokens 가 {한도 + 1} 이면 400", {"max_tokens": 한도 + 1})):
        s7, r7 = 묻기(a.url, "/v1/completions", dict({"prompt": 프롬프트글, "temperature": 0.0, "geulpu": {"logits": True}}, **덧))
        본다(이름, s7 == 400 and "geulpu.logits" in str(r7), f"HTTP {s7}")

    print("로짓 행(이 백엔드):", "통과" if not 못미침 else f"못 미침 — {못미침}")
    return 0 if not 못미침 else 1


if __name__ == "__main__":
    sys.exit(main())
