#!/usr/bin/env python3
"""MoE 재기 — 프록시 기록(요청 · 응답)을 엔진이 넣는 토큰 조각으로 바꾼다(llama-server 의 /apply-template · /tokenize).

  python 조각만들기.py <서버 주소> <기록.jsonl> <일감.json>

요청 i: P = 틀(요청), R = 틀(요청 + 그 응답의 assistant 메시지, add_generation_prompt 거짓), O = R − P(모델이 낸 글).
조각: 앞 상태 S 와 P 의 같은 앞부분 뒤만 prompt(같지 않으면 reset + reprocess 로 처음부터), 그다음 O 가 output.
llama.cpp 가 실제로 다시 셈한 양(timings 의 prompt_n · cache_n)도 함께 센다(문맥 저장소의 근거).
"""
import json
import sys
import urllib.request

URL = sys.argv[1].rstrip("/")


def 부르기(경로, 몸):
    r = urllib.request.Request(URL + 경로, json.dumps(몸).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())


def 토큰(글):
    return 부르기("/tokenize", {"content": 글, "add_special": False, "parse_special": True})["tokens"]


def 틀(몸, 생성프롬프트=True):
    b = {k: v for k, v in 몸.items() if k not in ("stream", "stream_options", "max_tokens", "n", "logprobs")}
    b["add_generation_prompt"] = 생성프롬프트
    return 부르기("/apply-template", b)["prompt"]


def 응답메시지(r):
    m = {"role": "assistant", "content": r["content"] or None}
    if r.get("reasoning_content"):
        m["reasoning_content"] = r["reasoning_content"]
    if r.get("tool_calls"):
        m["tool_calls"] = [{"id": t.get("id") or f"call_{k}", "type": "function",
                            "function": {"name": t["name"], "arguments": t["arguments"]}}
                           for k, t in sorted(r["tool_calls"].items(), key=lambda kv: int(kv[0]))]
    return m


조각, S = [], []
센 = {"requests": 0, "ideal_new": 0, "output": 0, "llama_prompt_n": 0, "llama_cache_n": 0, "reset": 0, "mismatch_out": 0}
for 줄 in open(sys.argv[2], encoding="utf-8"):
    j = json.loads(줄)
    if j["status"] != 200 or not j.get("request"):
        continue
    몸 = j["request"]
    P = 틀(몸)
    # llama.cpp 는 끝의 assistant 메시지를 '이어 쓰기'로 보아 도구 호출이 있으면 거절한다 — 뒤에 표시 메시지를 붙여 그리고 표시 앞에서 자른다
    am = 응답메시지(j["response"])
    표시 = "<<<표시-MoE재기>>>"
    if am.get("tool_calls"):
        뒤 = {"role": "tool", "tool_call_id": am["tool_calls"][0]["id"], "content": 표시}
    else:
        뒤 = {"role": "user", "content": 표시}
    R = 틀(dict(몸, messages=몸["messages"] + [am, 뒤]), False)
    k표 = R.rfind(표시)
    assert k표 > 0, "표시를 못 찾음"
    k표 = R.rfind("<|im_start|>", 0, k표)
    R = R[:k표]
    if not R.startswith(P):
        k = 0
        while k < min(len(P), len(R)) and P[k] == R[k]:
            k += 1
        센["mismatch_out"] += 1
        O = R[k:]
    else:
        O = R[len(P):]
    TP, TO = 토큰(P), 토큰(O)
    n = 0
    while n < min(len(S), len(TP)) and S[n] == TP[n]:
        n += 1
    i = j["i"]
    if n < len(S) - 64:
        조각.append({"tokens": TP[:n], "kind": "reprocess", "req": i, "reset": True})
        센["reset"] += 1
    elif n < len(S):
        센["tail_mismatch"] = 센.get("tail_mismatch", 0) + 1   # 끝 몇 토큰만 어긋남 — 다시 넣지 않고 이어 붙인다(위치가 몇 칸 밀림)
    if len(TP) > n:
        조각.append({"tokens": TP[n:], "kind": "prompt", "req": i})
    조각.append({"tokens": TO, "kind": "output", "req": i})
    S = TP + TO
    t = j.get("timings") or {}
    센["requests"] += 1
    센["ideal_new"] += len(TP) - n
    센["output"] += len(TO)
    센["llama_prompt_n"] += t.get("prompt_n") or 0
    센["llama_cache_n"] += t.get("cache_n") or 0
    print(i, "프롬프트", len(TP), "같은앞", n, "새", len(TP) - n, "출력", len(TO), "| llama.cpp 셈", t.get("prompt_n"),
          "재사용", t.get("cache_n"), flush=True)
json.dump({"spans": 조각, "counts": 센}, open(sys.argv[3], "w", encoding="utf-8"))
print(센)
