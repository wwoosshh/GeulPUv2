#!/usr/bin/env python3
"""모델 평가 1-1(문서/모델평가.md) — 서버의 geulpu.top_logprobs(상위 128 개까지)와 geulpu.token_ids_logprob(echo 의 위치마다 지정한
토큰의 로그확률)를 서버에 바로 묻는다. 서버를 --기록 폴더로 띄워 둔다(GPU 줄에 서서).

    python -I -X utf8 시험/평가서버.py run <주소> --log-dir <서버의 기록 폴더> --out <결과.json> [--max 64] [--name e11]
    python -I -X utf8 시험/평가서버.py compare <결과 a.json> <결과 b.json>        (백엔드를 건너 — CUDA ↔ Vulkan, 정밀 모드)

질문 셋마다(채팅 · 그리디 · 생각 끔):
  1. geulpu.top_logprobs 128 — 스텝마다 128 개가 기록한 로짓 행에서 계약 5.2 의 파이썬 참조(계약/로그확률.py)로 셈한 번호 · 값과 비트까지
     같다. OpenAI 호환의 top_logprobs(5)는 그 앞 5 개와 같다.
  2. geulpu.top_logprobs 없이 logprobs 20 — 토큰이 같고, 상위 20 개가 1 의 앞 20 개와 비트까지 같다(OpenAI 의 칸은 20 그대로).
  3. echo(프롬프트 + 1 의 답, echo_from = 프롬프트 길이) + geulpu.token_ids_logprob(위치마다 1 의 상위 64 개를 거꾸로 + 0 · V − 1 · 그
     위치의 토큰 — 끝 토큰의 위치는 셋만): 기록한 echo 행에서 파이썬 참조로 셈한 값과 비트까지 같고, 같은 위치의 prompt_top_logprobs(128)와 겹치는 토큰에서
     같고, 그 위치의 토큰은 token_logprobs 와 같다. prompt_top_ids 가 1 의 같은 스텝의 top_ids 와 같다(echo = 생성).
  4. 거절(400) 여덟 가지 — 채팅에 token_ids_logprob · 목록 수가 틀림 · 어휘 밖의 번호 · top_logprobs 129 · logprobs 없이 top_logprobs ·
     배열이 아님 · 한 위치에 257 개 · echo 없이.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "계약"))
import 로그확률 as LP          # noqa: E402

질문들 = ["프랑스의 수도는 어디인가요? 한 문장으로 답해 주세요.",
         "Write a haiku about the ocean.",
         "파이썬으로 리스트를 뒤집는 함수를 짧게 작성해 줘."]


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


def 비트(x):
    return LP.실수비트(float(x))


def 행읽기(경로, V):
    raw = open(경로, "rb").read()
    return [np.frombuffer(raw, np.float32, V, i * 4 * V) for i in range(len(raw) // (4 * V))]


def 채팅(a, i, 덧):
    몸 = {"messages": [{"role": "user", "content": 질문들[i]}], "max_tokens": a.max, "temperature": 0.0,
          "chat_template_kwargs": {"enable_thinking": False}}
    몸.update(덧)
    return 받기(a.url, "/v1/chat/completions", 몸)


def run(a):
    with urllib.request.urlopen(a.url + "/tokenizer_info", timeout=60) as r:
        V = json.loads(r.read().decode("utf-8"))["vocab_size"]
    합 = {k: 0 for k in ("steps", "answer_tokens", "top128_ref_diff", "openai5_diff", "tokens_diff", "top20_diff", "echo_rows", "ids_checked",
                         "ids_ref_diff", "ids_overlap_diff", "ids_target_diff", "echo_top_diff", "reject_fail")}
    결과 = {"url": a.url, "vocab": V, "requests": {}}
    for i in range(len(질문들)):
        이름 = f"{a.name}_{i}"
        # 1 — 상위 128
        r1 = 채팅(a, i, {"logprobs": True, "top_logprobs": 5,
                        "geulpu": {"top_logprobs": 128, "prompt_tokens": True, "record": 이름 + "_g"}})
        g1 = r1["geulpu"]
        n = len(g1["top_ids"])
        합["steps"] += n
        합["answer_tokens"] += len(g1["tokens"])
        행들 = 행읽기(os.path.join(a.log_dir, 이름 + "_g_로짓.bin"), V)
        내용 = r1["choices"][0]["logprobs"]["content"]
        for s in range(n):
            x = LP.로그확률행(행들[s], 위수=128)
            if [j for j, _ in x["위"]] != g1["top_ids"][s] or \
                    [LP.실수비트(v) for _, v in x["위"]] != [비트(v) for v in g1["top_logprobs"][s]]:
                합["top128_ref_diff"] += 1
            if s < len(내용) and [비트(t["logprob"]) for t in 내용[s]["top_logprobs"]] != [비트(v) for v in g1["top_logprobs"][s][:5]]:
                합["openai5_diff"] += 1
        # 2 — logprobs 20 만(새 칸 없이)
        r2 = 채팅(a, i, {"logprobs": True, "top_logprobs": 20, "geulpu": {}})
        g2 = r2["geulpu"]
        if g2["tokens"] != g1["tokens"]:
            합["tokens_diff"] += 1
        for s in range(min(n, len(g2["top_ids"]))):
            if g2["top_ids"][s] != g1["top_ids"][s][:20] or \
                    [비트(v) for v in g2["top_logprobs"][s]] != [비트(v) for v in g1["top_logprobs"][s][:20]]:
                합["top20_diff"] += 1
        # 3 — echo 의 지정한 토큰. 점수를 매기는 위치는 답의 토큰 모두(끝 토큰 포함 — 생성의 로그확률에는 끝 토큰이 없다)
        P = len(g1["prompt_tokens"])
        ids = g1["prompt_tokens"] + g1["tokens"]
        목록 = [(list(reversed(g1["top_ids"][s][:64])) if s < n else []) + [0, V - 1, g1["tokens"][s]] for s in range(len(g1["tokens"]))]
        r3 = 받기(a.url, "/v1/completions", {"prompt": ids, "echo": True, "max_tokens": 0, "logprobs": 20, "temperature": 0.0,
                                               "geulpu": {"echo_from": P, "top_logprobs": 128, "token_ids_logprob": 목록,
                                                          "record": 이름 + "_e"}})
        g3 = r3["geulpu"]
        e행 = 행읽기(os.path.join(a.log_dir, 이름 + "_e_프롬로짓.bin"), V)
        tl = r3["choices"][0]["logprobs"]["token_logprobs"]
        합["echo_rows"] += len(e행)
        for s in range(len(e행)):
            x = LP.로그확률행(e행[s])
            got = [비트(v) for v in g3["token_ids_logprob"][s]]
            want = [LP.실수비트(float(x["로그확률"][t])) for t in 목록[s]]
            합["ids_checked"] += len(want)
            합["ids_ref_diff"] += sum(1 for p, q in zip(got, want) if p != q) + abs(len(got) - len(want))
            위 = dict(zip(g3["prompt_top_ids"][P + s], [비트(v) for v in g3["prompt_top_logprobs"][P + s]]))
            합["ids_overlap_diff"] += sum(1 for t, v in zip(목록[s], got) if t in 위 and 위[t] != v)
            if not got or got[-1] != 비트(tl[P + s]):
                합["ids_target_diff"] += 1
            if s < n and g3["prompt_top_ids"][P + s] != g1["top_ids"][s]:
                합["echo_top_diff"] += 1
        결과["requests"][이름] = {"tokens": g1["tokens"], "top_ids": g1["top_ids"],
                                 "top_logprobs": [[비트(v) for v in row] for row in g1["top_logprobs"]],
                                 "token_ids_logprob": [[비트(v) for v in row] for row in g3["token_ids_logprob"]]}
        print(f"{이름}: 스텝 {n} · echo 행 {len(e행)}", flush=True)
    # 4 — 거절
    완 = {"prompt": ids, "echo": True, "max_tokens": 0, "logprobs": 1}
    거절 = [
        ("chat", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "logprobs": True,
                                          "geulpu": {"token_ids_logprob": [[1]]}}),
        ("rows", "/v1/completions", dict(완, geulpu={"echo_from": P, "token_ids_logprob": 목록[:-1]})),
        ("range", "/v1/completions", dict(완, geulpu={"echo_from": P, "token_ids_logprob": [[V]] + 목록[1:]})),
        ("top129", "/v1/completions", dict(완, geulpu={"echo_from": P, "top_logprobs": 129})),
        ("no_logprobs", "/v1/completions", {"prompt": ids, "echo": True, "max_tokens": 0, "geulpu": {"echo_from": P, "top_logprobs": 64}}),
        ("not_array", "/v1/completions", dict(완, geulpu={"echo_from": P, "token_ids_logprob": 5})),
        ("257", "/v1/completions", dict(완, geulpu={"echo_from": P, "token_ids_logprob": [list(range(257))] + 목록[1:]})),
        ("no_echo", "/v1/completions", {"prompt": ids, "max_tokens": 1, "logprobs": 1, "geulpu": {"echo_from": P, "token_ids_logprob": 목록}}),
    ]
    for 이름, 길, 몸 in 거절:
        s, _ = 묻기(a.url, 길, 몸)
        if s != 400:
            합["reject_fail"] += 1
            print(f"거절 {이름}: HTTP {s}(400 이어야 한다)", flush=True)
    결과["summary"] = 합
    json.dump(결과, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps(합, ensure_ascii=False))
    통과 = 합["steps"] > 0 and 합["echo_rows"] == 합["answer_tokens"] and all(
        합[k] == 0 for k in ("top128_ref_diff", "openai5_diff", "tokens_diff", "top20_diff", "ids_ref_diff", "ids_overlap_diff",
                             "ids_target_diff", "echo_top_diff", "reject_fail"))
    print("모델 평가 1-1(이 백엔드):", "통과" if 통과 else "못 미침")
    return 0 if 통과 else 1


def compare(a):
    x = json.load(open(a.a, encoding="utf-8"))["requests"]
    y = json.load(open(a.b, encoding="utf-8"))["requests"]
    다름 = [(이름, 칸) for 이름 in x for 칸 in ("tokens", "top_ids", "top_logprobs", "token_ids_logprob")
            if 이름 not in y or x[이름][칸] != y[이름][칸]]
    print(f"백엔드를 건너: 요청 {len(x)} · 다른 칸 {len(다름)}" + (f" — {다름[:6]}" if 다름 else ""))
    return 0 if not 다름 and len(x) == len(y) else 1


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("url")
    p.add_argument("--log-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max", type=int, default=64)
    p.add_argument("--name", default="e11")
    q = sub.add_parser("compare")
    q.add_argument("a")
    q.add_argument("b")
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
