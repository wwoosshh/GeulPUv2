#!/usr/bin/env python3
"""GeulPUv2 ③b-1 행동 확인(문서/긴문맥.md — 기록용, 판정에 쓰지 않음): 디버거 테스트 세트(시험/디버거/묶음50.jsonl)의 요청을 OpenAI 형식
/v1/chat/completions 로 보내 답을 적고, 두 실행의 답이 같은 문항 수와 처음 갈라진 위치를 센다. GeulPU 서버 · llama-server 모두에
같은 요청(그리디 디코딩 · 세트의 max_tokens · cache_prompt 끔 — 프롬프트 재사용이 llama.cpp 의 계산을 바꾸지 않게)을 보낸다. 표준 라이브러리만.

    python -I 재기/KV행동.py ask <url> <set.jsonl> <out.json> [--name GeulPU-F1]
    python -I 재기/KV행동.py compare <a.json> <b.json>
"""
import argparse
import json
import sys
import time
import urllib.request


def 보내기(주소, 몸, 시간=600):
    r = urllib.request.Request(주소.rstrip("/") + "/v1/chat/completions", data=json.dumps(몸, ensure_ascii=False).encode("utf-8"),
                               method="POST", headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=시간).read())


def 묻기(a):
    줄들 = [json.loads(x) for x in open(a.set, encoding="utf-8") if x.strip()]
    답들 = []
    t0 = time.time()
    for i, 요청 in enumerate(줄들):
        몸 = {"messages": 요청["messages"], "max_tokens": 요청.get("max_tokens", 96), "temperature": 0, "cache_prompt": False}
        r = 보내기(a.url, 몸)
        c = r["choices"][0]
        답들.append({"i": i, "text": c["message"].get("content") or "", "finish": c.get("finish_reason"),
                   "tokens": (r.get("usage") or {}).get("completion_tokens")})
        print(f"  {i + 1}/{len(줄들)} ({time.time() - t0:.0f} 초)", file=sys.stderr, flush=True)
    json.dump({"name": a.name, "url": a.url, "set": a.set, "answers": 답들}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{a.name}: {len(답들)} 문항 · {time.time() - t0:.0f} 초")


def 견주기(a):
    가, 나 = (json.load(open(p, encoding="utf-8")) for p in (a.a, a.b))
    같음, 다름 = 0, []
    for x, y in zip(가["answers"], 나["answers"]):
        if x["text"] == y["text"]:
            같음 += 1
            continue
        k = 0
        while k < min(len(x["text"]), len(y["text"])) and x["text"][k] == y["text"][k]:
            k += 1
        다름.append((x["i"], k, len(x["text"]), len(y["text"])))
    print(f"{가['name']} ↔ {나['name']}: 같은 답 {같음}/{len(가['answers'])}")
    for i, k, la, lb in 다름:
        print(f"  문항 {i}: 처음 다른 글자 {k}(답 길이 {la} · {lb})")
    return 같음, 다름


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ask")
    p.add_argument("url")
    p.add_argument("set")
    p.add_argument("out")
    p.add_argument("--name", default="")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"ask": 묻기, "compare": 견주기}[a.command](a)


if __name__ == "__main__":
    main()
