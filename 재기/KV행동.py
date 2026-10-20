#!/usr/bin/env python3
"""GeulPUv2 ③b-1 행동 확인(문서/긴문맥.md — 기록용, 판정에 쓰지 않음): 디버거 시험 셋(시험/디버거/묶음50.jsonl)의 요청을 OpenAI 꼴
/v1/chat/completions 로 보내 답을 적고, 두 실행의 답이 같은 문항 수와 처음 갈라진 자리를 센다. GeulPU 서버 · llama-server 모두에
같은 요청(탐욕 · 셋의 max_tokens · cache_prompt 끔 — 앞부분 재사용이 llama.cpp 의 셈을 바꾸지 않게)을 보낸다. 표준 라이브러리만.

    python -I 재기/KV행동.py 묻기 <주소> <묶음.jsonl> <결과.json> [--이름 GeulPU-F1]
    python -I 재기/KV행동.py 견주기 <가.json> <나.json>
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
    줄들 = [json.loads(x) for x in open(a.묶음, encoding="utf-8") if x.strip()]
    답들 = []
    t0 = time.time()
    for i, 요청 in enumerate(줄들):
        몸 = {"messages": 요청["messages"], "max_tokens": 요청.get("max_tokens", 96), "temperature": 0, "cache_prompt": False}
        r = 보내기(a.주소, 몸)
        c = r["choices"][0]
        답들.append({"i": i, "답": c["message"].get("content") or "", "끝": c.get("finish_reason"),
                   "토큰": (r.get("usage") or {}).get("completion_tokens")})
        print(f"  {i + 1}/{len(줄들)} ({time.time() - t0:.0f} 초)", file=sys.stderr, flush=True)
    json.dump({"이름": a.이름, "주소": a.주소, "묶음": a.묶음, "답": 답들}, open(a.결과, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{a.이름}: {len(답들)} 문항 · {time.time() - t0:.0f} 초")


def 견주기(a):
    가, 나 = (json.load(open(p, encoding="utf-8")) for p in (a.가, a.나))
    같음, 다름 = 0, []
    for x, y in zip(가["답"], 나["답"]):
        if x["답"] == y["답"]:
            같음 += 1
            continue
        k = 0
        while k < min(len(x["답"]), len(y["답"])) and x["답"][k] == y["답"][k]:
            k += 1
        다름.append((x["i"], k, len(x["답"]), len(y["답"])))
    print(f"{가['이름']} ↔ {나['이름']}: 같은 답 {같음}/{len(가['답'])}")
    for i, k, la, lb in 다름:
        print(f"  문항 {i}: 처음 다른 글자 {k}(답 길이 {la} · {lb})")
    return 같음, 다름


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="일", required=True)
    p = sub.add_parser("묻기")
    p.add_argument("주소")
    p.add_argument("묶음")
    p.add_argument("결과")
    p.add_argument("--이름", default="")
    p = sub.add_parser("견주기")
    p.add_argument("가")
    p.add_argument("나")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"묻기": 묻기, "견주기": 견주기}[a.일](a)


if __name__ == "__main__":
    main()
