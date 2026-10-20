"""YaRN 식이 맞는가(GeulPUv2 ③a — 문서/긴문맥.md 의 맞음 2): 같은 토큰열(wikitext-2 test — GeulPU 의 /tokenize)을 여러 자리에서 끊어,
다음 토큰의 분포(상위 20 의 로그확률)를 GeulPU 와 llama-server 의 /v1/completions(토큰 번호 배열 · 최대 1 토큰 · 온도 0)로 받아 견준다.
엔진이 달라 비트는 같지 않다 — YaRN 을 켠 64k 의 차이가 YaRN 없는 16k 의 차이와 같은 급이면 식이 맞다. 표준 라이브러리만.

    python -I 재기/YaRN견주기.py 토큰 <GeulPU 주소> <wiki.test.raw> <토큰.json> [--자리 16000,64000] [--개수 20] [--걸음 37]
    python -I 재기/YaRN견주기.py 얻기 <주소> <토큰.json> <결과.json> [--자리 64000] [--이름 GeulPU]
    python -I 재기/YaRN견주기.py 견주기 <결과 a.json> <결과 b.json>

견주기: 자리마다 첫째 토큰 일치, a 의 상위 20 위의 KL(a‖b)(b 에 없는 토큰은 b 의 20 등 로그확률로 — 위쪽 어림), a 의 첫째 토큰의 로그확률 차.
"""
import argparse
import json
import math
import statistics
import sys
import urllib.request


def 보내기(주소, 길, 몸, 시간=3600):
    r = urllib.request.Request(주소.rstrip("/") + 길, data=json.dumps(몸, ensure_ascii=False).encode("utf-8"), method="POST",
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=시간).read())


def 토큰(a):
    글 = open(a.글, encoding="utf-8").read()
    t = 보내기(a.주소, "/tokenize", {"prompt": 글})["tokens"]
    print(f"토큰 {len(t):,}", flush=True)
    끝 = max(int(x) for x in a.자리.split(",")) + a.개수 * a.걸음 + 8
    json.dump({"토큰": t[:끝], "걸음": a.걸음, "개수": a.개수}, open(a.토큰, "w", encoding="utf-8"))


def 얻기(a):
    d = json.load(open(a.토큰, encoding="utf-8"))
    결과 = []
    for 자리 in [int(x) for x in a.자리.split(",")]:
        for k in range(d["개수"]):
            n = 자리 + k * d["걸음"]
            몸 = {"prompt": d["토큰"][:n], "max_tokens": 1, "temperature": 0, "logprobs": 20, "stream": False}
            r = 보내기(a.주소, "/v1/completions", 몸)
            lp = r["choices"][0].get("logprobs") or {}
            if lp.get("content"):                       # llama-server 의 꼴 — content[0].top_logprobs = [{id, token, logprob} …]
                top = {t["token"]: t["logprob"] for t in lp["content"][0].get("top_logprobs") or []}
            else:                                       # OpenAI 완성(옛 꼴) — top_logprobs[0] = {글: 로그확률}
                top = (lp.get("top_logprobs") or [{}])[0]
            결과.append({"자리": 자리, "n": n, "글": r["choices"][0].get("text"), "상위": top})
            print(f"{a.이름} n {n:,} — {r['choices'][0].get('text')!r} · 상위 {len(top)}", flush=True)
    json.dump({"이름": a.이름, "결과": 결과}, open(a.결과, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def 견주기(a):
    A, B = json.load(open(a.a, encoding="utf-8")), json.load(open(a.b, encoding="utf-8"))
    rb = {x["n"]: x for x in B["결과"]}
    묶 = {}
    for x in A["결과"]:
        y = rb.get(x["n"])
        if not y or not x["상위"] or not y["상위"]:
            continue
        p, q = x["상위"], y["상위"]
        q바닥 = min(q.values())
        첫a = max(p, key=p.get)
        첫b = max(q, key=q.get)
        kl = sum(math.exp(lp) * (lp - q.get(t, q바닥)) for t, lp in p.items())
        묶.setdefault(x["자리"], []).append((첫a == 첫b, kl, abs(p[첫a] - q.get(첫a, q바닥))))
    print(f"{A['이름']} 대 {B['이름']}")
    for 자리, xs in sorted(묶.items()):
        print(f"  자리 {자리:,}: {len(xs)} 곳 — 첫째 일치 {sum(1 for x in xs if x[0])}/{len(xs)} · KL 가운데값 {statistics.median(x[1] for x in xs):.4f} "
              f"(평균 {statistics.mean(x[1] for x in xs):.4f}) · 첫째 로그확률 차 가운데값 {statistics.median(x[2] for x in xs):.4f}")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="명령", required=True)
    t = sp.add_parser("토큰")
    t.add_argument("주소")
    t.add_argument("글")
    t.add_argument("토큰")
    t.add_argument("--자리", default="16000,64000")
    t.add_argument("--개수", type=int, default=20)
    t.add_argument("--걸음", type=int, default=37)
    g = sp.add_parser("얻기")
    g.add_argument("주소")
    g.add_argument("토큰")
    g.add_argument("결과")
    g.add_argument("--자리", default="64000")
    g.add_argument("--이름", default="GeulPU")
    c = sp.add_parser("견주기")
    c.add_argument("a")
    c.add_argument("b")
    a = p.parse_args()
    {"토큰": 토큰, "얻기": 얻기, "견주기": 견주기}[a.명령](a)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
