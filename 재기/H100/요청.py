"""서버 GPU 의 속도 측정(문서/H100.md) — 세 엔진(GeulPU · llama.cpp · vLLM)에 똑같은 HTTP 요청을 보낸다.

  python3 재기/H100/요청.py --url http://127.0.0.1:44700 --tag 이름 --mode single|conc --out results.jsonl [--conc 1,4,8] [--reps 3] [--kind llama]

요청: OpenAI 호환 /v1/completions, 토큰 번호 프롬프트, 그리디(temperature 0), ignore_eos(정해진 수만큼 생성).
프롬프트는 요청마다 무작위로 새로 만든 보통 토큰 번호다(첫 토큰부터 달라 어느 엔진의 접두 캐시도 맞지 않는다).
  single: 빈 캐시 프롬프트 처리 — L 토큰 + max_tokens 1 의 시간(L = 512 · 2048 · 8192 · 30720).
          생성 — 깊이 D(64 · 8192 · 30720)에서 (D 토큰 + 257 토큰의 시간 − D 토큰 + 1 토큰의 시간) 으로 256 토큰의 생성 속도.
  conc:   동시 c 요청(각 512 토큰 프롬프트 + 256 토큰 생성)의 전체 생성 토큰 / 벽시계 시간.
값마다 --reps 번 측정한 중앙값. 처음에 덥히기 요청 하나. 결과는 JSONL 한 줄씩(태그 · 일 · 값 · 실제 생성 토큰 수).
표준 라이브러리만 쓴다.
"""
import argparse
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request

LO, HI = 1000, 100000          # 보통 토큰 번호(Qwen 의 특수 토큰 151643 부터를 피한다)
_rng = random.Random(20261027)
_lock = threading.Lock()


def 프롬프트(n):
    with _lock:
        return [_rng.randrange(LO, HI) for _ in range(n)]


def 보내기(a, n_prompt, n_new):
    body = {"model": a.model, "prompt": 프롬프트(n_prompt), "max_tokens": n_new, "temperature": 0, "ignore_eos": True, "stream": False}
    if a.kind == "llama":
        body["cache_prompt"] = False
    r = urllib.request.Request(a.url + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=3600) as f:
        d = json.load(f)
    dt = time.perf_counter() - t0
    u = d.get("usage") or {}
    n = u.get("completion_tokens")
    if n is None and isinstance((d.get("geulpu") or {}).get("tokens"), list):
        n = len(d["geulpu"]["tokens"])
    return dt, n


def 적기(a, rec):
    rec = {"tag": a.tag, "mode": a.mode, **rec, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(a.out, "a", encoding="utf-8") as w:
        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def single(a):
    for L in a.prefill:
        ts = [보내기(a, L, 1)[0] for _ in range(a.reps)]
        m = statistics.median(ts)
        적기(a, {"work": "prefill", "tokens": L, "times": ts, "median_s": m, "tok_per_s": L / m})
    for D in a.depths:
        t1, tn, ns = [], [], []
        for _ in range(a.reps):
            t1.append(보내기(a, D, 1)[0])
            dt, n = 보내기(a, D, a.new + 1)
            tn.append(dt)
            ns.append(n)
        gen = (min(x for x in ns if x is not None) - 1) if any(x is not None for x in ns) else a.new
        m1, mn = statistics.median(t1), statistics.median(tn)
        적기(a, {"work": "decode", "depth": D, "new": gen, "t_first": t1, "t_all": tn, "completion_tokens": ns,
                 "tok_per_s": gen / (mn - m1) if mn > m1 else None, "ms_per_tok": 1000 * (mn - m1) / gen if gen else None})


def conc(a):
    for c in a.conc:
        walls, toks = [], []
        for _ in range(a.reps):
            out = [None] * c

            def 일(i):
                out[i] = 보내기(a, a.conc_prompt, a.new)
            th = [threading.Thread(target=일, args=(i,)) for i in range(c)]
            t0 = time.perf_counter()
            for t in th:
                t.start()
            for t in th:
                t.join()
            walls.append(time.perf_counter() - t0)
            toks.append(sum((n if n is not None else a.new) for _, n in out))
        i = sorted(range(a.reps), key=lambda k: walls[k])[a.reps // 2]
        적기(a, {"work": "conc", "concurrency": c, "prompt": a.conc_prompt, "new": a.new, "walls": walls, "tokens": toks,
                 "tok_per_s": toks[i] / walls[i]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=("single", "conc"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="m")
    ap.add_argument("--kind", default="")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--prefill", default="512,2048,8192,30720")
    ap.add_argument("--depths", default="64,8192,30720")
    ap.add_argument("--conc", default="1,4,8")
    ap.add_argument("--conc-prompt", dest="conc_prompt", type=int, default=512)
    a = ap.parse_args()
    a.prefill = [int(x) for x in a.prefill.split(",") if x]
    a.depths = [int(x) for x in a.depths.split(",") if x]
    a.conc = [int(x) for x in a.conc.split(",") if x]
    보내기(a, 128, 8)                                      # 덥히기
    single(a) if a.mode == "single" else conc(a)


if __name__ == "__main__":
    main()
