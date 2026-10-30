#!/usr/bin/env python3
"""상태 스냅숏 시험(2026-10-31) — 상태가 있는 층의 구조(Qwen3.5 · 구조표의 스냅숏됨)에서 추론 엔진이 프롬프트의 마지막 행 앞(위치
n − 1)의 상태를 남겨, 새 요청의 앞이 슬롯의 끝보다 먼저 갈려도 그 위치에서 잇는지, 그리고 이은 답이 캐시 없이 처음부터 셈한 답과 같은
비트인지 본다. 표준 라이브러리만.

    python -I -X utf8 시험/스냅숏서버.py <서버.exe> <모델.gguf> <모듈.ptx> <작업폴더> [--포트 8131] [--문맥 4096] [--길이 700] [--번 2]

정한 무작위 토큰의 프롬프트(/v1/completions — temperature 0 · ignore_eos · 생성 24)를 슬롯 하나의 서버에 차례로 보낸다:
  A  P                          빈 슬롯 — 캐시 0
  B  P 다시                      같은 프롬프트 — 스냅숏(len(P) − 1)에서
  C  P + B 의 답 앞 5 + X        슬롯의 끝(먹인 답의 끝)보다 앞에서 갈림 — 스냅숏(len(P) − 1)에서
  D  C + C 의 답 + Y             슬롯의 끝까지 이음 — 지금 상태(캐시 = 슬롯이 가진 토큰 모두)
  E  P 앞 300 + Z                스냅숏(len(D) − 1)보다 앞에서 갈림 — 캐시 0
같은 요청을 다른 서버(같은 모델 · 모듈, 모두 cache_prompt false)에도 보내 뽑은 토큰 · 걸음마다의 로짓 해시가 모두 같은지와 캐시 토큰이 위와
같은지 본다. A … E 는 --번 차례 되풀이한다(차례마다 새 토큰 — 첫 차례의 시간은 처음 쓰는 행 수의 판 고르기를 품는다).
끝 줄: '스냅숏 서버: 모두 같음' 또는 '스냅숏 서버: 다름 N'(종료 1).
"""
import argparse
import http.client
import json
import os
import random
import subprocess
import sys
import time


def 요청(포트, 방법, 길, 몸=None, 시간=3600):
    c = http.client.HTTPConnection("127.0.0.1", 포트, timeout=시간)
    본 = None if 몸 is None else json.dumps(몸, ensure_ascii=False).encode("utf-8")
    c.request(방법, 길, body=본, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    d = r.read()
    c.close()
    try:
        return r.status, json.loads(d)
    except ValueError:
        return r.status, d


class 서버:
    def __init__(self, exe, gguf, 모듈, 포트, 문맥, 로그):
        self.포트 = 포트
        인자 = [os.path.abspath(exe), gguf, "--ptx", 모듈, "--포트", str(포트), "--슬롯", "1", "--문맥", str(문맥)]
        self.로그 = open(로그, "w", encoding="utf-8")
        self.p = subprocess.Popen(인자, stdout=self.로그, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise SystemExit(f"서버가 멈췄다(종료 {self.p.returncode}) — {로그}")
            try:
                s, _ = 요청(포트, "GET", "/health", 시간=5)
                if s == 200:
                    return
            except OSError:
                pass
            time.sleep(0.5)
        raise SystemExit(f"서버가 뜨지 않는다 — {로그}")

    def 끄기(self):
        self.p.kill()
        self.p.wait()
        self.로그.close()


def 완성(포트, 프롬프트, 캐시):
    s, d = 요청(포트, "POST", "/v1/completions", {"prompt": 프롬프트, "temperature": 0, "max_tokens": 24, "ignore_eos": True,
                                                 "cache_prompt": 캐시, "geulpu": {"logit_hash": True}})
    if s != 200:
        raise SystemExit(f"요청 실패 {s}: {d}")
    g = d["geulpu"]
    return {"tokens": g["tokens"], "hashes": g.get("logit_hashes"), "cached": g["prompt_tokens_cached"],
            "first_token_us": g.get("first_token_ms")}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="상태 스냅숏 시험(서버)")
    p.add_argument("exe")
    p.add_argument("gguf")
    p.add_argument("ptx")
    p.add_argument("work")
    p.add_argument("--포트", type=int, default=8131)
    p.add_argument("--문맥", type=int, default=4096)
    p.add_argument("--길이", type=int, default=700)
    p.add_argument("--번", type=int, default=2, help="A … E 를 되풀이하는 차례(차례마다 새 무작위 토큰 — 첫 차례는 판 고르기 · 데우기를 품는다)")
    a = p.parse_args()
    os.makedirs(a.work, exist_ok=True)
    r = random.Random(31)
    무작위 = lambda n: [r.randrange(1000, 150000) for _ in range(n)]
    재료 = [(무작위(a.길이), 무작위(40), 무작위(30), 무작위(100)) for _ in range(a.번)]

    # 시험 서버(cache_prompt true) — 프롬프트 C · D 는 앞 요청의 답으로 짓는다
    시험 = [{} for _ in range(a.번)]
    프롬 = [{} for _ in range(a.번)]
    srv = 서버(a.exe, a.gguf, a.ptx, a.포트, a.문맥, os.path.join(a.work, "서버_시험.txt"))
    try:
        for i, (P, X, Y, Z) in enumerate(재료):
            t, q = 시험[i], 프롬[i]
            q["A"] = q["B"] = P
            t["A"] = 완성(a.포트, q["A"], True)
            t["B"] = 완성(a.포트, q["B"], True)
            q["C"] = P + t["B"]["tokens"][:5] + X
            t["C"] = 완성(a.포트, q["C"], True)
            q["D"] = q["C"] + t["C"]["tokens"] + Y
            t["D"] = 완성(a.포트, q["D"], True)
            q["E"] = P[:300] + Z
            t["E"] = 완성(a.포트, q["E"], True)
    finally:
        srv.끄기()
    # 참조 서버 — 같은 프롬프트를 모두 캐시 없이(처음부터)
    참조 = [{} for _ in range(a.번)]
    srv = 서버(a.exe, a.gguf, a.ptx, a.포트, a.문맥, os.path.join(a.work, "서버_참조.txt"))
    try:
        for i in range(a.번):
            for k in "ABCDE":
                참조[i][k] = 완성(a.포트, 프롬[i][k], False)
    finally:
        srv.끄기()

    다름 = 0
    for i in range(a.번):
        t0, q = 시험[i], 프롬[i]
        n, nC, gC = len(q["A"]), len(q["C"]), len(t0["C"]["tokens"])
        기대 = {"A": (0, 0), "B": (n - 1, n - 1), "C": (n - 1, n - 1), "D": (nC + gC - 1, nC + gC), "E": (0, 0)}
        print(f"차례 {i + 1}")
        for k in "ABCDE":
            t, c = t0[k], 참조[i][k]
            같은비트 = t["tokens"] == c["tokens"] and t["hashes"] == c["hashes"] and t["hashes"] is not None
            lo, hi = 기대[k]
            캐시맞음 = lo <= t["cached"] <= hi and c["cached"] == 0
            다름 += (not 같은비트) + (not 캐시맞음)
            print(f"  {k}: 프롬프트 {len(q[k])} · 캐시 {t['cached']}(기대 {lo}{'' if lo == hi else f' ~ {hi}'}) · 토큰 · 해시 "
                  f"{'같음' if 같은비트 else '다름'} · 첫 토큰 {t['first_token_us'] / 1000:.1f} ms(캐시 없이 {c['first_token_us'] / 1000:.1f} ms)")
    with open(os.path.join(a.work, "결과.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"모델": a.gguf, "모듈": a.ptx, "시험": 시험, "참조": 참조,
                   "프롬프트길이": [{k: len(v) for k, v in q.items()} for q in 프롬]}, f, ensure_ascii=False, indent=1)
    print("스냅숏 서버: " + ("모두 같음" if not 다름 else f"다름 {다름}"))
    return 0 if not 다름 else 1


if __name__ == "__main__":
    sys.exit(main())
