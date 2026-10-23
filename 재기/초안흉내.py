#!/usr/bin/env python3
"""1.0.0 1-2b 앞 — 초안 모델 추측의 받아들임 흉내(문서/0.11.0.md). 큰 모델의 실제 답(재기/에이전트재생.py --토큰 의 결과)을 작은 모델에
echo 로 점수 매겨, 위치마다 작은 모델의 1위 토큰이 큰 모델의 토큰과 같은지 본다. 큰 모델의 앞부분이 맞는 동안 작은 모델이 탐욕으로 이어
쓰는 초안은 그 1위와 같으므로, 스텝마다 받는 초안의 길이를 정확히 셀 수 있다.

  python -I -X utf8 재기/초안흉내.py <재생 결과.json> <작은 모델 서버 url> <출력.json> [--t1 ms] [--td ms] [--c 표]

- 정책 셋을 같은 답에 흉내 낸다: (가) 추측 없음, (나) 프롬프트 찾기(지금의 3단계 규칙 — 끝의 n-gram n = 3 → 2, 앞에 나온 마지막 자리의
  뒤를 초안으로, 행 8), (다) 초안 모델(작은 모델의 탐욕 초안, 최대 K, 작은 모델의 1위 확률이 문턱 아래면 멈춤), (라) 둘을 섞음(프롬프트
  찾기가 초안을 내면 그것, 아니면 초안 모델).
- 비용: 큰 모델의 행 B 스텝 = c(B) · t1, 작은 모델의 토큰 하나 = td. 기본 t1 · td · c 는 재서 넣는다(--t1 · --td · --c "1:1,2:1.03,…").
"""
import json
import math
import sys
import urllib.request


def 인자():
    a = sys.argv[1:]
    opt = {"t1": 12.0, "td": 2.0, "c": "1:1,2:1.03,3:1.15,4:1.16,5:1.17,6:1.18,7:1.19,8:1.21,16:1.45", "K": 8, "p": 0.3}
    pos = []
    i = 0
    while i < len(a):
        if a[i] in ("--t1", "--td", "--p"):
            opt[a[i][2:]] = float(a[i + 1]); i += 2
        elif a[i] == "--c":
            opt["c"] = a[i + 1]; i += 2
        elif a[i] == "--K":
            opt["K"] = int(a[i + 1]); i += 2
        else:
            pos.append(a[i]); i += 1
    return pos, opt


def c표(s):
    pts = sorted((int(x.split(":")[0]), float(x.split(":")[1])) for x in s.split(","))

    def c(B):
        for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
            if b0 <= B <= b1:
                return v0 + (v1 - v0) * (B - b0) / (b1 - b0)
        return pts[-1][1] * B / pts[-1][0] if B > pts[-1][0] else pts[0][1]
    return c


def 점수(url, P, Y):
    몸 = {"prompt": P + Y, "echo": True, "max_tokens": 0, "logprobs": 1, "temperature": 0, "geulpu": {"echo_from": len(P)}}
    rq = urllib.request.Request(url + "/v1/completions", data=json.dumps(몸).encode(), method="POST",
                                headers={"Content-Type": "application/json"})
    j = json.load(urllib.request.urlopen(rq, timeout=3600))
    g = j["geulpu"]
    top = g["prompt_top_ids"][len(P):]
    tlp = g["prompt_top_logprobs"][len(P):]
    return [t[0] if t else -1 for t in top], [math.exp(t[0]) if t else 0.0 for t in tlp]


def 찾기초안(seq, K):
    # 3단계의 프롬프트 찾기: 끝의 n-gram(n = 3 → 2)이 앞에 나온 마지막 자리의 뒤 K 개
    for n in (3, 2):
        if len(seq) < n + 1:
            continue
        끝 = seq[-n:]
        for s in range(len(seq) - n - 1, -1, -1):
            if seq[s:s + n] == 끝:
                return seq[s + n:s + n + K]
    return []


def 흉내(P, Y, 맞음, 확률, opt, c):
    t1, td, K, p = opt["t1"], opt["td"], opt["K"], opt["p"]
    n = len(Y)
    out = {}
    out["없음"] = {"ms": n * t1, "steps": n}
    # 첫 토큰은 프롬프트 처리가 낸다 — 스텝은 둘째 토큰부터
    for 이름 in ("찾기", "모델", "섞음"):
        i, ms, steps, 받음, 낸 = 1, 0.0, 0, 0, 0
        while i < n:
            seq = P + Y[:i]
            초안 = []
            쓴작은 = 0
            if 이름 in ("찾기", "섞음"):
                초안 = 찾기초안(seq, K)
            if not 초안 and 이름 in ("모델", "섞음"):
                # 작은 모델의 탐욕 초안 — 앞부분이 맞는 동안은 echo 의 1위와 같다. 1위 확률이 문턱 아래면 멈춘다
                k = 0
                while k < K and i + k < n:
                    쓴작은 += 1
                    if 확률[i + k] < p:
                        break
                    if not 맞음[i + k]:
                        k += 1          # 틀린 초안도 하나는 낸다(검증에서 버려진다)
                        break
                    k += 1
                초안 = [Y[i + j] if j < k and 맞음[i + j] else -1 for j in range(k)]
            a = 0
            while a < len(초안) and i + a < n and 초안[a] == Y[i + a]:
                a += 1
            B = len(초안) + 1
            ms += c(B) * t1 + 쓴작은 * td
            steps += 1
            받음 += a
            낸 += len(초안)
            i += a + 1
        out[이름] = {"ms": ms, "steps": steps, "accepted": 받음, "proposed": 낸}
    return out


def main():
    pos, opt = 인자()
    재생, url, 출력 = pos
    c = c표(opt["c"])
    d = json.load(open(재생, encoding="utf-8"))
    합 = {}
    줄들 = []
    for s in d["sessions"]:
        for r in s["requests"]:
            P, Y = r.get("prompt_ids"), r.get("answer_ids")
            if not P or not Y or len(Y) < 2:
                continue
            top1, pr = 점수(url, P, Y)
            맞음 = [top1[j] == Y[j] for j in range(len(Y))]
            o = 흉내(P, Y, 맞음, pr, opt, c)
            o["n"] = len(Y)
            o["top1_match"] = sum(맞음) / len(Y)
            줄들.append(o)
            for k, v in o.items():
                if isinstance(v, dict):
                    합[k] = 합.get(k, 0.0) + v["ms"]
            print(f"{len(줄들):2d} 토큰 {len(Y):5d}  1위 같음 {o['top1_match']:.2f}  " + "  ".join(
                f"{k} {o[k]['ms'] / o['없음']['ms']:.2f}" for k in ("찾기", "모델", "섞음")), flush=True)
    없음 = 합["없음"]
    print("모두(추측 없음 대비 시간): " + "  ".join(f"{k} {v / 없음:.3f}" for k, v in 합.items()))
    json.dump({"opt": opt, "rows": 줄들, "sum_ms": 합}, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
