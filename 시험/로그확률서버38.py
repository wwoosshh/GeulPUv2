#!/usr/bin/env python3
"""38단계 — 서버의 로그확률(38a · 38b)을 서버에 바로 묻는다(관리자 없이 — 서버를 --기록 폴더로 띄운다).

    python -I -X utf8 시험/로그확률서버38.py 돌리기 <주소> --틀 qwen3|chatml|r1 --기록 <서버의 기록 폴더> --출력 <결과.json>
        [--최대 128] [--위수 5] [--이름 머리]
    python -I -X utf8 시험/로그확률서버38.py 견주기 <결과 a.json> <결과 b.json>       (38b (4) — 백엔드를 건너)

한 모델에 질문 셋 × (탐욕 · 뽑기 T 0.7 — 씨앗 고정, 2⁶³ 넘는 64비트 값):
  A. 생성(채팅 — r1 은 완성)하며 로그확률(상위 k) · 로짓 해시 · 프롬프트 토큰 · 기록(걸음마다의 로짓 행).
  38a  기록한 로짓 행으로 파이썬 참조(계약/로그확률.py)가 셈한 값 — 뽑은 토큰의 로그확률 · 엔트로피 · 상위 k(번호 · 값) — 이 응답과
       f64 비트까지 같은가. 행의 FNV 해시가 응답의 로짓 해시와 같은가(같은 행인가). echo 의 프롬프트 행(기록 _프롬로짓)도 같게.
  38b  (1) 같은 토큰을 echo 로 점수 매긴 값(캐시 켬 · 끔)이 생성의 값과 같은가(로그확률 · 상위 k · 엔트로피 · 로짓 해시),
       (2) 여섯 요청을 한꺼번에 보낸 생성(동시 처리)이 하나씩 보낸 것과 같은가(토큰 · 로그확률 · 해시),
       (3) 결과 파일끼리 — 백엔드를 건너(CUDA ↔ Vulkan) 같은가(견주기).
"""
import argparse
import concurrent.futures as cf
import json
import os
import struct
import sys
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "계약"))
import 로그확률 as LP          # noqa: E402

질문들 = ["프랑스의 수도는 어디인가요? 한 문장으로 답해 주세요.",
         "Write a haiku about the ocean.",
         "파이썬으로 리스트를 뒤집는 함수를 짧게 작성해 줘."]
씨앗 = (1 << 63) + 38          # 2⁵³ 위 — JSON 숫자 그대로 오가야 같은 뽑기
R1머리 = "<｜begin▁of▁sentence｜>"
R1사용자 = "<｜User｜>"
R1답 = "<｜Assistant｜>"


def 묻기(주소, 길, 몸):
    q = urllib.request.Request(주소 + 길, data=json.dumps(몸).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(q, timeout=600) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{길} → HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from None


def 비트(x):
    return None if x is None else LP.실수비트(float(x))


def 생성요청(a, i, 방식, 기록이름):
    """A — 생성 요청의 몸(질문 i, 방식 '탐욕' · '뽑기')."""
    몸 = {"max_tokens": a.최대, "temperature": 0.0 if 방식 == "탐욕" else 0.7,
          "geulpu": {"logit_hash": True, "prompt_tokens": True, "record": 기록이름}}
    몸["seed"] = 씨앗                    # 탐욕은 씨앗을 쓰지 않지만 결과 파일끼리 견주도록 고정
    if a.틀 == "r1":
        몸["prompt"] = R1머리 + R1사용자 + 질문들[i] + R1답
        몸["logprobs"] = a.위수
        return "/v1/completions", 몸
    몸["messages"] = [{"role": "user", "content": 질문들[i]}]
    몸["logprobs"] = True
    몸["top_logprobs"] = a.위수
    if a.틀 == "qwen3":
        몸["chat_template_kwargs"] = {"enable_thinking": False}
    return "/v1/chat/completions", 몸


def 걸음들(r):
    """응답에서 내용 토큰의 걸음마다 (로그확률, 엔트로피, 상위 번호, 상위 값) — 값은 비트. 상위 값은 geulpu.top_logprobs(번호와 나란히 —
    완성의 사전 꼴은 같은 글의 토큰을 합친다). 채팅이면 OpenAI 꼴(content)의 값도 그것과 같은지 본다."""
    g = r["geulpu"]
    c = r["choices"][0]
    out = []
    n = len(g["entropy"])
    if "content" in (c.get("logprobs") or {}):
        lps = [e["logprob"] for e in c["logprobs"]["content"]]
        for i, e in enumerate(c["logprobs"]["content"]):
            assert [t["logprob"] for t in e["top_logprobs"]] == g["top_logprobs"][i], ("채팅 꼴 ≠ geulpu.top_logprobs", i)
    else:
        lp = c["logprobs"]["token_logprobs"]
        lps = lp[len(lp) - n:]
    for i in range(n):
        out.append((비트(lps[i]), 비트(g["entropy"][i]), g["top_ids"][i], [비트(v) for v in g["top_logprobs"][i]]))
    return out


def 메아리걸음들(r, 첫):
    """echo 응답에서 프롬프트 토큰 첫 … 끝의 (로그확률, 엔트로피, 상위 번호, 상위 값)."""
    g = r["geulpu"]
    lp = r["choices"][0]["logprobs"]
    out = []
    for i in range(첫, len(lp["token_logprobs"])):
        out.append((비트(lp["token_logprobs"][i]), 비트(g["prompt_entropy"][i]), g["prompt_top_ids"][i],
                    [비트(v) for v in g["prompt_top_logprobs"][i]]))
    return out


def fnv(행):
    w = np.frombuffer(np.ascontiguousarray(행, np.float32).tobytes(), np.uint32)
    h = 0xCBF29CE484222325
    for x in w.tolist():
        h = ((h ^ x) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def 참조대조(행들, 표적들, 응답걸음, 해시들, k):
    """38a — 행마다 파이썬 참조와 응답(같은 걸음)의 비트 · 행 해시. (다른 걸음 수, 해시 다른 수, 첫 다름)"""
    다름, 해시다름, 첫 = 0, 0, None
    for i, (행, t) in enumerate(zip(행들, 표적들)):
        x = LP.로그확률행(행, 위수=k)
        참 = (LP.실수비트(float(x["로그확률"][t])), LP.실수비트(x["H"]), [j for j, _ in x["위"]], [LP.실수비트(v) for _, v in x["위"]])
        if i < len(응답걸음) and 참 != tuple(응답걸음[i]):
            다름 += 1
            첫 = 첫 if 첫 is not None else (i, 참[:2], tuple(응답걸음[i])[:2])
        if 해시들 is not None and i < len(해시들) and fnv(행) != 해시들[i]:
            해시다름 += 1
    return 다름, 해시다름, 첫


def 행읽기(경로, V):
    raw = open(경로, "rb").read()
    return [np.frombuffer(raw, np.float32, V, i * 4 * V) for i in range(len(raw) // (4 * V))]


def 돌리기(a):
    with urllib.request.urlopen(a.주소 + "/tokenizer_info", timeout=60) as r:
        V = json.loads(r.read().decode("utf-8"))["vocab_size"]
    k = a.위수
    결과 = {"주소": a.주소, "틀": a.틀, "위수": k, "최대": a.최대, "요청": {}, "38a": {}, "38b": {}}
    일들 = [(i, 방식) for i in range(len(질문들)) for 방식 in ("탐욕", "뽑기")]
    합 = {"38a 걸음": 0, "38a 다름": 0, "38a 해시다름": 0, "38a 메아리 걸음": 0, "38a 메아리 다름": 0, "38a 메아리 해시다름": 0,
         "38b 메아리 걸음": 0, "38b 메아리 다름": 0, "38b 동시 다름": 0}
    하나씩 = {}
    for i, 방식 in 일들:
        이름 = f"{a.이름}_{i}_{'g' if 방식 == '탐욕' else 's'}"
        길, 몸 = 생성요청(a, i, 방식, 이름)
        r = 묻기(a.주소, 길, 몸)
        g = r["geulpu"]
        걸 = 걸음들(r)
        하나씩[이름] = r
        # 38a — 생성의 걸음(기록한 로짓 행)
        행들 = 행읽기(os.path.join(a.기록, 이름 + "_로짓.bin"), V)
        토큰 = list(struct.unpack(f"<{len(행들)}q", open(os.path.join(a.기록, 이름 + "_토큰.bin"), "rb").read()))
        assert 토큰 == g["tokens"], (이름, "기록의 토큰 ≠ 응답의 토큰")
        d, hd, 첫 = 참조대조(행들, 토큰, 걸, g["logit_hashes"], k)
        합["38a 걸음"] += len(걸)
        합["38a 다름"] += d
        합["38a 해시다름"] += hd
        # 38b (1) — 같은 토큰을 echo 로(캐시 켬 · 끔)
        P = len(g["prompt_tokens"])
        메아리 = {}
        for 캐시 in (True, False):
            e이름 = 이름 + ("_ec" if 캐시 else "_e")
            em = {"prompt": g["prompt_tokens"] + g["tokens"], "echo": True, "max_tokens": 0, "logprobs": k, "temperature": 0.0,
                  "cache_prompt": 캐시, "geulpu": {"echo_from": P, "logit_hash": True, "record": e이름}}
            er = 묻기(a.주소, "/v1/completions", em)
            mg = 메아리걸음들(er, P)
            메아리[캐시] = er
            n = len(걸)
            다른 = sum(1 for x, y in zip(걸, mg[:n]) if tuple(x) != tuple(y))
            해시다른 = sum(1 for x, y in zip(g["logit_hashes"], er["geulpu"]["prompt_logit_hashes"]) if x != y)
            합["38b 메아리 걸음"] += n
            합["38b 메아리 다름"] += 다른 + 해시다른 + (0 if len(er["geulpu"]["prompt_logit_hashes"]) == len(g["logit_hashes"]) else 1)
            # 38a — echo 의 프롬프트 행(기록 _프롬로짓): 표적 = 다음 프롬프트 토큰
            e행 = 행읽기(os.path.join(a.기록, e이름 + "_프롬로짓.bin"), V)
            표적 = (g["prompt_tokens"] + g["tokens"])[P:P + len(e행)]
            d2, hd2, _ = 참조대조(e행, 표적, mg, er["geulpu"]["prompt_logit_hashes"], k)
            합["38a 메아리 걸음"] += len(e행)
            합["38a 메아리 다름"] += d2
            합["38a 메아리 해시다름"] += hd2
        결과["요청"][이름] = {"토큰": g["tokens"], "프롬프트": g["prompt_tokens"], "해시": g["logit_hashes"], "걸음": 걸,
                            "메아리": 메아리걸음들(메아리[True], P), "씨앗": str(g["seed"])}
        print(f"{이름}: 토큰 {len(g['tokens'])} · 38a 다름 {d} · 해시다름 {hd}" + (f" 첫 {첫}" if 첫 else "")
              + f" · echo(캐시/끔) 프롬프트 캐시 {메아리[True]['geulpu']['prompt_tokens_cached']}/{메아리[False]['geulpu']['prompt_tokens_cached']}", flush=True)
    # 38b (2) — 여섯을 한꺼번에(기록 없이 — 이름이 겹치지 않게)
    def 한꺼번에(일):
        i, 방식 = 일
        길, 몸 = 생성요청(a, i, 방식, f"{a.이름}_{i}_{'g' if 방식 == '탐욕' else 's'}_par")
        return 일, 묻기(a.주소, 길, 몸)
    with cf.ThreadPoolExecutor(len(일들)) as ex:
        동시 = list(ex.map(한꺼번에, 일들))
    슬롯들 = set()
    for (i, 방식), r in 동시:
        이름 = f"{a.이름}_{i}_{'g' if 방식 == '탐욕' else 's'}"
        s = 하나씩[이름]
        같다 = (r["geulpu"]["tokens"] == s["geulpu"]["tokens"] and r["geulpu"]["logit_hashes"] == s["geulpu"]["logit_hashes"]
                and 걸음들(r) == 걸음들(s))
        합["38b 동시 다름"] += 0 if 같다 else 1
        슬롯들.add(r["geulpu"]["slot"])
    합["38b 동시 슬롯"] = sorted(슬롯들)
    결과["합"] = 합
    print(json.dumps(합, ensure_ascii=False))
    json.dump(결과, open(a.출력, "w", encoding="utf-8"), ensure_ascii=False)
    통과 = 합["38a 다름"] == 0 and 합["38a 해시다름"] == 0 and 합["38a 메아리 다름"] == 0 and 합["38a 메아리 해시다름"] == 0 \
        and 합["38b 메아리 다름"] == 0 and 합["38b 동시 다름"] == 0
    print("38a · 38b(이 백엔드):", "통과" if 통과 else "못 미침")
    return 0 if 통과 else 1


def 견주기(a):
    x = json.load(open(a.a, encoding="utf-8"))
    y = json.load(open(a.b, encoding="utf-8"))
    다름 = []
    for 이름, rx in x["요청"].items():
        ry = y["요청"].get(이름)
        if ry is None:
            다름.append((이름, "없음"))
            continue
        for 칸 in ("토큰", "프롬프트", "해시", "걸음", "메아리", "씨앗"):
            if rx[칸] != ry[칸]:
                다름.append((이름, 칸))
    print(f"38b (3) 백엔드를 건너: 요청 {len(x['요청'])} · 다른 칸 {len(다름)}" + (f" — {다름[:6]}" if 다름 else ""))
    return 0 if not 다름 else 1


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="일", required=True)
    p = sub.add_parser("돌리기")
    p.add_argument("주소")
    p.add_argument("--틀", choices=["qwen3", "chatml", "r1"], required=True)
    p.add_argument("--기록", required=True)
    p.add_argument("--출력", required=True)
    p.add_argument("--최대", type=int, default=128)
    p.add_argument("--위수", type=int, default=5)
    p.add_argument("--이름", default="s38")
    q = sub.add_parser("견주기")
    q.add_argument("a")
    q.add_argument("b")
    a = ap.parse_args()
    return 돌리기(a) if a.일 == "돌리기" else 견주기(a)


if __name__ == "__main__":
    sys.exit(main())
