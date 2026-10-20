#!/usr/bin/env python3
"""디버거 v1 판정 5 — 기록 비용: 같은 요청들을 앱의 실행 기록을 끄고 · 켜고 번갈아 보내 걸린 시간을 비교한다(문서/디버거.md).

    python 재기/디버거기록비용.py [--앱 http://127.0.0.1:44639] [--묶음 시험/디버거/묶음50.jsonl] [--실행 qwencode3] [--돌림 3]
                                 [--결과 보고.json]

요청 세 갈래 — (가) 시험 묶음(흘림 끔 · 짧은 답), (나) 긴 답 둘(흘림 · 1,000 토큰), (다) 앱에 기록해 둔 에이전트 실행의 요청(흘림 ·
프롬프트 1.5 만 토큰대 — geulpu 칸은 빼고 보낸다). 씨앗을 박아 두 쪽의 일이 같다. 켬 = 앱의 실행 기록(요청에 logit_hash ·
prompt_tokens 를 넣고 응답을 파일로 — 서버가 걸음마다 로짓 해시를 셈하는 값까지 든다). 돌림마다 끔 · 켬의 차례를 바꾸고, 켬의 실행
파일은 크기만 적고 지운다. 표준 라이브러리만.
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

긴답 = [
    {"messages": [{"role": "user", "content": "한국의 사계절과 계절 음식에 대한 수필을 1,500자 정도로 써 줘."}],
     "max_tokens": 1000, "temperature": 0.7, "seed": 7001},
    {"messages": [{"role": "user", "content": "Write a short story (about 900 words) about a lighthouse keeper who finds a message in a bottle."}],
     "max_tokens": 1000, "temperature": 0.7, "seed": 7002},
]


def 묻기(앱, 방법, 길, 몸=None, 날것=False):
    r = urllib.request.Request(앱 + 길, data=None if 몸 is None else json.dumps(몸, ensure_ascii=False).encode("utf-8"), method=방법,
                               headers={"Content-Type": "application/json"})
    d = urllib.request.urlopen(r, timeout=3600).read()
    return d if 날것 else (json.loads(d) if d else None)


def 보내기(앱, 몸):
    """요청 하나 — 응답의 마지막 바이트까지 걸린 초"""
    r = urllib.request.Request(앱 + "/v1/chat/completions", data=json.dumps(몸, ensure_ascii=False).encode("utf-8"), method="POST",
                               headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=3600) as f:
        while f.read(65536):
            pass
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--앱", default="http://127.0.0.1:44639")
    p.add_argument("--묶음", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "시험", "디버거", "묶음50.jsonl"))
    p.add_argument("--실행", default="qwencode3", help="앱에 있는 실행 — 그 요청을 (다)로")
    p.add_argument("--돌림", type=int, default=3)
    p.add_argument("--결과")
    a = p.parse_args()
    앱 = a.앱.rstrip("/")
    갈래 = {"가": [], "나": [], "다": []}
    for i, l in enumerate(open(a.묶음, encoding="utf-8")):
        if l.strip():
            x = json.loads(l)
            x["stream"] = False
            x.setdefault("seed", 1000 + i)
            갈래["가"].append(x)
    for x in 긴답:
        갈래["나"].append(dict(x, stream=True, stream_options={"include_usage": True}))
    if a.실행:
        본 = 묻기(앱, "GET", "/api/debug/run?name=" + urllib.parse.quote(a.실행), 날것=True).decode("utf-8")
        for l in 본.splitlines()[1:]:
            x = json.loads(l)
            g = (x.get("response") or {}).get("geulpu") or {}
            if x.get("status") != 200 or "request" not in x or g.get("seed") is None:
                continue
            몸 = dict(x["request"])
            몸.pop("geulpu", None)
            몸["seed"] = g["seed"]
            갈래["다"].append(몸)
    print(f"요청 — 가 {len(갈래['가'])} · 나 {len(갈래['나'])} · 다 {len(갈래['다'])} · 돌림 {a.돌림}", flush=True)

    def 한바퀴(켬, 이름):
        if 켬:
            묻기(앱, "POST", "/api/debug/record/start", {"name": 이름})
        try:
            걸림 = {k: [보내기(앱, 몸) for 몸 in v] for k, v in 갈래.items()}
        finally:
            if 켬:
                묻기(앱, "POST", "/api/debug/record/stop", {})
        크기 = 0
        if 켬:
            크기 = next((r["bytes"] for r in 묻기(앱, "GET", "/api/debug")["runs"] if r["name"] == 이름), 0)
            묻기(앱, "POST", "/api/debug/run/delete", {"name": 이름})
        return 걸림, 크기

    한바퀴(False, None)                              # 데우기(슬롯 · 저장소 · 모듈)
    바퀴 = []
    for r in range(a.돌림):
        차례 = (False, True) if r % 2 == 0 else (True, False)
        for 켬 in 차례:
            걸림, 크기 = 한바퀴(켬, f"기록비용-{os.getpid()}-{r}")
            바퀴.append({"돌림": r, "켬": 켬, "걸림": 걸림, "바이트": 크기})
            print(f"돌림 {r} · {'켬' if 켬 else '끔'} · " + " · ".join(f"{k} {sum(v):.2f} 초" for k, v in 걸림.items())
                  + (f" · 실행 파일 {크기:,} 바이트" if 켬 else ""), flush=True)
    요약 = {}
    for k in 갈래:
        끔 = sum(sum(b["걸림"][k]) for b in 바퀴 if not b["켬"])
        켬 = sum(sum(b["걸림"][k]) for b in 바퀴 if b["켬"])
        요약[k] = {"끔_초": round(끔, 3), "켬_초": round(켬, 3), "늘어남_퍼센트": round((켬 / 끔 - 1) * 100, 2) if 끔 else None}
    끔 = sum(v["끔_초"] for v in 요약.values())
    켬 = sum(v["켬_초"] for v in 요약.values())
    요약["모두"] = {"끔_초": round(끔, 3), "켬_초": round(켬, 3), "늘어남_퍼센트": round((켬 / 끔 - 1) * 100, 2)}
    for k, v in 요약.items():
        print(f"{k}: 끔 {v['끔_초']:.2f} 초 · 켬 {v['켬_초']:.2f} 초 · {v['늘어남_퍼센트']:+.2f} %")
    if a.결과:
        json.dump({"요약": 요약, "바퀴": 바퀴}, open(a.결과, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
