"""GeulPUv2 문맥 저장소 4단계 재기 — 쓰임새의 판정 숫자(문서/문맥저장소.md): 새 작업의 첫 토큰 · 긴 대화를 서버를 다시 켠 뒤 다시
열기 · 그 대화의 앞 메시지로 되감아 다르게 잇기를 저장소 끔(다시 셈) · 켬(덩이 파일이 OS 캐시에) · 켬(차가운 SSD — 덩이 파일을 OS
캐시에서 뺀 뒤, 재기/캐시비우기.py)으로 재고, 세 길의 답 · 걸음마다 로짓 해시가 같은가 본다. 표준 라이브러리만.

    python -I 재기/문맥저장소.py <서버.exe> <모델.gguf> <모듈> <작업폴더> <기록 t1.jsonl> <기록 t2.jsonl> <긴 글 파일>
          [--포트 8099] [--문맥 32768] [--긴대화 31000] [--되풀이 3]

기록은 재기/MoE 의 Qwen Code 세션 기록(jsonl — 요청마다 {"request": …, "response": …}).
- 새 작업: t1 의 첫 요청(시스템 프롬프트 · 도구 · 스킬 · 작업 폴더 · 날짜 · 작업 지시)에서 작업 지시만 t2 의 것으로 바꾼 요청 —
  같은 폴더에서 같은 날 새 작업을 시작할 때. 저장소에는 t1 의 첫 요청이 있다.
- 긴 대화: t1 에서 가장 긴 요청 + 그 답 + 긴 글(문맥이 --긴대화 토큰이 되게 자름)을 붙인 사용자 말 = L. 저장소에는 L 과 그 답이 있다.
  다시 열기 = L + 그 답 + 다음 말.
- 되감기: L 의 가운데 메시지까지 + 다른 지시(되감은 지점까지 같은 앞부분).
차례: (0) 저장소 없는 서버 — 긴 글 자르기, 앞부분이 깨지는 자리(다른 폴더 · 다음 날 · 같은 날), 다시 셈(cache_prompt: false)을 되풀이
만큼 · L 의 답(다시 열기 요청을 짓는다). (1) 빈 저장소로 t1 의 첫 요청과 L 을 보내 쓰고 비우기. (2) 재는 실행마다 서버를 새로
(저장소쓰기 끔 — 읽기만이라 OS 캐시 · SSD 실행이 같은 저장소를 본다) 요청 하나: OS 캐시 · SSD(띄우기 전에 덩이 파일을 캐시에서 뺌).
결과: <작업폴더>/판정.json · 판정.txt(표)

    python -I 재기/문맥저장소.py 전환 <차례.json> <번갈아 · 저장소 끔.json> <번갈아 · 저장소 켬.json>

전환: 시험/저장소서버.py 돌리기의 결과 셋(같은 요청열 — 작업마다 차례로 · 작업 셋을 번갈아(슬롯 2 라 돌아올 때마다 제 문맥이 덮여
있다) 저장소 끔 · 켬)에서 요청마다 첫 토큰의 차(번갈아 − 차례 = 전환 비용)를 작업의 둘째 요청부터 모아 가운데값 · 합을 낸다.
"""
import argparse
import copy
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "시험"))
sys.path.insert(0, HERE)
import 저장소서버 as 틀        # noqa: E402 — 서버 · 요청
import 캐시비우기               # noqa: E402

씨앗 = {"새작업": 4101, "긴대화": 4102, "되감기": 4103, "L": 4100}
생성 = 32


def 기록읽기(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def 몸(원본, 이름, 최대=생성, 다시셈=False):
    b = {k: v for k, v in copy.deepcopy(원본).items() if k != "stream_options"}
    b.update(stream=False, max_tokens=최대, temperature=0.7, seed=씨앗[이름], geulpu={"logit_hash": True})
    if 다시셈:
        b["cache_prompt"] = False
    return b


def 보내기(포트, b):
    t0 = time.time()
    s, r = 틀.요청(포트, "POST", "/v1/chat/completions", b)
    if s != 200:
        raise SystemExit(f"HTTP {s} {r}")
    g = r.get("geulpu", {})
    st = g.get("store") or {}
    m = r["choices"][0]["message"]
    return {"프롬프트": r["usage"]["prompt_tokens"], "첫토큰ms": round((g.get("first_token_ms") or 0) / 1000, 1),    # 서버의 이 칸은 µs(이름만 ms)
            "걸림초": round(time.time() - t0, 3),
            "되살린": st.get("restored_tokens", 0), "되살림ms": st.get("restore_ms", 0.0), "슬롯캐시": st.get("slot_cached_tokens", 0),
            "복사": st.get("slot_copy_tokens", 0), "깨진자리": st.get("reuse_end"), "까닭": st.get("break"),
            "토큰": g.get("tokens"), "해시": g.get("logit_hashes"), "글": m.get("content"), "호출": m.get("tool_calls"),
            "끝": r["choices"][0]["finish_reason"]}


def 토큰수(포트, 글=None, 메시지=None, 도구=None):
    b = {"prompt": 글} if 글 is not None else {"messages": 메시지, "tools": 도구 or []}
    s, r = 틀.요청(포트, "POST", "/tokenize", b)
    if s != 200:
        raise SystemExit(f"/tokenize: HTTP {s} {r}")
    return r["tokens"]


def 같은앞(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def 요청들(t1, t2):
    """(새작업 원본, 새작업 요청, L 요청 — 긴 글은 나중에, 되감기 요청)"""
    첫 = t1[0]["request"]
    새 = copy.deepcopy(첫)
    새["messages"][1]["content"][-1]["text"] = t2[0]["request"]["messages"][1]["content"][-1]["text"]
    길이 = [(r.get("usage") or {}).get("prompt_tokens", 0) for r in t1]
    i = 길이.index(max(길이))
    긴 = copy.deepcopy(t1[i]["request"])
    답 = t1[i]["response"] or {}
    긴["messages"].append({"role": "assistant", "content": 답.get("content") or ""})
    m = 긴["messages"]
    j = len(m) // 2
    while j > 2 and m[j]["role"] != "tool":
        j -= 1
    되 = copy.deepcopy(긴)
    되["messages"] = 되["messages"][:j + 1] + [{"role": "user", "content": "잠깐, 방향을 바꾸자. 지금까지 만든 것은 그대로 두고, "
                                                                     "할 일마다 마감 날짜를 넣을 수 있게 설계부터 다시 잡아 줘."}]
    return 첫, 새, 긴, 되, i, j


def 서버켜기(a, 이름, 저장소=None, 쓰기="모두"):
    덧 = ["--저장소", 저장소, "--저장소크기", "40", "--저장소쓰기", 쓰기] if 저장소 else []
    return 틀.서버(a.서버, a.모델, a.모듈, a.포트, 2, a.문맥, 덧, os.path.join(a.작업폴더, 이름 + ".log"))


def 같음(x, y):
    return x["토큰"] == y["토큰"] and x["해시"] == y["해시"] and x["끝"] == y["끝"]


def main():
    p = argparse.ArgumentParser()
    for n in ("서버", "모델", "모듈", "작업폴더", "t1", "t2", "긴글"):
        p.add_argument(n)
    p.add_argument("--포트", type=int, default=8099)
    p.add_argument("--문맥", type=int, default=32768)
    p.add_argument("--긴대화", type=int, default=31000)
    p.add_argument("--되풀이", type=int, default=3)
    a = p.parse_args()
    os.makedirs(a.작업폴더, exist_ok=True)
    저장소 = os.path.join(a.작업폴더, "저장소")
    if os.path.exists(저장소):
        raise SystemExit(f"{저장소} 가 이미 있다 — 빈 폴더에서 시작한다")
    t1, t2 = 기록읽기(a.t1), 기록읽기(a.t2)
    첫, 새, 긴, 되, 긴i, 되j = 요청들(t1, t2)
    결과 = {"인자": sys.argv[1:], "긴 요청": 긴i, "되감은 메시지": 되j, "올림초": [], "다시셈": {}, "캐시": {}, "SSD": {}}

    # (0) 저장소 없는 서버 — 긴 글 · 깨지는 자리 · 다시 셈 · L 의 답
    s = 서버켜기(a, "0_다시셈")
    결과["올림초"].append(s.올림초)
    try:
        글 = open(a.긴글, encoding="utf-8").read()
        바탕 = len(토큰수(a.포트, 메시지=긴["messages"] + [{"role": "user", "content": ""}], 도구=긴.get("tools")))
        목표 = a.긴대화 - 바탕 - 40
        lo, hi = 0, len(글)
        while lo < hi:                                  # 토큰이 목표를 넘지 않는 가장 긴 앞부분
            mid = (lo + hi + 1) // 2
            if len(토큰수(a.포트, 글=글[:mid])) <= 목표:
                lo = mid
            else:
                hi = mid - 1
        긴["messages"].append({"role": "user", "content": "아래는 이 도구에 붙일 저장소 모듈의 코드다. 읽고, 지금 만든 할 일 도구에 "
                                                         "어떻게 쓸 수 있을지 정리해 줘.\n\n```\n" + 글[:lo] + "\n```"})
        L토큰 = 토큰수(a.포트, 메시지=긴["messages"], 도구=긴.get("tools"))
        # 앞부분이 깨지는 자리 — 새 작업이 다른 폴더 · 같은 폴더의 다음 날 · 같은 폴더 같은 날일 때(t1 의 첫 요청과 견줌)
        기준 = 토큰수(a.포트, 메시지=첫["messages"], 도구=첫.get("tools"))
        다른폴더 = 토큰수(a.포트, 메시지=t2[0]["request"]["messages"], 도구=t2[0]["request"].get("tools"))
        다음날 = copy.deepcopy(첫)
        for part in 다음날["messages"][1]["content"]:
            part["text"] = part["text"].replace("October 19, 2026", "October 20, 2026").replace("Monday", "Tuesday")
        다음날토큰 = 토큰수(a.포트, 메시지=다음날["messages"], 도구=다음날.get("tools"))
        같은날 = 토큰수(a.포트, 메시지=새["messages"], 도구=새.get("tools"))
        결과["깨지는 자리"] = {"첫 요청": len(기준), "다른 폴더": 같은앞(기준, 다른폴더), "같은 폴더 · 다음 날": 같은앞(기준, 다음날토큰),
                          "같은 폴더 · 같은 날": 같은앞(기준, 같은날), "새 작업 요청": len(같은날)}
        결과["L 토큰"] = len(L토큰)
        print("깨지는 자리:", 결과["깨지는 자리"], "· L", len(L토큰), flush=True)
        L답 = 보내기(a.포트, 몸(긴, "L", 64, 다시셈=True))
        다시 = copy.deepcopy(긴)
        다시["messages"] += [{"role": "assistant", "content": L답["글"] or json.dumps(L답["호출"], ensure_ascii=False)},
                           {"role": "user", "content": "좋아. 그 정리대로 첫 번째 것부터 해 줘."}]
        결과["L 답"] = {k: L답[k] for k in ("프롬프트", "끝", "글")}
        대상 = {"새작업": 새, "긴대화": 다시, "되감기": 되}
        for r in range(a.되풀이):
            for 이름, 요 in 대상.items():
                x = 보내기(a.포트, 몸(요, 이름, 다시셈=True))
                결과["다시셈"].setdefault(이름, []).append(x)
                print(f"다시 셈 {이름} {r}: 프롬프트 {x['프롬프트']} · 첫 토큰 {x['첫토큰ms']} ms", flush=True)
    finally:
        s.끄기()

    # (1) 빈 저장소에 쓰기 — t1 의 첫 요청, L(그 답이 (0)과 같아야 한다)
    s = 서버켜기(a, "1_쓰기", 저장소, "모두")
    try:
        x = 보내기(a.포트, 몸(첫, "새작업"))
        y = 보내기(a.포트, 몸(긴, "L", 64))
        결과["L 답 같음"] = 같음(y, L답)
        _, 결과["비우기"] = 틀.요청(a.포트, "POST", "/geulpu/store/flush", {})
        _, 결과["저장소상태"] = 틀.요청(a.포트, "GET", "/geulpu/store")
        print("쓰기:", x["프롬프트"], y["프롬프트"], "· L 답 같음", 결과["L 답 같음"], "·", json.dumps(결과["저장소상태"], ensure_ascii=False), flush=True)
    finally:
        s.끄기()

    # (2) 서버를 새로 켜 하나씩 — OS 캐시 · SSD
    for r in range(a.되풀이):
        for 곳 in ("캐시", "SSD"):
            for 이름, 요 in 대상.items():
                if 곳 == "SSD":
                    수 = sum(1 for f in 캐시비우기.파일들([저장소]) if 캐시비우기.비우기(f))
                s = 서버켜기(a, f"2_{곳}_{이름}_{r}", 저장소, "끔")
                결과["올림초"].append(s.올림초)
                try:
                    x = 보내기(a.포트, 몸(요, 이름))
                finally:
                    s.끄기()
                결과[곳].setdefault(이름, []).append(x)
                print(f"{곳} {이름} {r}: 프롬프트 {x['프롬프트']} · 되살린 {x['되살린']} ({x['되살림ms']:.0f} ms) · 첫 토큰 {x['첫토큰ms']} ms"
                      + (f" · 캐시에서 뺀 파일 {수}" if 곳 == "SSD" else ""), flush=True)

    # 표
    줄 = ["| 경우 | 프롬프트 | 다시 셈 | 켬 · OS 캐시 | 켬 · SSD | 되살린 토큰 | 되살림(캐시 · SSD) | 같음 |", "|---|---|---|---|---|---|---|---|"]
    for 이름 in 대상:
        d, c, ssd = 결과["다시셈"][이름], 결과["캐시"][이름], 결과["SSD"][이름]
        중 = lambda xs, k: statistics.median(x[k] for x in xs)        # noqa: E731
        모두같음 = all(같음(x, d[0]) for x in d + c + ssd)
        결과.setdefault("같음", {})[이름] = 모두같음
        줄.append(f"| {이름} | {d[0]['프롬프트']:,} | {중(d, '첫토큰ms'):,.0f} ms | {중(c, '첫토큰ms'):,.0f} ms | {중(ssd, '첫토큰ms'):,.0f} ms | "
                 f"{c[0]['되살린']:,} | {중(c, '되살림ms'):,.0f} · {중(ssd, '되살림ms'):,.0f} ms | {'같음' if 모두같음 else '다름'} |")
    표 = "\n".join(줄)
    print(표, flush=True)
    open(os.path.join(a.작업폴더, "판정.txt"), "w", encoding="utf-8").write(표 + "\n")
    json.dump(결과, open(os.path.join(a.작업폴더, "판정.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if all(결과["같음"].values()) and 결과["L 답 같음"] else 1)


def 전환():
    차례, 끔, 켬 = (json.load(open(p, encoding="utf-8")) for p in sys.argv[2:5])
    기준 = {x["번호"]: x for x in 차례["요청"]}
    줄 = ["| | 요청(작업의 둘째부터) | 첫 토큰 합 | 전환 비용 가운데값 | 전환 비용 합 | 되살린 토큰 | 같음 |", "|---|---|---|---|---|---|---|"]
    for 이름, d in (("차례(전환 없음)", 차례), ("번갈아 · 저장소 끔", 끔), ("번갈아 · 저장소 켬", 켬)):
        xs = [x for x in d["요청"] if not x["번호"].endswith(":0")]
        차 = [(x["첫토큰ms"] - 기준[x["번호"]]["첫토큰ms"]) / 1000 for x in xs]
        같음 = all(x["토큰"] == 기준[x["번호"]]["토큰"] and x["해시"] == 기준[x["번호"]]["해시"] for x in d["요청"])
        되 = sum((x["저장소"] or {}).get("restored_tokens", 0) for x in xs)
        줄.append(f"| {이름} | {len(xs)} | {sum(x['첫토큰ms'] for x in xs) / 1e6:.1f} 초 | {statistics.median(차):,.0f} ms | "
                 f"{sum(차) / 1000:.1f} 초 | {되:,} | {'같음' if 같음 else '다름'} |")
    print("\n".join(줄))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "전환":
        전환()                                       # 첫토큰ms 칸은 µs(시험/저장소서버.py 가 서버 값을 그대로 적음)
    else:
        main()          # 표준 출력은 시험/저장소서버.py 를 가져올 때 UTF-8 로 바뀐다
