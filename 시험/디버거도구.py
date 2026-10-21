#!/usr/bin/env python3
"""디버거 v1 2단계 — 도구/debugger.py 의 GPU 없는 시험(빠른 점검 4e — 시험/빠른점검.py 가 부른다). 실행 로그 나누기 · 분석 줄
덧붙이기(파일) · 분기 요청 짓기 · 요청 고치기(경로) · echo 응답 풀기 · 점수가 같은 서버의 것인가 · 중단점 찾기 · 비교 계산을, 손으로
정한 값과 이 파일에 따로 짠 단순한 참조(무작위 실행 여럿)로 본다. 같은 입력이면 같은 보고(글자까지)인지도 본다. node 가 있으면 앱
화면(앱/web/app.js)의 비교 · 중단점 · 큰 정수(64 비트 시드) JSON 이 도구와 같은 값인지도 본다(시험/디버거화면.js).

    python -X utf8 시험/디버거도구.py      → 마지막 줄 "디버거 도구: 사례 N 모두 통과"(종료 0), 틀리면 종료 1
"""
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("debugger", os.path.join(ROOT, "도구", "debugger.py"))
d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d)

사례 = 0
틀림 = []


def 같다(이름, 받은, 기대):
    global 사례
    사례 += 1
    if 받은 != 기대:
        틀림.append(f"{이름}: {받은!r} ≠ {기대!r}")


def 오류난다(이름, 함수, 종류):
    global 사례
    사례 += 1
    try:
        함수()
    except 종류:
        return
    except Exception as e:                            # noqa: BLE001 — 다른 오류도 틀림으로 적는다
        틀림.append(f"{이름}: {type(e).__name__} {e} (기대 {종류.__name__})")
        return
    틀림.append(f"{이름}: 오류가 나지 않았다(기대 {종류.__name__})")


서버A = {"sha256": "a" * 64, "layer": "cuda", "precision": "fast", "context": 32768, "yarn": 0}
서버B = {"sha256": "b" * 64, "layer": "cuda", "precision": "fast", "context": 32768, "yarn": 0}


def 요청줄(i, 프롬프트, 토큰, 서버=서버A, 요청=None, 해시=None, 시드=1000, 끝="stop", 상태=200, 호출=False):
    msg = {"role": "assistant", "content": "…"}
    if 호출:
        msg["tool_calls"] = [{"type": "function", "function": {"name": "f", "arguments": "{}"}}]
    return {"i": i, "at": "2026-10-20 12:00:00", "ms": 10, "path": "/v1/chat/completions", "status": 상태, "model": "m", "server": 서버,
            "request": 요청 if 요청 is not None else {"messages": [{"role": "user", "content": f"질문 {i}"}], "max_tokens": 64, "seed": 시드},
            "response": {"choices": [{"index": 0, "message": msg, "finish_reason": 끝}],
                         "geulpu": {"seed": 시드, "prompt_tokens": 프롬프트, "tokens": 토큰,
                                    "logit_hashes": 해시 if 해시 is not None else [f"{(i * 1000 + k):016x}" for k in range(len(토큰))]}}}


def 점수줄(i, 서버, 로그, 위번호, 위로그, 글=None, 해시=None):
    n = len(로그)
    return {"type": "score", "of": i, "at": "2026-10-20 12:00:01", "model": {"id": "m", "file": "m.gguf", "sha256": 서버["sha256"]},
            "server": 서버, "k": 2, "logprobs": 로그, "entropy": [0.5] * n, "top_ids": 위번호, "top_logprobs": 위로그,
            "token_text": 글 if 글 is not None else [f"t{k}" for k in range(n)], "logit_hashes": 해시 or [f"{k:016x}" for k in range(n)],
            "hash_check": "same"}


헤더A = {"format": "geulpu-run-log/1", "name": "A", "model": {"id": "ma", "sha256": 서버A["sha256"]}, "server": dict(서버A, slots=2)}
헤더B = {"format": "geulpu-run-log/1", "name": "B", "model": {"id": "mb", "sha256": 서버B["sha256"]}, "server": dict(서버B, slots=2)}


# ---- 1. 헤더 · 실행 나누기 ----
def 시험_실행():
    같다("헤더 새 꼴", d.헤더인가({"format": "geulpu-run-log/1"}), True)
    같다("헤더 옛 꼴", d.헤더인가({"geulrun": 1}), True)
    같다("헤더 아님", d.헤더인가({"i": 0}), False)
    같다("헤더 다른 형식", d.헤더인가({"format": "other/1"}), False)
    본 = "\n".join(json.dumps(x) for x in [헤더A, 요청줄(0, [1], [2]), {"type": "score", "of": 0}, {"type": "branch", "b": 0, "of": 0},
                                              요청줄(1, [1], [3]), {"type": "note"}]) + "\n"
    h, 줄들 = d.실행나누기(본, "x")
    실 = d.실행(h, 줄들, 파일="x.jsonl")
    같다("요청 줄 수", [x["i"] for x in 실.요청들], [0, 1])
    같다("점수 줄 수", len(실.점수들), 1)
    같다("분기 줄 수", len(실.분기들), 1)
    같다("요청 찾기", 실.요청(1)["i"], 1)
    오류난다("없는 요청", lambda: 실.요청(7), SystemExit)
    오류난다("헤더 없는 실행", lambda: d.실행나누기(json.dumps(요청줄(0, [1], [2])), "x"), SystemExit)
    같다("줄짓기", d.줄짓기({"a": [1, 2], "b": "한글"}), '{"a":[1,2],"b":"한글"}')


# ---- 2. 분석 줄 덧붙이기(파일) ----
def 시험_덧붙이기():
    with tempfile.TemporaryDirectory() as 폴더:
        for 끝줄바꿈 in (True, False):
            p = os.path.join(폴더, f"run{int(끝줄바꿈)}.jsonl")
            본 = json.dumps(헤더A) + "\n" + json.dumps(요청줄(0, [1], [2])) + ("\n" if 끝줄바꿈 else "")
            open(p, "w", encoding="utf-8", newline="").write(본)
            실 = d.실행불러오기("http://127.0.0.1:1", p)
            d.덧붙이기("http://127.0.0.1:1", 실, [{"type": "score", "of": 0, "x": 1.5}, {"type": "branch", "b": 0, "of": 0, "글": "줄\n바꿈"}])
            실2 = d.실행불러오기("http://127.0.0.1:1", p)
            같다(f"덧붙인 뒤 요청(끝 줄바꿈 {끝줄바꿈})", len(실2.요청들), 1)
            같다(f"덧붙인 점수(끝 줄바꿈 {끝줄바꿈})", 실2.점수들, [{"type": "score", "of": 0, "x": 1.5}])
            같다(f"덧붙인 분기의 글(끝 줄바꿈 {끝줄바꿈})", 실2.분기들[0]["글"], "줄\n바꿈")
            raw = open(p, "rb").read()
            같다(f"줄바꿈은 LF(끝 줄바꿈 {끝줄바꿈})", b"\r\n" in raw, False)
            같다(f"마지막은 줄바꿈(끝 줄바꿈 {끝줄바꿈})", raw.endswith(b"\n"), True)


# ---- 3. 분기 요청 · 요청 고치기 ----
def 시험_분기요청():
    요청 = {"messages": [{"role": "user", "content": "q"}], "temperature": 0.7, "top_p": 0.9, "max_tokens": 50, "stop": ["\n\n"],
            "tools": [{"type": "function", "function": {"name": "f"}}], "stream": True}
    x = 요청줄(3, [1, 2, 3], [10, 11, 12, 13], 요청=요청, 시드=1234)
    몸 = d.분기요청(x, 2, 99)
    같다("분기 프롬프트", 몸["prompt"], [1, 2, 3, 10, 11, 99])
    같다("분기 시드", 몸["seed"], 1234)
    같다("분기 step_offset", 몸["geulpu"]["step_offset"], 3)
    같다("분기 echo_from", 몸["geulpu"]["echo_from"], 3)
    같다("분기 echo · 해시 · 프롬프트 토큰", (몸["echo"], 몸["geulpu"]["logit_hash"], 몸["geulpu"]["prompt_tokens"]), (True, True, True))
    같다("분기 샘플링 설정", (몸["temperature"], 몸["top_p"], 몸["stop"]), (0.7, 0.9, ["\n\n"]))
    같다("분기에 없는 설정은 넣지 않는다", ("top_k" in 몸, "min_p" in 몸, "tools" in 몸, "messages" in 몸), (False, False, False, False))
    같다("분기 max_tokens 남은 몫", 몸["max_tokens"], 47)
    같다("분기 스트리밍 끔", 몸["stream"], False)
    같다("스텝 0 분기", d.분기요청(x, 0, 5)["prompt"], [1, 2, 3, 5])
    같다("마지막 스텝 분기의 max_tokens", d.분기요청(x, 3, 5)["max_tokens"], 46)
    짧은 = 요청줄(4, [1], [2, 3], 요청={"messages": [], "max_tokens": 2})
    같다("남은 몫이 없으면 max_tokens 0(점수만)", d.분기요청(짧은, 1, 7)["max_tokens"], 0)
    없음 = 요청줄(5, [1], [2, 3], 요청={"messages": []})
    같다("max_tokens 가 없으면 넣지 않는다", "max_tokens" in d.분기요청(없음, 0, 7), False)
    오류난다("답 밖 스텝", lambda: d.분기요청(x, 4, 1), SystemExit)
    오류난다("음수 스텝", lambda: d.분기요청(x, -1, 1), SystemExit)
    # 대안 토큰 고르기
    s = {"top_ids": [[10, 7], [5, 11], [12, 4], [13, 2]]}
    같다("기본 대안 — 원래가 1위", d.다른토큰(x, s, 0, None, None), 7)
    같다("기본 대안 — 원래가 2위", d.다른토큰(x, s, 1, None, None), 5)
    같다("--token same", d.다른토큰(x, s, 2, "same", None), 12)
    같다("--token 번호", d.다른토큰(x, s, 2, "77", None), 77)
    같다("--alt 2", d.다른토큰(x, s, 3, None, 2), 2)
    오류난다("--alt 범위 밖", lambda: d.다른토큰(x, s, 3, None, 3), SystemExit)
    오류난다("다른 토큰 없음", lambda: d.다른토큰(x, {"top_ids": [[10], [], [], []]}, 0, None, None), SystemExit)


def 시험_경로():
    v = {"messages": [{"role": "system", "content": "s"}, {"role": "tool", "content": "x"}], "temperature": 0.7}
    새 = d.경로바꾸기(v, "messages.1.content", "y")
    같다("경로 바꾸기", 새["messages"][1]["content"], "y")
    같다("원래는 그대로", v["messages"][1]["content"], "x")
    같다("음수 번호", d.경로바꾸기(v, "messages.-1.content", "z")["messages"][1]["content"], "z")
    같다("새 열쇠", d.경로바꾸기(v, "seed", 5)["seed"], 5)
    같다("객체 값", d.경로바꾸기(v, "messages.0", {"role": "system", "content": "n"})["messages"][0], {"role": "system", "content": "n"})
    오류난다("배열 밖", lambda: d.경로바꾸기(v, "messages.5.content", 1), ValueError)
    오류난다("배열에 이름", lambda: d.경로바꾸기(v, "messages.x", 1), ValueError)
    오류난다("없는 열쇠", lambda: d.경로바꾸기(v, "nope.a", 1), ValueError)
    오류난다("글 안으로", lambda: d.경로바꾸기(v, "messages.0.content.z", 1), ValueError)
    같다("--set 수", d.바꿀값("temperature=0"), ("temperature", 0))
    같다("--set 글", d.바꿀값("messages.0.content=안녕 = 반가워"), ("messages.0.content", "안녕 = 반가워"))
    같다("--set JSON", d.바꿀값('a={"x": [1, true]}'), ("a", {"x": [1, True]}))
    같다("--set 따옴표 글", d.바꿀값('a="12"'), ("a", "12"))
    오류난다("--set 꼴", lambda: d.바꿀값("temperature"), SystemExit)
    x = 요청줄(2, [1], [2], 요청={"messages": [{"role": "user", "content": "q"}], "stream": True, "stream_options": {"include_usage": True},
                                  "geulpu": {"store": False}}, 시드=77)
    몸 = d.요청분기요청(x, [("messages.0.content", "q2")])
    같다("요청 분기 — 고친 글", 몸["messages"][0]["content"], "q2")
    같다("요청 분기 — 시드 · 스트리밍", (몸["seed"], 몸["stream"], "stream_options" in 몸), (77, False, False))
    같다("요청 분기 — geulpu 합침", 몸["geulpu"], {"store": False, "logit_hash": True, "prompt_tokens": True})
    같다("요청 분기 — 시드도 고칠 수 있다", d.요청분기요청(x, [("seed", 5)])["seed"], 5)


# ---- 4. echo 응답 풀기 · 서버 · 해시 검사 ----
def 시험_점수():
    P = 3
    r = {"choices": [{"text": "abcXY", "logprobs": {"tokens": ["a", "b", "c", "X", "Y"], "token_logprobs": [None, None, None, -0.5, -1.25],
                                                "top_logprobs": [None, None, None, {}, {}], "text_offset": [0, 1, 2, 3, 4]}}],
         "geulpu": {"prompt_entropy": [None, None, None, 0.7, 1.1], "prompt_top_ids": [None, None, None, [8, 9], [9, 8]],
                    "prompt_top_logprobs": [None, None, None, [-0.5, -1.0], [-0.25, -1.25]], "prompt_logit_hashes": ["h3", "h4"]}}
    s = d.점수풀기(r, P, 2)
    같다("점수 풀기", s, {"logprobs": [-0.5, -1.25], "entropy": [0.7, 1.1], "top_ids": [[8, 9], [9, 8]], "top_logprobs": [[-0.5, -1.0], [-0.25, -1.25]],
                       "token_text": ["X", "Y"], "text": "XY", "text_offset": [0, 1], "logit_hashes": ["h3", "h4"]})
    # 한 글자(한 · 3 바이트)가 토큰 둘에 걸친 답 — 글은 바이트로 이어 붙인 것, 자리는 답의 처음부터
    r2 = {"choices": [{"text": "ab한글", "logprobs": {"tokens": ["a", "b", "�", "�", "글"], "token_logprobs": [None, None, -1.0, -2.0, -0.5],
                                                       "text_offset": [0, 1, 2, 4, 5]}}],
          "geulpu": {"prompt_entropy": [None, None, 1, 2, 3], "prompt_top_ids": [None, None, [1], [2], [3]],
                     "prompt_top_logprobs": [None, None, [-1.0], [-2.0], [-0.5]], "prompt_logit_hashes": ["x", "y", "z"]}}
    s2 = d.점수풀기(r2, 2, 3)
    같다("점수 풀기 — 바이트 조각", (s2["text"], s2["text_offset"], s2["token_text"]), ("한글", [0, 2, 3], ["�", "�", "글"]))
    몸 = d.점수요청([1, 2, 3], [8, 8])
    같다("점수 요청", (몸["prompt"], 몸["echo"], 몸["max_tokens"], 몸["logprobs"], 몸["geulpu"]["echo_from"], 몸["geulpu"]["logit_hash"]),
         ([1, 2, 3, 8, 8], True, 0, 2, 3, True))
    같다("같은 서버", d.서버다름(서버A, dict(서버A)), [])
    같다("다른 모델", len(d.서버다름(서버A, 서버B)), 1)
    같다("빠른 모드는 백엔드도", len(d.서버다름(서버A, dict(서버A, layer="vulkan"))), 1)
    정밀 = dict(서버A, precision="exact")
    같다("정밀 모드는 백엔드가 달라도 같다", d.서버다름(정밀, dict(정밀, layer="vulkan")), [])
    같다("문맥은 보지 않는다(YaRN 배만)", d.서버다름(서버A, dict(서버A, context=8192)), [])
    같다("YaRN 배", len(d.서버다름(서버A, dict(서버A, yarn=2))), 1)
    같다("KV 캐시 8비트", len(d.서버다름(서버A, dict(서버A, kv8=True))), 1)                 # GeulPUv2 ③b-2
    같다("KV 캐시 — 없으면 16비트", d.서버다름(dict(서버A, kv8=False), 서버A), [])
    같다("빠른 어텐션", len(d.서버다름(서버A, dict(서버A, kv8=True, fast_attention=True))), 2)    # GeulPUv2 빠른 어텐션(계약 F2)
    같다("빠른 어텐션 — 없으면 끔", d.서버다름(dict(서버A, fast_attention=False), 서버A), [])
    같다("해시 같음", d.해시검사(서버A, 서버A, ["1", "2"], ["1", "2"]), "same")
    같다("해시 다름", d.해시검사(서버A, 서버A, ["1", "2"], ["1", "3"]), "different")
    같다("다른 모델의 해시", d.해시검사(서버A, 서버B, ["1"], ["9"]), "other_model")
    같다("생성 해시 없음", d.해시검사(서버A, 서버A, None, ["1"]), "no_hashes")
    실 = d.실행(헤더A, [요청줄(0, [1], [2]), 점수줄(0, 서버A, [-0.1], [[2, 3]], [[-0.1, -2.0]]), 점수줄(0, 서버B, [-0.3], [[3, 2]], [[-0.2, -0.3]])])
    같다("점수 찾기 A", d.점수찾기(실, 0, 서버A)["logprobs"], [-0.1])
    같다("점수 찾기 B", d.점수찾기(실, 0, 서버B)["logprobs"], [-0.3])
    같다("점수 찾기 없음", d.점수찾기(실, 1, 서버A), None)
    같다("처음 다른 — 같음", d.처음다른([1, 2], [1, 2]), None)
    같다("처음 다른 — 가운데", d.처음다른([1, 2, 3], [1, 5, 3]), 1)
    같다("처음 다른 — 짧은 쪽", d.처음다른([1, 2], [1, 2, 3]), 2)
    같다("처음 다른 — 빈 것", d.처음다른([], [4]), 0)


# ---- 5. 중단점 ----
def 참조중단점(토큰, s, 마진기준, 확률기준, 도구호출):
    """따로 짠 참조 — (스텝, 종류들)"""
    결과 = []
    for j in range(len(토큰)):
        tl = s["top_logprobs"][j]
        종류 = []
        if len(tl) >= 2 and tl[0] - tl[1] < 마진기준:
            종류.append("low_margin")
        if math.exp(s["logprobs"][j]) < 확률기준:
            종류.append("low_prob")
        if "<tool_call>" in s["token_text"][j] or "<|python_tag|>" in s["token_text"][j]:
            종류.append("tool_call")
        if 종류:
            결과.append((j, 종류))
    if 도구호출 and 토큰 and not any("tool_call" in k for _, k in 결과):
        if 결과 and 결과[0][0] == 0:
            결과[0] = (0, 결과[0][1] + ["tool_call"])
        else:
            결과.insert(0, (0, ["tool_call"]))
    return 결과


def 시험_중단점():
    s = {"logprobs": [-0.01, -0.693, -3.5, -0.05], "top_ids": [[5, 9], [6, 4], [3, 2], [8, 1]],
         "top_logprobs": [[-0.01, -4.6], [-0.693, -0.72], [-0.1, -2.5], [-0.05, -3.0]], "token_text": ["가", "나", "다", "<tool_call>"]}
    bs = d.중단점찾기([5, 6, 7, 8], s, 0.05, 0.05)
    같다("중단점 스텝 · 종류", [(b["step"], b["kinds"]) for b in bs], [(1, ["low_margin"]), (2, ["low_prob"]), (3, ["tool_call"])])
    같다("중단점 마진", round(bs[0]["margin"], 6), 0.027)
    같다("중단점 확률", round(bs[1]["prob"], 6), round(math.exp(-3.5), 6))
    같다("중단점 상위", (bs[0]["top_ids"], [round(p, 6) for p in bs[0]["top_probs"]]), ([6, 4], [round(math.exp(-0.693), 6), round(math.exp(-0.72), 6)]))
    둘 = {"logprobs": [-3.2], "top_ids": [[4, 7]], "top_logprobs": [[-0.70, -0.71]], "token_text": ["x"]}
    같다("한 스텝에 둘", [b["kinds"] for b in d.중단점찾기([7], 둘, 0.05, 0.05)], [["low_margin", "low_prob"]])
    json꼴 = {"logprobs": [-0.01, -0.02], "top_ids": [[1, 2], [3, 4]], "top_logprobs": [[-0.01, -5.0], [-0.02, -6.0]], "token_text": ["{\"", "name"]}
    같다("Llama 3 의 JSON 꼴 호출 — 스텝 0", [(b["step"], b["kinds"]) for b in d.중단점찾기([1, 3], json꼴, 0.05, 0.05, 도구호출=True)], [(0, ["tool_call"])])
    같다("호출이 없으면 없음", d.중단점찾기([1, 3], json꼴, 0.05, 0.05), [])
    같다("기준을 바꾸면", [b["step"] for b in d.중단점찾기([5, 6, 7, 8], s, 3.0, 0.5)], [1, 2, 3])
    rng = random.Random(20261020)
    for 회 in range(200):
        n = rng.randint(0, 60)
        토큰, 로그, 위번호, 위로그, 글 = [], [], [], [], []
        for _ in range(n):
            a1 = -rng.random() * 3
            a2 = a1 - rng.choice([0.0, 0.001, 0.03, 0.2, 1.5, 6.0])
            t = rng.randint(0, 50)
            토큰.append(t)
            로그.append(a1 if rng.random() < 0.7 else a2 - rng.random() * 4)
            위번호.append([t, rng.randint(0, 50)])
            위로그.append([a1, a2])
            글.append(rng.choice(["a", "b", "<tool_call>", "<|python_tag|>", "\n"]) if rng.random() < 0.1 else "w")
        sc = {"logprobs": 로그, "top_ids": 위번호, "top_logprobs": 위로그, "token_text": 글}
        마, 확, 호 = rng.choice([0.01, 0.05, 0.5]), rng.choice([0.01, 0.05, 0.3]), rng.random() < 0.3
        같다(f"중단점 무작위 {회}", [(b["step"], b["kinds"]) for b in d.중단점찾기(토큰, sc, 마, 확, 호)], 참조중단점(토큰, sc, 마, 확, 호))


# ---- 6. 비교 ----
def 참조비교(A, B, 서버a, 서버b):
    """따로 짠 참조 — 요청마다 (i, same, first_divergence, delta_sum, b_top1_differs), 그리고 합"""
    B표 = {y["i"]: y for y in B["요청"]}
    행 = []
    for x in sorted(A["요청"], key=lambda v: v["i"]):
        y = B표.get(x["i"])
        if y is None:
            continue
        ta, tb = x["response"]["geulpu"]["tokens"], y["response"]["geulpu"]["tokens"]
        fd = None
        for k in range(max(len(ta), len(tb))):
            if k >= len(ta) or k >= len(tb) or ta[k] != tb[k]:
                fd = k
                break
        sa = next((s for s in A["점수"] if s["of"] == x["i"] and s["server"]["sha256"] == 서버a["sha256"]), None)
        sb = next((s for s in A["점수"] if s["of"] == x["i"] and s["server"]["sha256"] == 서버b["sha256"]), None)
        if sa and sb:
            합 = 0.0
            for j in range(len(ta)):
                합 += sb["logprobs"][j] - sa["logprobs"][j]
            b1 = sum(1 for j in range(len(ta)) if sb["top_ids"][j][0] != ta[j])
        else:
            합, b1 = None, None
        행.append((x["i"], ta == tb, fd, 합, b1))
    return 행


def 시험_비교():
    # 손으로 정한 실행 — 로그확률은 이진 분수라 합이 정확하다
    A요청 = [요청줄(0, [1, 2], [5, 6, 7]), 요청줄(1, [1, 3], [5, 6, 7, 8]), 요청줄(2, [1, 4], [9, 9, 9]), 요청줄(3, [1, 5], [4])]
    B요청 = [요청줄(0, [1, 2], [5, 6, 7], 서버=서버B), 요청줄(1, [1, 3], [5, 6, 2, 8], 서버=서버B), 요청줄(2, [1, 4], [9, 9], 서버=서버B, 끝="length")]
    A점수 = [점수줄(0, 서버A, [-0.5, -0.25, -0.125], [[5, 1], [6, 1], [7, 1]], [[-0.5, -1], [-0.25, -2], [-0.125, -3]]),
            점수줄(0, 서버B, [-1.0, -0.25, -0.5], [[5, 1], [1, 6], [7, 1]], [[-1, -2], [-0.2, -0.25], [-0.5, -1]]),
            점수줄(1, 서버A, [-0.5, -0.5, -0.5, -0.5], [[5, 0], [6, 0], [7, 0], [8, 0]], [[-0.5, -1]] * 4),
            점수줄(1, 서버B, [-0.25, -0.75, -2.5, -0.5], [[5, 0], [6, 0], [2, 7], [8, 0]], [[-0.25, -1], [-0.75, -1], [-0.1, -2.5], [-0.5, -1]]),
            점수줄(2, 서버A, [-0.125, -0.125, -0.125], [[9, 0]] * 3, [[-0.125, -3]] * 3)]
    A = d.실행(헤더A, A요청 + A점수, 이름="A")
    B = d.실행(헤더B, B요청, 이름="B")
    보고 = d.비교계산(A, B)
    r = {x["i"]: x for x in 보고["requests"]}
    같다("비교 0 같음", (r[0]["same"], r[0]["first_divergence"]), (True, None))
    같다("비교 0 로그확률 차", (r[0]["delta_sum"], r[0]["logprob_a"], r[0]["logprob_b"]), (-0.875, -0.875, -1.75))
    같다("비교 0 B 의 1위", (r[0]["b_top1_differs"], r[0]["a_top1_differs"]), (1, 0))
    같다("비교 0 가장 나빠진 스텝", [w["step"] for w in r[0]["worst"]], [0, 2, 1])
    같다("비교 1 갈라짐", (r[1]["same"], r[1]["first_divergence"]), (False, 2))
    같다("비교 1 차 합 · 평균", (r[1]["delta_sum"], r[1]["delta_mean"]), (-2.0, -0.5))
    같다("비교 1 B 의 1위", r[1]["b_top1_differs"], 1)
    같다("비교 2 짧은 쪽", (r[2]["same"], r[2]["first_divergence"], r[2]["finish_b"]), (False, 2, "length"))
    같다("비교 2 점수 없음", (r[2]["scored"], r[2]["missing"]), (False, ["a_by_b"]))
    같다("비교 3 B 에 없음", r[3]["compared"], False)
    s = 보고["summary"]
    같다("요약 요청", (s["requests"], s["not_compared"], s["same"], s["different"]), (3, 1, 1, 2))
    같다("요약 갈라짐", s["first_divergence"], {"min": 2, "median": 2, "max": 2, "histogram": {"0": 0, "1-4": 2, "5-16": 0, "17-64": 0, "65+": 0}})
    같다("요약 점수", (s["scored"], s["tokens"], s["delta_sum"], s["delta_mean_per_token"], s["b_top1_differs"]), (2, 7, -2.875, -2.875 / 7, 2))
    같다("요약 프롬프트 · 요청", (s["prompt_different"], s["request_different"]), (0, 0))
    같다("같은 입력이면 같은 보고", json.dumps(d.비교계산(A, B), ensure_ascii=False), json.dumps(보고, ensure_ascii=False))
    # 무작위 실행 — 참조와 같은가
    rng = random.Random(7)
    for 회 in range(60):
        n = rng.randint(1, 30)
        Ax, Bx, As = [], [], []
        for i in range(n):
            m = rng.randint(1, 40)
            ta = [rng.randint(0, 9) for _ in range(m)]
            tb = list(ta)
            if rng.random() < 0.6:
                k = rng.randint(0, m - 1)
                tb = tb[:k] + [rng.randint(10, 19)] + [rng.randint(0, 9) for _ in range(rng.randint(0, 20))]
            elif rng.random() < 0.2:
                tb = tb[:rng.randint(0, m)]
            Ax.append(요청줄(i, [1, i], ta))
            if rng.random() < 0.95:
                Bx.append(요청줄(i, [1, i], tb, 서버=서버B))
            for 서버 in (서버A, 서버B):
                if rng.random() < 0.9:
                    lp = [-rng.random() * 5 for _ in range(m)]
                    ids = [[rng.choice([t, t + 1]), 99] for t in ta]
                    As.append(점수줄(i, 서버, lp, ids, [[v, v - 1] for v in lp]))
        A = d.실행(헤더A, Ax + As, 이름="A")
        B = d.실행(헤더B, Bx, 이름="B")
        보고 = d.비교계산(A, B)
        받은 = [(x["i"], x["same"], x["first_divergence"], x.get("delta_sum"), x.get("b_top1_differs")) for x in 보고["requests"] if x["compared"]]
        같다(f"비교 무작위 {회}", 받은, 참조비교({"요청": Ax, "점수": As}, {"요청": Bx}, 서버A, 서버B))
        같다(f"비교 무작위 {회} — 다시 해도 같은 보고", json.dumps(d.비교계산(A, B)), json.dumps(보고))


# ---- 7. 앱 화면(app.js)의 같은 계산 — node 가 있으면 화면의 비교 · 중단점 · 큰 정수 JSON 이 도구와 같은 값인가 ----
def 줄글(헤더, 줄들):
    return "\n".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in [헤더] + 줄들) + "\n"


def 시험_화면():
    node = shutil.which("node")
    if not node:
        print("화면 시험: node 가 없어 건너뜀")
        return
    rng = random.Random(23)
    입력 = {"compare": [], "breaks": [], "big": []}
    기대 = []
    for 회 in range(40):
        n = rng.randint(1, 20)
        Ax, Bx, As = [], [], []
        for i in range(n):
            m = rng.randint(1, 30)
            ta = [rng.randint(0, 9) for _ in range(m)]
            tb = list(ta)
            if rng.random() < 0.6:
                k = rng.randint(0, m - 1)
                tb = tb[:k] + [rng.randint(10, 19)] + [rng.randint(0, 9) for _ in range(rng.randint(0, 10))]
            Ax.append(요청줄(i, [1, i], ta, 상태=200 if rng.random() < 0.95 else 400))
            if rng.random() < 0.9:
                Bx.append(요청줄(i, [1, i] if rng.random() < 0.9 else [2, i], tb, 서버=서버B))
            for 서버 in (서버A, 서버B):
                if rng.random() < 0.85:
                    lp = [-rng.random() * 6 for _ in range(m)]
                    As.append(점수줄(i, 서버, lp, [[rng.choice([t, t + 1]), 99] for t in ta], [[v, v - rng.random()] for v in lp]))
        rng.shuffle(As)
        입력["compare"].append({"a": 줄글(헤더A, Ax + As), "b": 줄글(헤더B, Bx)})
        기대.append(d.비교계산(d.실행(헤더A, Ax + As, 이름="A"), d.실행(헤더B, Bx, 이름="B")))
    기대중단 = []
    for 회 in range(80):
        n = rng.randint(0, 40)
        토큰 = [rng.randint(0, 50) for _ in range(n)]
        a1 = [-rng.random() * 3 for _ in range(n)]
        sc = {"logprobs": [v if rng.random() < 0.7 else v - rng.random() * 4 for v in a1], "top_ids": [[t, 0] for t in 토큰],
              "top_logprobs": [[v, v - rng.choice([0.0, 0.01, 0.04, 0.3, 2.0])] for v in a1],
              "token_text": [rng.choice(["a", "<tool_call>", "x<|python_tag|>", "한"]) if rng.random() < 0.1 else "w" for _ in range(n)]}
        마, 확, 호 = rng.choice([0.01, 0.05, 0.5]), rng.choice([0.01, 0.05, 0.3]), rng.random() < 0.3
        입력["breaks"].append({"tokens": 토큰, "score": sc, "margin": 마, "prob": 확, "tool_call": 호})
        기대중단.append([[b["step"], b["kinds"]] for b in d.중단점찾기(토큰, sc, 마, 확, 호)])
    큰 = ['{"seed":18446744073709551615,"x":[1,-9223372036854775808,0.5],"s":"a\\"b 12345678901234567890"}',
          '{"a":9007199254740993,"b":9007199254740992,"c":123}', '{"t":"\\\\","n":-0.25}']
    입력["big"] = 큰
    with tempfile.TemporaryDirectory() as 폴더:
        p = os.path.join(폴더, "입력.json")
        open(p, "w", encoding="utf-8").write(json.dumps(입력, ensure_ascii=False))
        r = subprocess.run([node, os.path.join(ROOT, "시험", "디버거화면.js"), os.path.join(ROOT, "앱", "web", "app.js"), p],
                           capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        같다("화면 시험 — node 실행", r.stderr[-500:], "")
        return
    받음 = json.loads(r.stdout)
    for 회, (js, py) in enumerate(zip(받음["compare"], 기대)):
        for jr, pr in zip(js["rows"], py["requests"]):
            열 = ["i", "compared"] + (["same", "first_divergence", "steps_a", "steps_b", "scored"] if pr["compared"] else [])
            if pr.get("scored"):
                열 += ["delta_sum", "delta_mean", "b_top1_differs", "a_top1_differs", "logprob_a", "logprob_b"]
            같다(f"화면 비교 {회} 요청 {pr['i']}", {k: jr.get(k) for k in 열}, {k: pr.get(k) for k in 열})
            if pr.get("scored"):
                같다(f"화면 비교 {회} 요청 {pr['i']} 가장 나빠진 스텝", jr["worst"], [w["step"] for w in pr["worst"]])
        ps, jsum = py["summary"], js["summary"]
        같다(f"화면 비교 {회} 요약", {k: jsum[k] for k in ("requests", "same", "different", "prompt_different", "scored", "tokens", "delta_sum", "b_top1_differs", "a_top1_differs")},
             {k: ps[k] for k in ("requests", "same", "different", "prompt_different", "scored", "tokens", "delta_sum", "b_top1_differs", "a_top1_differs")})
        같다(f"화면 비교 {회} 갈라진 스텝", [jsum["fd_min"], jsum["fd_median"], jsum["fd_max"]],
             [ps["first_divergence"]["min"], ps["first_divergence"]["median"], ps["first_divergence"]["max"]])
    for 회, (js, py) in enumerate(zip(받음["breaks"], 기대중단)):
        같다(f"화면 중단점 {회}", js, py)
    for t, js in zip(큰, 받음["big"]):
        같다(f"화면 큰 정수 JSON {t[:30]}", js, t)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for f in (시험_실행, 시험_덧붙이기, 시험_분기요청, 시험_경로, 시험_점수, 시험_중단점, 시험_비교, 시험_화면):
        f()
    for t in 틀림[:20]:
        print("틀림 —", t)
    if 틀림:
        print(f"디버거 도구: 사례 {사례} 가운데 틀림 {len(틀림)}")
        return 1
    print(f"디버거 도구: 사례 {사례} 모두 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
