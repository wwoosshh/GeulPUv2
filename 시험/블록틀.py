#!/usr/bin/env python3
"""0.25.0 25-2 — Inkling 2 의 답 블록 읽기(엔진/채팅틀.gl 의 블록걸음 · 블록호출풀기 — 엔진/시험_블록틀.gl)를 GPU · 어휘 없이 본다. 사례마다 생각(reasoning_content) ·
내용(content) · 도구 호출(이름 · 인자)이 기대와 같은가.

기대의 근거:
  · llama.cpp — b12662 의 tests/test-chat.cpp 가 Inkling 2(thinkingmachines-Inkling-2-Small.jinja)에 거는 사례 그대로(reasoning_format auto —
    GeulPU 의 서버는 Inkling 2 의 생각을 늘 reasoning_content 로 뗀다. 생각을 떼지 않는 reasoning_format none 의 사례는 GeulPU 에 그 선택이
    없어 넣지 않았다). 인자는 JSON 값으로 견준다(llama.cpp 는 모델이 쓴 글 그대로, GeulPU 는 다시 쓴 글 — 뜻이 같은지).
  · GeulPU — llama.cpp 의 문법이 받지 않는 꼴(닫지 않은 호출 · 풀 수 없는 호출 · 이름만 있는 블록 · 블록 밖의 글 등)을 GeulPU 가 읽는 규칙.
    GeulPU 는 생성을 문법으로 묶지 않아 이런 답을 받을 수 있다 — 글을 잃지 않게 내용으로 돌린다.
토큰 조각은 두 방식으로 나눈다: 표지 사이의 글을 통째로 한 토큰, 글자마다 한 토큰 — 둘의 결과가 같아야 한다.

  python -I 시험/블록틀.py <시험_블록틀 실행 파일>

끝 줄 "블록틀: 모두 맞음 (N 가지)"이면 종료 0, 아니면 1.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

H = "<|message_model|>"
E = "<|end_message|>"
TH = "<|content_thinking|>"
TX = "<|content_text|>"
TJ = "<|content_invoke_tool_json|>"
CALL = "special_function" + TJ + '{"name":"special_function","args":{"arg1":1}}' + E
HELLO = "Hello, world!\nWhat's up?"

# (근거, 아이디, 답의 글, 기대 생각, 기대 내용, 기대 호출 [(이름, 인자 값)])
사례들 = [
    ("llama.cpp", "내용만", H + TX + HELLO + E, "", HELLO, []),
    ("llama.cpp", "생각 · 내용", H + TH + "I'm thinking" + E + H + TX + HELLO + E, "I'm thinking", HELLO, []),
    ("llama.cpp", "생각 중에 잘림", H + TH + "I was still thinking", "I was still thinking", "", []),
    ("llama.cpp", "내용 중에 잘림", H + TH + "I'm thinking" + E + H + TX + "Hello, wor", "I'm thinking", "Hello, wor", []),
    ("llama.cpp", "호출 하나", H + CALL, "", "", [("special_function", {"arg1": 1})]),
    ("llama.cpp", "생각 · 내용 · 호출", H + TH + "I should call it" + E + H + TX + "On it." + E + H + CALL,
     "I should call it", "On it.", [("special_function", {"arg1": 1})]),
    ("llama.cpp", "도구가 있어도 글로 답함", H + TH + "No tool needed" + E + H + TX + HELLO + E, "No tool needed", HELLO, []),
    ("llama.cpp", "호출 둘(블록마다)",
     H + CALL + H + "special_function_with_opt" + TJ + '{"name":"special_function_with_opt","args":{"arg1":1,"arg2":2}}' + E,
     "", "", [("special_function", {"arg1": 1}), ("special_function_with_opt", {"arg1": 1, "arg2": 2})]),
    ("llama.cpp", "강제 호출(생각 뒤)", H + TH + "I should call it" + E + H + CALL, "I should call it", "", [("special_function", {"arg1": 1})]),
    ("llama.cpp", "이름 없이 도구 토큰부터(연구 모델)", H + TJ + '{"name":"special_function","args":{"arg1":1}}' + E,
     "", "", [("special_function", {"arg1": 1})]),
    ("llama.cpp", "목록에 없는 도구 · JSON 의 빈칸", H + "unknown_tool" + TJ + '{"name": "unknown_tool", "args": {"arg1": 1}}' + E,
     "", "", [("unknown_tool", {"arg1": 1})]),
    ("llama.cpp", "이름은 JSON 의 name", H + "special" + TJ + '{"name":"special_function","args":{"arg1":1}}' + E,
     "", "", [("special_function", {"arg1": 1})]),
    ("llama.cpp", "응답 형식(JSON 글)", H + TH + "I'm thinking" + E + H + TX + '{"amount": 123.45}' + E, "I'm thinking", '{"amount": 123.45}', []),
    # GeulPU 의 규칙(llama.cpp 의 문법이 받지 않는 답)
    ("GeulPU", "닫지 않은 호출(끝에서)", H + "f" + TJ + '{"name":"f","args":{', "", 'f{"name":"f","args":{', []),
    ("GeulPU", "풀 수 없는 호출", H + "f" + TJ + "{not json}" + E, "", "f{not json}", []),
    ("GeulPU", "JSON 뒤에 다른 글", H + "f" + TJ + '{"name":"f","args":{}} extra' + E, "", 'f{"name":"f","args":{}} extra', []),
    ("GeulPU", "name 이 없는 호출", H + TJ + '{"args":{}}' + E, "", '{"args":{}}', []),
    ("GeulPU", "인자 없는 호출", H + "g" + TJ + '{"name":"g"}' + E, "", "", [("g", {})]),
    ("GeulPU", "이름만 있고 호출 없음", H + "hello" + E, "", "hello", []),
    ("GeulPU", "닫지 않은 호출 뒤 새 머리", H + "f" + TJ + '{"name":"f"' + H + TX + "x" + E, "", 'f{"name":"f"x', []),
    ("GeulPU", "블록 밖의 글", "plain text", "", "plain text", []),
    ("GeulPU", "닫지 않은 생각 뒤 새 머리", H + TH + "a" + H + TX + "b" + E, "a", "b", []),
    ("GeulPU", "모르는 꼴의 블록", H + "<|content_xml|>x" + E, "", "x", []),
    ("GeulPU", "생각 둘 · 글 둘", H + TH + "a" + E + H + TX + "b" + E + H + TH + "c" + E + H + TX + "d" + E, "ac", "bd", []),
    ("GeulPU", "호출 안의 다른 제어 토큰은 버림", H + "f" + TJ + '{"name":"f",' + TX + '"args":{"k":"v"}}' + E, "", "", [("f", {"k": "v"})]),
    ("GeulPU", "호출 뒤의 글 블록", H + CALL + H + TX + "done" + E, "", "done", [("special_function", {"arg1": 1})]),
    ("GeulPU", "한국어 · 줄바꿈 · 따옴표", H + TH + "생각:\n\"따옴\"" + E + H + "w" + TJ + '{"name":"w","args":{"t":"줄\\n바꿈 \\"x\\""}}' + E,
     '생각:\n"따옴"', "", [("w", {"t": '줄\n바꿈 "x"'})]),
]

표지 = re.compile(r"(<\|[^|<>]+\|>)")


def 조각들(글, 글자마다):
    out = []
    for p in 표지.split(글):
        if not p:
            continue
        if 표지.fullmatch(p) or not 글자마다:
            out.append(p)
        else:
            out.extend(list(p))
    return out


def 돌리기(exe, 글자마다):
    with tempfile.TemporaryDirectory() as d:
        i, o = os.path.join(d, "사례.json"), os.path.join(d, "출력.json")
        json.dump([{"id": 아이디, "pieces": 조각들(글, 글자마다)} for _, 아이디, 글, *_ in 사례들],
                  open(i, "w", encoding="utf-8"), ensure_ascii=False)
        r = subprocess.run([exe, i, o], capture_output=True)
        if r.returncode != 0:
            raise SystemExit(f"시험_블록틀 실패: {(r.stdout + r.stderr).decode('utf-8', 'replace')[-400:]}")
        return json.load(open(o, encoding="utf-8"))


def main():
    exe = sys.argv[1]
    틀림 = 0
    결과들 = {갈: 돌리기(exe, 갈) for 갈 in (False, True)}
    for 근거, 아이디, _, 생각, 내용, 호출 in 사례들:
        for 갈, 결과 in 결과들.items():
            g = 결과.get(아이디)
            얻은호출 = [(c["name"], c["arguments"]) for c in (g or {}).get("calls", [])]
            맞음 = g is not None and g["reasoning"] == 생각 and g["content"] == 내용 and 얻은호출 == 호출
            if not 맞음:
                틀림 += 1
                print(f"틀림 [{근거}] {아이디}{' (글자마다)' if 갈 else ''}: 기대 생각 {생각!r} · 내용 {내용!r} · 호출 {호출} — 얻음 {g}")
    n = len(사례들)
    llama = sum(1 for c in 사례들 if c[0] == "llama.cpp")
    if 틀림:
        print(f"블록틀: 틀림 {틀림} (사례 {n} × 조각 두 방식)")
        sys.exit(1)
    print(f"블록틀: 모두 맞음 ({n} 가지 — llama.cpp 의 사례 {llama} · GeulPU 의 규칙 {n - llama}, 조각 두 방식)")


if __name__ == "__main__":
    main()
