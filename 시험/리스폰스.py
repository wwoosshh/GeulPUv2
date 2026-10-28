#!/usr/bin/env python3
"""에이전트 호환 2 단계 문 3 — OpenAI Responses(/v1/responses — Codex CLI, 상태 없이)를 불러 둔 모델로 본다(온도 0). 바깥 꾸러미 없이
urllib 로.

  python 시험/리스폰스.py <주소 — 예: http://127.0.0.1:44650>

보는 것: 온 답의 꼴(object response · status completed · output 의 message · function_call · usage 의 input · output · total), 글 답,
도구 호출(function_call — call_id · name · arguments JSON), 흘림의 사건(response.created … response.completed — sequence_number 가 0 부터
하나씩, response.completed 에 id · usage)과 흘림 = 흘림 아님(output_item.done 의 항목), function_call_output 을 돌려주면 글로 답함,
Codex 꼴의 기본 지시문(첫 developer 메시지) · 글 input, tool_choice required · 함수 지정, previous_response_id 는 400.
끝 줄 "모두 맞음"이면 종료 0.
"""
import json
import sys
import urllib.error
import urllib.request

날씨 = {"type": "function", "name": "get_weather", "description": "Get the current weather in a city.", "strict": False,
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}
시각 = {"type": "function", "name": "get_time", "description": "Get the current local time in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}
검색 = {"type": "web_search"}           # Codex 가 보내는 서버 도구 — 빼야 한다


def post(주소, 몸, 흘림=False):
    r = urllib.request.Request(주소 + "/v1/responses", data=json.dumps(몸, ensure_ascii=False).encode("utf-8"),
                               headers={"Content-Type": "application/json"})
    try:
        f = urllib.request.urlopen(r, timeout=600)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")
    if not 흘림:
        return 200, json.load(f)
    사건들 = []
    for raw in f:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            사건들.append(json.loads(line[5:].strip()))
    끝 = [d for d in 사건들 if d.get("type") in ("response.completed", "response.incomplete")]
    답 = dict(끝[-1]["response"]) if 끝 else {}
    답["_항목"] = [d["item"] for d in 사건들 if d.get("type") == "response.output_item.done"]
    답["_사건"] = 사건들
    return 200, 답


def 호출들(항목들):
    return [(x["name"], json.loads(x["arguments"])) for x in 항목들 if x.get("type") == "function_call"]


def 글(항목들):
    return "".join(c.get("text", "") for x in 항목들 if x.get("type") == "message" for c in x.get("content", []))


def main():
    주소 = sys.argv[1]
    틀림 = []
    기본 = {"model": "local", "max_output_tokens": 512, "temperature": 0, "store": False,
            "reasoning": {"effort": "medium", "summary": "auto"}, "include": ["reasoning.encrypted_content"], "prompt_cache_key": "x"}

    def 본(이름, ok, 덧=""):
        print(f"{'맞음' if ok else '틀림'}  {이름}{(' — ' + 덧) if 덧 else ''}")
        if not ok:
            틀림.append(이름)

    # 1 글 답(글 input · instructions)
    코드, a = post(주소, dict(기본, instructions="You are terse.", input="Say hello in one word."))
    본("글 답 200 · object response · completed", 코드 == 200 and a.get("object") == "response" and a.get("status") == "completed"
       and bool(글(a.get("output", [])).strip()), str(a)[:240])
    u = a.get("usage", {})
    본("usage 칸", all(k in u for k in ("input_tokens", "output_tokens", "total_tokens")) and u["total_tokens"] == u["input_tokens"] + u["output_tokens"], str(u))
    # 2 도구 호출 — Codex 꼴(첫 developer 메시지 · 서버 도구 섞임) — 온 답 · 흘림
    물음 = [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "You are a coding agent."}]},
           {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's the weather in Seoul right now?"}]}]
    코드2, b = post(주소, dict(기본, input=물음, tools=[날씨, 검색], tool_choice="auto", parallel_tool_calls=True))
    코드3, c = post(주소, dict(기본, input=물음, tools=[날씨, 검색], tool_choice="auto", parallel_tool_calls=True, stream=True), 흘림=True)
    cb = 호출들(b.get("output", []))
    본("function_call 항목", 코드2 == 200 and len(cb) >= 1 and cb[0][0] == "get_weather" and "city" in cb[0][1], str(b.get("output"))[:240])
    본("흘림 = 흘림 아님(항목의 호출)", 코드3 == 200 and 호출들(c.get("_항목", [])) == cb, str(호출들(c.get("_항목", []))))
    ev = c.get("_사건", [])
    순번 = [d.get("sequence_number") for d in ev]
    본("흘림 사건 — created 로 시작 · completed 로 끝 · 순번", bool(ev) and ev[0].get("type") == "response.created"
       and ev[-1].get("type") == "response.completed" and 순번 == list(range(len(ev))) and bool(c.get("id")) and "usage" in c,
       " ".join(d.get("type", "")[9:] for d in ev[:9]))
    # 3 결과를 돌려주면 글로
    if cb:
        fc = [x for x in b["output"] if x["type"] == "function_call"][0]
        둘째 = 물음 + [{"type": "function_call", "call_id": fc["call_id"], "name": fc["name"], "arguments": fc["arguments"]},
                    {"type": "function_call_output", "call_id": fc["call_id"], "output": json.dumps({"city": "Seoul", "temp_c": 21, "sky": "clear"})}]
        코드4, d = post(주소, dict(기본, input=둘째, tools=[날씨]))
        본("결과 뒤 글 답", 코드4 == 200 and d.get("status") == "completed" and bool(글(d.get("output", [])).strip()) and not 호출들(d.get("output", [])),
           글(d.get("output", []))[:160] if 코드4 == 200 else str(d)[:200])
    # 4 tool_choice
    인사 = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say hello in one word."}]}]
    코드5, e = post(주소, dict(기본, input=인사, tools=[날씨], tool_choice="required"))
    본("tool_choice required → 호출", 코드5 == 200 and 호출들(e.get("output", []))[:1] and 호출들(e["output"])[0][0] == "get_weather",
       str(호출들(e.get("output", []))) if 코드5 == 200 else str(e)[:200])
    코드6, g = post(주소, dict(기본, input=물음, tools=[날씨, 시각], tool_choice={"type": "function", "name": "get_time"}))
    본("tool_choice 함수 지정 → 그 이름", 코드6 == 200 and 호출들(g.get("output", []))[:1] and 호출들(g["output"])[0][0] == "get_time",
       str(호출들(g.get("output", []))) if 코드6 == 200 else str(g)[:200])
    # 5 상태 없음
    코드7, h = post(주소, dict(기본, input="hi", previous_response_id="resp_x"))
    본("previous_response_id → 400", 코드7 == 400 and "error" in h, str(h)[:160])
    print(f"      생각 글이 답에 섞이지 않음: {'<think>' not in 글(a.get('output', []))} · 글 답 {글(a.get('output', []))[:40]!r}")
    print("모두 맞음" if not 틀림 else f"틀림 {len(틀림)}: {틀림}")
    sys.exit(1 if 틀림 else 0)


if __name__ == "__main__":
    main()
