#!/usr/bin/env python3
"""GeulPUv2 ②단계 — 답에서 도구 호출을 읽는가(불러 둔 모델로, 온도 0). 바깥 꾸러미 없이 urllib 로.

  python 시험/도구호출.py <주소 — 예: http://127.0.0.1:44639> [--생각끔 | --생각켬] [--여럿]

보는 것: 흘림 아님 · 흘림이 같은 호출(이름 · 인자 JSON)과 같은 content · reasoning_content, finish_reason tool_calls, 인자가 맞는 JSON 객체,
결과를 돌려주면 글로 답함, parallel_tool_calls 거짓이면 호출 하나, tool_choice none 이면 호출 없음, --여럿이면 호출 둘(Qwen — 모델에
따라 하나일 수 있어 기록만). --생각켬(0.13.0 — 틀의 기본이 생각 끔인 Gemma 4)은 enable_thinking 참. 끝 줄 "모두 맞음"이면 종료 0.
에이전트 호환(문서/에이전트.md): tool_choice required 는 도구와 상관없는 물음에도 호출(흘림 = 흘림 아님), 함수 지정은 그 이름의 호출,
tools 에 없는 이름 · tools 없는 required 는 400.
"""
import json
import sys
import urllib.error
import urllib.request

날씨 = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}
시각 = {"type": "function", "function": {"name": "get_time", "description": "Get the current local time in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}


def post(주소, 몸, 흘림=False):
    r = urllib.request.Request(주소 + "/v1/chat/completions", data=json.dumps(몸, ensure_ascii=False).encode("utf-8"),
                               headers={"Content-Type": "application/json"})
    try:
        f = urllib.request.urlopen(r, timeout=600)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")
    if not 흘림:
        return 200, json.load(f)
    # SSE 를 모아 온 응답 꼴로
    content, reasoning, calls, finish = "", "", {}, None
    for raw in f:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            break
        j = json.loads(d)
        for ch in j.get("choices", []):
            dl = ch.get("delta", {})
            content += dl.get("content") or ""
            reasoning += dl.get("reasoning_content") or ""
            for tc in dl.get("tool_calls") or []:
                c = calls.setdefault(tc["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                c["id"] += tc.get("id", "")
                fn = tc.get("function", {})
                c["function"]["name"] += fn.get("name", "")
                c["function"]["arguments"] += fn.get("arguments", "")
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    msg = {"role": "assistant", "content": content or None}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = [calls[k] for k in sorted(calls)]
    return 200, {"choices": [{"message": msg, "finish_reason": finish}]}


def 호출들(답):
    m = 답["choices"][0]["message"]
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in m.get("tool_calls") or []]


def main():
    주소 = sys.argv[1]
    생각끔 = "--생각끔" in sys.argv
    여럿 = "--여럿" in sys.argv
    틀림 = []
    기본 = {"temperature": 0, "max_tokens": 1024, "tools": [날씨]}
    if 생각끔:
        기본["chat_template_kwargs"] = {"enable_thinking": False}
    if "--생각켬" in sys.argv:
        기본["chat_template_kwargs"] = {"enable_thinking": True}

    def 본(이름, ok, 덧=""):
        print(f"{'맞음' if ok else '틀림'}  {이름}{(' — ' + 덧) if 덧 else ''}")
        if not ok:
            틀림.append(이름)

    물음 = [{"role": "user", "content": "What's the weather in Seoul right now?"}]
    코드, a = post(주소, dict(기본, messages=물음))
    코드2, b = post(주소, dict(기본, messages=물음, stream=True), 흘림=True)
    본("흘림 아님 200", 코드 == 200, str(a)[:200] if 코드 != 200 else "")
    if 코드 != 200:
        print("멈춤")
        sys.exit(1)
    ma, mb = a["choices"][0]["message"], b["choices"][0]["message"]
    ca = 호출들(a)
    본("finish_reason tool_calls", a["choices"][0]["finish_reason"] == "tool_calls", f"{a['choices'][0]['finish_reason']} · content {str(ma.get('content'))[:120]!r}")
    본("호출 get_weather(city)", len(ca) >= 1 and ca[0][0] == "get_weather" and isinstance(ca[0][1], dict) and "city" in ca[0][1], str(ca))
    본("흘림 = 흘림 아님(호출)", 호출들(b) == ca, f"{호출들(b)}")
    본("흘림 = 흘림 아님(content)", (mb.get("content") or "") == (ma.get("content") or ""), f"{mb.get('content')!r} / {ma.get('content')!r}")
    본("흘림 = 흘림 아님(reasoning)", mb.get("reasoning_content", "") == ma.get("reasoning_content", ""))
    print(f"      reasoning {len(ma.get('reasoning_content', ''))} 자 · 아이디 {[c['id'] for c in ma.get('tool_calls') or []]}")
    # 결과를 돌려주면 글로 답한다
    if ca:
        둘째 = 물음 + [{"role": "assistant", "content": ma.get("content"), "tool_calls": ma["tool_calls"]},
                    {"role": "tool", "tool_call_id": ma["tool_calls"][0]["id"], "content": json.dumps({"city": "Seoul", "temp_c": 21, "sky": "clear"})}]
        코드3, c = post(주소, dict(기본, messages=둘째))
        본("결과 뒤 글 답", 코드3 == 200 and c["choices"][0]["finish_reason"] == "stop" and bool(c["choices"][0]["message"].get("content")),
           str(c["choices"][0]["message"].get("content"))[:160] if 코드3 == 200 else str(c)[:200])
    # parallel_tool_calls 거짓 · 여럿
    물음2 = [{"role": "user", "content": "What's the weather in Seoul and in Busan? Check both cities."}]
    코드4, d = post(주소, dict(기본, messages=물음2, parallel_tool_calls=False))
    본("parallel_tool_calls 거짓 → 호출 하나 이하", 코드4 == 200 and len(호출들(d)) <= 1, str(호출들(d)) if 코드4 == 200 else str(d)[:200])
    if 여럿:
        코드5, e = post(주소, dict(기본, messages=물음2))
        print(f"기록  여럿(parallel 기본 참): {호출들(e) if 코드5 == 200 else e}")
    # tool_choice none
    코드6, g = post(주소, dict(기본, messages=물음, tool_choice="none"))
    본("tool_choice none → 호출 없음", 코드6 == 200 and not 호출들(g) and g["choices"][0]["finish_reason"] != "tool_calls",
       str(g["choices"][0]["message"].get("content"))[:120] if 코드6 == 200 else str(g)[:200])
    # 에이전트 호환 — tool_choice required: 도구와 상관없는 물음에도 호출(흘림 = 흘림 아님)
    인사 = [{"role": "user", "content": "Say hello in one word."}]
    코드7, h = post(주소, dict(기본, messages=인사, tool_choice="required"))
    코드8, h2 = post(주소, dict(기본, messages=인사, tool_choice="required", stream=True), 흘림=True)
    본("tool_choice required → 호출(도구가 하나면 그 이름) · 호출 밖의 글 없음",
       코드7 == 200 and len(호출들(h)) >= 1 and 호출들(h)[0][0] == "get_weather" and h["choices"][0]["finish_reason"] == "tool_calls"
       and not (h["choices"][0]["message"].get("content") or "").strip(),
       f"{호출들(h)} · content {str(h['choices'][0]['message'].get('content'))[:120]!r}" if 코드7 == 200 else str(h)[:200])
    본("tool_choice required 흘림 = 흘림 아님", 코드8 == 200 and 코드7 == 200 and 호출들(h2) == 호출들(h), str(호출들(h2)) if 코드8 == 200 else str(h2)[:200])
    코드7b, h3 = post(주소, dict(기본, messages=인사, tools=[날씨, 시각], tool_choice="required"))
    print(f"기록  required · 도구 둘(이름은 모델이 고름): {호출들(h3) if 코드7b == 200 else h3} · "
          f"content {str(h3['choices'][0]['message'].get('content'))[:80]!r}" if 코드7b == 200 else f"기록  required · 도구 둘: {h3}")
    # 함수 지정 — 날씨를 묻지만 get_time 을 고르게 한다
    고름 = {"type": "function", "function": {"name": "get_time"}}
    코드9, k = post(주소, dict(기본, messages=물음, tools=[날씨, 시각], tool_choice=고름))
    본("tool_choice 함수 지정 → 그 이름", 코드9 == 200 and len(호출들(k)) >= 1 and 호출들(k)[0][0] == "get_time", str(호출들(k)) if 코드9 == 200 else str(k)[:200])
    코드10, m = post(주소, dict(기본, messages=물음, tool_choice={"type": "function", "function": {"name": "no_such_tool"}}))
    본("tool_choice 없는 이름 → 400", 코드10 == 400, str(m)[:160])
    없이 = {k2: v for k2, v in 기본.items() if k2 != "tools"}
    코드11, n = post(주소, dict(없이, messages=물음, tool_choice="required"))
    본("tools 없는 required → 400", 코드11 == 400, str(n)[:160])
    print("모두 맞음" if not 틀림 else f"틀림 {len(틀림)}: {틀림}")
    sys.exit(1 if 틀림 else 0)


if __name__ == "__main__":
    main()
