#!/usr/bin/env python3
"""에이전트 호환 2 단계 문 2 — Anthropic Messages(/v1/messages · /v1/messages/count_tokens)를 불러 둔 모델로 본다(온도 0). 바깥 꾸러미
없이 urllib 로.

  python 시험/앤트로픽.py <주소 — 예: http://127.0.0.1:44650> [--생각끔]

보는 것: 온 답의 꼴(type message · content 블록 · stop_reason · usage), 글 답(end_turn), 도구 호출(tool_use 블록 — input 객체 · id · name,
stop_reason tool_use), 흘림의 사건 차례(message_start → content_block_start/delta/stop … → message_delta → message_stop)와 흘림 = 흘림
아님(블록 · stop_reason), 도구 결과(tool_result)를 돌려주면 글로 답함(is_error 결과도 받음), tool_choice any · tool, system 블록 ·
count_tokens(= 온 답의 input_tokens + cache_read_input_tokens), 문맥을 넘으면 400 'prompt is too long', 잘못된 몸은 Anthropic 꼴 오류.
끝 줄 "모두 맞음"이면 종료 0.
"""
import json
import sys
import urllib.error
import urllib.request

날씨 = {"name": "get_weather", "description": "Get the current weather in a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}
시각 = {"name": "get_time", "description": "Get the current local time in a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}


def post(주소, 길, 몸, 흘림=False):
    r = urllib.request.Request(주소 + 길, data=json.dumps(몸, ensure_ascii=False).encode("utf-8"),
                               headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    try:
        f = urllib.request.urlopen(r, timeout=600)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")
    if not 흘림:
        return 200, json.load(f)
    # SSE(event: · data:)를 모아 사건 목록과 온 답 꼴로
    사건들, 이름 = [], None
    for raw in f:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("event:"):
            이름 = line[6:].strip()
        elif line.startswith("data:"):
            d = json.loads(line[5:].strip())
            사건들.append((이름, d))
    블록들, 답 = {}, {"content": [], "stop_reason": None, "usage": {}}
    for 이름, d in 사건들:
        t = d.get("type")
        if t == "message_start":
            답["usage"].update(d["message"].get("usage", {}))
        elif t == "content_block_start":
            b = dict(d["content_block"])
            if b["type"] == "tool_use":
                b["_json"] = ""
            블록들[d["index"]] = b
        elif t == "content_block_delta":
            b, dl = 블록들[d["index"]], d["delta"]
            if dl["type"] == "text_delta":
                b["text"] += dl["text"]
            elif dl["type"] == "input_json_delta":
                b["_json"] += dl["partial_json"]
        elif t == "message_delta":
            답["stop_reason"] = d["delta"].get("stop_reason")
            답["usage"].update(d.get("usage", {}))
    for k in sorted(블록들):
        b = 블록들[k]
        if b["type"] == "tool_use":
            b["input"] = json.loads(b.pop("_json") or "{}")
        답["content"].append(b)
    답["_사건"] = [이름 for 이름, _ in 사건들]
    return 200, 답


def 호출들(답):
    return [(b["name"], b["input"]) for b in 답.get("content", []) if b.get("type") == "tool_use"]


def 글(답):
    return "".join(b.get("text", "") for b in 답.get("content", []) if b.get("type") == "text")


def main():
    주소 = sys.argv[1]
    틀림 = []
    기본 = {"model": "local", "max_tokens": 512, "temperature": 0}
    if "--생각끔" in sys.argv:
        기본["chat_template_kwargs"] = {"enable_thinking": False}

    def 본(이름, ok, 덧=""):
        print(f"{'맞음' if ok else '틀림'}  {이름}{(' — ' + 덧) if 덧 else ''}")
        if not ok:
            틀림.append(이름)

    # 1 글 답
    인사 = [{"role": "user", "content": "Say hello in one word."}]
    코드, a = post(주소, "/v1/messages", dict(기본, messages=인사))
    본("글 답 200 · type message · end_turn", 코드 == 200 and a.get("type") == "message" and a.get("role") == "assistant"
       and a.get("stop_reason") == "end_turn" and bool(글(a).strip()), str(a)[:240])
    # 생각 끔(thinking disabled → 틀의 enable_thinking 거짓)이면 생각 글 없이 짧게
    코드1b, a2 = post(주소, "/v1/messages", dict(기본, messages=인사, thinking={"type": "disabled"}))
    print(f"      글 답 — 생각 기본: 생성 {a.get('usage', {}).get('output_tokens')} 토큰 {글(a)[:60]!r} · thinking disabled: 생성 "
          f"{a2.get('usage', {}).get('output_tokens')} 토큰 {글(a2)[:60]!r}")
    u = a.get("usage", {})
    본("usage 칸", all(k in u for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")), str(u))
    # 2 도구 호출 — 온 답 · 흘림
    물음 = [{"role": "user", "content": "What's the weather in Seoul right now?"}]
    코드2, b = post(주소, "/v1/messages", dict(기본, messages=물음, tools=[날씨]))
    코드3, c = post(주소, "/v1/messages", dict(기본, messages=물음, tools=[날씨], stream=True), 흘림=True)
    cb = 호출들(b)
    본("tool_use 블록 · stop_reason tool_use", 코드2 == 200 and b.get("stop_reason") == "tool_use" and len(cb) >= 1 and cb[0][0] == "get_weather"
       and isinstance(cb[0][1], dict) and "city" in cb[0][1], str(b.get("content"))[:240])
    본("흘림 = 흘림 아님(호출 · stop_reason)", 코드3 == 200 and 호출들(c) == cb and c.get("stop_reason") == b.get("stop_reason"), str(호출들(c)))
    ev = c.get("_사건", [])
    차례 = ev[:1] == ["message_start"] and ev[-2:] == ["message_delta", "message_stop"] and "content_block_start" in ev
    본("흘림 사건 차례", 차례, " ".join(ev[:12]))
    # 3 도구 결과를 돌려주면 글로(is_error 결과도)
    if cb:
        tu = [x for x in b["content"] if x["type"] == "tool_use"][0]
        둘째 = 물음 + [{"role": "assistant", "content": b["content"]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu["id"],
                                                  "content": json.dumps({"city": "Seoul", "temp_c": 21, "sky": "clear"})}]}]
        코드4, d = post(주소, "/v1/messages", dict(기본, messages=둘째, tools=[날씨]))
        본("결과 뒤 글 답", 코드4 == 200 and d.get("stop_reason") == "end_turn" and bool(글(d).strip()), 글(d)[:160] if 코드4 == 200 else str(d)[:200])
        셋째 = 물음 + [{"role": "assistant", "content": b["content"]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu["id"], "is_error": True,
                                                  "content": [{"type": "text", "text": "Error: weather service unavailable (permission denied)."}]},
                                                 {"type": "text", "text": "Please answer anyway."}]}]
        코드5, e = post(주소, "/v1/messages", dict(기본, messages=셋째, tools=[날씨]))
        본("is_error 결과 · 글 블록 함께 받음", 코드5 == 200, str(e)[:200] if 코드5 != 200 else f"{e.get('stop_reason')} · {글(e)[:100]!r}")
    # 4 tool_choice any · tool
    코드6, g = post(주소, "/v1/messages", dict(기본, messages=인사, tools=[날씨], tool_choice={"type": "any"}))
    본("tool_choice any → tool_use", 코드6 == 200 and 호출들(g)[:1] and 호출들(g)[0][0] == "get_weather", str(호출들(g)) if 코드6 == 200 else str(g)[:200])
    코드7, h = post(주소, "/v1/messages", dict(기본, messages=물음, tools=[날씨, 시각], tool_choice={"type": "tool", "name": "get_time"}))
    본("tool_choice tool → 그 이름", 코드7 == 200 and 호출들(h)[:1] and 호출들(h)[0][0] == "get_time", str(호출들(h)) if 코드7 == 200 else str(h)[:200])
    # 5 system 블록 · count_tokens(= 같은 요청의 input + cache_read)
    시 = [{"type": "text", "text": "You are a terse assistant."}, {"type": "text", "text": "Answer in at most five words.", "cache_control": {"type": "ephemeral"}}]
    몸s = dict(기본, system=시, messages=물음, tools=[날씨])
    코드8, k = post(주소, "/v1/messages", 몸s)
    코드9, m = post(주소, "/v1/messages/count_tokens", {kk: vv for kk, vv in 몸s.items() if kk not in ("max_tokens", "temperature")})
    합 = k.get("usage", {}).get("input_tokens", 0) + k.get("usage", {}).get("cache_read_input_tokens", 0) if 코드8 == 200 else -1
    본("system 블록 · count_tokens = input + cache_read", 코드8 == 200 and 코드9 == 200 and m.get("input_tokens") == 합, f"count {m} · usage {k.get('usage')}")
    # 6 오류 — 문맥 넘침 · 잘못된 몸
    긴 = [{"role": "user", "content": "word " * 200000}]
    코드10, n = post(주소, "/v1/messages", dict(기본, messages=긴))
    본("문맥 넘침 → 400 prompt is too long", 코드10 == 400 and n.get("type") == "error" and "prompt is too long" in n.get("error", {}).get("message", ""), str(n)[:200])
    코드11, o = post(주소, "/v1/messages", dict(기본, messages="hi"))
    본("잘못된 몸 → Anthropic 꼴 400", 코드11 == 400 and o.get("type") == "error" and o.get("error", {}).get("type") == "invalid_request_error", str(o)[:200])
    print(f"      생각 글이 답에 섞이지 않음: {'<think>' not in 글(a) and '<think>' not in 글(d if cb else a)}")
    print("모두 맞음" if not 틀림 else f"틀림 {len(틀림)}: {틀림}")
    sys.exit(1 if 틀림 else 0)


if __name__ == "__main__":
    main()
