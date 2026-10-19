#!/usr/bin/env python3
"""GeulPUv2 ②단계 — 도구(tools)를 낀 채팅 틀 대조: 서버의 /tokenize(messages · tools · tool_choice — 채팅 요청과 같은 틀)의 토큰이
GGUF 의 chat_template 을 jinja2 로 그린 참조 글(시험/도구틀참조.py)의 토큰과 같은가. 참조가 오류면(예: Llama 3 의 호출 둘) 서버도 400 이어야 한다.

  python 시험/도구틀.py <주소 — 예: http://127.0.0.1:44639> <참조.json> 시험/도구틀경우.json [--생각끔]

토큰은 둘 다 서버의 같은 나누기(특수 토큰의 글은 특수 토큰으로)라, 같은 토큰 = 같은 글이다. 끝 줄 "모두 같음" 이면 종료 0.
"""
import json
import sys
import urllib.error
import urllib.request


def post(주소, 길, 몸):
    r = urllib.request.Request(주소 + 길, data=json.dumps(몸, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=120) as f:
            return 200, json.load(f)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def main():
    주소, 참조파일, 경우파일 = sys.argv[1:4]
    생각끔 = "--생각끔" in sys.argv[4:]
    참조 = json.load(open(참조파일, encoding="utf-8"))
    경우 = json.load(open(경우파일, encoding="utf-8"))
    도구표 = 경우["도구"]
    틀림 = 0
    for c in 경우["경우"]:
        if c["id"] not in 참조:
            continue
        몸 = {"messages": c["messages"]}
        if c.get("tools"):
            몸["tools"] = [도구표[t] for t in c["tools"]]
        if c.get("tool_choice"):
            몸["tool_choice"] = c["tool_choice"]
        if 생각끔:
            몸["chat_template_kwargs"] = {"enable_thinking": False}
        코드, 답 = post(주소, "/tokenize", 몸)
        기대 = 참조[c["id"]]
        if isinstance(기대, dict):
            ok = 코드 == 400
            print(f"{'같음' if ok else '다름'}  {c['id']}: 참조 오류({기대['error'][:60]}) · 서버 {코드} {답.get('error', {}).get('message', '')[:80]}")
            틀림 += not ok
            continue
        if 코드 != 200:
            print(f"다름  {c['id']}: 서버 {코드} {답}")
            틀림 += 1
            continue
        코드2, 답2 = post(주소, "/tokenize", {"prompt": 기대, "add_special_tokens": False})
        a, b = 답["tokens"], 답2["tokens"]
        if a == b:
            print(f"같음  {c['id']}: 토큰 {len(a)}개")
        else:
            k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
            print(f"다름  {c['id']}: 서버 {len(a)} · 참조 {len(b)} 토큰, 처음 다른 자리 {k} — 서버 {a[k:k + 8]} · 참조 {b[k:k + 8]}")
            틀림 += 1
    print("모두 같음" if 틀림 == 0 else f"다름 {틀림}")
    sys.exit(1 if 틀림 else 0)


if __name__ == "__main__":
    main()
