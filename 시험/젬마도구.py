#!/usr/bin/env python3
"""0.13.0 — Gemma 4 의 도구 호출 풀기(엔진/젬마도구.gl — 엔진/시험_젬마도구.gl)를 GPU 없이 본다. 사례마다 llama.cpp(common/chat-peg-parser.cpp
의 gemma4_to_json)와 같은 이름 · 인자 JSON 글(빈칸 없는 꼴 — 수는 쓴 그대로)이 나오는가. llama.cpp 의 문법이 받지 않는 꼴 가운데 GeulPU 가
너그럽게 받는 것(낱말 사이의 빈칸 · 열쇠 앞뒤의 빈칸 · 따옴으로 감싼 열쇠)은 그 뜻대로 — GeulPU 는 생성을 문법으로 묶지 않는다. 못 풀어야 하는
꼴은 null 이다(서버는 그 글을 내용으로 돌린다).

  python -I 시험/젬마도구.py <시험_젬마도구 실행 파일>

끝 줄 "젬마 도구 호출: 모두 맞음 (N 가지)"이면 종료 0, 아니면 1.
"""
import json
import os
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

Q = '<|"|>'          # 따옴 — 사용자 정의 토큰 <|"|> 의 글(서버가 토큰 번호로 받아 도구글에 넣는다)

# (아이디, 도구글, 이름 · 인자 JSON 글 — 못 풀면 None)
사례들 = [
    ("글 하나", f"call:get_weather{{city:{Q}Seoul{Q}}}", ("get_weather", '{"city":"Seoul"}')),
    ("빈 인자", "call:get_time{}", ("get_time", "{}")),
    ("수 · 참거짓 · null", "call:f{a:1,b:-2.5,c:3e10,d:true,e:false,f:null,g:0.0,h:-0}",
     ("f", '{"a":1,"b":-2.5,"c":3e10,"d":true,"e":false,"f":null,"g":0.0,"h":-0}')),
    ("겹친 객체 · 배열", f"call:search{{query:{Q}gemma{Q},filters:{{lang:{Q}ko{Q},limit:5}},tags:[{Q}a{Q},{Q}b{Q}],m:[[1,2],[]]}}",
     ("search", '{"query":"gemma","filters":{"lang":"ko","limit":5},"tags":["a","b"],"m":[[1,2],[]]}')),
    ("탈출할 글자", f'call:write{{text:{Q}He said "hi"\n\tback\\slash{Q}}}', ("write", '{"text":"He said \\"hi\\"\\n\\tback\\\\slash"}')),
    ("한국어 글", f"call:get_weather{{city:{Q}서울{Q},note:{Q}맑음 · 21 °C{Q}}}", ("get_weather", '{"city":"서울","note":"맑음 · 21 °C"}')),
    ("글 안의 : } , [", f"call:f{{q:{Q}a:b}},c [x]{Q}}}", ("f", '{"q":"a:b},c [x]"}')),
    ("빈 글 · 빈 배열 · 빈 객체", f"call:f{{a:{Q}{Q},b:[],c:{{}}}}", ("f", '{"a":"","b":[],"c":{}}')),
    ("열쇠의 밑줄 · 숫자 · 하이픈", f"call:f{{max_results:3,v2-key:{Q}x{Q}}}", ("f", '{"max_results":3,"v2-key":"x"}')),
    ("JSON 꼴 열쇠(따옴표까지 열쇠 — llama.cpp 와 같다)", 'call:f{"a":1}', ("f", '{"\\"a\\"":1}')),
    ("이름의 점 · 하이픈", f"call:tools.get-weather{{city:{Q}Seoul{Q}}}", ("tools.get-weather", '{"city":"Seoul"}')),
    # GeulPU 가 너그럽게 받는 꼴
    ("빈칸(앞뒤 · 낱말 사이)", f"  call: get_weather {{ city : {Q}Seoul{Q} ,\n unit:{Q}c{Q} }}\n", ("get_weather", '{"city":"Seoul","unit":"c"}')),
    ("따옴으로 감싼 열쇠", f"call:f{{{Q}city{Q}:{Q}Seoul{Q}}}", ("f", '{"city":"Seoul"}')),
    ("배열 안의 빈칸", f"call:f{{a:[ 1 , {Q}x{Q} , [ ] ]}}", ("f", '{"a":[1,"x",[]]}')),
    # 못 푸는 꼴 — 내용으로 돌린다
    ("call: 이 없음", f"get_weather{{city:{Q}Seoul{Q}}}", None),
    ("이름이 없음", "call:{a:1}", None),
    ("이름에 빈칸", "call:get weather{a:1}", None),
    ("인자가 없음", "call:get_weather", None),
    ("닫히지 않은 글", f"call:f{{a:{Q}xx}}", None),
    ("닫히지 않은 객체", f"call:f{{a:{Q}x{Q}", None),
    ("따옴 없는 낱말 값", "call:f{a:hello}", None),
    ("0 으로 시작하는 수", "call:f{a:01}", None),
    ("끝의 점", "call:f{a:1.}", None),
    ("값이 없음", "call:f{a:}", None),
    ("열쇠가 없음", "call:f{:1}", None),
    ("쉼표 뒤에 끝", "call:f{a:1,}", None),
    ("뒤에 남은 글", "call:f{a:1} 그리고", None),
    ("호출 둘(토큰 하나 안에)", "call:f{a:1}call:g{b:2}", None),
    ("깊이 65", "call:f{a:" + "[" * 65 + "]" * 65 + "}", None),
]


def main():
    exe = sys.argv[1]
    with tempfile.TemporaryDirectory() as 임시:
        입력 = os.path.join(임시, "사례.json")
        출력 = os.path.join(임시, "출력.json")
        with open(입력, "w", encoding="utf-8", newline="\n") as f:
            json.dump([{"id": a, "text": t} for a, t, _ in 사례들], f, ensure_ascii=False)
        r = subprocess.run([exe, 입력, 출력], capture_output=True)
        if r.returncode != 0:
            print(r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace"))
            print(f"젬마 도구 호출: 시험_젬마도구 종료 {r.returncode}")
            return 1
        with open(출력, encoding="utf-8") as f:
            나온 = json.load(f)
    틀림 = 0
    for 아이디, 글, 답 in 사례들:
        n = 나온.get(아이디, "없음")
        if 답 is None:
            맞음 = n is None
        else:
            맞음 = isinstance(n, dict) and n.get("name") == 답[0] and n.get("arguments") == 답[1]
            if 맞음:
                json.loads(n["arguments"])             # 인자는 맞는 JSON
        if not 맞음:
            틀림 += 1
            print(f"틀림  {아이디}: {글!r}\n      나온 {n!r}\n      답   {답!r}")
    if 틀림:
        print(f"젬마 도구 호출: 틀림 {틀림} / {len(사례들)}")
        return 1
    print(f"젬마 도구 호출: 모두 맞음 ({len(사례들)} 가지)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
