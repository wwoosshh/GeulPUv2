#!/usr/bin/env python3
"""0.12.0 12-G1 — jinja 템플릿 해석기(엔진/진해석.gl — 엔진/시험_틀해석.gl)를 jinja2 와 견준다(GPU 없이). jinja2 의 환경은 transformers 가
채팅 템플릿에 쓰는 것과 같다 — ImmutableSandboxedEnvironment(trim_blocks · lstrip_blocks · loopcontrols · generation 태그), tojson =
json.dumps(ensure_ascii=False, …), 전역 raise_exception. strftime_now 는 두지 않는다(GeulPU 는 같은 요청이 같은 토큰이 되게 날짜를
고정한다 — 맥락의 date_string).

  python 시험/틀해석.py <시험_틀해석 실행 파일> [--틀 파일 …] [--hf 개수] [--보기]

  · 인자가 없으면 기능 사례(이 파일 안의 작은 템플릿 — 빠른 점검 · CI)만.
  · --틀: GGUF(그 안의 tokenizer.chat_template) · .jinja 파일의 템플릿을 표준 대화 묶음(시스템 · 여러 차례 · 생각 · 도구 정의 · 호출 ·
    결과, 생성 머리 켬 · 끔, enable_thinking 참 · 거짓 · 없음)으로.
  · --hf N: Hugging Face 에서 내려받기가 많은 GGUF 저장소 N 개의 템플릿(API 의 메타데이터만 — 모델 파일은 받지 않는다)을 같은 묶음으로.
  같음 = 두 글이 바이트까지 같거나, 둘 다 오류(오류 글은 견주지 않는다). 끝 줄 "틀해석: 모두 같음"이면 종료 0, jinja2 가 없으면 3.
  WSL 에서 윈도 실행 파일을 부를 때는 --임시 <윈도에서도 보이는 폴더 — /mnt/… > 를 준다.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")


# ---- jinja2 — transformers 의 _compile_jinja_template 과 같은 환경 ----
def 환경():
    import jinja2
    from jinja2 import nodes
    from jinja2.ext import Extension, loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    class 생성태그(Extension):
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return nodes.CallBlock(self.call_method("_생성", []), [], [], body).set_lineno(lineno)

        def _생성(self, caller):
            return caller()

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[생성태그, loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def 참조그리기(env, 틀, 맥락들):
    """[(id, 맥락)] → {id: 글 | {"error": 까닭}}"""
    out = {}
    try:
        t = env.from_string(틀)
    except Exception as e:                       # noqa: BLE001 — 템플릿 문법 오류도 '오류'로 견준다
        return {i: {"error": f"{type(e).__name__}: {e}"} for i, _ in 맥락들}
    for i, c in 맥락들:
        try:
            out[i] = t.render(**c)
        except Exception as e:                   # noqa: BLE001
            out[i] = {"error": f"{type(e).__name__}: {e}"}
    return out


def 실행경로(p):
    """WSL 에서 윈도 실행 파일(.exe)을 부를 때는 경로를 윈도 꼴로(wslpath -m)"""
    if os.name == "posix" and 임시폴더 and 실행파일.lower().endswith(".exe"):
        return subprocess.run(["wslpath", "-m", p], capture_output=True, text=True).stdout.strip()
    return p


임시폴더 = None
실행파일 = ""


def 글그리기(exe, 틀, 맥락들):
    with tempfile.TemporaryDirectory(dir=임시폴더) as d:
        tp, cp, op = (os.path.join(d, n) for n in ("t.jinja", "c.json", "o.json"))
        with open(tp, "wb") as f:
            f.write(틀.encode("utf-8"))
        with open(cp, "wb") as f:
            f.write(json.dumps([{"id": i, "ctx": c} for i, c in 맥락들], ensure_ascii=False).encode("utf-8"))
        r = subprocess.run([exe, 실행경로(tp), 실행경로(cp), 실행경로(op)], capture_output=True)
        if r.returncode != 0:
            return None, (r.stdout + r.stderr).decode("utf-8", "replace")
        with open(op, "rb") as f:
            return json.loads(f.read().decode("utf-8")), r.stdout.decode("utf-8", "replace").strip()


def 견주기(이름, 참조, 글, 보기):
    """같은 사례 수 · 다른 것들"""
    같음, 다름 = 0, []
    for i, 기대 in 참조.items():
        얻음 = 글.get(i)
        if isinstance(기대, dict) and isinstance(얻음, dict):
            같음 += 1
        elif 이름.startswith("한도-") and isinstance(얻음, dict):
            같음 += 1                           # 한도 사례 — 죽지 않고 오류로 끝나면 된다(jinja2 는 파이썬의 재귀 한도까지 간다)
        elif 기대 == 얻음:
            같음 += 1
        else:
            다름.append(i)
            if 보기 or len(다름) <= 3:
                if isinstance(기대, str) and isinstance(얻음, str):
                    k = next((j for j in range(min(len(기대), len(얻음))) if 기대[j] != 얻음[j]), min(len(기대), len(얻음)))
                    print(f"  다름 {이름}/{i} — 글자 {k}: 기대 {기대[max(0, k - 40):k + 60]!r}\n{' ' * 14}얻음 {얻음[max(0, k - 40):k + 60]!r}")
                else:
                    print(f"  다름 {이름}/{i} — 기대 {str(기대)[:200]!r}\n{' ' * 14}얻음 {str(얻음)[:200]!r}")
    return 같음, 다름


# ---- 기능 사례 — (이름, 템플릿, [맥락]) ----
사람 = {"name": "철수", "age": 31, "tags": ["a", "b", "c"], "score": 2.5, "ok": True, "none": None,
       "d": {"k1": "v1", "k2": [1, 2], "k3": {"x": 1}}, "s": "  Hello, World  ", "kor": "가나다라마", "n": 7}
기능 = [
    ("공백-기본", "A\n{% if true %}\n  B\n{% endif %}\nC\n", [사람]),
    ("공백-빼기", "A  {%- if true -%}  B  {%- endif -%}  C", [사람]),
    ("공백-더하기", "A\n    {%+ if true %}B{% endif +%}\nC", [사람]),
    ("공백-줄앞", "x\n    {% for t in tags %}\n    [{{ t }}]\n    {% endfor %}\ny", [사람]),
    ("공백-같은줄", "a {% if true %} b {% endif %} c", [사람]),
    ("공백-변수빼기", "a  {{- name -}}  b\n{{ age }}\n", [사람]),
    ("주석", "a{# 주석 #}b\n  {#- 빼기 -#}  c\n{# 줄 #}\nd", [사람]),
    ("날것", "{% raw %}{{ 그대로 }} {% if %}{% endraw %}!", [사람]),
    ("끝줄바꿈", "끝\n", [사람]),
    ("crlf", "a\r\n{% if true %}\r\nb\r\n{% endif %}\r\n", [사람]),
    ("만약", "{% if age > 40 %}old{% elif age > 30 %}mid{% else %}young{% endif %}|{% if none %}x{% elif not none %}y{% endif %}", [사람]),
    ("반복", "{% for t in tags %}{{ loop.index }}{{ loop.index0 }}{{ loop.revindex }}{{ loop.revindex0 }}{{ loop.first }}{{ loop.last }}"
     "{{ loop.length }}{{ loop.previtem }}{{ loop.nextitem }};{% endfor %}", [사람]),
    ("반복-else", "{% for t in [] %}x{% else %}비었음{% endfor %}|{% for t in tags if t != 'b' %}{{ t }}{{ loop.length }}{% endfor %}", [사람]),
    ("반복-멈춤", "{% for t in tags %}{% if t == 'b' %}{% continue %}{% endif %}{% if t == 'c' %}{% break %}{% endif %}{{ t }}{% endfor %}", [사람]),
    ("반복-사전", "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}{% for k in d %}{{ k }}{% endfor %}{% for c in kor %}[{{ c }}]{% endfor %}", [사람]),
    ("범위", "{% set x = 1 %}{% for t in tags %}{% set x = x + 1 %}{{ x }}{% endfor %}|{{ x }}", [사람]),
    ("이름공간", "{% set ns = namespace(c=0, f=false) %}{% for t in tags %}{% set ns.c = ns.c + 1 %}{% if t == 'b' %}{% set ns.f = true %}{% endif %}"
     "{% endfor %}{{ ns.c }} {{ ns.f }}", [사람]),
    ("매크로", "{% macro m(a, b=2, c='z') %}[{{ a }}-{{ b }}-{{ c }}]{% endmacro %}{{ m(1) }}{{ m(1, 3) }}{{ m(1, c='q') }}{{ m(a=5) }}|{{ m(1)|length }}", [사람]),
    ("매크로-전역", "{% set g = 'G' %}{% macro m() %}{{ g }}{{ name }}{% set g = 'L' %}{{ g }}{% endmacro %}{{ m() }}{{ g }}", [사람]),
    ("덩이두기", "{% set x %}  a{{ name }}  {% endset %}[{{ x }}]{% set y | upper %}abc{% endset %}{{ y }}", [사람]),
    ("거르개덩이", "{% filter upper %}hello {{ name }}{% endfilter %}", [사람]),
    ("글메서드", "{{ s.strip() }}|{{ s.lstrip() }}|{{ s.rstrip() }}|{{ 'xxhixx'.strip('x') }}|{{ s.split() }}|{{ 'a,b,,c'.split(',') }}|"
     "{{ 'a b c d'.split(None, 1) }}|{{ 'a,b,c'.rsplit(',', 1) }}|{{ s.strip().startswith('Hello') }}|{{ 'abc'.endswith(('x', 'c')) }}|"
     "{{ 'aaa'.replace('a', 'b', 2) }}|{{ 'abc'.upper() }}{{ 'ABC'.lower() }}{{ 'hello world'.title() }}{{ 'hELLO'.capitalize() }}|"
     "{{ 'abcabc'.find('c') }}{{ 'abcabc'.rfind('c') }}{{ 'abcabc'.count('bc') }}|{{ '-'.join(tags) }}|{{ kor.find('다') }}", [사람]),
    ("자르기", "{{ tags[1:] }}{{ tags[:-1] }}{{ tags[::-1] }}{{ tags[::2] }}{{ tags[-1] }}{{ tags[5] }}|{{ kor[1:3] }}{{ kor[::-1] }}{{ kor[-1] }}{{ 'abc'[10:] }}", [사람]),
    ("사전메서드", "{{ d.keys()|list }}{{ d.values()|list }}{{ d.get('k1') }}{{ d.get('zz', 'dflt') }}{{ d.get('zz') }}{{ d['k2'][1] }}{{ d.k3.x }}", [사람]),
    ("거르개", "{{ tags|length }}{{ kor|length }}{{ d|length }}|{{ s|trim }}|{{ tags|first }}{{ tags|last }}|{{ tags|join(', ') }}|{{ nope|default('기본') }}"
     "{{ ''|default('빈', true) }}|{{ name|upper }}{{ 'X'|lower }}|{{ 'a-b'|replace('-', '+') }}|{{ tags|reverse|list }}|{{ [3, 1, 2]|min }}{{ [3, 1, 2]|max }}"
     "{{ [1, 2, 3]|sum }}|{{ '42'|int + 1 }}{{ '3.5'|float }}{{ -3|abs }}|{{ 'a\nb\n\nc'|indent(2) }}|{{ 'a\nb'|indent(2, true) }}|{{ '<&>\"'|e }}", [사람]),
    ("거르개-고르기", "{{ [1, 2, 3, 4]|select('odd')|list }}{{ [1, 2, 3, 4]|reject('even')|list }}{{ ms|selectattr('role', 'equalto', 'user')|list|length }}"
     "{{ ms|rejectattr('role', 'equalto', 'user')|map(attribute='content')|join(',') }}{{ ms|map(attribute='role')|list }}"
     "{{ ['a', 'b']|map('upper')|list }}{{ ms|selectattr('extra')|list|length }}",
     [dict(사람, ms=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "extra": 1}, {"role": "user", "content": "q2"}])]),
    ("tojson", "{{ d|tojson }}|{{ tags|tojson(indent=2) }}|{{ {'한': '글', 'q': '\"'}|tojson }}|{{ {'한': 1}|tojson(ensure_ascii=true) }}|{{ score|tojson }}"
     "|{{ none|tojson }}{{ ok|tojson }}|{{ d|tojson(separators=(',', ':')) }}|{{ 1.0|tojson }}{{ 1e20|tojson }}{{ 0.1|tojson }}", [사람]),
    ("시험", "{{ x is defined }}{{ name is defined }}{{ none is none }}{{ name is string }}{{ age is number }}{{ age is integer }}{{ score is float }}"
     "{{ ok is boolean }}{{ ok is true }}{{ ok is false }}{{ d is mapping }}{{ tags is iterable }}{{ name is iterable }}{{ tags is sequence }}"
     "{{ n is even }}{{ n is odd }}{{ 9 is divisibleby 3 }}{{ age is eq 31 }}{{ age is ne 3 }}{{ 1 is lt 2 }}{{ 3 is gt 2 }}{{ 'a' is in tags }}"
     "{{ 'abc' is lower }}{{ 'ABC' is upper }}{{ none is sameas none }}{{ name is not string }}{{ x is not defined }}{{ ok is number }}{{ ok is integer }}", [사람]),
    ("셈", "{{ 1 + 2 }}{{ 7 - 10 }}{{ 3 * 4 }}{{ 7 / 2 }}{{ 7 // 2 }}{{ -7 // 2 }}{{ 7 % 3 }}{{ -7 % 3 }}{{ 2 ** 10 }}{{ 1.5 + 1 }}{{ 'a' + 'b' }}"
     "{{ 'ab' * 3 }}{{ [1] + [2] }}{{ 1 ~ 'x' ~ none ~ ok }}{{ age + true }}|{{ 3 > 2 > 1 }}{{ 1 < 2 > 3 }}{{ 'a' in 'cat' }}{{ 'z' not in tags }}"
     "{{ 1 == 1.0 }}{{ true == 1 }}{{ [1, 2] == [1, 2] }}{{ 'b' > 'a' }}|{{ 0 or 'x' }}{{ 1 and 'y' }}{{ '' and 3 }}{{ none or none }}"
     "{{ not 0 }}|{{ 'y' if ok else 'n' }}{{ 'y' if none }}{{ -2 ** 2 }}{{ (1, 2) }}{{ (1,) }}{{ d.items()|list }}", [사람]),
    ("오류-잇기우선", "{{ 1 + 2 ~ 3 }}", [사람]),
    ("찍기", "{{ none }}|{{ ok }}|{{ score }}|{{ 1e100 }}|{{ 0.1 + 0.2 }}|{{ tags }}|{{ d }}|{{ undefined_x }}|{{ [none, true, 'it\\'s', \"q\\\"\"] }}|{{ -0.0 }}"
     "|{{ 10 / 4 }}|{{ 1e-5 }}|{{ 123456789.0 * 1000000000 }}", [사람]),
    ("글상수", "{{ 'a\\nb' }}|{{ \"tab\\there\" }}|{{ '\\u00e9\\x41' }}|{{ 'one' 'two' }}|{{ '{{ not }}' }}|{{ \"%}\" }}", [사람]),
    ("목록사전식", "{{ [1, 'a', [2]] }}{{ {'a': 1, 'b': [2]} }}{{ {'a': 1}['a'] }}{{ [1, 2, 3][1] }}{{ ([1, 2])[0] }}", [사람]),
    ("오류-속성", "{{ nope.attr }}", [사람]),
    ("오류-던짐", "{{ raise_exception('멈춤: ' ~ name) }}", [사람]),
    ("오류-타입", "{{ 'a' + 1 }}", [사람]),
    ("오류-문법", "{% if %}", [사람]),
    ("오류-닫힘", "{% for x in tags %}", [사람]),
    ("범위함수", "{{ range(3)|list }}{{ range(1, 7, 2)|list }}{% for i in range(2) %}{{ i }}{% endfor %}", [사람]),
    ("생성태그", "a{% generation %}b{{ name }}{% endgeneration %}c", [사람]),
    ("글바꿈", "{{ kor|length }}{{ kor[2] }}{{ kor.replace('다', 'X') }}{{ '가 나 다'.split(' ') }}{{ ' 가 '|trim }}", [사람]),
    ("dictsort", "{% for k, v in {'b': 1, 'A': 2, 'a': 3, 'C': [1]}|dictsort %}{{ k }}={{ v }};{% endfor %}|{{ {'b': 1, 'a': 2}|dictsort(reverse=true) }}"
     "|{{ {'b': 1, 'B': 0, 'a': 2}|dictsort(true) }}|{{ {'x': 2, 'y': 1}|dictsort(by='value') }}", [사람]),
    ("format", "{{ '<{}|{}>'.format(name, age) }}{{ '{1}{0}'.format('a', 'b') }}{{ '{x}-{y}'.format(x=1, y=none) }}{{ '{{}}{}'.format(ok) }}"
     "{{ '{!r}'.format(name) }}", [사람]),
    ("오류-format", "{{ '{5}'.format(1) }}", [사람]),
    ("한도-깊은괄호", "{{ " + "(" * 300 + "1" + ")" * 300 + " }}", [사람]),
    ("한도-재귀매크로", "{% macro m(n) %}{% if n > 0 %}{{ m(n - 1) }}{% endif %}{{ n }}{% endmacro %}{{ m(30) }}|{{ m(200) }}", [사람]),
    ("깊은괄호", "{{ " + "(" * 30 + "1" + ")" * 30 + " }}", [사람]),
    ("한도-깊은목록", "{% set ns = namespace(x=1) %}{% for i in range(300) %}{% set ns.x = [ns.x] %}{% endfor %}{{ ns.x|length }}", [사람]),
    ("format거르개", "{{ '%s(%r) %d%%'|format(name, name, 3) }}{{ '%(a)s-%(b)s'|format(a=1, b='x') }}{{ '%i'|format(2.9) }}", [사람]),
    ("sort", "{{ [3, 1, 2]|sort }}{{ ['b', 'A', 'a', 'B']|sort }}{{ ['b', 'A', 'a', 'B']|sort(case_sensitive=true) }}{{ [3, 1, 2]|sort(reverse=true) }}"
     "{{ ms|sort(attribute='n')|map(attribute='n')|list }}{{ ['a', 'A', 'b', 'a']|unique|list }}{{ [1, 2, 1, 3]|unique|list }}",
     [dict(사람, ms=[{"n": 2}, {"n": 1}, {"n": 2, "x": 1}, {"n": 0}])]),
]


# ---- 표준 대화 묶음(실제 템플릿용) ----
도구 = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather in a city.",
                                         "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                                                        "required": ["city"]}}},
        {"type": "function", "function": {"name": "search", "description": "검색", "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}}}]
호출 = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                                                           "function": {"name": "get_weather", "arguments": {"city": "Seoul"}}}]}
결과 = {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "{\"temp_c\": 21}"}
대화들 = {
    "한": [{"role": "user", "content": "안녕? 자기소개 해 줘."}],
    "시스템": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}],
    "여러차례": [{"role": "system", "content": "짧게 답해."}, {"role": "user", "content": "수도는?"}, {"role": "assistant", "content": "서울."},
              {"role": "user", "content": "인구는?"}],
    "생각": [{"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "<think>\nadd\n</think>\n\n2"}, {"role": "user", "content": "2+2?"}],
    "공백": [{"role": "user", "content": "  앞뒤 공백  \n"}],
    "도구정의": [{"role": "user", "content": "What's the weather in Seoul?"}],
    "도구호출": [{"role": "user", "content": "What's the weather in Seoul?"}, 호출, 결과],
    "도구둘": [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Seoul and Busan?"},
             {"role": "assistant", "content": "Checking.", "tool_calls": [
                 {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Seoul"}}},
                 {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Busan"}}}]},
             {"role": "tool", "tool_call_id": "c1", "content": "21"}, {"role": "tool", "tool_call_id": "c2", "content": "19"}],
}


def 표준맥락들(bos, eos):
    out = []
    for 이름, ms in 대화들.items():
        도구씀 = 이름.startswith("도구")
        for 머리 in (True, False):
            for 생각 in (None, False, True):
                c = {"messages": ms, "add_generation_prompt": 머리, "bos_token": bos, "eos_token": eos, "date_string": "26 Jul 2024"}
                if 도구씀:
                    c["tools"] = 도구
                if 생각 is not None:
                    c["enable_thinking"] = 생각
                out.append((f"{이름}-{'머리' if 머리 else '끝'}-{'없음' if 생각 is None else ('켬' if 생각 else '끔')}", c))
    return out


# ---- GGUF 의 메타데이터 ----
def gguf_meta(path, keys):
    with open(path, "rb") as f:
        b = f.read(64 << 20)
    assert b[:4] == b"GGUF"
    _, _, nkv = struct.unpack_from("<IQQ", b, 4)
    o = 24
    out = {}

    def rs(o):
        n = struct.unpack_from("<Q", b, o)[0]
        return b[o + 8:o + 8 + n].decode("utf-8", "replace"), o + 8 + n
    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip(t, o, want=False):
        if t == 8:
            s, o = rs(o)
            return s, o
        if t == 9:
            et, n = struct.unpack_from("<IQ", b, o)
            o += 12
            vals = []
            for _ in range(n):
                v, o = skip(et, o, want)
                if want:
                    vals.append(v)
            return vals, o
        v = struct.unpack_from({0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t], b, o)[0]
        return v, o + SZ[t]
    for _ in range(nkv):
        k, o = rs(o)
        t = struct.unpack_from("<I", b, o)[0]
        o += 4
        v, o = skip(t, o, k in keys)
        if k in keys:
            out[k] = v
    return out


def 파일템플릿(path):
    """(템플릿, bos, eos) — GGUF 면 그 안의 것, 아니면 파일 글 그대로(bos · eos 는 흔한 것)"""
    if path.lower().endswith(".gguf"):
        m = gguf_meta(path, {"tokenizer.chat_template", "tokenizer.ggml.tokens", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"})
        toks = m.get("tokenizer.ggml.tokens") or []
        bos = toks[m["tokenizer.ggml.bos_token_id"]] if "tokenizer.ggml.bos_token_id" in m and m["tokenizer.ggml.bos_token_id"] < len(toks) else ""
        eos = toks[m["tokenizer.ggml.eos_token_id"]] if "tokenizer.ggml.eos_token_id" in m and m["tokenizer.ggml.eos_token_id"] < len(toks) else ""
        return m.get("tokenizer.chat_template"), bos, eos
    return open(path, encoding="utf-8").read(), "<s>", "</s>"


def hf템플릿들(개수):
    import urllib.request
    u = f"https://huggingface.co/api/models?filter=gguf&sort=downloads&limit={개수}&expand[]=gguf"
    r = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "geulpu-test"}), timeout=120).read())
    본 = {}
    for m in r:
        g = m.get("gguf") or {}
        t = g.get("chat_template")
        if t and t not in 본:
            본[t] = (m["id"], g.get("bos_token") or "", g.get("eos_token") or "")
    return [(이름, t, bos, eos) for t, (이름, bos, eos) in 본.items()]


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(2)
    global 임시폴더, 실행파일
    exe = args[0]
    실행파일 = exe
    보기 = "--보기" in args
    틀파일들 = []
    hf = 0
    i = 1
    while i < len(args):
        if args[i] == "--임시":
            임시폴더 = args[i + 1]
            i += 2
            continue
        if args[i] == "--틀":
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                틀파일들.append(args[i])
                i += 1
            continue
        if args[i] == "--hf":
            hf = int(args[i + 1])
            i += 2
            continue
        i += 1
    try:
        import jinja2  # noqa: F401
    except ImportError:
        print("틀해석: 건너뜀(jinja2 없음 — pip install -r requirements.txt)")
        sys.exit(3)
    env = 환경()
    묶음 = []
    if not 틀파일들 and not hf:
        for 이름, 틀, cs in 기능:
            묶음.append((이름, 틀, [("0", c) for c in cs]))
    for p in 틀파일들:
        틀, bos, eos = 파일템플릿(p)
        if 틀:
            묶음.append((os.path.basename(p), 틀, 표준맥락들(bos, eos)))
        else:
            print(f"  템플릿 없음: {p}")
    if hf:
        for 이름, 틀, bos, eos in hf템플릿들(hf):
            묶음.append((이름, 틀, 표준맥락들(bos, eos)))
    합같음, 합수, 틀린틀 = 0, 0, []
    for 이름, 틀, cs in 묶음:
        참조 = 참조그리기(env, 틀, cs)
        글, 알림 = 글그리기(exe, 틀, cs)
        if 글 is None:
            print(f"  실행 실패 {이름}: {알림[-300:]}")
            틀린틀.append(이름)
            합수 += len(cs)
            continue
        같음, 다름 = 견주기(이름, 참조, 글, 보기)
        합같음 += 같음
        합수 += len(cs)
        모두오류 = all(isinstance(v, dict) for v in 참조.values())
        if 다름:
            틀린틀.append(이름)
        if 보기 or 다름 or len(묶음) <= 60:
            print(f"{'같음' if not 다름 else '다름'} {이름}: {같음}/{len(cs)}{' (jinja2 도 모두 오류)' if 모두오류 else ''}")
    print(f"템플릿 {len(묶음)} · 사례 {합수} · 같음 {합같음} · 다른 템플릿 {len(틀린틀)}")
    print("틀해석: 모두 같음" if not 틀린틀 else f"틀해석: 다름 {len(틀린틀)}")
    sys.exit(0 if not 틀린틀 else 1)


if __name__ == "__main__":
    main()
