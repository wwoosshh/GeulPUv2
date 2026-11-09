"""기능 목록(앱/data/operations.json — geulpu-operations/1)의 점검 — 앱리디자인 단계 B1(문서/기능목록.md 3 · 4 절). GPU 없이, 표준 라이브러리만.

  python -I -X utf8 시험/기능목록점검.py              점검(못 미침이 있으면 종료 1)
  python -I -X utf8 시험/기능목록점검.py --자체시험    일부러 넣은 잘못 열둘을 모두 잡는지
  python -I -X utf8 시험/기능목록점검.py --뽑기        코드에서 뽑은 경로 · 메서드 · 키 · 서버 필드를 JSON 으로(목록을 고칠 때)

점검(문서/기능목록.md 3 절):
  1 형식 — 필드 · 형 · id 의 형식, status 와 stage, cli · title · description 의 두 언어.
  2 경로 대조 — 앱의 글(geul) 코드가 처리하는 /api/… 경로(메서드와 함께)와 목록의 경로가 같다(코드에만 · 목록에만 있는 것 0).
    전달 경로(/v1/* · /llama/*)와 서버 API 의 /v1 경로도 목록에 있다.
  3 CLI — 내부용이 아닌 모든 기능에 CLI 이름(ko · en)이 있고 같은 언어 안에서 겹치지 않는다.
  4 화면(앱리디자인 E6 — 문서/화면구현.md §6) — 화면 표(screens: id 마다 kind view · part 와 ko · en)의 view 가 웹 코드의 화면(app.js 의
    VIEW_GROUP)과 같다. 내부용이 아닌 있는 기능마다 gui.screens(표에 있는 화면 id)가 있고 gui.via 는 있는 기능이다. 앞으로의 기능은 gui.planned.
    결과 형식마다 screens(앞으로의 기능만의 결과면 planned)가 있고, 어느 기능의 결과다.
  5 웹 코드(앱리디자인 E6) — (가) 화면 코드(ui/ops.js · ui/command.js 밖)에 앱 · 서버 경로의 글(/api/ · /v1/ · /llama/ · /tokenize ·
    /detokenize · /tokenizer_info)이 없다 — 앱은 기능 목록 클라이언트로만 부른다. (나) 기능 id 형식의 글(점 앞이 목록의 기능 id 앞부분)은 목록에 있는
    있는 기능이다. (다) 내부용이 아닌 있는 기능의 id 가 모두 화면 코드에 있다(화면에서 닿는다) — 아니면 그 기능의 gui.via 가 모두.
  6 인자 — 경로를 처리하는 코드 블록이 직접 읽는 요청 키(GET 은 질의, POST 는 몸의 JSON)가 모두 그 기능의 params 에 있다.
    서버가 읽는 GeulPU 필드(geulpu.* · 그 밖의 확장 필드)가 모두 어느 기능의 params 에 있다.
"""
import copy
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(ROOT, "앱", "data", "operations.json")
# 경로를 처리하는 함수(파일, 함수 이름, 메서드). 관리나눠주기 안의 /api 경로는 '가져옴이면'(GET · HEAD) 블록에 있다.
처리함수 = [("앱/관리자.gl", "관리나눠주기", "GET"), ("앱/관리자.gl", "관리보냄", "POST"),
           ("앱/평가.gl", "평가가져오기", "GET"), ("앱/평가.gl", "평가보냄", "POST"),
           ("앱/라마재현.gl", "라마가져오기", "GET"), ("앱/라마재현.gl", "라마보냄", "POST")]
웹파일 = ["앱/web/app.js", "앱/web/eval.js"] + sorted("앱/web/ui/" + f for f in os.listdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "앱", "web", "ui"))
                                                if f.endswith(".js"))    # 앱리디자인 E1 — 화면 모듈
서버파일 = "엔진/서버몸.gl"
# 앱이 그대로 넘기는 경로(관리나눠주기): 추론 서버로 · 사용자의 llama-server(기록기)로
전달경로 = ["/v1/*", "/tokenize", "/detokenize", "/tokenizer_info", "/llama/*"]
# 서버가 요청의 맨 위에서 읽는 GeulPU 확장 필드(geulpu 객체 밖)
맨위필드 = ["echo", "ignore_eos", "cache_prompt", "geulpu_tool_errors", "add_special_tokens", "chat_template_kwargs.enable_thinking",
           "chat_template_kwargs.date_string"]
ID = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
형들 = {"string", "integer", "number", "boolean", "object", "array", "file"}


# ---------- 글(geul) 코드 읽기 ----------
def 짝괄호(t, i):
    """t[i] == '{' 의 짝 '}' 의 자리 — 글자열("…", 역빗금 이스케이프)과 주석((* … *))은 건너뛴다."""
    assert t[i] == "{"
    깊이 = 0; n = len(t); k = i
    while k < n:
        ch = t[k]
        if ch == '"':
            k += 1
            while k < n and t[k] != '"':
                k += 2 if t[k] == "\\" else 1
        elif ch == "(" and t.startswith("(*", k):
            e = t.find("*)", k + 2)
            k = n if e < 0 else e + 1
        elif ch == "{":
            깊이 += 1
        elif ch == "}":
            깊이 -= 1
            if 깊이 == 0:
                return k
        k += 1
    raise ValueError("짝 없는 { 의 자리 %d" % i)


def 함수몸(t, 이름):
    m = re.search(r"\[[^\[\]]*?" + re.escape(이름) + r"\]는[^{]*\{", t)
    if not m:
        raise ValueError("함수 '%s' 를 찾지 못함" % 이름)
    s = m.end() - 1
    return s, 짝괄호(t, s)


ROUTE = re.compile(r'문자열_같은가\(\s*길\s*,\s*"(/api/[^"]+)"\s*\)')
GETKEY = re.compile(r'물음값\(\s*[^,()]+\s*,\s*"([A-Za-z_][A-Za-z0-9_]*)"')
POSTKEY = re.compile(r'제[가-힣]+\(\s*o\s*,\s*"([A-Za-z_][A-Za-z0-9_]*)"')


def 경로블록들(t, 시작, 끝):
    """[시작, 끝) 안의 경로 조건마다 (경로, 블록 시작, 블록 끝, 조건 자리)."""
    out = []
    for m in ROUTE.finditer(t, 시작, 끝):
        e = t.find("이면", m.end(), 끝)
        b = t.find("{", e, 끝) if e >= 0 else -1
        if e < 0 or b < 0 or t[e + 2:b].strip():
            continue
        out.append((m.group(1), b, 짝괄호(t, b), m.start()))
    return out


def 코드뽑기(texts):
    """{(메서드, 경로): {"keys": set, "job": bool, "files": set}} — texts 는 {상대 경로: 글}."""
    일함수 = 일만드는함수들(texts)
    결과 = {}
    for 파일, 이름, 메서드 in 처리함수:
        t = texts[파일]
        s, e = 함수몸(t, 이름)
        블록들 = 경로블록들(t, s, e)
        for 경로, b, be, _ in 블록들:
            안쪽다른 = [(bb, bbe) for (p2, bb, bbe, _) in 블록들 if b < bb < be and p2 != 경로]
            조각 = []; k = b
            for bb, bbe in sorted(안쪽다른):
                if bb >= k:
                    조각.append(t[k:bb]); k = bbe + 1
            조각.append(t[k:be])
            몸 = "".join(조각)
            키 = set((GETKEY if 메서드 == "GET" else POSTKEY).findall(몸))
            일 = "일만들기(" in 몸 or any(f + "(" in 몸 for f in 일함수)
            x = 결과.setdefault((메서드, 경로), {"keys": set(), "job": False, "files": set()})
            x["keys"] |= 키; x["job"] = x["job"] or 일; x["files"].add(파일)
    return 결과


def 일만드는함수들(texts):
    이름들 = set()
    for 파일, t in texts.items():
        if not 파일.startswith("앱/"):
            continue
        for m in re.finditer(r"\[[^\[\]]*?([가-힣A-Za-z0-9_]+)\]는[^{]*\{", t):
            try:
                e = 짝괄호(t, m.end() - 1)
            except ValueError:
                continue
            if "일만들기(" in t[m.end():e] and m.group(1) != "일만들기":
                이름들.add(m.group(1))
    return 이름들


# 앱리디자인 E6 — 화면 코드: 기능 목록 클라이언트(ui/ops.js)와 명령 해석(ui/command.js)만 경로를 안다
클라이언트파일 = {"앱/web/ui/ops.js", "앱/web/ui/command.js"}
화면파일 = [f for f in 웹파일 if f not in 클라이언트파일]
정규식앞 = set("(,=:[!&|?{};+-*%<>~^")
정규식앞말 = {"return", "typeof", "case", "in", "of", "delete", "void", "throw", "new", "else", "do"}


def JS글들(t):
    """JS 코드의 글 그대로(작은 · 큰따옴표 글과 템플릿 — 템플릿은 ${…} 까지 그대로)와 그 줄. 주석 · 정규식은 건너뛴다(작은 훑기)."""
    out, i, n, 앞 = [], 0, len(t), ""
    while i < n:
        c = t[i]
        if c == "/" and t.startswith("//", i):
            j = t.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "/" and t.startswith("/*", i):
            j = t.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == "/" and (앞 in 정규식앞 or 앞 == ""):
            j, 대괄호안 = i + 1, False
            while j < n and t[j] != "\n":
                if t[j] == "\\":
                    j += 2
                    continue
                if t[j] == "[":
                    대괄호안 = True
                elif t[j] == "]":
                    대괄호안 = False
                elif t[j] == "/" and not 대괄호안:
                    break
                j += 1
            i, 앞 = j + 1, "x"
            continue
        if c in "'\"`":
            j, 깊이 = i + 1, 0
            while j < n:
                if t[j] == "\\":
                    j += 2
                    continue
                if c == "`" and t.startswith("${", j):
                    깊이 += 1
                    j += 2
                    continue
                if c == "`" and t[j] == "}" and 깊이:
                    깊이 -= 1
                elif t[j] == c and 깊이 == 0:
                    break
                j += 1
            out.append((t[i + 1:j], t.count("\n", 0, i) + 1))
            i, 앞 = j + 1, "x"
            continue
        if c.isalpha() or c in "_$":
            m = re.match(r"[A-Za-z_$][\w$]*", t[i:])
            if m:
                앞 = "(" if m.group(0) in 정규식앞말 else "x"
                i += len(m.group(0))
                continue
        if not c.isspace():
            앞 = c
        i += 1
    return out


def 경로글(v):
    return v.startswith(("/api/", "/v1/", "/llama/")) or re.match(r"^/(tokenize|detokenize|tokenizer_info)(\?|$)", v) is not None


def 웹경로들(texts):
    """화면 코드에 남은 경로의 글(--뽑기 · 점검 5 가)."""
    return {v.split("?")[0] for f in 화면파일 for v, _ in JS글들(texts[f]) if 경로글(v)}


def 웹화면들(texts):
    """app.js 의 VIEW_GROUP 키 — 웹 코드의 화면 이름."""
    m = re.search(r"const VIEW_GROUP = \{(.*?)\};", texts["앱/web/app.js"], re.S)
    return set(re.findall(r"([a-z]+):", m.group(1))) if m else set()


def 서버뽑기(t):
    """서버 API 의 /v1 경로와 GeulPU 필드(geulpu 객체의 키 · layer_lens 안의 키 · 맨 위 확장 필드)."""
    경로 = sorted(set(re.findall(r'"(/v1/[A-Za-z0-9_/]+)"', t)))
    i = t.find('v = 제찾기(r, "geulpu")')
    b = t.find("{", i)
    e = 짝괄호(t, b)
    몸 = t[b:e]
    필드 = {"geulpu." + k for k in re.findall(r'제[가-힣]*\(\s*v\s*,\s*"([A-Za-z_]+)"', 몸)}
    필드 |= {"geulpu.layer_lens." + k for k in re.findall(r'제찾기\(\s*ll\s*,\s*"([A-Za-z_]+)"', 몸)}
    필드 |= set(맨위필드)
    return 경로, sorted(필드)


def 읽기(경로들):
    return {p: open(os.path.join(ROOT, p), encoding="utf-8").read() for p in 경로들}


# ---------- 점검 ----------
def 점검(reg, texts):
    못 = []
    def 알림(점, 글):
        못.append(f"[{점}] {글}")
    # 1 형식
    if reg.get("format") != "geulpu-operations/1":
        알림(1, "format 이 geulpu-operations/1 이 아님")
    rt = reg.get("result_types") or {}
    ops = reg.get("operations") or []
    ids = set()
    for o in ops:
        i = o.get("id", "")
        if not ID.match(i):
            알림(1, f"id 형식: {i!r}")
        if i in ids:
            알림(1, f"id 겹침: {i}")
        ids.add(i)
        for f in ("title", "description"):
            v = o.get(f) or {}
            if not (isinstance(v, dict) and v.get("ko") and v.get("en")):
                알림(1, f"{i}: {f} 의 ko · en")
        st = o.get("status")
        if st not in ("exists", "planned"):
            알림(1, f"{i}: status")
        if st == "planned" and not o.get("stage"):
            알림(1, f"{i}: planned 인데 stage 없음")
        if st == "exists" and not o.get("http"):
            알림(1, f"{i}: exists 인데 http 없음")
        h = o.get("http")
        if h is not None and not (isinstance(h, dict) and h.get("method") in ("GET", "POST", "ANY") and str(h.get("path", "")).startswith("/")):
            알림(1, f"{i}: http 의 method · path")
        for p in o.get("params") or []:
            if not p.get("name") or p.get("type") not in 형들 or not isinstance(p.get("required"), bool) \
                    or not (isinstance(p.get("description"), dict) and p["description"].get("ko") and p["description"].get("en")):
                알림(1, f"{i}: 인자 {p.get('name')!r} 의 name · type · required · description")
        if o.get("result") not in rt:
            알림(1, f"{i}: result {o.get('result')!r} 가 result_types 에 없음")
        if not isinstance(o.get("job"), bool):
            알림(1, f"{i}: job 은 참 · 거짓")
        if not o.get("records"):
            알림(1, f"{i}: records")
        if o.get("internal") and not o.get("reason"):
            알림(1, f"{i}: internal 인데 reason 없음")
    # 2 경로 대조
    코드 = 코드뽑기(texts)
    목록경로 = {}
    for o in ops:
        h = o.get("http")
        if h:
            목록경로.setdefault((h["method"], h["path"]), []).append(o)
    for k in sorted(코드):
        if k not in 목록경로:
            알림(2, f"코드에만 있는 경로: {k[0]} {k[1]}")
    for k in sorted(목록경로):
        if k[1].startswith("/api/") and k not in 코드:
            알림(2, f"목록에만 있는 경로: {k[0]} {k[1]}")
    서버경로, 서버필드 = 서버뽑기(texts[서버파일])
    목록패스 = {k[1] for k in 목록경로}
    for p in 서버경로 + 전달경로:
        if p not in 목록패스:
            알림(2, f"목록에 없는 서버 · 전달 경로: {p}")
    # 3 CLI
    for 말 in ("ko", "en"):
        본 = {}
        for o in ops:
            if o.get("internal"):
                continue
            c = (o.get("cli") or {}).get(말, "")
            if not c:
                알림(3, f"{o.get('id')}: CLI 이름({말}) 없음")
            elif c in 본:
                알림(3, f"CLI 이름({말}) 겹침: {c!r} — {본[c]} · {o.get('id')}")
            else:
                본[c] = o.get("id")
    # 4 화면 — 화면 표 · 기능의 gui · 결과 형식의 screens(앱리디자인 E6)
    화면표 = reg.get("screens") or {}
    for 이름, v in 화면표.items():
        if not (isinstance(v, dict) and v.get("kind") in ("view", "part") and v.get("ko") and v.get("en")):
            알림(4, f"화면 {이름}: kind(view · part) · ko · en")
    웹화면 = 웹화면들(texts)
    표화면 = {k for k, v in 화면표.items() if isinstance(v, dict) and v.get("kind") == "view"}
    for v in sorted(웹화면 - 표화면):
        알림(4, f"웹 코드의 화면 {v} 가 기능 목록의 screens 에 없음")
    for v in sorted(표화면 - 웹화면):
        알림(4, f"기능 목록의 화면 {v} 가 웹 코드(VIEW_GROUP)에 없음")
    아이디 = {o.get("id"): o for o in ops}
    for o in ops:
        if o.get("internal"):
            continue
        g = o.get("gui") or {}
        i = o.get("id")
        if o.get("status") == "planned":
            if not g.get("planned"):
                알림(4, f"{i}: 앞으로의 기능인데 gui.planned 없음")
            continue
        sc = g.get("screens")
        if not (isinstance(sc, list) and sc):
            알림(4, f"{i}: gui.screens 없음")
            continue
        for x in sc:
            if x not in 화면표:
                알림(4, f"{i}: 없는 화면 id {x!r}")
        for v in g.get("via") or []:
            w = 아이디.get(v) or {}
            if w.get("status") != "exists" or w.get("internal"):
                알림(4, f"{i}: gui.via 의 {v!r} 가 내부용이 아닌 있는 기능이 아님")
    결과의기능 = {}
    for o in ops:
        결과의기능.setdefault(o.get("result"), []).append(o)
    for 이름, v in rt.items():
        v = v or {}
        생산 = 결과의기능.get(이름, [])
        if not 생산:
            알림(4, f"결과 형식 {이름}: 어느 기능의 결과도 아님")
        if v.get("planned"):
            if any(o.get("status") == "exists" for o in 생산):
                알림(4, f"결과 형식 {이름}: 있는 기능의 결과인데 planned")
            continue
        sc = v.get("screens")
        if not (isinstance(sc, list) and sc):
            알림(4, f"결과 형식 {이름}: screens 없음")
            continue
        for x in sc:
            if x not in 화면표:
                알림(4, f"결과 형식 {이름}: 없는 화면 id {x!r}")
    # 5 웹 코드 — (가) 경로 글 없음 (나) 기능 id 형식의 글은 있는 기능 (다) 내부용이 아닌 있는 기능은 화면 코드에서 불린다(앱리디자인 E6)
    앞부분들 = {i.split(".")[0] for i in ids}
    화면글 = {}
    for f in 화면파일:
        for v, 줄 in JS글들(texts[f]):
            화면글.setdefault(v, []).append(f"{f.split('/')[-1]}:{줄}")
    for v, 곳 in sorted(화면글.items()):
        if 경로글(v):
            알림(5, f"화면 코드의 경로 글: {곳[0]} {v[:60]!r} — 기능 id 로 ui/ops.js 를 부른다")
        elif ID.match(v) and v.split(".")[0] in 앞부분들:
            o = 아이디.get(v)
            if o is None:
                알림(5, f"목록에 없는 기능 id 를 부름: {곳[0]} {v!r}")
            elif o.get("status") != "exists":
                알림(5, f"앞으로의 기능을 부름: {곳[0]} {v!r}")
    for o in ops:
        if o.get("internal") or o.get("status") != "exists":
            continue
        via = (o.get("gui") or {}).get("via") or []
        if o["id"] in 화면글 or (via and all(v in 화면글 for v in via)):
            continue
        알림(5, f"{o['id']}: 화면 코드에서 불리지 않음(화면에서 닿지 않는 기능)")
    # 6 인자
    for k, x in sorted(코드.items()):
        for o in 목록경로.get(k, []):
            있는 = {p.get("name") for p in o.get("params") or []}
            for 키 in sorted(x["keys"] - 있는):
                알림(6, f"{o.get('id')}({k[0]} {k[1]}): 코드가 읽는 키 {키!r} 가 params 에 없음")
    모든인자 = {p.get("name") for o in ops for p in o.get("params") or []}
    for f in 서버필드:
        if f not in 모든인자:
            알림(6, f"서버의 GeulPU 필드 {f!r} 가 어느 기능의 params 에도 없음")
    return 못, 코드


def 자체시험(reg, texts):
    """잘못 열둘을 하나씩 넣어 모두 잡는지(문서/기능목록.md 3 절 7 · 문서/화면구현.md §6)."""
    경우 = []
    t = copy.deepcopy(texts)
    f, 이름, _ = 처리함수[1]                                 # 관리보냄(POST)
    s, e = 함수몸(t[f], 이름)
    t[f] = t[f][:s + 1] + '\n    문자열_같은가(길, "/api/zz/onlycode")이면 { 반환. }\n' + t[f][s + 1:]
    경우.append(("코드에만 있는 경로", reg, t, "[2] 코드에만 있는 경로: POST /api/zz/onlycode"))
    r = copy.deepcopy(reg)
    r["operations"].append({**copy.deepcopy(r["operations"][0]), "id": "zz.onlylist", "http": {"method": "GET", "path": "/api/zz/onlylist"},
                            "cli": {"ko": "zz 목록에만", "en": "zz onlylist"}})
    경우.append(("목록에만 있는 경로", r, texts, "[2] 목록에만 있는 경로: GET /api/zz/onlylist"))
    r = copy.deepcopy(reg)
    x = next(o for o in r["operations"] if not o.get("internal")); x["cli"] = {"ko": "", "en": x["cli"]["en"]}
    경우.append(("CLI 이름 없음", r, texts, f"[3] {x['id']}: CLI 이름(ko) 없음"))
    r = copy.deepcopy(reg)
    xs = [o for o in r["operations"] if not o.get("internal")]; xs[1]["cli"] = dict(xs[0]["cli"])
    경우.append(("CLI 이름 겹침", r, texts, f"[3] CLI 이름(ko) 겹침"))
    r = copy.deepcopy(reg)
    k0 = sorted(r["result_types"])[0]; r["result_types"][k0] = {**r["result_types"][k0], "screens": []}
    경우.append(("화면 없는 결과 형식", r, texts, f"[4] 결과 형식 {k0}: screens 없음"))
    t = copy.deepcopy(texts)
    t[웹파일[0]] += "\nfetch('/api/zz/webonly');\n"
    경우.append(("화면 코드의 경로 글", reg, t, "[5] 화면 코드의 경로 글: app.js:"))
    t = copy.deepcopy(texts)
    t[웹파일[0]] += "\ncallOp('run.zz_unknown');\n"
    경우.append(("목록에 없는 기능 id 를 부름", reg, t, "[5] 목록에 없는 기능 id 를 부름: app.js:"))
    r = copy.deepcopy(reg)
    st = next(o for o in r["operations"] if o["id"] == "system.state")
    r["operations"].append({**copy.deepcopy(st), "id": "system.zz_unreached", "cli": {"ko": "zz 닿지 않음", "en": "zz unreached"}})
    경우.append(("화면에서 닿지 않는 기능", r, texts, "[5] system.zz_unreached: 화면 코드에서 불리지 않음"))
    r = copy.deepcopy(reg)
    x = next(o for o in r["operations"] if not o.get("internal") and o.get("status") == "exists"); x["gui"] = {"screens": ["zz_screen"]}
    경우.append(("없는 화면 id", r, texts, f"[4] {x['id']}: 없는 화면 id 'zz_screen'"))
    r = copy.deepcopy(reg)
    del r["screens"]["perf"]
    경우.append(("기능 목록에 없는 화면", r, texts, "[4] 웹 코드의 화면 perf 가 기능 목록의 screens 에 없음"))
    r = copy.deepcopy(reg)
    r["result_types"]["zz_orphan"] = {"screens": ["common"], "description": {"ko": "zz", "en": "zz"}}
    경우.append(("어느 기능의 결과도 아닌 결과 형식", r, texts, "[4] 결과 형식 zz_orphan: 어느 기능의 결과도 아님"))
    t = copy.deepcopy(texts)
    코드 = 코드뽑기(texts)
    (m, p) = next(k for k in sorted(코드) if k[0] == "POST")
    f2 = sorted(코드[(m, p)]["files"])[0]
    j = t[f2].find('문자열_같은가(길, "%s")' % p)
    b = t[f2].find("{", t[f2].find("이면", j))
    t[f2] = t[f2][:b + 1] + '\n        문자열 zz = 제글보기(o, "zz_unlisted_key", "").\n' + t[f2][b + 1:]
    경우.append(("params 에 없는 키", reg, t, "zz_unlisted_key"))
    잡음 = 0
    for 이름, r, tx, 기대 in 경우:
        못, _ = 점검(r, tx)
        ok = any(기대 in x for x in 못)
        잡음 += ok
        print(f"{'잡음' if ok else '놓침'}  {이름}")
    print(f"자체시험: {잡음}/{len(경우)}")
    return 잡음 == len(경우)


def main():
    필요 = sorted({f for f, _, _ in 처리함수} | set(웹파일) | {서버파일} |
                {os.path.relpath(os.path.join(dp, n), ROOT).replace("\\", "/")
                 for dp, _, ns in os.walk(os.path.join(ROOT, "앱")) for n in ns if n.endswith(".gl")})
    texts = 읽기(필요)
    if "--뽑기" in sys.argv:
        코드 = 코드뽑기(texts); 서버경로, 서버필드 = 서버뽑기(texts[서버파일])
        print(json.dumps({"routes": [{"method": m, "path": p, "keys": sorted(x["keys"]), "job": x["job"], "files": sorted(x["files"])}
                                     for (m, p), x in sorted(코드.items())],
                          "web": sorted(웹경로들(texts)), "server_paths": 서버경로, "server_fields": 서버필드}, ensure_ascii=False, indent=1))
        return 0
    reg = json.load(open(REG, encoding="utf-8"))
    if "--자체시험" in sys.argv:
        return 0 if 자체시험(reg, texts) else 1
    못, 코드 = 점검(reg, texts)
    ops = reg["operations"]
    print(f"기능 {len(ops)} 개(있음 {sum(o['status'] == 'exists' for o in ops)} · 앞으로 {sum(o['status'] == 'planned' for o in ops)} · "
          f"내부용 {sum(bool(o.get('internal')) for o in ops)}), 코드의 경로(메서드마다) {len(코드)} 개, 결과 형식 {len(reg['result_types'])} 개")
    for x in 못:
        print("못 미침 " + x)
    print("통과" if not 못 else f"못 미침 {len(못)}")
    return 1 if 못 else 0


if __name__ == "__main__":
    sys.exit(main())
