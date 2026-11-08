#!/usr/bin/env python3
"""앱 화면(앱/web)의 GPU 없는 점검 — 빠른 점검 2c(시험/빠른점검.py 가 부른다).

  1. app.js · eval.js 문법 — node --check(node 가 있을 때).
  2. 영어 사전 — 화면 문구는 한국어로 쓰고 T('…') · T`…${x}…` 로 감싸 영어 화면이면 DICT 에서 바꾼다(app.js 머리말 — eval.js 의 문구는
     그 파일의 EVAL_DICT 가 DICT 에 더해진다). 두 파일의 T 호출
     가운데 첫 인자가 글 그대로인 것(작은따옴표 글 · 템플릿)과 index.html 의 정적 문구(aria-label · data-tip · placeholder · title ·
     글)가 모두 DICT 에 있는지 본다. 문맥을 붙인 T('파일', 'k') 는 '파일|k' 나 '파일' 가운데 하나. 주석 · 다른 글 안의 T 는 세지 않는다
     (작은 JS 훑기 — 글 · 템플릿 · 주석 · 정규식을 건너뛴다). DICT 의 같은 열쇠 두 번도 틀림으로 센다(뒤의 것이 앞을 덮는다).

    python -X utf8 시험/화면사전.py      → 마지막 줄 "화면 점검: 통과 — …"(종료 0), 틀리면 종료 1
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "앱", "web", "app.js")
EVAL = os.path.join(ROOT, "앱", "web", "eval.js")          # 모델 평가 1-3 — 사전은 EVAL_DICT
HTML = os.path.join(ROOT, "앱", "web", "index.html")
UI = os.path.join(ROOT, "앱", "web", "ui")                   # 앱리디자인 E1 — 화면 모듈(ES 모듈). 문구는 app.js 의 T 와 DICT 를 쓴다
UI_FILES = sorted(os.path.join(UI, f) for f in os.listdir(UI) if f.endswith(".js")) if os.path.isdir(UI) else []
한글 = re.compile(r"[가-힣]")
정규식앞 = set("(,=:[!&|?{};+-*%<>~^")
정규식앞말 = ("return", "typeof", "case", "in", "of", "delete", "void", "throw", "new", "else", "do")


def js풀기(t):
    """JS 글의 이스케이프를 푼 값(T 가 받는 글)"""
    out, i = [], 0
    while i < len(t):
        c = t[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        n = t[i + 1] if i + 1 < len(t) else ""
        if n == "u" and i + 2 < len(t) and t[i + 2] == "{":
            e = t.index("}", i + 3)
            out.append(chr(int(t[i + 3:e], 16)))
            i = e + 1
        elif n == "u":
            out.append(chr(int(t[i + 2:i + 6], 16)))
            i += 6
        elif n == "x":
            out.append(chr(int(t[i + 2:i + 4], 16)))
            i += 4
        elif n == "\n":
            i += 2
        else:
            out.append({"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}.get(n, n))
            i += 2
    return "".join(out)


def 글끝(s, i):
    """s[i] 가 따옴표 — 닫는 따옴표 다음 자리"""
    q, i = s[i], i + 1
    while s[i] != q:
        i += 2 if s[i] == "\\" else 1
    return i + 1


def 템플릿(s, i, out):
    """s[i] 가 백틱 — (닫는 백틱 다음 자리, ${…} 를 {} 로 바꾼 날 글). ${…} 안의 T 호출도 out 에"""
    i += 1
    날 = []
    while True:
        c = s[i]
        if c == "\\":
            날.append(s[i:i + 2])
            i += 2
        elif c == "`":
            return i + 1, "".join(날)
        elif s.startswith("${", i):
            i = 훑기(s, i + 2, out, 닫기=True)
            날.append("{}")
        else:
            날.append(c)
            i += 1


def T경계(s, i):
    """s[i:] 가 T 이고 앞이 이름의 일부가 아닌가"""
    return s[i] == "T" and (i == 0 or not (s[i - 1].isalnum() or s[i - 1] in "_$."))


def 뽑기(s):
    """app.js 의 T 열쇠 — [(열쇠, 문맥 또는 None, 줄 번호)]"""
    out = []
    훑기(s, 0, out)
    return out


def 훑기(s, i, out, 닫기=False):
    """JS 코드를 i 부터 훑어 T 호출을 out 에 — 닫기면 짝이 맞는 } 다음 자리에서 멈춘다(템플릿의 ${…})"""
    n = len(s)
    앞 = ";"                                    # 마지막 의미 있는 글자(정규식인가 나눗셈인가)
    앞말 = ""
    깊이 = 0
    while i < n:
        c = s[i]
        if 닫기 and c == "{":
            깊이 += 1
        elif 닫기 and c == "}":
            if 깊이 == 0:
                return i + 1
            깊이 -= 1
        if s.startswith("//", i):
            j = s.find("\n", i)
            i = n if j < 0 else j
            continue
        if s.startswith("/*", i):
            i = s.index("*/", i + 2) + 2
            continue
        if c in "'\"":
            j = 글끝(s, i)
            if c == "'" and i >= 2 and s[i - 1] == "(" and T경계(s, i - 2):
                열쇠, 문맥 = js풀기(s[i + 1:j - 1]), None
                k = j
                while s[k] in " \t\n":
                    k += 1
                if s[k] == ",":
                    k += 1
                    while s[k] in " \t\n":
                        k += 1
                    if s[k] == "'":
                        e = 글끝(s, k)
                        문맥 = js풀기(s[k + 1:e - 1])
                out.append((열쇠, 문맥, s.count("\n", 0, i) + 1))
            i, 앞, 앞말 = j, "a", ""
            continue
        if c == "`":
            j, 날 = 템플릿(s, i, out)
            if i >= 1 and T경계(s, i - 1):
                부분 = 날.split("{}")
                out.append(("{}".join(js풀기(p) for p in 부분), None, s.count("\n", 0, i) + 1))
            i, 앞, 앞말 = j, "a", ""
            continue
        if c == "/" and (앞 in 정규식앞 or 앞말 in 정규식앞말):  # 정규식 — 닫는 / 까지([…] 안의 / 는 건너뜀)
            i += 1
            묶음안 = False
            while True:
                d = s[i]
                if d == "\\":
                    i += 2
                    continue
                if d == "[":
                    묶음안 = True
                elif d == "]":
                    묶음안 = False
                elif d == "/" and not 묶음안:
                    break
                i += 1
            i += 1
            앞, 앞말 = "a", ""
            continue
        if c.isalnum() or c in "_$":
            j = i
            while j < n and (s[j].isalnum() or s[j] in "_$"):
                j += 1
            앞말, 앞 = s[i:j], "a"
            i = j
            continue
        if not c.isspace():
            앞, 앞말 = c, ""
        i += 1
    return i


def 사전(s, 머리="const DICT = {"):
    i = s.index(머리)
    j = s.index("\n};", i)
    열쇠, 겹침 = {}, []
    for k, line in enumerate(s[i:j].split("\n")[1:], start=s.count("\n", 0, i) + 2):
        t = line.strip()
        if not t.startswith('"'):
            continue
        try:
            (key, val), = json.loads("{" + t.rstrip(",") + "}").items()
        except ValueError:
            continue
        if key in 열쇠:
            겹침.append((key, 열쇠[key], k))
        열쇠[key] = k
    return 열쇠, 겹침


def html문구(h):
    """index.html 의 정적 문구 — 한글이 든 속성 · 글(applyStatic 이 T 로 바꾸는 것)"""
    out = [m.group(2) for m in re.finditer(r'\b(aria-label|data-tip|placeholder|title)="([^"]*)"', h) if 한글.search(m.group(2))]
    본 = re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", "\n", h, flags=re.S)
    out += [t.strip() for t in 본.split("\n") if t.strip() and 한글.search(t)]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    틀림 = []
    node = shutil.which("node")
    if node:
        for f in (APP, EVAL):
            r = subprocess.run([node, "--check", f], capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                틀림.append(os.path.basename(f) + " 문법: " + r.stderr.strip()[-400:])
        # 화면 모듈은 ES 모듈 — node 가 어느 버전이든 모듈로 읽도록 .mjs 로 옮겨 본다
        tmp = tempfile.mkdtemp(prefix="화면사전_")
        try:
            for f in UI_FILES:
                m = os.path.join(tmp, os.path.basename(f)[:-3] + ".mjs")
                shutil.copyfile(f, m)
                r = subprocess.run([node, "--check", m], capture_output=True, text=True, encoding="utf-8")
                if r.returncode != 0:
                    틀림.append("ui/" + os.path.basename(f) + " 문법: " + r.stderr.strip()[-400:])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        문법 = f"app.js · eval.js · 화면 모듈 {len(UI_FILES)} 개 문법 통과"
    else:
        문법 = "node 가 없어 app.js · eval.js · 화면 모듈 문법은 건너뜀"
    s = open(APP, encoding="utf-8").read()
    se = open(EVAL, encoding="utf-8").read()
    열쇠, 겹침 = 사전(s)
    for key, a, b in 겹침:
        틀림.append(f"DICT 의 같은 열쇠 두 번(줄 {a} · {b}): {key[:60]}")
    열쇠e, 겹침e = 사전(se, "const EVAL_DICT = {")
    for key, a, b in 겹침e:
        틀림.append(f"EVAL_DICT 의 같은 열쇠 두 번(줄 {a} · {b}): {key[:60]}")
    for key, b in 열쇠e.items():
        if key in 열쇠:
            틀림.append(f"DICT(app.js 줄 {열쇠[key]})와 EVAL_DICT(eval.js 줄 {b})에 같은 열쇠: {key[:60]}")
    열쇠 = {**열쇠, **열쇠e}
    호출 = []
    모듈 = [("ui/" + os.path.basename(f), open(f, encoding="utf-8").read()) for f in UI_FILES]
    for 이름, 글 in [("app.js", s), ("eval.js", se)] + 모듈:
        for key, 문맥, 줄 in 뽑기(글):
            if not 한글.search(key):
                continue
            호출.append((key, 문맥, 줄))
            if key in 열쇠 or (문맥 is not None and f"{key}|{문맥}" in 열쇠):
                continue
            틀림.append(f"{이름} 줄 {줄}: 사전에 없음 — {json.dumps(key, ensure_ascii=False)[:90]}")
    정적 = html문구(open(HTML, encoding="utf-8").read())
    for t in 정적:
        if t not in 열쇠:
            틀림.append(f"index.html: 사전에 없음 — {t[:60]}")
    for t in 틀림[:int(os.environ.get("GEULPU_DICT_SHOW", "30"))]:
        print("틀림 —", t)
    if 틀림:
        print(f"화면 점검: 못 미침 {len(틀림)} — {문법}")
        return 1
    print(f"화면 점검: 통과 — {문법} · 화면 문구 {len(호출):,}개와 index.html 문구 {len(정적)}개가 영어 사전({len(열쇠):,}개)에 있음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
