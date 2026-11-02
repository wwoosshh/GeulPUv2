#!/usr/bin/env python3
"""모델 평가 3-1(문서/모델평가.md 3.9) — 정답 채점(앱/평가채점.gl)의 파이썬 참조와 GPU 없는 대조(엔진/시험_채점.gl).

채점 규칙(number · choice · exact · tool_call)과 정답률의 부트스트랩 95 % 구간을 같은 규칙으로 계산해, 프로그램의 판정 · 구간의 비트가
같은가 본다. 사례는 손으로 정한 것(기대 판정을 함께 적음 — 규칙이 뜻대로인가)과 무작위로 지은 것(참조와 같은가).

    python -I -X utf8 시험/채점.py check <시험_채점.exe>        (빠른 점검 · CI)
"""
import json
import os
import random
import struct
import subprocess
import sys
import tempfile


def 숫자(c):
    return "0" <= c <= "9"


def 영문(c):
    return ("A" <= c <= "Z") or ("a" <= c <= "z")


def 수맞추기(t):
    음 = t.startswith("-")
    if 음:
        t = t[1:]
    정, _, 소 = t.replace(",", "").partition(".")
    정 = 정.lstrip("0") or "0"
    소 = 소.rstrip("0")
    영 = 정 == "0" and not 소
    return ("-" if 음 and not 영 else "") + 정 + ("." + 소 if 소 else "")


def 마지막수(s, a=0):
    b = s.encode("utf-8")
    n = len(b)
    시작 = 끝 = -1
    i = a
    while i < n:
        c = chr(b[i])
        if 숫자(c):
            b0 = i
            if b0 > a and b[b0 - 1] == 45 and (b0 - 1 == a or (not 숫자(chr(b[b0 - 2])) and not 영문(chr(b[b0 - 2])))):
                b0 -= 1
            j = i
            점 = False
            while j < n:
                d = chr(b[j])
                if 숫자(d):
                    j += 1
                elif d == "," and not 점 and j + 1 < n and 숫자(chr(b[j + 1])):
                    j += 1
                elif d == "." and not 점 and j + 1 < n and 숫자(chr(b[j + 1])):
                    점 = True
                    j += 1
                else:
                    break
            시작, 끝 = b0, j
            i = j
        else:
            i += 1
    if 시작 < 0:
        return None
    return 수맞추기(b[시작:끝].decode("ascii"))


def 정답수(s):
    b = s.encode("utf-8")
    a = 0
    for k in range(len(b) - 3):
        if b[k:k + 4] == b"####":
            a = k + 4
    return 마지막수(s, a)


def 글자(s):
    b = s.encode("utf-8")
    n = len(b)
    고름 = -1
    for k in range(n - 8):
        if bytes(c + 32 if 65 <= c <= 90 else c for c in b[k:k + 9]) == b"answer is":
            p = k + 9
            while p < n and b[p] in (32, 58, 9):
                p += 1
            if p < n and b[p] == 40:
                p += 1
            if p < n and 65 <= b[p] <= 74 and (p + 1 >= n or not 영문(chr(b[p + 1]))):
                고름 = b[p]
    if 고름 >= 0:
        return 고름
    a, e = 0, n
    while a < e and b[a] <= 32:
        a += 1
    while e > a and b[e - 1] <= 32:
        e -= 1
    if e - a == 3 and b[a] == 40 and b[a + 2] == 41:
        a += 1
        e -= 1
    if e - a == 2 and b[e - 1] in (46, 41):
        e -= 1
    if e - a == 1 and 65 <= b[a] <= 74:
        return b[a]
    return -1


def 다듬기(s):
    out = bytearray()
    빈칸 = False
    for c in s.encode("utf-8"):
        if c in (32, 9, 10, 13):
            빈칸 = True
        else:
            if 빈칸 and out:
                out.append(32)
            빈칸 = False
            out.append(c + 32 if 65 <= c <= 90 else c)
    while True:
        if out and out[-1] in (46, 33, 63):
            del out[-1]
        elif len(out) >= 3 and out[-3:] == b"\xe3\x80\x82":
            del out[-3:]
        else:
            break
    if out and out[-1] == 32:
        del out[-1]
    return bytes(out)


def 같은값(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b) if (isinstance(a, float) or isinstance(b, float)) else a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(같은값(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and 같은값(v, b[k]) for k, v in a.items())
    return a == b


def 인자(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return None
    return v


def 채점(grader, expected, answer, tool):
    answer = answer or ""
    if grader == "number":
        if not isinstance(expected, str):
            return -1
        x, y = 정답수(expected), 마지막수(answer)
        return 1 if x is not None and y is not None and x == y else 0
    if grader == "choice":
        if not isinstance(expected, str):
            return -1
        x, y = 글자(expected), 글자(answer)
        return 1 if x >= 0 and x == y else 0
    if grader == "exact":
        if not isinstance(expected, str):
            return -1
        return 1 if 다듬기(expected) == 다듬기(answer) else 0
    if grader == "tool_call":
        if not isinstance(expected, dict):
            return -1
        if not isinstance(tool, dict):
            return 0
        if "name" not in expected or "name" not in tool or not 같은값(expected["name"], tool["name"]):
            return 0
        return 1 if 같은값(인자(expected.get("arguments")), 인자(tool.get("arguments"))) else 0
    return -1


def 구간(맞음):
    n = len(맞음)
    if n < 1:
        return 0.0, 0.0
    x = 20261102
    값 = []
    for _ in range(1000):
        합 = 0
        for _ in range(n):
            x = (1103515245 * x + 12345) & 2147483647
            합 += 맞음[x % n]
        값.append(합 / n)
    값.sort()
    return 값[24], 값[974]


def 비트(x):
    return struct.unpack("<q", struct.pack("<d", x))[0]


# 손으로 정한 사례 — (grader, expected, answer, tool, 기대 판정)
손 = [
    ("number", "Natalia sold 48/2 = 24 clips in May.\n#### 72", "So the total is 72.", None, 1),
    ("number", "#### 1,000", "She pays $1000 in total.", None, 1),
    ("number", "#### 12.50", "The answer is 12.5 dollars", None, 1),
    ("number", "#### 3", "3.0", None, 1),
    ("number", "#### -7", "The temperature drops to -7 degrees.", None, 1),
    ("number", "#### 7", "The temperature drops to -7 degrees.", None, 0),
    ("number", "#### 18", "First 6 + 12 = 18, then 18 - 4 = 14.", None, 0),
    ("number", "#### 42", "", None, 0),
    ("number", "#### 0", "The difference is -0", None, 1),
    ("number", "#### 5", "x-5 is the gap; answer: 5", None, 1),
    ("number", "#### 2026", "Year 2,026.", None, 1),
    ("number", "#### 007", "7", None, 1),
    ("number", "답: 15", "정답은 15개입니다.", None, 1),
    ("choice", "C", "The answer is (C).", None, 1),
    ("choice", "C", "the answer is c", None, 0),
    ("choice", "B", "I think the answer is (A). Wait — the answer is (B)", None, 1),
    ("choice", "D", "D", None, 1),
    ("choice", "D", "(D)", None, 1),
    ("choice", "A", "A.", None, 1),
    ("choice", "A", "Answer is A because", None, 1),
    ("choice", "A", "The answer is Apple", None, 0),
    ("choice", "J", "answer is: (J)", None, 1),
    ("choice", "K", "The answer is (K)", None, 0),
    ("exact", "Paris", "  paris. ", None, 1),
    ("exact", "New York", "new   york!", None, 1),
    ("exact", "서울", "서울。", None, 1),
    ("exact", "Paris", "Paris, France", None, 0),
    ("tool_call", {"name": "get_weather", "arguments": {"city": "Seoul"}}, "", {"name": "get_weather", "arguments": "{\"city\": \"Seoul\"}"}, 1),
    ("tool_call", {"name": "get_weather", "arguments": {"city": "Seoul", "unit": "c"}}, "", {"name": "get_weather", "arguments": {"unit": "c", "city": "Seoul"}}, 1),
    ("tool_call", {"name": "get_weather", "arguments": {"city": "Seoul"}}, "", {"name": "get_time", "arguments": {"city": "Seoul"}}, 0),
    ("tool_call", {"name": "add", "arguments": {"a": 1, "b": 2.0}}, "", {"name": "add", "arguments": "{\"a\": 1.0, \"b\": 2}"}, 1),
    ("tool_call", {"name": "add", "arguments": {"a": 1}}, "", None, 0),
    ("tool_call", {"name": "add", "arguments": {"a": [1, 2]}}, "", {"name": "add", "arguments": {"a": [2, 1]}}, 0),
    ("nope", "x", "x", None, -1),
]


def 무작위(r, n):
    out = []
    글들 = ["The answer is (B).", "answer is c", "So 1,234.50 apples", "-3 and 4", "#### 10", "x-1", "Total: 2,000,000", "3.14159", "(E)", "e.", "Answer is: F",
           "The final answer is 0.0", "  Hello,   World!! ", "안녕하세요.", "12-5=7", "a1b2c3", "1.2.3", "-", "--5", "1,23,4"]
    for _ in range(n):
        g = r.choice(["number", "choice", "exact"])
        out.append((g, r.choice(글들), r.choice(글들), None, None))
    return out


def check(exe):
    sys.stdout.reconfigure(encoding="utf-8")
    r = random.Random(3)
    사례 = 손 + 무작위(r, 400)
    부트 = [[1] * 10, [0] * 7, [r.randint(0, 1) for _ in range(60)], [r.randint(0, 1) for _ in range(300)], [1, 0]]
    틀림 = []
    with tempfile.TemporaryDirectory() as 폴더:
        pi = os.path.join(폴더, "cases.json")
        po = os.path.join(폴더, "out.json")
        json.dump({"cases": [{"grader": g, "expected": e, "answer": a, "tool": t} for g, e, a, t, _ in 사례], "bootstrap": 부트}, open(pi, "w", encoding="utf-8"), ensure_ascii=False)
        rr = subprocess.run([os.path.abspath(exe), pi, po], capture_output=True, timeout=120)
        if rr.returncode != 0 or not os.path.exists(po):
            print(f"시험_채점 실패({rr.returncode}): {(rr.stdout + rr.stderr).decode('utf-8', 'replace')[-300:]}")
            return 1
        out = json.load(open(po, encoding="utf-8"))
    for k, (g, e, a, t, 기대) in enumerate(사례):
        참조 = 채점(g, e, a, t)
        if 기대 is not None and 참조 != 기대:
            틀림.append(f"참조가 기대와 다르다 — {g} {e!r} · {a!r}: {참조} (기대 {기대})")
        if out["verdicts"][k] != 참조:
            틀림.append(f"판정이 다르다 — {g} {e!r} · {a!r}: 프로그램 {out['verdicts'][k]} · 참조 {참조}")
    for k, m in enumerate(부트):
        lo, hi = 구간(m)
        if out["ci"][k] != [비트(lo), 비트(hi)]:
            틀림.append(f"구간이 다르다 — n {len(m)}: 프로그램 {out['ci'][k]} · 참조 {[비트(lo), 비트(hi)]} ({lo} · {hi})")
    for t in 틀림[:20]:
        print("틀림 —", t)
    if 틀림:
        print(f"정답 채점: 못 미침 {len(틀림)}")
        return 1
    print(f"정답 채점: 통과 — 판정 {len(사례)} 사례(손으로 정한 {len(손)} · 무작위 {len(사례) - len(손)}) · 부트스트랩 구간 {len(부트)} 경우의 비트가 참조와 같음")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "check":
        sys.exit(check(sys.argv[2]))
    print(__doc__)
    sys.exit(2)
