"""기록기의 JSON 문법 훑기 시험(GPU 없이) — 엔진/시험_디버거제이슨(앱/디버거제이슨.gl)의 맞음 · 틀림을 파이썬의 엄격한 풀기와 견준다:
UTF-8(엄격 — 넘는 꼴 · 대리 · U+10FFFF 너머는 틀림) → json.loads(NaN · Infinity 는 틀림), 그리고 깊이 128(배열 · 객체가 128 겹까지).
사례: 손으로 쓴 맞음 · 틀림, UTF-8 경계, 시험 묶음(시험/디버거/묶음50.jsonl)의 요청, 수가 많은 응답(prompt_tokens 2 만 개), 그것들을
씨앗을 박은 무작위로 바이트 하나 · 둘씩 고친 것. 빠른점검이 부른다.

    python -I 시험/디버거제이슨.py <시험_디버거제이슨 실행 파일> [--고침 4000] [--씨앗 7]
"""
import argparse
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
깊이최대 = 128


def 상수거부(c):
    raise ValueError(c)


def 깊이(o):
    if isinstance(o, list):
        return 1 + max((깊이(x) for x in o), default=0)
    if isinstance(o, dict):
        return 1 + max((깊이(x) for x in o.values()), default=0)
    return 0


def 기대(b):
    try:
        s = b.decode("utf-8")
        o = json.loads(s, parse_constant=상수거부)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return False
    return 깊이(o) <= 깊이최대


def 손사례():
    맞음 = ['{}', '[]', '0', '-0', '1.5e10', '-0.0E-0', '1e400', '"a"', '""', '"\\u00e9\\uD800\\/\\b\\f\\n\\r\\t\\"\\\\"', 'true',
           'false', 'null', ' {"a" : 1 } ', '{"a":[1,2,{"b":null}],"c":{"d":[]}}', '[[[]]]', '\t\r\n[1]\n', '"é한글😀"',
           '{"a":1,"a":2}', '[-1,0.5,123e-7,9223372036854775808]', '"\x7f"']
    틀림 = ['', ' ', '{', '}', '[', ']', '[1,]', '{"a":1,}', '01', '1.', '.5', '1e', '1e+', '-', '+1', '"abc', '"\\x"', '"\\u12"',
           '"\\u12G4"', 'tru', 'nul', 'falsy', '[1 2]', '{"a" 1}', '{1:2}', '"a\tb"', '"a\nb"', '﻿{}', '{} x', '{}{}', 'NaN',
           'Infinity', '-Infinity', '[NaN]', "{'a':1}", '[1,,2]', '{"a"::1}', '{,}', '[,]', '"\\"', '\x00', '[1]\x00', 'TRUE',
           '[01]', '-a', '1.e5', '{"a":1 "b":2}']
    사례 = [x.encode("utf-8") for x in 맞음 + 틀림]
    for d in (1, 2, 127, 128, 129, 200):           # 깊이 경계
        사례.append(b"[" * d + b"]" * d)
        사례.append(b'{"a":' * d + b"1" + b"}" * d)
    for 바이트 in (b"\x80", b"\xc0\x80", b"\xc1\xbf", b"\xc2\x80", b"\xdf\xbf", b"\xe0\x80\x80", b"\xe0\xa0\x80", b"\xed\x9f\xbf",
                 b"\xed\xa0\x80", b"\xef\xbf\xbf", b"\xf0\x8f\xbf\xbf", b"\xf0\x90\x80\x80", b"\xf4\x8f\xbf\xbf", b"\xf4\x90\x80\x80",
                 b"\xf5\x80\x80\x80", b"\xff", b"\xe2\x82", b"\xf0\x9f\x98", b"\xc3"):    # UTF-8 경계(글 안 · 글 끝)
        사례.append(b'"' + 바이트 + b'"')
        사례.append(b'["' + 바이트)
        사례.append(바이트)
    return 사례


def 바탕사례():
    out = []
    묶음 = os.path.join(ROOT, "시험", "디버거", "묶음50.jsonl")
    for l in open(묶음, encoding="utf-8"):
        if l.strip():
            out.append(l.strip().encode("utf-8"))
    r = random.Random(11)
    응답 = {"id": "chatcmpl-geulpu-1", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant",
            "content": "한국의 수도는 서울입니다.\n\"따옴표\" \\ 끝"}, "finish_reason": "stop"}],
            "geulpu": {"seed": 18446744073709551615, "tokens": [r.randrange(152064) for _ in range(500)],
                       "logit_hashes": [str(r.getrandbits(64)) for _ in range(500)],
                       "prompt_tokens": [r.randrange(152064) for _ in range(20000)], "first_token_ms": 2900679, "x": -1.25e-3}}
    out.append(json.dumps(응답, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    out.append(json.dumps(응답, ensure_ascii=True, indent=1).encode("utf-8"))
    return out


def 고치기(b, r):
    b = bytearray(b)
    글자 = b'{}[]",:0123456789eE.+- \t\nturefalsn\\u\x00\x1f\x7f\x80\xc3\xa9\xed\xa0\xf4\x90\xff'
    for _ in range(r.choice((1, 1, 2))):
        if not b:
            b.append(r.choice(글자))
            continue
        i = r.randrange(len(b))
        일 = r.randrange(3)
        if 일 == 0:
            del b[i]
        elif 일 == 1:
            b.insert(i, r.choice(글자))
        else:
            b[i] = r.choice(글자)
    return bytes(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exe")
    p.add_argument("--고침", type=int, default=4000)
    p.add_argument("--씨앗", type=int, default=7)
    a = p.parse_args()
    r = random.Random(a.씨앗)
    바탕 = 손사례() + 바탕사례()
    작은 = [x for x in 바탕 if len(x) < 4096]
    사례 = list(바탕)
    for k in range(a.고침):
        사례.append(고치기(r.choice(작은) if k % 8 else r.choice(바탕), r))
    기대값 = [기대(x) for x in 사례]
    뿌리 = tempfile.mkdtemp(prefix="디버거제이슨_")
    try:
        들 = os.path.join(뿌리, "사례.bin")
        날 = os.path.join(뿌리, "결과.bin")
        with open(들, "wb") as f:
            f.write(struct.pack("<I", len(사례)))
            for x in 사례:
                f.write(struct.pack("<I", len(x)))
                f.write(x)
        q = subprocess.run([a.exe, 들, 날], capture_output=True, stdin=subprocess.DEVNULL)
        줄 = (q.stdout or b"").decode("utf-8", "replace").strip()
        if q.returncode != 0 or not os.path.exists(날):
            print(f"기록기 JSON 훑기(GPU 없이): 못 미침 — 실행 실패(종료 {q.returncode}) {줄} {(q.stderr or b'').decode('utf-8', 'replace')[:300]}")
            return 1
        받음 = open(날, "rb").read()
    finally:
        shutil.rmtree(뿌리, ignore_errors=True)
    if len(받음) != len(사례):
        print(f"기록기 JSON 훑기(GPU 없이): 못 미침 — 결과 {len(받음)} 개 · 사례 {len(사례)} 개")
        return 1
    다름 = [i for i, (x, y) in enumerate(zip(받음, 기대값)) if bool(x) != y]
    맞음수 = sum(기대값)
    if 다름:
        for i in 다름[:8]:
            print(f"  사례 {i}: 파이썬 {기대값[i]} · 글(geul) {bool(받음[i])} — {사례[i][:120]!r}")
        print(f"기록기 JSON 훑기(GPU 없이): 못 미침 — 사례 {len(사례):,} 중 {len(다름)} 개가 파이썬과 다르다")
        return 1
    print(f"기록기 JSON 훑기(GPU 없이): 통과 — 사례 {len(사례):,}(맞음 {맞음수:,} · 틀림 {len(사례) - 맞음수:,})가 파이썬의 엄격한 풀기와 같다 · {줄}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
