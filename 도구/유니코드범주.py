#!/usr/bin/env python3
"""0.12.0 12-G2 — 정규식 사전 처리의 유니코드 범주 표(엔진/유니코드범주.gl)를 llama.cpp 의 src/unicode-data.cpp 에서 만든다.
llama.cpp 와 같은 토큰을 내려면 범주(숫자 · 글자 · 구두점 · 결합 문자 · 기호 …)와 공백이 llama.cpp 의 표와 같아야 한다
(그 표는 llama.cpp 의 scripts/gen-unicode-data.py 가 만든 것 — MIT, Copyright (c) 2023-2026 The ggml authors).

  python -I 도구/유니코드범주.py <llama.cpp 의 src/unicode-data.cpp> <llama.cpp 의 판 — 예: b11829> 엔진/유니코드범주.gl

표: unicode_ranges_flags 의 (시작, 범주) — 범주는 그 시작부터 다음 시작 앞까지(마지막은 0x110000), 그리고 unicode_set_whitespace.
"""
import re
import sys

src, 판, out = sys.argv[1:4]
t = open(src, encoding="utf-8").read()


def 덩이(이름):
    m = re.search(re.escape(이름) + r"\s*=\s*\{(.*?)\n\};", t, re.S)
    if not m:
        sys.exit(f"{이름} 를 찾지 못했다")
    return m.group(1)


쌍 = [(int(a, 16), int(b, 16)) for a, b in re.findall(r"\{0x([0-9A-Fa-f]+),\s*0x([0-9A-Fa-f]+)\}", 덩이("unicode_ranges_flags"))]
assert 쌍[0][0] == 0 and 쌍[-1][0] == 0x110000, (쌍[0], 쌍[-1])
assert all(쌍[i][0] < 쌍[i + 1][0] for i in range(len(쌍) - 1))
공백 = sorted(int(x, 16) for x in re.findall(r"0x([0-9A-Fa-f]+)", 덩이("unicode_set_whitespace")))
범주 = [f & 0xFF for _, f in 쌍[:-1]]                 # 마지막(0x110000)은 끝 표시 — 범주 0
assert all(b in (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80) for b in 범주), sorted(set(범주))

줄 = []
낱 = [f"{a:X} {b:X}" for (a, b) in 쌍]
for i in range(0, len(낱), 12):
    줄.append(f'    범주조각("{" ".join(낱[i:i + 12])}").')
글 = f"""(* llama.cpp {판} 의 src/unicode-data.cpp(scripts/gen-unicode-data.py 가 만든 표 — MIT, Copyright (c) 2023-2026 The ggml authors — THIRD_PARTY_NOTICES.md)에서
   도구/유니코드범주.py 가 만들었다. 고치지 말 것(다시 만든다). 0.12.0 12-G2 — 정규식 사전 처리의 범주(엔진/정규식.gl).
   범주조각: (시작 범주) 쌍 — 범주는 그 시작부터 다음 시작 앞까지(마지막 시작 110000 은 끝 표시). 범주 비트는 llama.cpp 의 unicode_cpt_flags 와
   같다(1 정의 없음 · 2 숫자 · 4 글자 · 8 구분자 · 10 결합 문자 · 20 구두점 · 40 기호 · 80 제어). 범주공백: unicode_set_whitespace({len(공백)} 개). *)

[범주표넣기]는 {{
{chr(10).join(줄)}
    범주공백("{" ".join(f"{c:X}" for c in 공백)}").
}}
"""
open(out, "wb").write(글.encode("utf-8"))
print(f"범위 {len(쌍) - 1} · 공백 {len(공백)} → {out}")
