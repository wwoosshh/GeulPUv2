#!/usr/bin/env python3
"""근사 명령 흉내의 표(백엔드/SPIRV/근사표.gl)를 ptx-math(MIT, github.com/bcarlet/ptx-math)의 C 표에서 만든다 — llama.cpp 실행의 비트
재현 3단계(문서/라마재현.md). 값은 C 파일의 16진수 그대로 십진수로 옮긴다.

  python -I -X utf8 시험/근사표만들기.py <ptx-math 의 src/ptxm/tables 폴더>
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
이름들 = [("ex2", "2^x"), ("rcp", "1/x"), ("rsqrt", "1/√x"), ("lg2", "log₂x"), ("sin", "sin · cos")]


def main():
    base = sys.argv[1]
    out = [
        "(* 근사 명령 흉내의 표 — ptx-math(MIT, github.com/bcarlet/ptx-math)의 src/ptxm/tables/*_table.c 를 그대로 옮긴 것(시험/근사표만들기.py 가",
        "   그 C 파일에서 만든다 — 손으로 고치지 말 것). 표마다 칸이 셋(c0 · c1 · c2)인 줄들: ex2 64 · rcp 128 · rsqrt 128 · lg2 64 · sin 64 줄.",
        "   SPIR-V 뒷단(근사.gl)이 커널 모듈에 상수 배열로 싣는다. ptx-math 의 저작권 고지는 THIRD_PARTY_NOTICES.md 에 둔다. *)",
        "",
        "정수목록 참조 근사표모음 = 없음.        (* 번호 → 낱말 목록(정수목록 참조를 정수로) *)",
        "",
        "(* 공백으로 가른 십진수들을 목록에 *)",
        "[정수목록 참조 L에 문자열 글을 근사수넣기]는 {",
        "    정수 n = 글자수(글).",
        "    정수 v = 0.",
        "    참거짓 안 = 거짓.",
        "    반복 (정수 k는 0부터 n 전까지) {",
        "        정수 c = 글[k] 으로 정수.",
        "        (c >= 48 그리고 c <= 57)이면 { v = v * 10 + (c - 48). 안 = 참. }",
        "        아니면 { 안이면 { 정수목록_추가(L, v). v = 0. 안 = 거짓. } }",
        "    }",
        "    안이면 { 정수목록_추가(L, v). }",
        "}",
        "",
    ]
    for i, (n, desc) in enumerate(이름들):
        s = open(os.path.join(base, f"{n}_table.c"), encoding="utf-8").read()
        nums = [int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", s)]
        assert len(nums) % 3 == 0, n
        out.append(f"(* {i}: {n}({desc}) — {len(nums) // 3} 줄 *)")
        out.append(f"[근사표{n}]는 -> 정수목록 참조 {{")
        out.append("    정수목록 참조 L = 정수목록_만들기().")
        for k in range(0, len(nums), 24):
            out.append('    L에 "' + " ".join(str(v) for v in nums[k:k + 24]) + '"을 근사수넣기하다.')
        out.append("    반환 L.")
        out.append("}")
        out.append("")
    out.append("[정수 번호를 근사표자료]는 -> 정수목록 참조 {")
    out.append("    근사표모음 == 없음이면 {")
    out.append("        근사표모음 = 정수목록_만들기().")
    for n, _ in 이름들:
        out.append(f"        정수목록_추가(근사표모음, 근사표{n}() 으로 정수).")
    out.append("    }")
    out.append("    반환 정수목록_값(근사표모음, 번호) 으로 정수목록 참조.")
    out.append("}")
    with open(os.path.join(ROOT, "백엔드", "SPIRV", "근사표.gl"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print(f"백엔드/SPIRV/근사표.gl: {len(out)} 줄")


if __name__ == "__main__":
    main()
