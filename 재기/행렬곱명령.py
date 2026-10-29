#!/usr/bin/env python3
"""프롬프트 행렬곱의 명령 구성(문서/프롬프트행렬곱.md 의 P0) — Nsight Compute 보고서(.ncu-repz — 엔진/재기_프로파일 이 고른 층의 판들)에서
텐서곱을 쓰는 판마다 텐서곱 하나에 붙는 워프 명령을 갈래(끝셈 · 반실수 · 텐서곱 · 공유 읽기 · 복사 · 정수 · MOV · 흐름 · 그 밖)로 나누고,
텐서 바쁨 · 명령 내기 · 점유 · 레지스터 · 멈춤 까닭의 첫째 셋과 SM · 클럭당 연산(m16n8k32 텐서곱 = 8,192 연산, m16n8k16 = 4,096)을 적는다.

  python -I -X utf8 재기/행렬곱명령.py <ncu.exe> <보고서.ncu-repz> …

ncu 는 설치 경로에 한글이 있으면 뜨지 않는다 — 영문 경로의 사본을 준다. 계약의 끝셈만으로 정해지는 천장은 8,192 / c (c = 칸 · 블록마다
끝셈 명령 — 빠른 Q8_0 2, 빠른 Q4_K · Q5_K 3)다.
"""
import csv
import io
import re
import subprocess
import sys

멈춤들 = ["math_pipe_throttle", "wait", "short_scoreboard", "long_scoreboard", "barrier", "not_selected", "selected", "mio_throttle",
         "dispatch_stall", "no_instruction", "lg_throttle", "branch_resolving", "drain", "membar", "sleeping", "tex_throttle", "imc_miss",
         "misc"]
지표 = ["gpu__time_duration.sum", "sm__cycles_elapsed.avg", "smsp__inst_executed.sum", "sass__inst_executed_per_opcode",
        "sass__inst_executed_per_opcode_with_modifier_all", "sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active", "sm__warps_active.avg.pct_of_peak_sustained_active",
        "launch__registers_per_thread", "launch__shared_mem_per_block", "device__attribute_multiprocessor_count"] + \
       [f"smsp__average_warps_issue_stalled_{m}_per_issue_active.ratio" for m in 멈춤들]

갈래표 = {
    "끝셈": {"FADD", "FFMA", "FMUL"},
    "반실수": {"HADD2", "HFMA2", "HMUL2"},
    "텐서곱": {"IMMA", "HMMA"},
    "공유 읽기": {"LDS", "LDSM"},
    "복사": {"LDGSTS", "LDGDEPBAR", "DEPBAR", "LDG", "STG", "STS"},
    "정수": {"IMAD", "IADD3", "SHF", "ISETP", "LOP3", "LEA", "SEL", "PRMT", "IABS", "IMNMX", "POPC", "FLO", "BMSK", "UIADD3", "UIMAD",
             "ULDC", "UMOV", "ULOP3", "USHF", "USEL", "S2UR", "S2R", "CS2R", "P2R", "R2P", "PLOP3", "I2F", "F2I", "F2F", "I2I",
             "VIMNMX", "UISETP", "ULEA", "IDP", "IMUL"},
    "MOV": {"MOV"},
    "흐름": {"BRA", "BSSY", "BSYNC", "BAR", "CALL", "RET", "EXIT", "WARPSYNC", "NOP", "YIELD", "BREAK"},
}
갈래들 = list(갈래표) + ["그 밖"]


def 갈래(op):
    for g, s in 갈래표.items():
        if op in s:
            return g
    return "그 밖"


def 수(글):
    return float(글.split(" ")[0].replace(",", "")) if 글 else 0.0


def 펼침(글):
    """'109,221,120 (FADD: 25,362,432; …)' → {FADD: 25362432, …}"""
    m = re.search(r"\((.*)\)", 글 or "")
    if not m:
        return {}
    d = {}
    for 쪽 in m.group(1).split(";"):
        if ":" in 쪽:
            k, v = 쪽.rsplit(":", 1)
            d[k.strip()] = float(v.strip().replace(",", ""))
    return d


def 이름풀기(s):
    if s.startswith("_G"):
        try:
            return bytes.fromhex(s[2:]).decode("utf-8")
        except ValueError:
            return s
    return s


def 읽기(ncu, 보고서):
    r = subprocess.run([ncu, "-i", 보고서, "--page", "raw", "--csv", "--print-metric-instances", "details", "--metrics", ",".join(지표)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    줄들 = list(csv.reader(io.StringIO(r.stdout)))
    if len(줄들) < 3:
        raise SystemExit(f"{보고서}: 표가 비었다 — {r.stderr[-300:]}")
    머리 = 줄들[0]
    판들 = []
    for 줄 in 줄들[2:]:
        v = dict(zip(머리, 줄))
        ops = 펼침(v.get("sass__inst_executed_per_opcode"))
        if not ops.get("IMMA") and not ops.get("HMMA"):
            continue
        꼴 = 펼침(v.get("sass__inst_executed_per_opcode_with_modifier_all"))
        k32 = sum(n for k, n in 꼴.items() if k.startswith("IMMA.16832"))
        k16 = sum(n for k, n in 꼴.items() if k.startswith("IMMA.16816"))
        텐서곱 = k32 + k16 or ops.get("IMMA", 0.0)
        연산 = 8192 * k32 + 4096 * k16
        명령 = 수(v.get("smsp__inst_executed.sum"))
        g = {x: 0.0 for x in 갈래들}
        for op, n in ops.items():
            g[갈래(op)] += n
        멈춤 = sorted(((m, 수(v.get(f"smsp__average_warps_issue_stalled_{m}_per_issue_active.ratio"))) for m in 멈춤들), key=lambda t: -t[1])
        sm = 수(v.get("device__attribute_multiprocessor_count")) or 60
        판들.append({
            "이름": 이름풀기(v.get("Kernel Name", "?")), "격자": v.get("Grid Size", ""), "µs": 수(v.get("gpu__time_duration.sum")) * 1000,
            "텐서곱": 텐서곱, "k16": k16, "명령/텐서곱": 명령 / 텐서곱, "갈래": {x: g[x] / 텐서곱 for x in 갈래들},
            "텐서 바쁨": 수(v.get("sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_active")),
            "명령 내기": 수(v.get("smsp__issue_active.avg.pct_of_peak_sustained_active")),
            "점유": 수(v.get("sm__warps_active.avg.pct_of_peak_sustained_active")),
            "레지스터": 수(v.get("launch__registers_per_thread")), "공유": 수(v.get("launch__shared_mem_per_block")),
            "SM클럭당": 연산 / (sm * 수(v.get("sm__cycles_elapsed.avg"))) if 수(v.get("sm__cycles_elapsed.avg")) else 0.0,
            "멈춤": 멈춤[:3],
        })
    return 판들


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    ncu = sys.argv[1]
    for 보고서 in sys.argv[2:]:
        판들 = 읽기(ncu, 보고서)
        print(f"## {보고서}\n")
        print("| 판 | 격자 | µs | 명령/텐서곱 | " + " | ".join(갈래들) + " | 텐서 바쁨 | 명령 내기 | 점유 | 레지스터 | SM·클럭당 연산 | 멈춤(명령당) |")
        print("|" + "---|" * (12 + len(갈래들)))
        for p in 판들:
            갈 = " | ".join(f"{p['갈래'][x]:.2f}" for x in 갈래들)
            멈 = " · ".join(f"{m} {r:.2f}" for m, r in p["멈춤"])
            꼴 = f" (k16 {p['k16'] / p['텐서곱']:.0%})" if p["k16"] else ""
            print(f"| {p['이름']}{꼴} | {p['격자']} | {p['µs']:.1f} | {p['명령/텐서곱']:.1f} | {갈} | {p['텐서 바쁨']:.0f} % | {p['명령 내기']:.0f} % |"
                  f" {p['점유']:.0f} % | {p['레지스터']:.0f} | {p['SM클럭당']:.0f} | {멈} |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
