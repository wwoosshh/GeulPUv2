#!/usr/bin/env python3
"""데이터센터 GPU 의 커널 프로파일(엔진/재기_걸음) — 입력 만들기 · 정리 · 두 기계 견주기.

  python -I -X utf8 재기/판시간.py 입력 <출력.bin> [--길이 32768] [--씨앗 1]
      재기_걸음 의 입력(재기_7B 꼴 — 채팅 0 개 · 긴 글): 보통 토큰(1,000 … 29,999 — 어느 어휘에서도 특수 토큰이 아니다)을 정한 씨앗으로.
      dense · Qwen3.5 의 시간은 토큰의 값에 따라 갈리지 않는다(MoE 는 라우팅이 값에 따라 갈린다).
  python -I -X utf8 재기/판시간.py 정리 <재기_걸음 출력.txt> … [--대역 GB/s] [--정수최고 TOPS] [--정밀]
      잰 것(생성 걸음 · 프롬프트)마다 판을 묶음(행렬곱 · 어텐션 · GDN · 정규화 · RoPE · 그 밖)으로 모아 몫을 적고, 바닥과 견준다:
      - 생성 걸음(행 1): (행렬 + 출력 헤드) 바이트 ÷ 대역 — 가중치를 한 번 읽는 시간. 형식마다(판 이름 끝 _Q4_K · _Q6_K …) 그 형식의
        바이트 ÷ 그 판들의 시간 = 이룬 대역.
      - 프롬프트: 행렬곱의 정수 셈 2 · P · 원소(--정밀 이면 활성값의 자리가 둘이라 두 배) ÷ 정수 최고치.
      대역 · 최고치는 그 GPU 에서 잰 값을 준다(엔진/진단 — 메모리 읽기 대역 · 정수 텐서 최고치). 주지 않으면 그 칸을 비운다.
  python -I -X utf8 재기/판시간.py 견주기 <가.txt> <나.txt>
      같은 모델 · 모듈을 두 기계에서 잰 출력 — 같은 잰 것마다 판의 ms 와 비(나 ÷ 가).
  python -I -X utf8 재기/판시간.py 요약 <출력.txt> … --대역 GB/s --정수최고 TOPS
      모델마다 한 줄 — 생성 걸음(행 1)의 시간 · 행렬곱 ÷ 대역 바닥 · 형식마다 이룬 대역 · 깊은 생성의 어텐션 몫, 가장 긴 프롬프트의
      시간 · 행렬곱 ÷ 정수 바닥 · 어텐션 몫. 정밀 모듈(이름에 _빠른 이 없다)은 정수 바닥을 두 배로.
"""
import random
import re
import sys

# 판 이름 → 묶음(앞의 것이 먼저). 이름은 글(geul) 커널 함수의 이름 — 재기_걸음 이 cuFuncGetName 을 풀어 적는다
묶음규칙 = [
    ("GDN", re.compile(r"GDN")),
    ("어텐션", re.compile(r"어텐션|접기|조각묶음|게이트256")),
    ("행렬곱", re.compile(r"줄하나|타일")),
    ("정규화", re.compile(r"RMS|정규화")),
    ("RoPE · KV", re.compile(r"RoPE")),
]
# 판 이름 끝의 형식 → GGUF 형식 번호
형식번호 = {"Q8_0": 8, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14, "IQ4_XS": 23}


def 묶음(이름):
    for 묶, 규칙 in 묶음규칙:
        if 규칙.search(이름):
            return 묶
    return "그 밖"


def 판형식(이름):
    m = re.search(r"_(Q8_0|Q4_K|Q5_K|Q6_K|IQ4_XS)$", 이름)
    return 형식번호[m.group(1)] if m else None


def 입력(argv):
    출력 = argv[0]
    길이 = int(argv[argv.index("--길이") + 1]) if "--길이" in argv else 32768
    씨앗 = int(argv[argv.index("--씨앗") + 1]) if "--씨앗" in argv else 1
    r = random.Random(씨앗)
    with open(출력, "wb") as f:
        f.write((0).to_bytes(8, "little", signed=True))
        f.write(길이.to_bytes(8, "little", signed=True))
        for _ in range(길이):
            f.write(r.randrange(1000, 30000).to_bytes(8, "little", signed=True))
    print(f"재기 입력: 채팅 0 · 긴 글 {길이} (씨앗 {씨앗}) → {출력}")


머리줄 = re.compile(r"^(깊이 (\d+) 행 (\d+)|프롬프트 (\d+)) ")
판줄 = re.compile(r"^\s+([\d.]+) ms \(\s*(\d+)\) (.+)$")


def 읽기(경로):
    """재기_걸음 의 출력 — {구조, 행렬바이트, 행렬원소, 출력바이트, 출력형식, 형식바이트, 잰것: [{종류, 깊이|P, 행, 실제, 그래프, 합, 판}]}"""
    d = {"경로": 경로, "구조": "?", "모듈": "?", "행렬바이트": 0, "행렬원소": 0, "출력바이트": 0, "출력형식": -1, "형식바이트": {},
         "잰것": [], "오류칸": None}
    지금 = None
    for 줄 in open(경로, encoding="utf-8", errors="replace"):
        줄 = 줄.rstrip("\n")
        m = re.match(r"^구조 (\S+) · 표 판 (\S+) · 모듈 (.+)$", 줄)
        if m:
            d["구조"], d["표판"], d["모듈"] = m.group(1), m.group(2), m.group(3)
            continue
        m = re.match(r"^가중치 행렬 (\d+) 바이트 · 원소 (\d+) · 출력 헤드 (\d+) 바이트\(형식 (-?\d+)", 줄)
        if m:
            d["행렬바이트"], d["행렬원소"], d["출력바이트"], d["출력형식"] = (int(m.group(i)) for i in range(1, 5))
            continue
        m = re.match(r"^가중치 행렬 형식 (\d+) (\d+) 바이트", 줄)
        if m:
            d["형식바이트"][int(m.group(1))] = int(m.group(2))
            continue
        m = re.match(r"^오류칸 (\d+)", 줄)
        if m:
            d["오류칸"] = int(m.group(1))
            continue
        m = 머리줄.match(줄)
        if m:
            def 값(이름):
                mm = re.search(이름 + r" ms ([\d.]+)", 줄)
                return float(mm.group(1)) if mm else None
            if m.group(2):
                지금 = {"종류": "생성", "깊이": int(m.group(2)), "행": int(m.group(3))}
            else:
                지금 = {"종류": "프롬프트", "P": int(m.group(4)), "행": int(m.group(4))}
            지금.update({"실제": 값("실제"), "그래프": 값("그래프"), "합": 값("띄움 합"), "판": []})
            d["잰것"].append(지금)
            continue
        m = 판줄.match(줄)
        if m and 지금 is not None:
            지금["판"].append((float(m.group(1)), int(m.group(2)), m.group(3).strip()))
    return d


def 잰것이름(e):
    return f"생성 깊이 {e['깊이']} 행 {e['행']}" if e["종류"] == "생성" else f"프롬프트 {e['P']}"


def 칸(v, 꼴="{:.2f}"):
    return "" if v is None else 꼴.format(v)


def 정리(argv):
    경로들 = []
    i = 0
    while i < len(argv):
        if argv[i] in ("--대역", "--정수최고"):
            i += 2
            continue
        if not argv[i].startswith("--"):
            경로들.append(argv[i])
        i += 1
    대역 = float(argv[argv.index("--대역") + 1]) if "--대역" in argv else None          # GB/s
    최고 = float(argv[argv.index("--정수최고") + 1]) if "--정수최고" in argv else None    # TOPS(정수 8 비트 · 밀집)
    정밀 = "--정밀" in argv
    for 경로 in 경로들:
        d = 읽기(경로)
        print(f"## {경로}\n\n구조 {d['구조']} · 모듈 {d['모듈']} · 행렬 {d['행렬바이트'] / 1e9:.3f} GB · 원소 {d['행렬원소'] / 1e9:.3f} G"
              f" · 출력 헤드 {d['출력바이트'] / 1e9:.3f} GB · 오류칸 {d['오류칸']}\n")
        print("| 잰 것 | 실제 ms | 그래프 ms | 띄움 합 ms | 묶음마다 ms (몫) | 바닥 ms | 행렬곱 ÷ 바닥 |")
        print("|---|---|---|---|---|---|---|")
        for e in d["잰것"]:
            묶음합 = {}
            for ms, _, 이름 in e["판"]:
                묶음합[묶음(이름)] = 묶음합.get(묶음(이름), 0.0) + ms
            합 = e["합"] or sum(묶음합.values()) or 1.0
            묶음글 = " · ".join(f"{k} {v:.2f} ({v / 합:.0%})" for k, v in sorted(묶음합.items(), key=lambda kv: -kv[1]))
            바닥 = None
            if e["종류"] == "생성" and 대역 and e["행"] == 1 and d["행렬바이트"] > 0:
                바닥 = (d["행렬바이트"] + d["출력바이트"]) / (대역 * 1e9) * 1e3
            if e["종류"] == "프롬프트" and 최고 and d["행렬원소"] > 0:
                바닥 = 2 * e["P"] * d["행렬원소"] * (2 if 정밀 else 1) / (최고 * 1e12) * 1e3
            비 = None if not 바닥 else 묶음합.get("행렬곱", 0.0) / 바닥
            print(f"| {잰것이름(e)} | {칸(e['실제'])} | {칸(e['그래프'])} | {칸(e['합'])} | {묶음글} | {칸(바닥)} | {칸(비)} |")
        print()
        # 생성 걸음(행 1) — 형식마다 이룬 대역
        for e in d["잰것"]:
            if e["종류"] != "생성" or e["행"] != 1:
                continue
            형식ms = {}
            for ms, _, 이름 in e["판"]:
                t = 판형식(이름)
                if t is not None and 묶음(이름) == "행렬곱":
                    형식ms[t] = 형식ms.get(t, 0.0) + ms
            줄들 = []
            for t, ms in sorted(형식ms.items()):
                바이트 = d["형식바이트"].get(t, 0) + (d["출력바이트"] if d["출력형식"] == t else 0)
                gbs = 바이트 / (ms / 1e3) / 1e9 if ms > 0 else 0.0
                이름 = next(k for k, v in 형식번호.items() if v == t)
                줄들.append(f"{이름} {바이트 / 1e9:.3f} GB ÷ {ms:.3f} ms = {gbs:.0f} GB/s" + (f" ({gbs / 대역:.0%})" if 대역 else ""))
            if 줄들:
                print(f"- {잰것이름(e)} — 형식마다 이룬 대역: " + " · ".join(줄들))
        print()
        for e in d["잰것"]:
            print(f"### {잰것이름(e)} — 판(시간이 큰 차례, 앞 12)\n")
            for ms, 수, 판 in e["판"][:12]:
                print(f"- {ms:.3f} ms ({수}) {판} — {묶음(판)}")
            print()


def 견주기(argv):
    가, 나 = 읽기(argv[0]), 읽기(argv[1])
    print(f"## {가['경로']} ↔ {나['경로']}\n")
    for e in 가["잰것"]:
        짝 = next((f for f in 나["잰것"] if f["종류"] == e["종류"] and f["행"] == e["행"] and f.get("깊이") == e.get("깊이")
                  and f.get("P") == e.get("P")), None)
        if 짝 is None:
            continue
        print(f"### {잰것이름(e)}: 띄움 합 {칸(e['합'])} → {칸(짝['합'])} ms"
              + (f" (비 {짝['합'] / e['합']:.2f})" if e["합"] and 짝["합"] else "") + "\n")
        print("| 판 | 가 ms | 나 ms | 나 ÷ 가 |")
        print("|---|---|---|---|")
        나판 = {이름: ms for ms, _, 이름 in 짝["판"]}
        for ms, _, 이름 in e["판"]:
            if 이름 in 나판:
                print(f"| {이름} | {ms:.3f} | {나판[이름]:.3f} | {나판[이름] / ms:.2f} |" if ms > 0 else f"| {이름} | {ms:.3f} | {나판[이름]:.3f} | |")
        print()


def 묶음합(e):
    합 = {}
    for ms, _, 이름 in e["판"]:
        합[묶음(이름)] = 합.get(묶음(이름), 0.0) + ms
    return 합


def 형식대역(d, e):
    """생성 걸음(행 1)의 형식마다 (이름, GB/s)"""
    형식ms = {}
    for ms, _, 이름 in e["판"]:
        t = 판형식(이름)
        if t is not None and 묶음(이름) == "행렬곱":
            형식ms[t] = 형식ms.get(t, 0.0) + ms
    out = []
    for t, ms in sorted(형식ms.items()):
        바이트 = d["형식바이트"].get(t, 0) + (d["출력바이트"] if d["출력형식"] == t else 0)
        out.append((next(k for k, v in 형식번호.items() if v == t), 바이트 / (ms / 1e3) / 1e9 if ms > 0 else 0.0))
    return out


def 요약(argv):
    대역 = float(argv[argv.index("--대역") + 1])
    최고 = float(argv[argv.index("--정수최고") + 1])
    경로들 = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] not in ("--대역", "--정수최고"))]
    print(f"대역 {대역:.0f} GB/s · 정수 최고 {최고:.0f} TOPS 기준\n")
    print("| 모델 | 구조 · 모드 | 생성 ms (깊이 1024) | 행렬곱 ÷ 대역 바닥 | 형식마다 이룬 대역 | 생성 ms (가장 깊은) · 어텐션 몫"
          " | 프롬프트 ms (가장 긴) | 행렬곱 ÷ 정수 바닥 | 어텐션 몫 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for 경로 in 경로들:
        d = 읽기(경로)
        정밀 = "_빠른" not in d["모듈"]
        이름 = re.sub(r"^(pc_|.*/)", "", 경로).replace(".txt", "")
        생성 = sorted((e for e in d["잰것"] if e["종류"] == "생성" and e["행"] == 1), key=lambda e: e["깊이"])
        프롬 = sorted((e for e in d["잰것"] if e["종류"] == "프롬프트"), key=lambda e: e["P"])
        칸들 = [이름, f"{d['구조']} · {'정밀' if 정밀 else '빠른'}"]
        if 생성:
            e = 생성[0]
            시간 = e["실제"] or e["그래프"]
            바닥 = (d["행렬바이트"] + d["출력바이트"]) / (대역 * 1e9) * 1e3
            칸들 += [f"{시간:.2f}" + (f" (그래프 {e['그래프']:.2f})" if e["실제"] and e["그래프"] else ""),
                    f"{묶음합(e).get('행렬곱', 0.0) / 바닥:.2f}" if 바닥 else "",
                    " · ".join(f"{n} {v:.0f}" for n, v in 형식대역(d, e))]
            깊은 = 생성[-1]
            합 = 깊은["합"] or 1.0
            칸들.append(f"{(깊은['실제'] or 깊은['그래프']):.2f} (깊이 {깊은['깊이']}) · {묶음합(깊은).get('어텐션', 0.0) / 합:.0%}")
        else:
            칸들 += ["", "", "", ""]
        if 프롬:
            e = 프롬[-1]
            바닥 = 2 * e["P"] * d["행렬원소"] * (2 if 정밀 else 1) / (최고 * 1e12) * 1e3
            합 = e["합"] or 1.0
            칸들 += [f"{e['실제']:.1f} (P {e['P']})", f"{묶음합(e).get('행렬곱', 0.0) / 바닥:.2f}" if 바닥 else "",
                    f"{묶음합(e).get('어텐션', 0.0) / 합:.0%}"]
        else:
            칸들 += ["", "", ""]
        print("| " + " | ".join(칸들) + " |")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("입력", "정리", "견주기", "요약"):
        print(__doc__)
        return 2
    {"입력": 입력, "정리": 정리, "견주기": 견주기, "요약": 요약}[sys.argv[1]](sys.argv[2:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
