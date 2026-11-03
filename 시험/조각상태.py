#!/usr/bin/env python3
"""층 나누기(엔진/조각.gl — 25단계)의 상태 목록 점검. 조각(장치)을 바꿀 때 엔진의 전역을 조각상태(묶음)에 담고 다음 조각의 것을 푼다 —
전역을 더하면서 그 목록에 넣지 않으면 여러 조각으로 나눈 실행(실행기의 --나누기 · 7B 시험 · 측정 — 서버는 나누지 않는다)에서만 조용히
틀린다(외부 리뷰의 지적). 이 점검은 엔진 전역을 하나 더할 때마다 둘 가운데 하나를 고르게 한다.

    python -I -X utf8 시험/조각상태.py check        (빠른 점검 · CI)

본다:
1. 조각상태의 칸마다 조각담기 가 전역을 읽어 `s→칸` 에 담고, 조각풀기 가 `s→칸` 에서 전역에 쓴다(수 · 참거짓 · 배열 · 계획의 꼴 모두).
2. 칸마다 엔진의 전역(아래 FILES)으로 있다 — 낡은 칸이 없다.
3. FILES 의 전역마다 조각상태의 칸이거나 시험/조각상태.txt 에 적혀 있다 — 새 전역은 조각.gl 의 조각상태 · 조각담기 · 조각풀기에 넣거나
   (조각마다 값이 다름 — 장치 주소 · 커널 손잡이 · 버퍼), 조각상태.txt 의 [공유] 에 적는다(모든 조각이 같음 — 모델 모양 · 설정 · 호스트 표).
4. 조각상태.txt 의 이름마다 아직 전역이다 — 낡은 줄이 없다.
조각상태.txt 의 [확인 필요] 는 2026-11-03 에 장치 자원처럼 보이나 조각상태에 없던 것들이다 — 층 나누기가 그 길(k-양자화 무리 · 표 판 ·
메가 커널 · 층 렌즈 · Qwen3.5)을 쓰면 틀릴 수 있다. 점검을 막지는 않고 수를 알린다.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["엔진.gl", "쿠다층.gl", "가중치.gl", "엔진K.gl", "가중치K.gl"]
선언 = re.compile(r"(?:^|(?<=\.)\s+)(?:정수|실수|참거짓|문자열|정수목록|문자열목록|바이트열|공허|중간정수|[가-힣A-Za-z0-9_]+)(?: 참조)?(?:\[[^\]]*\])? "
                r"([가-힣A-Za-z_][가-힣A-Za-z0-9_]*)\s*(?==|\.)")
건너뜀 = ("[", "묶음", "외부", "포함", "(*", " ", "\t", "반환", "}")


def 전역들():
    """FILES 의 맨 바깥 선언 — 이름 → 파일"""
    out = {}
    for f in FILES:
        for l in open(os.path.join(ROOT, "엔진", f), encoding="utf-8"):
            if l.startswith(건너뜀) or not l.strip():
                continue
            for m in 선언.finditer(l.split("(*")[0]):
                out.setdefault(m.group(1), f)
    return out


def 함수몸(글, 이름):
    """[… 이름]는 { … } 의 몸(줄 목록) — 맨 바깥 } 까지"""
    줄들 = 글.splitlines()
    for i, l in enumerate(줄들):
        if l.startswith("[") and f" {이름}]" in l:
            몸 = []
            for l2 in 줄들[i + 1:]:
                if l2.startswith("}"):
                    return 몸
                몸.append(l2)
    return None


def 목록읽기():
    """조각상태.txt — [공유] · [확인 필요] 의 이름들"""
    칸 = {}
    지금 = None
    for l in open(os.path.join(ROOT, "시험", "조각상태.txt"), encoding="utf-8"):
        l = l.split("#")[0].strip()
        if not l:
            continue
        if l.startswith("[") and l.endswith("]"):
            지금 = l[1:-1]
            칸[지금] = []
        elif 지금:
            칸[지금] += l.split()
    return 칸


def check():
    sys.stdout.reconfigure(encoding="utf-8")
    글 = open(os.path.join(ROOT, "엔진", "조각.gl"), encoding="utf-8").read()
    줄 = next(l for l in 글.splitlines() if l.startswith("묶음 조각상태는 "))
    칸들 = [x.strip().split()[-1] for x in 줄[len("묶음 조각상태는 "):].rstrip(".").split(",")]
    담기 = "\n".join(함수몸(글, "조각담기") or [])
    풀기 = "\n".join(함수몸(글, "조각풀기") or [])
    g = 전역들()
    목록 = 목록읽기()
    공유 = set(목록.get("공유", [])) | set(목록.get("확인 필요", []))
    틀림 = []
    for c in 칸들:
        e = re.escape(c)
        칸씀 = rf"s→{e}(?![\w])"                                  # 조각상태의 칸
        if not (re.search(칸씀, 담기) and re.search(rf"(?<![\w→]){e}", 담기)):          # 전역을 읽어 칸에(수 · 참거짓 · 배열)
            틀림.append(f"조각담기 에 '{c}' 가 없다")
        if not (re.search(칸씀, 풀기) and re.search(rf"(?<![\w→]){e}(\[[^\]]*\])?\s*=(?!=)", 풀기)):   # 칸에서 전역으로
            틀림.append(f"조각풀기 에 '{c}' 가 없다")
        if c not in g:
            틀림.append(f"조각상태의 칸 '{c}' 가 엔진의 전역이 아니다(낡은 칸)")
    for n, f in sorted(g.items()):
        if n not in 칸들 and n not in 공유:
            틀림.append(f"새 전역 '{n}'({f}) — 조각마다 다르면 조각.gl 의 조각상태 · 조각담기 · 조각풀기에, 모든 조각이 같으면 시험/조각상태.txt 의 [공유] 에")
    for n in sorted(공유):
        if n not in g:
            틀림.append(f"시험/조각상태.txt 의 '{n}' 가 이제 전역이 아니다(낡은 줄)")
        if n in 칸들:
            틀림.append(f"'{n}' 가 조각상태에도 시험/조각상태.txt 에도 있다")
    for t in 틀림[:20]:
        print("틀림 —", t)
    확인 = len(목록.get("확인 필요", []))
    if 틀림:
        print(f"층 나누기의 상태 목록: 못 미침 {len(틀림)}")
        return 1
    print(f"층 나누기의 상태 목록: 통과 — 엔진 전역 {len(g)} 개 = 조각상태 {len(칸들)} + 공유 {len(목록.get('공유', []))} + 확인 필요 {확인}"
          f"(장치 자원처럼 보이나 담지 않음 — 시험/조각상태.txt)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "check":
        print(__doc__)
        sys.exit(2)
    sys.exit(check())
