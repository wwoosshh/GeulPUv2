#!/usr/bin/env python3
"""바뀐 것의 조각 합치기 — PR 마다 문서/바뀐것.md · CHANGELOG.md 를 고치지 않고 조각 파일 하나(문서/바뀐것.d/<가지>.md)를 더한 것을,
릴리스 때 두 문서의 '다음 판' · 'Unreleased' 절에 모아 넣는다(여러 가지가 같은 줄을 고쳐 부딪히지 않게 — 문서/개발.md 의 '여러 세션으로
개발하기').

  python -I -X utf8 시험/바뀐것합치기.py            # 무엇이 들어갈지 보이기만
  python -I -X utf8 시험/바뀐것합치기.py --쓰기     # 두 문서에 넣고 조각을 지운다(릴리스 가지에서)

조각의 꼴(문서/바뀐것.d/_꼴.md — 밑줄로 시작하는 파일은 조각이 아니다):

  갈래: 더함                    (더함 · 바꿈 · 고침 — CHANGELOG 의 Added · Changed · Fixed)

  - **한국어 제목**: 내용 …     (문서/바뀐것.md 의 '## 다음 판' 끝에)

  ---

  - **English title.** Text …  (CHANGELOG.md 의 '## Unreleased' 아래 '### Added' 등의 끝에)

사용자에게 보이지 않는 변경(시험 · 내부 구조 · 문서만)은 조각을 쓰지 않는다.
"""
import argparse
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
조각곳 = os.path.join(ROOT, "문서", "바뀐것.d")
갈래들 = {"더함": "Added", "바꿈": "Changed", "고침": "Fixed"}


def 읽기(경로):
    with open(경로, "rb") as f:
        return f.read().decode("utf-8")


def 쓰기(경로, 글):
    with open(경로, "wb") as f:
        f.write(글.encode("utf-8"))


def 조각들():
    if not os.path.isdir(조각곳):
        return []
    나온 = []
    for 이름 in sorted(os.listdir(조각곳)):
        if not 이름.endswith(".md") or 이름.startswith("_"):
            continue
        글 = 읽기(os.path.join(조각곳, 이름)).replace("\r\n", "\n")
        줄들 = 글.split("\n")
        첫 = next((i for i, x in enumerate(줄들) if x.strip()), None)
        if 첫 is None or not 줄들[첫].startswith("갈래:"):
            raise SystemExit(f"{이름}: 첫 줄이 '갈래: 더함|바꿈|고침' 이 아니다")
        갈래 = 줄들[첫].split(":", 1)[1].strip()
        if 갈래 not in 갈래들:
            raise SystemExit(f"{이름}: 갈래 '{갈래}' — 더함 · 바꿈 · 고침 가운데 하나")
        몸 = "\n".join(줄들[첫 + 1:])
        if "\n---\n" not in "\n" + 몸 + "\n":
            raise SystemExit(f"{이름}: 한국어와 영어를 가르는 '---' 줄이 없다")
        ko, en = ("\n" + 몸 + "\n").split("\n---\n", 1)
        ko, en = ko.strip("\n"), en.strip("\n")
        if not ko.strip() or not en.strip():
            raise SystemExit(f"{이름}: 한국어 · 영어 가운데 빈 쪽이 있다")
        나온.append((이름, 갈래, ko, en))
    return 나온


def 절끝(줄들, 시작):
    """줄들[시작] 의 '## ' 절이 끝나는 자리(다음 '## ' 머리 — 없으면 끝)"""
    for i in range(시작 + 1, len(줄들)):
        if 줄들[i].startswith("## "):
            return i
    return len(줄들)


def 끝빈줄빼고(줄들, 시작, 끝):
    while 끝 > 시작 + 1 and not 줄들[끝 - 1].strip():
        끝 -= 1
    return 끝


def 한국어넣기(글, 조각):
    줄들 = 글.split("\n")
    i = next((k for k, x in enumerate(줄들) if x.strip() == "## 다음 판"), None)
    if i is None:
        i = next(k for k, x in enumerate(줄들) if x.startswith("## "))
        줄들[i:i] = ["## 다음 판", ""]
    끝 = 끝빈줄빼고(줄들, i, 절끝(줄들, i))
    새 = []
    for _, _, ko, _ in 조각:
        새 += ko.split("\n")
    줄들[끝:끝] = ([""] if 끝 == i + 1 else []) + 새
    return "\n".join(줄들)


def 영어넣기(글, 조각):
    줄들 = 글.split("\n")
    i = next((k for k, x in enumerate(줄들) if x.strip() == "## Unreleased"), None)
    if i is None:
        i = next(k for k, x in enumerate(줄들) if x.startswith("## "))
        줄들[i:i] = ["## Unreleased", ""]
    for 갈래, 머리 in 갈래들.items():
        새 = []
        for _, g, _, en in 조각:
            if g == 갈래:
                새 += en.split("\n")
        if not 새:
            continue
        끝 = 절끝(줄들, i)
        j = next((k for k in range(i + 1, 끝) if 줄들[k].strip() == f"### {머리}"), None)
        if j is None:
            # 갈래의 차례(Added · Changed · Fixed)를 지켜 그 앞의 머리 다음에 — 뒤의 머리가 있으면 그 앞에
            뒤 = [f"### {h}" for h in list(갈래들.values())[list(갈래들.values()).index(머리) + 1:]]
            k = next((k for k in range(i + 1, 끝) if 줄들[k].strip() in 뒤), None)
            if k is None:
                k = 끝빈줄빼고(줄들, i, 끝)
                줄들[k:k] = ["", f"### {머리}", ""] + 새
            else:
                줄들[k:k] = [f"### {머리}", ""] + 새 + [""]
            continue
        다음 = next((k for k in range(j + 1, 끝) if 줄들[k].startswith("### ")), 끝)
        끝j = 끝빈줄빼고(줄들, j, 다음)
        줄들[끝j:끝j] = ([""] if 끝j == j + 1 else []) + 새
    return "\n".join(줄들)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--쓰기", action="store_true", help="문서/바뀐것.md · CHANGELOG.md 에 넣고 조각을 지운다")
    a = p.parse_args()
    조각 = 조각들()
    if not 조각:
        print("조각 없음 — 문서/바뀐것.d/ 가 비었다")
        return 0
    for 이름, 갈래, ko, en in 조각:
        print(f"{이름} ({갈래} · {갈래들[갈래]}): {ko.splitlines()[0][:80]} / {en.splitlines()[0][:80]}")
    if not a.쓰기:
        print(f"조각 {len(조각)} 개 — --쓰기 로 넣는다")
        return 0
    ko경로 = os.path.join(ROOT, "문서", "바뀐것.md")
    en경로 = os.path.join(ROOT, "CHANGELOG.md")
    쓰기(ko경로, 한국어넣기(읽기(ko경로), 조각))
    쓰기(en경로, 영어넣기(읽기(en경로), 조각))
    for 이름, _, _, _ in 조각:
        os.remove(os.path.join(조각곳, 이름))
    print(f"조각 {len(조각)} 개를 문서/바뀐것.md · CHANGELOG.md 에 넣고 지웠다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
