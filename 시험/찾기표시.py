#!/usr/bin/env python3
"""모델 찾기의 저장소 표시(앱/web/app.js 의 fdSupport)를 GPU · 인터넷 없이 본다(빠른 점검 2e — 시험/빠른점검.py 가 부른다, node 가 있을 때).

고정 자료 시험/찾기표시경우.json: 2026-10-30 조사(E:\\geulpu-work\\compat 의 보고_2026-10-30.md — Hugging Face 의 GGUF 저장소를 트렌딩 순으로
200 개, 메타데이터의 구조 · 매개변수 수 · 파일 이름과, 엔진의 모델 확인 · 분류로 낸 판정)와 엔진의 구조 목록(서버 --구조들 의 archs).
표시 규칙은 시험/찾기표시.js 가 app.js 에서 떼어 돌린다 — 조사가 받는다고 본 저장소는 ok · check, 받지 못하는 저장소는 어느 것도 ok 가
아니어야 한다. --엔진 <서버> 를 주면 그 서버의 --구조들 과 고정 자료의 구조 목록(이름 · 형식 갈래)이 같은지도 본다 — 다르면 고정 자료의
archs 를 엔진에 맞춰 고친다(구조가 늘거나 받는 형식이 바뀌면 표시도 바뀌므로).

  python -I -X utf8 시험/찾기표시.py [--엔진 <서버 실행 파일>]
마지막 줄 '찾기 표시: 모두 통과 (N 저장소)'(종료 0) 또는 '찾기 표시: 틀림 N'(종료 1). node 가 없으면 '찾기 표시: 건너뜀'(종료 0).
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "앱", "web", "app.js")
JS = os.path.join(ROOT, "시험", "찾기표시.js")
경우 = os.path.join(ROOT, "시험", "찾기표시경우.json")


def 구조들(서버):
    """서버 --구조들 의 JSON — 서버가 뜨지 못하면(NVIDIA 드라이버가 없는 CI 등) 없음"""
    try:
        r = subprocess.run([서버, "--구조들"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout.decode("utf-8"))
    except ValueError:
        return None


def main():
    서버 = None
    args = sys.argv[1:]
    if "--엔진" in args:
        서버 = args[args.index("--엔진") + 1]
    node = shutil.which("node")
    if not node:
        print("찾기 표시: 건너뜀 (node 없음)")
        return 0
    with open(경우, encoding="utf-8") as f:
        fx = json.load(f)
    틀림 = 0
    if 서버:
        엔진 = 구조들(서버)
        if 엔진 is None:
            print(f"구조 목록: 건너뜀(서버가 뜨지 못함 — {os.path.basename(서버)} --구조들)")
        else:
            mine = sorted((a["name"], " ".join(a.get("quants") or [])) for a in fx["archs"])
            theirs = sorted((a["name"], " ".join(a.get("quants") or [])) for a in 엔진["archs"])
            if mine != theirs:
                print(f"고정 자료의 구조 목록이 엔진과 다르다 — 찾기표시경우.json 의 archs 를 고쳐라\n  자료: {mine}\n  엔진: {theirs}")
                틀림 += 1
            else:
                print(f"구조 목록 = 엔진 ({len(theirs)} 구조)")
    r = subprocess.run([node, JS, APP, 경우], capture_output=True, timeout=120)
    sys.stdout.write(r.stdout.decode("utf-8", "replace"))
    if r.returncode != 0:
        sys.stdout.write(r.stderr.decode("utf-8", "replace"))
        틀림 += 1
    if 틀림:
        if r.returncode == 0:
            print(f"찾기 표시: 틀림 {틀림}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
