#!/usr/bin/env python3
"""0.12.0 12-G1b — 서버의 채팅 틀(엔진/채팅틀.gl — GGUF 의 chat_template 을 jinja 해석기로)을 GPU 없이(엔진/시험_채팅틀.gl) 그린 글이 jinja2
참조(시험/도구틀참조.py — 서버와 같은 메시지 다듬기)와 글자까지 같은가. 모델마다 채팅 경우(시험/채팅틀경우.json)와 도구 경우(시험/도구틀경우.json)를,
틀이 enable_thinking 을 쓰면 --생각끔 으로도. jinja2 가 있는 곳(WSL Ubuntu)에서:

  python3 시험/채팅틀대조.py <시험_채팅틀 실행 파일> <모델.gguf> … [--임시 <윈도에서도 보이는 폴더 — WSL 에서 .exe 를 부를 때>]

같음 = 두 글이 바이트까지 같거나 둘 다 오류(서버는 400). 끝 줄 "채팅틀대조: 모두 같음"이면 종료 0.
"""
import json
import os
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import 도구틀참조 as 참조  # noqa: E402


def 윈도경로(p, exe):
    if os.name == "posix" and exe.lower().endswith(".exe"):
        return subprocess.run(["wslpath", "-m", p], capture_output=True, text=True).stdout.strip()
    return p


def main():
    args = sys.argv[1:]
    임시 = None
    if "--임시" in args:
        i = args.index("--임시")
        임시 = args[i + 1]
        args = args[:i] + args[i + 2:]
    exe, ggufs = args[0], args[1:]
    다른것 = 0
    합 = 0
    for gguf in ggufs:
        이름 = os.path.basename(gguf)
        틀글 = 참조.gguf_meta(gguf, {"tokenizer.chat_template"}).get("tokenizer.chat_template") or ""
        갈래들 = [False] + ([True] if "enable_thinking" in 틀글 else [])
        for 경우파일 in ("채팅틀경우.json", "도구틀경우.json"):
            경우 = json.load(open(os.path.join(HERE, 경우파일), encoding="utf-8"))
            for 생각끔 in 갈래들:
                기대, 꼴, 시스템받음 = 참조.참조들(gguf, 경우, 생각끔)
                with tempfile.TemporaryDirectory(dir=임시) as d:
                    cp, op = os.path.join(d, "c.json"), os.path.join(d, "o.json")
                    json.dump(경우, open(cp, "w", encoding="utf-8"), ensure_ascii=False)
                    cmd = [exe, 윈도경로(gguf, exe), 윈도경로(cp, exe), 윈도경로(op, exe)] + (["--생각끔"] if 생각끔 else [])
                    r = subprocess.run(cmd, capture_output=True)
                    알림 = (r.stdout + r.stderr).decode("utf-8", "replace").strip()
                    if r.returncode != 0:
                        print(f"  실행 실패 {이름} {경우파일}: {알림[-300:]}")
                        다른것 += 1
                        continue
                    얻음 = json.load(open(op, encoding="utf-8"))
                같음 = 0
                for i, e in 기대.items():
                    g = 얻음.get(i)
                    if (isinstance(e, dict) and isinstance(g, dict)) or e == g:
                        같음 += 1
                        continue
                    다른것 += 1
                    if isinstance(e, str) and isinstance(g, str):
                        k = next((j for j in range(min(len(e), len(g))) if e[j] != g[j]), min(len(e), len(g)))
                        print(f"  다름 {이름} {i}{' 생각끔' if 생각끔 else ''} — 글자 {k}: 기대 {e[max(0, k - 50):k + 50]!r}\n{' ' * 10}얻음 {g[max(0, k - 50):k + 50]!r}")
                    else:
                        print(f"  다름 {이름} {i}{' 생각끔' if 생각끔 else ''} — 기대 {str(e)[:160]!r}\n{' ' * 10}얻음 {str(g)[:160]!r}")
                합 += len(기대)
                print(f"{'같음' if 같음 == len(기대) else '다름'} {이름} {경우파일}{' --생각끔' if 생각끔 else ''}: {같음}/{len(기대)} · 도구 꼴 {꼴} · "
                      f"system {'그림' if 시스템받음 else '첫 user 말에 붙임'} · 도구 {알림.splitlines()[-1] if 알림 else ''}")
    print(f"경우 {합} · 다름 {다른것}")
    print("채팅틀대조: 모두 같음" if 다른것 == 0 else f"채팅틀대조: 다름 {다른것}")
    sys.exit(0 if 다른것 == 0 else 1)


if __name__ == "__main__":
    main()
