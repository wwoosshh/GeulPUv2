#!/usr/bin/env python3
"""모듈의 PTX → SASS(sm_89)를 WSL Ubuntu-24.04 의 ptxas(CUDA 13.0 — 넘침 값이 CI 의 12.9 와 같다)로 만들어, 커널마다 레지스터 · 넘침을 센다.
빠른점검.py 의 SASS() 와 같은 해석. 새 모듈 행의 시험/SASS넘침.txt 줄을 커밋 전에 얻으려고(윈도에는 ptxas 가 없다).

  python -X utf8 시험/SASS_WSL.py <작업 폴더(F: 의 ASCII 경로)> <이름=PTX 경로> … [--커널 <커널 이름> …]
출력: 모듈마다 커널 수 · 레지스터 최대 · 넘침 있는 커널, 그리고 SASS넘침.txt 꼴의 줄(모듈 | 커널 | 바이트).
GeulPUv2 4단계 — --커널: 모듈에서 그 커널(진입 함수)만 떼어 ptxas 에 준다(모듈 머리 + 함수 하나 — 64 MB 모듈의 몇 분 대신 몇 초).
커널마다 레지스터 · 넘침 저장 · 읽기도 찍는다. 생성기를 고치며 넘침을 빨리 보려고.
"""
import os
import re
import shutil
import subprocess
import sys


def 커널이름(기호):
    if 기호.startswith("_G"):
        try:
            return bytes.fromhex(기호[2:]).decode("utf-8")
        except ValueError:
            pass
    return 기호


def 떼기(원, 곳, 이름들):
    """PTX 모듈에서 머리(첫 진입 함수 앞)와 고른 커널의 진입 함수만 — 자체PTX 의 모듈은 .func 가 없고 진입 함수끼리 서로 부르지 않는다"""
    글 = open(원, encoding="ascii").read()
    조각 = re.split(r"(?m)^(?=\.visible \.entry )", 글)
    머리, 함수들 = 조각[0], 조각[1:]
    기호들 = {"_G" + n.encode("utf-8").hex() for n in 이름들}
    고른 = [f for f in 함수들 if re.match(r"\.visible \.entry (\S+)\(", f) and re.match(r"\.visible \.entry (\S+)\(", f).group(1) in 기호들]
    assert 고른, f"커널을 찾지 못했다: {이름들}"
    with open(곳, "w", encoding="ascii", newline="\n") as o:
        o.write(머리 + "".join(고른))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    작업 = os.path.abspath(sys.argv[1])
    assert all(ord(c) < 128 for c in 작업), "작업 폴더는 ASCII 경로여야 한다(ptxas)"
    줄들 = []
    인자 = sys.argv[2:]
    고른 = []
    if "--커널" in 인자:
        i = 인자.index("--커널")
        고른 = 인자[i + 1:]
        인자 = 인자[:i]
    for 항목 in 인자:
        이름, ptx = 항목.split("=", 1)
        d = os.path.join(작업, re.sub(r"[^A-Za-z0-9_]", "_", 이름.encode("utf-8").hex()[:40]))
        os.makedirs(d, exist_ok=True)
        if 고른:
            떼기(ptx, os.path.join(d, "m.ptx"), 고른)
        else:
            shutil.copyfile(ptx, os.path.join(d, "m.ptx"))
        드라이브, 나머지 = os.path.splitdrive(d)
        wsl경로 = "/mnt/" + 드라이브[0].lower() + 나머지.replace("\\", "/")
        r = subprocess.run(["wsl", "-d", "Ubuntu-24.04", "--exec", "bash", "-c",
                            f"cd '{wsl경로}' && /usr/local/cuda/bin/ptxas -arch=sm_89 -v m.ptx -o m.cubin"],
                           capture_output=True, env=dict(os.environ, MSYS_NO_PATHCONV="1"))
        글 = (r.stdout + r.stderr).decode("utf-8", "replace")
        커널들 = []
        for 줄 in 글.splitlines():
            m = re.search(r"Compiling entry function '([^']+)'", 줄)
            if m:
                커널들.append([커널이름(m.group(1)), 0, 0, 0])
                continue
            m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", 줄)
            if m and 커널들:
                커널들[-1][2], 커널들[-1][3] = int(m.group(1)), int(m.group(2))
            m = re.search(r"Used (\d+) registers", 줄)
            if m and 커널들:
                커널들[-1][1] = int(m.group(1))
        넘침 = {k[0]: k[2] + k[3] for k in 커널들 if k[2] or k[3]}
        print(f"{이름}: ptxas 종료 {r.returncode} · 커널 {len(커널들)} · 레지스터 최대 {max((k[1] for k in 커널들), default=0)} · 넘침 있는 커널 {len(넘침)}")
        if r.returncode != 0:
            print(글[-1500:])
        if 고른:
            for k in 커널들:
                print(f"  {k[0]}: 레지스터 {k[1]} · 넘침 저장 {k[2]} · 읽기 {k[3]} 바이트")
        for k, v in sorted(넘침.items()):
            줄들.append(f"{이름} | {k} | {v}")
        shutil.rmtree(d, ignore_errors=True)
    print("--- SASS넘침.txt 꼴 ---")
    print("\n".join(줄들))


if __name__ == "__main__":
    main()
