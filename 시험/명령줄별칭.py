#!/usr/bin/env python3
"""#137 — 명령줄 옵션의 영어 별칭을 GPU 없이 본다(빠른 점검 6b — 시험/빠른점검.py 가 빌드 뒤에 부른다).

  1. 커널생성 --help 와 GeulPU --help 가 0 으로 끝나고 한글 · 영어 이름을 함께 쓰는가(--빠른 · --fast, --포트 · --port).
  2. 커널생성이 영어 이름(qwen2 · --fast …)으로 만든 모듈 소스가 한글 이름(qwen2큰 · --빠른 …)으로 만든 것과 바이트까지 같은가 — 두 쌍:
     Qwen2.5-7B 의 K 모듈(--빠른 --KV8 --F2 — 앱의 기본)과 Qwen3-8B 의 K 모듈(--먼).
  3. 서버 · 실행기의 --help — 프로세스가 뜨면(NVIDIA 드라이버가 있는 기기) 0 으로 끝나고 두 이름을 쓰는가, 뜨지 못하면(CI) 건너뜀.

  python -I -X utf8 시험/명령줄별칭.py <빌드 폴더(커널생성 · GeulPU · 서버 · 실행기)> <작업 폴더>
마지막 줄 '명령줄 별칭: 모두 통과 (N 가지)'(종료 0) 또는 '명령줄 별칭: 틀림 N'(종료 1).
"""
import os
import subprocess
import sys

리눅스 = sys.platform.startswith("linux")
EXE = "" if 리눅스 else ".exe"
틀림 = []
가지 = 0


def 돌리기(명령, 폴더=None):
    try:
        r = subprocess.run(명령, capture_output=True, timeout=600, cwd=폴더)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"{e}"
    글 = (r.stdout + r.stderr).decode("utf-8", "replace")
    return r.returncode, 글


def 확인(이름, 된, 덧=""):
    global 가지
    가지 += 1
    print(f"{'통과' if 된 else '틀림'}  {이름}{(' — ' + 덧) if 덧 else ''}", flush=True)
    if not 된:
        틀림.append(이름)


def 도움말(exe, 이름, 한글, 영어, 건너뜀허용):
    rc, 글 = 돌리기([exe, "--help"])
    if rc is None or (건너뜀허용 and rc != 0 and "사용법" not in 글):
        print(f"건너뜀  {이름} --help (뜨지 못함 — {('rc ' + str(rc)) if rc is not None else 글.strip()[:80]})", flush=True)
        return
    확인(f"{이름} --help", rc == 0 and 한글 in 글 and 영어 in 글, f"rc {rc}, {한글} {'있음' if 한글 in 글 else '없음'} · {영어} {'있음' if 영어 in 글 else '없음'}")


def 모듈쌍(gen, 작업, 이름, 한글인자, 영어인자):
    a = os.path.join(작업, 이름 + "_한글.gl")
    b = os.path.join(작업, 이름 + "_영어.gl")
    for 출력, 인자 in ((a, 한글인자), (b, 영어인자)):
        if os.path.exists(출력):
            os.remove(출력)
        rc, 글 = 돌리기([gen] + 인자[:7] + [출력] + 인자[7:])
        if rc != 0 or not os.path.exists(출력):
            확인(f"모듈 {이름} ({'한글' if 출력 == a else '영어'} 이름)", False, f"rc {rc}: {글.strip()[-200:]}")
            return
    with open(a, "rb") as f:
        xa = f.read()
    with open(b, "rb") as f:
        xb = f.read()
    확인(f"모듈 {이름}: 한글 이름 = 영어 이름", xa == xb and len(xa) > 100000, f"{len(xa):,} 바이트 · {'같음' if xa == xb else '다름'}")


def main():
    빌드, 작업 = sys.argv[1], sys.argv[2]
    os.makedirs(작업, exist_ok=True)
    gen = os.path.join(빌드, "커널생성" + EXE)
    app = os.path.join(빌드, "GeulPU" + EXE)
    서버 = os.path.join(빌드, "서버" + EXE)
    실행기 = os.path.join(빌드, "실행기" + EXE)
    for p, 이름 in ((gen, "커널생성"), (app, "GeulPU")):
        if not os.path.exists(p):
            print(f"명령줄 별칭: 틀림 1 — {이름} 이 없다(빌드 실패): {p}")
            return 1
    도움말(gen, "커널생성", "--빠른", "--fast", False)
    도움말(app, "GeulPU", "--포트", "--port", False)
    for p, 이름, 한글, 영어 in ((서버, "서버", "--슬롯", "--slots"), (실행기, "실행기", "--최대길이", "--max-len")):
        if os.path.exists(p):
            도움말(p, 이름, 한글, 영어, True)
    # 모듈 소스 — 7B 꼴(Qwen2.5-7B: 너비 3584 · 머리 28 · KV 4 · 확장폭 18944 · 낱말 152064)과 Qwen3-8B(4096 · 32 · 8 · 12288 · 151936)
    모듈쌍(gen, 작업, "qwen2K_fast",
          ["qwen2큰", "K", "3584", "28", "4", "18944", "152064", "--빠른", "--KV8", "--F2"],
          ["qwen2", "K", "3584", "28", "4", "18944", "152064", "--fast", "--KV8", "--F2"])
    모듈쌍(gen, 작업, "qwen3K_exact",
          ["qwen3큰", "K", "4096", "32", "8", "12288", "151936", "--먼"],
          ["qwen3", "K", "4096", "32", "8", "12288", "151936", "--exact"])
    if 틀림:
        print(f"명령줄 별칭: 틀림 {len(틀림)} — {틀림}")
        return 1
    print(f"명령줄 별칭: 모두 통과 ({가지} 가지)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
