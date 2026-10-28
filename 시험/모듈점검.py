#!/usr/bin/env python3
"""빠른 점검의 모듈 · SASS 단계 — 시험/빠른점검.py 가 불러 쓴다(혼자 돌리지 않는다).
  5. 모듈(커널 소스 · PTX · SPIR-V 묶음)을 처음부터 만들어 시험/모듈해시.txt 와 바이트까지 견준다 — 바꾸려면 빠른점검.py --해시갱신
  7. PTX → SASS(sm_89) — 커널마다 레지스터 · 넘침(spill)을 시험/SASS넘침.txt 와 견준다 — 바꾸려면 빠른점검.py --넘침갱신
CI(.github/workflows/ci.yml)는 모듈을 정하는 파일(커널/ · 백엔드/ · 엔진/파일.gl · 엔진/리눅스/파일.gl · 시험/모듈해시.txt · 시험/SASS넘침.txt ·
시험/글저장소커밋.txt)과 이 파일 · ci.yml 이 앞 커밋과 같으면 이 둘을 건너뛴다(빠른점검 --모듈없이). 그래서 이 둘의 코드는 빠른점검.py 와
따로 둔다 — 빠른점검.py 의 다른 단계를 고친 커밋이 모듈 · SASS(러너에서 15 ~ 20 분)를 다시 돌지 않게. 해시 · 돌리기도 같은 까닭으로 여기에
따로 있다(이 파일만 보면 두 단계가 정해진다). 결과 줄은 빠른점검.py 의 알림으로 낸다.
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXE = "" if sys.platform.startswith("linux") else ".exe"


def 해시(경로):
    h = hashlib.sha256()
    with open(경로, "rb") as f:
        for 덩이 in iter(lambda: f.read(1 << 20), b""):
            h.update(덩이)
    return h.hexdigest()


def 돌리기(명령, 기록, 환경=None, 폴더=None):
    """명령을 돌려 표준 출력 · 오류를 기록 파일에 — (종료 코드, 마지막 몇 줄)."""
    with open(기록, "wb") as f:
        r = subprocess.run(명령, stdout=f, stderr=subprocess.STDOUT, env=환경, cwd=폴더, stdin=subprocess.DEVNULL)
    꼬리 = open(기록, "rb").read().decode("utf-8", "replace").strip().splitlines()[-3:]
    return r.returncode, " / ".join(꼬리)


def 모듈표읽기():
    줄들 = []
    for 줄 in open(os.path.join(ROOT, "시험", "모듈해시.txt"), encoding="utf-8"):
        if 줄.strip() and not 줄.startswith("#"):
            칸 = [x.strip() for x in 줄.split("|")]
            이름, 인자, gl, ptx = 칸[:4]
            줄들.append((이름, 인자.split(), gl, ptx, 칸[4] if len(칸) > 4 else ""))
    return 줄들


def 모듈(글, 출력, 병렬, 갱신, 알림):
    생성기 = os.path.join(출력, "커널생성" + EXE)
    뒷단 = os.path.join(출력, "ptx-backend" + EXE)         # 빠른점검.py 의 빌드()가 백엔드/ 에서 만든 것
    뒷단V = os.path.join(출력, "spirv-backend" + EXE)
    표 = 모듈표읽기()

    def 하나(항목):
        이름, 인자 = 항목[0], 항목[1]
        gl = os.path.join(출력, 이름 + ".gl")
        ptx = os.path.join(출력, 이름 + ".ptx")
        spvs = os.path.join(출력, 이름 + ".spvs")
        for x in (gl, ptx, spvs):
            if os.path.exists(x):
                os.remove(x)
        t0 = time.time()
        # 깃발(--공유32 같은 것)은 출력 경로 뒤에 — 커널생성의 인자 차례. 표의 깃발은 늘 끝에 있고, 값을 받는 깃발(--헤드 64 — 0.12.0
        # 12-3c-2)은 그 값과 함께 옮긴다
        i = next((k for k, x in enumerate(인자) if x.startswith("--")), len(인자))
        자리, 깃발 = 인자[:i], 인자[i:]
        rc, 꼬리 = 돌리기([생성기] + 자리 + [gl] + 깃발, os.path.join(출력, f"생성_{이름}.txt"))
        if rc != 0 or not os.path.exists(gl):
            return 이름, None, None, None, f"생성 실패: {꼬리}", 0.0
        rc, 꼬리 = 돌리기([뒷단, gl, "-o", ptx], os.path.join(출력, f"PTX_{이름}.txt"))
        if rc != 0 or not os.path.exists(ptx):
            return 이름, 해시(gl), None, None, f"PTX 실패(종료 {rc}): {꼬리[-300:]}", time.time() - t0
        if "--MoE" in 인자:                 # 0.12.0 12-3g-2 — MoE 의 판(원자 더하기)은 CUDA 만: SPIR-V 칸은 "-"
            return 이름, 해시(gl), 해시(ptx), "-", "", time.time() - t0
        rc, 꼬리 = 돌리기([뒷단V, gl, "-o", spvs], os.path.join(출력, f"SPIRV_{이름}.txt"))
        if rc != 0 or not os.path.exists(spvs):
            return 이름, 해시(gl), 해시(ptx), None, f"SPIR-V 실패(종료 {rc}): {꼬리[-300:]}", time.time() - t0
        return 이름, 해시(gl), 해시(ptx), 해시(spvs), "", time.time() - t0
    with ThreadPoolExecutor(병렬) as ex:
        결과 = list(ex.map(하나, 표))
    새줄 = []
    for (이름, 인자, 옛gl, 옛ptx, 옛spv), (_, gl, ptx, spv, 오류, 초) in zip(표, 결과):
        새줄.append(f"{이름} | {' '.join(인자)} | {gl or '?'} | {ptx or '?'} | {spv or '?'}")
        if 오류:
            알림(f"모듈 {이름}", False, 오류)
        elif 갱신:
            알림(f"모듈 {이름}", True, f"해시를 새로 적음 ({초:.0f} 초)")
        else:
            같음 = gl == 옛gl and ptx == 옛ptx and spv == 옛spv
            덧 = (f"소스 {'같음' if gl == 옛gl else '다름'} · PTX {'같음' if ptx == 옛ptx else '다름'} · "
                 f"SPIR-V {'같음' if spv == 옛spv else '다름'} ({초:.0f} 초)")
            if not 같음:
                덧 += f" — 새 해시 {gl} | {ptx} | {spv} (일부러 바꿨다면 --해시갱신)"
            알림(f"모듈 {이름}", 같음, 덧)
    머리 = [줄 for 줄 in open(os.path.join(ROOT, "시험", "모듈해시.txt"), encoding="utf-8") if 줄.startswith("#")]
    내용 = "".join(머리) + "\n".join(새줄) + "\n"
    open(os.path.join(출력, "모듈해시.txt"), "w", encoding="utf-8", newline="\n").write(내용)
    if 갱신 and all(not r[4] for r in 결과):
        open(os.path.join(ROOT, "시험", "모듈해시.txt"), "w", encoding="utf-8", newline="\n").write(내용)
        print("시험/모듈해시.txt 를 새 해시로 고쳤다", flush=True)


def 커널이름(기호):
    """PTX 의 이름(_G + UTF-8 16진) → 글 이름."""
    if 기호.startswith("_G"):
        try:
            return bytes.fromhex(기호[2:]).decode("utf-8")
        except ValueError:
            pass
    return 기호


def SASS(출력, ptxas, 병렬, 갱신, 알림):
    """7. PTX → SASS(sm_89) — NVIDIA ptxas 로(검사에만 — 엔진은 드라이버가 PTX 를 그대로 받는다). 커널마다 레지스터 · 넘침(spill) 바이트를
    SASS_<모듈>.txt 에 적는다. ptxas 가 받지 않거나, 시험/SASS넘침.txt(지금 넘치는 커널과 바이트)에 없는 커널이 넘치거나 더 많이 넘치면 못 미침
    — 레지스터 넘침은 GPU 없이 보이는 성능 회귀다. 일부러 바꿨다면 --넘침갱신."""
    표 = 모듈표읽기()
    넘침파일 = os.path.join(ROOT, "시험", "SASS넘침.txt")
    넘침기준 = {}
    if os.path.exists(넘침파일):
        for 줄 in open(넘침파일, encoding="utf-8"):
            if 줄.strip() and not 줄.startswith("#"):
                m_, k_, v_ = [x.strip() for x in 줄.split("|")]
                넘침기준.setdefault(m_, {})[k_] = int(v_)
    새줄 = []

    def 하나(항목):
        이름 = 항목[0]
        t0 = time.time()
        if not os.path.exists(os.path.join(출력, 이름 + ".ptx")):
            # 모듈 생성 · PTX 가 이미 못 미침으로 적혔다 — 여기서 멈추지 않고 그 모듈만 못 미침으로 적어 요약까지 간다
            return 이름, -1, "PTX 가 없다 — 위의 [모듈 " + 이름 + "] 줄(생성 또는 PTX 실패)", 0.0
        임시 = tempfile.mkdtemp(prefix="geulpu_sass_")     # ptxas 는 ASCII 가 아닌 경로를 열지 못한다(Windows 의 ANSI 코드 페이지)
        try:
            shutil.copyfile(os.path.join(출력, 이름 + ".ptx"), os.path.join(임시, "m.ptx"))
            r = subprocess.run([ptxas, "-arch=sm_89", "-v", "m.ptx", "-o", "m.cubin"], capture_output=True, stdin=subprocess.DEVNULL, cwd=임시)
        finally:
            shutil.rmtree(임시, ignore_errors=True)
        글 = (r.stdout + r.stderr).decode("utf-8", "replace")
        return 이름, r.returncode, 글, time.time() - t0
    with ThreadPoolExecutor(병렬) as ex:
        결과 = list(ex.map(하나, 표))
    for 이름, rc, 글, 초 in 결과:
        커널들 = []
        지금 = None
        for 줄 in 글.splitlines():
            m = re.search(r"Compiling entry function '([^']+)'", 줄)
            if m:
                지금 = 커널이름(m.group(1))
                커널들.append([지금, 0, 0, 0])
                continue
            m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", 줄)
            if m and 커널들:
                커널들[-1][2] = int(m.group(1))
                커널들[-1][3] = int(m.group(2))
            m = re.search(r"Used (\d+) registers", 줄)
            if m and 커널들:
                커널들[-1][1] = int(m.group(1))
        open(os.path.join(출력, f"SASS_{이름}.txt"), "w", encoding="utf-8").write(
            "커널 | 레지스터 | 넘침 저장 바이트 | 넘침 읽기 바이트\n" + "".join(f"{k} | {r_} | {s} | {l}\n" for k, r_, s, l in 커널들) +
            ("" if rc == 0 else "\n" + 글[-2000:]))
        넘침 = {k[0]: k[2] + k[3] for k in 커널들 if k[2] or k[3]}
        기준 = 넘침기준.get(이름, {})
        새것 = {k: v for k, v in 넘침.items() if v > 기준.get(k, 0)}          # 기준에 없거나 기준보다 많이 넘치는 커널
        for k, v in sorted(넘침.items()):
            print(f"    넘침 {이름} · {k}: {v} 바이트{'' if k not in 새것 else ' (새로 · 늘어남)'}", flush=True)
        새줄.extend(f"{이름} | {k} | {v}" for k, v in sorted(넘침.items()))
        덧 = f"커널 {len(커널들)} 개 · 레지스터 최대 {max((k[1] for k in 커널들), default=0)} · 넘침 있는 커널 {len(넘침)} ({초:.0f} 초)"
        if 새것:
            덧 += f" — 새로 넘치거나 늘어난 커널 {sorted(새것)} (일부러면 --넘침갱신)"
        알림(f"SASS {이름}", rc == 0 and bool(커널들) and (갱신 or not 새것), 덧 if rc == 0 else f"ptxas 종료 {rc}: {글.strip()[-300:]}")
    머리 = "# 빠른 점검의 SASS 단계(ptxas -arch=sm_89)에서 레지스터가 넘치는 커널 — 모듈 | 커널 | 넘침 저장 + 읽기 바이트\n"
    if os.path.exists(넘침파일):
        머리 = "".join(줄 for 줄 in open(넘침파일, encoding="utf-8") if 줄.startswith("#")) or 머리
    내용 = 머리 + "\n".join(새줄) + "\n"
    open(os.path.join(출력, "SASS넘침.txt"), "w", encoding="utf-8", newline="\n").write(내용)
    if 갱신 and all(r[1] == 0 for r in 결과):
        open(넘침파일, "w", encoding="utf-8", newline="\n").write(내용)
        print("시험/SASS넘침.txt 를 새로 고쳤다", flush=True)
