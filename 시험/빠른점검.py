#!/usr/bin/env python3
"""GeulPU 빠른 점검 — GPU · 모델 파일 없이 커밋마다 도는 검사(CI: .github/workflows/ci.yml 이 이것을 부른다).
단계마다의 공식 실행(GPU 의 비트 대조 · 속도 — 공식실행/)은 그대로 두고, 그 가운데 GPU 가 들지 않는 몫을 여기로:
  1. 파이썬 문법(저장소의 .py 모두)과 셸 스크립트 문법(공식실행/*.sh — bash -n)
  2. 계약의 CPU 참조 자체 시험(시험/계약자체.py)
  3. 판(타일 모양) 표가 생성기 · 엔진에서 같은가
  4. 글(geul) 언어 프로그램이 빌드되는가 — 커널 생성기와 엔진 프로그램 모두(글 컴파일러)
  5. 모듈(커널 소스 · PTX · SPIR-V 묶음)을 처음부터 만들어 시험/모듈해시.txt 와 바이트까지 견준다 — 바꾸려면 --해시갱신
     (31단계 — SPIR-V 는 글 저장소의 자체SPIRV 뒷단으로. GPU 없이 만들 수 있어 PTX 와 같이 CI 에서 본다)
  6. 실행기가 가져오는 DLL 이 kernel32 · shell32 · nvcuda 뿐인가(18단계 · 22단계의 조건) — 리눅스에서는 libc.so.6 · libcuda.so.1 뿐인가(29단계)

  python -X utf8 시험/빠른점검.py [--글 <글 저장소 — 기본 ../geul-lang-v2>] [--병렬 N] [--출력 <폴더 — 기본 빌드/점검>] [--해시갱신] [--계약없이]
                                [--모듈없이]
--모듈없이: 5 · 7(모듈 · SASS)을 건너뛴다 — CI 는 커널/ · 시험/모듈해시.txt · 시험/SASS넘침.txt · 시험/글저장소커밋.txt · 이 파일 · ci.yml 이
앞 커밋과 같으면 이것으로 돈다(모듈은 그 파일들로만 정해진다 — 엔진/ · 앱/ 은 모듈을 바꾸지 않는다). 하나라도 바뀌면 모두 본다.
글 저장소에는 build/self_컴파일러.exe · build/자체PTX.exe · build/자체SPIRV.exe(리눅스에서는 확장자 없이)가 있어야 한다
(README 의 빠른 시작 2). 리눅스(29단계 — 글 D-50)에서는 같은 프로그램을 ELF 로 빌드하고, 모듈은 윈도와 같은 시험/모듈해시.txt 와 견준다.
마지막 줄 "빠른 점검: 모두 통과"(종료 0) 또는 "못 미침 …"(종료 1).
"""
import argparse
import ast
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
리눅스 = sys.platform.startswith("linux")      # 29단계: 리눅스(글 D-50) — 실행 파일은 확장자 없는 ELF
EXE = "" if 리눅스 else ".exe"
sys.path.insert(0, os.path.join(ROOT, "시험"))
import 가져오기표       # noqa: E402

실패 = []
요약 = []


def 알림(단계, 통과, 덧=""):
    줄 = f"{'통과' if 통과 else '못 미침'}  [{단계}] {덧}"
    print(줄, flush=True)
    요약.append(줄)
    if not 통과:
        실패.append(단계)


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


# ---- 1. 문법 ----
def 파이썬문법():
    수, 틀림 = 0, []
    for 폴더, 아래, 파일들 in os.walk(ROOT):
        아래[:] = [d for d in 아래 if d not in (".git", "빌드", "모델", "도구", "__pycache__", "점검결과")]
        for 이름 in 파일들:
            if 이름.endswith(".py"):
                경로 = os.path.join(폴더, 이름)
                수 += 1
                try:
                    compile(open(경로, "rb").read(), 경로, "exec")
                except SyntaxError as e:
                    틀림.append(f"{os.path.relpath(경로, ROOT)}:{e.lineno} {e.msg}")
    알림("파이썬 문법", not 틀림, f"파일 {수} 개" + (f" — {틀림[:5]}" if 틀림 else ""))


def bash찾기():
    """GNU bash — Windows 에서는 System32 의 bash.exe(WSL 띄우개)가 먼저 잡힐 수 있어 Git 의 bash 를 앞에 둔다."""
    후보 = [os.environ.get("점검_BASH", ""), r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe", "bash"]
    for b in 후보:
        if not b:
            continue
        try:
            r = subprocess.run([b, "--version"], capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0 and b"GNU bash" in r.stdout:
            return b
    return None


def 셸문법():
    bash = bash찾기()
    if bash is None:
        알림("셸 문법", True, "GNU bash 가 없어 건너뜀")
        return
    폴더들 = [d for d in ("공식실행", "재기") if os.path.isdir(os.path.join(ROOT, d))]   # GeulPUv2 2단계 — 재기/*.sh(견주기 · 생성 · 품질)
    if not 폴더들:
        알림("셸 문법", True, "공식실행/ · 재기/ 가 없어 건너뜀")
        return
    틀림 = []
    수 = 0
    for d in 폴더들:
        폴더 = os.path.join(ROOT, d)
        for 이름 in sorted(os.listdir(폴더)):
            if 이름.endswith(".sh"):
                수 += 1
                r = subprocess.run([bash, "-n", 이름], capture_output=True, cwd=폴더)
                if r.returncode != 0:
                    틀림.append(f"{d}/{이름}: {r.stderr.decode('utf-8', 'replace').strip()[:200]}")
    알림("셸 문법", not 틀림, f"{' · '.join(d + '/*.sh' for d in 폴더들)} {수} 개" + (f" — {틀림}" if 틀림 else ""))


# ---- 2. 계약 자체 시험 ----
def 계약자체(출력):
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "계약자체.py")], os.path.join(출력, "계약자체.txt"))
    알림("계약 자체 시험", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


# ---- 3. 판 표 ----
def 판표들():
    """(이름, WM, WN, MT, NT) 의 차례 목록을 곳마다."""
    표 = {}
    s = open(os.path.join(ROOT, "커널", "생성타일.gl"), encoding="utf-8").read()
    표["커널/생성타일.gl"] = [(m[1], *map(int, m[2:6])) for m in re.findall(r'판하나\((\d+), "([^"]+)", (\d+), (\d+), (\d+), (\d+), \d+, \d+\)', s)]
    s = open(os.path.join(ROOT, "엔진", "엔진.gl"), encoding="utf-8").read()
    표["엔진/엔진.gl"] = [(m[1], *map(int, m[2:6])) for m in re.findall(r'판하나\((\d+), "([^"]+)", (\d+), (\d+), (\d+), (\d+)\)\.', s)]
    return 표


def 줄타일표들():
    """32단계 — 줄 타일 판 (이름, W, NW, BK, S) 의 차례 목록을 곳마다."""
    표 = {}
    for 파일 in ("커널/생성줄타일.gl", "엔진/엔진.gl"):
        s = open(os.path.join(ROOT, *파일.split("/")), encoding="utf-8").read()
        표[파일] = [(m[1], *map(int, m[2:6])) for m in re.findall(r'줄타일하나\((\d+), "([^"]+)", (\d+), (\d+), (\d+), (\d+)\)\.', s)]
    return 표


def 줄타일K표들():
    """32단계 — k-양자화의 줄 타일 판 (이름, W, NW, S) 의 차례 목록을 곳마다."""
    표 = {}
    for 파일 in ("커널/생성줄타일.gl", "엔진/엔진K.gl"):
        s = open(os.path.join(ROOT, *파일.split("/")), encoding="utf-8").read()
        표[파일] = [(m[1], *map(int, m[2:5])) for m in re.findall(r'줄타일K하나\((\d+), "([^"]+)", (\d+), (\d+), (\d+)\)\.', s)]
    return 표


def 판표():
    표 = 판표들()
    기준 = 표["커널/생성타일.gl"]
    다름 = [곳 for 곳, 값 in 표.items() if 값 != 기준]
    줄표 = 줄타일표들()
    줄기준 = 줄표["커널/생성줄타일.gl"]
    줄다름 = [곳 for 곳, 값 in 줄표.items() if 값 != 줄기준]
    K표 = 줄타일K표들()
    K기준 = K표["커널/생성줄타일.gl"]
    K다름 = [곳 for 곳, 값 in K표.items() if 값 != K기준]
    알림("판 표", bool(기준) and not 다름 and len(표) == 2 and bool(줄기준) and not 줄다름 and len(줄표) == 2 and bool(K기준) and not K다름 and len(K표) == 2,
         f"{len(기준)} 판 · 곳 {len(표)} · 줄 타일 판 {len(줄기준)} · 곳 {len(줄표)} · k-양자화 줄 타일 판 {len(K기준)} · 곳 {len(K표)}"
         + (f" — 생성기와 다른 곳 {다름 + 줄다름 + K다름}" if 다름 or 줄다름 or K다름 else ""))


# ---- 4 · 5 · 6. 빌드 · 모듈 · 가져오기 표 ----
def 호스트프로그램들():
    """엔진/ 에서 시작하기 가 있는 것(프로그램). 31단계 — 몸 파일(…몸.gl)은 GPU 층(쿠다.gl · 벌칸.gl)을 포함하지 않아 홀로 빌드되지
    않는다. 몸은 빼고, 몸을 포함하는 머리 파일(시험_엔진.gl · 시험_엔진V.gl …)을 프로그램으로 빌드한다."""
    폴더 = os.path.join(ROOT, "엔진")
    out = []
    for 이름 in os.listdir(폴더):
        if not 이름.endswith(".gl") or 이름.endswith("몸.gl"):
            continue
        s = open(os.path.join(폴더, 이름), encoding="utf-8").read()
        if "시작하기]는" in s or re.search(r'포함 "[^"]*몸\.gl"', s):
            out.append(이름[:-3])
    return sorted(out)


def 빌드(글, 출력, 병렬):
    컴파일러 = os.path.join(글, "build", "self_컴파일러" + EXE)
    환경 = dict(os.environ, GEUL_ROOT=글)
    일들 = [("커널생성", os.path.join(ROOT, "커널", "커널생성.gl"))] + [(p, os.path.join(ROOT, "엔진", p + ".gl")) for p in 호스트프로그램들()]
    if not 리눅스:
        일들.append(("GeulPU", os.path.join(ROOT, "앱", "관리자.gl")))      # 36단계 — 앱의 관리자(윈도만: WinHTTP · CNG · 셸)

    def 하나(일):
        이름, 소스 = 일
        exe = os.path.join(출력, 이름 + EXE)
        if os.path.exists(exe):
            os.remove(exe)
        rc, 꼬리 = 돌리기([컴파일러, 소스, "-o", exe], os.path.join(출력, f"빌드_{이름}.txt"), 환경)
        return 이름, rc == 0 and os.path.exists(exe), 꼬리
    t0 = time.time()
    with ThreadPoolExecutor(병렬) as ex:
        결과 = list(ex.map(하나, 일들))
    틀림 = [f"{이름}: {꼬리}" for 이름, 된, 꼬리 in 결과 if not 된]
    알림("글 빌드", not 틀림, f"생성기 + 호스트 프로그램 {len(일들) - 1} 개 ({time.time() - t0:.1f} 초)" + (f" — {틀림}" if 틀림 else ""))
    return not 틀림


def 모듈표읽기():
    줄들 = []
    for 줄 in open(os.path.join(ROOT, "시험", "모듈해시.txt"), encoding="utf-8"):
        if 줄.strip() and not 줄.startswith("#"):
            칸 = [x.strip() for x in 줄.split("|")]
            이름, 인자, gl, ptx = 칸[:4]
            줄들.append((이름, 인자.split(), gl, ptx, 칸[4] if len(칸) > 4 else ""))
    return 줄들


def 모듈(글, 출력, 병렬, 갱신):
    생성기 = os.path.join(출력, "커널생성" + EXE)
    뒷단 = os.path.join(글, "build", "자체PTX" + EXE)
    뒷단V = os.path.join(글, "build", "자체SPIRV" + EXE)
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
        # 깃발(--공유32 같은 것)은 출력 경로 뒤에 — 커널생성의 인자 차례
        자리 = [x for x in 인자 if not x.startswith("--")]
        깃발 = [x for x in 인자 if x.startswith("--")]
        rc, 꼬리 = 돌리기([생성기] + 자리 + [gl] + 깃발, os.path.join(출력, f"생성_{이름}.txt"))
        if rc != 0 or not os.path.exists(gl):
            return 이름, None, None, None, f"생성 실패: {꼬리}", 0.0
        rc, 꼬리 = 돌리기([뒷단, gl, "-o", ptx], os.path.join(출력, f"PTX_{이름}.txt"))
        if rc != 0 or not os.path.exists(ptx):
            return 이름, 해시(gl), None, None, f"PTX 실패(종료 {rc}): {꼬리[-300:]}", time.time() - t0
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


def SASS(출력, ptxas, 병렬, 갱신):
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


def 가져오기검사(출력):
    """윈도: 실행기는 kernel32 · shell32 · nvcuda, 서버는 그에 ws2_32(28단계). 리눅스(29단계): 둘 다 libc.so.6 · libcuda.so.1 뿐."""
    허용 = {"kernel32.dll", "shell32.dll", "nvcuda.dll"}
    허용서버 = 허용 | {"ws2_32.dll"}
    if 리눅스:
        허용 = 허용서버 = {"libc.so.6", "libcuda.so.1"}
    exe = os.path.join(출력, "실행기" + EXE)
    if not os.path.exists(exe):
        알림("실행기의 가져오기 표", False, f"{os.path.basename(exe)} 가 없다(빌드 실패)")
        return
    dll들 = sorted({d.lower() for d, _ in 가져오기표.가져오기(exe)})
    알림("실행기의 가져오기 표", set(dll들) <= 허용, f"{dll들}")
    서버 = os.path.join(출력, "서버" + EXE)
    if not os.path.exists(서버):
        알림("서버의 가져오기 표", False, f"{os.path.basename(서버)} 가 없다(빌드 실패)")
        return
    dll2 = sorted({d.lower() for d, _ in 가져오기표.가져오기(서버)})
    알림("서버의 가져오기 표", set(dll2) <= 허용서버, f"{dll2}")
    if not 리눅스:
        # 36단계 — 앱의 관리자는 윈도의 DLL 만(GPU DLL 은 관리자가 띄우는 서버 · 도구만)
        허용앱 = {"kernel32.dll", "shell32.dll", "ws2_32.dll", "advapi32.dll", "bcrypt.dll", "winhttp.dll"}
        앱 = os.path.join(출력, "GeulPU" + EXE)
        if not os.path.exists(앱):
            알림("앱 관리자의 가져오기 표", False, "GeulPU.exe 가 없다(빌드 실패)")
            return
        dll3 = sorted({d.lower() for d, _ in 가져오기표.가져오기(앱)})
        알림("앱 관리자의 가져오기 표", set(dll3) <= 허용앱, f"{dll3}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--글", default=os.path.normpath(os.path.join(ROOT, "..", "geul-lang-v2")))
    p.add_argument("--병렬", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    p.add_argument("--출력", default=os.path.join(ROOT, "빌드", "점검"))
    p.add_argument("--해시갱신", action="store_true")
    p.add_argument("--ptxas", default=os.environ.get("점검_PTXAS", ""), help="NVIDIA ptxas — 주면 PTX → SASS(sm_89)까지 본다")
    p.add_argument("--넘침갱신", action="store_true", help="시험/SASS넘침.txt 를 지금 넘치는 커널로 다시 쓴다")
    p.add_argument("--계약없이", action="store_true", help="계약 자체 시험(numpy)을 건너뛴다 — 운영체제와 무관한 CPU 시험이라 다른 곳에서 돌린 것으로 갈음할 때")
    p.add_argument("--모듈없이", action="store_true", help="모듈 · SASS 를 건너뛴다 — 커널 · 생성기 · 글 고정 커밋이 앞 커밋과 같을 때(CI)")
    a = p.parse_args()
    글 = os.path.abspath(a.글)
    출력 = os.path.abspath(a.출력)
    os.makedirs(출력, exist_ok=True)
    t0 = time.time()
    파이썬문법()
    셸문법()
    if a.계약없이:
        print("계약 자체 시험: 건너뜀(--계약없이)", flush=True)
        요약.append("건너뜀  [계약 자체 시험] --계약없이")
    else:
        계약자체(출력)
    판표()
    도구 = [os.path.join(글, "build", x + EXE) for x in ("self_컴파일러", "자체PTX", "자체SPIRV")]
    없는것 = [x for x in 도구 if not os.path.exists(x)]
    if 없는것:
        알림("글 도구", False, f"없다: {없는것} — README 의 빠른 시작 2")
    else:
        print("글 도구: " + " · ".join(f"{os.path.basename(x)} {해시(x)[:16]}" for x in 도구), flush=True)
        if 빌드(글, 출력, a.병렬):
            if a.모듈없이:
                print("모듈 · SASS: 건너뜀(--모듈없이 — 커널 · 생성기 · 글 고정 커밋이 그대로)", flush=True)
                요약.append("건너뜀  [모듈 · SASS] --모듈없이")
            else:
                모듈(글, 출력, a.병렬, a.해시갱신)
            가져오기검사(출력)
            if a.모듈없이:
                pass
            elif a.ptxas:
                SASS(출력, a.ptxas, a.병렬, a.넘침갱신)
            else:
                print("SASS: ptxas 를 주지 않아 건너뜀(--ptxas)", flush=True)
    open(os.path.join(출력, "요약.txt"), "w", encoding="utf-8").write("\n".join(요약) + "\n")
    print(f"빠른 점검: {'모두 통과' if not 실패 else '못 미침 ' + str(실패)} ({time.time() - t0:.0f} 초)", flush=True)
    return 0 if not 실패 else 1


if __name__ == "__main__":
    sys.exit(main())
