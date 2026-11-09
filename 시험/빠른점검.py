#!/usr/bin/env python3
"""GeulPU 빠른 점검 — GPU · 모델 파일 없이 커밋마다 도는 검사(CI: .github/workflows/ci.yml 이 이것을 부른다).
단계마다의 공식 실행(GPU 의 비트 대조 · 속도 — 공식실행/)은 그대로 두고, 그 가운데 GPU 가 들지 않는 몫을 여기로:
  1. 파이썬 문법(저장소의 .py 모두)과 셸 스크립트 문법(공식실행/*.sh — bash -n)
  2. 계약의 CPU 참조 자체 시험(시험/계약자체.py)
     2b. 디버거 v1 — 도구/debugger.py 의 실행 로그 나누기 · 분기 요청 · 중단점 · 비교 계산을 손으로 정한 값과 참조로, node 가 있으면
         화면(app.js)의 같은 계산도(시험/디버거도구.py · 시험/디버거화면.js)
     2c. 앱 화면 — app.js 문법(node) · 화면 문구가 모두 영어 사전에 있는가(시험/화면사전.py)
     2e. 앱의 모델 찾기 표시(app.js 의 fdSupport)가 2026-10-30 조사의 트렌딩 200 저장소(시험/찾기표시경우.json — Hugging Face 메타데이터 · 판정)에서
         받지 못하는 저장소를 '실행 가능'이라 하지 않는가(시험/찾기표시.py · 찾기표시.js — node 가 있을 때, 4 의 서버가 뜨면 --구조들 과도 견줌)
     2d. 여러 세션으로 개발하기 — GPU 줄의 자체 시험(시험/GPU줄.py — GPU 없이) · 바뀐 것 조각(문서/바뀐것.d/)의 꼴(시험/바뀐것합치기.py)
     2f. 파이썬 API(도구/pypi/geulpu/api.py — 시험/파이썬API.py): 가짜 앱(로짓 행을 문맥으로 정하는 서버)으로 요청 본문 · 응답 풀기 ·
         모델 더하기 · 불러오기 · 점수 · 분기 · 비교 · 설정 되돌리기, 휠의 파일 목록
     2g. 실험 기록 비교(앱리디자인 F1 — 시험/기록비교.py --self-test): 같은 사본 · 그때의 값만 다른 사본 · 스트리밍으로 받은 사본은 같다고,
         일부러 바꾼 사본 열한 가지는 다르다고 하는가
  3. 커널 변형(타일 모양) 표가 생성기 · 엔진에서 같은가
     3b. Vulkan 선언의 리눅스 판(엔진/리눅스/벌칸선언.gl — libvulkan.so.1, #135)이 윈도 판(엔진/벌칸선언.gl)과 같은 함수 · 서명인가
  4. 글(geul) 언어 프로그램이 빌드되는가 — 커널 생성기 · PTX · SPIR-V 백엔드(백엔드/)와 엔진 프로그램 모두(글 컴파일러)
     4b. 문맥 저장소의 GPU 없는 시험 — 가짜 장치 위의 저장소.gl(엔진/시험_저장소CPU.gl)을 파이썬 참조(시험/저장소CPU.py)와 비교
     4c. RoPE 표(GeulPUv2 ③a — YaRN 배 2 · 4 와 늘이지 않은 표, 0.13.0 — Gemma 4 의 넓은 표 둘) — 엔진/시험_로프표.gl 의 표를 CPU 참조(시험/로프표.py)와 바이트까지
     4d. 디버거 v1 — 앱 기록기의 JSON 문법 훑기(앱/디버거제이슨.gl — 엔진/시험_디버거제이슨.gl)를 파이썬의 엄격한 풀기(시험/디버거제이슨.py)와
     4e. 추론 서버의 브라우저 지킴(엔진/요청지킴.gl — Host 는 127.0.0.1 · localhost 만, Origin 이 있으면 거부)을 경우마다(엔진/시험_요청지킴.gl)
     4f. 망가진 GGUF — 엔진의 GGUF 열기(엔진/시험_GGUF열기.gl)가 조작된 파일 서른 남짓을 오류로 끝내는가(시험/망가진입력.py gguf)
     4g. 유한하지 않은 로짓 — 뽑기(엔진/시험_뽑기.gl)가 NaN · ±Inf 가 섞인 행에서 범위 안의 번호를 고르는가(시험/망가진입력.py 뽑기)
     4h. 앱의 모델 해시 지문(#19) — 같은 경로의 모델 파일이 바뀌면 옛 SHA-256 을 버리는가(시험/모델해시.py — 윈도만, 앱을 임시 자료로 띄워)
     4i. 앱의 모델 모양 — 셈할 수 없는 파일(qwen2 의 RMSNorm eps 가 10⁻⁶ · 10⁻⁵ 이 아님 — #16, qwen2 의 RoPE 늘이기 정보 — #17)을 까닭과
         함께 받지 않는가(시험/모델모양.py — 윈도만)
     4j. 앱의 메모리 누수(#22) — /api/state · 올바른 본문 · 잘린 본문을 되풀이해도 앱의 private 메모리가 늘지 않는가(시험/앱누수.py — 윈도만)
     4k. 뽑기 · 로그확률의 글 판이 계약의 파이썬 참조와 같은가(시험/뽑기.py 대조 · 시험/로그확률38.py 대조 — #27)
     4l. jinja 템플릿 해석기(0.12.0 12-G1 — 엔진/진해석.gl · 엔진/시험_틀해석.gl)가 jinja2(transformers 의 채팅 템플릿 환경)와 같은 글을
         내는가 — 기능 사례(시험/틀해석.py — jinja2 가 없으면 건너뜀)
     4m. 토크나이저의 정규식 사전 처리(0.12.0 12-G2 — 엔진/정규식.gl · 엔진/시험_토큰.gl)가 llama.cpp 의 어휘 시험(시험/어휘시험.txt 의 열네
         어휘 — 0.12.0 12-3h-2 에 gemma-4, --어휘 <폴더>, 시험/어휘받기.py 가 받는다. 폴더를 주지 않으면 건너뜀)과 같은 토큰을 내는가(시험/어휘시험.py)
     4n. Gemma 4 의 도구 호출 풀기(0.13.0 — 엔진/젬마도구.gl · 엔진/시험_젬마도구.gl)가 llama.cpp 의 gemma4_to_json 과 같은 이름 · 인자 JSON 을
         내는가(시험/젬마도구.py)
     4o. 도구 오류 층(에이전트 호환 4 단계 — 엔진/도구오류.gl · 엔진/시험_도구오류.gl): 4a 감지가 에이전트들이 실제로 보낸 도구 결과에서
         손으로 단 답과 같은 실패 · 종류 · 규칙을 내고 실패가 아닌 것은 잡지 않는가, 4b 꼴(C1 · C2 · C3)이 연구의 참조와 글자까지 같은가
         (시험/도구오류.py)
     4p. 모델 평가의 지표 계산(문서/모델평가.md 1-2 — 앱/평가계산.gl · 엔진/시험_평가계산.gl): KL divergence(상위 K + 나머지 확률) ·
         꼬리 · 백분위수 · 부호 검정 · 계약의 지수 · 로그가 파이썬 참조(시험/평가계산.py)와 비트까지 같은가
     4q. 모델 평가의 GGUF 다시 쓰기(문서/모델평가.md 2-1 — 앱/GGUF쓰기.gl · 엔진/시험_GGUF쓰기.gl): 메타데이터만 고친 GGUF 의 키 · 텐서
         정보 · 자료 구역 · 맞춤이 파이썬의 GGUF 읽기(시험/GGUF쓰기.py)로 본 기대와 같은가, 틀린 고침을 거절하는가
     4r. 모델 평가의 정답 채점(문서/모델평가.md 3-1 — 앱/평가채점.gl · 엔진/시험_채점.gl): number · choice · exact · tool_call 의 판정과
         정답률의 부트스트랩 구간이 파이썬 참조(시험/채점.py)와 같은가, 내장 데이터(앱/data/eval — 3-2)의 정답으로 지은 답이 둘 다에서 맞음인가
     4s. 층 나누기의 상태 목록(엔진/조각.gl — 시험/조각상태.py): 엔진 핵심 파일의 전역마다 조각상태(조각담기 · 조각풀기)에 있거나
         시험/조각상태.txt 에 공유로 적혀 있는가 — 전역을 더하며 조각마다의 상태를 빠뜨리지 않게(외부 리뷰)
     4t. 파이썬 API 의 geulpu.start · add_model · close 를 빌드한 앱으로 — 창 없이 띄우고, 그 자료 폴더의 앱인지 보고, SHA-256 을
         받고, 끈다(시험/파이썬API.py --앱 — 임시 자료 · GPU 없이 · 윈도만)
     4u. llama-server 의 로그 확률 흉내(llama.cpp 재현 4단계 — 엔진/라마로그확률.gl · 엔진/시험_라마로그확률.gl)가 서버 코드를 옮긴
         파이썬 참조(시험/라마영수증.py — MSVC partial_sort 의 자리 옮김 · 같은 C 런타임의 expf · logf)와 비트까지 같은가(GPU 없이)
     4v. llama-server 의 고르기 사슬 흉내(llama.cpp 재현 5단계 — 엔진/라마고르기.gl · 엔진/시험_라마고르기.gl)가 서버 코드를 옮긴
         파이썬 참조(시험/라마고르기.py — top_k · top_p · min_p · temperature → dist, std::mt19937 · MSVC uniform_real_distribution)와
         같은 토큰을 고르는가(GPU 없이)
     4w. llama-server 한 슬롯의 프롬프트 캐시 셈 흉내(llama.cpp 재현 7단계 — 앱/라마사슬.gl · 엔진/시험_라마사슬.gl)가 서버 코드를
         옮긴 파이썬 참조(시험/라마사슬.py — 슬롯 고르기 검사 · n_past · 체크포인트 되살리기 · 덩어리 나눔 · 생성 걸음)와 같은
         cache_n · ubatch 사슬을 내는가(GPU 없이). 24 · 25 · 26단계 — 서버 단위 흉내(모든 슬롯 · 슬롯 고르기 · RAM 캐시 · 쉬는 슬롯 비움)와
         통합 KV 의 칸 배치(칸 · n_kv)까지
  5. 모듈(커널 소스 · PTX · SPIR-V 묶음)을 처음부터 만들어 시험/모듈해시.txt 와 바이트까지 견준다 — 바꾸려면 --해시갱신
     (31단계 — SPIR-V 는 SPIR-V 백엔드로. GPU 없이 만들 수 있어 PTX 와 같이 CI 에서 본다. 두 백엔드는 4 에서 빌드한 것.
     0.12.0 12-3g-2 — --MoE 행은 CUDA 만이라 SPIR-V 칸이 "-"). 5 와 7(SASS)의 코드는 시험/모듈점검.py
  6. 실행기가 가져오는 DLL 이 kernel32 · shell32 · nvcuda 뿐인가(18단계 · 22단계의 조건) — 리눅스에서는 libc.so.6 · libcuda.so.1 뿐인가(29단계),
     리눅스의 Vulkan 판(서버V · 장치V · 진단V)은 libc.so.6 · libvulkan.so.1 뿐인가(#135)
     6b. 명령줄 옵션의 영어 별칭(#137 — 시험/명령줄별칭.py): 커널생성 · GeulPU(· 서버 · 실행기 — 뜨면)의 --help 가 한글 · 영어 이름을 함께 쓰고,
         커널생성이 영어 이름(qwen2 · --fast)으로 만든 모듈 소스가 한글 이름(qwen2큰 · --빠른)으로 만든 것과 바이트까지 같은가

  python -X utf8 시험/빠른점검.py [--글 <글 저장소 — 기본 ../geul-lang-v2>] [--병렬 N] [--출력 <폴더 — 기본 빌드/점검>] [--해시갱신] [--계약없이]
                                [--모듈없이 | --기준커밋 <커밋>] [--어휘 <llama.cpp 의 어휘 시험 폴더>]
--모듈없이: 5 · 7(모듈 · SASS)을 건너뛴다 — CI 는 커널/ · 백엔드/ · 엔진/파일.gl · 엔진/리눅스/파일.gl · 시험/모듈해시.txt · 시험/SASS넘침.txt ·
시험/글저장소커밋.txt · 시험/모듈점검.py · ci.yml 이 앞 커밋과 같으면 이것으로 돈다(모듈은 그 파일들로만 정해진다 — 엔진/ 의 나머지 · 앱/ · 이
파일의 다른 단계는 모듈을 바꾸지 않는다).
--기준커밋: 그 커밋과 견주어 바뀐 모듈만 PTX · SPIR-V · SASS 까지 본다 — 소스는 모두 처음부터 만들어 견주고, 뒷단을 정하는 것(백엔드/ ·
글 고정 커밋 · 시험/모듈점검.py · ci.yml 등)이 그 커밋과 다르면 모두 본다(시험/모듈점검.py). CI 는 견준 커밋(PR 의 base · push 의 앞 커밋)을
준다. 주지 않으면 모두 본다.
글 저장소(고정 판 — 시험/글저장소커밋.txt)에는 build/self_컴파일러.exe(리눅스에서는 확장자 없이)가 있어야 한다(문서/개발.md 의
'소스에서 빌드하기' 2). 리눅스(29단계 — 글 D-50)에서는 같은 프로그램을 ELF 로 빌드하고, 모듈은 윈도와 같은 시험/모듈해시.txt 와 견준다.
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
import 모듈점검         # noqa: E402 — 5 · 7(모듈 · SASS)의 코드: CI 는 이 파일을 고친 커밋에서만 모듈 · SASS 를 다시 본다

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
        아래[:] = [d for d in 아래 if d not in (".git", "빌드", "모델", "__pycache__", "점검결과")]      # #27 — 도구/ 도 본다
        for 이름 in 파일들:
            if 이름.endswith(".py"):
                경로 = os.path.join(폴더, 이름)
                수 += 1
                try:
                    compile(open(경로, "rb").read(), 경로, "exec")
                except SyntaxError as e:
                    틀림.append(f"{os.path.relpath(경로, ROOT)}:{e.lineno} {e.msg}")
    알림("파이썬 문법", not 틀림, f"파일 {수} 개" + (f" — {틀림[:5]}" if 틀림 else ""))
    # #27 — 사용자가 직접 돌리는 도구는 문법만이 아니라 불러오기(import)까지: --help 가 종료 0 인가
    for 이름 in ("debugger.py", "repro_check.py", "jlens_fit.py"):         # #96 2단계 — jlens_fit 은 torch 를 인자를 읽은 뒤에 부른다
        rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "도구", 이름), "--help"],
                          os.path.join(tempfile.gettempdir(), f"geulpu_help_{이름}.txt"))
        알림(f"도구 {이름} --help", rc == 0, 꼬리[-200:] if rc != 0 else "")


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


def 화면점검(출력):
    """앱 화면 — app.js 문법(node 가 있으면) · 화면 문구(T 호출 · index.html)가 모두 영어 사전에 있는가. 앱리디자인 E1 — node 가 있으면 명령 창의
    명령 해석이 CLI 와 같은 요청을 만드는가(시험/명령창점검.mjs — CLI 쪽은 4 의 명령줄 점검)"""
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "화면사전.py")], os.path.join(출력, "화면사전.txt"))
    알림("앱 화면(문법 · 영어 사전)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")
    node = shutil.which("node")
    if node:
        t0 = time.time()
        rc, 꼬리 = 돌리기([node, os.path.join(ROOT, "시험", "명령창점검.mjs")], os.path.join(출력, "명령창점검.txt"))
        알림("명령 창의 명령 해석(CLI 와 같은 요청)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 디버거도구(출력):
    """디버거 v1 2단계 — 도구/debugger.py 의 GPU 없는 몫(실행 로그 · 분기 요청 · 중단점 · 비교 계산)을 손으로 정한 값과 참조로"""
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "디버거도구.py")], os.path.join(출력, "디버거도구.txt"))
    알림("디버거 도구(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 파이썬API(출력):
    """2f — 파이썬 API(도구/pypi/geulpu/api.py)를 가짜 앱으로: 요청 본문 · 응답 풀기 · 불러오기 · 점수 · 분기 · 비교(GPU · 앱 없이)"""
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "파이썬API.py")], os.path.join(출력, "파이썬API.txt"))
    알림("파이썬 API(가짜 앱 · GPU 없이)", rc == 0, f"{꼬리.split(' / ')[-1]} ({time.time() - t0:.1f} 초)")


def 명령줄(출력):
    """앱리디자인 C1(문서/명령줄.md) — 빌드한 GeulPU 의 하위 명령(CLI): 부를 수 있는 모든 기능을 세 이름으로 --dry-run 해 기능 목록과
    대조하고, 사용법 오류 · 앱 없음의 종료 코드를 본다(GPU · 앱 없이)"""
    exe = os.path.join(출력, "GeulPU" + EXE)
    if not os.path.exists(exe):
        알림("명령줄(CLI — GPU 없이)", False, "GeulPU 가 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "명령줄점검.py"), exe], os.path.join(출력, "명령줄.txt"))
    알림("명령줄(CLI — GPU 없이)", rc == 0, f"{꼬리.split(' / ')[-1]} ({time.time() - t0:.1f} 초)")


def 파이썬API앱(출력):
    """4t — 파이썬 API 의 start · add_model · close 를 빌드한 앱으로(임시 자료 · GPU 없이 — 윈도만)"""
    if 리눅스:
        return
    exe = os.path.join(출력, "GeulPU" + EXE)
    if not os.path.exists(exe):
        알림("파이썬 API — 앱 띄우고 끄기(GPU 없이)", False, "GeulPU 가 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "파이썬API.py"), "--앱", exe, "--자료",
                       os.path.join(출력, "파이썬API앱")], os.path.join(출력, "파이썬API앱.txt"))
    알림("파이썬 API — 앱 띄우고 끄기(GPU 없이)", rc == 0, f"{꼬리.split(' / ')[-1]} ({time.time() - t0:.1f} 초)")


def 기능목록(출력):
    """앱리디자인 B1(문서/기능목록.md) — 기능 목록(앱/data/operations.json)과 앱 · 웹 · 서버 코드의 대조, 그리고 잘못 일곱을 잡는 자체 시험(GPU 없이)"""
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "기능목록점검.py")], os.path.join(출력, "기능목록.txt"))
    알림("기능 목록(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "기능목록점검.py"), "--자체시험"],
                      os.path.join(출력, "기능목록자체.txt"))
    알림("기능 목록 — 자체 시험", rc == 0, 꼬리)


def 실험기록(출력):
    """앱리디자인 B2(문서/기능목록.md §6) — experiments/ 의 실험 기록 점검(형식 · 해시 · 크기 · index.md)과, 올바른 실험의 복사본을
    망가뜨려 잘못마다 잡는지 보는 자체 시험(GPU 없이)"""
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "도구", "experiment_check.py")], os.path.join(출력, "실험기록.txt"))
    알림("실험 기록(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "도구", "experiment_check.py"), "--self-test"],
                      os.path.join(출력, "실험기록자체.txt"))
    알림("실험 기록 — 자체 시험", rc == 0, 꼬리)


def 기록비교(출력):
    """앱리디자인 F1(문서/사용성시험.md §1) — 실험 폴더 둘의 기록 비교 도구가 같은 사본은 같다고, 일부러 바꾼 사본은 다르다고 하는가(GPU 없이)"""
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "기록비교.py"), "--self-test"],
                      os.path.join(출력, "기록비교자체.txt"))
    알림("기록 비교 — 자체 시험(GPU 없이)", rc == 0, f"{꼬리.split(' / ')[-1]} ({time.time() - t0:.1f} 초)")


def 개발도구(출력):
    """여러 세션으로 개발하기(문서/개발.md) — GPU 줄의 자체 시험(GPU 없이: 겹침 · 선 차례 · 죽은 대기자의 자리)과 바뀐 것 조각의 꼴"""
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "GPU줄.py"), "자체시험"], os.path.join(출력, "GPU줄.txt"))
    알림("GPU 줄(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "바뀐것합치기.py")], os.path.join(출력, "바뀐것조각.txt"))
    알림("바뀐 것 조각의 꼴(문서/바뀐것.d)", rc == 0, 꼬리)


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


def 벌칸선언짝():
    """3b — #135: 리눅스 대상이면 컴파일러가 엔진/벌칸선언.gl 대신 엔진/리눅스/벌칸선언.gl 을 포함한다(D-50). 두 판의 Vulkan 함수(기호 ·
    서명)가 같은가 — 라이브러리 이름(vulkan-1.dll · libvulkan.so.1)만 다르다. 윈도 판에만 함수를 더하면 리눅스 빌드가 그 함수를 모른다"""
    def 목록(경로, 라이브러리):
        머리 = f'외부 "{라이브러리}" '
        return sorted(줄[len(머리):] for 줄 in open(경로, encoding="utf-8").read().splitlines() if 줄.startswith(머리))
    윈 = 목록(os.path.join(ROOT, "엔진", "벌칸선언.gl"), "vulkan-1.dll")
    리 = 목록(os.path.join(ROOT, "엔진", "리눅스", "벌칸선언.gl"), "libvulkan.so.1")
    한쪽 = sorted({줄.split('"')[1] for 줄 in set(윈) ^ set(리)})
    알림("Vulkan 선언의 리눅스 판(#135)", bool(윈) and 윈 == 리, f"함수 {len(윈)} · {len(리)}" + (f" — 한쪽에만 있거나 서명이 다름: {한쪽}" if 한쪽 else ""))


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
    백엔드들 = [("ptx-backend", os.path.join(ROOT, "백엔드", "PTX", "PTX.gl")), ("spirv-backend", os.path.join(ROOT, "백엔드", "SPIRV", "SPIRV.gl"))]
    일들 += 백엔드들
    일들.append(("GeulPU", os.path.join(ROOT, "앱", "관리자.gl")))      # 36단계 — 앱의 관리자(리눅스는 앱/리눅스/윈도.gl · 받기망.gl)

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
    알림("글 빌드", not 틀림, f"생성기 + 백엔드 둘 + 호스트 프로그램 {len(일들) - 3} 개 ({time.time() - t0:.1f} 초)" + (f" — {틀림}" if 틀림 else ""))
    if not 틀림:
        print("백엔드: " + " · ".join(f"{이름} {해시(os.path.join(출력, 이름 + EXE))[:16]}" for 이름, _ in 백엔드들), flush=True)
    return not 틀림


def 저장소CPU(출력):
    """문서/문맥저장소.md 4단계 — 이름 사슬 · 머리말 · 내용 합 · 배치 · 되살리기 · 색인 차례 · 크기 한도를 GPU 없이(윈도 · 리눅스)"""
    exe = os.path.join(출력, "시험_저장소CPU" + EXE)
    if not os.path.exists(exe):
        알림("문맥 저장소(GPU 없이)", False, "시험_저장소CPU 가 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "저장소CPU.py"), exe], os.path.join(출력, "저장소CPU.txt"))
    알림("문맥 저장소(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 로프표(출력):
    """GeulPUv2 ③a — 로프.gl 의 표(늘이지 않음 · YaRN 배 2 · 4 — 위치 131,072 까지)가 CPU 참조와 바이트까지 같은가(GPU 없이)"""
    exe = os.path.join(출력, "시험_로프표" + EXE)
    if not os.path.exists(exe):
        알림("RoPE 표(GPU 없이)", False, "시험_로프표 가 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "로프표.py"), exe], os.path.join(출력, "로프표.txt"))
    알림("RoPE 표(GPU 없이)", rc == 0, f"{꼬리.split(' / ')[-1]} ({time.time() - t0:.1f} 초)")


def 디버거제이슨(출력):
    """디버거 v1 — 앱의 기록기가 기록 줄마다 쓰는 JSON 문법 훑기(앱/디버거제이슨.gl)가 파이썬의 엄격한 풀기와 같은가(GPU 없이)"""
    exe = os.path.join(출력, "시험_디버거제이슨" + EXE)
    if not os.path.exists(exe):
        알림("기록기 JSON 훑기(GPU 없이)", False, "시험_디버거제이슨 이 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "디버거제이슨.py"), exe],
                      os.path.join(출력, "디버거제이슨.txt"))
    알림("기록기 JSON 훑기(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 틀해석(출력):
    """0.12.0 12-G1 — 서버가 GGUF 의 채팅 템플릿을 그대로 실행하는 jinja 해석기(엔진/진해석.gl)가 jinja2 와 바이트까지 같은 글을 내는가(GPU 없이)"""
    exe = os.path.join(출력, "시험_틀해석" + EXE)
    if not os.path.exists(exe):
        알림("jinja 템플릿 해석기(GPU 없이)", False, "시험_틀해석 이 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "틀해석.py"), exe], os.path.join(출력, "틀해석.txt"))
    if rc == 3:
        print("jinja 템플릿 해석기: 건너뜀(jinja2 없음 — pip install -r requirements.txt)", flush=True)
        요약.append("건너뜀  [jinja 템플릿 해석기] jinja2 없음")
        return
    알림("jinja 템플릿 해석기(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 젬마도구(출력):
    """0.13.0 — Gemma 4 의 도구 호출(<|tool_call>call:이름{…}<tool_call|>)을 이름 · JSON 인자로 푸는가(GPU 없이)"""
    exe = os.path.join(출력, "시험_젬마도구" + EXE)
    if not os.path.exists(exe):
        알림("Gemma 4 도구 호출 풀기(GPU 없이)", False, "시험_젬마도구 가 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "젬마도구.py"), exe], os.path.join(출력, "젬마도구.txt"))
    알림("Gemma 4 도구 호출 풀기(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 도구오류(출력):
    """에이전트 호환 4 단계 4a — 서버가 대화 속의 실패한 도구 결과를 찾고 종류를 가르는가(GPU 없이 — 출력은 바꾸지 않는 감지)"""
    exe = os.path.join(출력, "시험_도구오류" + EXE)
    if not os.path.exists(exe):
        알림("도구 오류 감지(GPU 없이)", False, "시험_도구오류 가 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "도구오류.py"), exe], os.path.join(출력, "도구오류.txt"))
    알림("도구 오류 감지(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 어휘시험(출력, 폴더):
    """0.12.0 12-G2 — 토크나이저(정규식 사전 처리 · 손으로 짠 갈래 · SentencePiece)가 llama.cpp 의 어휘 시험과 같은 토큰을 내는가(GPU 없이)"""
    if not 폴더:
        print("어휘 시험: 건너뜀(--어휘 를 주지 않음 — 시험/어휘받기.py 로 받는다)", flush=True)
        요약.append("건너뜀  [어휘 시험] --어휘 없음")
        return
    exe = os.path.join(출력, "시험_토큰" + EXE)
    if not os.path.exists(exe):
        알림("토크나이저 어휘 시험(GPU 없이)", False, "시험_토큰 이 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "어휘시험.py"), exe, 폴더], os.path.join(출력, "어휘시험.txt"))
    알림("토크나이저 어휘 시험(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 요청지킴(출력):
    """추론 서버의 브라우저 지킴(엔진/요청지킴.gl) — DNS 리바인딩 Host · 웹페이지의 Origin 을 막고 프로그램 · 앱 프록시는 받는가(GPU 없이)"""
    exe = os.path.join(출력, "시험_요청지킴" + EXE)
    if not os.path.exists(exe):
        알림("서버의 요청 지킴(GPU 없이)", False, "시험_요청지킴 이 없다(빌드 실패)")
        return
    rc, 꼬리 = 돌리기([exe], os.path.join(출력, "요청지킴.txt"))
    알림("서버의 요청 지킴(GPU 없이)", rc == 0, 꼬리)


def 계약대조(출력):
    """#27 — 호스트 쪽 셈(뽑기 · 로그확률)의 글 판이 계약의 파이썬 참조와 같은가 — 시험/뽑기.py 대조(합성 행 · 매개변수 열두 벌) ·
    시험/로그확률38.py 대조(한 번에 · 하나씩). 커널은 모듈 해시가 지키지만 이 셈은 글(geul) 컴파일러의 f64 코드가 바뀌어도 CI 가 몰랐다"""
    for 이름, 스크립트, exe이름 in (("뽑기의 계약 대조(GPU 없이)", "뽑기.py", "시험_뽑기"), ("로그확률의 계약 대조(GPU 없이)", "로그확률38.py", "시험_로그확률")):
        exe = os.path.join(출력, exe이름 + EXE)
        if not os.path.exists(exe):
            알림(이름, False, f"{exe이름} 이 없다(빌드 실패)")
            continue
        t0 = time.time()
        rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", 스크립트), "대조", exe],
                          os.path.join(출력, f"계약대조_{exe이름}.txt"))
        다름들 = [int(x) for x in re.findall(r"다름 (\d+)", 꼬리)]               # 뽑기.py 는 다름이 있어도 종료 0 이라 글로 본다
        알림(이름, rc == 0 and bool(다름들) and not any(다름들), f"{꼬리} ({time.time() - t0:.1f} 초)")


def 평가계산(출력):
    """모델 평가 1-2 — 앱의 지표 계산(앱/평가계산.gl)이 파이썬 참조(시험/평가계산.py)와 비트까지 같은가(GPU 없이)"""
    exe = os.path.join(출력, "시험_평가계산" + EXE)
    if not os.path.exists(exe):
        알림("평가 지표의 계산(GPU 없이)", False, "시험_평가계산 이 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "평가계산.py"), "check", exe],
                      os.path.join(출력, "평가계산.txt"))
    알림("평가 지표의 계산(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 라마로그확률(출력):
    """4u — llama.cpp 재현 4단계: llama-server 의 로그 확률 흉내(글)가 서버 코드를 옮긴 파이썬 참조와 비트까지 같은가(GPU 없이)"""
    exe = os.path.join(출력, "시험_라마로그확률" + EXE)
    if not os.path.exists(exe):
        알림("llama-server 로그 확률의 흉내(GPU 없이)", False, "시험_라마로그확률 이 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "라마영수증.py"), exe],
                      os.path.join(출력, "라마영수증.txt"))
    알림("llama-server 로그 확률의 흉내(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 라마고르기(출력):
    """4v — llama.cpp 재현 5단계: llama-server 의 고르기 사슬 흉내(글)가 서버 코드를 옮긴 파이썬 참조와 같은 토큰을 고르는가(GPU 없이)"""
    exe = os.path.join(출력, "시험_라마고르기" + EXE)
    if not os.path.exists(exe):
        알림("llama-server 고르기 사슬의 흉내(GPU 없이)", False, "시험_라마고르기 가 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "라마고르기.py"), exe],
                      os.path.join(출력, "라마고르기.txt"))
    알림("llama-server 고르기 사슬의 흉내(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 라마사슬(출력):
    """4w — llama.cpp 재현 7단계: llama-server 한 슬롯의 프롬프트 캐시 셈 흉내(글)가 서버 코드를 옮긴 파이썬 참조와 같은가(GPU 없이)"""
    exe = os.path.join(출력, "시험_라마사슬" + EXE)
    if not os.path.exists(exe):
        알림("llama-server 프롬프트 캐시 셈의 흉내(GPU 없이)", False, "시험_라마사슬 이 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "라마사슬.py"), exe],
                      os.path.join(출력, "라마사슬.txt"))
    알림("llama-server 프롬프트 캐시 셈의 흉내(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def GGUF쓰기(출력):
    """모델 평가 2-1 — 메타데이터만 고친 GGUF(앱/GGUF쓰기.gl)가 키 · 텐서 정보 · 자료 구역 · 맞춤에서 기대와 같은가(GPU 없이)"""
    exe = os.path.join(출력, "시험_GGUF쓰기" + EXE)
    if not os.path.exists(exe):
        알림("GGUF 다시 쓰기(GPU 없이)", False, "시험_GGUF쓰기 가 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "GGUF쓰기.py"), "check", exe],
                      os.path.join(출력, "GGUF쓰기.txt"))
    알림("GGUF 다시 쓰기(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 채점(출력):
    """모델 평가 3-1 — 정답 채점(앱/평가채점.gl)의 판정 · 부트스트랩 구간이 파이썬 참조(시험/채점.py)와 같은가, 3-2 — 내장 데이터의 정답으로
    지은 답이 맞음인가(GPU 없이)"""
    exe = os.path.join(출력, "시험_채점" + EXE)
    if not os.path.exists(exe):
        알림("정답 채점(GPU 없이)", False, "시험_채점 이 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "채점.py"), "check", exe],
                      os.path.join(출력, "채점.txt"))
    알림("정답 채점(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 조각상태(출력):
    """층 나누기(엔진/조각.gl)가 조각을 바꿀 때 담고 푸는 전역의 목록이 엔진의 전역과 맞는가 — 새 전역은 조각상태나 시험/조각상태.txt 의
    [공유] 에 있어야 한다(GPU · 빌드 없이)"""
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "조각상태.py"), "check"],
                      os.path.join(출력, "조각상태.txt"))
    알림("층 나누기의 상태 목록(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 망가진입력(출력):
    """GHSA-4v72 — 조작된 GGUF 를 엔진의 GGUF 열기가 오류로 끝내는가(접근 위반 · 멈춤 · 그냥 열림이 아니라), NaN · ±Inf 로짓에서 뽑기가
    범위 안의 번호를 고르는가(GPU 없이)"""
    for 이름, 갈래, exe이름 in (("망가진 GGUF 열기(GPU 없이)", "gguf", "시험_GGUF열기"), ("유한하지 않은 로짓의 뽑기(GPU 없이)", "뽑기", "시험_뽑기")):
        exe = os.path.join(출력, exe이름 + EXE)
        if not os.path.exists(exe):
            알림(이름, False, f"{exe이름} 이 없다(빌드 실패)")
            continue
        t0 = time.time()
        rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "망가진입력.py"), 갈래, exe],
                          os.path.join(출력, f"망가진입력_{갈래}.txt"))
        알림(이름, rc == 0, f"{꼬리.split(' / ')[-1]} ({time.time() - t0:.1f} 초)")


def 모델해시(출력):
    """#19 — 앱이 모델 파일의 SHA-256 과 함께 크기 · 수정 시각을 두고, 같은 경로의 파일이 바뀌면 옛 해시를 버리는가(GPU 없이 — 앱은 윈도만)"""
    if 리눅스:
        return
    exe = os.path.join(출력, "GeulPU" + EXE)
    if not os.path.exists(exe):
        알림("앱의 모델 해시 지문(GPU 없이)", False, "GeulPU 가 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "모델해시.py"), exe, os.path.join(출력, "모델해시자료")],
                      os.path.join(출력, "모델해시.txt"))
    알림("앱의 모델 해시 지문(GPU 없이)", rc == 0, f"{꼬리.split(' / ')[-1]} ({time.time() - t0:.1f} 초)")


def 모델모양(출력):
    """앱이 셈할 수 없는 모델 파일을 까닭과 함께 받지 않는가 — 머리만 있는 작은 GGUF 로(GPU 없이 — 앱은 윈도만)"""
    if 리눅스:
        return
    exe = os.path.join(출력, "GeulPU" + EXE)
    if not os.path.exists(exe):
        알림("앱의 모델 모양(GPU 없이)", False, "GeulPU 가 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "모델모양.py"), exe, os.path.join(출력, "모델모양자료")],
                      os.path.join(출력, "모델모양.txt"))
    알림("앱의 모델 모양(GPU 없이)", rc == 0, f"{꼬리.split(' / ')[-1]} ({time.time() - t0:.1f} 초)")
    # 2026-10-24 구조 개선 — 엔진의 모델 확인의 길(앱 옆에 서버를 두고 — 서버가 뜨지 못하는 기기면 건너뜀)
    서버 = os.path.join(출력, "서버" + EXE)
    if os.path.exists(서버):
        t1 = time.time()
        rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "모델모양.py"), exe, os.path.join(출력, "모델모양엔진자료"),
                          "--엔진", 서버], os.path.join(출력, "모델모양엔진.txt"))
        알림("앱의 모델 확인 — 엔진의 길(GPU 없이)", rc == 0, f"{꼬리.split(' / ')[-1]} ({time.time() - t1:.1f} 초)")


def 명령줄별칭(출력):
    """6b — 명령줄 옵션의 영어 별칭(#137): 빌드한 커널생성 · GeulPU(· 서버 · 실행기 — 뜨면)의 --help, 영어 이름으로 만든 모듈 소스 = 한글 이름의 것"""
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "명령줄별칭.py"), 출력, os.path.join(출력, "명령줄별칭자료")],
                      os.path.join(출력, "명령줄별칭.txt"))
    알림("명령줄 옵션의 영어 별칭(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 찾기표시(출력):
    """2e — 앱의 모델 찾기 표시(app.js 의 fdSupport)를 2026-10-30 조사의 트렌딩 200 저장소로(GPU · 인터넷 없이 — node 가 있을 때). 빌드된 서버가
    뜨면 고정 자료의 구조 목록을 서버 --구조들 과도 견준다(구조 · 받는 형식이 바뀌면 자료를 고치라고)"""
    t0 = time.time()
    명령 = [sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "찾기표시.py")]
    서버 = os.path.join(출력, "서버" + EXE)
    if os.path.exists(서버):
        명령 += ["--엔진", 서버]
    rc, 꼬리 = 돌리기(명령, os.path.join(출력, "찾기표시.txt"))
    알림("앱의 모델 찾기 표시(GPU 없이)", rc == 0, f"{꼬리} ({time.time() - t0:.1f} 초)")


def 앱누수(출력):
    """#22 — 오래 켜 둔 앱의 메모리가 늘지 않는가: /api/state 와 JSON 본문(올바른 · 잘린)을 되풀이하며 private 메모리를 잰다(GPU 없이 — 윈도만)"""
    if 리눅스:
        return
    exe = os.path.join(출력, "GeulPU" + EXE)
    if not os.path.exists(exe):
        알림("앱의 메모리 누수(GPU 없이)", False, "GeulPU 가 없다(빌드 실패)")
        return
    t0 = time.time()
    rc, 꼬리 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, "시험", "앱누수.py"), exe, os.path.join(출력, "앱누수"),
                       os.path.join(ROOT, "앱", "data", "models.json")], os.path.join(출력, "앱누수.txt"))
    알림("앱의 메모리 누수(GPU 없이)", rc == 0, f"{꼬리.split(' / ')[-1]} ({time.time() - t0:.1f} 초)")


def 가져오기검사(출력):
    """윈도: 실행기는 kernel32 · shell32 · nvcuda, 서버는 그에 ws2_32(28단계). 리눅스(29단계): 둘 다 libc.so.6 · libcuda.so.1 뿐 —
    앱의 관리자는 libc.so.6 만, 장치C · 진단은 libc.so.6 · libcuda.so.1 만."""
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
    # 36단계 — 앱의 관리자는 체계의 라이브러리만(GPU 라이브러리는 관리자가 띄우는 서버 · 도구만). 리눅스는 libc.so.6 뿐
    허용앱 = {"libc.so.6"} if 리눅스 else {"kernel32.dll", "shell32.dll", "ws2_32.dll", "advapi32.dll", "bcrypt.dll", "winhttp.dll"}
    앱 = os.path.join(출력, "GeulPU" + EXE)
    if not os.path.exists(앱):
        알림("앱 관리자의 가져오기 표", False, f"{os.path.basename(앱)} 가 없다(빌드 실패)")
        return
    dll3 = sorted({d.lower() for d, _ in 가져오기표.가져오기(앱)})
    알림("앱 관리자의 가져오기 표", set(dll3) <= 허용앱, f"{dll3}")
    if 리눅스:
        # 앱이 띄우는 장치 확인 · 진단도 리눅스의 라이브러리만(윈도 이름 nvcuda.dll 을 가져오면 뜨지 않는다)
        for 이름 in ("장치C", "진단"):
            도구 = os.path.join(출력, 이름)
            if not os.path.exists(도구):
                알림(f"{이름} 의 가져오기 표", False, f"{이름} 가 없다(빌드 실패)")
                continue
            dll4 = sorted({d.lower() for d, _ in 가져오기표.가져오기(도구)})
            알림(f"{이름} 의 가져오기 표", set(dll4) <= 허용서버, f"{dll4}")
        # #135 — Vulkan 판(server-vulkan · device-vulkan · diag-vulkan)은 libc.so.6 · libvulkan.so.1 만(윈도 이름 vulkan-1.dll 이면 뜨지 않는다)
        허용벌칸 = {"libc.so.6", "libvulkan.so.1"}
        for 이름 in ("서버V", "장치V", "진단V"):
            도구 = os.path.join(출력, 이름)
            if not os.path.exists(도구):
                알림(f"{이름} 의 가져오기 표", False, f"{이름} 가 없다(빌드 실패)")
                continue
            dll5 = sorted({d.lower() for d, _ in 가져오기표.가져오기(도구)})
            알림(f"{이름} 의 가져오기 표", set(dll5) <= 허용벌칸 and "libvulkan.so.1" in dll5, f"{dll5}")


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
    p.add_argument("--기준커밋", default="", help="그 커밋과 견주어 바뀐 모듈만 PTX · SPIR-V · SASS 까지 본다(뒷단이 같을 때 — CI)")
    p.add_argument("--어휘", default="", help="llama.cpp 의 어휘 시험 파일 폴더(시험/어휘받기.py) — 주면 토크나이저를 견준다")
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
    디버거도구(출력)
    파이썬API(출력)
    화면점검(출력)
    기능목록(출력)
    실험기록(출력)
    기록비교(출력)
    개발도구(출력)
    판표()
    벌칸선언짝()
    도구 = [os.path.join(글, "build", "self_컴파일러" + EXE)]
    없는것 = [x for x in 도구 if not os.path.exists(x)]
    if 없는것:
        알림("글 컴파일러", False, f"없다: {없는것} — 문서/개발.md 의 '소스에서 빌드하기' 2")
    else:
        print("글 컴파일러: " + " · ".join(f"{os.path.basename(x)} {해시(x)[:16]}" for x in 도구), flush=True)
        if 빌드(글, 출력, a.병렬):
            저장소CPU(출력)
            로프표(출력)
            디버거제이슨(출력)
            틀해석(출력)
            젬마도구(출력)
            도구오류(출력)
            어휘시험(출력, os.path.abspath(a.어휘) if a.어휘 else "")
            요청지킴(출력)
            망가진입력(출력)
            모델해시(출력)
            파이썬API앱(출력)
            명령줄(출력)
            모델모양(출력)
            찾기표시(출력)
            명령줄별칭(출력)
            앱누수(출력)
            계약대조(출력)
            평가계산(출력)
            GGUF쓰기(출력)
            채점(출력)
            조각상태(출력)
            라마로그확률(출력)
            라마고르기(출력)
            라마사슬(출력)
            건너뛴 = set()
            if a.모듈없이:
                print("모듈 · SASS: 건너뜀(--모듈없이 — 커널 · 생성기 · 글 고정 커밋이 그대로)", flush=True)
                요약.append("건너뜀  [모듈 · SASS] --모듈없이")
            else:
                건너뛴 = 모듈점검.모듈(글, 출력, a.병렬, a.해시갱신, 알림, a.기준커밋)
            가져오기검사(출력)
            if a.모듈없이:
                pass
            elif a.ptxas:
                모듈점검.SASS(출력, a.ptxas, a.병렬, a.넘침갱신, 알림, 건너뛴)
            else:
                print("SASS: ptxas 를 주지 않아 건너뜀(--ptxas)", flush=True)
    open(os.path.join(출력, "요약.txt"), "w", encoding="utf-8").write("\n".join(요약) + "\n")
    print(f"빠른 점검: {'모두 통과' if not 실패 else '못 미침 ' + str(실패)} ({time.time() - t0:.0f} 초)", flush=True)
    return 0 if not 실패 else 1


if __name__ == "__main__":
    sys.exit(main())
