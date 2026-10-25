#!/usr/bin/env python3
"""0.12.0 12-G3 — 자동 모델 검증: GGUF 파일 하나를 명령 하나로 이 PC 에서 시험해 판정 표와 앱 모델 목록 항목의 초안을 낸다(문서/모델검증.md).
사람은 '못 미침' · '건너뜀' 과 목록 초안의 빈칸(모델 설명 · 저장소)만 본다. 모델 파일은 받지 않는다 — 받기는 늘 사용자의 허락 뒤에.

  python -X utf8 재기/모델검증.py <모델.gguf> --work <폴더 — SSD> --app <패키지한 앱 폴더(앱/꾸리기.ps1 -NoZip)>
      --tokenizer-llama <llama.cpp 폴더 — llama-tokenize(최신 공식 버전)> --quality-llama <llama.cpp 폴더 — 품질 비교의 기준 b11719>
      [--stages header,bits,…] [--wiki <wikitext-2 의 wiki.test.raw>] [--repo <Hugging Face 저장소 — 예: Qwen/Qwen3-4B-GGUF>]
      [--port 44651] [--app-data <앱의 임시 데이터 폴더 — 기본 <work>/app-data>] [--geul <글(geul) 저장소>]

단계(--stages, 기본은 모두 — 줄 것이 없는 단계는 '건너뜀'):
  header    엔진의 모델 확인(이 저장소의 서버 --모델확인 — 엔진/모델확인.gl: 받는가 · 모듈 이름 · 커널 생성기 인자 · 정밀만 · 8비트 KV ·
            빠른 어텐션 — 2026-10-25, 엔진이 받는 구조 모두), GGUF 의 메타데이터와 패키지한 앱의 모델 확인(/api/state 의 shape — 앱이
            받지 않으면 여기서 멈춘다): 구조 · 형식 · 모양 · 토크나이저 · 템플릿, 앱의 기본 모드(빠른 모드 — 정밀만인 구조와 F16 · BF16 ·
            F32 파일은 정밀 — 앱의 precision_unquantized)
  bits      앱이 이 모델에 쓰는 CUDA 모듈마다(정밀 --먼 · 빠른 모드, 허용되면 8비트 KV · 빠른 어텐션) 시험_7B 의 다섯 실행 경로.
            모델 파일이 6 GB 이하면 앱의 기본 모듈과 다른 끝(기본이 빠르면 정밀, 정밀이면 가장 빠른 모듈)으로 시험_긴문맥(문맥 32,768 ·
            기본은 --far 0,0 도) · 시험_추측
  ref       CPU 참조(시험/참조7B.py · Qwen3.5 · MoE 는 시험/참조35.py · Gemma 4 는 시험/참조gemma4.py — 시험_7B 의 모든 행 로짓과
            불일치 0): bits 의 모듈마다(6 GB 넘는 모델은 앱의 기본만), 넷까지 함께. 입력7B.py 의 토크나이저가 모르는 구조는 정한 무작위
            토큰열로(시험_긴문맥 · 시험_추측 은 qwen2 · qwen3 · llama 만)
  quality   llama.cpp(f16 KV) 대비 다음 토큰 KL · 1위 일치(시험/품질.py — 재기/품질2b.sh 와 같은 텍스트 · 같은 기준): 앱의 기본 모드가
            판정 B′(텍스트마다 평균 KL 이 llama.cpp 의 2.5 배 이하 · 1위 일치가 0.5 %p 넘게 낮지 않음)를 넘는가. 다른 모드는 참고.
            작업 폴더의 디스크에 50 GB 가 비어 있어야 한다(로짓)
  template  서버의 채팅 템플릿(GPU 없이 — 엔진/시험_채팅틀) = jinja2(WSL 의 Ubuntu-24.04 — 시험/채팅틀대조.py)
  tokenizer 토크나이저(GPU 없이 — 엔진/시험_토큰) = llama.cpp 의 llama-tokenize(특이한 텍스트 열 · 문서 다섯 — 특수 토큰 · BOS ·
            escape 없이)
  app       패키지한 앱(임시 데이터 폴더 · --port)에서: 앱이 고른 모드 = 판정한 모드, 서버의 /tokenize = jinja2(채팅 · 도구 사례 —
            시험/도구틀.py), 채팅 둘(한국어 · 영어), 템플릿에 도구가 있으면 도구 호출(시험/도구호출.py — 생각을 켜고 끄는 템플릿이면 둘 다)
  vulkan    같은 앱의 GPU 백엔드를 Vulkan 으로(늘 정밀 모드) 바꿔 불러와 채팅 둘
  catalog   앱 모델 목록(앱/data/models.json) 항목과 영어 화면의 번역 줄(앱/web/app.js 의 DICT)의 초안 — 확인한 것만 적는다
결과: <work>/결과.json · 결과.md · 목록초안.json · 기록.txt, 끝 줄 '모델 검증: 모두 통과' 또는 '모델 검증: 못 미침 N'(종료 코드 1).
"""
import argparse
import hashlib
import json
import mmap
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
EXE = ".exe" if os.name == "nt" else ""
단계들 = ["header", "bits", "ref", "quality", "template", "tokenizer", "app", "vulkan", "catalog"]
문서들 = ["사용안내.md", "README.ko.md", "docs/user-guide.md", "연구자안내.md", "CHANGELOG.md"]
특이한텍스트 = [                   # 12-1 의 토큰 비교와 같은 텍스트 — 숫자 · 공백 · 줄바꿈 · CJK · 이모지 · 따옴표
    "안녕하세요, 반갑습니다! 오늘은 2026년 10월 23일입니다.",
    "I'm sure we'll see they've done it; don't you think? It's 3:45pm.",
    "1234567 12 3 0.5 1,000,000 3.14159 007 x2y34z567",
    "  앞 공백   여러 칸\n\n\n  줄 앞 공백\t탭\t\t끝  ",
    "def f(x):\n    return x ** 2  # 제곱\n\nprint(f(12345))\n",
    "😀 emoji 👍🏽 and 中文字符 与 日本語のテキスト、カタカナ。",
    "abc123def 456ghi  789\r\n윈도 줄바꿈\r\n",
    "https://example.com/path?q=1&b=%20x — em dash · middle dot … ellipsis",
    " 123  456   789\n 7",
    "한국어 문장이 길게 이어질 때: 엔진은 같은 바이트면 같은 토큰을 내야 하고, 숫자 2,048 과 8,192 와 32k 같은 표기도 흔하다. "
    "또 '따옴표' \"큰따옴표\" (괄호) [대괄호] {중괄호} <꺾쇠> 도 섞인다.",
]
질문들 = ["대한민국의 수도는 어디야? 한 문장으로 답해.", "Write one sentence about the sea."]
큰모델 = 6 * 1024 ** 3              # 이보다 큰 파일: 문맥 32,768 의 비트 시험을 건너뛰고(12 GB 에 들지 않는다) 품질은 청크 128 행
결과들 = []
작업 = ""


def 적기(글):
    줄 = f"[{time.strftime('%H:%M:%S')}] {글}"
    print(줄, flush=True)
    with open(os.path.join(작업, "기록.txt"), "a", encoding="utf-8") as f:
        f.write(줄 + "\n")


def 판(단계, 이름, 통과, 글="", 참고=False):
    """통과: 참 · 거짓 · 없음(건너뜀). 참고면 거짓이어도 못 미침으로 세지 않는다(앱의 기본이 아닌 모드의 품질)"""
    결과들.append({"단계": 단계, "이름": 이름, "통과": 통과, "참고": 참고, "글": 글})
    적기(f"{판글(통과, 참고)}  [{단계}] {이름}{(' — ' + 글) if 글 else ''}")


def 판글(통과, 참고):
    if 참고:
        return "참고(통과)" if 통과 else "참고(못 미침)"
    return {True: "통과", False: "못 미침", None: "건너뜀"}[통과]


def 꼬리(경로):
    try:
        줄들 = open(경로, encoding="utf-8", errors="replace").read().strip().splitlines()
    except OSError:
        return ""
    return 줄들[-1] if 줄들 else ""


def 돌리기(cmd, 기록, cwd=None, env=None):
    with open(기록, "wb") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=cwd, env=env)
    return r.returncode, 꼬리(기록)


def 파이썬(스크립트, *인자):
    return [sys.executable, "-I", "-X", "utf8", os.path.join(ROOT, *스크립트.split("/"))] + list(인자)


def wsl경로(p):
    p = os.path.abspath(p).replace("\\", "/")
    return f"/mnt/{p[0].lower()}{p[2:]}" if len(p) > 1 and p[1] == ":" else p


def GPU비었나():
    """다른 작업이 GPU 메모리를 쓰고 있으면(1.5 GB 넘게) 멈춘다 — 사용자의 앱 · 다른 시험과 겹쳐 메모리가 모자라지 않게"""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return int(r.stdout.split()[0]) < 1500
    except (OSError, ValueError, IndexError):
        return True


def GPU이름():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
        return r.stdout.strip().splitlines()[0].replace("NVIDIA ", "").replace("GeForce ", "").strip()
    except (OSError, IndexError):
        return "<GPU>"


# ---- GGUF 헤더 ----
def gguf헤더(경로):
    """메타데이터(배열은 읽지 않는다)와 텐서(형식, 모양)"""
    f = open(경로, "rb")
    b = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    if b[:4] != b"GGUF":
        raise SystemExit(f"GGUF 파일이 아니다: {경로}")
    _, nt, nkv = struct.unpack_from("<IQQ", b, 4)
    o = 24
    fm = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def 글(o):
        n = struct.unpack_from("<Q", b, o)[0]
        return b[o + 8:o + 8 + n].decode("utf-8", "replace"), o + 8 + n

    def 값(o, t):
        if t in fm:
            return struct.unpack_from(fm[t], b, o)[0], o + struct.calcsize(fm[t])
        if t == 8:
            return 글(o)
        et, n = struct.unpack_from("<IQ", b, o)
        o += 12
        if et in fm:
            return None, o + n * struct.calcsize(fm[et])
        for _ in range(n):
            _, o = 값(o, et)
        return None, o
    meta = {}
    for _ in range(nkv):
        k, o = 글(o)
        t = struct.unpack_from("<I", b, o)[0]
        meta[k], o = 값(o + 4, t)
    텐서 = {}
    for _ in range(nt):
        이름, o = 글(o)
        nd = struct.unpack_from("<I", b, o)[0]
        d = struct.unpack_from(f"<{nd}Q", b, o + 4)
        텐서[이름] = (struct.unpack_from("<I", b, o + 4 + 8 * nd)[0], list(d))
        o += 4 + 8 * nd + 12
    b.close()
    f.close()
    return meta, 텐서


def 모델모양(meta, 텐서):
    """GGUF 만으로 읽은 모양 — 앱의 확인(모델보기.gl)이 있으면 단계헤더가 그것과 비교하고 KV8 허용 · 매개변수 수는 앱의 값을 쓴다"""
    구조 = meta.get("general.architecture", "?")
    키 = lambda x, d=0: meta.get(f"{구조}.{x}", d)       # noqa: E731
    q형 = 텐서.get("blk.0.attn_q.weight", (-1, []))[0]
    형식글 = {8: "Q8_0", 2: "Q4_0"}.get(q형)
    if q형 in (12, 13, 14, 23):
        형식글 = "KI" if any(t == 23 for n, (t, _) in 텐서.items() if n.startswith("blk.")) else "K"
    if q형 in (0, 1, 30):                        # 0.12.0 12-3d-1 — F32 · F16 · BF16: 모듈 W16(16비트 블록 가중치 — 계약 1.3)
        형식글 = "W16"
    템플릿 = meta.get("tokenizer.chat_template") or ""
    매개 = 0
    for _, d in 텐서.values():
        n = 1
        for x in d:
            n *= x
        매개 += n
    return {"구조": 구조, "이름": meta.get("general.name") or "", "크기표": meta.get("general.size_label") or "",
            "파일형식": meta.get("general.file_type"), "라이선스": meta.get("general.license") or "",
            "너비": 키("embedding_length"), "헤드": 키("attention.head_count"), "KV헤드": 키("attention.head_count_kv"),
            "확장폭": 키("feed_forward_length"), "어휘": 텐서.get("token_embd.weight", (0, [0, 0]))[1][-1],
            "eps": float(키("attention.layer_norm_rms_epsilon", 0.0) or 0.0), "형식글": 형식글 or f"?({q형})", "매개변수": 매개,
            "헤드크기": 키("attention.key_length", 0), "RoPE": 키("rope.dimension_count", 0),
            "토큰모델": meta.get("tokenizer.ggml.model") or "", "토큰앞": meta.get("tokenizer.ggml.pre") or "",
            "템플릿": bool(템플릿), "템플릿도구": "tools" in 템플릿, "생각켬끔": "enable_thinking" in 템플릿}


def 엔진확인(서버, gguf):
    """엔진의 모델 확인(서버 --모델확인 — 엔진/모델확인.gl): (JSON, None) 또는 (None, 엔진의 까닭)"""
    r = subprocess.run([서버, "--모델확인", gguf], capture_output=True, stdin=subprocess.DEVNULL, timeout=600)
    글 = (r.stdout + r.stderr).decode("utf-8", "replace").strip()
    if r.returncode == 0:
        for 줄 in reversed(r.stdout.decode("utf-8", "replace").splitlines()):
            if 줄.startswith("{"):
                return json.loads(줄), None
    끝줄 = 글.splitlines()[-1] if 글 else f"종료 코드 {r.returncode}"
    return None, 끝줄


def 엔진규칙(모, 확인):
    """2026-10-25 — 모듈 이름 · 커널 생성기 인자 · 할 수 있는 것은 엔진의 모델 확인(앱/일.gl 의 모듈준비 와 같은 것 — 엔진이 받는 구조
    모두: qwen2 · qwen3 · llama · qwen35 · qwen35moe · gemma4 …)"""
    모["모듈"] = 확인["module"]
    모["형식글"] = 확인.get("quant") or 모["형식글"]
    모["생성기인자"] = [str(x) for x in 확인["gen"]]
    모["머리옵션"] = [str(x) for x in 확인.get("gen_opts") or []]          # 그 구조의 깃발(--헤드 · --RoPE · --MoE …)
    모["정밀만"] = bool(확인.get("exact_only"))
    모["먼됨"] = bool(확인.get("far_ok", True))
    모["KV8"] = bool(확인.get("kv8_ok")) and not 모["정밀만"]
    모["빠른어텐션"] = ("F2B" if 확인.get("f2b") else "F2") if 모["KV8"] and 확인.get("fast_attention_ok") else None


def 변형들(모):
    """앱이 이 모델에 쓰는 CUDA 모듈(앱/일.gl 의 모듈준비 — 정밀은 --먼, 그 구조가 받으면): (이름, 모듈 이름, 생성기 옵션, 참조 옵션).
    처음이 정밀, 마지막이 가장 빠른 모듈 — 앱의 기본은 기본변형. 정밀만인 구조는 정밀 하나"""
    m = 모["모듈"]
    h = 모.get("머리옵션", [])
    먼 = ["--먼"] if 모.get("먼됨", True) else []
    out = [("정밀", m + ("_먼" if 먼 else ""), 먼 + h, [])]
    if 모.get("정밀만"):
        return out
    out.append(("F1", m + "_빠른", ["--빠른"] + h, ["--빠른"]))
    if 모["KV8"]:
        fa = 모["빠른어텐션"]
        out.append(("KV8", m + "_빠른_KV8", ["--빠른", "--KV8"] + h, ["--KV8"]))
        out.append((f"{fa}_KV8", f"{m}_빠른_{fa}_KV8", ["--빠른", "--KV8", f"--{fa}"] + h, [f"--{fa}"]))
    return out


def 기본변형(모):
    """앱의 기본 모드(앱/일.gl — 모델계산설정): F16 · BF16 · F32 파일(W16)과 정밀만인 구조는 정밀(설정 precision_unquantized 의 기본
    exact — 0.12.0), 그 밖에는 가장 빠른 모듈(설정 precision 의 기본 fast)"""
    v = 변형들(모)
    return v[0] if 모["형식글"] == "W16" or 모.get("정밀만") else v[-1]


표준구조 = ("qwen2", "qwen3", "llama")               # 입력7B.py 의 토크나이저 · 시험_긴문맥 · 시험_추측 · 참조7B.py 가 아는 구조


# ---- 빌드 · 모듈 ----
도구들 = {"kgen": "커널/커널생성.gl", "ptxb": "백엔드/PTX/PTX.gl", "t7b": "엔진/시험_7B.gl", "tlong": "엔진/시험_긴문맥.gl",
        "tspec": "엔진/시험_추측.gl", "tchat": "엔진/시험_채팅틀.gl", "ttok": "엔진/시험_토큰.gl", "srv": "엔진/서버.gl"}
단계도구 = {"bits": ["kgen", "ptxb", "t7b", "tlong", "tspec"], "quality": ["kgen", "ptxb", "t7b"], "template": ["tchat"], "tokenizer": ["ttok"]}


def 빌드(글, bin폴더, 이름들):
    """이 저장소의 소스를 글(geul) 컴파일러로 — 이름들(도구들의 키)만"""
    컴 = os.path.join(글, "build", "self_컴파일러" + EXE)
    env = dict(os.environ, GEUL_ROOT=글)
    os.makedirs(bin폴더, exist_ok=True)
    틀림 = []
    for 이름, 소스 in 도구들.items():
        if 이름 not in 이름들:
            continue
        rc, 끝 = 돌리기([컴, 소스, "-o", os.path.join(bin폴더, 이름 + EXE)], os.path.join(bin폴더, f"빌드_{이름}.txt"), cwd=ROOT, env=env)
        if rc != 0:
            틀림.append(f"{이름}: {끝}")
    return 틀림


def 모듈짓기(bin폴더, 모듈폴더, 이름, 인자, 옵션):
    p = os.path.join(모듈폴더, 이름 + ".ptx")
    if os.path.exists(p):
        return p
    g = os.path.join(모듈폴더, 이름 + ".gl")
    rc, 끝 = 돌리기([os.path.join(bin폴더, "kgen" + EXE)] + 인자 + [g] + 옵션, os.path.join(모듈폴더, f"생성_{이름}.txt"))
    if rc == 0:
        rc, 끝 = 돌리기([os.path.join(bin폴더, "ptxb" + EXE), g, "-o", p], os.path.join(모듈폴더, f"ptx_{이름}.txt"))
    if os.path.exists(g):
        os.remove(g)
    if rc != 0 or not os.path.exists(p):
        raise RuntimeError(f"모듈 {이름}: {끝}")
    return p


# ---- 패키지한 앱 ----
def 같은경로(a, b):
    return os.path.normcase(os.path.normpath(a or "")) == os.path.normcase(os.path.normpath(b or ""))


class 앱:
    """앱/꾸리기.ps1 로 패키지한 앱을 임시 데이터 폴더 · 다른 포트로 켠다(사용자의 앱 · 데이터와 겹치지 않게)"""

    def __init__(self, 폴더, 데이터, 포트):
        self.주소 = f"http://127.0.0.1:{포트}"
        os.makedirs(데이터, exist_ok=True)
        self.기록 = open(os.path.join(작업, "앱.log"), "ab")
        self.p = subprocess.Popen([os.path.join(폴더, "GeulPU" + EXE), "--포트", str(포트), "--자료", 데이터, "--열지않음"],
                                  stdout=self.기록, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        for _ in range(240):
            try:
                self.묻기("/api/ping")
                break
            except OSError:
                if self.p.poll() is not None:
                    raise RuntimeError(f"앱이 끝났다(종료 코드 {self.p.returncode}) — {os.path.join(작업, '앱.log')}")
                time.sleep(0.5)
        self.장치확인()

    def 묻기(self, 길, 몸=None, 시간=1800):
        r = urllib.request.Request(self.주소 + 길, data=None if 몸 is None else json.dumps(몸, ensure_ascii=False).encode("utf-8"),
                                   method="GET" if 몸 is None else "POST", headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(r, timeout=시간).read() or b"null")

    def 작업기다림(self, 번):
        while True:
            time.sleep(1)
            j = self.묻기("/api/state").get("job") or {}
            if j.get("id") == 번 and j.get("state") != "running":
                return j

    def 장치확인(self, 다시=False):
        """앱이 켜질 때 스스로 띄운 장치 확인과 겹치면(409) 그 작업이 끝나기를 기다린다. 다시: 백엔드를 바꾼 뒤"""
        for _ in range(900):
            st = self.묻기("/api/state")
            if not 다시 and st["device"].get("checked") and (st.get("job") or {}).get("state") != "running":
                return
            try:
                self.작업기다림(self.묻기("/api/device/check", {})["job"])
                return
            except urllib.error.HTTPError as e:
                if e.code != 409:
                    raise
            time.sleep(1)

    def 모델(self, 경로):
        m = next((x for x in self.묻기("/api/state")["models"] if 같은경로(x.get("path"), 경로)), None)
        if not m:
            self.묻기("/api/models/add", {"path": 경로})
            m = next(x for x in self.묻기("/api/state")["models"] if 같은경로(x.get("path"), 경로))
        return m

    def 멈춤(self):
        if self.묻기("/api/state")["server"].get("state") == "ready":
            self.묻기("/api/server/stop", {})
            while self.묻기("/api/state")["server"].get("state") not in ("off", "failed"):
                time.sleep(0.5)

    def 올리기(self, 경로):
        m = self.모델(경로)
        self.멈춤()
        j = self.묻기("/api/server/start", {"model": m["id"]})
        t0 = time.time()
        while time.time() - t0 < 3600:                  # 처음에는 모듈을 짓는다(커널 생성기 · 백엔드 — 몇 분)
            time.sleep(1)
            st = self.묻기("/api/state")
            sv, job = st["server"], st.get("job") or {}
            if sv.get("state") == "ready" and job.get("id") == j.get("job") and job.get("state") == "done":
                return sv
            if sv.get("state") == "failed" or (job.get("id") == j.get("job") and job.get("state") == "failed"):
                return {"failed": f"{job.get('reason') or ''} {sv.get('reason') or ''}".strip()}
        return {"failed": "불러오기가 한 시간을 넘었다"}

    def 채팅(self):
        답 = []
        for q in 질문들:
            a = self.묻기("/v1/chat/completions", {"model": "geulpu", "messages": [{"role": "user", "content": q}], "max_tokens": 160,
                                                 "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
            답.append((a["choices"][0]["message"]["content"] or "").strip())
        return 답

    def 끄기(self):
        try:
            self.멈춤()
        except OSError:
            pass
        self.p.terminate()
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.기록.close()


# ---- 단계 ----
def 단계헤더(gguf, 모, 앱하나, 서버):
    if 서버 is None:
        판("header", "엔진의 모델 확인", False, "서버(엔진/서버.gl)를 빌드하지 못했다")
        return False
    확인, 까닭 = 엔진확인(서버, gguf)
    if 확인 is None:
        판("header", "엔진의 모델 확인", False, f"엔진이 받지 않는다 — {까닭}")
        return False
    엔진규칙(모, 확인)
    판("header", "엔진의 모델 확인", True, f"{확인.get('arch')} · {확인.get('quant')} · 모듈 {확인.get('module')}"
       f"{' · 정밀만' if 모['정밀만'] else ''}{' · Vulkan 못 함 — ' + 확인['vulkan_reason'] if 확인.get('vulkan_reason') else ''}")
    if 앱하나 is None:
        판("header", "앱의 모델 확인", None, "--app 을 주지 않았다 — 엔진의 확인과 GGUF 에서 읽은 모양으로 시험한다")
    else:
        sh = 앱하나.모델(gguf).get("shape") or {}
        if not sh.get("ok"):
            판("header", "앱의 모델 확인", False, f"앱이 받지 않는다 — {sh.get('reason')}")
            return False
        # 도구가 GGUF 에서 읽지 못한 값(층마다의 배열 — Gemma 4 의 KV 헤드 등)은 견주지 않는다
        다름 = [f"{이름} 도구 {모[키]} · 앱 {sh.get(앱키)}" for 키, 앱키, 이름 in (("형식글", "quant", "형식"), ("너비", "width", "너비"),
                ("헤드", "heads", "헤드"), ("KV헤드", "kv_heads", "KV 헤드"), ("확장폭", "ffn", "FFN"), ("어휘", "vocab", "어휘"))
                if 모[키] is not None and 모[키] != sh.get(앱키)]
        모["매개변수"] = sh.get("params") or 모["매개변수"]
        판("header", "앱의 모델 확인", not 다름, "; ".join(다름) or f"받음 — 사전 처리 {sh.get('pre') or '-'} · 문맥 {sh.get('context')}")
        if 다름:
            return False
    기본 = 기본변형(모)[0]
    판("header", "모델 모양", True, f"{모['구조']} · 너비 {모['너비']} · 헤드 {모['헤드']}/{모['KV헤드']} · FFN {모['확장폭']} · 어휘 {모['어휘']} · "
       f"{모['형식글']} · 토크나이저 {모['토큰모델']}/{모['토큰앞'] or '-'} · 템플릿 {'있음' if 모['템플릿'] else '없음(기본 ChatML)'}"
       f"{' · 도구' if 모['템플릿도구'] else ''}{' · 생각 켬 · 끔' if 모['생각켬끔'] else ''} · 모듈 {모['모듈']} · 앱의 기본 모드 {기본}")
    return True


def 단계비트(gguf, 모, bin폴더, 모듈폴더, 출력, 라마):
    """돌려줌: CPU 참조와 비교할 시험_7B 출력 — (이름, 폴더, 참조7B 옵션)"""
    라마토큰 = []
    if 모["구조"] == "llama":                     # 입력7B.py — 구조 llama 의 토큰은 llama.cpp 의 llama-tokenize 로
        if not 라마:
            판("bits", "입력", None, "구조 llama 는 --tokenizer-llama(llama-tokenize)가 있어야 입력을 만든다")
            return []
        라마토큰 = ["--라마토큰", os.path.join(라마, "llama-tokenize" + EXE)]
    elif 모["구조"] not in 표준구조:              # 2026-10-25 — 입력7B.py 의 토크나이저가 모르는 구조: 정한 무작위 토큰열(같은 길이 — 비트 대조에는 충분)
        라마토큰 = ["--무작위", "24"]
    입력 = os.path.join(출력, "입력.bin")
    rc, 끝 = 돌리기(파이썬("시험/입력7B.py", gguf, 입력, *라마토큰), os.path.join(출력, "입력.txt"))
    if rc != 0:
        판("bits", "입력", False, 끝)
        return []
    길 = []
    for 이름, 모듈, 옵션, 참조옵션 in 변형들(모):
        p = 모듈짓기(bin폴더, 모듈폴더, 모듈, 모["생성기인자"], 옵션)
        o = os.path.join(출력, f"e2e_{이름}")
        os.makedirs(o, exist_ok=True)
        if not GPU비었나():
            판("bits", f"시험_7B {이름}", None, "GPU 메모리를 다른 작업이 쓰고 있다")
            continue
        rc, 끝 = 돌리기([os.path.join(bin폴더, "t7b" + EXE), gguf, p, 입력, o], os.path.join(출력, f"e2e_{이름}.txt"))
        된 = rc == 0 and "모두 같음 1" in 끝 and "오류칸 0" in 끝
        판("bits", f"시험_7B {이름}(다섯 실행 경로)", 된, 끝)
        if 된:
            길.append((이름, o, 참조옵션))
    if os.path.getsize(gguf) > 큰모델:
        판("bits", "시험_긴문맥 · 시험_추측", None, "모델 파일이 6 GB 를 넘는다 — 문맥 32,768 이 12 GB 에 들지 않는다")
        return 길
    if 모["구조"] not in 표준구조:
        판("bits", "시험_긴문맥 · 시험_추측", None, f"구조 {모['구조']} 는 시험_긴문맥 · 시험_추측 이 아직 받지 않는다(그 구조의 시험으로)")
        return 길
    재기 = os.path.join(출력, "재기입력.bin")
    rc, 끝 = 돌리기(파이썬("시험/입력7B.py", gguf, 재기, "--재기", *라마토큰), os.path.join(출력, "재기입력.txt"))
    if rc != 0:
        판("bits", "측정 입력", False, 끝)
        return 길
    b = open(재기, "rb").read()
    o, 열 = 0, []
    while o < len(b):
        n = struct.unpack_from("<q", b, o)[0]
        열.append(list(struct.unpack_from(f"<{n}q", b, o + 8)))
        o += 8 + 8 * n
    긴 = os.path.join(출력, "긴입력.bin")
    with open(긴, "wb") as f:                     # 측정 입력의 긴 텍스트(8,192)를 되풀이한 73,728 토큰 — 앞의 채팅 하나와 함께(12-2 와 같다)
        for s in (열[0], (열[1] * 9)[:73728]):
            f.write(struct.pack("<q", len(s)) + struct.pack(f"<{len(s)}q", *s))
    기본 = 기본변형(모)
    다른 = 변형들(모)[-1] if 기본[0] == "정밀" else 변형들(모)[0]       # 다른 끝 — 기본이 정밀이면 가장 빠른 모듈
    p기본 = 모듈짓기(bin폴더, 모듈폴더, 기본[1], 모["생성기인자"], 기본[2])
    p다른 = 모듈짓기(bin폴더, 모듈폴더, 다른[1], 모["생성기인자"], 다른[2])
    for 이름, 인자 in ((f"시험_긴문맥 {기본[0]}", [p기본, 긴, "32768", "1000,4100,8190,16384,32760"]),
                     (f"시험_긴문맥 {기본[0]} --far 0,0", [p기본, 긴, "32768", "4100,16384,32760", "--far", "0,0"]),
                     (f"시험_긴문맥 {다른[0]}", [p다른, 긴, "32768", "1000,8190,32760"])):
        if not GPU비었나():
            판("bits", 이름, None, "GPU 메모리를 다른 작업이 쓰고 있다")
            continue
        rc, 끝 = 돌리기([os.path.join(bin폴더, "tlong" + EXE), gguf] + 인자, os.path.join(출력, 이름.replace(" ", "_") + ".txt"))
        판("bits", f"{이름}(문맥 32,768)", rc == 0 and "다른 경우 0" in 끝 and "오류칸 0" in 끝, 끝)
    for 이름, p in ((f"시험_추측 {기본[0]}", p기본), (f"시험_추측 {다른[0]}", p다른)):
        if not GPU비었나():
            판("bits", 이름, None, "GPU 메모리를 다른 작업이 쓰고 있다")
            continue
        rc, 끝 = 돌리기([os.path.join(bin폴더, "tspec" + EXE), gguf, p, 재기], os.path.join(출력, 이름.replace(" ", "_") + ".txt"))
        판("bits", 이름, rc == 0 and "다른 칸 0" in 끝 and "다른 번호 0" in 끝 and "오류칸 0" in 끝, 끝)
    return 길


def 참조시작(gguf, 모, 길, 출력):
    """시험_7B 의 출력마다 CPU 참조(넷까지 함께) — 끝에서 참조기다림"""
    if os.path.getsize(gguf) > 큰모델:            # 큰 모델은 앱의 기본만(하나에 한 시간 가까이)
        길 = [x for x in 길 if x[0] == 기본변형(모)[0]]
    일들 = []
    for 이름, o, 옵션 in 길:
        while sum(p.poll() is None for _, p, _ in 일들) >= 4:
            time.sleep(5)
        결과 = os.path.join(출력, f"ref_{이름}.json")
        if 모["구조"] in 표준구조:
            cmd = 파이썬("시험/참조7B.py", gguf, o, 결과, *옵션)
        elif 모["구조"] in ("qwen35", "qwen35moe"):    # 2026-10-25 — 구조의 CPU 참조(같은 '모두 0: True' 꼴)
            cmd = 파이썬("시험/참조35.py", gguf, "--엔진", o, 결과, *[x for x in 옵션 if x == "--빠른"])
        elif 모["구조"] == "gemma4":
            cmd = 파이썬("시험/참조gemma4.py", gguf, "--엔진", o, 결과, "--입력", os.path.join(출력, "입력.bin"))
        else:
            판("ref", f"CPU 참조 {이름}", None, f"구조 {모['구조']} 의 CPU 참조를 이 도구가 아직 모른다")
            continue
        f = open(os.path.join(출력, f"ref_{이름}.txt"), "wb")
        일들.append((이름, subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=ROOT), f))
    return 일들


def 참조기다림(일들, 출력):
    for 이름, p, f in 일들:
        rc = p.wait()
        f.close()
        끝 = 꼬리(os.path.join(출력, f"ref_{이름}.txt"))
        판("ref", f"CPU 참조 {이름}(모든 행 로짓)", rc == 0 and "모두 0: True" in 끝, 끝)


def 단계품질(gguf, 모, bin폴더, 모듈폴더, 출력, 품질라마, 위키):
    d = os.path.join(출력, "품질")
    try:
        품질측정(gguf, 모, bin폴더, 모듈폴더, 출력, 품질라마, 위키, d)
    finally:
        shutil.rmtree(d, ignore_errors=True)        # 로짓(수십 GB)


def 품질측정(gguf, 모, bin폴더, 모듈폴더, 출력, 품질라마, 위키, d):
    if 모.get("정밀만"):
        판("quality", "llama.cpp 대비", None, "정밀 모드만인 구조 — 빠른 모드의 품질 판정이 없다")
        return
    if not os.path.exists(위키):
        판("quality", "llama.cpp 대비", None, f"wikitext-2 가 없다({위키} — --wiki)")
        return
    if shutil.disk_usage(출력).free < 50 * 1000 ** 3:
        판("quality", "llama.cpp 대비", None, "작업 폴더의 디스크에 50 GB 가 비어 있지 않다(로짓)")
        return
    if not GPU비었나():
        판("quality", "llama.cpp 대비", None, "GPU 메모리를 다른 작업이 쓰고 있다")
        return
    라 = os.path.join(d, "라마")
    os.makedirs(라, exist_ok=True)
    rc, 끝 = 돌리기(파이썬("시험/품질.py", "입력", gguf, 품질라마, 라, "--글", 위키, "--글", 위키 + "@400000", "--글", 위키 + "@800000",
                         "--글", os.path.join(ROOT, "연구자안내.md"), "--긴", 위키 + "@1100000"), os.path.join(출력, "품질_입력.txt"))
    if rc != 0:
        판("quality", "입력", False, 끝)
        return
    # 2026-10-25 — 12 GB 에 들지 않는 MoE 는 llama.cpp 의 전문가 행렬을 호스트 메모리에(시험/품질.py --전문가CPU — llama-server 의 --cpu-moe)
    전문가 = ["--전문가CPU"] if "--MoE" in 모.get("머리옵션", []) and os.path.getsize(gguf) > 큰모델 else []
    for kv in (None, "q8_0"):
        rc, 끝 = 돌리기(파이썬("시험/품질.py", "라마", gguf, 품질라마, 라, *(["--kv", kv] if kv else []), *전문가),
                       os.path.join(출력, f"품질_라마{kv or ''}.txt"))
        if rc != 0:
            판("quality", f"llama.cpp({kv or 'f16'} KV)", False, 끝)
            return
    행 = ["--최대행", "128"] if os.path.getsize(gguf) > 큰모델 else []      # 12 GB 에 문맥 8,192 가 들게(비트는 청크 행 수와 상관없다)
    짝 = []
    for 이름, 모듈, 옵션, _ in 변형들(모):
        p = 모듈짓기(bin폴더, 모듈폴더, 모듈, 모["생성기인자"], 옵션)
        o = os.path.join(d, 이름)
        os.makedirs(o, exist_ok=True)
        rc, 끝 = 돌리기([os.path.join(bin폴더, "t7b" + EXE), gguf, p, os.path.join(라, "입력.bin"), o, "--한길"] + 행,
                       os.path.join(출력, f"품질_엔진{이름}.txt"))
        if rc != 0:
            판("quality", f"엔진 {이름}", False, 끝)
            return
        if 이름 != "정밀":
            짝.append(f"{이름}={o}")
    비교 = os.path.join(출력, "품질_비교.txt")
    rc, 끝 = 돌리기(파이썬("시험/품질.py", "견주기", 라, os.path.join(d, "정밀"), os.path.join(출력, "품질.json"), *짝), 비교)
    줄들 = open(비교, encoding="utf-8", errors="replace").read().splitlines()
    판정 = {줄.split("·")[1].split(":")[0].strip(): 줄 for 줄 in 줄들 if 줄.startswith("판정") and "·" in 줄}
    기본 = 기본변형(모)[0]
    모은값 = {}
    for i, 줄 in enumerate(줄들):                 # '글 4 개를 모아서' 아래의 모드마다 줄 — KL 비 · 1위
        if 줄.startswith("글 ") and "모아서" in 줄:
            for 다음 in 줄들[i + 1:]:
                if not 다음.startswith("    "):
                    break
                모은값[다음.strip().split(":")[0]] = 다음.strip()
    if rc != 0 or (기본 != "정밀" and 기본 not in 판정):
        판("quality", f"{기본} — 앱의 기본", False, f"비교 실패 — {끝}")
    elif 기본 == "정밀":                           # F16 · BF16 · F32 파일 — 앱의 기본이 견줌의 기준 그 자체
        판("quality", "정밀 — 앱의 기본(B′)", True, "견줌의 기준(정밀 모드) 그 자체 — 아래의 다른 모드는 참고")
    for 이름, 줄 in 판정.items():
        통과 = "B′(글마다) 통과" in 줄
        판("quality", f"{이름}{' — 앱의 기본' if 이름 == 기본 else ''}(B′)", 통과,
           f"{줄.split(':', 1)[-1].strip()} | 네 텍스트를 모은 값: {모은값.get(이름, '-').split(':', 1)[-1].strip()}", 참고=이름 != 기본)


def 단계템플릿(gguf, bin폴더, 출력):
    t = os.path.join(출력, "template")
    os.makedirs(t, exist_ok=True)
    cmd = ["wsl", "-d", "Ubuntu-24.04", "--", "python3", wsl경로(os.path.join(ROOT, "시험", "채팅틀대조.py")),
           wsl경로(os.path.join(bin폴더, "tchat" + EXE)), wsl경로(gguf), "--임시", wsl경로(t)]
    try:
        rc, 끝 = 돌리기(cmd, os.path.join(출력, "템플릿비교.txt"))
    except OSError:
        판("template", "채팅 템플릿 = jinja2", None, "WSL 이 없다")
        return
    판("template", "서버의 채팅 템플릿 = jinja2(채팅 · 도구 사례)", rc == 0 and "모두 같음" in 끝, 끝)


def 단계토크나이저(gguf, bin폴더, 출력, 라마):
    T = os.path.join(라마, "llama-tokenize" + EXE)
    if not os.path.exists(T):
        판("tokenizer", "토크나이저 = llama-tokenize", None, f"{T} 가 없다")
        return
    t = os.path.join(출력, "tokenizer")
    os.makedirs(t, exist_ok=True)
    텍스트들 = [(f"텍스트{k}", s) for k, s in enumerate(특이한텍스트)] + [(f, open(os.path.join(ROOT, f), encoding="utf-8").read()) for f in 문서들]
    inp = os.path.join(t, "텍스트.inp")
    with open(inp, "w", encoding="utf-8", newline="") as f:
        for _, s in 텍스트들:
            f.write(s + "\n__ggml_vocab_test__\n")
    rc, 끝 = 돌리기([os.path.join(bin폴더, "ttok" + EXE), gguf, inp, os.path.join(t, "엔진.txt")], os.path.join(t, "시험_토큰.txt"))
    if rc != 0:
        판("tokenizer", "토크나이저 = llama-tokenize", False, f"시험_토큰 실패 — {끝}")
        return
    엔진 = [[int(x) for x in 줄.split()] for 줄 in open(os.path.join(t, "엔진.txt"), encoding="utf-8").read().split("\n")]
    다름 = []
    아스키 = tempfile.mkdtemp(prefix="geulpu-tok-")    # llama.cpp 도구는 한글 경로를 열지 못한다
    for k, (이름, s) in enumerate(텍스트들):
        p = os.path.join(아스키, f"t{k}.txt")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(s)
        r = subprocess.run([T, "-m", gguf, "-f", p, "--ids", "--no-bos", "--log-disable", "--no-parse-special", "--no-escape"],
                           capture_output=True, stdin=subprocess.DEVNULL)
        줄 = next((x for x in reversed(r.stdout.decode("utf-8", "replace").strip().splitlines()) if x.startswith("[")), None)
        if r.returncode != 0 or 줄 is None:
            다름.append(f"{이름}: llama-tokenize 실패({r.stderr.decode('utf-8', 'replace').strip()[-200:]})")
            continue
        저쪽 = json.loads(줄)
        이쪽 = 엔진[k] if k < len(엔진) else []
        if 이쪽 != 저쪽:
            i = next((j for j in range(min(len(이쪽), len(저쪽))) if 이쪽[j] != 저쪽[j]), min(len(이쪽), len(저쪽)))
            다름.append(f"{이름}: 엔진 {len(이쪽)} · llama.cpp {len(저쪽)} 토큰, 처음 다른 위치 {i}")
    shutil.rmtree(아스키, ignore_errors=True)
    판("tokenizer", f"토크나이저 = llama-tokenize(텍스트 {len(텍스트들)})", not 다름, "; ".join(다름) or f"모두 같음({os.path.basename(라마)})")


def 단계앱(앱하나, gguf, 모, 출력, 벌칸):
    확인 = {"cuda": False, "vulkan": False, "도구": None, "문맥": None}
    sv = 앱하나.올리기(gguf)
    if "failed" in sv:
        판("app", "불러오기(CUDA)", False, sv["failed"])
        return 확인
    기대 = ("exact", False, False) if 기본변형(모)[0] == "정밀" else ("fast", 모["KV8"], 모["KV8"])
    얻음 = (sv.get("precision"), sv.get("kv8"), sv.get("fast_attention"))
    확인["문맥"] = sv.get("context")
    판("app", "불러오기(CUDA) · 앱이 고른 모드 = 판정한 모드", sv.get("layer") == "cuda" and 얻음 == 기대,
       f"{sv.get('layer')} · {sv.get('precision')} · 8비트 KV {sv.get('kv8')} · 빠른 어텐션 {sv.get('fast_attention')} · 문맥 {sv.get('context')}"
       + ("" if 얻음 == 기대 else f" — 기대 {기대}"))
    생각들 = [[], ["--생각끔"]] if 모["생각켬끔"] else [[]]
    for 사례 in ("채팅틀경우", "도구틀경우"):
        for 생각 in 생각들:
            이름 = f"서버 /tokenize = jinja2({'채팅' if 사례 == '채팅틀경우' else '도구'} 사례{' · 생각 끔' if 생각 else ''})"
            ref = os.path.join(출력, f"참조_{사례}{''.join(생각)}.json")
            r = subprocess.run(["wsl", "-d", "Ubuntu-24.04", "--", "python3", wsl경로(os.path.join(ROOT, "시험", "도구틀참조.py")), wsl경로(gguf),
                                wsl경로(os.path.join(ROOT, "시험", f"{사례}.json")), wsl경로(ref)] + 생각, capture_output=True, stdin=subprocess.DEVNULL)
            if r.returncode != 0:
                판("app", 이름, None, "WSL 의 jinja2 로 참조를 만들지 못했다")
                continue
            rc, 끝 = 돌리기([sys.executable, "-X", "utf8", os.path.join(ROOT, "시험", "도구틀.py"), 앱하나.주소, ref,
                            os.path.join(ROOT, "시험", f"{사례}.json")] + 생각, os.path.join(출력, f"도구틀_{사례}{''.join(생각)}.txt"))
            판("app", 이름, rc == 0, 끝)
    답 = 앱하나.채팅()
    확인["cuda"] = all(답)
    판("app", "채팅(CUDA — 한국어 · 영어)", all(답), " / ".join(x[:80].replace("\n", " ") for x in 답))
    if 모["템플릿도구"]:
        된 = True
        for 생각 in 생각들:
            rc, 끝 = 돌리기([sys.executable, "-X", "utf8", os.path.join(ROOT, "시험", "도구호출.py"), 앱하나.주소] + 생각,
                           os.path.join(출력, f"도구호출{''.join(생각)}.txt"))
            판("app", f"도구 호출{' · 생각 끔' if 생각 else ''}", rc == 0, 끝)
            된 = 된 and rc == 0
        확인["도구"] = 된
    else:
        판("app", "도구 호출", None, "채팅 템플릿에 도구가 없다")
    if not 벌칸:
        return 확인
    앱하나.멈춤()
    앱하나.묻기("/api/settings", {"gpu_layer": "vulkan"})
    try:
        앱하나.장치확인(다시=True)
        sv2 = 앱하나.올리기(gguf)
        if "failed" in sv2:
            판("vulkan", "불러오기(Vulkan)", False, sv2["failed"])
        elif sv2.get("layer") != "vulkan":
            판("vulkan", "불러오기(Vulkan)", False, f"앱이 고른 백엔드 {sv2.get('layer')} — {앱하나.묻기('/api/state')['device'].get('reason')}")
        else:
            답2 = 앱하나.채팅()
            확인["vulkan"] = all(답2)
            판("vulkan", f"채팅(Vulkan · {sv2.get('precision')})", all(답2), " / ".join(x[:80].replace("\n", " ") for x in 답2))
    finally:
        앱하나.멈춤()
        앱하나.묻기("/api/settings", {"gpu_layer": "auto"})
    return 확인


def 단계목록(gguf, 모, 확인, 저장소, 앱봄):
    h = hashlib.sha256()
    with open(gguf, "rb") as f:
        for 청크 in iter(lambda: f.read(1 << 24), b""):
            h.update(청크)
    파일 = os.path.basename(gguf)
    매개 = 모["매개변수"]
    params = 모["크기표"] or (f"{매개 / 1e9:.1f}B".replace(".0B", "B") if 매개 >= 1e9 else f"{매개 / 1e6:.0f}M")
    양자 = {7: "Q8_0", 2: "Q4_0", 15: "Q4_K_M", 14: "Q4_K_S", 17: "Q5_K_M", 16: "Q5_K_S", 18: "Q6_K", 30: "IQ4_XS",
            0: "F32", 1: "F16", 32: "BF16"}.get(모["파일형식"], 모["형식글"])
    라이선스 = {"apache-2.0": "Apache-2.0", "mit": "MIT"}.get(모["라이선스"].lower(), 모["라이선스"] or "<라이선스>")
    날 = time.strftime("%Y-%m-%d")
    gpu = GPU이름()
    ko, en = [], []
    if 모["생각켬끔"]:
        ko.append("답하기 전에 생각 과정을 쓰며, 생성 설정에서 끌 수 있습니다.")
        en.append("It writes a thought process before answering, which you can turn off in the generation settings.")
    if 모["템플릿"] and not 모["템플릿도구"]:
        ko.append("채팅 템플릿에 도구가 없어 도구 호출은 받지 않습니다.")
        en.append("Its chat template has no tools, so it does not take tool calls.")
    if 확인.get("문맥") and 확인["문맥"] < 8192:
        ko.append("문맥 길이는 엔진이 그래픽 메모리에 맞춰 자동으로 줄입니다.")
        en.append("The engine reduces the context length automatically to fit the graphics memory.")
    if 모["형식글"] in ("Q8_0", "Q4_0"):
        ko.append(f"{모['형식글']} 파일이라 공유 메모리가 32 KB인 GPU(AMD 등)에서도 실행됩니다.")
        en.append(f"As a {모['형식글']} file it also runs on GPUs with 32 KB of shared memory (AMD and others).")
    else:
        ko.append("CUDA나 공유 메모리 48 KB 이상인 Vulkan GPU에서 실행됩니다.")
        en.append("Runs on CUDA or on Vulkan GPUs with 48 KB or more of shared memory.")
    if not 앱봄 or not 확인.get("cuda"):
        ko.append("<확인한 장치 — app 단계를 돌리지 않았거나 통과하지 못했다>")
        en.append("<checked devices>")
    elif 확인.get("vulkan") and 확인.get("도구"):
        ko.append(f"이 파일은 CUDA({gpu})와 Vulkan에서 실행을, CUDA에서 도구 호출을 확인했습니다({날}).")
        en.append(f"This file was checked to run on CUDA ({gpu}) and Vulkan, and to make tool calls on CUDA ({날}).")
    elif 확인.get("vulkan"):
        ko.append(f"이 파일은 CUDA({gpu})와 Vulkan에서 실행을 확인했습니다({날}).")
        en.append(f"This file was checked to run on CUDA ({gpu}) and Vulkan ({날}).")
    elif 확인.get("도구"):
        ko.append(f"이 파일은 CUDA({gpu})에서 실행과 도구 호출을 확인했습니다({날}).")
        en.append(f"This file was checked to run and make tool calls on CUDA ({gpu}) ({날}).")
    else:
        ko.append(f"이 파일은 CUDA({gpu})에서 실행을 확인했습니다({날}).")
        en.append(f"This file was checked to run on CUDA ({gpu}) ({날}).")
    note = "<모델 설명(크기 GB)> " + " ".join(ko)
    항목 = {"id": os.path.splitext(파일)[0].lower(), "name": 모["이름"] or os.path.splitext(파일)[0], "params": params, "quant": 양자,
            "file": 파일, "url": f"https://huggingface.co/{저장소}/resolve/main/{파일}" if 저장소 else "<저장소의 resolve/main 주소>",
            "source": f"huggingface.co/{저장소}" if 저장소 else "<huggingface.co/저장소>", "size": os.path.getsize(gguf),
            "sha256": h.hexdigest(), "license": 라이선스, "check": "tested", "devices": "all" if 모["형식글"] in ("Q8_0", "Q4_0") else "k",
            "note": note}
    번역 = f'  {json.dumps(note, ensure_ascii=False)}: {json.dumps("<model description (size GB)> " + " ".join(en), ensure_ascii=False)},'
    with open(os.path.join(작업, "목록초안.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"models.json 항목": 항목, "app.js DICT 줄": 번역}, f, ensure_ascii=False, indent=2)
    판("catalog", "모델 목록 항목 · 번역 줄의 초안", True, "목록초안.json — 모델 설명 · 저장소 · 라이선스의 조건은 사람이 채운다")


def 해보기(단계, 일, *인자):
    """단계 하나가 예외로 끝나도(모듈 생성 실패 · 앱 응답 없음 …) 판정 표는 남긴다"""
    try:
        return 일(*인자)
    except Exception as e:                          # noqa: BLE001 — 까닭을 표에 적고 다음 단계로
        판(단계, "도구 오류", False, f"{type(e).__name__}: {e}")
        return None


def main():
    global 작업
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="자동 모델 검증(0.12.0 12-G3 — 문서/모델검증.md)")
    p.add_argument("model", help="GGUF 파일(ASCII 경로)")
    p.add_argument("--work", required=True, help="작업 폴더(SSD)")
    p.add_argument("--geul", default=os.path.normpath(os.path.join(ROOT, "..", "geul-lang-v2")), help="글(geul) 저장소 — 컴파일러")
    p.add_argument("--app", default="", help="앱/꾸리기.ps1 로 패키지한 앱 폴더")
    p.add_argument("--app-data", default="", help="앱의 임시 데이터 폴더(기본 <work>/app-data)")
    p.add_argument("--tokenizer-llama", default="", help="llama-tokenize 가 있는 llama.cpp 폴더")
    p.add_argument("--quality-llama", default="", help="품질 비교의 기준 llama.cpp 폴더(b11719)")
    p.add_argument("--wiki", default=os.path.join(ROOT, "빌드", "wikitext-2", "wiki.test.raw"))
    p.add_argument("--repo", default="", help="Hugging Face 저장소 — 목록 초안의 주소")
    p.add_argument("--stages", default=",".join(단계들))
    p.add_argument("--port", type=int, default=44651)
    a = p.parse_args()
    gguf = os.path.abspath(a.model)
    작업 = os.path.abspath(a.work)
    os.makedirs(작업, exist_ok=True)
    할 = [x for x in a.stages.split(",") if x]
    모름 = [x for x in 할 if x not in 단계들]
    if 모름:
        raise SystemExit(f"모르는 단계: {', '.join(모름)} — {', '.join(단계들)}")
    for 이름, 경로 in (("model", gguf), ("--tokenizer-llama", a.tokenizer_llama), ("--quality-llama", a.quality_llama)):
        if not 경로.isascii():
            raise SystemExit(f"{이름} 의 경로는 ASCII 여야 한다(llama.cpp 도구가 한글 경로를 열지 못한다): {경로}")
    t0 = time.time()
    적기(f"모델 검증: {gguf} — 단계 {', '.join(할)}")
    모 = 모델모양(*gguf헤더(gguf))
    출력 = os.path.join(작업, "r")
    bin폴더, 모듈폴더 = os.path.join(작업, "bin"), os.path.join(작업, "모듈")
    os.makedirs(출력, exist_ok=True)
    os.makedirs(모듈폴더, exist_ok=True)
    앱하나 = None
    참조일 = []
    확인 = {}
    try:
        if a.app:
            앱하나 = 해보기("header", 앱, os.path.abspath(a.app), os.path.abspath(a.app_data or os.path.join(작업, "app-data")), a.port)
        # 2026-10-25 — 모듈 이름 · 커널 생성기 인자는 엔진의 모델 확인으로: 이 저장소의 서버를 먼저 빌드한다
        틀림 = 빌드(os.path.abspath(a.geul), bin폴더, ["srv"])
        if 틀림:
            판("header", "서버 빌드(엔진의 모델 확인)", False, "; ".join(틀림))
        서버 = None if 틀림 else os.path.join(bin폴더, "srv" + EXE)
        if (앱하나 is not None or not a.app) and 해보기("header", 단계헤더, gguf, 모, 앱하나, 서버):
            with open(os.path.join(작업, "모양.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(모, f, ensure_ascii=False, indent=1)
            쓸도구 = sorted({x for 단계 in 할 for x in 단계도구.get(단계, [])})
            if 쓸도구:
                틀림 = 빌드(os.path.abspath(a.geul), bin폴더, 쓸도구)
                판("header", f"도구 빌드(글(geul) 컴파일러 — {len(쓸도구)})", not 틀림, "; ".join(틀림))
                if 틀림:
                    할 = [x for x in 할 if x in ("app", "vulkan", "catalog")]
            라마 = os.path.abspath(a.tokenizer_llama) if a.tokenizer_llama else ""
            if "bits" in 할:
                길 = 해보기("bits", 단계비트, gguf, 모, bin폴더, 모듈폴더, 출력, 라마) or []
                if "ref" in 할:
                    참조일 = 참조시작(gguf, 모, 길, 출력)
            if "quality" in 할:
                if a.quality_llama:
                    해보기("quality", 단계품질, gguf, 모, bin폴더, 모듈폴더, 출력, os.path.abspath(a.quality_llama), os.path.abspath(a.wiki))
                else:
                    판("quality", "llama.cpp 대비", None, "--quality-llama 를 주지 않았다")
            if "template" in 할:
                해보기("template", 단계템플릿, gguf, bin폴더, 출력)
            if "tokenizer" in 할:
                if 라마:
                    해보기("tokenizer", 단계토크나이저, gguf, bin폴더, 출력, 라마)
                else:
                    판("tokenizer", "토크나이저 = llama-tokenize", None, "--tokenizer-llama 를 주지 않았다")
            if "app" in 할 or "vulkan" in 할:
                if 앱하나 is None:
                    판("app", "패키지한 앱", None, "--app 을 주지 않았다")
                elif not GPU비었나():
                    판("app", "패키지한 앱", None, "GPU 메모리를 다른 작업이 쓰고 있다")
                else:
                    확인 = 해보기("app", 단계앱, 앱하나, gguf, 모, 출력, "vulkan" in 할) or {}
            if 참조일:
                적기("CPU 참조를 기다린다")
                참조기다림(참조일, 출력)
                참조일 = []
            if "catalog" in 할:
                해보기("catalog", 단계목록, gguf, 모, 확인, a.repo, "app" in 할 and 앱하나 is not None)
    finally:
        for _, pr, f in 참조일:
            pr.kill()
            f.close()
        if 앱하나 is not None:
            앱하나.끄기()
    못 = [x for x in 결과들 if x["통과"] is False and not x["참고"]]
    건너뜀 = [x for x in 결과들 if x["통과"] is None]
    with open(os.path.join(작업, "결과.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"모델": gguf, "모양": 모, "결과": 결과들, "초": round(time.time() - t0)}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(작업, "결과.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# 모델 검증 — {os.path.basename(gguf)}\n\n{time.strftime('%Y-%m-%d')} · {time.time() - t0:.0f} 초\n\n"
                "| 단계 | 시험 | 판정 | 내용 |\n|---|---|---|---|\n")
        for x in 결과들:
            글 = x["글"].replace("|", "·").replace("\n", " ")[:400]
            표 = 판글(x["통과"], x["참고"])
            f.write(f"| {x['단계']} | {x['이름']} | {'**' + 표 + '**' if x['통과'] is False and not x['참고'] else 표} | {글} |\n")
    끝 = "모두 통과" if not 못 else f"못 미침 {len(못)}"
    적기(f"모델 검증: {끝}{f'(건너뜀 {len(건너뜀)})' if 건너뜀 else ''} — {time.time() - t0:.0f} 초 · {os.path.join(작업, '결과.md')}")
    return 0 if not 못 else 1


if __name__ == "__main__":
    sys.exit(main())
