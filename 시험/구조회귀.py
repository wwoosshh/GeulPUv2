#!/usr/bin/env python3
"""구조 회귀 시험 — 공유 코어(엔진 · 커널 생성기 · 구조 레지스트리)를 고친 뒤, 구조마다 작은 모델 파일로 엔진의 로짓 · KV 가 비트까지
그대로인지 본다(문서/개발.md 의 '여러 세션으로 개발하기' — 검증 ②). 고치기 전에 기준을 적고(--쓰기), 고친 뒤 같은 명령으로 견준다.

  python -I -X utf8 시험/구조회귀.py --모델 E:/geulpu-work --작업 <폴더 — SSD> --기준 <기준.json> [--쓰기] [--경우 이름,…] [--글 ../geul-lang-v2]

- 빌드: 커널생성 · ptx-backend · 시험_7B 를 이 저장소의 소스에서 <작업>/bin 에(글(geul) 컴파일러 — --글).
- 경우마다: 모듈(커널생성 → PTX — 이름은 엔진이 모드를 읽는 꼴 그대로)을 만들고, 정한 무작위 토큰열 셋(길이 40 · 73 은 잡기시작 0 — 모든
  행, 길이 1,100 은 잡기시작 1,000 — 덩이 경계 1,024)으로 시험_7B 의 다섯 길(생성 · 한 번에 · 넷으로 · 한 행씩 · 다시)을 GPU 줄
  (시험/GPU줄.py)에 서서 돌린다. 시험_7B 의 요약(경우마다 토큰 · 로짓 해시 · 생성 로짓 해시 · KV 해시 · 길끼리의 같음)과 모듈 PTX 의
  SHA-256 을 적는다.
- --쓰기 면 기준 파일에 쓰고, 아니면 기준과 견준다 — 마지막 줄 '구조 회귀: 모두 같음' 또는 '구조 회귀: 다름 N'(종료 코드 1).
모델 파일은 받지 않는다(--모델 아래의 경로 — 없는 경우는 건너뜀). Qwen3.5 MoE 는 시험/작은MoE.py 로 뗀 파일(층 4 · 전문가 32).
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
EXE = ".exe" if os.name == "nt" else ""

# (이름, --모델 아래의 GGUF, 모듈 이름(엔진이 모드를 읽는 꼴), 커널생성 인자) — 구조 · 형식 · 모드마다 하나씩, 이 PC 의 작은 파일로
경우들 = [
    ("qwen2-h64-Q8-정밀", "v1/models/qwen2.5-0.5b-instruct-q8_0.gguf", "qwen2Q8_896x14x2x4864x151936_h64_먼",
     "qwen2큰 Q8_0 896 14 2 4864 151936 --먼 --헤드 64 --RoPE 64"),
    ("qwen2-h64-Q8-빠른", "v1/models/qwen2.5-0.5b-instruct-q8_0.gguf", "qwen2Q8_896x14x2x4864x151936_h64_빠른",
     "qwen2큰 Q8_0 896 14 2 4864 151936 --빠른 --헤드 64 --RoPE 64"),
    ("qwen2-h64-W16-정밀", "v12/gen/qwen2.5-0.5b-F16.gguf", "qwen2W16_896x14x2x4864x151936_h64_먼",
     "qwen2큰 W16 896 14 2 4864 151936 --먼 --헤드 64 --RoPE 64"),
    ("qwen2-Q8-정밀", "v12/models/qwen2.5-1.5b-instruct-q8_0.gguf", "qwen2Q8_1536x12x2x8960x151936_먼",
     "qwen2큰 Q8_0 1536 12 2 8960 151936 --먼"),
    ("qwen2-K-F2", "f2/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf", "qwen2K_3584x28x4x18944x152064_빠른_F2_KV8",
     "qwen2큰 K 3584 28 4 18944 152064 --빠른 --F2"),
    ("qwen3-Q8-정밀", "v12/models/Qwen3-0.6B-Q8_0.gguf", "qwen3Q8_1024x16x8x3072x151936_먼",
     "qwen3큰 Q8_0 1024 16 8 3072 151936 --먼"),
    ("qwen3-K-빠른", "v1/models/Qwen3-0.6B-Q4_K_M-rq.gguf", "qwen3K_1024x16x8x3072x151936_빠른",
     "qwen3큰 K 1024 16 8 3072 151936 --빠른"),
    ("qwen3-KI-정밀", "v1/models/Qwen3-0.6B-IQ4_XS-rq.gguf", "qwen3KI_1024x16x8x3072x151936_먼",
     "qwen3큰 KI 1024 16 8 3072 151936 --먼"),
    ("llama-Q8-정밀", "모델/Llama-3.2-3B-Instruct-Q8_0.gguf", "llamaQ8_3072x24x8x8192x128256_먼",
     "llama큰 Q8_0 3072 24 8 8192 128256 --먼"),
    ("llama-K-F2", "v12/models/hyperclovax-seed-text-instruct-1.5b-q4_k_m.gguf", "llamaK_2048x16x8x7168x110592_빠른_F2_KV8",
     "llama큰 K 2048 16 8 7168 110592 --빠른 --F2"),
    ("llamae6-K-정밀", "v12/models/MiniCPM5-2B-Q4_K_M.gguf", "llamae6K_2048x16x2x6144x130560_먼",
     "llamae6큰 K 2048 16 2 6144 130560 --먼"),
    ("qwen35-Q8", "v12/models/Qwen3.5-0.8B-Q8_0.gguf", "qwen35Q8_1024x8x2x3584x248320",
     "qwen35큰 Q8_0 1024 8 2 3584 248320"),
    ("qwen35-K", "v12/models/Qwen3.5-4B-Q4_K_M.gguf", "qwen35K_2560x16x4x9216x248320",
     "qwen35큰 K 2560 16 4 9216 248320"),
    ("qwen35moe-작은", "구조회귀/작은MoE.gguf", "qwen35moeQ8_2048x16x2x512x248320_MoEKN",
     "qwen35큰 Q8_0 2048 16 2 512 248320 --MoE K --W16 --K낱말"),
]
도구들 = {"커널생성": "커널/커널생성.gl", "ptx-backend": "백엔드/PTX/PTX.gl", "시험_7B": "엔진/시험_7B.gl"}


def 해시(경로):
    h = hashlib.sha256()
    with open(경로, "rb") as f:
        for 덩이 in iter(lambda: f.read(1 << 20), b""):
            h.update(덩이)
    return h.hexdigest()


def 돌리기(명령, 기록, **kw):
    with open(기록, "wb") as f:
        r = subprocess.run(명령, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
    return r.returncode, open(기록, "rb").read().decode("utf-8", "replace").strip().splitlines()[-3:]


def 빌드(글, bin폴더):
    os.makedirs(bin폴더, exist_ok=True)
    env = dict(os.environ, GEUL_ROOT=글)
    컴 = os.path.join(글, "build", "self_컴파일러" + EXE)

    def 하나(항목):
        이름, 소스 = 항목
        rc, 끝 = 돌리기([컴, 소스, "-o", os.path.join(bin폴더, 이름 + EXE)], os.path.join(bin폴더, f"빌드_{이름}.txt"), cwd=ROOT, env=env)
        return 이름, rc, 끝
    with ThreadPoolExecutor(3) as ex:
        for 이름, rc, 끝 in ex.map(하나, 도구들.items()):
            if rc != 0:
                raise SystemExit(f"빌드 실패 {이름}: {끝}")


def 입력쓰기(경로, 어휘):
    """[최대길이][로짓행][경우 수], 경우마다 [p][n][잡기시작][토큰 p 개] — 8 바이트 정수. 토큰은 정한 씨앗의 무작위(어휘 앞 100,000 안)"""
    import random
    r = random.Random(20261024)
    상한 = min(어휘, 100000)
    경우 = [(40, 17, 0), (73, 17, 0), (1100, 17, 1000)]
    with open(경로, "wb") as f:
        f.write(struct.pack("<qqq", 4096, 128, len(경우)))
        for p, n, 잡기 in 경우:
            f.write(struct.pack("<qqq", p, n, 잡기))
            f.write(struct.pack(f"<{p}q", *[r.randrange(상한) for _ in range(p)]))


def 어휘수(gguf):
    sys.path.insert(0, os.path.join(ROOT, "계약"))
    import gguf읽기 as GR
    g = GR.GGUF(gguf)
    return g.tensors["token_embd.weight"][1][1]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--모델", required=True, help="모델 파일의 바탕 폴더(경우의 경로가 그 아래)")
    p.add_argument("--작업", required=True, help="빌드 · 모듈 · 출력 폴더(SSD)")
    p.add_argument("--기준", required=True, help="기준 JSON — --쓰기 면 쓰고, 아니면 견준다")
    p.add_argument("--쓰기", action="store_true")
    p.add_argument("--경우", default="", help="쉼표로 가른 경우 이름(기본: 모두)")
    p.add_argument("--글", default=os.path.normpath(os.path.join(ROOT, "..", "geul-lang-v2")))
    a = p.parse_args()
    고른 = [x for x in 경우들 if not a.경우 or x[0] in a.경우.split(",")]
    작업 = os.path.abspath(a.작업)
    bin폴더 = os.path.join(작업, "bin")
    빌드(os.path.abspath(a.글), bin폴더)
    print(f"빌드: {' · '.join(f'{k} {해시(os.path.join(bin폴더, k + EXE))[:12]}' for k in 도구들)}", flush=True)

    def 모듈(경우):
        이름, gguf, 모듈이름, 인자 = 경우
        d = os.path.join(작업, 이름)
        os.makedirs(d, exist_ok=True)
        gl, ptx = os.path.join(d, 모듈이름 + ".gl"), os.path.join(d, 모듈이름 + ".ptx")
        args = 인자.split()
        i = next((k for k, x in enumerate(args) if x.startswith("--")), len(args))
        rc, 끝 = 돌리기([os.path.join(bin폴더, "커널생성" + EXE)] + args[:i] + [gl] + args[i:], os.path.join(d, "생성.txt"))
        if rc != 0:
            return 이름, f"생성 실패: {끝}"
        rc, 끝 = 돌리기([os.path.join(bin폴더, "ptx-backend" + EXE), gl, "-o", ptx], os.path.join(d, "PTX.txt"))
        return 이름, None if rc == 0 else f"PTX 실패: {끝}"
    있는 = [x for x in 고른 if os.path.exists(os.path.join(a.모델, x[1]))]
    for x in 고른:
        if x not in 있는:
            print(f"건너뜀 {x[0]}: 모델 파일이 없다({x[1]})", flush=True)
    with ThreadPoolExecutor(4) as ex:
        모듈오류 = dict(ex.map(모듈, 있는))
    결과 = {}
    for 이름, gguf, 모듈이름, 인자 in 있는:
        if 모듈오류[이름]:
            결과[이름] = {"오류": 모듈오류[이름]}
            print(f"{이름}: {모듈오류[이름]}", flush=True)
            continue
        d = os.path.join(작업, 이름)
        경로 = os.path.join(a.모델, gguf)
        입력 = os.path.join(d, "입력.bin")
        입력쓰기(입력, 어휘수(경로))
        출력 = os.path.join(d, "출력")
        os.makedirs(출력, exist_ok=True)
        if os.path.exists(os.path.join(출력, "요약.txt")):
            os.remove(os.path.join(출력, "요약.txt"))
        ptx = os.path.join(d, 모듈이름 + ".ptx")
        rc, 끝 = 돌리기([sys.executable, "-I", "-X", "utf8", os.path.join(HERE, "GPU줄.py"), "실행", "--조용히", "--이름",
                         f"구조 회귀 — {이름}", "--", os.path.join(bin폴더, "시험_7B" + EXE), 경로, ptx, 입력, 출력],
                        os.path.join(d, "시험_7B.txt"))
        요약 = os.path.join(출력, "요약.txt")
        if rc != 0 or not os.path.exists(요약):
            결과[이름] = {"오류": f"시험_7B 종료 {rc}: {끝}"}
            print(f"{이름}: 시험_7B 종료 {rc} — {끝}", flush=True)
            continue
        글 = open(요약, encoding="utf-8").read()
        결과[이름] = {"모듈PTX": 해시(ptx), "요약": 글}
        print(f"{이름}: {'모두 같음 1' if '모두 같음 1' in 글 else '길끼리 다름!'} · 요약 {hashlib.sha256(글.encode()).hexdigest()[:12]}", flush=True)
    if a.쓰기:
        with open(a.기준, "w", encoding="utf-8", newline="\n") as f:
            json.dump(결과, f, ensure_ascii=False, indent=1)
        나쁜 = [k for k, v in 결과.items() if "오류" in v or "모두 같음 1" not in v["요약"]]
        print(f"구조 회귀: 기준을 적음 — 경우 {len(결과)} 개" + (f", 문제 {나쁜}" if 나쁜 else ""))
        return 1 if 나쁜 else 0
    기준 = json.load(open(a.기준, encoding="utf-8"))
    다름 = []
    for 이름, v in 결과.items():
        b = 기준.get(이름)
        if b is None:
            print(f"{이름}: 기준에 없다")
            continue
        if "오류" in v or v.get("요약") != b.get("요약"):
            다름.append(이름)
            print(f"다름 {이름}: " + (v.get("오류") or "요약이 다르다"))
        elif v["모듈PTX"] != b.get("모듈PTX"):
            print(f"{이름}: 요약은 같고 모듈 PTX 가 다르다(커널 생성기가 바뀜)")
    print("구조 회귀: 모두 같음" if not 다름 else f"구조 회귀: 다름 {len(다름)} {다름}")
    return 1 if 다름 else 0


if __name__ == "__main__":
    sys.exit(main())
