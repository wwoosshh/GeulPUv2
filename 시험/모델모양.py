#!/usr/bin/env python3
"""GeulPUv2 — 앱이 셈할 수 없는 모델 파일을 까닭과 함께 받지 않는가를 GPU 없이 본다(빠른 점검 4i — 윈도만). 머리만 있는 작은 GGUF
(텐서 없음)를 임시 자료 폴더의 모델 폴더에 두고 앱을 띄워 /api/state 의 models[].shape(ok · reason)를 본다:
  - qwen2 의 RMSNorm eps 가 10⁻⁶ · 10⁻⁵ 이 아니면 그 까닭(#16 — 모듈의 상수와 다른 eps 로 셈하지 않게)
  - eps 10⁻⁶ · 10⁻⁵ · 열쇠 없음은 eps 규칙을 지나 다음 검사(모델 크기)에서 멈춘다
  - qwen2 에 RoPE 늘이기 정보(rope.scaling.type — none 이 아님)가 있으면 그 까닭(#17 — 늘이지 않은 RoPE 로 셈하지 않게)
  - GeulPUv2 1.0.0 1-6a — 질의 폭(머리 × 128)이 너비와 다른 qwen3(0.6B 의 모양)은 모양 규칙을 지나 다음 검사(토크나이저)에서 멈추고,
    key_length 가 없어 머리 크기(너비 / 머리 수)가 128 이 아니면 그 까닭
  - 0.12.0 12-G1b — 채팅 템플릿은 서버의 jinja 해석기가 풀지 못할 때만 그 까닭(풀리거나 없으면 — 기본 ChatML — 다음 검사에서 멈춘다)
  - 0.12.0 12-G2 — 바이트 BPE 의 사전 처리는 llama.cpp 의 표에 있는 바이트 꼴 이름만(qwen35 · gpt-2 는 지나고, gemma4 · kimi-k2 · 모르는 이름은 그 까닭)

  python -I 시험/모델모양.py <GeulPU.exe> <빈 자료 폴더>

마지막 줄 '모델 모양: 모두 통과 (N 가지)'(종료 0) 또는 '… 틀림 N'(종료 1)."""
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.request

EPS_까닭 = "RMSNorm eps가 10⁻⁶ · 10⁻⁵이 아닌 Qwen2 파일은 지원하지 않습니다"
크기_까닭 = "지원하지 않는 모델 크기입니다"
ROPE_까닭 = "RoPE 늘이기(yarn)가 들어간 Qwen2 파일은 지원하지 않습니다"
토큰_까닭 = "지원하지 않는 토크나이저입니다"
머리_까닭 = "지원하지 않는 헤드 크기입니다"
틀_까닭 = "이 파일의 채팅 템플릿(tokenizer.chat_template)을 GeulPU의 jinja 해석기가 아직 풀지 못합니다"
낱말_까닭 = "token_embd.weight 텐서가 없습니다"
앞_까닭 = "지원하지 않는 토크나이저 사전 처리입니다"


def 글(x):
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def 머리(메타):
    """GGUF v3 머리 — 텐서 0 개, 메타는 (열쇠, 값) — 값이 글이면 string(8), 정수면 uint32(4), 실수면 float32(6)."""
    몸 = b"GGUF" + struct.pack("<I", 3) + struct.pack("<q", 0) + struct.pack("<q", len(메타))
    for 열쇠, 값 in 메타:
        if isinstance(값, str):
            몸 += 글(열쇠) + struct.pack("<I", 8) + 글(값)
        elif isinstance(값, int):
            몸 += 글(열쇠) + struct.pack("<I", 4) + struct.pack("<I", 값)
        else:
            몸 += 글(열쇠) + struct.pack("<I", 6) + struct.pack("<f", 값)
    return 몸 + b"\0" * ((-len(몸)) % 32)


def 경우들():
    구조 = ("general.architecture", "qwen2")
    eps = "qwen2.attention.layer_norm_rms_epsilon"
    return [
        ("eps 10⁻⁴ 은 받지 않는다", [구조, (eps, 1e-4)], EPS_까닭),
        ("eps 10⁻⁷ 은 받지 않는다", [구조, (eps, 1e-7)], EPS_까닭),
        ("eps 10⁻⁶ 은 eps 규칙을 지난다", [구조, (eps, 1e-6)], 크기_까닭),
        ("eps 10⁻⁵ 은 eps 규칙을 지난다(qwen2e5 모듈)", [구조, (eps, 1e-5)], 크기_까닭),
        ("eps 열쇠가 없으면 10⁻⁶ 으로 지난다", [구조], 크기_까닭),
        ("RoPE 늘이기 yarn 은 받지 않는다", [구조, (eps, 1e-6), ("qwen2.rope.scaling.type", "yarn")], ROPE_까닭),
        ("RoPE 늘이기 none 은 지난다", [구조, (eps, 1e-6), ("qwen2.rope.scaling.type", "none")], 크기_까닭),
    ] + 질의폭경우들() + 틀경우들() + 사전처리경우들()


def 질의폭경우들():
    """1-6a — qwen3 의 질의 폭(머리 × 128)이 너비와 다른 모양(Qwen3-0.6B: 너비 1024 · 머리 16 · KV 머리 8 · 확장폭 3072)."""
    q3 = [("general.architecture", "qwen3"), ("qwen3.attention.layer_norm_rms_epsilon", 1e-6), ("qwen3.block_count", 28),
          ("qwen3.embedding_length", 1024), ("qwen3.feed_forward_length", 3072)]
    머리 = [("qwen3.attention.head_count", 16), ("qwen3.attention.head_count_kv", 8)]
    길이 = [("qwen3.attention.key_length", 128), ("qwen3.attention.value_length", 128)]
    return [
        ("qwen3 질의 폭 2048 ≠ 너비 1024 는 모양 규칙을 지난다", q3 + 머리 + 길이, 토큰_까닭),
        ("qwen3 key_length 가 없으면 머리 크기 너비 / 머리 = 64 — 받지 않는다", q3 + 머리, 머리_까닭),
        ("qwen3 머리 수가 없으면 모델 크기에서 멈춘다", q3 + 길이, 크기_까닭),
    ]


def 틀경우들():
    """0.12.0 12-G1b — 서버가 GGUF 의 채팅 템플릿을 jinja 해석기로 그대로 실행한다(Qwen2.5-1.5B 의 모양 · 토크나이저 — 텐서는 없다)."""
    q2 = [("general.architecture", "qwen2"), ("qwen2.attention.layer_norm_rms_epsilon", 1e-6), ("qwen2.embedding_length", 1536),
          ("qwen2.feed_forward_length", 8960), ("qwen2.attention.head_count", 12), ("qwen2.attention.head_count_kv", 2),
          ("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", "qwen2")]
    return [
        ("풀리지 않는 채팅 템플릿(endfor 없음)은 받지 않는다", q2 + [("tokenizer.chat_template", "{% for m in messages %}{{ m.content }}")], 틀_까닭),
        ("풀리는 채팅 템플릿은 템플릿 검사를 지난다", q2 + [("tokenizer.chat_template", "{% for m in messages %}{{ m.content }}{% endfor %}")],
         낱말_까닭),
        ("채팅 템플릿이 없으면(기본 ChatML) 템플릿 검사를 지난다", q2, 낱말_까닭),
    ]



def 사전처리경우들():
    """0.12.0 12-G2 — 바이트 BPE 의 사전 처리(tokenizer.ggml.pre)는 llama.cpp 의 표(엔진/사전처리표.gl)로 — Qwen2.5-1.5B 의 모양에 이름만 바꿔."""
    q2 = [("general.architecture", "qwen2"), ("qwen2.attention.layer_norm_rms_epsilon", 1e-6), ("qwen2.embedding_length", 1536),
          ("qwen2.feed_forward_length", 8960), ("qwen2.attention.head_count", 12), ("qwen2.attention.head_count_kv", 2),
          ("tokenizer.ggml.model", "gpt2")]
    return [
        ("사전 처리 qwen35(Qwen3.5)는 지난다", q2 + [("tokenizer.ggml.pre", "qwen35")], 낱말_까닭),
        ("사전 처리 gpt-2 는 지난다(정규식 엔진)", q2 + [("tokenizer.ggml.pre", "gpt-2")], 낱말_까닭),
        ("사전 처리 gemma4(SentencePiece 꼴 BPE)는 받지 않는다", q2 + [("tokenizer.ggml.pre", "gemma4")], 앞_까닭),
        ("사전 처리 kimi-k2(손으로 짠 나누기만)는 받지 않는다", q2 + [("tokenizer.ggml.pre", "kimi-k2")], 앞_까닭),
        ("모르는 사전 처리는 받지 않는다", q2 + [("tokenizer.ggml.pre", "no-such-pre")], 앞_까닭),
    ]

def 빈포트():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    exe, 자료 = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    shutil.rmtree(자료, ignore_errors=True)
    os.makedirs(os.path.join(자료, "모델"))
    표 = 경우들()
    for i, (_, 메타, _) in enumerate(표):
        with open(os.path.join(자료, "모델", f"m{i}.gguf"), "wb") as f:
            f.write(머리(메타))
    포트 = 빈포트()
    주소 = f"http://127.0.0.1:{포트}"
    기록 = open(os.path.join(os.path.dirname(자료.rstrip("\\/")), "모델모양_앱.txt"), "ab")
    p = subprocess.Popen([exe, "--포트", str(포트), "--자료", 자료, "--열지않음"], stdout=기록, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL)
    틀림 = []
    try:
        상태 = None
        for _ in range(300):
            try:
                상태 = json.loads(urllib.request.urlopen(주소 + "/api/state", timeout=10).read())
                break
            except Exception:
                if p.poll() is not None:
                    break
                time.sleep(0.1)
        if 상태 is None:
            raise SystemExit(f"앱이 뜨지 않았다(종료 코드 {p.poll()})")
        칸 = {os.path.basename(m["path"]): m for m in 상태["models"]}
        for i, (이름, _, 기대) in enumerate(표):
            s = (칸.get(f"m{i}.gguf") or {}).get("shape") or {}
            까닭 = s.get("reason") or ""
            맞음 = s.get("ok") is False and 까닭.startswith(기대)
            print(f"{'통과' if 맞음 else '틀림'}  {이름} — {까닭[:70]}", flush=True)
            if not 맞음:
                틀림.append(이름)
    finally:
        p.terminate()
        p.wait(30)
    print(f"모델 모양: 모두 통과 ({len(표)} 가지)" if not 틀림 else f"모델 모양: 틀림 {len(틀림)} — {'; '.join(틀림)}")
    sys.exit(1 if 틀림 else 0)


if __name__ == "__main__":
    main()
