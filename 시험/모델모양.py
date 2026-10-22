#!/usr/bin/env python3
"""GeulPUv2 — 앱이 셈할 수 없는 모델 파일을 까닭과 함께 받지 않는가를 GPU 없이 본다(빠른 점검 4i — 윈도만). 머리만 있는 작은 GGUF
(텐서 없음)를 임시 자료 폴더의 모델 폴더에 두고 앱을 띄워 /api/state 의 models[].shape(ok · reason)를 본다:
  - qwen2 의 RMSNorm eps 가 10⁻⁶ · 10⁻⁵ 이 아니면 그 까닭(#16 — 모듈의 상수와 다른 eps 로 셈하지 않게)
  - eps 10⁻⁶ · 10⁻⁵ · 열쇠 없음은 eps 규칙을 지나 다음 검사(모델 크기)에서 멈춘다

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


def 글(x):
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def 머리(메타):
    """GGUF v3 머리 — 텐서 0 개, 메타는 (열쇠, 값) — 값이 글이면 string(8), 실수면 float32(6)."""
    몸 = b"GGUF" + struct.pack("<I", 3) + struct.pack("<q", 0) + struct.pack("<q", len(메타))
    for 열쇠, 값 in 메타:
        if isinstance(값, str):
            몸 += 글(열쇠) + struct.pack("<I", 8) + 글(값)
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
