#!/usr/bin/env python3
"""0.13.0 12-3h-4 — Gemma 4 의 RoPE 표(구조/gemma4/로프표.gl — 엔진/로프.gl 의 함수로 넓은 표를 짓는 길)를 CPU 참조(계약/정수계약.rope표)와
바이트까지 견준다(GPU 없이 — 시험/로프표.py 의 꼴).

  python -I 시험/젬마로프표.py <시험_로프표.exe — 구조/gemma4/시험_로프표.gl> <작업 폴더> [--gguf <gemma4 gguf>] [--길이 4096]

경우: 윈도 층(θ 10⁴ · 헤드 256 — 표 폭 128)과 전체 층(θ 10⁶ · 헤드 512 — 표 폭 256 · rope_freqs). rope_freqs 는 --gguf 의 것, 없으면 12B 의
값(1 이 64 칸 · 10³⁰ 이 192 칸 — 부분 회전 0.25). 마지막 줄 '젬마 RoPE 표: 모두 같음' 또는 '젬마 RoPE 표: 다름 N'(종료 1).
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.normpath(os.path.join(HERE, "..")), "계약"))
import gguf읽기 as GR        # noqa: E402
import 정수계약 as C         # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    exe, 작업 = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    L = int(sys.argv[sys.argv.index("--길이") + 1]) if "--길이" in sys.argv else 4096
    os.makedirs(작업, exist_ok=True)
    if "--gguf" in sys.argv:
        g = GR.GGUF(sys.argv[sys.argv.index("--gguf") + 1])
        rf = g.풀기("rope_freqs.weight").astype(np.float32)
        θ윈도, θ전체 = float(g.meta["gemma4.rope.freq_base_swa"]), float(g.meta["gemma4.rope.freq_base"])
    else:
        rf = np.concatenate([np.ones(64), np.full(192, 1e30)]).astype(np.float32)
        θ윈도, θ전체 = 10000.0, 1000000.0
    다름 = 0
    for 이름, θ, hd, 인자 in (("윈도", θ윈도, 256, None), ("전체", θ전체, 512, rf)):
        반 = hd // 2
        rf경로 = "-"
        if 인자 is not None:
            rf경로 = os.path.join(작업, "rope_freqs.bin")
            np.asarray(인자, np.float32).tofile(rf경로)
        출력 = os.path.join(작업, f"표_{이름}.bin")
        r = subprocess.run([exe, str(L), repr(θ), str(반), rf경로, 출력], capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            print(f"{이름}: 시험_로프표 종료 {r.returncode} — {r.stdout.strip()} {r.stderr.strip()}")
            다름 += 1
            continue
        b = np.fromfile(출력, np.float32)
        ce, se = b[:L * 반].reshape(L, 반), b[L * 반:].reshape(L, 반)
        cr, sr = C.rope표(L, θ, hd=hd, 인자=인자)
        dc = int(np.count_nonzero(ce.view(np.int32) != np.asarray(cr, np.float32).view(np.int32)))
        ds = int(np.count_nonzero(se.view(np.int32) != np.asarray(sr, np.float32).view(np.int32)))
        print(f"{이름}(θ {θ:g} · 헤드 {hd} · 위치 {L}): 다른 칸 cos {dc} · sin {ds} — {r.stdout.strip()}", flush=True)
        다름 += dc + ds
    print("젬마 RoPE 표: 모두 같음" if 다름 == 0 else f"젬마 RoPE 표: 다름 {다름}")
    return 1 if 다름 else 0


if __name__ == "__main__":
    sys.exit(main())
