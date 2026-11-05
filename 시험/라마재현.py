#!/usr/bin/env python3
"""llama.cpp 실행의 비트 재현(문서/라마재현.md) — 판정 1-2: 재현 호스트(엔진/라마재현.gl)의 생성 걸음 로짓을 llama.cpp 의 로짓과 비트까지.

llama.cpp 가 프롬프트를 마친 뒤의 상태(기록 — 생성 1걸음의 노드 입력: 합성곱 · 재귀 상태, KV 캐시)와 그래프의 상수(op_params · GGUF)로
상태 파일을 만들고, 서버가 걸음마다 넣은 토큰을 넣어 호스트를 돌린다. 하네스 A 의 로짓(같은 DLL · 같은 실행 모양 — 서버의 로짓과 같다)과
걸음마다 견준다.

  python -I 시험/라마재현.py <라마재현.exe> <라마재현.ptx> <기록 폴더> <모델.gguf> <로짓.bin> <서버 응답.json> [걸음 수, 기본 8]
마지막 줄 "모두 0: True|False".
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "계약"))
sys.path.insert(0, HERE)
import gguf읽기             # noqa: E402
from 라마재현판 import 기록   # noqa: E402

f32 = np.float32
표시 = 0x3150524C


def 비트(x):
    return int(np.array([x], f32).view(np.int32)[0])


def 찾기(R, op, 앞):
    for n in R.노드들:
        rec = R.텐서[n["ptr"]]
        if rec["op"] == op and n["srcs"] and not n["srcs"][0].get("weight"):
            if R.텐서[n["srcs"][0]["ptr"]]["name"].startswith(앞):
                return n, rec
    raise KeyError((op, 앞))


def 상태파일(R, M, 경로, 병렬=7):
    층수 = M.meta["qwen35.block_count"]
    eps = f32(M.meta["qwen35.attention.layer_norm_rms_epsilon"])
    _, rq = 찾기(R, "RMS_NORM", "q_conv-0")                         # l2 정규화 = rms_norm(eps / 128)
    _, sq = R.노드("q_conv_predelta-0")                               # SCALE(눈금, 치우침 0)
    assert sq["op_params"][1] == 0
    rope_n = [n for n in R.노드들 if n["name"] == "Qcur-3" and R.텐서[n["ptr"]]["op"] == "ROPE"][0]
    op = R.텐서[rope_n["ptr"]]["op_params"]
    pos0 = int(R.입력(rope_n, 1).reshape(-1)[0])
    fb = np.array([op[5]], np.int32).view(f32)[0]
    ts = f32(np.power(np.float64(fb), -2.0 / op[1]))                 # 호스트의 powf(freq_base, −2/n) — 두 셈이 같았다(타당성 시험)
    fa_n, fa = 찾기(R, "FLASH_ATTN_EXT", "Qcur-3 ")
    KV칸 = R.텐서[fa_n["srcs"][1]["ptr"]]["ne"][1]
    머리 = np.zeros(16, np.int32)
    머리[:14] = [표시, 층수, pos0, KV칸, 비트(eps), rq["op_params"][0], sq["op_params"][0],
               비트((f32(1) / np.sqrt(f32(128))).astype(f32)), 비트(ts), op[6], op[8], op[1], fa["op_params"][0], 병렬]
    덩이 = [머리.tobytes()]
    for L in range(층수):
        if (L + 1) % 4 == 0:
            fn, _ = 찾기(R, "FLASH_ATTN_EXT", f"Qcur-{L} ")
            for i in (1, 2):
                c = np.ascontiguousarray(np.transpose(R.입력(fn, i)[0], (1, 0, 2))).astype(np.float16)
                덩이.append(c.tobytes())
        else:
            n, _ = R.노드(f"conv_input-{L}")
            덩이.append(np.ascontiguousarray(R.입력(n, 0).astype(f32).reshape(-1)).tobytes())
            g, _ = 찾기(R, "GATED_DELTA_NET", f"q_conv_predelta-{L}")
            덩이.append(np.ascontiguousarray(R.입력(g, 5).astype(f32).reshape(-1)).tobytes())
    with open(경로, "wb") as f:
        for d in 덩이:
            f.write(d)
    return pos0, KV칸


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    exe, ptx, 폴더, 모델, 로짓, 응답 = sys.argv[1:7]
    exe, ptx = os.path.abspath(exe), os.path.abspath(ptx)
    걸음수 = int(sys.argv[7]) if len(sys.argv) > 7 else 8
    R, M = 기록(폴더), gguf읽기.GGUF(모델)
    gen = json.load(open(응답, encoding="utf-8"))["response"]["tokens"]
    with tempfile.TemporaryDirectory() as td:
        상태 = os.path.join(td, "상태.bin")
        pos0, KV칸 = 상태파일(R, M, 상태)
        토큰 = os.path.join(td, "토큰.bin")
        np.array([걸음수] + gen[:걸음수], np.int32).tofile(토큰)
        출력 = os.path.join(td, "로짓.bin")
        print(f"처음 위치 {pos0} · KV 칸 {KV칸} · 걸음 {걸음수}", flush=True)
        r = subprocess.run([exe, 모델, ptx, 상태, 토큰, 출력], capture_output=True, text=True, encoding="utf-8")
        print(r.stdout, end="")
        if r.returncode != 0:
            print(r.stderr)
            print("모두 0: False")
            return 1
        got = np.fromfile(출력, f32).reshape(걸음수, -1)
    want = np.fromfile(로짓, f32).reshape(-1, got.shape[1])
    나쁨 = 0
    for k in range(걸음수):
        d = int(np.count_nonzero(got[k].view(np.uint32) != want[k + 1].view(np.uint32)))
        나쁨 += d
        am = int(np.argmax(got[k]))
        print(f"로짓 생성 {k + 1}걸음(토큰 {gen[k]}): 값 {got.shape[1]} · 불일치 {d} · 가장 큰 값의 토큰 {am}"
              f"(서버의 다음 토큰 {gen[k + 1] if k + 1 < len(gen) else None})", flush=True)
    print(f"합계: 값 {got.size} · 불일치 {나쁨}")
    print(f"모두 0: {나쁨 == 0}")
    return 0 if 나쁨 == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
