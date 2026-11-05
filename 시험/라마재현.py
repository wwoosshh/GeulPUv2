#!/usr/bin/env python3
"""llama.cpp 실행의 비트 재현(문서/라마재현.md) — 재현 호스트(엔진/라마재현.gl)의 로짓을 llama.cpp 의 로짓과 비트까지.

판정 1-2(기본): llama.cpp 가 프롬프트를 마친 뒤의 상태(기록 — 생성 1걸음의 노드 입력: 합성곱 · 재귀 상태, KV 캐시)와 그래프의
상수(op_params · GGUF)로 상태 파일을 만들고, 서버가 걸음마다 넣은 토큰을 넣어 호스트를 돌린다. 하네스의 로짓(같은 DLL · 같은 실행 모양 —
서버의 로짓과 같다)의 걸음 1 … 과 견준다.
판정 2-2 · 2-3(--프롬프트부터): 상태 0 에서 프롬프트 토큰만으로 — 서버와 같은 덩어리(체크포인트: 끝에서 4 + 512 · 4 토큰 앞, 덩어리
2,048 이하) · ubatch(512)로 나눠 넣고, 프롬프트 끝 로짓과 걸음마다의 로짓을 하네스 로짓의 줄 0 … 과 견준다. 기록 폴더는 상수만 쓴다.

  python -I 시험/라마재현.py <라마재현.exe> <라마재현.ptx> <기록 폴더> <모델.gguf> <로짓.bin> <서버 응답.json> [걸음 수, 기본 8] [--프롬프트부터]
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


def 서버덩어리(n, n_batch=2048, n_ubatch=512):
    """llama-server b12008(server-context.cpp)의 프롬프트 나눔 — 체크포인트를 위해 끝에서 4 + n_ubatch, 4 토큰 앞에서 끊고 덩어리는
    n_batch 이하. 덩어리마다 llama_decode 가 ubatch(n_ubatch)로 나눈다. 돌려주는 것: ubatch 토큰 수들"""
    offs = (min(n_batch, 4 + n_ubatch), min(n_batch, 4))
    ub, start = [], 0
    while start < n:
        end = start
        while end < n:
            end += 1
            if end - start >= n_batch or (n - end) in offs:
                break
        m = end - start
        ub += [n_ubatch] * (m // n_ubatch) + ([m % n_ubatch] if m % n_ubatch else [])
        start = end
    return ub


def 상태파일(R, M, 경로, 병렬=7, 프롬프트부터=False, KV칸=0):
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
    if 프롬프트부터:
        pos0 = 0
    else:
        KV칸 = R.텐서[fa_n["srcs"][1]["ptr"]]["ne"][1]
    머리 = np.zeros(16, np.int32)
    머리[:15] = [표시, 층수, pos0, KV칸, 비트(eps), rq["op_params"][0], sq["op_params"][0],
               비트((f32(1) / np.sqrt(f32(128))).astype(f32)), 비트(ts), op[6], op[8], op[1], fa["op_params"][0], 병렬,
               1 if 프롬프트부터 else 0]
    덩이 = [머리.tobytes()]
    for L in range(0 if 프롬프트부터 else 층수):
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
    프롬프트부터 = "--프롬프트부터" in sys.argv
    인자 = [a for a in sys.argv[1:] if a != "--프롬프트부터"]
    exe, ptx, 폴더, 모델, 로짓, 응답 = 인자[:6]
    exe, ptx = os.path.abspath(exe), os.path.abspath(ptx)
    걸음수 = int(인자[6]) if len(인자) > 6 else 8
    R, M = 기록(폴더), gguf읽기.GGUF(모델)
    run = json.load(open(응답, encoding="utf-8"))
    gen = run["response"]["tokens"]
    with tempfile.TemporaryDirectory() as td:
        상태 = os.path.join(td, "상태.bin")
        토큰 = os.path.join(td, "토큰.bin")
        if 프롬프트부터:
            prompt = run["prompt_tokens"]
            ub = 서버덩어리(len(prompt))
            KV칸 = (len(prompt) + 걸음수 + 255) // 256 * 256
            pos0, _ = 상태파일(R, M, 상태, 프롬프트부터=True, KV칸=KV칸)
            np.array([len(ub)] + ub + [len(prompt)] + prompt + [걸음수] + gen[:걸음수], np.int32).tofile(토큰)
            print(f"프롬프트 {len(prompt)} 토큰 · ubatch {ub} · KV 칸 {KV칸} · 걸음 {걸음수}", flush=True)
            줄수, 첫줄 = 걸음수 + 1, 0
        else:
            pos0, KV칸 = 상태파일(R, M, 상태)
            np.array([걸음수] + gen[:걸음수], np.int32).tofile(토큰)
            print(f"처음 위치 {pos0} · KV 칸 {KV칸} · 걸음 {걸음수}", flush=True)
            줄수, 첫줄 = 걸음수, 1
        출력 = os.path.join(td, "로짓.bin")
        r = subprocess.run([exe, 모델, ptx, 상태, 토큰, 출력], capture_output=True, text=True, encoding="utf-8")
        print(r.stdout, end="")
        if r.returncode != 0:
            print(r.stderr)
            print("모두 0: False")
            return 1
        got = np.fromfile(출력, f32).reshape(줄수, -1)
    want = np.fromfile(로짓, f32).reshape(-1, got.shape[1])
    나쁨 = 0
    for k in range(줄수):
        d = int(np.count_nonzero(got[k].view(np.uint32) != want[k + 첫줄].view(np.uint32)))
        나쁨 += d
        am = int(np.argmax(got[k]))
        j = k + 첫줄                                                    # 하네스 줄 j 는 서버 토큰 gen[j] 를 낸다
        이름 = "프롬프트 끝" if j == 0 else f"생성 {j}걸음(토큰 {gen[j - 1]})"
        if k < 10 or d:
            print(f"로짓 {이름}: 값 {got.shape[1]} · 불일치 {d} · 가장 큰 값의 토큰 {am}"
                  f"(서버 토큰 {gen[j] if j < len(gen) else None})", flush=True)
    print(f"합계: 값 {got.size} · 불일치 {나쁨}")
    print(f"모두 0: {나쁨 == 0}")
    return 0 if 나쁨 == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
