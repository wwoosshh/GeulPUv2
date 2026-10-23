#!/usr/bin/env python3
"""0.12.0 12-3c — 16비트 블록 가중치(계약 1.3 — F16 · BF16 · F32)의 판(커널생성 … --W16 — 커널/생성W16.gl)을 CPU 참조(계약/정수계약.행렬곱16)와
비트까지. 무작위 가중치(행 · 블록마다 크기가 다르게, 0 블록, 블록 최댓값보다 아주 작은 값)와 무작위 활성값으로:
  타일 판 11 개 × 끝손질 "" · "_잔차", 한 행 줄 판 × "" · "_잔차" · "_SwiGLU정수", SwiGLU정수로_W16(타일 판 "" 의 gate · up 출력에서),
  임베딩 F16 · BF16 · F32 와 그 칸 판(비정규수 값까지 비트 그대로).
모양: 행이 타일의 배수가 아닌 것(100), 열이 64 의 배수가 아닌 것(1000), 깊이가 32 의 배수가 아닌 것(1100 — 0 을 채운 1120 · 블록 35 개 —
줄 판의 마지막 판이 블록 넷보다 적다).

  python -I 시험/판16.py <ptx> <output.json> [--fast] [--extreme] [--width D]
<ptx>: 커널생성 qwen2큰 … <출력.gl> --W16 [--빠른] 의 모듈. --fast: 모듈이 --빠른(자리 하나 — 계약 1.3 의 3)일 때. --width: 모듈의 너비(임베딩
판의 행 길이 — 기본 1536). --extreme: 활성값 블록의 크기를 2⁻⁹⁰ … 2⁻¹²⁰ · 0 에서, 가중치 블록의 절반을 2⁻⁹⁰ … 2⁻¹¹⁰ 둘레로 — 지수가 −98 에서
잘리고 2^(E_x − P) · 2^(E_w − 13) 의 곱이 2⁻¹²⁶ 아래로 내려가는 칸(규칙 R 의 경계).
마지막 줄 "경우 … · 모두 0: True|False".
"""
import ctypes
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "계약"))
sys.path.insert(0, HERE)
import 쿠다드라이버 as G          # noqa: E402
import 시험장치               # noqa: E402
import 정수계약 as C        # noqa: E402
from 머리128 import 자리바이트    # noqa: E402

f32 = np.float32
u64, i64 = G.c_u64, G.c_i64
# 판 표 (생성타일.gl 의 판표준비 — 이름 · WM · WN · MT · NT)
판들 = [("정수타일선형", 2, 2, 2, 8), ("정수좁은타일선형", 2, 2, 2, 4), ("정수작은타일선형", 2, 2, 1, 4), ("정수아주작은타일선형", 2, 2, 1, 2),
       ("정수넓은타일선형", 2, 4, 2, 8), ("정수긴타일선형", 1, 4, 2, 8), ("정수긴4타일선형", 1, 4, 2, 8), ("정수긴2타일선형", 1, 4, 2, 8),
       ("정수깊은타일선형", 1, 4, 4, 4), ("정수깊은넓은타일선형", 2, 2, 4, 4), ("정수세판타일선형", 2, 2, 2, 8)]


def 무작위가중치(r, N, K, 극단):
    """[N][K] fp32 — 행마다 크기 10^U(−3, 1), 블록(32 칸)마다 2^U(−6, 6), 블록의 2 % 는 0, 칸의 5 % 는 2⁻²⁰ 배(블록 최댓값보다 아주 작다).
    극단: 블록의 절반을 최댓값이 2⁻⁹⁰ · 2⁻¹⁰⁰ · 2⁻¹¹⁰ 둘레가 되게."""
    KB = (K + 31) // 32
    w = r.standard_normal((N, KB * 32)) * 10.0 ** r.uniform(-3, 1, (N, 1))
    w = w.reshape(N, KB, 32) * (2.0 ** r.uniform(-6, 6, (N, KB)))[:, :, None]
    w = w * (r.random((N, KB)) >= 0.02)[:, :, None]
    w = np.where(r.random((N, KB, 32)) < 0.05, w * 2.0 ** -20, w)
    if 극단:
        크기 = r.choice(np.array([2.0 ** -90, 2.0 ** -100, 2.0 ** -110]), (N, KB))
        m = np.abs(w).max(2, keepdims=True)
        작게 = (r.random((N, KB)) < 0.5)[:, :, None] & (m > 0)
        w = np.where(작게, w / np.where(m > 0, m, 1.0) * 크기[:, :, None], w)
    return w.reshape(N, KB * 32)[:, :K].astype(f32)


def 극단활성값(r, M, K):
    """[M][K] — 블록(32 칸)마다 크기를 2⁻⁹⁰ · 2⁻⁹⁶ · 2⁻¹⁰² · 2⁻¹⁰⁸ · 2⁻¹¹⁴ · 2⁻¹²⁰ · 0 에서 고른다(판23.py 와 같다)."""
    KB = (K + 31) // 32
    크기 = r.choice(np.array([2.0 ** -90, 2.0 ** -96, 2.0 ** -102, 2.0 ** -108, 2.0 ** -114, 2.0 ** -120, 0.0]), (M, KB))
    x = r.standard_normal((M, KB, 32)) * 크기[:, :, None]
    return x.reshape(M, KB * 32)[:, :K].astype(f32)


def 깔기16(wn, wE):
    """엔진 배치 — 값 [KB][N][64 바이트](자리 1 의 32 바이트 · 자리 0 의 32 바이트, 칸 차례), 지수 [KB][Np] 바이트(E + 114, 남는 칸 0)."""
    N, Kp = wn.shape
    KB = Kp // 32
    d0 = ((wn + 128) & 255) - 128
    d1 = (wn - d0) >> 8
    assert np.abs(d1).max() <= 64
    값 = np.empty((KB, N, 64), np.int8)
    값[:, :, :32] = d1.reshape(N, KB, 32).transpose(1, 0, 2)
    값[:, :, 32:] = d0.reshape(N, KB, 32).transpose(1, 0, 2)
    Np = (N + 63) // 64 * 64
    지수 = np.zeros((KB, Np), np.uint8)
    지수[:, :N] = (np.asarray(wE).T + 114).astype(np.uint8)
    return 값.tobytes(), 지수.tobytes()


def 채움(X, Kp):
    """깊이를 Kp 로 0 채움."""
    return np.concatenate([X, np.zeros((X.shape[0], Kp - X.shape[1]), f32)], 1) if X.shape[1] < Kp else X


def 다른칸(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.dtype == f32:
        a, b = a.view(np.int32), np.asarray(b, f32).view(np.int32)
    return int(np.count_nonzero(a != b))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ptx, 출력 = sys.argv[1:3]
    자리수 = 1 if "--fast" in sys.argv else 2
    극단 = "--extreme" in sys.argv
    너비 = int(sys.argv[sys.argv.index("--width") + 1]) if "--width" in sys.argv else 1536
    dr = 시험장치.드라이버()
    mod = dr.모듈(open(ptx, "rb").read())
    fn = {}

    def 띄우기(이름, grid, block, params):
        if 이름 not in fn:
            fn[이름] = dr.함수(mod, 이름)
        ps = [u64(p) if 참조 else i64(p) for 참조, p in params]
        args = (ctypes.c_void_p * len(ps))(*[ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in ps])
        rr = dr.cu.cuLaunchKernel(fn[이름], grid[0], grid[1], 1, block, 1, 1, 0, None, args, None)
        assert rr == 0, (이름, rr)

    def up(b):
        b = b if isinstance(b, bytes) else np.ascontiguousarray(b).tobytes()
        p = dr.할당(len(b))
        dr.확인(dr.cu.cuMemcpyHtoD_v2(p, b, len(b)))
        return p

    def 놓기():
        dr.맞추기()
        for p in dr.할당들:
            dr.확인(dr.cu.cuMemFree_v2(u64(p)))
        dr.할당들 = []

    결과 = {"자리수": 자리수, "극단": 극단, "경우": []}

    def 적기(한):
        결과["경우"].append(한)
        if sum(한["다른 칸"].values()):
            print("다름!", 한, flush=True)

    r = np.random.default_rng(16)
    M = 100
    F = 2048
    # 이름 → (N, K, 끝손질) — gu 는 [gate 32 | up 32] 64 열씩(엔진의 gate · up 배치)
    행렬들 = [("qkv", 2048, 1536, ""), ("o", 1536, 1536, "_잔차"), ("gu", 2 * F, 1536, "_SwiGLU정수"), ("down", 1536, F, "_잔차"),
            ("모양", 1000, 1100, "")]
    for 이름, N, K, 끝 in 행렬들:
        w = 무작위가중치(r, N, K, 극단)
        wn, wE = C.가중치16(w)
        Kp = wn.shape[1]
        KB = Kp // 32
        Np = (N + 63) // 64 * 64
        값, 지수 = 깔기16(wn, wE)
        p값, p지 = up(값), up(지수)
        치 = (r.standard_normal(N) * 0.1).astype(f32) if 끝 == "" else np.zeros(N, f32)
        p치 = up(치)
        X = 극단활성값(r, M, K) if 극단 else (r.standard_normal((M, K)) * 10.0 ** r.uniform(-1, 1, (M, 1))).astype(f32)
        바이트, M0 = 자리바이트(채움(X, Kp), 자리수)
        폭 = (M + 63) // 64 * 64
        정보 = np.zeros((KB, 폭), np.int32)
        정보[:, :M] = M0.T
        p자, p정 = up(바이트), up(정보)
        잔 = (r.standard_normal((M, N)) * 3.0).astype(f32)
        y = C.행렬곱16(X, wn, wE, 치, 자리수=자리수)
        참 = C.더(잔, y) if 끝 == "_잔차" else y
        타일끝 = "_잔차" if 끝 == "_잔차" else ""
        # 타일 판 — 띄움의 열 블록은 가짜 열(2N)로
        p출 = dr.할당(M * N * 4)
        for 판, WM, WN, MT, NT in 판들:
            BM, BN, T = 16 * MT * WM, 8 * NT * WN, 32 * WM * WN
            kn = 판 + 타일끝 + "_W16"
            grid = ((M + BM - 1) // BM, (2 * N + BN - 1) // BN)
            앞 = [(1, p자), (1, p정), (1, p값), (1, p지), (1, p치)]
            뒤 = [(0, M), (0, Kp), (0, N), (0, Np), (0, 0), (0, 0), (0, N)]
            if 타일끝 == "_잔차":
                dr.확인(dr.cu.cuMemcpyHtoD_v2(p출, 잔.ctypes.data, 잔.nbytes))
                띄우기(kn, grid, T, 앞 + [(1, p출), (1, p출)] + 뒤)
            else:
                띄우기(kn, grid, T, 앞 + [(1, p출)] + 뒤)
            dr.맞추기()
            글 = dr.내리기(p출, np.empty((M, N), f32))
            적기({"가중치": 이름, "판": kn, "행": M, "K": K, "N": N, "다른 칸": {"출력": 다른칸(글, 참)}})
            if 이름 == "gu" and 판 == "정수타일선형":
                # SwiGLU정수로_W16 — 이 판의 gate · up 출력에서 down 의 자리 · 정보
                gcol = np.concatenate([np.arange(64 * j, 64 * j + 32) for j in range(N // 64)])
                참자리, 참정보 = 자리바이트(C.swiglu(y[:, gcol], y[:, gcol + 32]), 자리수)
                p다자, p다정 = dr.할당(자리수 * M * F), dr.할당((F // 32) * 폭 * 4)
                띄우기("SwiGLU정수로_W16", ((F // 32 + 31) // 32, M), 256, [(1, p출), (1, p다자), (1, p다정), (0, M), (0, F)])
                dr.맞추기()
                다 = {"자리": 다른칸(dr.내리기(p다자, np.empty((자리수, M, F), np.int8)), 참자리),
                     "정보": 다른칸(dr.내리기(p다정, np.empty((F // 32, 폭), np.int32))[:, :M].T, 참정보)}
                적기({"가중치": 이름, "판": "SwiGLU정수로_W16", "행": M, "K": K, "N": N, "다른 칸": 다})
        # 한 행 줄 판(생성 그래프 — 행 하나): 블록 하나가 열 W 개
        W = 64 if 끝 == "_SwiGLU정수" else 32
        kn = "정수줄하나" + 끝 + "_W16"
        X1 = X[:1]
        바이트1, M01 = 자리바이트(채움(X1, Kp), 자리수)
        정보1 = np.zeros((KB, 64), np.int32)
        정보1[:, :1] = M01.T
        p자1, p정1 = up(바이트1), up(정보1)
        앞 = [(1, p자1), (1, p정1), (1, p값), (1, p지), (1, p치)]
        뒤 = [(0, 1), (0, Kp), (0, N)]
        grid = ((N + W - 1) // W, 1)
        if 끝 == "_SwiGLU정수":
            gcol = np.concatenate([np.arange(64 * j, 64 * j + 32) for j in range(N // 64)])
            참자리, 참정보 = 자리바이트(C.swiglu(y[:1, gcol], y[:1, gcol + 32]), 자리수)
            p다자, p다정 = dr.할당(자리수 * F), dr.할당((F // 32) * 64 * 4)
            띄우기(kn, grid, 128, 앞 + [(1, p다정), (1, p다자)] + 뒤)
            dr.맞추기()
            다 = {"자리": 다른칸(dr.내리기(p다자, np.empty((자리수, 1, F), np.int8)), 참자리),
                 "정보": 다른칸(dr.내리기(p다정, np.empty((F // 32, 64), np.int32))[:, :1].T, 참정보)}
        else:
            p출1 = dr.할당(N * 4)
            if 끝 == "_잔차":
                p잔1 = up(잔[:1])
                띄우기(kn, grid, 128, 앞 + [(1, p잔1), (1, p출1)] + 뒤)
            else:
                띄우기(kn, grid, 128, 앞 + [(1, p출1)] + 뒤)
            dr.맞추기()
            다 = {"출력": 다른칸(dr.내리기(p출1, np.empty((1, N), f32)), 참[:1])}
        적기({"가중치": 이름, "판": kn, "행": 1, "K": K, "N": N, "다른 칸": 다})
        print(이름, "끝 — 경우", sum(1 for c in 결과["경우"] if c["가중치"] == 이름), flush=True)
        놓기()
    # 임베딩 — 파일의 값 그대로(F16 · BF16 → fp32 는 정확 · F32 는 비트 그대로). 비정규수 값을 섞는다
    V, D = 1000, 너비
    토큰 = r.integers(0, V, 37).astype(np.int64)
    표들 = {}
    h = (r.standard_normal((V, D)) * 0.05).astype(np.float16)
    작은 = r.random((V, D)) < 0.05
    h[작은] = (r.integers(1, 1024, int(작은.sum())) * 2.0 ** -24).astype(np.float16)            # 반실수 비정규수 k · 2⁻²⁴
    표들["F16"] = (h.tobytes(), h.astype(f32))
    b = ((r.standard_normal((V, D)) * 0.05).astype(f32).view(np.uint32) >> 16).astype(np.uint16)
    작은 = r.random((V, D)) < 0.05
    b[작은] = r.integers(1, 128, int(작은.sum())).astype(np.uint16)                               # BF16 비정규수(지수 0)
    표들["BF16"] = (b.tobytes(), (b.astype(np.uint32) << 16).view(f32))
    x32 = (r.standard_normal((V, D)) * 0.05).astype(f32)
    x32[r.random((V, D)) < 0.05] = np.float32(2.0 ** -130)                                     # fp32 비정규수
    표들["F32"] = (x32.tobytes(), x32)
    for 형식, (표, 참표) in 표들.items():
        p표 = up(표)
        p척 = dr.할당(64)
        참 = 참표[토큰]
        p토, p출 = up(토큰), dr.할당(37 * D * 4)
        띄우기("임베딩_" + 형식, ((37 * D + 255) // 256, 1), 256, [(1, p토), (1, p표), (1, p척), (1, p출), (0, 37)])
        dr.맞추기()
        적기({"가중치": "낱말표", "판": "임베딩_" + 형식, "행": 37, "다른 칸": {"출력": 다른칸(dr.내리기(p출, np.empty((37, D), f32)), 참)}})
        # 칸 판 — 토큰[걸음칸[1] + r]
        토큰칸 = np.concatenate([np.zeros(5, np.int64), 토큰])
        p토칸, p걸음 = up(토큰칸), up(np.array([0, 5], np.int64))
        p출칸 = dr.할당(37 * D * 4)
        띄우기("임베딩칸_" + 형식, ((37 * D + 255) // 256, 1), 256, [(1, p토칸), (1, p표), (1, p척), (1, p출칸), (0, 37), (1, p걸음)])
        dr.맞추기()
        적기({"가중치": "낱말표", "판": "임베딩칸_" + 형식, "행": 37, "다른 칸": {"출력": 다른칸(dr.내리기(p출칸, np.empty((37, D), f32)), 참)}})
        놓기()
    결과["모두 0"] = all(v == 0 for c in 결과["경우"] for v in c["다른 칸"].values())
    print("경우", len(결과["경우"]), "· 모두 0:", 결과["모두 0"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
