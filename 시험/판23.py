#!/usr/bin/env python3
"""23a (1) — 7B 모듈(qwen2Q8_7b · qwen2Q4_7b)의 행렬곱 판을 CPU 참조(계약/정수계약.행렬곱블라스 — q 는 Q8_0 의 바이트 · Q4_0 의 니블 − 8)와
비트까지. 7B 층 0 의 실제 가중치(GGUF 의 형식 그대로 — 엔진과 같은 배치로 깔아)와 무작위 활성값으로:
  타일 판 11 개 × 끝손질 ""(qkv + 치우침) · "_잔차"(o · down) · "_SwiGLU정수"(gate·up — 열이 64 인 판), 열먼저(잔차 · SwiGLU), 로짓 판(""),
  줄 판(정수줄자리 × 끝손질 셋 × 행 1 … 4 · 정수줄하나 × 끝손질 셋 × 행 1), 임베딩(낱말표의 q · d),
  RMS(RMS정규화정수줄 · RMS정규화정수줄펼침 — 행 9 개).

  python -I 시험/판23.py <ptx> <7B gguf> <출력.json> [--빠르게] [--극단]
--극단(31단계 — 규칙 R 의 경계): 척도 d 의 절반을 2⁻¹⁴ 배(반실수 비정규수 둘레)로, 활성값은 블록마다 크기를 2⁻⁹⁰ … 2⁻¹²⁰ 과 0 으로
섞는다 — 지수 E 가 −98 에서 잘리고 P · d 가 2⁻¹²⁶ 아래로 내려가는 칸이 생긴다(예전의 타일 끝셈 S · (P · d) 가 줄 판과 갈리던 경우).
"""
import ctypes
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
for _d in ("계약",):
    sys.path.insert(0, os.path.join(ROOT, _d))
sys.path.insert(0, HERE)
import 쿠다드라이버 as G          # noqa: E402
import 시험장치               # noqa: E402
import gguf읽기 as GR       # noqa: E402
import 정수계약 as C        # noqa: E402
from 머리128 import 자리바이트    # noqa: E402

f32 = np.float32
u64, i64 = G.c_u64, G.c_i64
# 판 표 (생성타일.gl 의 판표준비 — 이름 · WM · WN · MT · NT)
판들 = [("정수타일선형", 2, 2, 2, 8), ("정수좁은타일선형", 2, 2, 2, 4), ("정수작은타일선형", 2, 2, 1, 4), ("정수아주작은타일선형", 2, 2, 1, 2),
       ("정수넓은타일선형", 2, 4, 2, 8), ("정수긴타일선형", 1, 4, 2, 8), ("정수긴4타일선형", 1, 4, 2, 8), ("정수긴2타일선형", 1, 4, 2, 8),
       ("정수깊은타일선형", 1, 4, 4, 4), ("정수깊은넓은타일선형", 2, 2, 4, 4), ("정수세판타일선형", 2, 2, 2, 8)]
# 32단계 — 줄 타일 판 표 (생성줄타일.gl 의 줄타일표준비 — 이름 · W · NW · BK · S). Q8_0 · Q4_0 모듈에 있다
줄타일들 = [("정수줄타일", 64, 4, 4, 4), ("정수줄타일깊은", 64, 4, 8, 2), ("정수줄타일좁은", 32, 2, 8, 3), ("정수줄타일좁은얕은", 32, 2, 4, 6),
         ("정수줄타일넓은", 128, 4, 4, 2)]


def 양자읽기(g, name):
    """GGUF Q8_0 · Q4_0 텐서 → (q [N][K] int8 (Q4_0 은 니블 − 8), d [N][K/32] 짧은실수, 형식)."""
    t, dims, raw = g.원(name)
    K, N = dims
    if t == GR.Q8_0:
        bl = np.asarray(raw).reshape(N, K // 32, 34)
        d = np.ascontiguousarray(bl[:, :, :2]).view(np.float16).reshape(N, K // 32).astype(f32)
        return np.ascontiguousarray(bl[:, :, 2:]).view(np.int8).reshape(N, K), d, "Q8_0"
    assert t == GR.Q4_0, (name, t)
    bl = np.asarray(raw).reshape(N, K // 32, 18)
    d = np.ascontiguousarray(bl[:, :, :2]).view(np.float16).reshape(N, K // 32).astype(f32)
    qs = bl[:, :, 2:]
    q = (np.concatenate([qs & 15, qs >> 4], 2).astype(np.int16) - 8).astype(np.int8).reshape(N, K)
    return q, d, "Q4_0"


def 깔기(q, d, 형식):
    """엔진 배치의 바이트 — 값 [KB][N][32 바이트](Q8_0) 또는 [KB][N][16 바이트](Q4_0 — 낱말 j 의 바이트 m: 아래 니블 q[8j + m] + 8,
    위 니블 q[8j + 4 + m] + 8), 척도 [KB][Np] 반실수(남는 칸 0)."""
    N, K = q.shape
    KB = K // 32
    Np = (N + 63) // 64 * 64
    if 형식 == "Q8_0":
        값 = np.ascontiguousarray(q.reshape(N, KB, 32).transpose(1, 0, 2))
    else:
        u = (q.astype(np.int16) + 8).astype(np.uint8).reshape(N, KB, 4, 2, 4).transpose(1, 0, 2, 3, 4)   # [KB][N][j][h][m]
        값 = np.ascontiguousarray(u[:, :, :, 0, :] | (u[:, :, :, 1, :] << 4))
    척 = np.zeros((KB, Np), np.float16)
    척[:, :N] = d.T.astype(np.float16)
    return 값.tobytes(), 척.tobytes()


def 극단척도(r, d):
    """d 의 절반(무작위 칸)을 2⁻¹⁴ 배로 — 반실수로 반올림한 값(엔진이 쓰는 값 그대로)."""
    작게 = r.random(d.shape) < 0.5
    return np.where(작게, (d * f32(2.0 ** -14)).astype(np.float16).astype(f32), d).astype(f32)


def 극단활성값(r, M, K):
    """[M][K] — 블록(32 칸)마다 크기를 2⁻⁹⁰ · 2⁻⁹⁶ · 2⁻¹⁰² · 2⁻¹⁰⁸ · 2⁻¹¹⁴ · 2⁻¹²⁰ · 0 에서 고른다(지수가 −98 에서 잘리는 블록이 많다)."""
    크기 = r.choice(np.array([2.0 ** -90, 2.0 ** -96, 2.0 ** -102, 2.0 ** -108, 2.0 ** -114, 2.0 ** -120, 0.0]), (M, K // 32))
    x = r.standard_normal((M, K)).reshape(M, K // 32, 32) * 크기[:, :, None]
    return x.reshape(M, K).astype(f32)


def 다른칸(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.dtype == f32:
        a, b = a.view(np.int32), np.asarray(b, f32).view(np.int32)
    return int(np.count_nonzero(a != b))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ptx, gguf, 출력 = sys.argv[1:4]
    빠르게 = "--빠르게" in sys.argv
    극단 = "--극단" in sys.argv
    줄타일만 = "--줄타일만" in sys.argv          # 32단계 — 줄 타일 판만(행 1 … 8)
    자리수 = 1 if ("--자리하나" in sys.argv or "--빠른" in sys.argv) else 2    # 33단계 V1 · GeulPUv2 빠른 계약 F1(커널생성 --빠른 의 모듈)
    dr = 시험장치.드라이버()
    mod = dr.모듈(open(ptx, "rb").read())
    fn = {}

    def 함수(이름):
        if 이름 not in fn:
            fn[이름] = dr.함수(mod, 이름)
        return fn[이름]

    def 띄우기(이름, grid, block, params):
        ps = [u64(p) if 참조 else i64(p) for 참조, p in params]
        args = (ctypes.c_void_p * len(ps))(*[ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in ps])
        r = dr.cu.cuLaunchKernel(함수(이름), grid[0], grid[1], 1, block, 1, 1, 0, None, args, None)
        assert r == 0, (이름, r)

    def up(b):
        b = b if isinstance(b, bytes) else np.ascontiguousarray(b).tobytes()
        p = dr.할당(len(b))
        dr.확인(dr.cu.cuMemcpyHtoD_v2(p, b, len(b)))
        return p

    def 넘는가(kn, 한):
        """31단계 — Vulkan 장치가 돌릴 수 없는 커널(공유 메모리 한도 · 뒷단이 뺀 것)이면 결과에 '장치 한도' 로 적고 참. CUDA 는 늘 거짓"""
        까닭 = 시험장치.한도넘음(dr, kn)
        if 까닭 is None:
            return False
        결과["경우"].append(dict(한, 판=kn, **{"장치 한도": 까닭}))
        print("장치 한도:", kn, 까닭, flush=True)
        return True

    def 놓기():
        dr.맞추기()
        for p in dr.할당들:
            dr.확인(dr.cu.cuMemFree_v2(u64(p)))
        dr.할당들 = []

    g = GR.GGUF(gguf)
    r = np.random.default_rng(23)
    가중 = {}
    qs, ds, fs = zip(*(양자읽기(g, f"blk.0.attn_{a}.weight") for a in ("q", "k", "v")))
    형식 = fs[0]
    가중["qkv"] = (np.concatenate(qs), np.concatenate(ds), np.concatenate([g.풀기(f"blk.0.attn_{a}.bias") for a in ("q", "k", "v")]))
    가중["o"] = 양자읽기(g, "blk.0.attn_output.weight")[:2] + (None,)
    가중["down"] = 양자읽기(g, "blk.0.ffn_down.weight")[:2] + (None,)
    wg, dg, _ = 양자읽기(g, "blk.0.ffn_gate.weight")
    wu, du, _ = 양자읽기(g, "blk.0.ffn_up.weight")
    F = wg.shape[0]
    순 = np.empty(2 * F, np.int64)
    for j in range(F // 32):
        순[64 * j:64 * j + 32] = np.arange(32 * j, 32 * j + 32)
        순[64 * j + 32:64 * j + 64] = F + np.arange(32 * j, 32 * j + 32)
    가중["gu"] = (np.concatenate([wg, wu])[순], np.concatenate([dg, du])[순], None)
    가중["로짓"] = 양자읽기(g, "output.weight")[:2] + (None,)
    if 극단:
        가중 = {이름: (q, 극단척도(r, d), b) for 이름, (q, d, b) in 가중.items()}
    결과 = {"형식": 형식, "극단": 극단, "경우": []}
    print("형식", 형식, "· 극단" if 극단 else "", flush=True)
    M = 100                                                  # 타일의 행(64 의 배수가 아니게 — 가장자리)
    for 이름, (q, d, b) in 가중.items():
        if 빠르게 and 이름 in ("down", "로짓"):
            continue
        N, K = q.shape
        값, 척 = 깔기(q, d, 형식)
        p값, p척 = up(값), up(척)
        치 = b if b is not None else np.zeros(N, f32)
        p치 = up(np.ascontiguousarray(치, f32))
        끝들 = {"qkv": [""], "로짓": [""], "o": ["_잔차"], "down": ["_잔차"], "gu": ["_SwiGLU정수"]}[이름]
        X = 극단활성값(r, M, K) if 극단 else (r.standard_normal((M, K)) * 10.0 ** r.uniform(-1, 1, (M, 1))).astype(f32)
        바이트, M0 = 자리바이트(X, 자리수)
        폭 = (M + 63) // 64 * 64
        정보 = np.zeros((K // 32, 폭), np.int32)
        정보[:, :M] = M0.T
        p자, p정 = up(바이트), up(정보)
        잔 = (r.standard_normal((M, N)) * 3.0).astype(f32)
        p잔 = up(잔)
        y = C.행렬곱블라스(X, q, d, 치, 자리수=자리수)
        if 끝들[0] == "_SwiGLU정수":
            gcol = np.concatenate([np.arange(64 * j, 64 * j + 32) for j in range(N // 64)])
            참자리, 참정보 = 자리바이트(C.swiglu(y[:, gcol], y[:, gcol + 32]), 자리수)
        참 = C.더(잔, y) if 끝들[0] == "_잔차" else y
        Np = (N + 63) // 64 * 64
        # 32단계 — 줄 타일 판(행 1 … 8): 블록 하나가 열 W 개, 스레드 32 · NW (Q8_0 · Q4_0 모듈)
        if 형식 in ("Q8_0", "Q4_0"):
            e = 끝들[0]
            for 판, W, NW, BK, S in 줄타일들:
                if e == "_SwiGLU정수" and W != 64:
                    continue
                kn = ("로짓" if 이름 == "로짓" else "") + 판 + e
                if 넘는가(kn, {"가중치": 이름}):
                    continue
                for m in range(1, 9):
                    바이트m, M0m = 자리바이트(X[:m], 자리수)
                    정보m = np.zeros((K // 32, 64), np.int32)
                    정보m[:, :m] = M0m.T
                    p자m, p정m = up(바이트m), up(정보m)
                    앞 = [(1, p자m), (1, p정m), (1, p값), (1, p척), (1, p치)]
                    뒤 = [(0, m), (0, K), (0, N), (0, Np), (0, 0), (0, 0)]
                    grid = ((N + W - 1) // W, 1)
                    if e == "_SwiGLU정수":
                        p다자, p다정 = dr.할당(자리수 * m * (N // 2)), dr.할당((N // 64) * 64 * 4)
                        띄우기(kn, grid, 32 * NW, 앞 + [(1, p다정), (1, p다자)] + 뒤)
                        dr.맞추기()
                        다 = {"자리": 다른칸(dr.내리기(p다자, np.empty((자리수, m, N // 2), np.int8)), 참자리[:, :m]),
                             "정보": 다른칸(dr.내리기(p다정, np.empty((N // 64, 64), np.int32))[:, :m].T, 참정보[:m])}
                    else:
                        p출 = dr.할당(m * N * 4)
                        if e == "_잔차":
                            p잔m = up(잔[:m])
                            띄우기(kn, grid, 32 * NW, 앞 + [(1, p잔m), (1, p출)] + 뒤)
                        else:
                            띄우기(kn, grid, 32 * NW, 앞 + [(1, p출)] + 뒤)
                        dr.맞추기()
                        다 = {"출력": 다른칸(dr.내리기(p출, np.empty((m, N), f32)), 참[:m])}
                    한 = {"가중치": 이름, "판": kn, "행": m, "K": K, "N": N, "다른 칸": 다}
                    결과["경우"].append(한)
                    if sum(다.values()):
                        print("다름!", 한, flush=True)
        if 줄타일만:
            print(이름, "끝 — 경우", sum(1 for c in 결과["경우"] if c["가중치"] == 이름), flush=True)
            놓기()
            continue
        # 타일 판
        for 판, WM, WN, MT, NT in 판들:
            BM, BN, T = 16 * MT * WM, 8 * NT * WN, 32 * WM * WN
            for 열먼저 in (False, True):
                for e in 끝들:
                    if e == "_SwiGLU정수" and NT != 8:
                        continue
                    if 열먼저 and e == "":
                        continue
                    kn = ("로짓" if 이름 == "로짓" else "") + 판 + ("열먼저" if 열먼저 else "") + e
                    if 넘는가(kn, {"가중치": 이름}):
                        continue
                    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM) if 열먼저 else ((M + BM - 1) // BM, (N + BN - 1) // BN)
                    앞 = [(1, p자), (1, p정), (1, p값), (1, p척), (1, p치)]
                    뒤 = [(0, M), (0, K), (0, N), (0, Np), (0, 0), (0, 0)]
                    if e == "_SwiGLU정수":
                        p다자, p다정 = dr.할당(자리수 * M * (N // 2)), dr.할당((N // 64) * 폭 * 4)
                        띄우기(kn, grid, T, 앞 + [(1, p다정), (1, p다자)] + 뒤)
                        dr.맞추기()
                        글자리 = dr.내리기(p다자, np.empty((자리수, M, N // 2), np.int8))
                        글정보 = dr.내리기(p다정, np.empty((N // 64, 폭), np.int32))[:, :M].T
                        다 = {"자리": 다른칸(글자리, 참자리), "정보": 다른칸(글정보, 참정보)}
                    else:
                        p출 = dr.할당(M * N * 4)
                        if e == "_잔차":
                            dr.확인(dr.cu.cuMemcpyHtoD_v2(p출, 잔.ctypes.data, 잔.nbytes))
                            띄우기(kn, grid, T, 앞 + [(1, p출), (1, p출)] + 뒤)
                        else:
                            띄우기(kn, grid, T, 앞 + [(1, p출)] + 뒤)
                        dr.맞추기()
                        다 = {"출력": 다른칸(dr.내리기(p출, np.empty((M, N), f32)), 참)}
                    한 = {"가중치": 이름, "판": kn, "행": M, "K": K, "N": N, "다른 칸": 다}
                    결과["경우"].append(한)
                    if sum(다.values()):
                        print("다름!", 한, flush=True)
        # 줄 판 (행 1 … 4)
        e = 끝들[0]
        W = 64 if e == "_SwiGLU정수" else 32
        for m in (1, 2, 3, 4):
            Xm = X[:m]
            바이트m, M0m = 자리바이트(Xm, 자리수)
            정보m = np.zeros((K // 32, 64), np.int32)
            정보m[:, :m] = M0m.T
            p자m, p정m = up(바이트m), up(정보m)
            p잔m = up(잔[:m]) if e == "_잔차" else None
            if not 넘는가("정수줄자리" + e, {"가중치": 이름, "행": m}):
                if e == "_SwiGLU정수":
                    p다자, p다정 = dr.할당(자리수 * m * (N // 2)), dr.할당((N // 64) * 64 * 4)
                    띄우기("정수줄자리" + e, ((N + W - 1) // W, 1), 256, [(1, p자m), (1, p정m), (1, p값), (1, p척), (1, p치), (1, p다정), (1, p다자), (0, m), (0, K), (0, N)])
                    dr.맞추기()
                    다 = {"자리": 다른칸(dr.내리기(p다자, np.empty((자리수, m, N // 2), np.int8)), 참자리[:, :m]),
                         "정보": 다른칸(dr.내리기(p다정, np.empty((N // 64, 64), np.int32))[:, :m].T, 참정보[:m])}
                else:
                    p출 = dr.할당(m * N * 4)
                    if e == "_잔차":
                        띄우기("정수줄자리" + e, ((N + W - 1) // W, 1), 256, [(1, p자m), (1, p정m), (1, p값), (1, p척), (1, p치), (1, p잔m), (1, p출), (0, m), (0, K), (0, N)])
                    else:
                        띄우기("정수줄자리" + e, ((N + W - 1) // W, 1), 256, [(1, p자m), (1, p정m), (1, p값), (1, p척), (1, p치), (1, p출), (0, m), (0, K), (0, N)])
                    dr.맞추기()
                    다 = {"출력": 다른칸(dr.내리기(p출, np.empty((m, N), f32)), 참[:m])}
                한 = {"가중치": 이름, "판": "정수줄자리" + e, "행": m, "K": K, "N": N, "다른 칸": 다}
                결과["경우"].append(한)
                if sum(다.values()):
                    print("다름!", 한, flush=True)
            if m == 1 and not 넘는가("정수줄하나" + e, {"가중치": 이름, "행": 1}):    # 23단계 — 한 행 줄 판(생성 그래프): 스레드 128, 열 W 개
                if e == "_SwiGLU정수":
                    p다자, p다정 = dr.할당(자리수 * N // 2), dr.할당((N // 64) * 64 * 4)
                    띄우기("정수줄하나" + e, ((N + W - 1) // W, 1), 128, [(1, p자m), (1, p정m), (1, p값), (1, p척), (1, p치), (1, p다정), (1, p다자), (0, 1), (0, K), (0, N)])
                    dr.맞추기()
                    다 = {"자리": 다른칸(dr.내리기(p다자, np.empty((자리수, 1, N // 2), np.int8)), 참자리[:, :1]),
                         "정보": 다른칸(dr.내리기(p다정, np.empty((N // 64, 64), np.int32))[:, :1].T, 참정보[:1])}
                else:
                    p출 = dr.할당(N * 4)
                    if e == "_잔차":
                        띄우기("정수줄하나" + e, ((N + W - 1) // W, 1), 128, [(1, p자m), (1, p정m), (1, p값), (1, p척), (1, p치), (1, p잔m), (1, p출), (0, 1), (0, K), (0, N)])
                    else:
                        띄우기("정수줄하나" + e, ((N + W - 1) // W, 1), 128, [(1, p자m), (1, p정m), (1, p값), (1, p척), (1, p치), (1, p출), (0, 1), (0, K), (0, N)])
                    dr.맞추기()
                    다 = {"출력": 다른칸(dr.내리기(p출, np.empty((1, N), f32)), 참[:1])}
                한 = {"가중치": 이름, "판": "정수줄하나" + e, "행": 1, "K": K, "N": N, "다른 칸": 다}
                결과["경우"].append(한)
                if sum(다.values()):
                    print("다름!", 한, flush=True)
        print(이름, "끝 — 경우", sum(1 for c in 결과["경우"] if c["가중치"] == 이름), flush=True)
        놓기()
    if 줄타일만:
        결과["모두 0"] = all(v == 0 for c in 결과["경우"] for v in c.get("다른 칸", {}).values())
        결과["장치 한도"] = sum(1 for c in 결과["경우"] if "장치 한도" in c)
        print("경우", len(결과["경우"]), "· 모두 0:", 결과["모두 0"], flush=True)
        json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return
    # 23단계 — 생성의 RMS(RMS정규화정수줄펼침 — 읽기를 앞으로)와 RMS정규화정수줄 · CPU 참조(rms정규화 → 자리 둘): 행 9 개(크기가 다르게)
    D7 = 가중["o"][0].shape[0]
    감마 = np.ascontiguousarray(g.풀기("blk.0.attn_norm.weight"), f32)
    Xr = (r.standard_normal((9, D7)) * 10.0 ** r.uniform(-3, 3, (9, 1))).astype(f32)
    eps = float(g.meta["qwen2.attention.layer_norm_rms_epsilon"])
    참자리r, 참정보r = 자리바이트(C.rms정규화(Xr, 감마, eps), 자리수)
    pX, pG = up(Xr), up(감마)
    for kn in ("RMS정규화정수줄", "RMS정규화정수줄펼침"):
        if 넘는가(kn, {"가중치": "attn_norm"}):
            continue
        p자r, p정r = dr.할당(자리수 * 9 * D7), dr.할당((D7 // 32) * 64 * 4)
        띄우기(kn, (9, 1), 256, [(1, pX), (1, pG), (1, p자r), (1, p정r), (0, 9)])
        dr.맞추기()
        다 = {"자리": 다른칸(dr.내리기(p자r, np.empty((자리수, 9, D7), np.int8)), 참자리r),
             "정보": 다른칸(dr.내리기(p정r, np.empty((D7 // 32, 64), np.int32))[:, :9].T, 참정보r)}
        결과["경우"].append({"가중치": "attn_norm", "판": kn, "행": 9, "다른 칸": 다})
        if sum(다.values()):
            print("다름!", kn, 다, flush=True)
    놓기()
    # 임베딩 — 낱말표(token_embd)의 q · d
    q, d, _ = 양자읽기(g, "token_embd.weight")
    V, D = q.shape
    값, 척 = 깔기(q, d, 형식)
    p값, p척 = up(값), up(척)
    토큰 = r.integers(0, V, 37).astype(np.int64)
    p토, p출 = up(토큰), dr.할당(37 * D * 4)
    띄우기("임베딩", ((37 * D + 255) // 256, 1), 256, [(1, p토), (1, p값), (1, p척), (1, p출), (0, 37)])
    dr.맞추기()
    참 = (q[토큰].astype(f32) * np.repeat(d[토큰], 32, axis=1)).astype(f32)
    다 = {"출력": 다른칸(dr.내리기(p출, np.empty((37, D), f32)), 참)}
    결과["경우"].append({"가중치": "낱말표", "판": "임베딩", "행": 37, "다른 칸": 다})
    놓기()
    결과["모두 0"] = all(v == 0 for c in 결과["경우"] for v in c.get("다른 칸", {}).values())
    결과["장치 한도"] = sum(1 for c in 결과["경우"] if "장치 한도" in c)
    print("경우", len(결과["경우"]), "· 모두 0:", 결과["모두 0"], "· 장치 한도:", 결과["장치 한도"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
