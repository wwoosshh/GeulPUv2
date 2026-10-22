#!/usr/bin/env python3
"""26a (1) — k-양자화(Q4_K · Q5_K · Q6_K)의 판을 CPU 참조(계약/정수계약.행렬곱K블라스)와 비트까지. 7B k-양자화 파일의 층 0 의 실제 가중치
(파일의 정수 · 척도 그대로 — 엔진과 같은 배치로 깔아)와 무작위 활성값으로: 정수줄하나 × 끝손질 셋(qkv 의 구간 · o · down 잔차 · gate·up
SwiGLU정수 · 로짓), (뒤에) 타일 판 · 임베딩.

  python -I 시험/판26.py <ptx> <gguf …> <출력.json> [--극단] [--빠른] [--행합]
--행합(1.0.0 1-1 — 행 합 판의 모듈: 정밀 CUDA k-양자화 모듈 — 커널생성 --먼): SwiGLU정수 끝손질이 정보 뒤에 쓴 블록마다의 행 합 T 도
본다. 입력의 정보는 늘 두 배로 올리고 뒤에 T 를 둔다(행 합 판의 타일 판이 읽는다 — 다른 모듈은 읽지 않는다).
--극단(31단계 — 규칙 R 의 경계): 슈퍼블록의 절반에서 반실수 척도(d · dmin, Q6_K 는 d)를 2⁻¹⁴ 배로, 활성값은 블록마다 크기를
2⁻⁹⁰ … 2⁻¹²⁰ 과 0 으로 — P · a 가 2⁻¹²⁶ 아래로 내려가는 칸이 생긴다(예전의 타일 끝셈 S · (P · a) 가 줄 판과 갈리던 경우).
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
from 판23 import 극단활성값    # noqa: E402

f32 = np.float32
u64, i64 = G.c_u64, G.c_i64
형식이름 = {12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 23: "IQ4_XS"}
블록바이트 = {12: 144, 13: 176, 14: 210, 23: 136}                 # 1.0.0 1-6c — IQ4_XS
# 타일 판(생성타일.gl 의 판 표에서 — 이름 · BM = 16 · MT · WM · BN = 8 · NT · WN · 스레드 32 · WM · WN)
_판표 = [("정수타일선형", 2, 2, 2, 8), ("정수좁은타일선형", 2, 2, 2, 4), ("정수작은타일선형", 2, 2, 1, 4), ("정수아주작은타일선형", 2, 2, 1, 2),
        ("정수넓은타일선형", 2, 4, 2, 8), ("정수긴타일선형", 1, 4, 2, 8), ("정수긴4타일선형", 1, 4, 2, 8), ("정수긴2타일선형", 1, 4, 2, 8),
        ("정수깊은타일선형", 1, 4, 4, 4), ("정수깊은넓은타일선형", 2, 2, 4, 4), ("정수세판타일선형", 2, 2, 2, 8)]
타일판들 = [(이름, 16 * MT * WM, 8 * NT * WN, 32 * WM * WN, NT) for 이름, WM, WN, MT, NT in _판표]
# 32단계 — k-양자화의 줄 타일 판 표 (생성줄타일.gl 의 줄타일K표준비 — 이름 · W · NW · 판 수 S)
줄타일K들 = [("정수줄타일", 64, 4, 3), ("정수줄타일좁은", 32, 2, 4), ("정수줄타일두워프", 64, 2, 3)]


def 엔진배치K(t, raw, N, K):
    """GGUF k-양자화 텐서(행 N · 깊이 K) → 엔진 배치의 (값 바이트 — 값 [KB][N][16] 뒤에 높은 [KB][N][4 · 8], 척 바이트 — [KS][N][16] 뒤에
    (Q6_K) d [KS][Np] 반실수) — 커널/생성K.gl 의 머리말 그대로."""
    형식, q, a, b = C.풀기K(t, raw, N, K)
    KB, KS = K // 32, K // 256
    Np = (N + 63) // 64 * 64
    if 형식 == "IQ4_XS":                                                    # 1.0.0 1-6c — 엔진/가중치K.gl 의 IQ4낱말 · 척 그대로
        bl = np.asarray(raw).reshape(N, KS, 136)
        qs = bl[:, :, 8:136].reshape(N, KS, 8, 16).astype(np.uint32)
        번호 = np.concatenate([qs & 15, qs >> 4], 3).reshape(N, KB, 32)       # 칸 i < 16 은 아래 니블, 16 + i 는 위 니블
        낱 = np.zeros((KB, N, 4), np.uint32)
        for tw, (b0, b1) in enumerate(((0, 8), (16, 24), (4, 12), (20, 28))):
            for i in range(4):
                낱[:, :, tw] |= 번호[:, :, b0 + i].T << (4 * i)
                낱[:, :, tw] |= 번호[:, :, b1 + i].T << (16 + 4 * i)
        척 = np.zeros((KS, N, 16), np.uint8)
        척[:, :, 0:2] = bl[:, :, 0:2].transpose(1, 0, 2)
        sh = (bl[:, :, 2].astype(np.int64) | (bl[:, :, 3].astype(np.int64) << 8))
        for j in range(8):
            ls = ((bl[:, :, 4 + j // 2].astype(np.int64) >> (4 * (j % 2))) & 15) | (((sh >> (2 * j)) & 3) << 4)
            척[:, :, 2 + j] = ((ls - 32) & 255).astype(np.uint8).T
        return np.ascontiguousarray(낱).tobytes(), np.ascontiguousarray(척).tobytes(), (형식, q, a, b)
    qq = q.astype(np.int16)
    if 형식 == "Q6_K":
        qq = qq + 32                                                        # 6비트 그대로(0 … 63)
    u = qq.astype(np.uint8).reshape(N, KB, 4, 2, 4).transpose(1, 0, 2, 3, 4)   # [KB][N][j][h][m] — 칸 8j + 4h + m
    아래 = (u & 15)
    값 = (아래[:, :, :, 0, :] | (아래[:, :, :, 1, :] << 4)).astype(np.uint8)    # [KB][N][4][4]
    부분 = [np.ascontiguousarray(값).tobytes()]
    if 형식 == "Q5_K":
        h = ((u >> 4) & 1).astype(np.uint32)                                # [KB][N][j][h][m]
        바 = (h << (4 * np.arange(2)[None, None, None, :, None] + np.arange(4)[None, None, None, None, :])).sum((3, 4)).astype(np.uint8)  # [KB][N][j]
        부분.append(np.ascontiguousarray(바).tobytes())
    elif 형식 == "Q6_K":                                                    # 27단계 — 낱말 h′ 의 바이트 i, 비트 6 − 2G′ = 칸 16h′ + 4G′ + i 의 (높은 ^ 2)
        h = (((u >> 4) & 3) ^ 2).astype(np.uint32).reshape(KB, N, 2, 2, 2, 4)  # [KB][N][h′][j % 2][4 칸 묶음 반][i] — 칸 16h′ + 8(j % 2) + 4hh + i
        jl, hh, i = np.meshgrid(np.arange(2), np.arange(2), np.arange(4), indexing="ij")
        밀기 = (8 * i + 6 - 2 * (2 * jl + hh)).astype(np.uint32)            # G′ = 2(j % 2) + hh
        낱 = (h << 밀기[None, None, None]).sum((3, 4, 5)).astype(np.uint32)   # [KB][N][2]
        부분.append(np.ascontiguousarray(낱).tobytes())
    bb = 블록바이트[t]
    bl = np.asarray(raw).reshape(N, KS, bb)
    if 형식 == "Q6_K":
        척 = np.ascontiguousarray(bl[:, :, 192:208].transpose(1, 0, 2)).tobytes()
        d = np.zeros((KS, Np), np.uint16)
        d[:, :N] = np.ascontiguousarray(bl[:, :, 208:210]).view(np.uint16)[:, :, 0].T
        척 += d.tobytes()
    else:
        척 = np.ascontiguousarray(bl[:, :, 0:16].transpose(1, 0, 2)).tobytes()
    return b"".join(부분), 척, (형식, q, a, b)


def 극단날(t, raw, N, K, r):
    """k-양자화 날 바이트의 반실수 척도(d · dmin — Q6_K 는 d)를 슈퍼블록의 절반(무작위)에서 2⁻¹⁴ 배로(반실수로 반올림)."""
    bb = 블록바이트[t]
    KS = K // 256
    bl = np.array(raw, dtype=np.uint8, copy=True).reshape(N, KS, bb)
    작게 = r.random((N, KS)) < 0.5
    for a in ((0, 2) if t in (12, 13) else (0,) if t == 23 else (208,)):
        h = np.ascontiguousarray(bl[:, :, a:a + 2]).view(np.float16)[..., 0]
        h2 = np.where(작게, (h.astype(f32) * f32(2.0 ** -14)).astype(np.float16), h).astype(np.float16)
        bl[:, :, a:a + 2] = np.ascontiguousarray(h2).view(np.uint8).reshape(N, KS, 2)
    return bl.reshape(-1)


def 자리행합(x, 자리수=2):
    """1.0.0 1-1 — 행 합 판: 블록(32 칸)마다 정수 n = Σ_j 256^j · D_j 의 합 T [행][블록] (int32 — 정확)."""
    Dg, _ = C.자리로(np.asarray(x, f32), 자리수)
    n, w = Dg.shape[1], Dg.shape[2]
    v = sum((256 ** j) * Dg[j].astype(np.int64) for j in range(자리수))
    return v.reshape(n, w // 32, 32).sum(axis=2).astype(np.int32)


def 정보판(M0, T, 폭):
    """정보 [블록][폭] 과 그 뒤의 행 합 T [블록][폭] — [2 · 블록][폭]"""
    n, nb = M0.shape
    정보 = np.zeros((2 * nb, 폭), np.int32)
    정보[:nb, :n] = M0.T
    정보[nb:, :n] = T.T
    return 정보


def 다른칸(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.dtype == f32:
        a, b = a.view(np.int32), np.asarray(b, f32).view(np.int32)
    return int(np.count_nonzero(a != b))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    극단 = "--극단" in sys.argv
    줄타일만 = "--줄타일만" in sys.argv          # 32단계 — 줄 타일 판만(행 1 … 8)
    자리수 = 1 if ("--빠른" in sys.argv or "--자리하나" in sys.argv) else 2     # GeulPUv2 빠른 계약 F1(커널생성 --빠른 의 모듈)
    행합 = "--행합" in sys.argv          # 1.0.0 1-1 — 행 합 판의 모듈: 끝손질이 쓴 T 도 본다
    인자 = [a for a in sys.argv[1:] if a not in ("--극단", "--줄타일만", "--빠른", "--자리하나", "--행합")]
    ptx, 출력 = 인자[0], 인자[-1]
    파일들 = 인자[1:-1]
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

    r = np.random.default_rng(26)
    결과 = {"극단": 극단, "경우": []}
    for 파일 in 파일들:
        g = GR.GGUF(파일)
        이름파일 = os.path.basename(파일)
        print("파일", 이름파일, flush=True)
        # 행렬: (이름, 텐서들(열로 잇는다), 끝손질, 치우침)
        행렬들 = [("q·k", ["blk.0.attn_q.weight", "blk.0.attn_k.weight"], "", ["blk.0.attn_q.bias", "blk.0.attn_k.bias"]),
                 ("v", ["blk.0.attn_v.weight"], "", ["blk.0.attn_v.bias"]),
                 ("o", ["blk.0.attn_output.weight"], "_잔차", None),
                 ("down", ["blk.0.ffn_down.weight"], "_잔차", None),
                 ("down3", ["blk.3.ffn_down.weight"], "_잔차", None),
                 ("gu", ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"], "_SwiGLU정수", None),
                 ("로짓", ["output.weight" if "output.weight" in g.tensors else "token_embd.weight"], "", None)]     # 37단계 — 묶인 출력(Qwen3)
        for 이름, 텐서들, e, 치들 in 행렬들:
            원들 = [g.원(n) for n in 텐서들]
            ts = {t for t, _, _ in 원들}
            assert len(ts) == 1, (이름, ts)
            t = ts.pop()
            K = 원들[0][1][0]
            Ns = [dims[1] for _, dims, _ in 원들]
            N = sum(Ns)
            raw = np.concatenate([np.asarray(x) for _, _, x in 원들])
            if 이름 == "gu":                                     # gate · up 을 출력 32 열마다 번갈아(엔진과 같다)
                F = Ns[0]
                bb = 블록바이트[t]
                rows = raw.reshape(N, K // 256 * bb)
                순 = np.empty(2 * F, np.int64)
                for j in range(F // 32):
                    순[64 * j:64 * j + 32] = np.arange(32 * j, 32 * j + 32)
                    순[64 * j + 32:64 * j + 64] = F + np.arange(32 * j, 32 * j + 32)
                raw = np.ascontiguousarray(rows[순]).reshape(-1)
            if 극단:
                raw = 극단날(t, raw, N, K, r)
            값, 척, w = 엔진배치K(t, raw, N, K)
            형식 = w[0]
            p값, p척 = up(값), up(척)
            치 = (np.concatenate([g.풀기(n) if n in g.tensors else np.zeros(Ns[k], f32) for k, n in enumerate(치들)]).astype(f32)
                 if 치들 else np.zeros(N, f32))                       # 치우침이 없는 모델(Qwen3)은 0
            p치 = up(치)
            X = 극단활성값(r, 1, K) if 극단 else (r.standard_normal((1, K)) * 10.0 ** r.uniform(-1, 1, (1, 1))).astype(f32)
            바이트, M0 = 자리바이트(X, 자리수)
            정보 = 정보판(M0, 자리행합(X, 자리수), 64)
            p자, p정 = up(바이트), up(정보)
            y = C.행렬곱K블라스(X, w, 치, 자리수=자리수)
            W = 64 if e == "_SwiGLU정수" else 32
            kn = f"정수줄하나{e}_{형식}"
            if 줄타일만:
                # 32단계 — 줄 타일 판(행 1 … 8): 블록 하나가 열 W 개, 스레드 32 · NW. 출력폭 = N
                M8 = 8
                X8 = 극단활성값(r, M8, K) if 극단 else (r.standard_normal((M8, K)) * 10.0 ** r.uniform(-1, 1, (M8, 1))).astype(f32)
                y8 = C.행렬곱K블라스(X8, w, 치, 자리수=자리수)
                Np = (N + 63) // 64 * 64
                잔8 = (r.standard_normal((M8, N)) * 3.0).astype(f32)
                for 판, W8, NW, _S in 줄타일K들:
                    if e == "_SwiGLU정수" and W8 != 64:
                        continue
                    kn8 = f"{판}{e}_{형식}"
                    if 넘는가(kn8, {"파일": 이름파일, "가중치": 이름, "형식": 형식}):
                        continue
                    for m in range(1, 9):
                        바이트m, M0m = 자리바이트(X8[:m], 자리수)
                        정보m = 정보판(M0m, 자리행합(X8[:m], 자리수), 64)
                        p자m, p정m = up(바이트m), up(정보m)
                        앞 = [(1, p자m), (1, p정m), (1, p값), (1, p척), (1, p치)]
                        뒤 = [(0, m), (0, K), (0, N), (0, Np), (0, 0), (0, 0), (0, N)]
                        grid = ((N + W8 - 1) // W8, 1)
                        if e == "_SwiGLU정수":
                            gcol = np.concatenate([np.arange(64 * j, 64 * j + 32) for j in range(N // 64)])
                            h8 = C.swiglu(y8[:m][:, gcol], y8[:m][:, gcol + 32])
                            참자리8, 참정보8 = 자리바이트(h8, 자리수)
                            p다자, p다정 = dr.할당(자리수 * m * (N // 2)), dr.할당(2 * (N // 64) * 64 * 4)
                            띄우기(kn8, grid, 32 * NW, 앞 + [(1, p다정), (1, p다자)] + 뒤)
                            dr.맞추기()
                            글정8 = dr.내리기(p다정, np.empty((2 * (N // 64), 64), np.int32))
                            다8 = {"자리": 다른칸(dr.내리기(p다자, np.empty((자리수, m, N // 2), np.int8)), 참자리8),
                                  "정보": 다른칸(글정8[:N // 64, :m].T, 참정보8)}
                            if 행합:
                                다8["행합"] = 다른칸(글정8[N // 64:, :m].T, 자리행합(h8, 자리수))
                        else:
                            p출 = dr.할당(m * N * 4)
                            if e == "_잔차":
                                dr.확인(dr.cu.cuMemcpyHtoD_v2(p출, np.ascontiguousarray(잔8[:m]).ctypes.data, m * N * 4))
                                띄우기(kn8, grid, 32 * NW, 앞 + [(1, p출), (1, p출)] + 뒤)
                                참8 = C.더(잔8[:m], y8[:m])
                            else:
                                띄우기(kn8, grid, 32 * NW, 앞 + [(1, p출)] + 뒤)
                                참8 = y8[:m]
                            dr.맞추기()
                            다8 = {"출력": 다른칸(dr.내리기(p출, np.empty((m, N), f32)), 참8)}
                        한 = {"파일": 이름파일, "가중치": 이름, "형식": 형식, "판": kn8, "행": m, "K": int(K), "N": int(N), "다른 칸": 다8}
                        결과["경우"].append(한)
                        if sum(다8.values()):
                            print("다름!", 한, flush=True)
                    print(f"{이름} ({형식}) {kn8} 행 1 … 8 끝", flush=True)
                놓기()
                continue
            if not 넘는가(kn, {"파일": 이름파일, "가중치": 이름, "형식": 형식}):
                if e == "_SwiGLU정수":
                    gcol = np.concatenate([np.arange(64 * j, 64 * j + 32) for j in range(N // 64)])
                    h1 = C.swiglu(y[:, gcol], y[:, gcol + 32])
                    참자리, 참정보 = 자리바이트(h1, 자리수)
                    p다자, p다정 = dr.할당(자리수 * N // 2), dr.할당(2 * (N // 64) * 64 * 4)
                    띄우기(kn, ((N + W - 1) // W, 1), 128, [(1, p자), (1, p정), (1, p값), (1, p척), (1, p치), (1, p다정), (1, p다자), (0, 1), (0, K), (0, N)])
                    dr.맞추기()
                    글정 = dr.내리기(p다정, np.empty((2 * (N // 64), 64), np.int32))
                    다 = {"자리": 다른칸(dr.내리기(p다자, np.empty((자리수, 1, N // 2), np.int8)), 참자리[:, :1]),
                         "정보": 다른칸(글정[:N // 64, :1].T, 참정보[:1])}
                    if 행합:
                        다["행합"] = 다른칸(글정[N // 64:, :1].T, 자리행합(h1, 자리수)[:1])
                else:
                    p출 = dr.할당(N * 4)
                    참 = y
                    if e == "_잔차":
                        잔 = (r.standard_normal((1, N)) * 3.0).astype(f32)
                        p잔 = up(잔)
                        참 = C.더(잔, y)
                        띄우기(kn, ((N + W - 1) // W, 1), 128, [(1, p자), (1, p정), (1, p값), (1, p척), (1, p치), (1, p잔), (1, p출), (0, 1), (0, K), (0, N)])
                    else:
                        띄우기(kn, ((N + W - 1) // W, 1), 128, [(1, p자), (1, p정), (1, p값), (1, p척), (1, p치), (1, p출), (0, 1), (0, K), (0, N)])
                    dr.맞추기()
                    다 = {"출력": 다른칸(dr.내리기(p출, np.empty((1, N), f32)), 참)}
                한 = {"파일": 이름파일, "가중치": 이름, "형식": 형식, "판": kn, "K": int(K), "N": int(N), "다른 칸": 다}
                결과["경우"].append(한)
                print(("다름! " if sum(다.values()) else "") + f"{이름} ({형식}) {kn} K {K} · N {N}: {다}", flush=True)
            # 타일 판 — 행 M (64 의 배수가 아니게)
            M = 100
            Xm = 극단활성값(r, M, K) if 극단 else (r.standard_normal((M, K)) * 10.0 ** r.uniform(-1, 1, (M, 1))).astype(f32)
            바이트m, M0m = 자리바이트(Xm, 자리수)
            폭 = (M + 63) // 64 * 64
            정보m = 정보판(M0m, 자리행합(Xm, 자리수), 폭)
            p자m, p정m = up(바이트m), up(정보m)
            ym = C.행렬곱K블라스(Xm, w, 치, 자리수=자리수)
            Np = (N + 63) // 64 * 64
            for 판, BM, BN, T, NT in 타일판들:
                if e == "_SwiGLU정수" and NT != 8:
                    continue
                kn2 = f"{판}{e}_{형식}"
                if 넘는가(kn2, {"파일": 이름파일, "가중치": 이름, "형식": 형식}):
                    continue
                grid = ((M + BM - 1) // BM, (N + BN - 1) // BN)
                앞 = [(1, p자m), (1, p정m), (1, p값), (1, p척), (1, p치)]
                뒤 = [(0, M), (0, K), (0, N), (0, Np), (0, 0), (0, 0), (0, N)]          # 출력폭 = N
                if e == "_SwiGLU정수":
                    gcol = np.concatenate([np.arange(64 * j, 64 * j + 32) for j in range(N // 64)])
                    hm = C.swiglu(ym[:, gcol], ym[:, gcol + 32])
                    참자리m, 참정보m = 자리바이트(hm, 자리수)
                    p다자, p다정 = dr.할당(자리수 * M * (N // 2)), dr.할당(2 * (N // 64) * 폭 * 4)
                    띄우기(kn2, grid, T, 앞 + [(1, p다정), (1, p다자)] + 뒤)
                    dr.맞추기()
                    글정m = dr.내리기(p다정, np.empty((2 * (N // 64), 폭), np.int32))
                    다2 = {"자리": 다른칸(dr.내리기(p다자, np.empty((자리수, M, N // 2), np.int8)), 참자리m),
                          "정보": 다른칸(글정m[:N // 64, :M].T, 참정보m)}
                    if 행합:
                        다2["행합"] = 다른칸(글정m[N // 64:, :M].T, 자리행합(hm, 자리수))
                else:
                    p출 = dr.할당(M * N * 4)
                    참m = ym
                    if e == "_잔차":
                        잔m = (r.standard_normal((M, N)) * 3.0).astype(f32)
                        참m = C.더(잔m, ym)
                        dr.확인(dr.cu.cuMemcpyHtoD_v2(p출, 잔m.ctypes.data, 잔m.nbytes))
                        띄우기(kn2, grid, T, 앞 + [(1, p출), (1, p출)] + 뒤)
                    else:
                        띄우기(kn2, grid, T, 앞 + [(1, p출)] + 뒤)
                    dr.맞추기()
                    다2 = {"출력": 다른칸(dr.내리기(p출, np.empty((M, N), f32)), 참m)}
                한 = {"파일": 이름파일, "가중치": 이름, "형식": 형식, "판": kn2, "행": M, "K": int(K), "N": int(N), "다른 칸": 다2}
                결과["경우"].append(한)
                print(("다름! " if sum(다2.values()) else "") + f"{이름} ({형식}) {kn2} 행 {M}: {다2}", flush=True)
            놓기()
        if 줄타일만:
            continue
        # 임베딩 — 낱말표(token_embd)
        t, dims, raw = g.원("token_embd.weight")
        D, V = dims
        값, 척, w = 엔진배치K(t, raw, V, D)
        p값, p척 = up(값), up(척)
        토큰 = r.integers(0, V, 37).astype(np.int64)
        for 칸 in (False, True):
            kn3 = ("임베딩칸_" if 칸 else "임베딩_") + w[0]
            if 넘는가(kn3, {"파일": 이름파일, "가중치": "낱말표", "형식": w[0]}):
                continue
            if 칸:
                걸음 = np.array([5, 3], np.int64)
                p토 = up(np.concatenate([np.zeros(3, np.int64), 토큰]))
                p걸 = up(걸음)
                p출 = dr.할당(D * 4)
                띄우기(kn3, ((D + 255) // 256, 1), 256, [(1, p토), (1, p값), (1, p척), (1, p출), (0, 1), (1, p걸)])
                dr.맞추기()
                참e = C.임베딩K(w, 토큰[:1])
                다3 = {"출력": 다른칸(dr.내리기(p출, np.empty((1, D), f32)), 참e)}
            else:
                p토, p출 = up(토큰), dr.할당(37 * D * 4)
                띄우기(kn3, ((37 * D + 255) // 256, 1), 256, [(1, p토), (1, p값), (1, p척), (1, p출), (0, 37)])
                dr.맞추기()
                다3 = {"출력": 다른칸(dr.내리기(p출, np.empty((37, D), f32)), C.임베딩K(w, 토큰))}
            결과["경우"].append({"파일": 이름파일, "가중치": "낱말표", "형식": w[0], "판": kn3, "다른 칸": 다3})
            print(("다름! " if sum(다3.values()) else "") + f"낱말표 ({w[0]}) {kn3}: {다3}", flush=True)
        놓기()
    결과["모두 0"] = all(v == 0 for c in 결과["경우"] for v in c.get("다른 칸", {}).values())
    결과["장치 한도"] = sum(1 for c in 결과["경우"] if "장치 한도" in c)
    print("경우", len(결과["경우"]), "· 모두 0:", 결과["모두 0"], "· 장치 한도:", 결과["장치 한도"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
