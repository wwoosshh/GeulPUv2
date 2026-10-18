#!/usr/bin/env python3
"""22a (1) — 머리 128 의 새 커널(모듈 qwen2Q8_7b 의 RoPEKV정수로 · 정수어텐션묶음 · 정수묶음접기 · 정수묶음접기짧은)을 CPU 참조
(계약/정수계약.py — 머리 128 로 넓힌 계약)와 비트까지 견준다. 다른 칸이 0 이어야 한다.

  입력 (가) 무작위 q · k · v — 행 2100 (묶음 셋: 위치 1024 · 2048 을 넘는다), 머리마다 크기가 다르고 튀는 값을 섞는다.
       (나) 7B 층 0 의 실제 q · k · v — 채팅 질문 셋의 프롬프트(CPU 참조의 임베딩 · RMS · 행렬곱으로).
  덩이 — 한 번에(덩이 하나) · 나눠서(묶음 경계를 넘는 덩이) · 한 행씩(묶음 판을 행 1 로) · 생성 판 행 1 · 행 4(생성 길의 판 — 정수조각어텐션
  (KV 머리 × 키 조각, 질의 머리 일곱을 가상 행으로) → 정수조각접기 · 정수조각접기짧은(두 층 접기)). 덩이마다 RoPEKV정수로 → 어텐션 →
  접기(o 의 자리 둘 · 정보, 짧은실수 O/L)를 띄우고, 끝에 KV 캐시(바이트 · 지수)를 내린다.

  python -I 시험/머리128.py <qwen2Q8_7b.ptx> <7B gguf> <출력.json> [--빠르게]
"""
import ctypes
import json
import os
import sys
import time

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

f32 = np.float32
u64, i64 = G.c_u64, G.c_i64
D, NH, NKV, HD = 3584, 28, 4, 128         # 7B — 24단계부터는 main 이 GGUF 의 설정으로 다시 둔다(모양두기)
QKV = D + 2 * NKV * HD                     # 4608
O자리수 = 2                                # 33단계 — 접기의 출력(o 의 입력) 자리 수: --자리하나 면 1 (어텐션 안의 q · k · v · e 는 늘 둘)
KB = D // 32                               # 112
SH = HD + 4                            # 묶음 상태 하나의 칸 (M, L, 빈 둘, O 128)


def 모양두기(경로):
    """24단계 — GGUF 의 메타데이터(바이트 몇 십 MB 만 읽는다)에서 너비 · 머리 · KV 머리. 머리 크기는 128."""
    import struct
    with open(경로, "rb") as f:
        b = f.read(96 * 1024 * 1024)
    _, nkv = struct.unpack_from("<QQ", b, 8)
    o = 24
    메타 = {}
    for _ in range(nkv):
        n = struct.unpack_from("<Q", b, o)[0]
        키 = b[o + 8:o + 8 + n].decode("utf-8")
        t = struct.unpack_from("<I", b, o + 8 + n)[0]
        v, o = GR._값(b, o + 12 + n, t)
        메타[키] = v
    global D, NH, NKV, QKV, KB
    D = 메타["qwen2.embedding_length"]
    NH = 메타["qwen2.attention.head_count"]
    NKV = 메타["qwen2.attention.head_count_kv"]
    assert D == NH * HD, (D, NH)
    QKV = D + 2 * NKV * HD
    KB = D // 32
    return 메타
질문들 = ["What is the capital of France? Answer in one word.", "대한민국의 수도는 어디야? 한 단어로 답해.",
         "Write a short poem about the sea."]


class 글커널:
    def __init__(self, ptx):
        self.dr = 시험장치.드라이버()
        mod = self.dr.모듈(open(ptx, "rb").read())
        이름들 = ["RoPEKV정수로", "정수어텐션묶음", "정수묶음접기", "정수묶음접기짧은", "정수조각어텐션", "정수조각접기", "정수조각접기짧은"]
        self.넷 = b"_G" + "정수조각어텐션넷".encode().hex().encode() in open(ptx, "rb").read()
        if self.넷:                                   # 23단계 — 생성 판 넷(조각 셈을 워프 넷으로) · 접기 넓(묶음끼리 나란히)
            이름들 += ["정수조각어텐션넷", "정수조각접기넓"]
        self.f = {n: self.dr.함수(mod, n) for n in 이름들}
        self.한도 = {n: 시험장치.한도넘음(self.dr, n) for n in 이름들}     # 31단계 — Vulkan 장치가 못 돌리는 커널의 까닭(아니면 None)
        self.dr.cu.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]

    def 속성(self):
        """커널마다 레지스터 · 넘친 지역 메모리 · 공유 메모리(바이트)."""
        out = {}
        for n, fn in self.f.items():
            v = []
            for a in (4, 3, 1):                       # NUM_REGS · LOCAL_SIZE_BYTES · SHARED_SIZE_BYTES
                x = ctypes.c_int()
                self.dr.확인(self.dr.cu.cuFuncGetAttribute(ctypes.byref(x), a, fn))
                v.append(x.value)
            out[n] = {"레지스터": v[0], "지역(넘침) 바이트": v[1], "공유 바이트": v[2]}
        return out

    def 띄우기(self, 이름, grid, block, params):
        ps = [u64(p) if 참조 else i64(p) for 참조, p in params]
        args = (ctypes.c_void_p * len(ps))(*[ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in ps])
        r = self.dr.cu.cuLaunchKernel(self.f[이름], grid[0], grid[1], 1, block, 1, 1, 0, None, args, None)
        assert r == 0, (이름, r)

    판커널 = {"묶음": ("RoPEKV정수로", "정수어텐션묶음", "정수묶음접기", "정수묶음접기짧은"),
              "조각": ("RoPEKV정수로", "정수조각어텐션", "정수조각접기", "정수조각접기짧은"),
              "넷": ("RoPEKV정수로", "정수조각어텐션넷", "정수조각접기넓", "정수조각접기짧은")}

    def 못돌림(self, 덩이들):
        """31단계 — 덩이들이 쓰는 커널 가운데 장치 한도를 넘는 것 {이름: 까닭} (CUDA 는 늘 빈 것)"""
        쓰는 = {n for d in 덩이들 for n in self.판커널[d[2] if len(d) > 2 else "묶음"]}
        return {n: self.한도[n] for n in sorted(쓰는) if self.한도.get(n)}

    def 돌리기(self, X, 덩이들, 최대길이, cos, sin, RoPE만=False):
        """X [n][QKV] 짧은실수를 덩이들 [(위치시작, 행[, 판])] 차례로 — 행마다의 출력과 끝의 캐시. 판: "묶음"(기본) · "조각"(생성 길, 행 넷까지) ·
        "넷"(23단계 생성 그래프의 판 — 정수조각어텐션넷 → 정수조각접기넓, 행 하나). RoPE만: RoPEKV정수로만 띄우고 q · KV 캐시만 낸다
        (31단계 — 어텐션 커널이 장치 한도를 넘는 Vulkan 장치에서도 RoPE 는 본다)."""
        dr = self.dr
        n = X.shape[0]
        덩이들 = [(d[0], d[1], d[2] if len(d) > 2 else "묶음") for d in 덩이들]
        Mmax = max(m for _, m, _ in 덩이들)
        조각칸폭 = 최대길이 // 64
        p조 = dr.할당(4 * NH * 조각칸폭 * SH * 4)
        폭max = (Mmax + 63) // 64 * 64
        묶음수 = (최대길이 + 1023) // 1024
        p입력 = dr.할당(Mmax * QKV * 4)
        p자q, p정q = dr.할당(2 * Mmax * D), dr.할당(KB * 폭max * 4)
        p키, p값 = dr.할당(NKV * 2 * 최대길이 * HD), dr.할당(NKV * 최대길이 * 2 * HD)
        p키지, p값지 = dr.할당(NKV * 최대길이 * 4), dr.할당(NKV * 최대길이 * 4)
        p묶 = dr.할당(Mmax * NH * 묶음수 * SH * 4)
        p자, p정 = dr.할당(2 * Mmax * D), dr.할당(KB * 폭max * 4)
        p출 = dr.할당(Mmax * D * 4)
        p코, p사 = dr.할당(최대길이 * 64 * 4), dr.할당(최대길이 * 64 * 4)
        dr.올리기(p코, np.ascontiguousarray(cos, f32))
        dr.올리기(p사, np.ascontiguousarray(sin, f32))
        out = {"q자리": np.zeros((2, n, D), np.int8), "q정보": np.zeros((n, KB), np.int32), "o자리": np.zeros((O자리수, n, D), np.int8),
               "o정보": np.zeros((n, KB), np.int32), "o짧은": np.zeros((n, D), f32)}
        KV = [(1, p키), (1, p값), (1, p키지), (1, p값지)]
        for s, m, 판 in 덩이들:
            폭 = (m + 63) // 64 * 64
            dr.올리기(p입력, np.ascontiguousarray(X[s:s + m], f32))
            self.띄우기("RoPEKV정수로", ((QKV // 32 + 31) // 32, m), 256,
                     [(1, p입력), (1, p자q), (1, p정q)] + KV + [(1, p코), (1, p사), (0, m), (0, m), (0, s), (0, 최대길이)])
            if RoPE만:
                pass
            elif 판 == "넷":
                assert m == 1
                self.띄우기("정수조각어텐션넷", (조각칸폭, NKV), 128,
                         [(1, p자q), (1, p정q)] + KV + [(1, p조), (0, m), (0, m), (0, s), (0, 최대길이)])
                self.띄우기("정수조각접기넓", (NH, m), 1024, [(1, p조), (1, p정), (1, p자), (0, m), (0, m), (0, s), (0, 최대길이)])
                self.띄우기("정수조각접기짧은", (NH, m), 128, [(1, p조), (1, p출), (0, m), (0, m), (0, s), (0, 최대길이)])
            elif 판 == "조각":
                assert m <= 4
                self.띄우기("정수조각어텐션", (조각칸폭, NKV), 128,
                         [(1, p자q), (1, p정q)] + KV + [(1, p조), (0, m), (0, m), (0, s), (0, 최대길이)])
                self.띄우기("정수조각접기", (NH, m), 128, [(1, p조), (1, p정), (1, p자), (0, m), (0, m), (0, s), (0, 최대길이)])
                self.띄우기("정수조각접기짧은", (NH, m), 128, [(1, p조), (1, p출), (0, m), (0, m), (0, s), (0, 최대길이)])
            else:
                묶 = (s + m - 1) // 1024 + 1
                self.띄우기("정수어텐션묶음", ((m + 63) // 64 * NH * 묶, 1), 128,
                         [(1, p자q), (1, p정q)] + KV + [(1, p묶), (0, m), (0, m), (0, s), (0, 최대길이)])
                self.띄우기("정수묶음접기", (NH, m), 128, [(1, p묶), (1, p정), (1, p자), (0, m), (0, m), (0, s), (0, 최대길이)])
                self.띄우기("정수묶음접기짧은", (NH, m), 128, [(1, p묶), (1, p출), (0, m), (0, m), (0, s), (0, 최대길이)])
            dr.맞추기()
            out["q자리"][:, s:s + m] = dr.내리기(p자q, np.empty((2, m, D), np.int8))
            out["q정보"][s:s + m] = dr.내리기(p정q, np.empty((KB, 폭), np.int32))[:, :m].T
            if RoPE만:
                continue
            out["o자리"][:, s:s + m] = dr.내리기(p자, np.empty((O자리수, m, D), np.int8))
            out["o정보"][s:s + m] = dr.내리기(p정, np.empty((KB, 폭), np.int32))[:, :m].T
            out["o짧은"][s:s + m] = dr.내리기(p출, np.empty((m, D), f32))
        out["키"] = dr.내리기(p키, np.empty((NKV, 2, 최대길이, HD), np.int8))
        out["값"] = dr.내리기(p값, np.empty((NKV, 최대길이 // 32, 2, HD, 32), np.int8))
        out["키지수"] = dr.내리기(p키지, np.empty((NKV, 최대길이, 4), np.uint8))
        out["값지수"] = dr.내리기(p값지, np.empty((NKV, 최대길이, 4), np.uint8))
        if RoPE만:
            for 키 in ("o자리", "o정보", "o짧은"):
                out.pop(키)
        dr.맞추기()
        for p in dr.할당들:                          # 메모리만 돌려준다(모듈은 둔다 — 다음 경우가 쓴다)
            dr.확인(dr.cu.cuMemFree_v2(u64(p)))
        dr.할당들 = []
        return out


def 자리바이트(x, 자리수=2):
    """x [n][너비] → (자리 바이트 [자리수][n][너비] — 블록마다 낱말 차례, 정보 M_0 [n][블록]) — 정수로 · RoPEKV · 묶음 접기의 출력 배치.
    33단계 — 자리수 1 은 같은 최적화의 대조(V1): M_0 = 1.5 · 2²³ · 2^(E − P), P = 8 · 자리수 − 3."""
    Dg, E = C.자리로(np.asarray(x, f32), 자리수)
    n, w = Dg.shape[1], Dg.shape[2]
    바이트 = C.낱말차례(Dg.reshape(자리수, n, w // 32, 32)).reshape(자리수, n, w).astype(np.int8)
    M0 = (((153 - 8 * 자리수 + E) << 23) + 4194304).astype(np.int64).astype(np.int32)
    return 바이트, M0


def 참조(X, 최대길이, cos, sin):
    n = X.shape[0]
    위치 = np.arange(n)
    q, k, v = X[:, :D], X[:, D:D + NKV * HD], X[:, D + NKV * HD:]
    qr = np.concatenate([C.rope(q[:, HD * a:HD * a + HD], cos, sin, 위치) for a in range(NH)], 1)
    kr = [C.rope(k[:, HD * g:HD * g + HD], cos, sin, 위치) for g in range(NKV)]
    vv = [np.ascontiguousarray(v[:, HD * g:HD * g + HD]) for g in range(NKV)]
    out = {}
    out["q자리"], out["q정보"] = 자리바이트(qr)
    키들 = [C.키캐시(kr[g], 최대길이) for g in range(NKV)]
    값들 = [C.값캐시(vv[g], 최대길이) for g in range(NKV)]
    out["키"] = np.stack([a for a, _ in 키들])
    out["키지수"] = np.stack([b for _, b in 키들])
    out["값"] = np.stack([a for a, _ in 값들])
    out["값지수"] = np.stack([b for _, b in 값들]).astype(np.uint8)
    G_ = NH // NKV
    y = np.concatenate([C.어텐션(qr[:, HD * a:HD * a + HD], kr[a // G_], vv[a // G_]) for a in range(NH)], 1)
    out["o짧은"] = y
    out["o자리"], out["o정보"] = 자리바이트(y, O자리수)
    return out


def 견주기(글, 참):
    결과 = {}
    for 이름 in ("q자리", "q정보", "키", "키지수", "값", "값지수", "o자리", "o정보", "o짧은"):
        if 이름 not in 글:                           # RoPE만 — 어텐션 · 접기의 출력이 없다
            continue
        a, b = np.asarray(글[이름]), np.asarray(참[이름])
        if a.dtype == f32:
            a, b = a.view(np.int32), b.astype(f32).view(np.int32)
        결과[이름] = int(np.count_nonzero(a.reshape(-1) != b.reshape(-1).astype(a.dtype)))
    return 결과


def 무작위(n, 씨):
    """머리마다 크기가 다르고(10^±1.5) 몇 칸은 튀는(×30) q · k · v."""
    r = np.random.default_rng(씨)
    X = r.standard_normal((n, QKV)).astype(f32)
    크기 = (10.0 ** r.uniform(-1.5, 1.5, QKV // HD)).astype(f32)
    X *= np.repeat(크기, HD)[None, :]
    튐 = r.random((n, QKV)) < 0.002
    X[튐] *= f32(30.0)
    return X.astype(f32)


def 층0qkv(gguf, ids):
    """7B 층 0 의 q · k · v (bias 포함, RoPE 앞) — CPU 참조의 임베딩 · RMS · 행렬곱빠른(계약 그대로)."""
    g = GR.GGUF(gguf)

    def q8(name):                                   # Q8_0 · Q4_0 (니블 − 8) — 24단계
        t, dims, raw = g.원(name)
        K, N = dims
        if t == GR.Q4_0:
            bl = np.asarray(raw).reshape(N, K // 32, 18)
            d = np.ascontiguousarray(bl[:, :, :2]).view(np.float16).reshape(N, K // 32).astype(f32)
            qs = bl[:, :, 2:]
            q = (np.concatenate([qs & 15, qs >> 4], 2).astype(np.int16) - 8).astype(np.int8).reshape(N, K)
            return q, d
        bl = np.asarray(raw).reshape(N, K // 32, 34)
        d = np.ascontiguousarray(bl[:, :, :2]).view(np.float16).reshape(N, K // 32).astype(f32)
        q = np.ascontiguousarray(bl[:, :, 2:]).view(np.int8).reshape(N, K)
        return q, d
    eq, ed = q8("token_embd.weight")
    x = (eq[ids].astype(f32) * np.repeat(ed[ids], 32, axis=1)).astype(f32)
    h = C.rms정규화(x, g.풀기("blk.0.attn_norm.weight"), float(g.meta["qwen2.attention.layer_norm_rms_epsilon"]))
    out = []
    for a in ("q", "k", "v"):
        q, d = q8(f"blk.0.attn_{a}.weight")
        out.append(C.행렬곱빠른(h, q.T, np.ascontiguousarray(d.T), g.풀기(f"blk.0.attn_{a}.bias")))
    return np.concatenate(out, 1).astype(f32)


def 채팅토큰(질문):
    import 토크나이저 as TK
    tk = TK.Qwen토크나이저(os.path.join(ROOT, "모델", "qwen2.5-0.5b-instruct"))
    return tk.나누기(TK.채팅틀([{"role": "user", "content": 질문}]))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ptx, gguf, 출력 = sys.argv[1:4]
    빠르게 = "--빠르게" in sys.argv
    global O자리수
    if "--자리하나" in sys.argv:                 # 33단계 — 같은 최적화의 대조 V1(커널생성 --자리하나 의 모듈)
        O자리수 = 1
    모양두기(gguf)
    print(f"모양: 너비 {D} · 머리 {NH} · KV 머리 {NKV} (GQA {NH // NKV})", flush=True)
    k = 글커널(ptx)
    결과 = {"커널 속성": k.속성(), "경우": []}
    print(json.dumps(결과["커널 속성"], ensure_ascii=False), flush=True)
    경우들 = []
    n = 300 if 빠르게 else 2100
    X = 무작위(n, 22)
    if 빠르게:
        경우들.append(("무작위 300", X, 1024, {"한 번에": [(0, n)], "나눠서": [(0, 100), (100, 150), (250, 50)],
                                          "한 행씩": [(0, 260)] + [(s, 1) for s in range(260, n)],
                                          "생성 판 행 1": [(0, 260)] + [(s, 1, "조각") for s in range(260, n)],
                                          "생성 판 행 4": [(0, 258)] + [(s, min(4, n - s), "조각") for s in range(258, n, 4)]}))
        if k.넷:
            경우들[-1][3]["생성 판 넷"] = [(0, 260)] + [(s, 1, "넷") for s in range(260, n)]
    else:
        경우들.append(("무작위 2100", X, 3072, {"한 번에": [(0, n)], "나눠서": [(0, 700), (700, 700), (1400, 700)],
                                           "한 행씩": [(0, 1024), (1024, 1000)] + [(s, 1) for s in range(2024, n)],
                                           "생성 판 행 1": [(0, 1024), (1024, 1000)] + [(s, 1, "조각") for s in range(2024, n)],
                                           "생성 판 행 4": [(0, 1024), (1024, 1022)] + [(s, min(4, n - s), "조각") for s in range(2046, n, 4)]}))
        if k.넷:
            경우들[-1][3]["생성 판 넷"] = [(0, 1024), (1024, 1000)] + [(s, 1, "넷") for s in range(2024, n)]
        for qi, 질문 in enumerate(질문들):
            ids = 채팅토큰(질문)
            Xr = 층0qkv(gguf, ids)
            m = len(ids)
            경우들.append((f"층 0 · 질문 {qi + 1} (토큰 {m})", Xr, 1024,
                        {"한 번에": [(0, m)], "나눠서": [(0, m // 2), (m // 2, m - m // 2)], "한 행씩": [(s, 1) for s in range(m)],
                         "생성 판 행 1": [(s, 1, "조각") for s in range(m)],
                         "생성 판 행 4": [(s, min(4, m - s), "조각") for s in range(0, m, 4)]}))
            if k.넷:
                경우들[-1][3]["생성 판 넷"] = [(s, 1, "넷") for s in range(m)]
    for 이름, X, 최대길이, 길들 in 경우들:
        t0 = time.time()
        cos, sin = C.rope표(최대길이, 1e6, hd=HD)
        참 = 참조(X, 최대길이, cos, sin)
        t1 = time.time()
        한 = {"경우": 이름, "행": int(X.shape[0]), "최대길이": 최대길이, "참조 초": round(t1 - t0, 1), "길": {}}
        for 길, 덩이들 in 길들.items():
            못 = k.못돌림(덩이들)
            if 못:                                  # 31단계 — 장치 한도: 그 길은 적고, RoPE 는 같은 덩이로 따로 본다
                한["길"][길] = {"덩이 수": len(덩이들), "장치 한도": 못}
                print(이름, 길, 한["길"][길], flush=True)
                if k.한도.get("RoPEKV정수로"):
                    continue
                글 = k.돌리기(X, 덩이들, 최대길이, cos, sin, RoPE만=True)
                길 = 길 + " (RoPE만)"
            else:
                글 = k.돌리기(X, 덩이들, 최대길이, cos, sin)
            한["길"][길] = {"덩이 수": len(덩이들), "다른 칸": 견주기(글, 참)}
            print(이름, 길, 한["길"][길], flush=True)
        결과["경우"].append(한)
    if k.넷 and not 빠르게:
        # 23단계 — 긴 문맥(묶음 아홉 — 접기 넓의 묶음 자리 여덟을 넘는다): 생성 판 넷 == 생성 판(22단계 — 위에서 CPU 참조와 같음을 본 판).
        # CPU 참조의 어텐션(n²)이 길어 GPU 둘을 견준다. 프롬프트 8296 행(묶음 판) 뒤 행 넷을 행 하나씩.
        n2 = 8300
        X2 = 무작위(n2, 23)
        cos, sin = C.rope표(8448, 1e6, hd=HD)
        앞 = [(a, min(1024, 8296 - a)) for a in range(0, 8296, 1024)]
        못 = k.못돌림(앞 + [(8296, 1, "조각"), (8296, 1, "넷")])
        if 못:                                      # 31단계 — 장치 한도
            길 = {"장치 한도": 못}
        else:
            옛 = k.돌리기(X2, 앞 + [(s, 1, "조각") for s in range(8296, n2)], 8448, cos, sin)
            새 = k.돌리기(X2, 앞 + [(s, 1, "넷") for s in range(8296, n2)], 8448, cos, sin)
            길 = {"다른 칸": {키: int(np.count_nonzero(np.asarray(옛[키]).view(np.uint8) != np.asarray(새[키]).view(np.uint8))) for 키 in 옛}}
        한 = {"경우": f"무작위 {n2} (긴 문맥 — 생성 판 넷 대 생성 판)", "행": n2, "최대길이": 8448, "길": {"생성 판 넷 == 생성 판": 길}}
        print(한["경우"], 길, flush=True)
        결과["경우"].append(한)
    결과["모두 0"] = all(v == 0 for c in 결과["경우"] for g in c["길"].values() for v in g.get("다른 칸", {}).values())
    결과["장치 한도"] = sum(1 for c in 결과["경우"] for g in c["길"].values() if "장치 한도" in g)
    print("모두 0:", 결과["모두 0"], "· 장치 한도 길:", 결과["장치 한도"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
