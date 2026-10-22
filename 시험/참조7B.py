#!/usr/bin/env python3
"""22a (2) — 7B 의 끝에서 끝: 글 엔진(엔진/시험_7B.gl — GeulPU 저장소)이 낸 모든 행 로짓(채팅 질문 셋 — 프롬프트 + 생성 16 토큰의 토큰열 S, 행마다)을
CPU 참조(정수 계약의 Qwen2 앞으로 — 계약/정수계약.py, 행렬곱은 행렬곱블라스)와 비트까지 견준다. 다른 칸이 0 이어야 한다.

  python -I 시험/참조7B.py <7B gguf> <글 도구의 출력폴더> <출력.json> [--자체시험만] [--빠른] [--어텐션 KV|QKV|KV8|V8] [--KV8] [--F2]
  (GeulPUv2: --빠른 은 빠른 계약 F1 — 선형층 자리 하나, --어텐션 은 2b단계 F1A 에뮬레이션 — 커널생성의 같은 플래그로 만든 모듈과 비교한다.
   --KV8 은 ③b-2 의 8비트 KV 캐시(계약 F1A′ — 커널생성 --빠른 --KV8 의 모듈) — --빠른 을 함께 뜻한다.
   --F2 는 빠른 어텐션 계약 F2(계약.md 2.2 — 커널생성 --빠른 --F2 의 모듈) — --빠른 · --KV8 을 함께 뜻한다)

먼저 빠른 판(행렬곱블라스)이 정수계약.행렬곱(자리수=2)과 비트까지 같은지 무작위 값과 7B 층 0 의 실제 가중치 조각으로 본다(문서/22단계.md 에
미리 적은 대로). 앞으로는 층마다 가중치를 한 번 풀어 세 경우의 행을 함께 행렬곱하고, RoPE · 어텐션은 경우마다(위치 0 부터).
37단계 — 구조 qwen3(Qwen3)도: 열쇠 앞머리는 general.architecture 에서, 치우침 텐서가 없으면 0(계약 4.5), RoPE 앞에 머리마다 q · k 정규화
(계약 4.6 — attn_q_norm · attn_k_norm), output.weight 가 없으면 낱말표(묶인 출력), 머리 크기는 key_length(없으면 너비 / 머리 수).
39단계 — 구조 llama(Llama 3.x · Mistral): q · k 무게의 줄을 머리마다 되돌리고(GGUF 변환기의 permute 를 거꾸로 — 계약 4.7), RoPE 표에
늘이기 인자(rope_freqs — 있으면), eps 는 파일의 값.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
for _d in ("계약",):
    sys.path.insert(0, os.path.join(ROOT, _d))
import gguf읽기 as GR       # noqa: E402
import 정수계약 as C        # noqa: E402

f32 = np.float32
선형자리수 = 2          # GeulPUv2 빠른 계약 F1(--빠른): 선형층의 활성값 자리 하나 — 어텐션은 자리 둘 그대로


def q8(g, name):
    """GGUF Q8_0 · Q4_0 텐서 → (q [N][K] int8 — Q4_0 은 니블 − 8, d [N][K/32] 짧은실수) — GGUF 의 값 그대로(23단계에 Q4_0)."""
    t, dims, raw = g.원(name)
    K, N = dims
    if t == GR.Q4_0:
        bl = np.asarray(raw).reshape(N, K // 32, 18)
        d = np.ascontiguousarray(bl[:, :, :2]).view(np.float16).reshape(N, K // 32).astype(f32)
        qs = bl[:, :, 2:]
        q = (np.concatenate([qs & 15, qs >> 4], 2).astype(np.int16) - 8).astype(np.int8).reshape(N, K)
        return q, d
    assert t == GR.Q8_0, name
    bl = np.asarray(raw).reshape(N, K // 32, 34)
    d = np.ascontiguousarray(bl[:, :, :2]).view(np.float16).reshape(N, K // 32).astype(f32)
    q = np.ascontiguousarray(bl[:, :, 2:]).view(np.int8).reshape(N, K)
    return q, d


def 가중(g, name):
    """26단계 — 텐서마다 형식: (형식, q [N][K], a, b). Q8_0 · Q4_0 은 (형식, q, d, None) — 계약 1절 4 (y ← 곱해더하기(u, d, y)),
    k-양자화(Q4_K · Q5_K · Q6_K)는 정수계약.풀기K."""
    t, dims, raw = g.원(name)
    if t in (GR.Q8_0, GR.Q4_0):
        q, d = q8(g, name)
        return ("Q8_0" if t == GR.Q8_0 else "Q4_0"), q, d, None
    K, N = dims
    return C.풀기K(t, raw, N, K)


def 곱(h, w, bias):
    """가중치의 형식에 맞는 CPU 참조 행렬곱(행렬곱블라스 · 행렬곱K블라스)."""
    if w[0] in ("Q8_0", "Q4_0"):
        return C.행렬곱블라스(h, w[1], w[2], bias, 자리수=선형자리수)
    return C.행렬곱K블라스(h, w, bias, 자리수=선형자리수)


def 다른칸(a, b):
    a, b = np.asarray(a, f32), np.asarray(b, f32)
    assert a.shape == b.shape, (a.shape, b.shape)
    return int(np.count_nonzero(a.view(np.int32) != b.view(np.int32)))


def 자체시험(g):
    """행렬곱블라스 == 행렬곱(자리수=2)."""
    r = np.random.default_rng(7)
    결과 = []
    for M, K, N in ((5, 256, 96), (37, 1024, 200), (3, 3584, 64)):
        x = (r.standard_normal((M, K)) * 10.0 ** r.uniform(-3, 3, (M, 1))).astype(f32)
        q = r.integers(-128, 128, (N, K), dtype=np.int64).astype(np.int8)
        d = (r.standard_normal((N, K // 32)) * 0.01).astype(np.float16).astype(f32)
        b = r.standard_normal(N).astype(f32)
        a = C.행렬곱블라스(x, q, d, b, 자리수=선형자리수)
        c = C.행렬곱(x, q.T.astype(np.int64), d.T, b, 자리수=선형자리수)
        결과.append({"꼴": f"무작위 M {M} · K {K} · N {N}", "다른 칸": 다른칸(a, c)})
    for t in (12, 13, 14):                                                   # 26단계 — k-양자화: 행렬곱K블라스 == 행렬곱K (무작위 블록)
        bb = {12: 144, 13: 176, 14: 210}[t]
        for M, K, N in ((5, 256, 96), (3, 3584, 64)):
            raw = r.integers(0, 256, (N, K // 256, bb), dtype=np.uint8)
            hh = (r.standard_normal((N, K // 256, 2)) * 0.01).astype(np.float16).view(np.uint8).reshape(N, K // 256, 4)
            if t == 14:
                raw[:, :, 208:210] = hh[:, :, :2]
            else:
                raw[:, :, 0:4] = hh
            w = C.풀기K(t, raw, N, K)
            x = (r.standard_normal((M, K)) * 10.0 ** r.uniform(-3, 3, (M, 1))).astype(f32)
            b = r.standard_normal(N).astype(f32)
            결과.append({"꼴": f"무작위 {w[0]} M {M} · K {K} · N {N}", "다른 칸": 다른칸(C.행렬곱K블라스(x, w, b, 자리수=선형자리수), C.행렬곱K(x, w, b, 자리수=선형자리수))})
    for 이름 in ("blk.0.ffn_gate.weight", "blk.0.ffn_down.weight"):          # K 는 텐서에서(24단계 — 모양과 상관없이)
        w = 가중(g, 이름)
        if w[0] not in ("Q8_0", "Q4_0"):                                       # 26단계 — 파일의 k-양자화
            w = (w[0], w[1][:96], w[2][:96], None if w[3] is None else w[3][:96])
            K = w[1].shape[1]
            x = (r.standard_normal((6, K)) * 0.5).astype(f32)
            결과.append({"꼴": f"{이름} ({w[0]}) 열 96 · K {K} · 행 6",
                       "다른 칸": 다른칸(C.행렬곱K블라스(x, w, np.zeros(96, f32), 자리수=선형자리수), C.행렬곱K(x, w, np.zeros(96, f32), 자리수=선형자리수))})
            continue
        q, d = w[1][:96], w[2][:96]
        K = q.shape[1]
        x = (r.standard_normal((6, K)) * 0.5).astype(f32)
        a = C.행렬곱블라스(x, q, d, np.zeros(96, f32), 자리수=선형자리수)
        c = C.행렬곱(x, q.T.astype(np.int64), np.ascontiguousarray(d.T), np.zeros(96, f32), 자리수=선형자리수)
        결과.append({"꼴": f"{이름} 열 96 · K {K} · 행 6", "다른 칸": 다른칸(a, c)})
    for x in 결과:
        print("자체시험", x, flush=True)
    return 결과


def 설정(g):
    """모델 머리의 설정 — 구조(qwen2 · qwen3), 층 · 머리 · KV 머리, eps, RoPE θ, 머리 크기(key_length — 없으면 None: 너비 / 머리 수)."""
    m = g.meta
    구조 = m.get("general.architecture", "qwen2")
    assert 구조 in ("qwen2", "qwen3", "llama"), f"받지 않는 구조 {구조}"
    k = 구조 + "."
    assert m.get(k + "rope.scaling.type", "none") == "none", "RoPE 늘이기 열쇠(rope.scaling.*)는 받지 않는다 — llama 의 늘이기는 rope_freqs 텐서로"
    return {"구조": 구조, "층": m[k + "block_count"], "머리": m[k + "attention.head_count"], "KV머리": m[k + "attention.head_count_kv"],
            "eps": float(m[k + "attention.layer_norm_rms_epsilon"]), "θ": float(m[k + "rope.freq_base"]),
            "머리크기": m.get(k + "attention.key_length")}


def 줄되돌리기(w, 머리수, hd):
    """39단계 — 계약 4.7: llama 의 q · k 무게(줄 = 머리수 × hd)를 머리마다 원래 차례로. GGUF 변환기(convert_hf_to_gguf.py 의 permute)는
    머리 하나를 [2][hd/2] 로 보고 두 축을 바꿔 원래 줄 h · hd/2 + j 를 파일의 줄 2j + h 에 두었다 — 거꾸로 새 줄 j < hd/2 는 파일의 2j,
    hd/2 + j 는 2j + 1. 줄을 통째로 옮길 뿐이다(Q8_0 · k-양자화는 줄마다 따로 양자화되어 있다)."""
    한 = np.concatenate([np.arange(0, hd, 2), np.arange(1, hd, 2)])
    차례 = (np.arange(머리수)[:, None] * hd + 한[None, :]).reshape(-1)
    if isinstance(w, tuple):                                          # (형식, q, d 또는 a, b) — 배열은 줄마다, 형식 · None 은 그대로
        return tuple(a[차례] if isinstance(a, np.ndarray) else a for a in w)
    return np.asarray(w)[차례]


def 치우침(g, 이름, n):
    """치우침 텐서(없으면 0 — 계약 4.5: 치우침 없는 선형은 0 을 더한다)."""
    return g.풀기(이름) if 이름 in g.tensors else np.zeros(n, f32)


def 앞으로(g, 경우, 말=True, 끝행만=False, 얀=None):
    """정수 계약의 앞으로 — 토큰열들(경우마다 위치 0 부터)의 모든 행 로짓 [Σ 행][어휘]과 경우의 시작 행. 끝행만: 경우마다 끝 행의 로짓만
    [경우][어휘] (탐욕 생성 — 시험/참조생성.py; 행마다의 값은 같다). 얀: YaRN (배, 원래 길이)(계약 4.8) — 서버가 그 요청을 셈할 때 쓴
    배(재현 파일 · 실행 로그의 yarn). 없으면 늘이지 않는다(#23 — 긴 문맥의 서버가 낸 답을 늘이지 않은 RoPE 로 다시 셈해 '다름'이 났다)."""
    s_ = 설정(g)
    NL, NH, NKV, eps, θ = s_["층"], s_["머리"], s_["KV머리"], s_["eps"], s_["θ"]
    qwen3 = s_["구조"] == "qwen3"
    llama = s_["구조"] == "llama"
    we = 가중(g, "token_embd.weight")
    D = we[1].shape[1]
    HD = int(s_["머리크기"] or D // NH)
    A = NH * HD                                                       # 질의 폭(1.0.0 1-6a — 너비와 다를 수 있다: Qwen3-0.6B · 4B)
    KVW = NKV * HD
    G_ = NH // NKV
    if we[0] in ("Q8_0", "Q4_0"):
        eq, ed = we[1], we[2]
        x = np.concatenate([(eq[S].astype(f32) * np.repeat(ed[S], 32, axis=1)).astype(f32) for S in 경우], 0)
        del eq, ed
    else:
        x = np.concatenate([C.임베딩K(we, S) for S in 경우], 0)              # 26단계
    del we
    시작 = np.cumsum([0] + [len(S) for S in 경우])
    인자 = g.풀기("rope_freqs.weight") if llama and "rope_freqs.weight" in g.tensors else None      # 39단계 — 계약 4.7 (2)
    cos, sin = C.rope표(max(64, int(max(len(S) for S in 경우))), θ, hd=HD, 인자=인자, 얀=얀)
    for l in range(NL):
        tl = time.time()
        p = f"blk.{l}."
        h = C.rms정규화(x, g.풀기(p + "attn_norm.weight"), eps)
        ws = [가중(g, p + f"attn_{a}.weight") for a in ("q", "k", "v")]
        bs = [치우침(g, p + f"attn_{a}.bias", w[1].shape[0]) for w, a in zip(ws, ("q", "k", "v"))]
        if llama:                                                                       # 39단계 — 계약 4.7 (1): q · k 의 줄 되돌리기
            ws[0], bs[0] = 줄되돌리기(ws[0], NH, HD), 줄되돌리기(bs[0], NH, HD)
            ws[1], bs[1] = 줄되돌리기(ws[1], NKV, HD), 줄되돌리기(bs[1], NKV, HD)
        if all(w[0] == ws[0][0] and w[0] in ("Q8_0", "Q4_0") for w in ws):          # 22 · 23단계와 같은 길
            qkv = C.행렬곱블라스(h, np.concatenate([w[1] for w in ws]), np.concatenate([w[2] for w in ws]), np.concatenate(bs),
                              자리수=선형자리수)                                        # GeulPUv2 F1 — 이 길도 선형자리수(빠뜨렸던 것)
        else:                                                                           # 26단계 — 텐서마다 형식(열마다 따로라 값은 같다)
            qkv = np.concatenate([곱(h, w, b) for w, b in zip(ws, bs)], 1)
        del ws
        if qwen3:                                                                       # 37단계 — 계약 4.6: RoPE 앞, v 는 그대로
            qkv = np.concatenate([C.머리정규화(qkv[:, :A], g.풀기(p + "attn_q_norm.weight"), eps, HD),
                                  C.머리정규화(qkv[:, A:A + KVW], g.풀기(p + "attn_k_norm.weight"), eps, HD), qkv[:, A + KVW:]], 1)
        ys = []
        for s, S in enumerate(경우):
            n = len(S)
            r = qkv[시작[s]:시작[s + 1]]
            위치 = np.arange(n)
            qh = [C.rope(r[:, HD * a:HD * a + HD], cos, sin, 위치) for a in range(NH)]
            kh = [C.rope(r[:, A + HD * a:A + HD * a + HD], cos, sin, 위치) for a in range(NKV)]
            vh = [np.ascontiguousarray(r[:, A + KVW + HD * a:A + KVW + HD * a + HD]) for a in range(NKV)]
            ys.append(np.concatenate([C.어텐션(qh[a], kh[a // G_], vh[a // G_]) for a in range(NH)], 1))
        y = np.concatenate(ys, 0)
        x = (x + 곱(y, 가중(g, p + "attn_output.weight"), np.zeros(D, f32))).astype(f32)
        h = C.rms정규화(x, g.풀기(p + "ffn_norm.weight"), eps)
        wg = 가중(g, p + "ffn_gate.weight")
        F = wg[1].shape[0]
        gg = 곱(h, wg, np.zeros(F, f32))
        del wg
        uu = 곱(h, 가중(g, p + "ffn_up.weight"), np.zeros(F, f32))
        f = C.swiglu(gg, uu)
        x = (x + 곱(f, 가중(g, p + "ffn_down.weight"), np.zeros(D, f32))).astype(f32)
        if 말:
            print(f"층 {l}: {time.time() - tl:.1f} 초", flush=True)
    if 끝행만:
        x = x[시작[1:] - 1]
    h = C.rms정규화(x, g.풀기("output_norm.weight"), eps)
    wh = 가중(g, "output.weight" if "output.weight" in g.tensors else "token_embd.weight")     # 37단계 — 묶인 출력(Qwen3-1.7B)
    로짓 = 곱(h, wh, np.zeros(wh[1].shape[0], f32))
    return 로짓, 시작


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    global 선형자리수
    if "--빠른" in sys.argv or "--KV8" in sys.argv or "--F2" in sys.argv:
        선형자리수 = 1
    if "--KV8" in sys.argv or "--F2" in sys.argv:     # GeulPUv2 ③b-2 — 8비트 KV 캐시(계약 F1A′)
        C.KV8 = True
    if "--F2" in sys.argv:                      # GeulPUv2 빠른 어텐션 계약 F2(계약.md 2.2)
        C.F2 = True
    인자 = list(sys.argv[1:])
    if "--어텐션" in 인자:                      # GeulPUv2 2b단계 — F1A 에뮬레이션 · ③b-1 — 8비트 KV 에뮬레이션(커널생성 --어텐션한자리 의 모듈)
        i = 인자.index("--어텐션")
        assert 인자[i + 1] in ("KV", "QKV", "KV8", "V8"), 인자[i + 1]
        C.어텐션한자리 = 인자[i + 1].lower()
        del 인자[i:i + 2]
    gguf, 폴더, 출력 = [a for a in 인자 if not a.startswith("--")][:3]
    g = GR.GGUF(gguf)
    결과 = {"자체시험": 자체시험(g)}
    if "--자체시험만" in sys.argv:
        json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return
    # 경우들 — 글 도구가 쓴 토큰열 S 와 로짓
    경우 = []
    i = 0
    while os.path.exists(os.path.join(폴더, f"경우{i}_토큰.bin")):
        S = np.fromfile(os.path.join(폴더, f"경우{i}_토큰.bin"), np.int64)
        경우.append(S)
        i += 1
    print("경우", [len(S) for S in 경우], flush=True)
    t0 = time.time()
    로짓, 시작 = 앞으로(g, 경우)
    V = 로짓.shape[1]
    결과["경우"] = []
    for s, S in enumerate(경우):
        A = np.fromfile(os.path.join(폴더, f"경우{s}_로짓.bin"), f32).reshape(len(S), V)
        R = 로짓[시작[s]:시작[s + 1]]
        한 = {"토큰": len(S), "다른 칸": 다른칸(A, R), "칸": int(A.size), "행마다 다른 칸": [다른칸(A[k], R[k]) for k in range(len(S))]}
        결과["경우"].append(한)
        print(f"경우 {s}: 행 {len(S)} — 다른 칸 {한['다른 칸']} / {한['칸']}", flush=True)
    결과["CPU 참조 초"] = round(time.time() - t0, 1)
    결과["모두 0"] = all(c["다른 칸"] == 0 for c in 결과["경우"]) and all(x["다른 칸"] == 0 for x in 결과["자체시험"])
    print("모두 0:", 결과["모두 0"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
