#!/usr/bin/env python3
"""0.12.0 12-3h-1 — Gemma 4(GGUF 구조 gemma4)의 수학 참조: 파일의 가중치를 풀어(양자화 값 그대로) 활성값을 fp64 로 계산한 순전파.
qwen35수학.py 와 같은 쓰임 — GGUF 의 뜻(텐서의 배치 · 층의 차례 · 두 어텐션 모양)을 llama.cpp 와 맞춰 본다(문서/블록조합.md 의 'Gemma 4 (12-3h)').
dense(12B · 31B)와 E2B · E4B(층마다의 임베딩(PLE) · KV 공유 · 층마다의 FFN 폭 — 0.24.0 의 24-5), 26B-A4B(MoE — 아래).

  python -I 시험/gemma4수학.py <gemma4 gguf> <llama.cpp 폴더> <작업 폴더> [--text <경로[@바이트]> …] [--length 256] [--no-llama]
  python -I 시험/gemma4수학.py <gemma4 gguf> <llama.cpp 폴더> <작업 폴더> --채팅 [--n 100]
  두 쓰임 모두 --llama-only(llama.cpp 의 토큰 · 로짓만 남기고 끝낸다 — GPU 줄 안에서) → --saved-llama(그것으로 수학 참조의 순전파와
  비교만 — GPU 없이)로 나눠 돌릴 수 있다.

판정은 qwen35수학.py 와 같다 — 끝 줄 '수학 참조: 맞음' 은 평균 KL(참조 ‖ llama.cpp) < 0.01 · 1위 일치 ≥ 95 %. llama.cpp 는 두 설정 —
기본(플래시 어텐션 자동 · KV f16 — 라마{i}.npy, 판정은 이것으로)과 플래시 어텐션 끔 · KV f32(라마f32{i}.npy) — 으로 돌리고 둘끼리의 KL 도 적는다
(Gemma 4 의 글에서는 llama.cpp 의 두 설정끼리도 판정을 넘지 못한다 — 문서/0.13.0.md 의 12-3h-1).
--채팅: 채팅 확인 — 채팅 프롬프트 셋에 llama.cpp(플래시 어텐션 끔 · KV f32)가 그리디로 이어 쓴 답 N 토큰을 붙여(채팅토큰.json), 답을 예측하는
행에서 수학 참조의 1위가 그 그리디(실제 다음 토큰)와 같은지 · KL(채팅수학{i}.npy · 채팅라마{i}.npy). 끝 줄 '채팅 확인: 맞음' 은 1위 일치 ≥ 95 % ·
평균 KL < 0.01.

층(llama.cpp 의 src/models/gemma4.cpp 와 같은 식 — master 7894e8b):
  입력 x = token_embd[토큰] · √너비
  층 l: 슬라이딩 윈도 층(attention.sliding_window_pattern[l] 참)과 전체 어텐션 층의 모양이 다르다 —
    헤드 크기 key_length_swa · key_length, KV 헤드 attention.head_count_kv[l], RoPE θ rope.freq_base_swa · rope.freq_base 와 차원
    rope.dimension_count_swa · rope.dimension_count(NEOX 짝 (i, i + n/2), θ_i = θ^(−2i/n) — 전체 층은 rope_freqs 로 나눈다: 인자가 10³⁰ 인
    차원은 사실상 돌지 않는다(12B 는 짝 64 개만 돈다))
    h = RMSNorm(x; attn_norm)
    q = RMSNorm(헤드마다 attn_q · h; attn_q_norm) → RoPE,  k₀ = attn_k · h,  v₀ = attn_v · h(없으면 v₀ = k₀ — 12B 의 전체 층)
    k = RMSNorm(헤드마다 k₀; attn_k_norm) → RoPE,  v = RMSNorm(헤드마다 v₀; 가중치 없이)
    어텐션: 점수 = q · k(배율 1 — 1/√d 없이), 인과 마스크, 슬라이딩 윈도 층은 (질의 위치 − 키 위치 ≥ sliding_window) 도 가린다,
      GQA(질의 헤드 j → KV 헤드 ⌊j / (H / H_kv)⌋), attn_output
    x ← x + RMSNorm(어텐션; post_attention_norm)
    x ← x + RMSNorm(ffn_down · (GELU(ffn_gate · h₂) ⊙ ffn_up · h₂); post_ffw_norm),  h₂ = RMSNorm(x; ffn_norm),
      GELU(z) = ½ z (1 + tanh(√(2/π) (z + 0.044715 z³)))(tanh 근사 — gelu_pytorch_tanh)
    (PLE — 아래) x ← x + RMSNorm(proj · (GELU(inp_gate · x) ⊙ p_l); post_norm)
    x ← x · layer_output_scale[l]
  끝: RMSNorm(x; output_norm), 로짓 = token_embd · x(출력은 임베딩과 묶임 — output.weight 가 있으면 그것), 소프트캡 c · tanh(로짓 / c)
  (c = final_logit_softcapping — 없으면 소프트캡 없음)

0.24.0 의 24-5 — E2B · E4B(llama.cpp b12224 의 gemma4.cpp):
  층마다의 입력(PLE — n = embedding_length_per_layer_input): 토큰마다 p_l = RMSNorm0(M_l · x₀; eps · 너비) ⊙ (per_layer_proj_norm / √2)
    + per_layer_token_embd[토큰][l n … (l + 1) n) · (√n / √2). x₀ 는 위의 입력(임베딩 · √너비), M_l 은 per_layer_model_proj 의 출력 행
    l n … (l + 1) n, RMSNorm0 은 가중치 없는 RMSNorm(그 n 칸에서). llama.cpp 는 원래 식의 곱 1/√너비(투영) · 1/√2(합)를 eps 와 가중치 ·
    임베딩의 스케일에 접어 넣었다 — 같은 값이다.
  KV 공유(shared_kv_layers = N): 뒤의 N 층(l ≥ L − N)은 자기 k · v 를 만들지 않고, 자기 KV 가 있는 마지막 층 가운데 같은 종류의 것을 본다 —
    슬라이딩 윈도 층은 층 L − N − 2, 전체 어텐션 층은 층 L − N − 1 의 k · v(정규화 · RoPE 뒤의 것 그대로). q 는 자기 attn_q · attn_q_norm.
    (llama.cpp 는 KV 를 쓰지 않는 그 층들에서 출력에 필요한 행만 계산한다 — 값은 같다.)
  FFN 폭은 층마다(feed_forward_length 가 배열 — E2B 는 KV 공유 층이 두 배).

0.24.0 의 24-5 — 26B-A4B(llama.cpp b12224 의 gemma4.cpp · build_moe_ffn): ffn_gate_inp 가 있는 층은 FFN 자리가 셋이다(a = 어텐션 잔차 뒤의 x):
  dense 가지 y₁ = RMSNorm(ffn_down · (GELU(ffn_gate · h₂) ⊙ ffn_up · h₂); post_ffw_norm_1),  h₂ = RMSNorm(a; ffn_norm)
  라우터: 로짓 = ffn_gate_inp · (RMSNorm0(a) / √너비 ⊙ ffn_gate_inp.scale), p = softmax(로짓), 상위 k(expert_used_count — 로짓의 내림차순,
    같으면 작은 번호), w_r = p_(e_r) / max(Σ p, 2⁻¹⁴) · ffn_down_exps.scale[e_r]
  전문가: [g | u] = ffn_gate_up_exps[e] · h₃(앞 F 행이 gate · 뒤 F 행이 up — F = expert_feed_forward_length),  h₃ = RMSNorm(a; pre_ffw_norm_2),
    y_e = ffn_down_exps[e] · (GELU(g) ⊙ u),  y₂ = RMSNorm(Σ_r w_r y_(e_r); post_ffw_norm_2)
  x = a + RMSNorm(y₁ + y₂; post_ffw_norm) — 그다음 층 배율. 고른 전문가를 {채팅}라우팅{i}.npy([층][행][k])로 남긴다.
  llama.cpp 는 전문가 배율을 down 의 출력에 곱한다(실수로는 같은 값). 파일이 이 PC 의 그래픽 메모리보다 커서 llama.cpp 는 전문가 행렬을
  호스트 메모리에 둔다(라마.py 의 전문가CPU — llama-server 의 --cpu-moe).

0.24.0 — Gemma 3(GGUF 구조 gemma3 — 계약 4.14)도 읽는다. llama.cpp 의 src/models/gemma3.cpp(b12224)와 같은 식 — 위에서 다른 것만:
  층의 모양이 모두 같다(헤드 크기 key_length · KV 헤드 head_count_kv). 층 l 은 l mod P < P − 1 이면 슬라이딩 윈도 층(P =
    sliding_window_pattern — 없으면 6), sliding_window 가 없으면 모든 층이 전체 어텐션 층
  RoPE: 윈도 층 θ = rope.freq_base_swa(없으면 10⁴), 전체 층 θ = rope.freq_base 에 선형 스케일링(각 = 위치 · θ_i / s, s = rope.scaling.factor —
    rope.scaling.type 이 linear 이거나 없을 때. 4B · 12B 는 8, 270M · 1B 는 없음)
  v 는 정규화하지 않는다, 점수 = (q · k) / √헤드 크기(층이 62 인 27B 만 1/√(너비 / 헤드 수)), 층 배율 없음
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "계약"))
sys.path.insert(0, HERE)
import gguf읽기 as GR       # noqa: E402
import qwen35수학 as Q      # noqa: E402 — 풀어 · 선형 · rms · 로그소프트맥스

f64 = np.float64


def gelu(z):
    return 0.5 * z * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * z * z * z)))


def rms0(x, eps):
    """가중치 없는 RMSNorm(마지막 축)."""
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps)


def 윈도층들(무늬, L):
    """층마다 슬라이딩 윈도 층인가 — 불 배열(Gemma 4)이면 그대로, 정수 P(Gemma 3 — 없으면 6)면 l mod P < P − 1(llama.cpp 의
    set_swa_pattern: P 가 0 이면 모든 층이 윈도 층)."""
    if isinstance(무늬, (list, tuple)):
        assert len(무늬) == L, (len(무늬), L)
        return [bool(x) for x in 무늬]
    P = 6 if 무늬 is None else int(무늬)
    return [P == 0 or l % P < P - 1 for l in range(L)]


def KV원층(l, n0, 윈도):
    """층 l 이 보는 키 · 값의 층(Gemma 4 의 KV 공유 — 계약 4.12 의 8, llama.cpp b12224 의 kv_source): 자기 KV 가 있는 층(l < n0)은 자기,
    그 뒤의 층은 같은 종류의 마지막 것 — 슬라이딩 윈도 층은 n0 − 2, 전체 어텐션 층은 n0 − 1."""
    return l if l < n0 else n0 - (2 if 윈도 else 1)


def 선형스케일(k):
    """Gemma 3 의 전체 층 RoPE 의 선형 스케일링 s(각 = 위치 · θ_i / s) — rope.scaling.type 이 linear(없으면 linear)이고 rope.scaling.factor 가
    0 이 아닐 때(llama.cpp: rope_freq_scale_train = 1 / factor, 종류가 none 이면 늘 1). 그 밖의 종류(yarn 등)는 받지 않는다."""
    종류 = k("rope.scaling.type") or "linear"
    assert 종류 in ("linear", "none"), f"rope.scaling.type {종류} 는 받지 않는다"
    s = k("rope.scaling.factor")
    if s is None:
        s = k("rope.scale_linear")                                  # 옛 이름(llama.cpp 가 factor 가 없을 때 읽는다)
    s = float(s or 0.0)
    return s if 종류 == "linear" and s != 0.0 else 1.0


class 모델:
    def __init__(self, path):
        self.g = g = GR.GGUF(path)
        m = g.meta
        구조 = self.구조 = m["general.architecture"]
        assert 구조 in ("gemma4", "gemma3"), 구조
        k = lambda x, d=None: m.get(구조 + "." + x, d)               # noqa: E731
        self.L, self.D = k("block_count"), k("embedding_length")
        self.고름수, self.전문가폭 = int(k("expert_used_count", 0) or 0), int(k("expert_feed_forward_length", 0) or 0)   # 26B-A4B — MoE
        self.라우팅 = []                                            # MoE 층마다 고른 전문가 [행][k]
        self.H = k("attention.head_count")
        hkv = k("attention.head_count_kv")
        self.Hkv = list(hkv) if isinstance(hkv, (list, tuple)) else [hkv] * self.L
        무늬 = k("attention.sliding_window_pattern")
        assert 구조 == "gemma3" or isinstance(무늬, (list, tuple)), "Gemma 4 의 sliding_window_pattern 은 층마다의 배열"
        self.swa = 윈도층들(무늬, self.L)
        hd = k("attention.key_length")
        self.hd = {True: k("attention.key_length_swa", hd), False: hd}
        assert k("attention.value_length_swa", self.hd[True]) == self.hd[True] and k("attention.value_length", hd) == hd
        self.rot = {True: k("rope.dimension_count_swa", self.hd[True]), False: k("rope.dimension_count", self.hd[False])}
        self.theta = {True: float(k("rope.freq_base_swa", 10000.0)), False: float(k("rope.freq_base", 10000.0))}
        self.창 = k("attention.sliding_window")
        self.eps = float(k("attention.layer_norm_rms_epsilon"))
        self.캡 = float(k("final_logit_softcapping", 0.0) or 0.0)
        self.인자 = g.풀기("rope_freqs.weight").astype(f64) if "rope_freqs.weight" in g.tensors else None
        self.묶인출력 = "output.weight" not in g.tensors
        self.npl = int(k("embedding_length_per_layer_input", 0) or 0)   # 0.24.0 24-5 — 층마다의 입력(PLE)의 너비(0 이면 없다)
        self.n0 = self.L - int(k("attention.shared_kv_layers", 0) or 0)  # 자기 KV 가 있는 층 수(그 뒤의 층은 KV 공유)
        assert self.n0 == self.L or (self.n0 >= 2 and not self.swa[self.n0 - 1] and self.swa[self.n0 - 2]), (
            "KV 공유: 자기 KV 가 있는 마지막 두 층이 윈도 층 · 전체 어텐션 층의 차례여야 한다")
        self.v정규화, self.눈금, self.스케일 = True, 1.0, 1.0            # Gemma 4 — v 도 정규화(가중치 없이) · 점수 배율 1 · 스케일링 없음
        if 구조 == "gemma3":                                         # 계약 4.14 — llama.cpp 의 gemma3.cpp(b12224)
            assert self.인자 is None, "Gemma 3 의 rope_freqs 는 받지 않는다"
            if not self.창:                                          # sliding_window 가 없으면 모든 층이 전체 어텐션 층
                self.swa = [False] * self.L
            self.v정규화 = False
            self.눈금 = 1.0 / math.sqrt(float(self.D // self.H if self.L == 62 else hd))
            self.스케일 = 선형스케일(k)

    def 돌리기(self, x, l, T):
        """NEOX RoPE — 짝 (i, i + n/2). 전체 층은 θ_i 를 rope_freqs[i] 로(Gemma 4) · 선형 스케일링 s 로(Gemma 3) 나눈다."""
        s = self.swa[l]
        n = self.rot[s] // 2
        f = self.theta[s] ** (-2.0 * np.arange(n, dtype=f64) / self.rot[s])
        if not s and self.인자 is not None:
            f = f / self.인자[:n]
        if not s and self.스케일 != 1.0:
            f = f / self.스케일
        각 = np.arange(T, dtype=f64)[:, None] * f[None, :]
        cos, sin = np.cos(각)[:, None, :], np.sin(각)[:, None, :]
        y = x.copy()
        a, b = x[..., :n], x[..., n:2 * n]
        y[..., :n] = a * cos - b * sin
        y[..., n:2 * n] = b * cos + a * sin
        return y

    def 어텐션(self, l, h, T, kv):
        g, H = self.g, self.H
        s = self.swa[l]
        원 = KV원층(l, self.n0, s)                                    # KV 공유 층은 같은 종류의, 자기 KV 가 있는 마지막 층의 것을 본다
        hd, Hk = self.hd[s], self.Hkv[원]
        q = Q.선형(h, g, f"blk.{l}.attn_q.weight").reshape(T, H, hd)
        q = self.돌리기(Q.rms(q, g.풀기(f"blk.{l}.attn_q_norm.weight"), self.eps), l, T)
        if 원 == l:
            k0 = Q.선형(h, g, f"blk.{l}.attn_k.weight").reshape(T, Hk, hd)
            이름v = f"blk.{l}.attn_v.weight"
            v0 = Q.선형(h, g, 이름v).reshape(T, Hk, hd) if 이름v in g.tensors else k0
            kk = self.돌리기(Q.rms(k0, g.풀기(f"blk.{l}.attn_k_norm.weight"), self.eps), l, T)
            vv = rms0(v0, self.eps) if self.v정규화 else v0           # Gemma 3 는 v 를 정규화하지 않는다
            if l >= self.n0 - 2:
                kv[l] = (kk, vv)                                     # 뒤의 KV 공유 층이 본다
        else:
            kk, vv = kv[원]
        묶 = H // Hk
        kk, vv = np.repeat(kk, 묶, 1), np.repeat(vv, 묶, 1)          # 질의 헤드 j → KV 헤드 j // 묶
        sc = np.einsum("thd,shd->hts", q, kk)                        # 배율 1(Gemma 4)
        if self.눈금 != 1.0:
            sc = sc * self.눈금                                       # Gemma 3 — 1/√헤드 크기
        i = np.arange(T)
        가림 = i[None, :] > i[:, None]
        if s:
            가림 = 가림 | (i[:, None] - i[None, :] >= self.창)
        sc = np.where(가림[None], -np.inf, sc)
        sc = np.exp(sc - sc.max(-1, keepdims=True))
        sc /= sc.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", sc, vv)
        return Q.선형(o.reshape(T, H * hd), g, f"blk.{l}.attn_output.weight")

    def 층입력(self, ids, x0):
        """층마다의 입력(PLE) [층][행][n] — 위 설명의 p_l."""
        g, T, L, n = self.g, len(ids), self.L, self.npl
        p = Q.풀어(g, "per_layer_token_embd.weight", ids).astype(f64).reshape(T, L, n) * (math.sqrt(n) / math.sqrt(2.0))
        y = rms0(Q.선형(x0, g, "per_layer_model_proj.weight").reshape(T, L, n), self.eps * self.D)
        y = y * (g.풀기("per_layer_proj_norm.weight").astype(f64) / math.sqrt(2.0))
        return np.ascontiguousarray((y + p).transpose(1, 0, 2))

    def MoE(self, l, a):
        """26B-A4B — 위 설명의 FFN 자리 셋(dense 가지 · 라우터 · 전문가)의 합 RMSNorm(y₁ + y₂; post_ffw_norm)(a 에 더할 값)."""
        g, T, p = self.g, a.shape[0], f"blk.{l}."
        h = Q.rms(a, g.풀기(p + "ffn_norm.weight"), self.eps)
        y1 = Q.rms(Q.선형(gelu(Q.선형(h, g, p + "ffn_gate.weight")) * Q.선형(h, g, p + "ffn_up.weight"), g, p + "ffn_down.weight"),
                   g.풀기(p + "post_ffw_norm_1.weight"), self.eps)
        r = rms0(a, self.eps) / math.sqrt(self.D) * g.풀기(p + "ffn_gate_inp.scale").astype(f64)
        로짓 = Q.선형(r, g, p + "ffn_gate_inp.weight")                 # (T, 전문가)
        pr = np.exp(로짓 - 로짓.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        고른 = np.argsort(-로짓, axis=-1, kind="stable")[:, :self.고름수]
        self.라우팅.append(고른.astype(np.int16))
        w = np.take_along_axis(pr, 고른, -1)
        w = w / np.maximum(w.sum(-1, keepdims=True), 2.0 ** -14) * g.풀기(p + "ffn_down_exps.scale").astype(f64)[고른]
        h3 = Q.rms(a, g.풀기(p + "pre_ffw_norm_2.weight"), self.eps)
        F = self.전문가폭
        o = np.zeros((T, self.D))
        for e in np.unique(고른):
            행, 순위 = np.nonzero(고른 == e)
            gu = h3[행] @ Q.풀어전문가(g, p + "ffn_gate_up_exps.weight", e).astype(f64).T
            y = (gelu(gu[:, :F]) * gu[:, F:]) @ Q.풀어전문가(g, p + "ffn_down_exps.weight", e).astype(f64).T
            o[행] += w[행, 순위][:, None] * y
        y2 = Q.rms(o, g.풀기(p + "post_ffw_norm_2.weight"), self.eps)
        return Q.rms(y1 + y2, g.풀기(p + "post_ffw_norm.weight"), self.eps)

    def 앞으로(self, ids):
        g, T = self.g, len(ids)
        x = Q.풀어(g, "token_embd.weight", ids).astype(f64) * math.sqrt(self.D)
        층입 = self.층입력(ids, x) if self.npl else None
        kv = {}
        for l in range(self.L):
            h = Q.rms(x, g.풀기(f"blk.{l}.attn_norm.weight"), self.eps)
            x = x + Q.rms(self.어텐션(l, h, T, kv), g.풀기(f"blk.{l}.post_attention_norm.weight"), self.eps)
            if f"blk.{l}.ffn_gate_inp.weight" in g.tensors:             # 26B-A4B — MoE(dense 가지 + 전문가)
                x = x + self.MoE(l, x)
            else:
                h = Q.rms(x, g.풀기(f"blk.{l}.ffn_norm.weight"), self.eps)
                f = Q.선형(gelu(Q.선형(h, g, f"blk.{l}.ffn_gate.weight")) * Q.선형(h, g, f"blk.{l}.ffn_up.weight"), g,
                          f"blk.{l}.ffn_down.weight")
                x = x + Q.rms(f, g.풀기(f"blk.{l}.post_ffw_norm.weight"), self.eps)
            if 층입 is not None:                                     # PLE — FFN 의 잔차 뒤, 층 배율 앞
                y = Q.선형(gelu(Q.선형(x, g, f"blk.{l}.inp_gate.weight")) * 층입[l], g, f"blk.{l}.proj.weight")
                x = x + Q.rms(y, g.풀기(f"blk.{l}.post_norm.weight"), self.eps)
            이름s = f"blk.{l}.layer_output_scale.weight"
            if 이름s in g.tensors:
                x = x * float(g.풀기(이름s)[0])
        x = Q.rms(x, g.풀기("output_norm.weight"), self.eps)
        이름 = "token_embd.weight" if self.묶인출력 else "output.weight"
        V = g.tensors[이름][1][1]
        로짓 = np.empty((T, V))
        for s in range(0, V, 16384):
            로짓[:, s:s + 16384] = x @ Q.풀어(g, 이름, np.arange(s, min(V, s + 16384))).astype(f64).T
        if self.캡:
            로짓 = self.캡 * np.tanh(로짓 / self.캡)
        return 로짓


채팅질문 = ["What is the capital of France? Answer in one sentence, then list three famous landmarks there.",
         "대한민국의 수도는 어디야? 한 문장으로 답하고, 그 도시의 유명한 곳 세 군데를 적어 줘.",
         "Write a Python function that returns the n-th Fibonacci number iteratively."]
채팅틀 = {"gemma4": "<|turn>user\n{}<turn|>\n<|turn>model\n",                # 사용자 차례 하나 + 생성 프롬프트(BOS 는 토크나이저가 붙인다)
         "gemma3": "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"}


def 라마부르기(판):
    os.environ["GEULPU_LLAMA"] = 판
    sys.path.insert(0, os.path.join(ROOT, "재기"))
    import 라마 as LM                                               # noqa: E402
    return LM


def 전문가CPU인가(gguf):
    """MoE 파일(26B-A4B — 16.8 GB)이면 llama.cpp 는 전문가 행렬을 호스트 메모리에 둔다(이 PC 의 그래픽 메모리 12 GB 에 들지 않는다)."""
    return any(n.endswith("_exps.weight") for n in GR.GGUF(gguf).tensors)


def 라마로짓(lm, ids):
    """llama.cpp 의 모든 행 로짓 — 위치 0 부터."""
    lm.지우기()
    lm.넣기(ids, list(range(len(ids))), [0] * len(ids), [1] * len(ids))
    return lm.모든로짓(len(ids))


def 채팅(gguf, 판, 작업, N):
    토큰파일 = os.path.join(작업, "채팅토큰.json")
    if "--saved-llama" in sys.argv:                                 # 앞의 --llama-only 가 남긴 토큰 · 로짓으로(GPU 없이)
        경우 = [(ids, P, np.load(os.path.join(작업, f"채팅라마{i}.npy")))
              for i, (ids, P) in enumerate(json.load(open(토큰파일, encoding="utf-8"))["경우"])]
    else:
        경우 = 채팅라마(gguf, 판, N)
        json.dump({"경우": [[ids, P] for ids, P, _ in 경우]}, open(토큰파일, "w", encoding="utf-8"))
        for i, (_, _, z) in enumerate(경우):
            np.save(os.path.join(작업, f"채팅라마{i}.npy"), z)
        if "--llama-only" in sys.argv:
            return
    m = 모델(gguf)
    kls, 맞음, 행수 = [], 0, 0
    for i, (ids, P, z) in enumerate(경우):
        t0 = time.time()
        m.라우팅 = []
        p = Q.로그소프트맥스(m.앞으로(ids))
        np.save(os.path.join(작업, f"채팅수학{i}.npy"), p.astype(np.float32))
        if m.라우팅:
            np.save(os.path.join(작업, f"채팅라우팅{i}.npy"), np.stack(m.라우팅))   # [MoE 층][행][k]
        r = Q.로그소프트맥스(z)
        kl = (np.exp(p) * (p - r)).sum(-1)
        답 = slice(P - 1, len(ids) - 1)                              # 행 P − 1 … 끝 − 2 가 답 토큰을 예측
        같 = int((p[답].argmax(1) == np.array(ids[P:])).sum())
        kls.append(kl[답])
        맞음 += 같
        행수 += len(ids) - P
        print(f"경우 {i}: {time.time() - t0:.0f} 초 — 답 구간 {len(ids) - P} 행: 수학 참조의 1위 = llama.cpp 의 그리디 {같}/{len(ids) - P} · "
              f"KL 평균 {kl[답].mean():.3e} · 최대 {kl[답].max():.3e} | 프롬프트 구간 KL 평균 {kl[:P - 1].mean():.3e}", flush=True)
    평균 = float(np.concatenate(kls).mean())
    print(f"모두: 답 {행수} 행 — 1위 = 그리디 {100 * 맞음 / 행수:.2f} % · KL 평균 {평균:.3e}")
    좋음 = 맞음 / 행수 >= 0.95 and 평균 < 0.01
    print("채팅 확인: 맞음" if 좋음 else "채팅 확인: 다름")
    sys.exit(0 if 좋음 else 1)


def 채팅라마(gguf, 판, N):
    """llama.cpp(플래시 어텐션 끔 · KV f32)가 채팅 프롬프트 셋을 그리디로 N 토큰 이어 쓴 것 — [(토큰열, 프롬프트 길이, 모든 행의 로짓)]"""
    import ctypes
    LM = 라마부르기(판)
    lm = LM.라마(gguf, n_ctx=512, n_batch=512, n_ubatch=512, n_seq_max=1, flash=0, kv형식=0, 전문가CPU=전문가CPU인가(gguf))
    L = lm.L
    L.llama_model_get_vocab.restype = ctypes.c_void_p
    L.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    L.llama_tokenize.restype = ctypes.c_int32
    L.llama_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
                                 ctypes.c_bool, ctypes.c_bool]
    어휘 = L.llama_model_get_vocab(lm.model)
    경우 = []
    틀 = 채팅틀[GR.GGUF(gguf).meta["general.architecture"]]
    for 질문 in 채팅질문:
        글 = 틀.format(질문).encode("utf-8")                         # GGUF 의 채팅 템플릿(생성 프롬프트까지)
        buf = (ctypes.c_int32 * 1024)()
        ids = [int(x) for x in buf[:L.llama_tokenize(어휘, 글, len(글), buf, 1024, True, True)]]      # BOS · 특수 토큰
        P = len(ids)
        lm.지우기()
        lm.넣기(ids, list(range(P)), [0] * P, [0] * (P - 1) + [1])
        for _ in range(N):
            t = int(np.argmax(lm.로짓(-1)))                          # 그리디
            ids.append(t)
            lm.넣기([t], [len(ids) - 1], [0], [1])
        경우.append((ids, P, 라마로짓(lm, ids)))
        print(f"프롬프트 {P} + 답 {N} 토큰 — 앞 {ids[:6]} … 답의 처음 {ids[P:P + 8]}", flush=True)
    lm.닫기()
    return 경우


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    gguf, 판, 작업 = sys.argv[1:4]
    os.makedirs(작업, exist_ok=True)
    if "--채팅" in sys.argv:
        return 채팅(gguf, 판, 작업, int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100)
    길이 = int(sys.argv[sys.argv.index("--length") + 1]) if "--length" in sys.argv else 256
    글들 = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--text"] or [os.path.join(ROOT, "연구자안내.md"),
                                                                                   os.path.join(ROOT, "README.md")]
    if "--saved-llama" in sys.argv:                                 # 앞의 --llama-only 가 남긴 토큰 · 로짓으로(GPU 없이)
        토큰들 = json.load(open(os.path.join(작업, "토큰.json"), encoding="utf-8"))["토큰"]
    else:
        import 품질 as QQ                                            # noqa: E402 — 텍스트 고르기 · llama.cpp 의 llama_tokenize
        어휘 = QQ.라마어휘(판, gguf)
        토큰들 = [어휘.토큰(QQ._글바이트(지정)[1], 길이) for 지정 in 글들]
        json.dump({"gguf": os.path.basename(gguf), "글": 글들, "토큰": 토큰들}, open(os.path.join(작업, "토큰.json"), "w", encoding="utf-8"))
    if "--no-llama" not in sys.argv and "--saved-llama" not in sys.argv:
        LM = 라마부르기(판)
        for 이름, flash, kv in (("라마", -1, 1), ("라마f32", 0, 0)):     # 기본 · 플래시 어텐션 끔 + KV f32
            lm = LM.라마(gguf, n_ctx=길이, n_batch=길이, n_ubatch=min(512, 길이), n_seq_max=1, flash=flash, kv형식=kv,
                        전문가CPU=전문가CPU인가(gguf))
            for i, ids in enumerate(토큰들):
                np.save(os.path.join(작업, f"{이름}{i}.npy"), 라마로짓(lm, ids))
            lm.닫기()
    if "--llama-only" in sys.argv:
        return
    m = 모델(gguf)
    print(f"{os.path.basename(gguf)}: 층 {m.L}(전체 어텐션 {sum(not s for s in m.swa)}) · 너비 {m.D} · 질의 헤드 {m.H} · "
          f"윈도 층 헤드 {m.hd[True]} × KV {m.Hkv[m.swa.index(True)]} · 전체 층 헤드 {m.hd[False]} × KV {m.Hkv[m.swa.index(False)]} · "
          f"윈도 {m.창} · 소프트캡 {m.캡:g} · 임베딩과 묶인 출력 {m.묶인출력}", flush=True)
    kls, 끼리들, 같음, 같끼리, 행수 = [], [], 0, 0, 0
    for i, ids in enumerate(토큰들):
        t0 = time.time()
        m.라우팅 = []
        p = Q.로그소프트맥스(m.앞으로(ids))
        np.save(os.path.join(작업, f"수학{i}.npy"), p.astype(np.float32))
        if m.라우팅:
            np.save(os.path.join(작업, f"라우팅{i}.npy"), np.stack(m.라우팅))       # [MoE 층][행][k]
        if "--no-llama" in sys.argv:
            print(f"텍스트 {i}: {len(ids)} 토큰 · {time.time() - t0:.0f} 초 — 1위(앞 8 행) {p[:8].argmax(1).tolist()}", flush=True)
            continue
        r = Q.로그소프트맥스(np.load(os.path.join(작업, f"라마{i}.npy")))
        r2 = Q.로그소프트맥스(np.load(os.path.join(작업, f"라마f32{i}.npy")))
        kl = (np.exp(p) * (p - r)).sum(-1)
        kl2 = (np.exp(p) * (p - r2)).sum(-1)
        끼리 = 0.5 * ((np.exp(r) * (r - r2)).sum(-1) + (np.exp(r2) * (r2 - r)).sum(-1))
        같 = int((p.argmax(1) == r.argmax(1)).sum())
        kls.append(kl)
        끼리들.append(끼리)
        같음 += 같
        같끼리 += int((r.argmax(1) == r2.argmax(1)).sum())
        행수 += len(ids)
        print(f"텍스트 {i}: {len(ids)} 토큰 · {time.time() - t0:.0f} 초 — KL 평균 {kl.mean():.3e} · 최대 {kl.max():.3e} · 1위 일치 {같}/{len(ids)}",
              flush=True)
        print(f"  llama.cpp 플래시 어텐션 끔 · KV f32: KL 평균 {kl2.mean():.3e} · 1위 일치 {int((p.argmax(1) == r2.argmax(1)).sum())}/{len(ids)} | "
              f"llama.cpp 의 두 설정끼리: 대칭 KL 평균 {끼리.mean():.3e} · 1위 일치 {int((r.argmax(1) == r2.argmax(1)).sum())}/{len(ids)}", flush=True)
    if kls:
        평균 = float(np.concatenate(kls).mean())
        일치 = 같음 / 행수
        평균끼리 = float(np.concatenate(끼리들).mean())
        print(f"모두: KL 평균 {평균:.3e} · 1위 일치 {100 * 일치:.2f} % | llama.cpp 의 두 설정끼리: 대칭 KL 평균 {평균끼리:.3e} · "
              f"1위 일치 {100 * 같끼리 / 행수:.2f} %")
        if 평균끼리 >= 0.01 or 같끼리 / 행수 < 0.95:
            print("llama.cpp 의 두 설정끼리도 판정을 넘지 못한다 — 이 글에서는 출력이 작은 수치 차이에 민감하다(--채팅 의 확인을 보라)")
        print("수학 참조: 맞음" if 평균 < 0.01 and 일치 >= 0.95 else "수학 참조: 다름")
        sys.exit(0 if 평균 < 0.01 and 일치 >= 0.95 else 1)


if __name__ == "__main__":
    main()
