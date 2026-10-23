#!/usr/bin/env python3
"""0.12.0 12-3a — Qwen3.5(GGUF 구조 qwen35)의 수학 참조: 파일의 가중치를 풀어(양자화 값 그대로) 활성값을 fp64 로 계산한 순전파(forward pass).
엔진의 계약(정수 블록 · 적힌 지수)이 아니라 "양자화된 가중치 그대로의 이상적인 모델"이다 — GGUF 의 뜻(텐서의 배치 · 층의 차례 · 헤드의
짝 · 합성곱 상태)을 llama.cpp 와 맞춰 보는 용도(문서/블록조합.md 의 12-3a). 계약의 CPU 참조(정수계약.py)는 12-3b 에서 이것과 비교한다.

  python -I 시험/qwen35수학.py <qwen35 gguf> <llama.cpp 폴더> <작업 폴더> [--text <경로[@바이트]> …] [--length 256] [--no-llama]

텍스트마다 앞 --length 토큰(llama.cpp 의 llama_tokenize — BOS 는 모델대로, 특수 토큰은 풀지 않음)의 모든 행 로짓을 (가) 이 참조, (나) llama.cpp
(f16 KV · 모든 층 GPU — 재기/라마.py)로 내어 행마다 KL(참조 ‖ llama.cpp) · 1위 일치를 본다. 끝 줄 '수학 참조: 맞음' 은 평균 KL < 0.01 ·
1위 일치 ≥ 95 % (GGUF 를 잘못 읽으면 KL 이 0.1 을 크게 넘는다 — 헤드 짝 · 게이트 배치 · 합성곱 차례 하나만 틀려도).

층(llama.cpp 의 src/models/qwen35.cpp 와 같은 식):
  x ← x + 섞기(RMSNorm(x; attn_norm)),  x ← x + FFN(RMSNorm(x; post_attention_norm))  — FFN 은 SwiGLU(ffn_gate · ffn_up · ffn_down)
  섞기 = 전체 어텐션((l + 1) % full_attention_interval == 0) 또는 선형 어텐션(gated delta net — 나머지)
  전체 어텐션: attn_q 가 헤드마다 [q(헤드 크기), 게이트(헤드 크기)] — q · k 에 헤드마다 RMSNorm(attn_q_norm · attn_k_norm), RoPE 는 헤드의 앞
    rope.dimension_count 차원만(짝 (i, i + n/2), θ_i = freq_base^(−2i/n) — 텍스트만이면 MRoPE 의 섹션 위치가 모두 같다), 점수 × 1/√헤드 크기,
    인과 소프트맥스, GQA(질의 헤드 h → KV 헤드 ⌊h / (H / H_kv)⌋), 출력 × sigmoid(게이트), attn_output
  선형 어텐션: attn_qkv → [q(H_k·d) | k(H_k·d) | v(H_v·d)] 를 합성곱(ssm_conv1d — 채널마다 커널 4, 처음 상태 0)과 SiLU 에 지나게,
    q · k 는 헤드마다 L2 정규화(x / max(‖x‖, eps)), q × 1/√d, β = sigmoid(ssm_beta·x), g = softplus(ssm_alpha·x + ssm_dt) · ssm_a,
    v 헤드 j 는 k 헤드 j mod H_k(llama.cpp 의 ggml_repeat — 헤드 목록을 통째로 되풀이한 차례), 토큰마다 상태 S(d × d): S ← e^g · S,
    δ = (v − Sᵀk) · β, S ← S + k δᵀ, o = Sᵀq. 출력 RMSNorm(o; ssm_norm — 헤드마다) × SiLU(attn_gate·x), ssm_out
  끝: RMSNorm(x; output_norm), 로짓 = output(없으면 token_embd — 임베딩과 공유한 출력)
0.12.0 12-3g — 구조 qwen35moe(Qwen3.5 · 3.6-35B-A3B): FFN 이 MoE — 라우터(ffn_gate_inp) softmax 의 상위 expert_used_count 개를 합으로 나눈
  무게로 전문가 SwiGLU(ffn_gate_exps · ffn_up_exps · ffn_down_exps — [전문가][…] 3 차원)를 더하고, 공유 전문가(ffn_*_shexp)에
  sigmoid(ffn_gate_inp_shexp · x) 를 곱해 더한다(llama.cpp 의 build_moe_ffn — softmax · norm_w). 고른 전문가를 라우팅{i}.npy([층][행][k])로 남긴다.
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
import 정수계약 as C        # noqa: E402

f64 = np.float64


def 풀어(g, name, 행=None):
    """텐서 → fp32 배열 (N, K) (1 차원은 (K,)) — 양자화 값 그대로(k-양자화는 계약의 풀기K 와 같은 a · q − b). 행: 행 번호 배열(임베딩의 일부)."""
    t, dims, raw = g.원(name)
    if len(dims) == 1 or t in (GR.F32, GR.F16):
        x = g.풀기(name)
        return x if 행 is None else x[행]
    K, N = dims[0], dims[1]
    vb, bb = GR.블록[t]
    줄바이트 = K // vb * bb
    if 행 is not None:
        r = raw.reshape(N, 줄바이트)[np.asarray(행)].reshape(-1)
        N = len(행)
    else:
        r = raw
    return 풀기바이트(t, r, N, K)


def 풀어전문가(g, name, e):
    """0.12.0 12-3g — 3 차원 전문가 텐서 [전문가][N][K] 의 전문가 e 하나 → fp32 (N, K)."""
    t, dims, raw = g.원(name)
    K, N = dims[0], dims[1]
    if t == GR.F32:
        return np.asarray(raw).view(np.float32).reshape(-1, N, K)[e].copy()
    vb, bb = GR.블록[t]
    한 = N * (K // vb * bb)
    return 풀기바이트(t, np.asarray(raw)[e * 한:(e + 1) * 한], N, K)


def 풀기바이트(t, r, N, K):
    """양자화 블록 바이트 r(행 N · 깊이 K) → fp32 (N, K)."""
    vb, bb = GR.블록[t]
    if t in (GR.Q8_0, GR.Q4_0):
        bl = r.reshape(-1, bb)
        d = bl[:, :2].copy().view(np.float16).astype(np.float32)
        if t == GR.Q8_0:
            q = bl[:, 2:].view(np.int8).astype(np.float32)
        else:
            qs = bl[:, 2:]
            q = np.concatenate([(qs & 15).astype(np.int8) - 8, (qs >> 4).astype(np.int8) - 8], 1).astype(np.float32)
        return (q * d).astype(np.float32).reshape(N, K)
    형식, q, a, b = C.풀기K(t, r, N, K)
    q = q.astype(np.float32)
    if 형식 == "Q6_K":
        return (q.reshape(N, K // 16, 16) * a[..., None]).reshape(N, K)
    x = q.reshape(N, K // 32, 32) * a[..., None]
    if b is not None:
        x = x - b[..., None]
    return x.reshape(N, K).astype(np.float32)


def 선형(x, g, name):
    return x @ 풀어(g, name).astype(f64).T


def rms(x, w, eps):
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w.astype(f64)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


class 모델:
    def __init__(self, path):
        self.g = g = GR.GGUF(path)
        m = g.meta
        구조 = m["general.architecture"]
        assert 구조 in ("qwen35", "qwen35moe"), 구조
        k = lambda x: m[구조 + "." + x]                          # noqa: E731
        self.moe = 구조 == "qwen35moe"                              # 0.12.0 12-3g
        if self.moe:
            self.전문가수, self.고름수 = k("expert_count"), k("expert_used_count")
        self.라우팅 = []                                            # MoE 층마다 고른 전문가 [행][k]
        self.L, self.D = k("block_count"), k("embedding_length")
        self.H, self.Hkv, self.hd = k("attention.head_count"), k("attention.head_count_kv"), k("attention.key_length")
        assert k("attention.value_length") == self.hd
        self.eps = float(k("attention.layer_norm_rms_epsilon"))
        self.rot, self.theta = k("rope.dimension_count"), float(k("rope.freq_base"))
        self.간격 = k("full_attention_interval")
        self.d, self.Hk, self.Hv = k("ssm.state_size"), k("ssm.group_count"), k("ssm.time_step_rank")
        self.합성곱 = k("ssm.conv_kernel")
        assert k("ssm.inner_size") == self.Hv * self.d and self.Hv % self.Hk == 0
        self.묶인출력 = "output.weight" not in g.tensors

    def 선형층인가(self, l):
        return (l + 1) % self.간격 != 0

    def 어텐션(self, l, h, T):
        g, H, Hkv, hd = self.g, self.H, self.Hkv, self.hd
        qg = 선형(h, g, f"blk.{l}.attn_q.weight").reshape(T, H, 2, hd)
        q, 게이트 = qg[:, :, 0], qg[:, :, 1]
        k = 선형(h, g, f"blk.{l}.attn_k.weight").reshape(T, Hkv, hd)
        v = 선형(h, g, f"blk.{l}.attn_v.weight").reshape(T, Hkv, hd)
        q = rms(q, g.풀기(f"blk.{l}.attn_q_norm.weight"), self.eps)
        k = rms(k, g.풀기(f"blk.{l}.attn_k_norm.weight"), self.eps)
        n = self.rot // 2
        각 = np.arange(T, dtype=f64)[:, None] * (self.theta ** (-2.0 * np.arange(n, dtype=f64) / self.rot))[None, :]
        cos, sin = np.cos(각)[:, None, :], np.sin(각)[:, None, :]

        def 돌리기(x):
            y = x.copy()
            a, b = x[..., :n], x[..., n:2 * n]
            y[..., :n] = a * cos - b * sin
            y[..., n:2 * n] = b * cos + a * sin
            return y
        q, k = 돌리기(q), 돌리기(k)
        묶 = H // Hkv
        kk, vv = np.repeat(k, 묶, 1), np.repeat(v, 묶, 1)          # 질의 헤드 h → KV 헤드 h // 묶
        s = np.einsum("thd,shd->hts", q, kk) / math.sqrt(hd)
        s = s + np.triu(np.full((T, T), -np.inf), 1)[None]
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", s, vv) * sigmoid(게이트)
        return 선형(o.reshape(T, H * hd), g, f"blk.{l}.attn_output.weight")

    def 선형어텐션(self, l, h, T):
        g, d, Hk, Hv = self.g, self.d, self.Hk, self.Hv
        qkv = 선형(h, g, f"blk.{l}.attn_qkv.weight")                # (T, 2·Hk·d + Hv·d)
        z = 선형(h, g, f"blk.{l}.attn_gate.weight")                 # (T, Hv·d)
        b = 선형(h, g, f"blk.{l}.ssm_beta.weight")                  # (T, Hv)
        a = 선형(h, g, f"blk.{l}.ssm_alpha.weight")                 # (T, Hv)
        w = g.풀기(f"blk.{l}.ssm_conv1d.weight").astype(f64)          # (채널, 커널) — 커널 칸 0 이 가장 오래된 토큰
        K = self.합성곱
        xp = np.concatenate([np.zeros((K - 1, qkv.shape[1])), qkv], 0)
        y = sum(xp[j:j + T] * w[:, j][None, :] for j in range(K))
        y = silu(y)
        kd = Hk * d
        q = y[:, :kd].reshape(T, Hk, d)
        k = y[:, kd:2 * kd].reshape(T, Hk, d)
        v = y[:, 2 * kd:].reshape(T, Hv, d)
        q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), self.eps)
        k = k / np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), self.eps)
        q = q / math.sqrt(d)
        짝 = np.arange(Hv) % Hk                                      # v 헤드 j → k 헤드 j mod H_k
        q, k = q[:, 짝], k[:, 짝]
        β = sigmoid(b)
        gg = softplus(a + g.풀기(f"blk.{l}.ssm_dt.bias").astype(f64)) * g.풀기(f"blk.{l}.ssm_a").astype(f64)
        S = np.zeros((Hv, d, d))                                     # S[헤드][k 칸][v 칸]
        o = np.empty((T, Hv, d))
        for t in range(T):
            S *= np.exp(gg[t])[:, None, None]
            kv = np.einsum("hij,hi->hj", S, k[t])
            δ = (v[t] - kv) * β[t][:, None]
            S += k[t][:, :, None] * δ[:, None, :]
            o[t] = np.einsum("hij,hi->hj", S, q[t])
        o = rms(o, g.풀기(f"blk.{l}.ssm_norm.weight"), self.eps) * silu(z.reshape(T, Hv, d))
        return 선형(o.reshape(T, Hv * d), g, f"blk.{l}.ssm_out.weight")

    def MoE(self, l, h):
        """0.12.0 12-3g — 라우터 softmax 의 상위 k(같으면 작은 번호 — 계약 4.11 과 같은 규칙) · 합으로 나눈 무게 · 전문가 SwiGLU · 공유 전문가 × sigmoid."""
        g, T = self.g, h.shape[0]
        로짓 = 선형(h, g, f"blk.{l}.ffn_gate_inp.weight")              # (T, 전문가)
        p = np.exp(로짓 - 로짓.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        고른 = np.argsort(-로짓, axis=-1, kind="stable")[:, :self.고름수]
        self.라우팅.append(고른.astype(np.int16))
        w = np.take_along_axis(p, 고른, -1)
        w = w / np.maximum(w.sum(-1, keepdims=True), 2.0 ** -14)
        o = np.zeros((T, self.D))
        for e in np.unique(고른):
            행, 순위 = np.nonzero(고른 == e)
            he = h[행]
            gg = he @ 풀어전문가(g, f"blk.{l}.ffn_gate_exps.weight", e).astype(f64).T
            uu = he @ 풀어전문가(g, f"blk.{l}.ffn_up_exps.weight", e).astype(f64).T
            y = (silu(gg) * uu) @ 풀어전문가(g, f"blk.{l}.ffn_down_exps.weight", e).astype(f64).T
            o[행] += w[행, 순위][:, None] * y
        ys = 선형(silu(선형(h, g, f"blk.{l}.ffn_gate_shexp.weight")) * 선형(h, g, f"blk.{l}.ffn_up_shexp.weight"), g, f"blk.{l}.ffn_down_shexp.weight")
        gs = sigmoid(h @ 풀어(g, f"blk.{l}.ffn_gate_inp_shexp.weight").astype(f64))
        return o + gs[:, None] * ys

    def 앞으로(self, ids):
        g, T = self.g, len(ids)
        x = 풀어(g, "token_embd.weight", ids).astype(f64)
        for l in range(self.L):
            h = rms(x, g.풀기(f"blk.{l}.attn_norm.weight"), self.eps)
            x = x + (self.선형어텐션(l, h, T) if self.선형층인가(l) else self.어텐션(l, h, T))
            h = rms(x, g.풀기(f"blk.{l}.post_attention_norm.weight"), self.eps)
            if self.moe:
                x = x + self.MoE(l, h)
            else:
                x = x + 선형(silu(선형(h, g, f"blk.{l}.ffn_gate.weight")) * 선형(h, g, f"blk.{l}.ffn_up.weight"), g, f"blk.{l}.ffn_down.weight")
        x = rms(x, g.풀기("output_norm.weight"), self.eps)
        이름 = "token_embd.weight" if self.묶인출력 else "output.weight"
        V = g.tensors[이름][1][1]
        로짓 = np.empty((T, V))
        for s in range(0, V, 16384):
            로짓[:, s:s + 16384] = x @ 풀어(g, 이름, np.arange(s, min(V, s + 16384))).astype(f64).T
        return 로짓


def 로그소프트맥스(x):
    x = np.asarray(x, f64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    gguf, 판, 작업 = sys.argv[1:4]
    길이 = int(sys.argv[sys.argv.index("--length") + 1]) if "--length" in sys.argv else 256
    글들 = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--text"] or [os.path.join(ROOT, "연구자안내.md"),
                                                                                   os.path.join(ROOT, "README.md")]
    os.makedirs(작업, exist_ok=True)
    import 품질 as Q                                                 # noqa: E402 — 텍스트 고르기 · llama.cpp 의 llama_tokenize
    어휘 = Q.라마어휘(판, gguf)
    토큰들 = [어휘.토큰(Q._글바이트(지정)[1], 길이) for 지정 in 글들]
    json.dump({"gguf": os.path.basename(gguf), "글": 글들, "토큰": 토큰들}, open(os.path.join(작업, "토큰.json"), "w", encoding="utf-8"))
    if "--no-llama" not in sys.argv:
        os.environ["GEULPU_LLAMA"] = 판
        sys.path.insert(0, os.path.join(ROOT, "재기"))
        import 라마 as LM                                           # noqa: E402
        lm = LM.라마(gguf, n_ctx=길이, n_batch=길이, n_ubatch=min(512, 길이), n_seq_max=1, flash=-1, kv형식=1)
        for i, ids in enumerate(토큰들):
            lm.지우기()
            lm.넣기(ids, list(range(len(ids))), [0] * len(ids), [1] * len(ids))
            np.save(os.path.join(작업, f"라마{i}.npy"), lm.모든로짓(len(ids)))
        lm.닫기()
    m = 모델(gguf)
    print(f"{os.path.basename(gguf)}: 층 {m.L}(전체 어텐션 {sum(not m.선형층인가(l) for l in range(m.L))}) · 너비 {m.D} · 헤드 {m.H}/{m.Hkv} × {m.hd} · "
          f"RoPE {m.rot} 차원 θ {m.theta:g} · 선형 k 헤드 {m.Hk} · v 헤드 {m.Hv} × {m.d} · 임베딩과 공유한 출력 {m.묶인출력}", flush=True)
    kls, 같음, 행수 = [], 0, 0
    for i, ids in enumerate(토큰들):
        t0 = time.time()
        m.라우팅 = []
        p = 로그소프트맥스(m.앞으로(ids))
        np.save(os.path.join(작업, f"수학{i}.npy"), p.astype(np.float32))
        if m.moe:
            np.save(os.path.join(작업, f"라우팅{i}.npy"), np.stack(m.라우팅))      # [MoE 층][행][k]
        if "--no-llama" in sys.argv:
            print(f"텍스트 {i}: {len(ids)} 토큰 · {time.time() - t0:.0f} 초 — 1위(앞 8 행) {p[:8].argmax(1).tolist()}", flush=True)
            continue
        r = 로그소프트맥스(np.load(os.path.join(작업, f"라마{i}.npy")))
        kl = (np.exp(p) * (p - r)).sum(-1)
        같 = int((p.argmax(1) == r.argmax(1)).sum())
        kls.append(kl)
        같음 += 같
        행수 += len(ids)
        print(f"텍스트 {i}: {len(ids)} 토큰 · {time.time() - t0:.0f} 초 — KL 평균 {kl.mean():.3e} · 최대 {kl.max():.3e} · 1위 일치 {같}/{len(ids)}",
              flush=True)
    if kls:
        평균 = float(np.concatenate(kls).mean())
        일치 = 같음 / 행수
        print(f"모두: KL 평균 {평균:.3e} · 1위 일치 {100 * 일치:.2f} %")
        print("수학 참조: 맞음" if 평균 < 0.01 and 일치 >= 0.95 else "수학 참조: 다름")
        sys.exit(0 if 평균 < 0.01 and 일치 >= 0.95 else 1)


if __name__ == "__main__":
    main()
