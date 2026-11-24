#!/usr/bin/env python3
"""0.25.0 25-1 — 작은 합성 Inkling 2(GGUF 구조 inkling2_loop) 파일: 무작위 가중치(F32)로 llama.cpp 와 수학 참조 · 계약의 CPU 참조를 GPU 없이
맞춰 본다(문서/0.25.0.md). 값은 말이 되지 않는다 — 식을 읽은 방식(텐서의 배치 · 층의 차례 · 반복 · 블록 어텐션 잔차 · 합성곱의 탭 ·
상대 위치 편향 · 싱크 · MoE)이 같은지만 본다.

  python -I -X utf8 시험/합성inkling2.py <출력.gguf> [--seed 7]

모양(실제 Inkling-2-Mini 의 구성을 줄인 것 — llama.cpp 의 src/models/inkling2.cpp · inkling2-loop.cpp 가 받는 조건 안에서):
  너비 64 · 층 6(블록 2 층 — 0 1 | 2 3 | 4 5) · 반복 [2, 4) × 3 패스 · 질의 헤드 4 · KV 헤드 2 · 헤드 16 · 슬라이딩 윈도 층 0 2 4(윈도 8) ·
  전체 어텐션 층 1 3 5 · 상대 위치 편향 차원 4 · 범위 6(전체 층 — 윈도 층은 윈도 8) · 합성곱 4 · 전문가 8 중 2 · 전문가 FFN 32 · 어휘 300
  (행렬은 320 으로 패딩). 클리핑 문턱 · τ 의 바닥 · SITU 의 두 소프트캡 · 끝 소프트캡은 짧은 입력에서도 걸리도록 작게 둔다.
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import GGUF쓰기 as W         # noqa: E402 — 짓기(키들, 텐서들, 맞춤)

구조 = "inkling2_loop"
D, L, H, HKV, HD = 64, 6, 4, 2, 16
창, 범위, R, 탭 = 8, 6, 4, 4
E, K, F = 8, 2, 32
V, V패딩 = 300, 320
시작, 끝, 패스, 랭크, 블록 = 2, 4, 3, 3, 2
윈도 = [l % 2 == 0 for l in range(L)]                 # 층 0 2 4 슬라이딩 윈도, 1 3 5 전체 어텐션


def _바이트문자():
    """GPT-2 의 bytes_to_unicode — 바이트 b 마다 BPE 어휘의 글자."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def 키들():
    p = 구조 + "."
    u32 = lambda k, v: (p + k, 4, int(v))              # noqa: E731
    f = lambda k, v: (p + k, 6, float(v))              # noqa: E731
    바 = _바이트문자()
    토큰 = [바[b] for b in range(256)] + [f"zz{i}" for i in range(V - 256)]
    return [
        ("general.architecture", 8, 구조), ("general.name", 8, "inkling2-loop-synthetic"), ("general.alignment", 4, 32),
        u32("context_length", 256), u32("embedding_length", D), u32("block_count", L), u32("feed_forward_length", F),
        u32("expert_feed_forward_length", F), u32("expert_count", E), u32("expert_used_count", K), f("expert_weights_scale", 2.5),
        u32("attention.head_count", H), u32("attention.head_count_kv", HKV),
        u32("attention.key_length", HD), u32("attention.value_length", HD),
        u32("attention.key_length_swa", HD), u32("attention.value_length_swa", HD),
        f("attention.layer_norm_rms_epsilon", 1e-6),
        u32("attention.sliding_window", 창), (p + "attention.sliding_window_pattern", 9, (7, [1 if s else 0 for s in 윈도])),
        u32("attention.relative_length", R), u32("attention.relative_extent", 범위),
        f("attention.qk_clip", 1.5), f("attention.relative_clip", 0.7),
        u32("attention.log_scale_floor", 4), f("attention.log_scale_alpha", 0.3),
        u32("shortconv.l_cache", 탭), u32("attn_res.block_size", 블록),
        f("activation.situ_beta", 2.0), f("activation.situ_linear_beta", 3.0),
        f("final_logit_softcapping", 6.0),
        u32("vocab_size", V패딩), u32("vocab_size_unpadded", V), u32("rope.dimension_count", 0),
        u32("loop.start", 시작), u32("loop.end", 끝), u32("loop.count", 패스), u32("loop.gate_rank", 랭크), (p + "loop.varying", 7, 0),
        ("tokenizer.ggml.model", 8, "gpt2"), ("tokenizer.ggml.pre", 8, "inkling2"),
        ("tokenizer.ggml.tokens", 9, (8, 토큰)), ("tokenizer.ggml.token_type", 9, (5, [1] * V)),
        ("tokenizer.ggml.merges", 9, (8, [])), ("tokenizer.ggml.eos_token_id", 4, V - 1), ("tokenizer.ggml.add_bos_token", 7, 0),
    ]


def 텐서들(씨앗):
    rng = np.random.default_rng(씨앗)
    t = []

    def 넣기(이름, a):
        a = np.ascontiguousarray(a, np.float32)
        t.append((이름, list(reversed(a.shape)), 0, a.tobytes()))      # ggml 의 차원 차례 [ne0, ne1, …] = 배열 모양의 뒤집음

    def 행렬(n_out, n_in, 배=1.0):
        return rng.normal(0, 배 / np.sqrt(n_in), (n_out, n_in))

    def 정규(n):
        return 1.0 + 0.1 * rng.normal(0, 1, n)

    넣기("token_embd.weight", rng.normal(0, 1, (V패딩, D)))
    넣기("token_embd_norm.weight", 정규(D))
    for l in range(L):
        b = f"blk.{l}."
        n범위 = 창 if 윈도[l] else 범위
        넣기(b + "attn_q.weight", 행렬(H * HD, D, 2.0))
        넣기(b + "attn_k.weight", 행렬(HKV * HD, D, 2.0))
        넣기(b + "attn_v.weight", 행렬(HKV * HD, D))
        넣기(b + "attn_output.weight", 행렬(D, H * HD))
        넣기(b + "attn_gate.weight", 행렬(H * HD, D))
        넣기(b + "attn_q_clip.weight", 정규((H, HD)))
        넣기(b + "attn_sink_proj.weight", 행렬(H, D, 2.0))
        넣기(b + "attn_rel.weight", 행렬(H * R, D))
        넣기(b + "attn_rel_bias.weight", rng.normal(0, 1.5, (n범위, R)))       # ggml {R, 범위}
        넣기(b + "attn_rel_clip.weight", 정규((H, R)))
        넣기(b + "attn_k_conv.weight", rng.normal(0, 0.4, (HKV * HD, 탭)))     # ggml {탭, 채널} — 칸 0 이 가장 오래된 토큰
        넣기(b + "attn_v_conv.weight", rng.normal(0, 0.4, (HKV * HD, 탭)))
        넣기(b + "attn_post_conv.weight", rng.normal(0, 0.4, (D, 탭)))
        넣기(b + "ffn_post_conv.weight", rng.normal(0, 0.4, (D, 탭)))
        if 윈도[l]:
            넣기(b + "attn_norm.weight", 정규(D))
        else:
            넣기(b + "attn_q_inp_norm.weight", 정규(D))
            넣기(b + "attn_k_inp_norm.weight", 정규(D))
            넣기(b + "attn_v_inp_norm.weight", 정규(D))
            넣기(b + "attn_k_conv_proj.weight", 행렬(HKV * 탭, D, 0.5))
            넣기(b + "attn_v_conv_proj.weight", 행렬(HKV * 탭, D, 0.5))
        if l > 0:
            넣기(b + "attn_res_query.weight", rng.normal(0, 1, (1 if 윈도[l] else 3, D)))
        넣기(b + "ffn_res_query.weight", rng.normal(0, 1, D))
        넣기(b + "ffn_norm.weight", 정규(D))
        넣기(b + "ffn_gate_inp.weight", 행렬(E, D, 2.0))
        넣기(b + "exp_probs_b.bias", rng.normal(0, 0.1, E))
        넣기(b + "ffn_gate_up_exps.weight", rng.normal(0, 1.5 / np.sqrt(D), (E, 2 * F, D)))
        넣기(b + "ffn_down_exps.weight", rng.normal(0, 1 / np.sqrt(F), (E, D, F)))
        if 시작 <= l < 끝:
            넣기(b + "loop_msg_query.weight", rng.normal(0, 1, (4, D)))
            for c in ("q", "k", "v", "ffn"):
                넣기(b + f"loop_{c}_ctrl_down.weight", 행렬(랭크, 2 * D))
                넣기(b + f"loop_{c}_ctrl_gain.weight", 행렬(D, 랭크, 0.8))
                넣기(b + f"loop_{c}_ctrl_forget.weight", 행렬(D, 랭크, 0.8))
    넣기("output_res_query.weight", rng.normal(0, 1, D))
    넣기("output_norm.weight", 정규(D))
    넣기("output.weight", 행렬(V패딩, D, 3.0))
    return t


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    global E, K
    출력 = sys.argv[1]
    씨앗 = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 7
    if "--experts" in sys.argv:                                    # 전문가 수 · 고름 수(라우팅이 갈리지 않는 경우 — 전문가 2 중 2 — 의 비교용)
        E = int(sys.argv[sys.argv.index("--experts") + 1])
        K = int(sys.argv[sys.argv.index("--used") + 1]) if "--used" in sys.argv else min(K, E)
    b = W.짓기(키들(), 텐서들(씨앗), 32)
    with open(출력, "wb") as f:
        f.write(b)
    print(f"{출력}: {len(b):,} 바이트 — 구조 {구조}, 층 {L}, 반복 [{시작}, {끝}) × {패스}, 너비 {D}, 전문가 {E} 중 {K}, 씨앗 {씨앗}")


if __name__ == "__main__":
    main()
