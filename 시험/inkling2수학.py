#!/usr/bin/env python3
"""0.25.0 25-1 — Inkling 2(GGUF 구조 inkling2_loop · inkling2)의 수학 참조: 파일의 가중치를 풀어(양자화 값 그대로) 활성값을 fp64 로 계산한
순전파. gemma4수학.py 와 같은 쓰임 — GGUF 의 뜻을 llama.cpp 와 맞춰 보고, 계약의 CPU 참조(시험/참조inkling2.py)의 손실을 잰다(문서/0.25.0.md).

  python -I -X utf8 시험/inkling2수학.py <gguf> <llama.cpp 폴더> <작업 폴더> --random 48 [--seed 1] [--llama-bias]
  python -I -X utf8 시험/inkling2수학.py <gguf> <llama.cpp 폴더> <작업 폴더> [--text <경로[@바이트]> …] [--length 256] [--no-llama]
  python -I -X utf8 시험/inkling2수학.py <gguf> <llama.cpp 폴더> <작업 폴더> --chat [--n 100] [--workers 3] [--llama-only | --saved-llama]

--chat: 채팅 확인(gemma4수학.py 의 --채팅 과 같은 판정) — 채팅 프롬프트 셋(GGUF 의 채팅 템플릿대로, 생각의 정도 0.9)에 llama.cpp 가 그리디로
이어 쓴 답 N 토큰을 붙여, 답을 예측하는 행에서 수학 참조의 1위가 그 그리디와 같은지 · KL. 끝 줄 '채팅 확인: 맞음' 은 1위 일치 ≥ 95 % ·
평균 KL < 0.01. --workers 는 경우마다 프로세스 하나(같은 파일을 메모리 맵으로 함께 읽는다).

--random N: 무작위 토큰 N 개(합성 파일 — 시험/합성inkling2.py)로 llama.cpp(플래시 어텐션 끔 · KV f32)의 모든 행 로짓과 비교한다.
끝 줄 '수학 참조: 맞음' 은 평균 KL(참조 ‖ llama.cpp) < 0.01 · 1위 일치 ≥ 95 % (gemma4수학.py 와 같은 판정).

식 — transformers 의 models/inkling2(_loop)(modeling · modular)와 llama.cpp 의 src/models/inkling2.cpp · inkling2-loop.cpp(master af5c583,
2026-11-24)를 읽은 것. 둘이 다른 곳은 하나 — 상대 위치 편향과 싱크의 배율(아래 '편향'):
  입력 x = RMSNorm(token_embd[토큰]; token_embd_norm)
  층 l(물리 층 번호 p — 반복의 패스마다 같은 가중치):
    블록 어텐션 잔차 섞기(B, x; Q) = Σ_s softmax_s(RMSNorm0(S_s) · Q / √D) · RMSNorm0(S_s), S = [블록 상태들 …, x], RMSNorm0 은 가중치 없는 것
      (eps 10⁻⁶). 층의 입력: 층 0 은 x, 그 밖은 섞기(attn_res_query — 슬라이딩 윈도 층은 질의 하나, 전체 어텐션 층은 q · k · v 셋)
    반복의 두 번째 패스부터: 입력 j(q · k · v · FFN)마다 제어(c, m) = c − tanh(f) · c + tanh(g) · m, [f | g] 는 a = ctrl_down · [RMSNorm0(c),
      RMSNorm0(m)] 의 ctrl_forget · ctrl_gain 투영, m = 섞기(앞 패스의 블록 상태 가운데 그 패스가 처음 더한 것 뒤의 것들, 앞 패스의 출력;
      loop_msg_query 의 j 번째 질의)
    정규화: 슬라이딩 윈도 층은 attn_norm(입력이 셋으로 갈리면 셋 모두), 전체 층은 attn_q_inp_norm · attn_k_inp_norm · attn_v_inp_norm
    q = attn_q · h_q, k = 합성곱(attn_k · h_k), v = 합성곱(attn_v · h_v), r = attn_rel · h_q(헤드마다 relative_length 차원), t = attn_sink_proj · h_q
      합성곱(y; w, 탭) = (y + Σ_j w_j · y_(t−3+j) + Σ_lag a_lag · y_(t−lag)) / √(탭 수 · (동적이면 2) + 1) — 정적 탭 w(칸 0 이 가장 오래된
      토큰)와, 전체 층은 동적 탭 a = attn_{k,v}_conv_proj · h(KV 헤드마다 탭 4 — lag 0 이 지금 토큰). 시퀀스의 처음은 0
    q = 클립(q ⊙ attn_q_clip; qk_clip), k = 클립(k; qk_clip), v = RMSNorm0(헤드마다 v; eps), r = 클립(r ⊙ attn_rel_clip; relative_clip),
      클립(x; c) = x · min(1, c / (‖x‖ + 10⁻⁶)). 전체 층은 q · r 에 τ = 1 + log_scale_alpha · ln(max(1, (위치 + 1) / log_scale_floor))
    점수 = q · k / √헤드 크기 + 편향, 편향 = r · attn_rel_bias[거리](0 ≤ 거리 < 범위 — 윈도 층의 범위는 sliding_window, 전체 층은
      relative_extent — 그 밖은 0). 인과 가림, 윈도 층은 거리 ≥ sliding_window 도 가린다. softmax 의 분모에 싱크 e^t 를 더한다(값은 0)
    o = attn_output · (어텐션 ⊙ σ(attn_gate · h_q)) → 합성곱(attn_post_conv — 정적)
    잔차: p 가 attn_res.block_size 의 배수면 층 입력 x 를 블록 상태에 더하고 x ← o(잔차 스트림을 새로 시작), 아니면 x ← x + o
    FFN: h = RMSNorm(섞기(ffn_res_query)[, 제어]; ffn_norm), 라우터 σ(ffn_gate_inp · h) 에 exp_probs_b 를 더한 점수로 상위 expert_used_count,
      무게 = 고른 σ / (합 + 10⁻²⁰) × expert_weights_scale, 전문가 [g | u] = ffn_gate_up_exps · h(앞 절반이 gate),
      SITU = β tanh(g/β) σ(g) · λ tanh(u/λ)(β = activation.situ_beta, λ = activation.situ_linear_beta), y = ffn_down_exps · SITU,
      x ← x + 합성곱(Σ 무게 · y; ffn_post_conv)
  반복(inkling2_loop): 층 [0, loop.start) → 층 [loop.start, loop.end) 를 loop.count 번(두 번째 패스부터 앞 6 층의 출력 · 블록 상태에서 다시
    시작하고 위의 제어) → 층 [loop.end, 층 수)
  끝: 섞기(output_res_query) → RMSNorm(output_norm) → output → 소프트캡(final_logit_softcapping) → 앞 vocab_size_unpadded 개
편향: transformers(eager_attention_forward — 점수 × 배율 뒤에 편향을 더하고 싱크를 로짓 하나로 붙인다)와 vLLM 의 커널("(qk + bias/scale) ·
  scale")은 편향 · 싱크에 1/√헤드 크기를 곱하지 않는다. llama.cpp 는 편향 − 싱크를 kq 에 더한 뒤 ggml_soft_max_ext 의 배율을 곱한다 —
  편향과 싱크가 1/√헤드 크기 배가 된다. 이 참조는 transformers 를 따르고, --llama-bias 는 llama.cpp 의 식으로 계산한다(차이의 확인용).
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
import qwen35수학 as Q      # noqa: E402 — 풀어 · 풀어전문가 · 선형 · rms · 로그소프트맥스

f64 = np.float64
섞기eps = 1e-6                # llama.cpp 의 mix() · control() 의 ggml_rms_norm(…, 1e-6f) — transformers 는 rms_norm_eps(Inkling-2-Mini 1e-6)


def rms0(x, eps):
    """가중치 없는 RMSNorm(마지막 축)."""
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps)


def 클립(x, c):
    """L2 노름 클리핑(마지막 축): x · min(1, c / (‖x‖ + 10⁻⁶))."""
    n = np.sqrt((x * x).sum(-1, keepdims=True))
    return x * np.minimum(1.0, c / (n + 1e-6))


def 시그(x):
    return 1.0 / (1.0 + np.exp(-x))


class 모델:
    def __init__(self, path, llama편향=False):
        self.g = g = GR.GGUF(path)
        m = g.meta
        구조 = self.구조 = m["general.architecture"]
        assert 구조 in ("inkling2_loop", "inkling2"), 구조
        k = lambda x, d=None: m.get(구조 + "." + x, d)               # noqa: E731
        self.L, self.D = k("block_count"), k("embedding_length")
        층마다 = lambda v: list(v) if isinstance(v, (list, tuple)) else [v] * self.L    # noqa: E731 — 실제 파일은 머리 수를 층마다의 배열로 적는다
        self.Hs, self.Hkv = 층마다(k("attention.head_count")), 층마다(k("attention.head_count_kv"))
        self.hd = {False: k("attention.key_length"), True: k("attention.key_length_swa", k("attention.key_length"))}
        self.eps = float(k("attention.layer_norm_rms_epsilon"))
        self.swa = [bool(x) for x in k("attention.sliding_window_pattern")]
        assert len(self.swa) == self.L
        self.창, self.R, self.범위 = k("attention.sliding_window"), k("attention.relative_length"), k("attention.relative_extent")
        self.qk클립, self.r클립 = float(k("attention.qk_clip")), float(k("attention.relative_clip"))
        self.바닥, self.알파 = float(k("attention.log_scale_floor")), float(k("attention.log_scale_alpha"))
        self.탭, self.블록 = k("shortconv.l_cache"), k("attn_res.block_size")
        self.베타, self.선베타 = float(k("activation.situ_beta")), float(k("activation.situ_linear_beta"))
        self.배율 = float(k("expert_weights_scale", 0.0) or 0.0)
        self.캡 = float(k("final_logit_softcapping", 0.0) or 0.0)
        self.Vu = k("vocab_size_unpadded")
        self.고름, self.F = k("expert_used_count"), k("expert_feed_forward_length")
        if 구조 == "inkling2_loop":
            self.시작, self.끝, self.패스 = k("loop.start"), k("loop.end"), k("loop.count")
            assert not k("loop.varying", False), "loop.varying 은 받지 않는다"
        else:
            self.시작 = self.끝 = self.L
            self.패스 = 1
        self.llama편향 = llama편향
        self.라우팅 = []                                            # 층 실행마다 고른 전문가 [행][k]

    def 풀(self, 이름):
        return self.g.풀기(이름).astype(f64)

    def 섞기(self, 상태들, x, 질의):
        """블록 어텐션 잔차 — 상태들(블록 상태 [T][D] 의 목록)과 x 를 질의 [nq][D] 로 섞은 [T][nq][D]."""
        S = rms0(np.stack(list(상태들) + [x], 1), 섞기eps)          # [T][상태][D]
        sc = np.einsum("tsd,qd->tqs", S, np.atleast_2d(질의)) / math.sqrt(self.D)
        sc = np.exp(sc - sc.max(-1, keepdims=True))
        sc /= sc.sum(-1, keepdims=True)
        return np.einsum("tqs,tsd->tqd", sc, S)

    def 제어(self, c, m, l, 이름):
        """반복의 두 번째 패스부터 — c − tanh(f) · c + tanh(g) · m."""
        b = f"blk.{l}.loop_{이름}_ctrl_"
        a = np.concatenate([rms0(c, 섞기eps), rms0(m, 섞기eps)], -1) @ self.풀(b + "down.weight").T
        g = np.tanh(a @ self.풀(b + "gain.weight").T)
        f = np.tanh(a @ self.풀(b + "forget.weight").T)
        return c - f * c + g * m

    def 합성곱(self, y, w, a, Hk):
        """짧은 인과 합성곱 — y [T][C], w [C][탭](칸 0 이 가장 오래된 토큰), a [T][Hk · 탭](동적 탭 — lag 0 이 지금 토큰) 또는 None."""
        T, C = y.shape
        n = self.탭
        yp = np.concatenate([np.zeros((n - 1, C)), y], 0)
        s = np.zeros_like(y)
        for j in range(n):
            s += w[None, :, j] * yp[j:j + T]
        if a is not None:
            a = a.reshape(T, Hk, n)
            for lag in range(n):
                s += np.repeat(a[:, :, lag], C // Hk, axis=1) * yp[n - 1 - lag:n - 1 - lag + T]
        return (y + s) / math.sqrt(n * (2 if a is not None else 1) + 1)

    def 층(self, l, x, 블록들, τ, 메시지=None):
        g, T, H = self.g, x.shape[0], self.Hs[l]
        s = self.swa[l]
        Hk, hd = self.Hkv[l], self.hd[s]
        b = f"blk.{l}."
        입력 = x
        if b + "attn_res_query.weight" in g.tensors:
            섞임 = self.섞기(블록들, x, self.풀(b + "attn_res_query.weight"))
            h = [섞임[:, 0 if s else j] for j in range(3)]
        else:
            h = [x, x, x]
        if 메시지 is not None:
            h = [self.제어(h[j], 메시지[:, j], l, c) for j, c in enumerate(("q", "k", "v"))]
        if s:
            w = self.풀(b + "attn_norm.weight")
            h = [Q.rms(v, w, self.eps) for v in h] if 메시지 is not None else [Q.rms(h[0], w, self.eps)] * 3
        else:
            h = [Q.rms(h[j], self.풀(b + f"attn_{c}_inp_norm.weight"), self.eps) for j, c in enumerate(("q", "k", "v"))]
        q = Q.선형(h[0], g, b + "attn_q.weight").reshape(T, H, hd)
        kk = self.합성곱(Q.선형(h[1], g, b + "attn_k.weight"), self.풀(b + "attn_k_conv.weight"),
                       None if s else Q.선형(h[1], g, b + "attn_k_conv_proj.weight"), Hk).reshape(T, Hk, hd)
        vv = self.합성곱(Q.선형(h[2], g, b + "attn_v.weight"), self.풀(b + "attn_v_conv.weight"),
                       None if s else Q.선형(h[2], g, b + "attn_v_conv_proj.weight"), Hk).reshape(T, Hk, hd)
        r = Q.선형(h[0], g, b + "attn_rel.weight").reshape(T, H, self.R)
        q = 클립(q * self.풀(b + "attn_q_clip.weight")[None], self.qk클립)
        kk = 클립(kk, self.qk클립)
        vv = rms0(vv, self.eps)
        r = 클립(r * self.풀(b + "attn_rel_clip.weight")[None], self.r클립)
        if not s:
            q = q * τ[:, None, None]
            r = r * τ[:, None, None]
        t = Q.선형(h[0], g, b + "attn_sink_proj.weight").T                # [H][T]
        표 = self.풀(b + "attn_rel_bias.weight")                         # [범위][R]
        범위 = 표.shape[0]
        c = 1.0 / math.sqrt(hd)
        묶 = H // Hk
        kr, vr = np.repeat(kk, 묶, 1), np.repeat(vv, 묶, 1)
        sc = np.einsum("thd,shd->hts", q, kr) * c
        i = np.arange(T)
        거리 = i[:, None] - i[None, :]
        편향 = np.einsum("thr,dr->htd", r, 표)                            # [H][T][범위]
        안 = (거리 >= 0) & (거리 < 범위)
        bias = np.where(안[None], np.take_along_axis(편향, np.clip(거리, 0, 범위 - 1)[None].repeat(H, 0), 2), 0.0)
        if self.llama편향:                                               # llama.cpp — 편향과 싱크도 배율 c 를 곱한다
            sc, t = sc + c * bias, c * t
        else:
            sc = sc + bias
        가림 = 거리 < 0
        if s:
            가림 = 가림 | (거리 >= self.창)
        sc = np.where(가림[None], -np.inf, sc)
        mx = np.maximum(sc.max(-1), t)                                    # [H][T]
        e = np.exp(sc - mx[..., None])
        p = e / (e.sum(-1) + np.exp(t - mx))[..., None]
        o = np.einsum("hts,shd->thd", p, vr).reshape(T, H * hd)
        o = o * 시그(Q.선형(h[0], g, b + "attn_gate.weight"))
        o = self.합성곱(Q.선형(o, g, b + "attn_output.weight"), self.풀(b + "attn_post_conv.weight"), None, Hk)
        if l % self.블록 == 0:
            블록들.append(입력)
            x = o
        else:
            x = 입력 + o
        ffn입력 = x
        hf = self.섞기(블록들, x, self.풀(b + "ffn_res_query.weight"))[:, 0]
        if 메시지 is not None:
            hf = self.제어(hf, 메시지[:, 3], l, "ffn")
        hf = Q.rms(hf, self.풀(b + "ffn_norm.weight"), self.eps)
        o = self.합성곱(self.MoE(l, hf), self.풀(b + "ffn_post_conv.weight"), None, Hk)
        return ffn입력 + o

    def MoE(self, l, h):
        g, b, F = self.g, f"blk.{l}.", self.F
        p = 시그(h @ self.풀(b + "ffn_gate_inp.weight").T)               # [T][전문가]
        고름점수 = p + self.풀(b + "exp_probs_b.bias")[None] if b + "exp_probs_b.bias" in g.tensors else p
        고른 = np.argsort(-고름점수, axis=-1, kind="stable")[:, :self.고름]
        self.라우팅.append(고른.astype(np.int16))
        w = np.take_along_axis(p, 고른, -1)
        w = w / (w.sum(-1, keepdims=True) + 1e-20)
        if self.배율 not in (0.0, 1.0):
            w = w * self.배율
        o = np.zeros_like(h)
        for e in np.unique(고른):
            행, 순위 = np.nonzero(고른 == e)
            gu = h[행] @ Q.풀어전문가(g, b + "ffn_gate_up_exps.weight", e).astype(f64).T
            gg, uu = gu[:, :F], gu[:, F:]
            act = self.베타 * np.tanh(gg / self.베타) * 시그(gg)
            if self.선베타 > 0:
                uu = self.선베타 * np.tanh(uu / self.선베타)
            y = (act * uu) @ Q.풀어전문가(g, b + "ffn_down_exps.weight", e).astype(f64).T
            o[행] += w[행, 순위][:, None] * y
        return o

    def 앞으로(self, ids):
        g, T = self.g, len(ids)
        x = Q.rms(Q.풀어(g, "token_embd.weight", ids).astype(f64), self.풀("token_embd_norm.weight"), self.eps)
        τ = 1.0 + self.알파 * np.log(np.maximum(1.0, (np.arange(T) + 1.0) / self.바닥))
        블록들 = []
        for l in range(self.시작):
            x = self.층(l, x, 블록들, τ)
        앞x, 앞블록 = x, list(블록들)
        for 패스 in range(self.패스 if self.시작 < self.끝 else 0):
            if 패스 > 0:
                은행, 은행출력 = 블록들[len(앞블록) + 1:], x                  # 앞 패스가 처음 더한 블록 상태 뒤의 것들 · 앞 패스의 출력
                x, 블록들 = 앞x, list(앞블록)
            for l in range(self.시작, self.끝):
                메시지 = None if 패스 == 0 else self.섞기(은행, 은행출력, self.풀(f"blk.{l}.loop_msg_query.weight"))
                x = self.층(l, x, 블록들, τ, 메시지)
        for l in range(self.끝, self.L):
            x = self.층(l, x, 블록들, τ)
        x = self.섞기(블록들, x, self.풀("output_res_query.weight"))[:, 0]
        x = Q.rms(x, self.풀("output_norm.weight"), self.eps)
        V = g.tensors["output.weight"][1][1]
        로짓 = np.empty((T, V))
        for s in range(0, V, 16384):
            로짓[:, s:s + 16384] = x @ Q.풀어(g, "output.weight", np.arange(s, min(V, s + 16384))).astype(f64).T
        if self.캡:
            로짓 = self.캡 * np.tanh(로짓 / self.캡)
        return 로짓[:, :self.Vu]


def 라마부르기(판):
    os.environ["GEULPU_LLAMA"] = 판
    sys.path.insert(0, os.path.join(ROOT, "재기"))
    import 라마 as LM                                               # noqa: E402
    return LM


def 라마로짓(lm, ids):
    lm.지우기()
    lm.넣기(ids, list(range(len(ids))), [0] * len(ids), [1] * len(ids))
    return lm.모든로짓(len(ids))


def 비교(p, z):
    """(KL(p ‖ z) 행마다, 1위 일치 수) — p · z 는 로그확률 · 로짓."""
    r = Q.로그소프트맥스(z)
    return (np.exp(p) * (p - r)).sum(-1), int((p.argmax(1) == r.argmax(1)).sum())


채팅질문 = ["What is the capital of France? Answer in one sentence, then list three famous landmarks there.",
         "대한민국의 수도는 어디야? 한 문장으로 답하고, 그 도시의 유명한 곳 세 군데를 적어 줘.",
         "Write a Python function that returns the n-th Fibonacci number iteratively."]


def 채팅글(질문):
    """GGUF 의 채팅 템플릿이 사용자 메시지 하나에 내는 글 — 생각의 정도(기본 0.9)를 적은 시스템 메시지 + 사용자 메시지. 템플릿은 생성
    프롬프트를 붙이지 않는다(모델이 <|message_model|> 부터 낸다). BOS 는 없다(pre inkling2)."""
    return ("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
            f"<|message_user|><|content_text|>{질문}<|end_message|>")


def 채팅라마(gguf, 판, N):
    """llama.cpp(플래시 어텐션 끔 · KV f32)가 채팅 프롬프트 셋을 그리디로 N 토큰 이어 쓴 것 — [(토큰열, 프롬프트 길이, 모든 행의 로짓)]."""
    import ctypes
    LM = 라마부르기(판)
    lm = LM.라마(gguf, n_ctx=512, n_batch=512, n_ubatch=512, n_seq_max=1, flash=0, kv형식=0)
    L = lm.L
    L.llama_model_get_vocab.restype = ctypes.c_void_p
    L.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    L.llama_tokenize.restype = ctypes.c_int32
    L.llama_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
                                 ctypes.c_bool, ctypes.c_bool]
    어휘 = L.llama_model_get_vocab(lm.model)
    경우 = []
    for 질문 in 채팅질문:
        글 = 채팅글(질문).encode("utf-8")
        buf = (ctypes.c_int32 * 1024)()
        ids = [int(x) for x in buf[:L.llama_tokenize(어휘, 글, len(글), buf, 1024, True, True)]]       # 특수 토큰을 푼다
        P = len(ids)
        lm.지우기()
        lm.넣기(ids, list(range(P)), [0] * P, [0] * (P - 1) + [1])
        for _ in range(N):
            t = int(np.argmax(lm.로짓(-1)))                              # 그리디
            ids.append(t)
            lm.넣기([t], [len(ids) - 1], [0], [1])
        경우.append((ids, P, 라마로짓(lm, ids)))
        print(f"프롬프트 {P} + 답 {N} 토큰 — 앞 {ids[:6]} … 답의 처음 {ids[P:P + 8]}", flush=True)
    lm.닫기()
    return 경우


def _수학일(인자들):
    """일꾼 프로세스 하나 — (gguf, 토큰열, llama.cpp 식의 편향인가) → (로그확률 fp32, 라우팅)."""
    gguf, ids, l편향 = 인자들
    m = 모델(gguf, llama편향=l편향)
    p = Q.로그소프트맥스(m.앞으로(ids))
    return p.astype(np.float32), np.stack(m.라우팅)


def 채팅(gguf, 판, 작업, N, 일꾼):
    토큰파일 = os.path.join(작업, "채팅토큰.json")
    if "--saved-llama" in sys.argv:                                     # 앞의 --llama-only 가 남긴 토큰 · 로짓으로
        경우 = [(ids, P, np.load(os.path.join(작업, f"채팅라마{i}.npy")))
              for i, (ids, P) in enumerate(json.load(open(토큰파일, encoding="utf-8"))["경우"])]
    else:
        경우 = 채팅라마(gguf, 판, N)
        json.dump({"경우": [[ids, P] for ids, P, _ in 경우]}, open(토큰파일, "w", encoding="utf-8"))
        for i, (_, _, z) in enumerate(경우):
            np.save(os.path.join(작업, f"채팅라마{i}.npy"), z)
        if "--llama-only" in sys.argv:
            return
    t0 = time.time()
    일들 = [(gguf, ids, "--llama-bias" in sys.argv) for ids, _, _ in 경우]
    if 일꾼 > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(일꾼) as ex:
            결과 = list(ex.map(_수학일, 일들))
    else:
        결과 = [_수학일(a) for a in 일들]
    kls, 맞음, 행수 = [], 0, 0
    for i, ((ids, P, z), (p, 라우팅)) in enumerate(zip(경우, 결과)):
        np.save(os.path.join(작업, f"채팅수학{i}.npy"), p)
        np.save(os.path.join(작업, f"채팅라우팅{i}.npy"), 라우팅)               # [층 실행][행][k]
        p = p.astype(f64)
        r = Q.로그소프트맥스(z)
        kl = (np.exp(p) * (p - r)).sum(-1)
        답 = slice(P - 1, len(ids) - 1)                                  # 행 P − 1 … 끝 − 2 가 답 토큰을 예측
        같 = int((p[답].argmax(1) == np.array(ids[P:])).sum())
        kls.append(kl[답])
        맞음 += 같
        행수 += len(ids) - P
        print(f"경우 {i}: 답 구간 {len(ids) - P} 행: 수학 참조의 1위 = llama.cpp 의 그리디 {같}/{len(ids) - P} · KL 평균 {kl[답].mean():.3e} · "
              f"최대 {kl[답].max():.3e} | 프롬프트 구간 KL 평균 {kl[:P - 1].mean():.3e}", flush=True)
    평균 = float(np.concatenate(kls).mean())
    print(f"모두({time.time() - t0:.0f} 초): 답 {행수} 행 — 1위 = 그리디 {100 * 맞음 / 행수:.2f} % · KL 평균 {평균:.3e}")
    좋음 = 맞음 / 행수 >= 0.95 and 평균 < 0.01
    print("채팅 확인: 맞음" if 좋음 else "채팅 확인: 다름")
    sys.exit(0 if 좋음 else 1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    gguf, 판, 작업 = sys.argv[1:4]
    os.makedirs(작업, exist_ok=True)
    인자 = lambda 이름, d: sys.argv[sys.argv.index(이름) + 1] if 이름 in sys.argv else d    # noqa: E731
    if "--chat" in sys.argv:
        return 채팅(gguf, 판, 작업, int(인자("--n", 100)), int(인자("--workers", 1)))
    if "--random" in sys.argv:
        n, 씨앗 = int(인자("--random", 48)), int(인자("--seed", 1))
        m0 = 모델(gguf)
        토큰들 = [np.random.default_rng(씨앗 + i).integers(0, m0.Vu, n).tolist() for i in range(2)]
    else:
        길이 = int(인자("--length", 256))
        글들 = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--text"] or [os.path.join(ROOT, "연구자안내.md"),
                                                                                       os.path.join(ROOT, "README.md")]
        import 품질 as QQ                                            # noqa: E402 — 텍스트 고르기 · llama.cpp 의 llama_tokenize
        어휘 = QQ.라마어휘(판, gguf)
        토큰들 = [어휘.토큰(QQ._글바이트(지정)[1], 길이) for 지정 in 글들]
    json.dump({"gguf": os.path.basename(gguf), "토큰": 토큰들}, open(os.path.join(작업, "토큰.json"), "w", encoding="utf-8"))
    if "--no-llama" not in sys.argv:
        LM = 라마부르기(판)
        n = max(len(t) for t in 토큰들)
        lm = LM.라마(gguf, n_ctx=max(64, n), n_batch=max(64, n), n_ubatch=max(64, n), n_seq_max=1, flash=0, kv형식=0)
        for i, ids in enumerate(토큰들):
            np.save(os.path.join(작업, f"라마{i}.npy"), 라마로짓(lm, ids))
        lm.닫기()
    결과 = {}
    for 방식 in ((False, True) if "--both" in sys.argv else (("--llama-bias" in sys.argv),)):
        m = 모델(gguf, llama편향=방식)
        이름 = "llama.cpp 의 편향 배율" if 방식 else "transformers 의 편향 배율"
        kls, 같음, 행수 = [], 0, 0
        for i, ids in enumerate(토큰들):
            t0 = time.time()
            p = Q.로그소프트맥스(m.앞으로(ids))
            np.save(os.path.join(작업, f"수학{'L' if 방식 else ''}{i}.npy"), p.astype(np.float32))
            if "--no-llama" in sys.argv:
                print(f"[{이름}] 경우 {i}: {len(ids)} 토큰 · {time.time() - t0:.1f} 초 — 1위(앞 8 행) {p[:8].argmax(1).tolist()}", flush=True)
                continue
            kl, 같 = 비교(p, np.load(os.path.join(작업, f"라마{i}.npy")))
            kls.append(kl)
            같음 += 같
            행수 += len(ids)
            print(f"[{이름}] 경우 {i}: {len(ids)} 토큰 · {time.time() - t0:.1f} 초 — KL 평균 {kl.mean():.3e} · 최대 {kl.max():.3e} · "
                  f"1위 일치 {같}/{len(ids)}", flush=True)
        if kls:
            평균 = float(np.concatenate(kls).mean())
            결과[이름] = (평균, 같음 / 행수)
            print(f"[{이름}] 모두: KL 평균 {평균:.3e} · 1위 일치 {100 * 같음 / 행수:.2f} %")
    for 이름, (평균, 일치) in 결과.items():
        print(f"수학 참조({이름}): {'맞음' if 평균 < 0.01 and 일치 >= 0.95 else '다름'}")


if __name__ == "__main__":
    main()
