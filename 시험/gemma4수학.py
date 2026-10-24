#!/usr/bin/env python3
"""0.12.0 12-3h-1 — Gemma 4(GGUF 구조 gemma4)의 수학 참조: 파일의 가중치를 풀어(양자화 값 그대로) 활성값을 fp64 로 계산한 순전파.
qwen35수학.py 와 같은 쓰임 — GGUF 의 뜻(텐서의 배치 · 층의 차례 · 두 어텐션 모양)을 llama.cpp 와 맞춰 본다(문서/블록조합.md 의 'Gemma 4 (12-3h)').
지금은 dense(PLE · KV 공유 · MoE 없음 — 12B · 31B)만.

  python -I 시험/gemma4수학.py <gemma4 gguf> <llama.cpp 폴더> <작업 폴더> [--text <경로[@바이트]> …] [--length 256] [--no-llama]
  python -I 시험/gemma4수학.py <gemma4 gguf> <llama.cpp 폴더> <작업 폴더> --채팅 [--n 100]

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
    x ← x · layer_output_scale[l]
  끝: RMSNorm(x; output_norm), 로짓 = token_embd · x(출력은 임베딩과 묶임 — output.weight 가 있으면 그것), 소프트캡 c · tanh(로짓 / c)
  (c = final_logit_softcapping — 없으면 소프트캡 없음)
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


class 모델:
    def __init__(self, path):
        self.g = g = GR.GGUF(path)
        m = g.meta
        assert m["general.architecture"] == "gemma4", m["general.architecture"]
        k = lambda x, d=None: m.get("gemma4." + x, d)               # noqa: E731
        assert not k("embedding_length_per_layer_input", 0), "PLE(E2B · E4B)는 아직"
        assert not k("attention.shared_kv_layers", 0), "KV 공유는 아직"
        assert "blk.0.ffn_gate_inp.weight" not in g.tensors, "MoE(26B-A4B)는 아직"
        self.L, self.D = k("block_count"), k("embedding_length")
        self.H = k("attention.head_count")
        hkv = k("attention.head_count_kv")
        self.Hkv = list(hkv) if isinstance(hkv, (list, tuple)) else [hkv] * self.L
        self.swa = list(k("attention.sliding_window_pattern"))
        self.hd = {True: k("attention.key_length_swa"), False: k("attention.key_length")}
        assert k("attention.value_length_swa") == self.hd[True] and k("attention.value_length") == self.hd[False]
        self.rot = {True: k("rope.dimension_count_swa", self.hd[True]), False: k("rope.dimension_count", self.hd[False])}
        self.theta = {True: float(k("rope.freq_base_swa")), False: float(k("rope.freq_base"))}
        self.창 = k("attention.sliding_window")
        self.eps = float(k("attention.layer_norm_rms_epsilon"))
        self.캡 = float(k("final_logit_softcapping", 0.0) or 0.0)
        self.인자 = g.풀기("rope_freqs.weight").astype(f64) if "rope_freqs.weight" in g.tensors else None
        self.묶인출력 = "output.weight" not in g.tensors

    def 돌리기(self, x, l, T):
        """NEOX RoPE — 짝 (i, i + n/2). 전체 층은 θ_i 를 rope_freqs[i] 로 나눈다."""
        s = self.swa[l]
        n = self.rot[s] // 2
        f = self.theta[s] ** (-2.0 * np.arange(n, dtype=f64) / self.rot[s])
        if not s and self.인자 is not None:
            f = f / self.인자[:n]
        각 = np.arange(T, dtype=f64)[:, None] * f[None, :]
        cos, sin = np.cos(각)[:, None, :], np.sin(각)[:, None, :]
        y = x.copy()
        a, b = x[..., :n], x[..., n:2 * n]
        y[..., :n] = a * cos - b * sin
        y[..., n:2 * n] = b * cos + a * sin
        return y

    def 어텐션(self, l, h, T):
        g, H = self.g, self.H
        s = self.swa[l]
        hd, Hk = self.hd[s], self.Hkv[l]
        q = Q.선형(h, g, f"blk.{l}.attn_q.weight").reshape(T, H, hd)
        k0 = Q.선형(h, g, f"blk.{l}.attn_k.weight").reshape(T, Hk, hd)
        이름v = f"blk.{l}.attn_v.weight"
        v0 = Q.선형(h, g, 이름v).reshape(T, Hk, hd) if 이름v in g.tensors else k0
        q = self.돌리기(Q.rms(q, g.풀기(f"blk.{l}.attn_q_norm.weight"), self.eps), l, T)
        kk = self.돌리기(Q.rms(k0, g.풀기(f"blk.{l}.attn_k_norm.weight"), self.eps), l, T)
        vv = rms0(v0, self.eps)
        묶 = H // Hk
        kk, vv = np.repeat(kk, 묶, 1), np.repeat(vv, 묶, 1)          # 질의 헤드 j → KV 헤드 j // 묶
        sc = np.einsum("thd,shd->hts", q, kk)                        # 배율 1
        i = np.arange(T)
        가림 = i[None, :] > i[:, None]
        if s:
            가림 = 가림 | (i[:, None] - i[None, :] >= self.창)
        sc = np.where(가림[None], -np.inf, sc)
        sc = np.exp(sc - sc.max(-1, keepdims=True))
        sc /= sc.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", sc, vv)
        return Q.선형(o.reshape(T, H * hd), g, f"blk.{l}.attn_output.weight")

    def 앞으로(self, ids):
        g, T = self.g, len(ids)
        x = Q.풀어(g, "token_embd.weight", ids).astype(f64) * math.sqrt(self.D)
        for l in range(self.L):
            h = Q.rms(x, g.풀기(f"blk.{l}.attn_norm.weight"), self.eps)
            x = x + Q.rms(self.어텐션(l, h, T), g.풀기(f"blk.{l}.post_attention_norm.weight"), self.eps)
            h = Q.rms(x, g.풀기(f"blk.{l}.ffn_norm.weight"), self.eps)
            f = Q.선형(gelu(Q.선형(h, g, f"blk.{l}.ffn_gate.weight")) * Q.선형(h, g, f"blk.{l}.ffn_up.weight"), g, f"blk.{l}.ffn_down.weight")
            x = x + Q.rms(f, g.풀기(f"blk.{l}.post_ffw_norm.weight"), self.eps)
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


def 라마부르기(판):
    os.environ["GEULPU_LLAMA"] = 판
    sys.path.insert(0, os.path.join(ROOT, "재기"))
    import 라마 as LM                                               # noqa: E402
    return LM


def 라마로짓(lm, ids):
    """llama.cpp 의 모든 행 로짓 — 위치 0 부터."""
    lm.지우기()
    lm.넣기(ids, list(range(len(ids))), [0] * len(ids), [1] * len(ids))
    return lm.모든로짓(len(ids))


def 채팅(gguf, 판, 작업, N):
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
        글 = f"<|turn>user\n{질문}<turn|>\n<|turn>model\n".encode("utf-8")            # GGUF 의 채팅 템플릿(생성 프롬프트까지)
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
    json.dump({"경우": [[ids, P] for ids, P, _ in 경우]}, open(os.path.join(작업, "채팅토큰.json"), "w", encoding="utf-8"))
    m = 모델(gguf)
    kls, 맞음, 행수 = [], 0, 0
    for i, (ids, P, z) in enumerate(경우):
        t0 = time.time()
        p = Q.로그소프트맥스(m.앞으로(ids))
        np.save(os.path.join(작업, f"채팅수학{i}.npy"), p.astype(np.float32))
        np.save(os.path.join(작업, f"채팅라마{i}.npy"), z)
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    gguf, 판, 작업 = sys.argv[1:4]
    os.makedirs(작업, exist_ok=True)
    if "--채팅" in sys.argv:
        return 채팅(gguf, 판, 작업, int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100)
    길이 = int(sys.argv[sys.argv.index("--length") + 1]) if "--length" in sys.argv else 256
    글들 = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--text"] or [os.path.join(ROOT, "연구자안내.md"),
                                                                                   os.path.join(ROOT, "README.md")]
    import 품질 as QQ                                                # noqa: E402 — 텍스트 고르기 · llama.cpp 의 llama_tokenize
    어휘 = QQ.라마어휘(판, gguf)
    토큰들 = [어휘.토큰(QQ._글바이트(지정)[1], 길이) for 지정 in 글들]
    json.dump({"gguf": os.path.basename(gguf), "글": 글들, "토큰": 토큰들}, open(os.path.join(작업, "토큰.json"), "w", encoding="utf-8"))
    if "--no-llama" not in sys.argv:
        LM = 라마부르기(판)
        for 이름, flash, kv in (("라마", -1, 1), ("라마f32", 0, 0)):     # 기본 · 플래시 어텐션 끔 + KV f32
            lm = LM.라마(gguf, n_ctx=길이, n_batch=길이, n_ubatch=min(512, 길이), n_seq_max=1, flash=flash, kv형식=kv)
            for i, ids in enumerate(토큰들):
                np.save(os.path.join(작업, f"{이름}{i}.npy"), 라마로짓(lm, ids))
            lm.닫기()
    m = 모델(gguf)
    print(f"{os.path.basename(gguf)}: 층 {m.L}(전체 어텐션 {sum(not s for s in m.swa)}) · 너비 {m.D} · 질의 헤드 {m.H} · "
          f"윈도 층 헤드 {m.hd[True]} × KV {m.Hkv[m.swa.index(True)]} · 전체 층 헤드 {m.hd[False]} × KV {m.Hkv[m.swa.index(False)]} · "
          f"윈도 {m.창} · 소프트캡 {m.캡:g} · 임베딩과 묶인 출력 {m.묶인출력}", flush=True)
    kls, 끼리들, 같음, 같끼리, 행수 = [], [], 0, 0, 0
    for i, ids in enumerate(토큰들):
        t0 = time.time()
        p = Q.로그소프트맥스(m.앞으로(ids))
        np.save(os.path.join(작업, f"수학{i}.npy"), p.astype(np.float32))
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
