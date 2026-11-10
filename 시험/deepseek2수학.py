#!/usr/bin/env python3
"""0.24.0 24-2 — deepseek2(GGUF 구조 deepseek2 — MLA 어텐션 + MoE)의 수학 참조: 파일의 가중치를 풀어(양자화 값 그대로) 활성값을 fp64 로
계산한 순전파. gemma4수학.py 와 같은 쓰임 — GGUF 의 뜻(텐서의 배치 · 층의 차례 · 흡수한 MLA · 시그모이드 라우팅)을 llama.cpp 와 맞춰 본다
(구조/deepseek2/기록.md). 대상: GLM-4.7-Flash · Kanana-2-30B-A3B.

  python -I 시험/deepseek2수학.py <gguf> <llama.cpp 폴더> <작업 폴더> [--text <경로[@바이트]> …] [--length 256] [--llama-only] [--no-llama]

1. 토큰(llama.cpp 의 llama_tokenize — 토큰.json)과 llama.cpp 의 모든 행 로짓 두 설정(기본 — 플래시 어텐션 자동 · KV f16: 라마{i}.npy,
   플래시 어텐션 끔 · KV f32: 라마f32{i}.npy). 전문가 행렬은 호스트 메모리에(--cpu-moe). 이미 있으면 다시 만들지 않는다. GPU 를 쓰므로
   GPU 줄에서 --llama-only 로 돌리고, 수학 참조(CPU 만)는 줄 밖에서 다시 부른다.
2. 수학 참조의 로그 확률(수학{i}.npy)과 층마다 고른 전문가(라우팅.npz — 계약의 CPU 참조와 비교), 판정은 gemma4수학.py 와 같다 — 끝 줄
   '수학 참조: 맞음' 은 평균 KL(참조 ‖ llama.cpp) < 0.01 · 1위 일치 ≥ 95 %. 글들의 토큰열을 층마다 함께 계산한다(전문가 행렬을 한 번만 푼다).

층(llama.cpp b12224 의 src/models/deepseek2.cpp · build_moe_ffn 과 같은 식):
  h = RMSNorm(x; attn_norm)
  q = attn_q_b · RMSNorm(attn_q_a · h; attn_q_a_norm)(LoRA 가 없으면 attn_q · h) — 헤드마다 [nope N | rope R]
  kv = attn_kv_a_mqa · h → c = kv[:C] · k_pe = kv[C:], c′ = RMSNorm(c; attn_kv_a_norm)
  RoPE NORM(이웃 짝 (2i, 2i+1), θ_i = θ^(−2i/R)) — q 의 rope 부분 · k_pe
  흡수: q_abs,h = attn_k_b[h] · q_nope,h(C × N),  점수 = (q_abs,h · c′ + q_pe,h · k_pe) / √(N + R), 인과 마스크, softmax
  o_h = Σ softmax · c′,  o′_h = attn_v_b[h] · o_h(V × C),  x ← x + attn_output · [o′_0 | …]
  h₂ = RMSNorm(x; ffn_norm) — 층 < leading_dense_block_count: SwiGLU, 아니면 MoE:
    p = σ(ffn_gate_inp · h₂), 상위 k 를 p + exp_probs_b 로, w = p(고른 것), w ← w / max(Σw, 6.103515625e-5)(expert_weights_norm),
    w ← w · expert_weights_scale, 출력 = Σ w_r · SwiGLU_e_r(h₂) + 공유 전문가 SwiGLU(h₂)
  x ← x + 출력.  끝: RMSNorm(x; output_norm), 로짓 = output · x
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
import qwen35수학 as Q      # noqa: E402 — 풀어 · 풀어전문가 · 선형 · rms · silu · sigmoid · 로그소프트맥스

f64 = np.float64


def 설정(g):
    """메타데이터 → 모양. 이 참조가 받지 않는 파일은 까닭과 함께 멈춘다."""
    m = g.meta
    assert m["general.architecture"] == "deepseek2", m["general.architecture"]
    k = lambda x, d=None: m.get("deepseek2." + x, d)                 # noqa: E731
    assert "blk.0.attn_k_b.weight" in g.tensors, "흡수하지 않은 옛 파일(attn_kv_b)은 받지 않는다"
    assert k("expert_group_count", 1) <= 1, "그룹 고르기(expert_group_count > 1)는 아직"
    assert k("expert_gating_func") == 2, f"라우팅 함수 {k('expert_gating_func')} — 시그모이드(2)만"
    assert m.get("deepseek2.rope.scaling.type", "none") in ("none", None), "RoPE 늘이기는 아직"
    assert not k("nextn_predict_layers", 0), "MTP 층이 있는 파일은 아직"
    R = k("rope.dimension_count")
    s = {"층": k("block_count"), "너비": k("embedding_length"), "머리": k("attention.head_count"), "eps": float(k("attention.layer_norm_rms_epsilon")),
         "C": k("attention.kv_lora_rank"), "R": R, "N": k("attention.key_length_mla") - R, "V": k("attention.value_length_mla"),
         "qa": k("attention.q_lora_rank", 0), "θ": float(k("rope.freq_base")), "dense": k("leading_dense_block_count", 0),
         "전문가": k("expert_count"), "고름": k("expert_used_count"), "정규화": bool(k("expert_weights_norm", False)),
         "배율": float(k("expert_weights_scale", 1.0))}
    assert k("attention.key_length") == s["C"] + R and k("attention.value_length") == s["C"], "흡수한 키 · 값의 길이가 C + R · C 가 아니다"
    return s


def rope이웃(x, 위치, θ, R):
    """RoPE NORM — 짝 (2i, 2i + 1), θ_i = θ^(−2i/R). x [..., R], 위치 [T](x 의 첫 축)."""
    f = θ ** (-2.0 * np.arange(R // 2, dtype=f64) / R)
    각 = 위치.astype(f64)[:, None] * f[None, :]
    shape = [len(위치)] + [1] * (x.ndim - 2) + [R // 2]
    cos, sin = np.cos(각).reshape(shape), np.sin(각).reshape(shape)
    y = x.copy()
    a, b = x[..., 0::2], x[..., 1::2]
    y[..., 0::2] = a * cos - b * sin
    y[..., 1::2] = b * cos + a * sin
    return y


class 모델:
    def __init__(self, path):
        self.g = GR.GGUF(path)
        self.s = 설정(self.g)
        self.라우팅 = {}                                              # 층 → [Σ 행][k] 고른 전문가

    def 어텐션(self, l, h, 경우):
        g, s = self.g, self.s
        H, C, R, N, V = s["머리"], s["C"], s["R"], s["N"], s["V"]
        p = f"blk.{l}."
        if s["qa"]:
            a = Q.rms(Q.선형(h, g, p + "attn_q_a.weight"), g.풀기(p + "attn_q_a_norm.weight"), s["eps"])
            q = Q.선형(a, g, p + "attn_q_b.weight")
        else:
            q = Q.선형(h, g, p + "attn_q.weight")
        q = q.reshape(len(h), H, N + R)
        kv = Q.선형(h, g, p + "attn_kv_a_mqa.weight")
        c = Q.rms(kv[:, :C], g.풀기(p + "attn_kv_a_norm.weight"), s["eps"])
        wk = np.stack([Q.풀어전문가(g, p + "attn_k_b.weight", i) for i in range(H)]).astype(f64)        # [H][C][N]
        wv = np.stack([Q.풀어전문가(g, p + "attn_v_b.weight", i) for i in range(H)]).astype(f64)        # [H][V][C]
        out = np.empty((len(h), H, V))
        for a0, a1 in 경우:
            T = a1 - a0
            위치 = np.arange(T)
            qpe = rope이웃(q[a0:a1, :, N:], 위치, s["θ"], R)           # [T][H][R]
            kpe = rope이웃(kv[a0:a1, C:], 위치, s["θ"], R)             # [T][R]
            qabs = np.einsum("hcn,thn->thc", wk, q[a0:a1, :, :N])       # [T][H][C]
            cc = c[a0:a1]
            sc = (np.einsum("thc,sc->hts", qabs, cc) + np.einsum("thr,sr->hts", qpe, kpe)) / math.sqrt(N + R)
            i = np.arange(T)
            sc = np.where((i[None, :] > i[:, None])[None], -np.inf, sc)
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            sc /= sc.sum(-1, keepdims=True)
            o = np.einsum("hts,sc->thc", sc, cc)                         # [T][H][C]
            out[a0:a1] = np.einsum("hvc,thc->thv", wv, o)
        return Q.선형(out.reshape(len(h), H * V), g, p + "attn_output.weight")

    def FFN(self, l, h):
        g, s = self.g, self.s
        p = f"blk.{l}."
        if l < s["dense"]:
            return Q.선형(Q.silu(Q.선형(h, g, p + "ffn_gate.weight")) * Q.선형(h, g, p + "ffn_up.weight"), g, p + "ffn_down.weight")
        로짓 = h @ g.풀기(p + "ffn_gate_inp.weight").reshape(s["전문가"], -1).astype(f64).T
        pr = Q.sigmoid(로짓)
        b = g.풀기(p + "exp_probs_b.bias").astype(f64) if p + "exp_probs_b.bias" in g.tensors else 0.0
        고른 = np.argsort(-(pr + b), axis=-1, kind="stable")[:, :s["고름"]]
        self.라우팅[l] = 고른.astype(np.int16)
        w = np.take_along_axis(pr, 고른, -1)
        if s["정규화"]:
            w = w / np.maximum(w.sum(-1, keepdims=True), 6.103515625e-5)
        w = w * s["배율"]
        out = np.zeros_like(h)
        for e in np.unique(고른):
            행, 순위 = np.nonzero(고른 == e)
            he = h[행]
            gg = he @ Q.풀어전문가(g, p + "ffn_gate_exps.weight", e).astype(f64).T
            uu = he @ Q.풀어전문가(g, p + "ffn_up_exps.weight", e).astype(f64).T
            y = (Q.silu(gg) * uu) @ Q.풀어전문가(g, p + "ffn_down_exps.weight", e).astype(f64).T
            np.add.at(out, 행, w[행, 순위][:, None] * y)
        공 = Q.선형(Q.silu(Q.선형(h, g, p + "ffn_gate_shexp.weight")) * Q.선형(h, g, p + "ffn_up_shexp.weight"), g, p + "ffn_down_shexp.weight")
        return out + 공

    def 앞으로(self, 토큰열들, 말=True):
        """토큰열들(각각 위치 0 부터) → 로짓 [Σ 행][어휘](fp64)과 시작 행."""
        g, s = self.g, self.s
        ids = np.concatenate([np.asarray(t, np.int64) for t in 토큰열들])
        시작 = np.cumsum([0] + [len(t) for t in 토큰열들])
        경우 = [(int(시작[i]), int(시작[i + 1])) for i in range(len(토큰열들))]
        x = Q.풀어(g, "token_embd.weight", ids).astype(f64)
        for l in range(s["층"]):
            t0 = time.time()
            x = x + self.어텐션(l, Q.rms(x, g.풀기(f"blk.{l}.attn_norm.weight"), s["eps"]), 경우)
            x = x + self.FFN(l, Q.rms(x, g.풀기(f"blk.{l}.ffn_norm.weight"), s["eps"]))
            if 말:
                print(f"층 {l}: {time.time() - t0:.1f} 초", flush=True)
        x = Q.rms(x, g.풀기("output_norm.weight"), s["eps"])
        이름 = "output.weight" if "output.weight" in g.tensors else "token_embd.weight"
        V = g.tensors[이름][1][1]
        로짓 = np.empty((len(x), V))
        for a in range(0, V, 16384):
            로짓[:, a:a + 16384] = x @ Q.풀어(g, 이름, np.arange(a, min(V, a + 16384))).astype(f64).T
        return 로짓, 시작


def 라마부르기(판):
    os.environ["GEULPU_LLAMA"] = 판
    sys.path.insert(0, os.path.join(ROOT, "재기"))
    import 라마 as LM                                               # noqa: E402
    return LM


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    gguf, 판, 작업 = sys.argv[1:4]
    os.makedirs(작업, exist_ok=True)
    길이 = int(sys.argv[sys.argv.index("--length") + 1]) if "--length" in sys.argv else 256
    글들 = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--text"] or [os.path.join(ROOT, "연구자안내.md"),
                                                                                   os.path.join(ROOT, "README.md")]
    토큰경로 = os.path.join(작업, "토큰.json")
    if os.path.exists(토큰경로):
        토큰들 = json.load(open(토큰경로, encoding="utf-8"))["토큰"]
    else:
        import 품질 as QQ                                            # noqa: E402 — 텍스트 고르기 · llama.cpp 의 llama_tokenize
        어휘 = QQ.라마어휘(판, gguf)
        토큰들 = [어휘.토큰(QQ._글바이트(지정)[1], 길이) for 지정 in 글들]
        json.dump({"gguf": os.path.basename(gguf), "글": 글들, "토큰": 토큰들}, open(토큰경로, "w", encoding="utf-8"))
    if "--no-llama" not in sys.argv:
        LM = None
        for 이름, flash, kv in (("라마", -1, 1), ("라마f32", 0, 0)):     # 기본 · 플래시 어텐션 끔 + KV f32
            if all(os.path.exists(os.path.join(작업, f"{이름}{i}.npy")) for i in range(len(토큰들))):
                continue
            LM = LM or 라마부르기(판)
            n = max(len(t) for t in 토큰들)
            lm = LM.라마(gguf, n_ctx=n, n_batch=n, n_ubatch=min(512, n), n_seq_max=1, flash=flash, kv형식=kv, 전문가CPU=True)
            for i, ids in enumerate(토큰들):
                lm.지우기()
                lm.넣기(ids, list(range(len(ids))), [0] * len(ids), [1] * len(ids))
                np.save(os.path.join(작업, f"{이름}{i}.npy"), lm.모든로짓(len(ids)))
            lm.닫기()
            print(f"llama.cpp {이름}: 글 {len(토큰들)} 개", flush=True)
    if "--llama-only" in sys.argv:
        return
    m = 모델(gguf)
    s = m.s
    print(f"{os.path.basename(gguf)}: 층 {s['층']}(dense {s['dense']}) · 너비 {s['너비']} · 헤드 {s['머리']} · C {s['C']} · R {s['R']} · N {s['N']} · "
          f"V {s['V']} · q LoRA {s['qa']} · 전문가 {s['전문가']} 가운데 {s['고름']} · 배율 {s['배율']:g}", flush=True)
    t0 = time.time()
    로짓, 시작 = m.앞으로(토큰들)
    print(f"수학 참조: {time.time() - t0:.0f} 초", flush=True)
    np.savez_compressed(os.path.join(작업, "라우팅.npz"), **{str(l): v for l, v in m.라우팅.items()})
    kls, 끼리들, 같음, 같끼리, 행수 = [], [], 0, 0, 0
    for i, ids in enumerate(토큰들):
        p = Q.로그소프트맥스(로짓[시작[i]:시작[i + 1]])
        np.save(os.path.join(작업, f"수학{i}.npy"), p.astype(np.float32))
        if not os.path.exists(os.path.join(작업, f"라마{i}.npy")):
            print(f"글 {i}: {len(ids)} 토큰 — 1위(앞 8 행) {p[:8].argmax(1).tolist()}", flush=True)
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
        print(f"글 {i}: {len(ids)} 토큰 — KL 평균 {kl.mean():.3e} · 최대 {kl.max():.3e} · 1위 일치 {같}/{len(ids)}", flush=True)
        print(f"  llama.cpp 플래시 어텐션 끔 · KV f32: KL 평균 {kl2.mean():.3e} · 1위 일치 {int((p.argmax(1) == r2.argmax(1)).sum())}/{len(ids)} | "
              f"llama.cpp 의 두 설정끼리: 대칭 KL 평균 {끼리.mean():.3e} · 1위 일치 {int((r.argmax(1) == r2.argmax(1)).sum())}/{len(ids)}", flush=True)
    if kls:
        평균 = float(np.concatenate(kls).mean())
        일치 = 같음 / 행수
        평균끼리 = float(np.concatenate(끼리들).mean())
        print(f"모두: KL 평균 {평균:.3e} · 1위 일치 {100 * 일치:.2f} % | llama.cpp 의 두 설정끼리: 대칭 KL 평균 {평균끼리:.3e} · "
              f"1위 일치 {100 * 같끼리 / 행수:.2f} %")
        좋음 = 평균 < 0.01 and 일치 >= 0.95
        print("수학 참조: 맞음" if 좋음 else "수학 참조: 다름")
        sys.exit(0 if 좋음 else 1)


if __name__ == "__main__":
    main()
