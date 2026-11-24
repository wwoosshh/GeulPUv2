#!/usr/bin/env python3
"""0.25.0 25-1 — Inkling 2(GGUF 구조 inkling2_loop · inkling2)의 계약 CPU 참조: 지금의 계약(정수 블록 행렬곱 · 어텐션 · RMSNorm · σ · 소프트캡 ·
σ 라우팅 MoE)에 계약 4.15(블록 어텐션 잔차 · 층 반복의 제어 · 짧은 합성곱 · L2 노름 클리핑 · 로그 스케일 τ · 상대 위치 편향 · 어텐션 싱크 · SITU —
계약/정수계약.py)를 더한 순전파. 층의 뜻은 시험/inkling2수학.py(transformers 의 models/inkling2(_loop) · llama.cpp 의 src/models/inkling2.cpp 와
같음) 그대로이고, 연산마다 계약의 차례 · 반올림이다(문서/0.25.0.md).

  python -I -X utf8 시험/참조inkling2.py <gguf> --수학 <폴더> <출력.json> [--일꾼 3] [--저장된]

--수학 <폴더>: 시험/inkling2수학.py 의 토큰.json · 수학{i}.npy(fp64 의 이상적인 모델) · 라마{i}.npy(llama.cpp), 그리고 있으면 채팅 확인의
채팅토큰.json · 채팅수학{i}.npy · 채팅라마{i}.npy 와 견준다 — 계약의 손실 KL(수학 ‖ 계약) · 1위 일치, 같은 자리의 KL(수학 ‖ llama.cpp).
채팅(답을 예측하는 행)이 있으면 그것, 없으면 글(모든 행). 계약의 로짓을 {채팅|글}계약{i}.npy 로 남기고, --저장된 은 순전파 없이 그 파일로
견준다. 판정(문서/0.25.0.md 의 25-1 ②): KL(수학 ‖ 계약) 의 평균 ≤ KL(수학 ‖ llama.cpp) 의 평균 / 10.
"""
import json
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
import 참조7B as R7         # noqa: E402 — 가중 · 곱 · 헤드채움
import 참조35 as R35        # noqa: E402 — 라우터로짓 · 벡터가중
import 참조gemma4 as RG4    # noqa: E402 — 가중전문가(Q8_0 · Q5_0 · k-양자화)
import qwen35수학 as Q      # noqa: E402 — 풀어(임베딩의 행 — 파일의 값 그대로)

f32 = np.float32


def 설정(g):
    m = g.meta
    구조 = m["general.architecture"]
    assert 구조 in ("inkling2_loop", "inkling2"), 구조
    k = lambda x, d=None: m.get(구조 + "." + x, d)                       # noqa: E731
    L = k("block_count")
    hkv = k("attention.head_count_kv")
    hd = k("attention.key_length")
    s = {"구조": 구조, "층": L, "너비": k("embedding_length"), "머리": k("attention.head_count"),
         "KV머리": list(hkv) if isinstance(hkv, (list, tuple)) else [hkv] * L,
         "헤드": {False: hd, True: k("attention.key_length_swa", hd)},
         "윈도층": [bool(x) for x in k("attention.sliding_window_pattern")], "창": k("attention.sliding_window"),
         "eps": float(k("attention.layer_norm_rms_epsilon")), "R": k("attention.relative_length"),
         "qk클립": float(k("attention.qk_clip")), "r클립": float(k("attention.relative_clip")),
         "바닥": k("attention.log_scale_floor"), "알파": float(k("attention.log_scale_alpha")), "블록": k("attn_res.block_size"),
         "베타": float(k("activation.situ_beta")), "선베타": float(k("activation.situ_linear_beta")),
         "배율": float(k("expert_weights_scale", 0.0) or 0.0), "캡": float(k("final_logit_softcapping", 0.0) or 0.0),
         "어휘": k("vocab_size_unpadded"), "전문가": k("expert_count"), "고름": k("expert_used_count"),
         "전문가폭": k("expert_feed_forward_length")}
    if 구조 == "inkling2_loop":
        s["시작"], s["끝"], s["패스"] = k("loop.start"), k("loop.end"), k("loop.count")
        assert not k("loop.varying", False), "loop.varying 은 받지 않는다"
    else:
        s["시작"] = s["끝"] = L
        s["패스"] = 1
    assert len(s["윈도층"]) == L and s["전문가폭"] % 32 == 0 and s["너비"] % 32 == 0
    return s


def 가중16(g, 이름):
    """F32 행렬(또는 벡터 — 행 하나) → 1.3 의 16비트 블록 (n, E)."""
    w = g.풀기(이름)
    return C.가중치16(w.reshape(1, -1) if w.ndim == 1 else w)


def 가중전문가(g, 이름, e):
    """전문가 e 의 행렬 — F32 · F16 · BF16 은 1.3 의 W16(합성 파일), 그 밖은 참조gemma4.가중전문가."""
    t, dims, raw = g.원(이름)
    if t in (GR.F32, GR.F16, GR.BF16):
        n, E = C.가중치16(Q.풀어전문가(g, 이름, e))
        return "W16", n, E, None
    return RG4.가중전문가(g, 이름, e)


def 곱(h, g, 이름, 열):
    return R7.곱(np.ascontiguousarray(h, f32), R7.가중(g, 이름), np.zeros(열, f32))


def 헤드어텐션(q, k, v, s, swa, H, Hk, hd, 표, t):
    """헤드마다 계약 어텐션(4.15 — 편향 · 싱크) — 헤드 크기가 32 의 배수가 아니면 0 을 채워(계약 '헤드 크기 · RoPE 차원'의 1) 눈금은 실제 헤드의 것."""
    T = q.shape[0]
    hdp = (hd + 31) // 32 * 32
    o = np.zeros((T, H * hd), f32)
    눈금 = C.점수눈금(hd)
    for h in range(H):
        kv = h // (H // Hk)
        qh = R7.헤드채움(q[:, h * hd:(h + 1) * hd], hd, hdp)
        kh = R7.헤드채움(k[:, kv * hd:(kv + 1) * hd], hd, hdp)
        vh = R7.헤드채움(v[:, kv * hd:(kv + 1) * hd], hd, hdp)
        oh = C.어텐션(qh, kh, vh, 눈금=눈금, 창=(s["창"] if swa else None), 편향=표[:, h, :], 범위=표.shape[2], 싱크=t[:, h])
        o[:, h * hd:(h + 1) * hd] = oh[:, :hd]
    return o


def 층(g, s, l, x, 블록들, τ, 메시지=None):
    T, D, H, eps = x.shape[0], s["너비"], s["머리"], s["eps"]
    swa = s["윈도층"][l]
    Hk, hd, R = s["KV머리"][l], s["헤드"][swa], s["R"]
    p = f"blk.{l}."
    입력 = x
    if p + "attn_res_query.weight" in g.tensors:
        섞임 = C.블록섞기(블록들, x, *가중16(g, p + "attn_res_query.weight"), eps)
        h = [섞임[:, 0 if swa else j] for j in range(3)]
    else:
        h = [x, x, x]
    if 메시지 is not None:
        h = [제어(g, p, c, h[j], 메시지[:, j], eps) for j, c in enumerate(("q", "k", "v"))]
    if swa:
        w = g.풀기(p + "attn_norm.weight")
        h = [C.rms정규화(v, w, eps) for v in h] if 메시지 is not None else [C.rms정규화(h[0], w, eps)] * 3
    else:
        h = [C.rms정규화(h[j], g.풀기(p + f"attn_{c}_inp_norm.weight"), eps) for j, c in enumerate(("q", "k", "v"))]
    q = 곱(h[0], g, p + "attn_q.weight", H * hd)
    kd = None if swa else 곱(h[1], g, p + "attn_k_conv_proj.weight", Hk * 4)
    vd = None if swa else 곱(h[2], g, p + "attn_v_conv_proj.weight", Hk * 4)
    k, _ = C.짧은합성곱(곱(h[1], g, p + "attn_k.weight", Hk * hd), g.풀기(p + "attn_k_conv.weight"), None, kd, Hk)
    v, _ = C.짧은합성곱(곱(h[2], g, p + "attn_v.weight", Hk * hd), g.풀기(p + "attn_v_conv.weight"), None, vd, Hk)
    r = 곱(h[0], g, p + "attn_rel.weight", H * R)
    q = C.L2클립(C.곱(q, g.풀기(p + "attn_q_clip.weight").reshape(1, -1)), s["qk클립"], hd)
    k = C.L2클립(k, s["qk클립"], hd)
    v = C.머리정규화(v, np.ones(hd, f32), eps, hd)
    r = C.L2클립(C.곱(r, g.풀기(p + "attn_rel_clip.weight").reshape(1, -1)), s["r클립"], R)
    if not swa:
        q = C.곱(q, τ[:, None])
        r = C.곱(r, τ[:, None])
    t = 곱(h[0], g, p + "attn_sink_proj.weight", H)
    표 = C.상대편향표(r.reshape(T, H, R), g.풀기(p + "attn_rel_bias.weight"))
    o = 헤드어텐션(q, k, v, s, swa, H, Hk, hd, 표, t)
    o = C.게이트(o, 곱(h[0], g, p + "attn_gate.weight", H * hd))
    o, _ = C.짧은합성곱(곱(o, g, p + "attn_output.weight", D), g.풀기(p + "attn_post_conv.weight"))
    if l % s["블록"] == 0:
        블록들.append(입력)
        x = o
    else:
        x = C.더(입력, o)
    ffn입력 = x
    hf = C.블록섞기(블록들, x, *가중16(g, p + "ffn_res_query.weight"), eps)[:, 0]
    if 메시지 is not None:
        hf = 제어(g, p, "ffn", hf, 메시지[:, 3], eps)
    hf = C.rms정규화(hf, g.풀기(p + "ffn_norm.weight"), eps)
    o, _ = C.짧은합성곱(MoE(g, s, p, hf), g.풀기(p + "ffn_post_conv.weight"))
    return C.더(ffn입력, o)


def 제어(g, p, c, x, m, eps):
    b = p + f"loop_{c}_ctrl_"
    return C.반복제어(x, m, 가중16(g, b + "down.weight"), 가중16(g, b + "gain.weight"), 가중16(g, b + "forget.weight"), eps)


라우팅기록 = []


def MoE(g, s, p, h):
    T, D, F = h.shape[0], s["너비"], s["전문가폭"]
    l = R35.라우터로짓(h, g, p + "ffn_gate_inp.weight", s["전문가"])
    b = g.풀기(p + "exp_probs_b.bias") if p + "exp_probs_b.bias" in g.tensors else None
    고른, w = C.시그모이드전문가(l, b, s["고름"], True, s["배율"] if s["배율"] else 1.0)
    라우팅기록.append(고른.astype(np.int16))
    ys = np.zeros((s["고름"], T, D), f32)
    for e in np.unique(고른):
        행, 순위 = np.nonzero(고른 == e)
        gu = R7.곱(np.ascontiguousarray(h[행]), 가중전문가(g, p + "ffn_gate_up_exps.weight", e), np.zeros(2 * F, f32))
        a = C.SITU(gu[:, :F], gu[:, F:], s["베타"], s["선베타"])
        ys[순위, 행] = R7.곱(a, 가중전문가(g, p + "ffn_down_exps.weight", e), np.zeros(D, f32))
    return C.전문가합(w, ys)


def 앞으로(g, s, ids):
    """계약의 순전파 — 토큰열(위치 0 부터)의 모든 행 로짓 [행][어휘](소프트캡 뒤, 패딩한 어휘를 뺀 것)."""
    T, eps = len(ids), s["eps"]
    x = C.rms정규화(Q.풀어(g, "token_embd.weight", np.asarray(ids)), g.풀기("token_embd_norm.weight"), eps)
    τ = C.로그스케일(np.arange(T), s["바닥"], s["알파"])
    블록들 = []
    for l in range(s["시작"]):
        x = 층(g, s, l, x, 블록들, τ)
    앞x, 앞블록 = x, list(블록들)
    for 패스 in range(s["패스"] if s["시작"] < s["끝"] else 0):
        if 패스 > 0:
            은행, 은행출력 = 블록들[len(앞블록) + 1:], x
            x, 블록들 = 앞x, list(앞블록)
        for l in range(s["시작"], s["끝"]):
            메시지 = None if 패스 == 0 else C.블록섞기(은행, 은행출력, *가중16(g, f"blk.{l}.loop_msg_query.weight"), eps)
            x = 층(g, s, l, x, 블록들, τ, 메시지)
    for l in range(s["끝"], s["층"]):
        x = 층(g, s, l, x, 블록들, τ)
    x = C.블록섞기(블록들, x, *가중16(g, "output_res_query.weight"), eps)[:, 0]
    x = C.rms정규화(x, g.풀기("output_norm.weight"), eps)
    w = R7.가중(g, "output.weight")
    z = R7.곱(x, w, np.zeros(w[1].shape[0], f32))
    if s["캡"]:
        z = C.소프트캡(z, s["캡"])
    return z[:, :s["어휘"]]


def _일(인자들):
    gguf, ids = 인자들
    g = GR.GGUF(gguf)
    라우팅기록.clear()
    z = 앞으로(g, 설정(g), ids)
    return z, np.stack(라우팅기록)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    gguf = sys.argv[1]
    폴더, 출력 = sys.argv[sys.argv.index("--수학") + 1], sys.argv[sys.argv.index("--수학") + 2]
    일꾼 = int(sys.argv[sys.argv.index("--일꾼") + 1]) if "--일꾼" in sys.argv else 1
    채팅 = os.path.exists(os.path.join(폴더, "채팅토큰.json"))
    if 채팅:
        경우 = json.load(open(os.path.join(폴더, "채팅토큰.json"), encoding="utf-8"))["경우"]
        앞, 구간 = "채팅", [(P - 1, len(ids) - 1) for ids, P in 경우]
        토큰들 = [ids for ids, _ in 경우]
    else:
        토큰들 = json.load(open(os.path.join(폴더, "토큰.json"), encoding="utf-8"))["토큰"]
        앞, 구간 = "", [(0, len(ids)) for ids in 토큰들]
    t0 = time.time()
    if "--저장된" in sys.argv:
        결과 = [(np.load(os.path.join(폴더, f"{앞 or '글'}계약{i}.npy")), None) for i in range(len(토큰들))]
    elif 일꾼 > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(일꾼) as ex:
            결과 = list(ex.map(_일, [(gguf, ids) for ids in 토큰들]))
    else:
        결과 = [_일((gguf, ids)) for ids in 토큰들]
    줄 = {"gguf": os.path.basename(gguf), "경우": []}
    합계약, 합라마, 같음, 행수 = [], [], 0, 0
    for i, ((z, 라우팅), (a, b)) in enumerate(zip(결과, 구간)):
        if 라우팅 is not None:
            np.save(os.path.join(폴더, f"{앞 or '글'}계약{i}.npy"), z)
            np.save(os.path.join(폴더, f"{앞 or '글'}계약라우팅{i}.npy"), 라우팅)
        p = np.load(os.path.join(폴더, f"{앞}수학{i}.npy")).astype(np.float64)[a:b]
        r = Q.로그소프트맥스(np.asarray(z, np.float64)[a:b])
        kl = (np.exp(p) * (p - r)).sum(-1)
        같 = int((p.argmax(1) == r.argmax(1)).sum())
        항 = {"행": b - a, "KL계약": float(kl.mean()), "1위같음": 같}
        라마 = os.path.join(폴더, f"{앞}라마{i}.npy")
        if os.path.exists(라마):
            rl = Q.로그소프트맥스(np.load(라마).astype(np.float64)[a:b])
            kll = (np.exp(p) * (p - rl)).sum(-1)
            항["KL라마"] = float(kll.mean())
            합라마.append(kll)
        수학라우팅 = os.path.join(폴더, f"{앞}라우팅{i}.npy" if 앞 else f"라우팅{i}.npy")
        if 라우팅 is not None and os.path.exists(수학라우팅):
            mr = np.load(수학라우팅)
            if mr.shape == 라우팅.shape:
                항["라우팅같음"] = float((np.sort(mr, -1) == np.sort(라우팅, -1)).all(-1).mean())
        합계약.append(kl)
        같음 += 같
        행수 += b - a
        줄["경우"].append(항)
        print(f"경우 {i}: {b - a} 행 — KL(수학 ‖ 계약) 평균 {kl.mean():.3e} · 최대 {kl.max():.3e} · 1위 일치 {같}/{b - a}"
              + (f" | KL(수학 ‖ llama.cpp) 평균 {항['KL라마']:.3e}" if "KL라마" in 항 else "")
              + (f" | 고른 전문가 집합이 수학과 같은 행 {100 * 항['라우팅같음']:.2f} %" if "라우팅같음" in 항 else ""), flush=True)
    kc = float(np.concatenate(합계약).mean())
    줄.update({"KL계약": kc, "1위일치": 같음 / 행수, "초": time.time() - t0})
    글 = f"모두({time.time() - t0:.0f} 초): {행수} 행 — KL(수학 ‖ 계약) 평균 {kc:.3e} · 1위 일치 {100 * 같음 / 행수:.2f} %"
    if 합라마:
        kl라마 = float(np.concatenate(합라마).mean())
        줄["KL라마"] = kl라마
        글 += f" | KL(수학 ‖ llama.cpp) 평균 {kl라마:.3e} — 계약은 그 {kc / kl라마:.2e} 배"
        print(글)
        print("계약의 손실: " + ("맞음" if kc <= kl라마 / 10 else "다름") + "(KL(수학 ‖ 계약) ≤ KL(수학 ‖ llama.cpp) / 10)")
    else:
        print(글)
    json.dump(줄, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
