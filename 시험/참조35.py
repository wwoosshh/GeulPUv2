#!/usr/bin/env python3
"""0.12.0 12-3e — Qwen3.5(GGUF 구조 qwen35)의 계약 CPU 참조: 지금의 계약(정수 블록 행렬곱 · 어텐션 · RMSNorm · RoPE · SwiGLU)에 계약 4.9(출력
게이트) · 4.10(선형 어텐션 GDN — 계약/정수계약.py)을 더한 순전파. 층의 뜻은 시험/qwen35수학.py(12-3a — llama.cpp 의 src/models/qwen35.cpp 와 같음)
그대로이고, 연산마다 계약의 차례 · 반올림이다.

  python -I 시험/참조35.py <qwen35 gguf> --수학 <폴더> <출력.json> [--빠른] [--라우터fma]
  python -I 시험/참조35.py <qwen35 gguf> --엔진 <시험_7B 출력 폴더> <출력.json> [--빠른]

--수학 <폴더>: 12-3a 의 시험/qwen35수학.py 가 남긴 토큰.json · 수학{i}.npy(fp64 의 이상적인 모델 — 로그 확률) · 라마{i}.npy(llama.cpp 로짓)와
견준다 — 계약의 손실 KL(수학 ‖ 계약) · 1위 일치, 그리고 같은 자리의 KL(수학 ‖ llama.cpp). --빠른: 빠른 계약 F1(선형층의 활성값 자리 하나 —
어텐션 · GDN 은 그대로).
--엔진 <폴더>(12-3e-4): 시험_7B 가 남긴 경우{i}_토큰.bin · 경우{i}_로짓.bin(잡기시작 0 의 경우 — 모든 행)과 계약의 순전파를 비트까지(참조7B.py 의 꼴).
0.12.0 12-3g — 구조 qwen35moe(Qwen3.5 · 3.6-35B-A3B): FFN 이 MoE(계약 4.11). --라우터fma 는 라우터 로짓을 fp32 FMA 체인으로(기본은 1 절의 정수
블록 계약 — F32 가중치는 계약 1.3), --수학 의 라우팅{i}.npy 가 있으면 고른 전문가 집합이 수학 참조와 같은 비율을 낸다.
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
import 참조7B as R7         # noqa: E402

f32 = np.float32


def 설정35(g):
    m = g.meta
    구조 = m["general.architecture"]
    assert 구조 in ("qwen35", "qwen35moe"), 구조
    k = 구조 + "."
    s = {"층": m[k + "block_count"], "너비": m[k + "embedding_length"], "머리": m[k + "attention.head_count"],
         "KV머리": m[k + "attention.head_count_kv"], "헤드": m[k + "attention.key_length"], "eps": float(m[k + "attention.layer_norm_rms_epsilon"]),
         "돌림": m[k + "rope.dimension_count"], "θ": float(m[k + "rope.freq_base"]), "간격": m[k + "full_attention_interval"],
         "d": m[k + "ssm.state_size"], "Hk": m[k + "ssm.group_count"], "Hv": m[k + "ssm.time_step_rank"], "합성곱": m[k + "ssm.conv_kernel"]}
    assert m[k + "attention.value_length"] == s["헤드"] and s["합성곱"] == 4 and s["d"] == 128 and s["Hv"] % s["Hk"] == 0, s
    assert m.get(k + "rope.scaling.type", "none") == "none", "RoPE 늘이기는 받지 않는다"
    s["moe"] = 구조 == "qwen35moe"                                  # 0.12.0 12-3g — FFN 이 MoE(계약 4.11)
    if s["moe"]:
        s["전문가"], s["고름"] = m[k + "expert_count"], m[k + "expert_used_count"]
    return s


라우터길 = "블록"          # 계약 4.11 1 — "블록"(1 절의 정수 블록 계약 · F32 가중치는 계약 1.3 — 계약) 또는 "fma"(fp32 FMA 체인 — --라우터fma, 12-3g-1 의 비교용)
라우팅기록 = []           # MoE 층마다 고른 전문가 [행][k] — 수학 참조의 라우팅과 비교


def 가중전문가(g, name, e):
    """0.12.0 12-3g — 3 차원 전문가 텐서 [전문가][N][K] 의 전문가 e 하나를 참조7B.가중 의 꼴로(Q8_0 은 (형식, q, d, None), k-양자화는 풀기K)."""
    t, dims, raw = g.원(name)
    K, N = dims[0], dims[1]
    vb, bb = GR.블록[t]
    한 = N * (K // vb * bb)
    r = np.asarray(raw)[e * 한:(e + 1) * 한]
    if t == GR.Q8_0:
        bl = r.reshape(N, K // 32, 34)
        d = np.ascontiguousarray(bl[:, :, :2]).view(np.float16).reshape(N, K // 32).astype(f32)
        return "Q8_0", np.ascontiguousarray(bl[:, :, 2:]).view(np.int8).reshape(N, K), d, None
    return C.풀기K(t, r, N, K)


def 벡터가중(g, name):
    """1 차원 F32 가중치(공유 전문가의 게이트 ffn_gate_inp_shexp — 열 하나의 행렬)를 참조7B.가중 의 W16 꼴로(계약 1.3)."""
    n, E = C.가중치16(g.풀기(name).reshape(1, -1))
    return "W16", n, E, None


def 라우터로짓(h, g, name, 열):
    """계약 4.11 1 — 라우터길 이 "블록" 이면 1 절의 정수 블록 계약(F32 는 계약 1.3), "fma" 면 fp32 FMA 체인(활성값은 16비트 블록 형식)."""
    if 라우터길 == "fma":
        W = g.풀기(name).reshape(열, -1)
        return C.라우터FMA(h, W, R7.선형자리수)
    w = 벡터가중(g, name) if 열 == 1 else R7.가중(g, name)
    return R7.곱(h, w, np.zeros(열, f32))


def MoE(g, p, h, s):
    """계약 4.11 — 라우터 → softmax → 상위 k → 무게 → 전문가 SwiGLU(행을 전문가마다 묶어 — 행마다 같은 셈) → 순위 차례의 합 → 공유 전문가 × σ."""
    T, D = h.shape
    l = 라우터로짓(h, g, p + "ffn_gate_inp.weight", s["전문가"])
    pr = C.소프트맥스전문가(l)
    고른 = C.상위k(l, s["고름"])
    라우팅기록.append(고른.astype(np.int16))
    w = C.전문가무게(pr, 고른)
    ys = np.zeros((s["고름"], T, D), f32)
    for e in np.unique(고른):
        행, 순위 = np.nonzero(고른 == e)
        he = np.ascontiguousarray(h[행])
        wg = 가중전문가(g, p + "ffn_gate_exps.weight", e)
        Fe = wg[1].shape[0]
        gg = R7.곱(he, wg, np.zeros(Fe, f32))
        uu = R7.곱(he, 가중전문가(g, p + "ffn_up_exps.weight", e), np.zeros(Fe, f32))
        ys[순위, 행] = R7.곱(C.swiglu(gg, uu), 가중전문가(g, p + "ffn_down_exps.weight", e), np.zeros(D, f32))
    o = C.전문가합(w, ys)
    wg = R7.가중(g, p + "ffn_gate_shexp.weight")
    Fs = wg[1].shape[0]
    gs = R7.곱(h, wg, np.zeros(Fs, f32))
    us = R7.곱(h, R7.가중(g, p + "ffn_up_shexp.weight"), np.zeros(Fs, f32))
    ysh = R7.곱(C.swiglu(gs, us), R7.가중(g, p + "ffn_down_shexp.weight"), np.zeros(D, f32))
    게 = C.시그모이드(라우터로짓(h, g, p + "ffn_gate_inp_shexp.weight", 1)[:, 0])
    return C.fma32(np.broadcast_to(게[:, None], o.shape), ysh, o)


def 임베딩(g, 경우):
    """참조7B.앞으로 의 임베딩과 같다 — Q8_0 · Q4_0 은 q · d(정확), k-양자화는 임베딩K, F16 · BF16 · F32 는 파일의 값."""
    if g.tensors["token_embd.weight"][0] in (GR.F16, GR.BF16, GR.F32):
        ew = g.풀기("token_embd.weight")
        return np.concatenate([ew[S] for S in 경우], 0).astype(f32)
    we = R7.가중(g, "token_embd.weight")
    if we[0] in ("Q8_0", "Q4_0"):
        eq, ed = we[1], we[2]
        return np.concatenate([(eq[S].astype(f32) * np.repeat(ed[S], 32, axis=1)).astype(f32) for S in 경우], 0)
    return np.concatenate([C.임베딩K(we, S) for S in 경우], 0)


def 앞으로(g, 경우, 말=True):
    """계약의 순전파 — 토큰열들(경우마다 위치 0 부터 · 합성곱과 GDN 의 상태는 0 부터)의 모든 행 로짓 [Σ 행][어휘]과 경우의 시작 행."""
    s = 설정35(g)
    NL, D, H, Hkv, hd, eps = s["층"], s["너비"], s["머리"], s["KV머리"], s["헤드"], s["eps"]
    d, Hk, Hv, rot = s["d"], s["Hk"], s["Hv"], s["돌림"]
    x = 임베딩(g, 경우)
    시작 = np.cumsum([0] + [len(S) for S in 경우])
    cos, sin = C.rope표(max(64, int(max(len(S) for S in 경우))), s["θ"], hd=rot)
    눈금 = f32(1.0 / np.sqrt(np.float64(hd)))                      # 헤드 256 이면 1/16(정확)
    G_ = H // Hkv
    짝 = np.arange(Hv) % Hk                                       # v 헤드 j → k 헤드 j mod H_k(llama.cpp 의 ggml_repeat)
    kd = Hk * d

    def z0(n):
        return np.zeros(n, f32)
    for l in range(NL):
        tl = time.time()
        p = f"blk.{l}."
        h = C.rms정규화(x, g.풀기(p + "attn_norm.weight"), eps)
        if (l + 1) % s["간격"] == 0:                                # 전체 어텐션(계약 4.9)
            qg = R7.곱(h, R7.가중(g, p + "attn_q.weight"), z0(2 * H * hd)).reshape(-1, H, 2, hd)
            q = np.ascontiguousarray(qg[:, :, 0]).reshape(-1, H * hd)
            gate = np.ascontiguousarray(qg[:, :, 1]).reshape(-1, H * hd)
            kk = R7.곱(h, R7.가중(g, p + "attn_k.weight"), z0(Hkv * hd))
            vv = R7.곱(h, R7.가중(g, p + "attn_v.weight"), z0(Hkv * hd))
            q = C.머리정규화(q, g.풀기(p + "attn_q_norm.weight"), eps, hd)
            kk = C.머리정규화(kk, g.풀기(p + "attn_k_norm.weight"), eps, hd)
            ys = []
            for si in range(len(경우)):
                a0, a1 = 시작[si], 시작[si + 1]
                위치 = np.arange(a1 - a0)

                def 돌(r):
                    return np.concatenate([C.rope(r[:, :rot], cos, sin, 위치), r[:, rot:]], 1)
                qh = [돌(q[a0:a1, hd * a:hd * a + hd]) for a in range(H)]
                kh = [돌(kk[a0:a1, hd * a:hd * a + hd]) for a in range(Hkv)]
                vh = [np.ascontiguousarray(vv[a0:a1, hd * a:hd * a + hd]) for a in range(Hkv)]
                ys.append(np.concatenate([C.어텐션(qh[a], kh[a // G_], vh[a // G_], 눈금=눈금) for a in range(H)], 1))
            y = C.게이트(np.concatenate(ys, 0), gate)
            x = (x + R7.곱(y, R7.가중(g, p + "attn_output.weight"), z0(D))).astype(f32)
        else:                                                       # 선형 어텐션 GDN(계약 4.10)
            Wqkv = R7.가중(g, p + "attn_qkv.weight")
            qkv = R7.곱(h, Wqkv, z0(Wqkv[1].shape[0]))
            del Wqkv
            zg = R7.곱(h, R7.가중(g, p + "attn_gate.weight"), z0(Hv * d))
            bb = R7.곱(h, R7.가중(g, p + "ssm_beta.weight"), z0(Hv))
            aa = R7.곱(h, R7.가중(g, p + "ssm_alpha.weight"), z0(Hv))
            cw = g.풀기(p + "ssm_conv1d.weight")                       # [채널][4] — 칸 0 이 가장 오래된 토큰
            dt, A_ = g.풀기(p + "ssm_dt.bias"), g.풀기(p + "ssm_a")
            nw = g.풀기(p + "ssm_norm.weight")
            입력 = []
            for si in range(len(경우)):
                a0, a1 = 시작[si], 시작[si + 1]
                T = a1 - a0
                yc = C.실루(C.합성곱4(qkv[a0:a1], cw)[0])
                qn = C.곱(C.L2정규화(yc[:, :kd], eps, d), C.눈금128)
                kn = C.L2정규화(yc[:, kd:2 * kd], eps, d)
                입력.append((np.ascontiguousarray(qn.reshape(T, Hk, d)[:, 짝]), np.ascontiguousarray(kn.reshape(T, Hk, d)[:, 짝]),
                           np.ascontiguousarray(yc[:, 2 * kd:]).reshape(T, Hv, d), C.시그모이드(bb[a0:a1]),
                           C.적힌지수(C.곱(C.소프트플러스(C.더(aa[a0:a1], dt[None, :])), A_[None, :]))))
            ob = [None] * len(경우)
            for T in sorted({len(S) for S in 경우}):                  # 길이가 같은 경우는 헤드 축으로 묶어 한 번에(헤드끼리 따로 — 값은 같다)
                묶 = [si for si in range(len(경우)) if len(경우[si]) == T]
                o, _ = C.GDN(*(np.concatenate([입력[si][j] for si in 묶], 1) for j in range(5)))
                for n_, si in enumerate(묶):
                    ob[si] = o[:, n_ * Hv:(n_ + 1) * Hv]
            os_ = []
            for si in range(len(경우)):
                a0, a1 = 시작[si], 시작[si + 1]
                o = C.머리정규화(np.ascontiguousarray(ob[si]).reshape(a1 - a0, Hv * d), nw, eps, d)
                os_.append(C.곱(o, C.실루(zg[a0:a1])))
            x = (x + R7.곱(np.concatenate(os_, 0), R7.가중(g, p + "ssm_out.weight"), z0(D))).astype(f32)
        h = C.rms정규화(x, g.풀기(p + "post_attention_norm.weight"), eps)
        if s["moe"]:                                                # 0.12.0 12-3g — 계약 4.11
            x = C.더(x, MoE(g, p, h, s))
            if 말:
                print(f"층 {l}({'어텐션' if (l + 1) % s['간격'] == 0 else 'GDN'} · MoE): {time.time() - tl:.1f} 초", flush=True)
            continue
        wg = R7.가중(g, p + "ffn_gate.weight")
        F = wg[1].shape[0]
        gg = R7.곱(h, wg, z0(F))
        del wg
        uu = R7.곱(h, R7.가중(g, p + "ffn_up.weight"), z0(F))
        x = (x + R7.곱(C.swiglu(gg, uu), R7.가중(g, p + "ffn_down.weight"), z0(D))).astype(f32)
        if 말:
            print(f"층 {l}({'어텐션' if (l + 1) % s['간격'] == 0 else 'GDN'}): {time.time() - tl:.1f} 초", flush=True)
    h = C.rms정규화(x, g.풀기("output_norm.weight"), eps)
    wh = R7.가중(g, "output.weight" if "output.weight" in g.tensors else "token_embd.weight")
    return R7.곱(h, wh, z0(wh[1].shape[0])), 시작


def 로그소프트맥스(x):
    x = np.asarray(x, np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def 엔진대조(gguf, 폴더, 출력):
    """12-3e-4 — 시험_7B 의 출력(경우마다 토큰열 S 와 모든 행의 로짓)을 계약의 순전파와 비트까지."""
    g = GR.GGUF(gguf)
    경우 = []
    while os.path.exists(os.path.join(폴더, f"경우{len(경우)}_토큰.bin")):
        경우.append(np.fromfile(os.path.join(폴더, f"경우{len(경우)}_토큰.bin"), np.int64))
    print("경우", [len(S) for S in 경우], flush=True)
    t0 = time.time()
    로짓, 시작 = 앞으로(g, 경우)
    V = 로짓.shape[1]
    결과 = {"gguf": os.path.basename(gguf), "모드": "F1" if R7.선형자리수 == 1 else "정밀", "경우": []}
    for i, S in enumerate(경우):
        A = np.fromfile(os.path.join(폴더, f"경우{i}_로짓.bin"), f32).reshape(len(S), V)
        R = 로짓[시작[i]:시작[i + 1]]
        한 = {"토큰": len(S), "다른 칸": R7.다른칸(A, R), "칸": int(A.size), "행마다 다른 칸": [R7.다른칸(A[k], R[k]) for k in range(len(S))]}
        결과["경우"].append(한)
        print(f"경우 {i}: 행 {len(S)} — 다른 칸 {한['다른 칸']} / {한['칸']}", flush=True)
    결과["CPU 참조 초"] = round(time.time() - t0, 1)
    결과["모두 0"] = all(c["다른 칸"] == 0 for c in 결과["경우"])
    print("모두 0:", 결과["모두 0"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def main():
    global 라우터길
    sys.stdout.reconfigure(encoding="utf-8")
    if "--빠른" in sys.argv:
        R7.선형자리수 = 1
    if "--라우터fma" in sys.argv:                 # 0.12.0 12-3g — 계약 4.11 1 의 다른 길(fp32 FMA 체인)
        라우터길 = "fma"
    인자 = [a for a in sys.argv[1:] if not a.startswith("--")]
    gguf = 인자[0]
    if "--엔진" in sys.argv:
        엔진대조(gguf, sys.argv[sys.argv.index("--엔진") + 1], [a for a in 인자[1:] if a != sys.argv[sys.argv.index("--엔진") + 1]][0])
        return
    폴더 = sys.argv[sys.argv.index("--수학") + 1]
    출력 = [a for a in 인자[1:] if a != 폴더][0]
    g = GR.GGUF(gguf)
    토큰들 = json.load(open(os.path.join(폴더, "토큰.json"), encoding="utf-8"))["토큰"]
    t0 = time.time()
    로짓, 시작 = 앞으로(g, [np.asarray(t, np.int64) for t in 토큰들])
    결과 = {"gguf": os.path.basename(gguf), "모드": "F1" if R7.선형자리수 == 1 else "정밀", "초": round(time.time() - t0), "글": []}
    if 라우팅기록:                                                   # 0.12.0 12-3g — 고른 전문가(집합)가 수학 참조와 같은가
        결과["라우터"] = 라우터길
        우리 = np.stack(라우팅기록)                                     # [MoE 층][Σ 행][k]
        같은집합, 겹침, 칸 = 0, 0, 0
        for i in range(len(토큰들)):
            경로 = os.path.join(폴더, f"라우팅{i}.npy")
            if not os.path.exists(경로):
                continue
            수 = np.sort(np.load(경로).astype(np.int64), -1)
            우 = np.sort(우리[:, 시작[i]:시작[i + 1]].astype(np.int64), -1)
            같은집합 += int((수 == 우).all(-1).sum())
            겹침 += int(sum((np.isin(우[a, b], 수[a, b])).sum() for a in range(수.shape[0]) for b in range(수.shape[1])))
            칸 += 수.shape[0] * 수.shape[1]
        if 칸:
            결과["라우팅"] = {"같은 집합 %": 100 * 같은집합 / 칸, "겹침 %": 100 * 겹침 / (칸 * 우리.shape[2]), "(층, 행)": 칸}
            print("라우팅:", 결과["라우팅"], flush=True)
    모든 = {"계약": [], "라마": [], "같음계약": 0, "같음라마": 0, "행": 0}
    for i in range(len(토큰들)):
        pc = 로그소프트맥스(로짓[시작[i]:시작[i + 1]])
        pm = np.load(os.path.join(폴더, f"수학{i}.npy")).astype(np.float64)
        pr = 로그소프트맥스(np.load(os.path.join(폴더, f"라마{i}.npy")))
        em = np.exp(pm)
        kc, kr = (em * (pm - pc)).sum(-1), (em * (pm - pr)).sum(-1)
        같c, 같r = int((pm.argmax(1) == pc.argmax(1)).sum()), int((pm.argmax(1) == pr.argmax(1)).sum())
        한 = {"토큰": len(토큰들[i]), "KL(수학‖계약)": float(kc.mean()), "최대": float(kc.max()), "1위 일치(계약)": 같c,
             "KL(수학‖llama.cpp)": float(kr.mean()), "1위 일치(llama.cpp)": 같r}
        결과["글"].append(한)
        모든["계약"].append(kc)
        모든["라마"].append(kr)
        모든["같음계약"] += 같c
        모든["같음라마"] += 같r
        모든["행"] += len(토큰들[i])
        print(f"글 {i}: {한}", flush=True)
    kc, kr = np.concatenate(모든["계약"]), np.concatenate(모든["라마"])
    결과["모두"] = {"KL(수학‖계약)": float(kc.mean()), "1위 일치(계약) %": 100 * 모든["같음계약"] / 모든["행"],
                  "KL(수학‖llama.cpp)": float(kr.mean()), "1위 일치(llama.cpp) %": 100 * 모든["같음라마"] / 모든["행"],
                  "비(계약 / llama.cpp)": float(kc.mean() / kr.mean())}
    print("모두:", 결과["모두"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
