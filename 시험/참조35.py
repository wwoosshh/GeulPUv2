#!/usr/bin/env python3
"""0.12.0 12-3e — Qwen3.5(GGUF 구조 qwen35)의 계약 CPU 참조: 지금의 계약(정수 블록 행렬곱 · 어텐션 · RMSNorm · RoPE · SwiGLU)에 계약 4.9(출력
게이트) · 4.10(선형 어텐션 GDN — 계약/정수계약.py)을 더한 순전파. 층의 뜻은 시험/qwen35수학.py(12-3a — llama.cpp 의 src/models/qwen35.cpp 와 같음)
그대로이고, 연산마다 계약의 차례 · 반올림이다.

  python -I 시험/참조35.py <qwen35 gguf> --수학 <폴더> <출력.json> [--빠른]

--수학 <폴더>: 12-3a 의 시험/qwen35수학.py 가 남긴 토큰.json · 수학{i}.npy(fp64 의 이상적인 모델 — 로그 확률) · 라마{i}.npy(llama.cpp 로짓)와
견준다 — 계약의 손실 KL(수학 ‖ 계약) · 1위 일치, 그리고 같은 자리의 KL(수학 ‖ llama.cpp). --빠른: 빠른 계약 F1(선형층의 활성값 자리 하나 —
어텐션 · GDN 은 그대로).
엔진(12-3e 의 커널 · 엔진)이 생기면 그 출력 폴더와 비트까지 견주는 길을 더한다(참조7B.py 의 꼴).
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
    k = "qwen35."
    assert m["general.architecture"] == "qwen35", m["general.architecture"]
    s = {"층": m[k + "block_count"], "너비": m[k + "embedding_length"], "머리": m[k + "attention.head_count"],
         "KV머리": m[k + "attention.head_count_kv"], "헤드": m[k + "attention.key_length"], "eps": float(m[k + "attention.layer_norm_rms_epsilon"]),
         "돌림": m[k + "rope.dimension_count"], "θ": float(m[k + "rope.freq_base"]), "간격": m[k + "full_attention_interval"],
         "d": m[k + "ssm.state_size"], "Hk": m[k + "ssm.group_count"], "Hv": m[k + "ssm.time_step_rank"], "합성곱": m[k + "ssm.conv_kernel"]}
    assert m[k + "attention.value_length"] == s["헤드"] and s["합성곱"] == 4 and s["d"] == 128 and s["Hv"] % s["Hk"] == 0, s
    assert m.get(k + "rope.scaling.type", "none") == "none", "RoPE 늘이기는 받지 않는다"
    return s


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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if "--빠른" in sys.argv:
        R7.선형자리수 = 1
    인자 = [a for a in sys.argv[1:] if not a.startswith("--")]
    gguf = 인자[0]
    폴더 = sys.argv[sys.argv.index("--수학") + 1]
    출력 = [a for a in 인자[1:] if a != 폴더][0]
    g = GR.GGUF(gguf)
    토큰들 = json.load(open(os.path.join(폴더, "토큰.json"), encoding="utf-8"))["토큰"]
    t0 = time.time()
    로짓, 시작 = 앞으로(g, [np.asarray(t, np.int64) for t in 토큰들])
    결과 = {"gguf": os.path.basename(gguf), "모드": "F1" if R7.선형자리수 == 1 else "정밀", "초": round(time.time() - t0), "글": []}
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
