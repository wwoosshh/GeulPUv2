#!/usr/bin/env python3
"""0.12.0 12-3h-1 — Gemma 4(GGUF 구조 gemma4)의 계약 CPU 참조: 지금의 계약(정수 블록 행렬곱 · 어텐션 · RMSNorm · RoPE)에 계약 4.12(Gemma 4 —
GeGLU · 소프트캡 · 슬라이딩 윈도 · 가중치 없는 v 정규화 · 배율 1 의 어텐션 · 임베딩 × √너비 · 층 배율 — 계약/정수계약.py)를 더한 순전파. 층의 뜻은
시험/gemma4수학.py(llama.cpp 의 src/models/gemma4.cpp 와 같음) 그대로이고, 연산마다 계약의 차례 · 반올림이다. 지금은 dense(12B · 31B)만.

  python -I 시험/참조gemma4.py <gemma4 gguf> --수학 <폴더> <출력.json> [--빠른] [--저장된]
  python -I 시험/참조gemma4.py <gemma4 gguf> --엔진 <시험_7B 출력 폴더> <출력.json> [--입력 <시험_7B 의 입력 파일>]

--수학 <폴더>: 시험/gemma4수학.py 의 토큰.json · 수학{i}.npy(fp64 의 이상적인 모델) · 라마{i}.npy(llama.cpp), 그리고 있으면 채팅 확인의
채팅토큰.json · 채팅수학{i}.npy · 채팅라마{i}.npy 와 견준다 — 계약의 손실 KL(수학 ‖ 계약) · 1위 일치, 같은 자리의 KL(수학 ‖ llama.cpp).
채팅(답을 예측하는 행)이 먼저, 글(모든 행)이 다음. 계약의 로짓(소프트캡 뒤 — fp32 그대로)을 {채팅|글}계약{i}.npy 로 남기고,
--저장된 은 순전파 없이 그 파일로 견준다(비교만 다시).
--엔진 <폴더>(0.13.0 12-3h-4): 시험_7B 의 출력과 계약의 순전파를 비트까지 — 잡기시작 0 의 경우는 모든 행(경우{i}_토큰.bin · 경우{i}_로짓.bin
— 참조35.py 의 꼴), 잡기시작 뒤의 행만 남긴 경우(경우{i}_뒤로짓.bin — 1,100 토큰처럼 슬라이딩 윈도 1,024 를 넘는 위치)는 --입력 의 프롬프트와
요약.txt 의 생성 토큰으로 토큰열을 다시 짓고 행 [잡기시작, 끝) 만. 로짓은 견줄 행만 셈한다(행마다 따로라 값은 같다).
"""
import json
import os
import struct
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
import 참조35 as R35        # noqa: E402 — 임베딩

f32 = np.float32


def 설정(g):
    m = g.meta
    assert m["general.architecture"] == "gemma4", m["general.architecture"]
    k = lambda x, d=None: m.get("gemma4." + x, d)                   # noqa: E731
    assert not k("embedding_length_per_layer_input", 0) and not k("attention.shared_kv_layers", 0), "PLE · KV 공유는 아직"
    assert "blk.0.ffn_gate_inp.weight" not in g.tensors, "MoE 는 아직"
    L = k("block_count")
    hkv = k("attention.head_count_kv")
    s = {"층": L, "너비": k("embedding_length"), "머리": k("attention.head_count"),
         "KV머리": list(hkv) if isinstance(hkv, (list, tuple)) else [hkv] * L, "윈도층": list(k("attention.sliding_window_pattern")),
         "헤드": {True: k("attention.key_length_swa"), False: k("attention.key_length")},
         "돌림": {True: k("rope.dimension_count_swa"), False: k("rope.dimension_count")},
         "θ": {True: float(k("rope.freq_base_swa")), False: float(k("rope.freq_base"))},
         "창": k("attention.sliding_window"), "eps": float(k("attention.layer_norm_rms_epsilon")),
         "캡": float(k("final_logit_softcapping", 0.0) or 0.0)}
    for w in (True, False):
        assert s["돌림"][w] == s["헤드"][w], "Gemma 4 의 RoPE 는 헤드 전체(부분 회전은 rope_freqs 로)"
    return s


def 앞으로(g, 경우, 말=True, 행=None):
    """계약의 순전파 — 토큰열들(경우마다 위치 0 부터)의 모든 행 로짓 [Σ 행][어휘](소프트캡 뒤)과 경우의 시작 행. 행 = 경우마다 (a, b) 면
    로짓은 그 행 [a, b) 만(시작 행도 그 행들의 것)."""
    s = 설정(g)
    NL, D, H, eps = s["층"], s["너비"], s["머리"], s["eps"]
    x = C.곱(R35.임베딩(g, 경우), C.제곱근(f32(D)))               # 계약 4.12 — 임베딩 × fp32(√너비)(√ 는 IEEE 바른 반올림)
    시작 = np.cumsum([0] + [len(S) for S in 경우])
    최대 = max(64, int(max(len(S) for S in 경우)))
    인자 = g.풀기("rope_freqs.weight") if "rope_freqs.weight" in g.tensors else None
    표 = {True: C.rope표(최대, s["θ"][True], hd=s["돌림"][True]),
         False: C.rope표(최대, s["θ"][False], hd=s["돌림"][False], 인자=인자)}

    def z0(n):
        return np.zeros(n, f32)
    for l in range(NL):
        tl = time.time()
        p = f"blk.{l}."
        w = s["윈도층"][l]
        hd, Hkv = s["헤드"][w], s["KV머리"][l]
        G_ = H // Hkv
        cos, sin = 표[w]
        h = C.rms정규화(x, g.풀기(p + "attn_norm.weight"), eps)
        q = R7.곱(h, R7.가중(g, p + "attn_q.weight"), z0(H * hd))
        k0 = R7.곱(h, R7.가중(g, p + "attn_k.weight"), z0(Hkv * hd))
        v0 = R7.곱(h, R7.가중(g, p + "attn_v.weight"), z0(Hkv * hd)) if p + "attn_v.weight" in g.tensors else k0
        q = C.머리정규화(q, g.풀기(p + "attn_q_norm.weight"), eps, hd)
        kk = C.머리정규화(k0, g.풀기(p + "attn_k_norm.weight"), eps, hd)
        vv = C.머리정규화(v0, np.ones(hd, f32), eps, hd)             # 계약 4.12 — v 는 가중치 없이(× 1 은 정확)
        ys = []
        for si in range(len(경우)):
            a0, a1 = 시작[si], 시작[si + 1]
            위치 = np.arange(a1 - a0)
            qh = [C.rope(q[a0:a1, hd * a:hd * a + hd], cos, sin, 위치) for a in range(H)]
            kh = [C.rope(kk[a0:a1, hd * a:hd * a + hd], cos, sin, 위치) for a in range(Hkv)]
            vh = [np.ascontiguousarray(vv[a0:a1, hd * a:hd * a + hd]) for a in range(Hkv)]
            ys.append(np.concatenate([C.어텐션(qh[a], kh[a // G_], vh[a // G_], 눈금=1.0, 창=s["창"] if w else None)
                                      for a in range(H)], 1))
        o = R7.곱(np.concatenate(ys, 0), R7.가중(g, p + "attn_output.weight"), z0(D))
        x = C.더(x, C.rms정규화(o, g.풀기(p + "post_attention_norm.weight"), eps))
        h = C.rms정규화(x, g.풀기(p + "ffn_norm.weight"), eps)
        wg = R7.가중(g, p + "ffn_gate.weight")
        F = wg[1].shape[0]
        gg = R7.곱(h, wg, z0(F))
        del wg
        uu = R7.곱(h, R7.가중(g, p + "ffn_up.weight"), z0(F))
        f = R7.곱(C.겔루(gg, uu), R7.가중(g, p + "ffn_down.weight"), z0(D))
        x = C.더(x, C.rms정규화(f, g.풀기(p + "post_ffw_norm.weight"), eps))
        if p + "layer_output_scale.weight" in g.tensors:             # 계약 4.12 — 층 배율(잔차 전체에 곱 하나)
            x = C.곱(x, f32(g.풀기(p + "layer_output_scale.weight")[0]))
        if 말:
            print(f"층 {l}({'윈도' if w else '전체'}): {time.time() - tl:.1f} 초", flush=True)
    if 행 is not None:
        x = np.concatenate([x[시작[si] + a:시작[si] + b] for si, (a, b) in enumerate(행)], 0)
        시작 = np.cumsum([0] + [b - a for a, b in 행])
    h = C.rms정규화(x, g.풀기("output_norm.weight"), eps)
    wh = R7.가중(g, "output.weight" if "output.weight" in g.tensors else "token_embd.weight")
    로짓 = R7.곱(h, wh, z0(wh[1].shape[0]))
    if s["캡"]:
        로짓 = C.소프트캡(로짓, s["캡"])
    return 로짓, 시작


def 입력경우들(경로):
    """시험_7B 의 입력 파일 — [최대길이][로짓행][경우 수], 경우마다 [p][n][잡기시작][토큰 p 개](8 바이트 정수) → [(p, n, 잡기, 프롬프트)]"""
    b = open(경로, "rb").read()
    _, _, k = struct.unpack_from("<qqq", b, 0)
    o, 경우 = 24, []
    for _ in range(k):
        p, n, 잡기 = struct.unpack_from("<qqq", b, o)
        경우.append((p, n, 잡기, np.frombuffer(b, np.int64, p, o + 24).copy()))
        o += 24 + 8 * p
    return 경우


def 엔진대조(gguf, 폴더, 출력, 입력=None):
    """12-3h-4 — 시험_7B 의 출력(경우마다 토큰열과 로짓)을 계약의 순전파와 비트까지."""
    g = GR.GGUF(gguf)
    입력경우 = 입력경우들(입력) if 입력 else None
    생성 = {}
    요약 = os.path.join(폴더, "요약.txt")
    if os.path.exists(요약):
        for 줄 in open(요약, encoding="utf-8"):
            w = 줄.split()
            if len(w) > 2 and w[0] == "경우" and w[2] == "토큰":
                생성[int(w[1])] = [int(t) for t in w[3:]]
    경우, 구간, 파일, 번호 = [], [], [], []
    i = 0
    while True:
        tp, bp = os.path.join(폴더, f"경우{i}_토큰.bin"), os.path.join(폴더, f"경우{i}_뒤로짓.bin")
        if os.path.exists(tp):
            S = np.fromfile(tp, np.int64)
            경우.append(S), 구간.append((0, len(S))), 파일.append(os.path.join(폴더, f"경우{i}_로짓.bin")), 번호.append(i)
        elif os.path.exists(bp) and 입력경우 is not None:
            p, n, 잡기, 프롬프트 = 입력경우[i]
            S = np.concatenate([프롬프트, np.asarray(생성[i][:n - 1], np.int64)])
            경우.append(S), 구간.append((잡기, len(S))), 파일.append(bp), 번호.append(i)
        elif os.path.exists(bp):
            print(f"경우 {i}: 잡기시작 뒤의 로짓만 있다 — --입력 이 없어 건너뜀", flush=True)
        else:
            break
        i += 1
    print("경우", [(번호[k], len(S), 구간[k]) for k, S in enumerate(경우)], flush=True)
    t0 = time.time()
    로짓, 시작 = 앞으로(g, 경우, 행=구간)
    V = 로짓.shape[1]
    결과 = {"gguf": os.path.basename(gguf), "모드": "F1" if R7.선형자리수 == 1 else "정밀", "경우": []}
    for k in range(len(경우)):
        a, b = 구간[k]
        A = np.fromfile(파일[k], f32).reshape(b - a, V)
        R = 로짓[시작[k]:시작[k + 1]]
        한 = {"경우": 번호[k], "토큰": len(경우[k]), "행": [int(a), int(b)], "다른 칸": R7.다른칸(A, R), "칸": int(A.size),
             "행마다 다른 칸": [R7.다른칸(A[j], R[j]) for j in range(b - a)]}
        결과["경우"].append(한)
        print(f"경우 {번호[k]}: 행 {a} … {b - 1} — 다른 칸 {한['다른 칸']} / {한['칸']}", flush=True)
    결과["CPU 참조 초"] = round(time.time() - t0, 1)
    결과["모두 0"] = bool(결과["경우"]) and all(c["다른 칸"] == 0 for c in 결과["경우"])
    print("모두 0:", 결과["모두 0"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def 로그소프트맥스(x):
    x = np.asarray(x, np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if "--빠른" in sys.argv:
        R7.선형자리수 = 1
    저장된 = "--저장된" in sys.argv
    인자 = [a for a in sys.argv[1:] if not a.startswith("--")]
    gguf = 인자[0]
    if "--엔진" in sys.argv:
        폴더e = sys.argv[sys.argv.index("--엔진") + 1]
        입력 = sys.argv[sys.argv.index("--입력") + 1] if "--입력" in sys.argv else None
        엔진대조(gguf, 폴더e, [a for a in 인자[1:] if a not in (폴더e, 입력)][0], 입력)
        return
    폴더 = sys.argv[sys.argv.index("--수학") + 1]
    출력 = [a for a in 인자[1:] if a != 폴더][0]
    g = GR.GGUF(gguf)
    묶음들 = []                                                      # (이름, 토큰열들, 수학 파일 앞머리, llama.cpp 파일 앞머리, 견줄 행의 [시작, 끝)들)
    경로 = os.path.join(폴더, "채팅토큰.json")
    if os.path.exists(경로):                                         # 프롬프트 P 토큰 + 답 — 답을 예측하는 행은 P − 1 … 끝 − 2
        경우 = json.load(open(경로, encoding="utf-8"))["경우"]
        묶음들.append(("채팅", [e[0] for e in 경우], "채팅수학", "채팅라마", [(e[1] - 1, len(e[0]) - 1) for e in 경우]))
    토큰들 = json.load(open(os.path.join(폴더, "토큰.json"), encoding="utf-8"))["토큰"]
    묶음들.append(("글", 토큰들, "수학", "라마", [(0, len(t)) for t in 토큰들]))
    결과 = {"gguf": os.path.basename(gguf), "모드": "F1" if R7.선형자리수 == 1 else "정밀", "묶음": {}}
    for 이름, 토큰열들, 수학, 라마, 구간들 in 묶음들:
        t0 = time.time()
        파일 = [os.path.join(폴더, f"{이름}계약{i}.npy") for i in range(len(토큰열들))]
        if 저장된:
            로짓들 = [np.load(p) for p in 파일]
        else:
            로짓, 시작 = 앞으로(g, [np.asarray(t, np.int64) for t in 토큰열들])
            로짓들 = [로짓[시작[i]:시작[i + 1]] for i in range(len(토큰열들))]
            for p, z in zip(파일, 로짓들):
                np.save(p, z)
        모든 = {"계약": [], "라마": [], "같음계약": 0, "같음라마": 0, "행": 0}
        글들 = []
        for i in range(len(토큰열들)):
            pc = 로그소프트맥스(로짓들[i])
            pm = np.load(os.path.join(폴더, f"{수학}{i}.npy")).astype(np.float64)
            pr = 로그소프트맥스(np.load(os.path.join(폴더, f"{라마}{i}.npy")))
            a, b = 구간들[i]
            pm, pc, pr = pm[a:b], pc[a:b], pr[a:b]
            em = np.exp(pm)
            kc, kr = (em * (pm - pc)).sum(-1), (em * (pm - pr)).sum(-1)
            같c, 같r = int((pm.argmax(1) == pc.argmax(1)).sum()), int((pm.argmax(1) == pr.argmax(1)).sum())
            한 = {"행": len(kc), "KL(수학‖계약)": float(kc.mean()), "최대": float(kc.max()), "1위 일치(계약)": 같c,
                 "KL(수학‖llama.cpp)": float(kr.mean()), "1위 일치(llama.cpp)": 같r}
            글들.append(한)
            모든["계약"].append(kc)
            모든["라마"].append(kr)
            모든["같음계약"] += 같c
            모든["같음라마"] += 같r
            모든["행"] += len(kc)
            print(f"{이름} {i}: {한}", flush=True)
        kc, kr = np.concatenate(모든["계약"]), np.concatenate(모든["라마"])
        결과["묶음"][이름] = {"글": 글들, "초": round(time.time() - t0),
                          "모두": {"KL(수학‖계약)": float(kc.mean()), "1위 일치(계약) %": 100 * 모든["같음계약"] / 모든["행"],
                                 "KL(수학‖llama.cpp)": float(kr.mean()), "1위 일치(llama.cpp) %": 100 * 모든["같음라마"] / 모든["행"],
                                 "비(계약 / llama.cpp)": float(kc.mean() / kr.mean())}}
        print(f"{이름} 모두:", 결과["묶음"][이름]["모두"], flush=True)
        json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
