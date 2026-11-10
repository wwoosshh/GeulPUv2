#!/usr/bin/env python3
"""0.24.0 24-2 — deepseek2(GGUF 구조 deepseek2 — MLA 어텐션 + MoE)의 계약 CPU 참조: 지금의 계약(정수 블록 행렬곱 · 어텐션 · RMSNorm · SwiGLU ·
4.11 의 MoE)에 계약 4.13(흡수한 MLA · NORM RoPE · 헤드마다의 행렬곱 · 시그모이드 라우팅 — 계약/정수계약.py)을 더한 순전파. 층의 뜻은
시험/deepseek2수학.py(llama.cpp 의 src/models/deepseek2.cpp 와 같음) 그대로이고, 연산마다 계약의 차례 · 반올림이다.

  python -I 시험/참조deepseek2.py <gguf> --수학 <폴더> <출력.json> [--저장된]
  python -I 시험/참조deepseek2.py <gguf> --엔진 <시험_7B 출력 폴더> <출력.json> [--입력 <시험_7B 의 입력 파일>]
  [--일꾼 N] — 리눅스(fork)에서 MoE 의 전문가를 프로세스 N 개로 나눠 셈(값은 같다 — 전문가마다 따로인 계산). 윈도는 늘 하나씩

--수학 <폴더>: 시험/deepseek2수학.py 가 남긴 토큰.json · 수학{i}.npy(fp64 의 이상적인 모델 — 로그 확률) · 라마{i}.npy(llama.cpp 로짓) ·
라우팅.npz 와 비교한다 — 계약의 손실 KL(수학 ‖ 계약) · 1위 일치, 같은 자리의 KL(수학 ‖ llama.cpp), 층마다 고른 전문가가 수학 참조와 같은 행의
비율. 계약의 로짓(fp32 그대로)을 계약{i}.npy 로 남기고, --저장된 은 순전파 없이 그 파일로 비교한다.
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
import 참조35 as R35        # noqa: E402 — 임베딩 · 라우터로짓 · 가중전문가
import deepseek2수학 as M   # noqa: E402 — 설정

_g = None                   # 전문가 일꾼(fork)이 물려받는 GGUF
_풀 = None                  # --일꾼 N 의 프로세스 풀(리눅스만)


def _풀열기(g):
    """--일꾼 N(리눅스)이면 GGUF 를 연 뒤에 fork 풀을 연다 — 일꾼은 _g 를 물려받는다"""
    global _g, _풀
    _g = g
    if "--일꾼" in sys.argv and os.name == "posix":
        import multiprocessing
        _풀 = multiprocessing.get_context("fork").Pool(int(sys.argv[sys.argv.index("--일꾼") + 1]))


def _전문가하나(일):
    """전문가 e 의 SwiGLU(4.11 5) — 그 전문가를 고른 행들 he 만"""
    p, e, he = 일
    wg = 가중헤드(_g, p + "ffn_gate_exps.weight", e)
    Fe = wg[1].shape[0]
    gg = R7.곱(he, wg, np.zeros(Fe, f32))
    uu = R7.곱(he, 가중헤드(_g, p + "ffn_up_exps.weight", e), np.zeros(Fe, f32))
    return R7.곱(C.swiglu(gg, uu), 가중헤드(_g, p + "ffn_down_exps.weight", e), np.zeros(he.shape[1], f32))

f32 = np.float32
라우팅기록 = {}             # 층 → [Σ 행][k] 고른 전문가(계약)


def 가중헤드(g, name, i):
    """3 차원 텐서 [헤드 · 전문가][N][K] 의 조각 i 를 참조7B.가중 의 꼴로 — Q8_0 · Q5_0(계약 4.13 — Q8_0 의 식) · Q4_0 은 (형식, q, d, None),
    k-양자화는 풀기K, F16 · BF16 · F32 는 16비트 블록(계약 1.3)."""
    t, dims, raw = g.원(name)
    K, N = dims[0], dims[1]
    vb, bb = GR.블록[t]
    한 = N * (K // vb * bb)
    r = np.asarray(raw)[i * 한:(i + 1) * 한]
    if t in (GR.Q8_0, GR.Q5_0, GR.Q4_0):
        q, d = R7.q8바이트(t, r, N, K)
        return ("Q4_0" if t == GR.Q4_0 else "Q8_0"), q, d, None
    if t in (GR.F16, GR.BF16, GR.F32):
        n, E = C.가중치16(g.풀기(name).reshape(-1, N, K)[i])
        return "W16", n, E, None
    return C.풀기K(t, r, N, K)


def MoE(g, p, h, s):
    """계약 4.13 8 — 라우터(4.11 1) → 시그모이드 라우팅(σ · p + b 로 상위 k · 정규화 · 배율) → 전문가 SwiGLU(4.11 5) → 순위 차례의 합(4.11 6)
    → 공유 전문가를 더하기 하나로."""
    T, D = h.shape
    l = R35.라우터로짓(h, g, p + "ffn_gate_inp.weight", s["전문가"])
    b = g.풀기(p + "exp_probs_b.bias") if p + "exp_probs_b.bias" in g.tensors else None
    고른, w = C.시그모이드전문가(l, b, s["고름"], s["정규화"], s["배율"])
    라우팅기록[int(p.split(".")[1])] = 고른.astype(np.int16)
    ys = np.zeros((s["고름"], T, D), f32)
    global _g
    _g = g
    일들 = []
    for e in np.unique(고른):
        행, 순위 = np.nonzero(고른 == e)
        일들.append((행, 순위, (p, int(e), np.ascontiguousarray(h[행]))))
    결과들 = _풀.imap(_전문가하나, [x[2] for x in 일들]) if _풀 is not None else map(_전문가하나, [x[2] for x in 일들])
    for (행, 순위, _), y in zip(일들, 결과들):
        ys[순위, 행] = y
    o = C.전문가합(w, ys)
    wg = R7.가중(g, p + "ffn_gate_shexp.weight")
    Fs = wg[1].shape[0]
    gs = R7.곱(h, wg, np.zeros(Fs, f32))
    us = R7.곱(h, R7.가중(g, p + "ffn_up_shexp.weight"), np.zeros(Fs, f32))
    return C.더(o, R7.곱(C.swiglu(gs, us), R7.가중(g, p + "ffn_down_shexp.weight"), np.zeros(D, f32)))


def 앞으로(g, 경우, 말=True, 행=None):
    """계약의 순전파 — 토큰열들(경우마다 위치 0 부터)의 모든 행 로짓 [Σ 행][어휘]과 경우의 시작 행. 행 = 경우마다 (a, b) 면 로짓은 그 행
    [a, b) 만(참조gemma4.py 와 같은 꼴)."""
    s = M.설정(g)
    NL, D, H, eps = s["층"], s["너비"], s["머리"], s["eps"]
    Cc, Rr, N, V = s["C"], s["R"], s["N"], s["V"]
    x = R35.임베딩(g, 경우)
    시작 = np.cumsum([0] + [len(S) for S in 경우])
    cos, sin = C.rope표(max(64, int(max(len(S) for S in 경우))), s["θ"], hd=Rr)
    눈금 = f32(1.0 / np.sqrt(np.float64(N + Rr)))                     # 계약 4.13 6 — fp32(1/√(N + R))(GLM 은 1/16 으로 정확)

    def z0(n):
        return np.zeros(n, f32)
    for l in range(NL):
        tl = time.time()
        p = f"blk.{l}."
        h = C.rms정규화(x, g.풀기(p + "attn_norm.weight"), eps)
        if s["qa"]:                                                  # 4.13 1 — q 의 LoRA(RMSNorm 사이)
            a = R7.곱(h, R7.가중(g, p + "attn_q_a.weight"), z0(s["qa"]))
            a = C.rms정규화(a, g.풀기(p + "attn_q_a_norm.weight"), eps)
            q = R7.곱(a, R7.가중(g, p + "attn_q_b.weight"), z0(H * (N + Rr)))
        else:
            q = R7.곱(h, R7.가중(g, p + "attn_q.weight"), z0(H * (N + Rr)))
        kv = R7.곱(h, R7.가중(g, p + "attn_kv_a_mqa.weight"), z0(Cc + Rr))
        c = C.rms정규화(np.ascontiguousarray(kv[:, :Cc]), g.풀기(p + "attn_kv_a_norm.weight"), eps)     # 4.13 2
        wk = [가중헤드(g, p + "attn_k_b.weight", i) for i in range(H)]
        wv = [가중헤드(g, p + "attn_v_b.weight", i) for i in range(H)]
        ys = []
        for si in range(len(경우)):
            a0, a1 = 시작[si], 시작[si + 1]
            위치 = np.arange(a1 - a0)
            kpe = C.rope이웃(kv[a0:a1, Cc:], cos, sin, 위치)              # 4.13 3
            K = np.ascontiguousarray(np.concatenate([c[a0:a1], kpe], 1))   # 4.13 5 — 키 [c′ | k′_pe]
            Vv = np.ascontiguousarray(c[a0:a1])                          # 값 c′(키의 앞 C 원소 — 같은 16비트 형식)
            os_ = []
            for i in range(H):
                qi = q[a0:a1, i * (N + Rr):(i + 1) * (N + Rr)]
                qabs = R7.곱(np.ascontiguousarray(qi[:, :N]), wk[i], z0(Cc))     # 4.13 4 — 흡수
                qpe = C.rope이웃(qi[:, N:], cos, sin, 위치)
                o = C.어텐션(np.concatenate([qabs, qpe], 1), K, Vv, 눈금=눈금)   # 4.13 6 — MQA(KV 헤드 하나)
                os_.append(R7.곱(o, wv[i], z0(V)))                       # 4.13 7 — 값 풀기
            ys.append(np.concatenate(os_, 1))
        x = C.더(x, R7.곱(np.concatenate(ys, 0), R7.가중(g, p + "attn_output.weight"), z0(D)))
        h2 = C.rms정규화(x, g.풀기(p + "ffn_norm.weight"), eps)
        if l < s["dense"]:                                           # 4.13 8 — 앞의 dense 층은 SwiGLU(4.3)
            wg = R7.가중(g, p + "ffn_gate.weight")
            F = wg[1].shape[0]
            gg = R7.곱(h2, wg, z0(F))
            del wg
            uu = R7.곱(h2, R7.가중(g, p + "ffn_up.weight"), z0(F))
            f = R7.곱(C.swiglu(gg, uu), R7.가중(g, p + "ffn_down.weight"), z0(D))
        else:
            f = MoE(g, p, h2, s)
        x = C.더(x, f)
        if 말:
            print(f"층 {l}: {time.time() - tl:.1f} 초", flush=True)
    if 행 is not None:
        x = np.concatenate([x[시작[si] + a:시작[si] + b] for si, (a, b) in enumerate(행)], 0)
        시작 = np.cumsum([0] + [b - a for a, b in 행])
    h = C.rms정규화(x, g.풀기("output_norm.weight"), eps)
    wh = R7.가중(g, "output.weight" if "output.weight" in g.tensors else "token_embd.weight")
    return R7.곱(h, wh, z0(wh[1].shape[0])), 시작


def 로그소프트맥스(x):
    x = np.asarray(x, np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def 엔진대조(gguf, 폴더, 출력, 입력=None):
    """시험_7B 의 출력(경우마다 토큰열과 로짓)을 계약의 순전파와 비트까지 — 참조gemma4.py 의 엔진대조 와 같은 꼴(잡기시작 뒤의 행만 남긴
    경우는 --입력 의 프롬프트와 요약.txt 의 생성 토큰으로 토큰열을 다시 짓고 행 [잡기시작, 끝) 만)."""
    import 참조gemma4 as RG                                       # noqa: E402 — 입력경우들
    g = GR.GGUF(gguf)
    _풀열기(g)
    입력경우 = RG.입력경우들(입력) if 입력 else None
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
    결과 = {"gguf": os.path.basename(gguf), "모드": "정밀", "경우": []}
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    저장된 = "--저장된" in sys.argv
    인자 = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--일꾼"]
    gguf = 인자[0]
    if "--엔진" in sys.argv:
        폴더e = sys.argv[sys.argv.index("--엔진") + 1]
        입력 = sys.argv[sys.argv.index("--입력") + 1] if "--입력" in sys.argv else None
        엔진대조(gguf, 폴더e, [a for a in 인자[1:] if a not in (폴더e, 입력)][0], 입력)
        return
    폴더 = sys.argv[sys.argv.index("--수학") + 1]
    출력 = [a for a in 인자[1:] if a != 폴더][0]
    g = GR.GGUF(gguf)
    _풀열기(g)
    토큰들 = json.load(open(os.path.join(폴더, "토큰.json"), encoding="utf-8"))["토큰"]
    파일 = [os.path.join(폴더, f"계약{i}.npy") for i in range(len(토큰들))]
    t0 = time.time()
    if 저장된:
        로짓들 = [np.load(p) for p in 파일]
    else:
        로짓, 시작 = 앞으로(g, [np.asarray(t, np.int64) for t in 토큰들])
        로짓들 = [로짓[시작[i]:시작[i + 1]] for i in range(len(토큰들))]
        for p, z in zip(파일, 로짓들):
            np.save(p, z)
    결과 = {"gguf": os.path.basename(gguf), "모드": "정밀", "글": [], "초": round(time.time() - t0)}
    모든 = {"계약": [], "라마": [], "같음계약": 0, "같음라마": 0, "행": 0}
    for i in range(len(토큰들)):
        pc = 로그소프트맥스(로짓들[i])
        pm = np.load(os.path.join(폴더, f"수학{i}.npy")).astype(np.float64)
        pr = 로그소프트맥스(np.load(os.path.join(폴더, f"라마{i}.npy")))
        em = np.exp(pm)
        kc, kr = (em * (pm - pc)).sum(-1), (em * (pm - pr)).sum(-1)
        같c, 같r = int((pm.argmax(1) == pc.argmax(1)).sum()), int((pm.argmax(1) == pr.argmax(1)).sum())
        한 = {"행": len(kc), "KL(수학‖계약)": float(kc.mean()), "최대": float(kc.max()), "1위 일치(계약)": 같c,
             "KL(수학‖llama.cpp)": float(kr.mean()), "1위 일치(llama.cpp)": 같r}
        결과["글"].append(한)
        모든["계약"].append(kc)
        모든["라마"].append(kr)
        모든["같음계약"] += 같c
        모든["같음라마"] += 같r
        모든["행"] += len(kc)
        print(f"글 {i}: {한}", flush=True)
    kc, kr = np.concatenate(모든["계약"]), np.concatenate(모든["라마"])
    결과["모두"] = {"KL(수학‖계약)": float(kc.mean()), "1위 일치(계약) %": 100 * 모든["같음계약"] / 모든["행"],
                  "KL(수학‖llama.cpp)": float(kr.mean()), "1위 일치(llama.cpp) %": 100 * 모든["같음라마"] / 모든["행"],
                  "비(계약 / llama.cpp)": float(kc.mean() / kr.mean())}
    경로 = os.path.join(폴더, "라우팅.npz")
    if 라우팅기록 and os.path.exists(경로):                           # 층마다 고른 전문가 집합이 수학 참조와 같은 행의 비율
        수 = np.load(경로)
        같, 모 = 0, 0
        for l, 고른 in 라우팅기록.items():
            a, b2 = np.sort(고른, 1), np.sort(수[str(l)], 1)
            같 += int((a == b2).all(1).sum())
            모 += len(a)
        결과["모두"]["라우팅이 수학 참조와 같은 행 %"] = 100 * 같 / 모
    print("모두:", 결과["모두"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
