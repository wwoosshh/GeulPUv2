#!/usr/bin/env python3
"""2단계 판정 B · 2b단계 판정 B″ — 숨은 손실의 견줌: 정밀 모드(계약 자리 둘 — fp64 에 가까운 기준)의 다음 토큰 분포를 기준으로, 빠른 모드의
변형들(F1 · F1+F1A 흉내 …)과 llama.cpp(최신 공식 판 · 기본 설정 — KV f16, 참고로 q8_0 KV)가 얼마나 벗어나는가.
GeulPU 39단계의 시험/독립39.py(엔진 대 llama.cpp 의 모든 행 로짓 대조)에서 왔다.

  python -I 시험/품질.py 입력 <gguf> <llama 폴더> <작업폴더> [--글 <경로[@바이트]> …] [--긴 <경로[@바이트]>]
      글마다 앞 2,048 토큰 — llama.cpp 의 llama_tokenize(BOS 는 모델대로 · 특수 토큰은 풀지 않음 — llama-perplexity 와 같다). 바이트 자리를
      주면 그 뒤 첫 줄바꿈 다음부터. 기본 글은 둘(wikitext-2 test 앞 · 연구자안내.md — 2단계). --긴: 8,192 토큰 창 하나(행 7,168 … 8,191 의
      로짓 — 2b단계). → 작업폴더/토큰.json · 입력.bin(엔진/시험_7B.gl 의 입력 — 로짓행 1024, 최대길이 2048 · 긴 창이 있으면 8192): 글마다
      경우 둘 — (프롬프트 1024, 잡기 0) → 행 0 … 1023, (프롬프트 2048, 잡기 1024) → 행 1024 … 2047, 긴 창은 (8192, 잡기 7168).
  python -I 시험/품질.py 라마 <gguf> <llama 폴더> <작업폴더> [--kv q8_0]
      llama.cpp C API(재기/라마.py): n_ctx = n_batch = 2048(긴 창이 있으면 8192) · n_ubatch 512 · 플래시 어텐션 자동 · KV f16(--kv q8_0 이면
      q8_0 — 참고값) · 모든 층 GPU → 라마{i}.npy · 라마긴.npy (q8_0 KV 는 라마q8_{i}.npy · 라마q8_긴.npy)
  python -I 시험/품질.py 견주기 <작업폴더> <정밀 시험_7B 출력> <결과.json> <이름=시험_7B 출력> [<이름=출력> …]
      (2단계 꼴도 받는다: 견주기 <작업폴더> <정밀 출력> <F1 출력> <결과.json> — 이름 F1)
      행마다(실수 64 비트) KL(정밀 ‖ x) = Σ p_정밀 (log p_정밀 − log p_x), 1위 토큰이 정밀과 같은가 — x = 변형들 · llama.cpp(f16 KV · 있으면
      q8_0 KV). 판정(미리 정함 — 문서/2단계.md · 2b단계.md):
        B  — 평균 KL(정밀 ‖ x) ≤ 평균 KL(정밀 ‖ llama.cpp) 이고 1위 일치가 llama.cpp 이상(2단계의 처음 판정).
        B′ · B″ — 평균 KL 이 llama.cpp(f16 KV)의 2.5 배 이하이고 1위 일치가 0.5 %p 넘게 낮지 않다. 글마다 · 2,048 글을 모아서 · 긴 창 따로.
"""
import ctypes
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
길이 = 2048
반 = 1024
긴길이 = 8192
긴잡기 = 7168
덩이 = 128                     # 로짓 행을 이만큼씩 읽어 셈한다(조각 길이 1024 · 2048 의 약수)


class 라마어휘:
    """llama.cpp C API 의 어휘만(vocab_only) — llama_tokenize 를 부르려고. 재기/라마.py 의 DLL 부르기를 쓴다(GEULPU_LLAMA = 판 폴더).
    (GeulPU 시험/토크나이저39.py 의 같은 이름 클래스에서 — b11669 와 b11719 의 llama.h 는 같다)"""

    def __init__(self, 판폴더, gguf):
        os.environ["GEULPU_LLAMA"] = 판폴더
        sys.path.insert(0, os.path.join(ROOT, "재기"))
        import 라마 as LM                      # noqa: E402
        L, _ = LM._불러오기(True)
        c_p = ctypes.c_void_p
        L.llama_model_default_params.restype = c_p
        L.llama_model_default_params.argtypes = [c_p]
        L.llama_model_load_from_file.restype = c_p
        L.llama_model_load_from_file.argtypes = [ctypes.c_char_p, c_p]
        L.llama_model_get_vocab.restype = c_p
        L.llama_model_get_vocab.argtypes = [c_p]
        mp = ctypes.create_string_buffer(256)
        L.llama_model_default_params(mp)
        assert mp.raw[72:78] == bytes([0, 0, 1, 0, 0, 0]), f"llama_model_params 의 참거짓 칸이 예상과 다르다: {mp.raw[64:80].hex()}"
        struct.pack_into("<B", mp, 72, 1)
        self.L = L
        self.model = L.llama_model_load_from_file(gguf.encode("utf-8"), mp)
        assert self.model, "llama.cpp 가 어휘를 읽지 못했다"
        self.vocab = L.llama_model_get_vocab(self.model)
        L.llama_tokenize.restype = ctypes.c_int32
        L.llama_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
                                     ctypes.c_bool, ctypes.c_bool]

    def 토큰(self, b, n):
        buf = (ctypes.c_int32 * (len(b) + 16))()
        k = self.L.llama_tokenize(self.vocab, b, len(b), buf, len(b) + 16, True, False)
        assert k >= n, (k, n)
        return [int(x) for x in buf[:n]]


def _글바이트(지정):
    """'경로' 또는 '경로@바이트' → (이름, 바이트) — 바이트 자리를 주면 그 뒤 첫 줄바꿈 다음부터(UTF-8 글자를 가르지 않게)."""
    경로, _, 자리 = 지정.partition("@")
    b = open(경로, "rb").read()
    if 자리:
        nl = b.find(b"\n", int(자리))
        assert nl >= 0, 지정
        b = b[nl + 1:]
    return os.path.basename(경로) + (f"@{자리}" if 자리 else ""), b[:400000]


def 입력(gguf, 판폴더, 작업, 글들, 긴):
    os.makedirs(작업, exist_ok=True)
    어휘 = 라마어휘(판폴더, gguf)
    이름들, 토큰들 = [], []
    for 지정 in 글들:
        이름, b = _글바이트(지정)
        이름들.append(이름)
        토큰들.append(어휘.토큰(b, 길이))
    긴이름, 긴토큰 = None, None
    if 긴:
        긴이름, b = _글바이트(긴)
        긴토큰 = 어휘.토큰(b, 긴길이)
    with open(os.path.join(작업, "토큰.json"), "w", encoding="utf-8") as f:
        json.dump({"gguf": os.path.basename(gguf), "글": 이름들, "토큰": 토큰들, "긴글": 긴이름, "긴": 긴토큰}, f)
    최대 = 긴길이 if 긴토큰 else 길이
    with open(os.path.join(작업, "입력.bin"), "wb") as f:
        w = lambda v: f.write(int(v).to_bytes(8, "little", signed=True))
        w(최대), w(반), w(2 * len(토큰들) + (1 if 긴토큰 else 0))
        for ids in 토큰들:
            for p, 잡기 in ((반, 0), (길이, 반)):
                w(p), w(1), w(잡기)
                for t in ids[:p]:
                    w(t)
        if 긴토큰:
            w(긴길이), w(1), w(긴잡기)
            for t in 긴토큰:
                w(t)
    print("입력: 글", len(토큰들), "· 2,048 토큰씩" + (f" · 긴 창 {긴이름} 8,192 토큰" if 긴토큰 else ""), "· 첫 토큰", [t[:4] for t in 토큰들])


def 라마(gguf, 판폴더, 작업, kv="f16"):
    os.environ["GEULPU_LLAMA"] = 판폴더
    sys.path.insert(0, os.path.join(ROOT, "재기"))
    import 라마 as LM                     # noqa: E402
    T = json.load(open(os.path.join(작업, "토큰.json"), encoding="utf-8"))
    토큰들, 긴 = T["토큰"], T.get("긴")
    n_ctx = 긴길이 if 긴 else 길이
    앞 = {"f16": "라마", "q8_0": "라마q8_"}[kv]
    lm = LM.라마(gguf, n_ctx=n_ctx, n_batch=n_ctx, n_ubatch=512, n_seq_max=1, flash=-1, kv형식={"f16": 1, "q8_0": 8}[kv])
    print("llama.cpp 문맥:", " | ".join(l.strip() for l in lm.문맥로그.splitlines() if any(x in l for x in ("flash", "Flash", "n_ctx", "n_ubatch", "CUDA", "type_k"))))
    for i, ids in enumerate(토큰들):
        lm.지우기()
        lm.넣기(ids, list(range(len(ids))), [0] * len(ids), [1] * len(ids))
        x = lm.모든로짓(len(ids))
        np.save(os.path.join(작업, f"{앞}{i}.npy"), x)
        print("라마", kv, i, x.shape, "1위(앞 8 행)", x[:8].argmax(1).tolist(), flush=True)
    if 긴:
        lm.지우기()
        lm.넣기(긴, list(range(긴길이)), [0] * 긴길이, [0] * 긴잡기 + [1] * (긴길이 - 긴잡기))
        x = lm.모든로짓(긴길이 - 긴잡기)
        np.save(os.path.join(작업, f"{앞}긴.npy"), x)
        print("라마", kv, "긴", x.shape, "1위(앞 8 행)", x[:8].argmax(1).tolist(), flush=True)
    lm.닫기()


def _로그소프트맥스(x):
    x = np.asarray(x, np.float64)
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def _엔진조각(폴더, 경우들, V):
    """시험_7B 출력의 경우들 [(번호, 잡기)] → 로짓 조각(읽기만 하는 memmap) 목록."""
    return [np.memmap(os.path.join(폴더, f"경우{k}_{'로짓' if 잡기 == 0 else '뒤로짓'}.bin"), np.float32, "r").reshape(-1, V)
            for k, 잡기 in 경우들]


def _덩이들(조각들):
    for c in 조각들:
        for a in range(0, c.shape[0], 덩이):
            yield np.asarray(c[a:a + 덩이])


def _견줌(기준, 대상들):
    """기준 조각들(정밀)과 대상들(이름 → 조각들)의 행마다 KL(정밀 ‖ x) · 1위 같음."""
    kl = {n: [] for n in 대상들}
    같 = {n: [] for n in 대상들}
    이음 = {n: _덩이들(c) for n, c in 대상들.items()}
    for blk in _덩이들(기준):
        lp = _로그소프트맥스(blk)
        p = np.exp(lp)
        top = blk.argmax(1)
        for n, it in 이음.items():
            x = next(it)
            assert x.shape == blk.shape, (n, x.shape, blk.shape)
            kl[n].append((p * (lp - _로그소프트맥스(x))).sum(1))
            같[n].append(x.argmax(1) == top)
    return {n: (np.concatenate(kl[n]), np.concatenate(같[n])) for n in 대상들}


def _통계(kl, 같):
    return {"행": int(len(kl)), "평균 KL": float(kl.mean()), "KL 중앙값": float(np.median(kl)), "KL 최대": float(kl.max()),
            "1위 같음 비율": float(같.mean())}


def _판정(x, 라):
    비 = x["평균 KL"] / max(라["평균 KL"], 1e-30)
    return {"KL 비": 비, "B": bool(x["평균 KL"] <= 라["평균 KL"] and x["1위 같음 비율"] >= 라["1위 같음 비율"]),
            "B″": bool(비 <= 2.5 and x["1위 같음 비율"] >= 라["1위 같음 비율"] - 0.005)}


def 견주기(작업, 정밀폴더, 변형들, 출력):
    T = json.load(open(os.path.join(작업, "토큰.json"), encoding="utf-8"))
    토큰들, 긴 = T["토큰"], T.get("긴")
    q8 = os.path.exists(os.path.join(작업, "라마q8_0.npy"))
    V = np.load(os.path.join(작업, "라마0.npy"), mmap_mode="r").shape[1]
    for 폴더 in [정밀폴더] + list(변형들.values()):
        for i, ids in enumerate(토큰들):
            S = np.fromfile(os.path.join(폴더, f"경우{2 * i}_토큰.bin"), dtype=np.int64)
            assert S.tolist() == ids[:len(S)], f"글 {i}: {폴더} 의 토큰열이 입력과 다르다"
    결과 = {"gguf": T["gguf"], "글": T["글"], "긴글": T.get("긴글"), "변형": list(변형들), "llama q8_0 KV": q8, "글마다": [], "모음": {}, "긴": {}}
    모음 = {}

    def 한벌(기준조각, 대상조각, 라조각, 라q8조각):
        대상 = dict(대상조각)
        대상["llama.cpp"] = 라조각
        if 라q8조각 is not None:
            대상["llama.cpp q8_0 KV"] = 라q8조각
        return _견줌(기준조각, 대상)

    def 적기(표, 이름표):
        라 = _통계(*표["llama.cpp"])
        한 = {"llama.cpp": 라}
        for n, v in 표.items():
            if n == "llama.cpp":
                continue
            한[n] = _통계(*v)
            한[n].update(_판정(한[n], 라))
        print(f"{이름표}: llama.cpp KL {라['평균 KL']:.3e} · 1위 {라['1위 같음 비율'] * 100:.2f} %", flush=True)
        for n in 한:
            if n == "llama.cpp":
                continue
            x = 한[n]
            print(f"    {n}: KL {x['평균 KL']:.3e} (비 {x['KL 비']:.2f}) · 1위 {x['1위 같음 비율'] * 100:.2f} % — B″ {'통과' if x['B″'] else '못 미침'}",
                  flush=True)
        return 한

    for i in range(len(토큰들)):
        정 = _엔진조각(정밀폴더, [(2 * i, 0), (2 * i + 1, 반)], V)
        대 = {n: _엔진조각(f, [(2 * i, 0), (2 * i + 1, 반)], V) for n, f in 변형들.items()}
        라 = [np.load(os.path.join(작업, f"라마{i}.npy"), mmap_mode="r")]
        라8 = [np.load(os.path.join(작업, f"라마q8_{i}.npy"), mmap_mode="r")] if q8 else None
        표 = 한벌(정, 대, 라, 라8)
        for n, v in 표.items():
            모음.setdefault(n, []).append(v)
        결과["글마다"].append({"글": T["글"][i], **적기(표, f"글 {i} ({T['글'][i]})")})
    합 = {n: (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) for n, v in 모음.items()}
    결과["모음"] = 적기(합, f"글 {len(토큰들)} 개를 모아서")
    if 긴:
        k = 2 * len(토큰들)
        정 = _엔진조각(정밀폴더, [(k, 긴잡기)], V)
        대 = {n: _엔진조각(f, [(k, 긴잡기)], V) for n, f in 변형들.items()}
        라 = [np.load(os.path.join(작업, "라마긴.npy"), mmap_mode="r")]
        라8 = [np.load(os.path.join(작업, "라마q8_긴.npy"), mmap_mode="r")] if q8 else None
        결과["긴"] = 적기(한벌(정, 대, 라, 라8), f"긴 창 ({T.get('긴글')} — 행 {긴잡기} … {긴길이 - 1})")
    판 = {}
    for n in 변형들:
        칸들 = [결과["모음"][n]] + ([결과["긴"][n]] if 긴 else [])
        판[n] = {"B″(모음 · 긴 창)": all(c["B″"] for c in 칸들), "B′(글마다)": all(g[n]["B″"] for g in 결과["글마다"]),
                 "B(글마다)": all(g[n]["B"] for g in 결과["글마다"])}
    결과["판정"] = 판
    # 2단계 꼴의 열쇠(재기/품질2.sh 의 요약이 읽는다)
    결과["통과"] = all(v["B(글마다)"] for v in 판.values())
    결과["통과 B′"] = all(v["B′(글마다)"] for v in 판.values())
    with open(출력, "w", encoding="utf-8") as f:
        json.dump(결과, f, ensure_ascii=False, indent=1)
    for n, v in 판.items():
        print(f"판정 {T['gguf']} · {n}: B″(모음 · 긴 창) {'통과' if v['B″(모음 · 긴 창)'] else '못 미침'} · B′(글마다) "
              f"{'통과' if v['B′(글마다)'] else '못 미침'} · B(글마다) {'통과' if v['B(글마다)'] else '못 미침'}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    op = a[0] if a else ""
    if op == "입력":
        글들, 긴 = [], None
        나머지 = a[4:]
        while 나머지:
            if 나머지[0] == "--글":
                글들.append(나머지[1])
                나머지 = 나머지[2:]
            elif 나머지[0] == "--긴":
                긴 = 나머지[1]
                나머지 = 나머지[2:]
            else:                                   # 2단계 꼴: 글 둘을 그대로
                글들.append(나머지[0])
                나머지 = 나머지[1:]
        if not 글들:
            글들 = [os.path.join(ROOT, "빌드", "wikitext-2", "wiki.test.raw"), os.path.join(ROOT, "연구자안내.md")]
        입력(a[1], a[2], a[3], 글들, 긴)
    elif op == "라마":
        kv = a[a.index("--kv") + 1] if "--kv" in a else "f16"
        라마(a[1], a[2], a[3], kv)
    elif op == "견주기":
        작업, 정밀 = a[1], a[2]
        if len(a) == 5 and "=" not in a[3]:          # 2단계 꼴: 견주기 <작업> <정밀> <F1> <결과.json>
            견주기(작업, 정밀, {"F1": a[3]}, a[4])
        else:
            견주기(작업, 정밀, dict(x.split("=", 1) for x in a[4:]), a[3])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
