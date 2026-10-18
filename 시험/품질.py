#!/usr/bin/env python3
"""2단계 판정 B — 숨은 손실의 견줌: 정밀 모드(계약 자리 둘 — fp64 에 가까운 기준)의 다음 토큰 분포를 기준으로, 빠른 모드(F1)와
llama.cpp(최신 공식 판 · 기본 설정)가 얼마나 벗어나는가. GeulPU 39단계의 시험/독립39.py(엔진 대 llama.cpp 의 모든 행 로짓 대조)에서 왔다.

  python -I 시험/품질.py 입력 <gguf> <llama 폴더> <작업폴더> [<글1> <글2>]
      글 둘(기본: wikitext-2 test 앞 · 연구자안내.md)을 llama.cpp 의 llama_tokenize(BOS 는 모델대로 · 특수 토큰은 풀지 않음 —
      llama-perplexity 와 같다)로 앞 2,048 토큰씩 → 작업폴더/토큰.json · 입력.bin(엔진/시험_7B.gl 의 입력 — 최대길이 2048 · 로짓행 1024:
      글마다 경우 둘 — (프롬프트 1024, 잡기 0) → 행 0 … 1023, (프롬프트 2048, 잡기 1024) → 행 1024 … 2047).
  python -I 시험/품질.py 라마 <gguf> <llama 폴더> <작업폴더>
      llama.cpp C API(재기/라마.py): n_ctx 2048 · n_batch 2048 · n_ubatch 512 · 플래시 어텐션 자동 · KV F16 · 모든 층 GPU → 라마{i}.npy
  python -I 시험/품질.py 견주기 <작업폴더> <정밀 시험_7B 출력> <F1 시험_7B 출력> <결과.json>
      행마다(실수 64 비트) KL(정밀 ‖ x) = Σ p_정밀 (log p_정밀 − log p_x), 1위 토큰이 정밀과 같은가 — x = F1 · llama.cpp.
      판정(2단계 B — 미리 정함): 글마다 평균 KL(정밀 ‖ F1) ≤ 평균 KL(정밀 ‖ llama.cpp) 이고 1위 일치 F1 ≥ llama.cpp.
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


def 입력(gguf, 판폴더, 작업, 글들):
    os.makedirs(작업, exist_ok=True)
    어휘 = 라마어휘(판폴더, gguf)
    L = 어휘.L
    L.llama_tokenize.restype = ctypes.c_int32
    L.llama_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
                                 ctypes.c_bool, ctypes.c_bool]
    토큰들 = []
    for p in 글들:
        b = open(p, "rb").read()[:200000]
        buf = (ctypes.c_int32 * (len(b) + 16))()
        n = L.llama_tokenize(어휘.vocab, b, len(b), buf, len(b) + 16, True, False)
        assert n >= 길이, (p, n)
        토큰들.append([int(x) for x in buf[:길이]])
    with open(os.path.join(작업, "토큰.json"), "w", encoding="utf-8") as f:
        json.dump({"gguf": os.path.basename(gguf), "글": [os.path.basename(p) for p in 글들], "토큰": 토큰들}, f)
    반 = 길이 // 2
    with open(os.path.join(작업, "입력.bin"), "wb") as f:
        w = lambda v: f.write(int(v).to_bytes(8, "little", signed=True))
        w(길이), w(반), w(2 * len(토큰들))
        for ids in 토큰들:
            for p, 잡기 in ((반, 0), (길이, 반)):
                w(p), w(1), w(잡기)
                for t in ids[:p]:
                    w(t)
    print("입력: 경우", 2 * len(토큰들), "· 토큰", 길이, "씩 · 첫 토큰", [t[:4] for t in 토큰들])


def 라마(gguf, 판폴더, 작업):
    os.environ["GEULPU_LLAMA"] = 판폴더
    sys.path.insert(0, os.path.join(ROOT, "재기"))
    import 라마 as LM                     # noqa: E402
    토큰들 = json.load(open(os.path.join(작업, "토큰.json"), encoding="utf-8"))["토큰"]
    lm = LM.라마(gguf, n_ctx=길이, n_batch=길이, n_ubatch=512, n_seq_max=1, flash=-1, kv형식=1)
    print("llama.cpp 문맥:", " | ".join(l.strip() for l in lm.문맥로그.splitlines() if any(x in l for x in ("flash", "Flash", "n_ctx", "n_ubatch", "CUDA"))))
    for i, ids in enumerate(토큰들):
        lm.지우기()
        lm.넣기(ids, list(range(len(ids))), [0] * len(ids), [1] * len(ids))
        x = lm.모든로짓(len(ids))
        np.save(os.path.join(작업, f"라마{i}.npy"), x)
        print("라마", i, x.shape, "1위(앞 8 행)", x[:8].argmax(1).tolist())
    lm.닫기()


def _로그소프트맥스(x):
    x = np.asarray(x, np.float64)
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def _글로짓(폴더, i, V):
    앞 = np.fromfile(os.path.join(폴더, f"경우{2 * i}_로짓.bin"), dtype=np.float32).reshape(-1, V)
    뒤 = np.fromfile(os.path.join(폴더, f"경우{2 * i + 1}_뒤로짓.bin"), dtype=np.float32).reshape(-1, V)
    return np.concatenate([앞, 뒤], 0)


def 견주기(작업, 정밀폴더, 빠른폴더, 출력):
    토큰들 = json.load(open(os.path.join(작업, "토큰.json"), encoding="utf-8"))
    결과 = {"gguf": 토큰들["gguf"], "글": 토큰들["글"], "경우": []}
    통과 = True
    for i, ids in enumerate(토큰들["토큰"]):
        for 폴더 in (정밀폴더, 빠른폴더):
            S = np.fromfile(os.path.join(폴더, f"경우{2 * i}_토큰.bin"), dtype=np.int64)
            assert S.tolist() == ids[:len(S)], f"글 {i}: {폴더} 의 토큰열이 입력과 다르다"
        라 = np.load(os.path.join(작업, f"라마{i}.npy"), mmap_mode="r")
        V = 라.shape[1]
        정, 빠 = _글로짓(정밀폴더, i, V), _글로짓(빠른폴더, i, V)
        assert 정.shape == 라.shape == 빠.shape, (정.shape, 빠.shape, 라.shape)
        n = len(ids)
        kl = {"F1": np.zeros(n), "llama.cpp": np.zeros(n)}
        같음 = {"F1": np.zeros(n, bool), "llama.cpp": np.zeros(n, bool)}
        for a in range(0, n, 128):
            b = min(n, a + 128)
            lp = _로그소프트맥스(정[a:b])
            p = np.exp(lp)
            top = np.asarray(정[a:b]).argmax(1)
            for 이름, x in (("F1", 빠), ("llama.cpp", 라)):
                lq = _로그소프트맥스(x[a:b])
                kl[이름][a:b] = (p * (lp - lq)).sum(1)
                같음[이름][a:b] = np.asarray(x[a:b]).argmax(1) == top
        한 = {"경우": i, "행": n}
        for 이름 in ("F1", "llama.cpp"):
            한[이름] = {"평균 KL": float(kl[이름].mean()), "KL 중앙값": float(np.median(kl[이름])), "KL 최대": float(kl[이름].max()),
                       "1위 같음 비율": float(같음[이름].mean())}
        한["판정"] = bool(한["F1"]["평균 KL"] <= 한["llama.cpp"]["평균 KL"] and 한["F1"]["1위 같음 비율"] >= 한["llama.cpp"]["1위 같음 비율"])
        # 2단계 B′ 같은 급(독립 검토 뒤 · 측정 전에 정함 — 문서/2단계.md): KL ≤ 2.5 배, 1위 일치가 0.5 %p 넘게 낮지 않다
        한["KL 비"] = 한["F1"]["평균 KL"] / max(한["llama.cpp"]["평균 KL"], 1e-30)
        한["판정 B′"] = bool(한["KL 비"] <= 2.5 and 한["F1"]["1위 같음 비율"] >= 한["llama.cpp"]["1위 같음 비율"] - 0.005)
        통과 &= 한["판정"]
        통과B2 = 결과.get("통과 B′", True) and 한["판정 B′"]
        결과["통과 B′"] = 통과B2
        결과["경우"].append(한)
        print(f"글 {i}: KL(정밀‖F1) {한['F1']['평균 KL']:.3e} · KL(정밀‖llama.cpp) {한['llama.cpp']['평균 KL']:.3e} (비 {한['KL 비']:.2f}) · "
              f"1위 같음 F1 {한['F1']['1위 같음 비율'] * 100:.2f} % · llama.cpp {한['llama.cpp']['1위 같음 비율'] * 100:.2f} % — "
              f"B {'통과' if 한['판정'] else '못 미침'} · B′ {'통과' if 한['판정 B′'] else '못 미침'}")
    결과["통과"] = 통과
    with open(출력, "w", encoding="utf-8") as f:
        json.dump(결과, f, ensure_ascii=False, indent=1)
    print("2단계 B", 결과["gguf"], "— 통과" if 통과 else "— 못 미침", "· B′", "통과" if 결과.get("통과 B′") else "못 미침")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    op = sys.argv[1] if len(sys.argv) > 1 else ""
    if op == "입력":
        글들 = sys.argv[5:7] if len(sys.argv) > 6 else [os.path.join(ROOT, "빌드", "wikitext-2", "wiki.test.raw"),
                                                       os.path.join(ROOT, "연구자안내.md")]
        입력(sys.argv[2], sys.argv[3], sys.argv[4], 글들)
    elif op == "라마":
        라마(sys.argv[2], sys.argv[3], sys.argv[4])
    elif op == "견주기":
        견주기(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
