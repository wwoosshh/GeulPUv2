#!/usr/bin/env python3
"""22단계 — 글 시험 도구(호스트/시험_7B.gl)의 입력: 채팅 질문 셋의 채팅 틀(잡기시작 0 — 모든 행 로짓을 CPU 참조와 대조)과
문장 1 을 되풀이한 2100 토큰(잡기시작 2000 — 묶음 경계 1024 · 2048 을 넘는 길끼리의 대조). 걸음 17 (생성 16 걸음).
토크나이저는 7B GGUF 의 것(엔진/토크나이저.py). 37단계 — 채팅 틀은 구조에 맞춰(qwen3 는 Qwen3 의 틀 — 기본 시스템 말 없음, 생각 켬).

  python -I 시험/입력7B.py <7B gguf> <입력.bin> [--최대길이 4096] [--로짓행 128]
  python -I 시험/입력7B.py <7B gguf> <재기입력.bin> --재기        (22d — 호스트/재기_7B.gl 의 입력)
  python -I 시험/입력7B.py <gguf> <입력.bin> --무작위 24          (24단계 — 합성 모델: 같은 길이의 정한 무작위 토큰열)
39단계 — 구조 llama(Llama 3 · Mistral)는 채팅 틀을 서버(호스트/서버몸.gl 의 라마3틀 · 미스트랄틀)와 같은 글로 짓고, 토큰 번호는
llama.cpp 의 llama-tokenize(특수 토큰 알아봄 · BOS 를 덧붙이지 않음 — 틀의 글에 BOS 가 있다)로: --라마토큰 <llama-tokenize.exe>.
무작위 토큰은 어휘 앞쪽의 보통 토큰(종류 1)에서(Llama 3 은 0 … 127999).
0.12.0 12-3e-4 — --토큰 <토큰.json>: 토크나이저 없이 JSON 의 토큰열들(시험/qwen35수학.py 의 토큰.json — "토큰": [[…], …])을 경우로
(걸음 17 · 잡기시작 0)와 첫 토큰열을 되풀이한 1100 토큰(잡기시작 1000 — 묶음 경계 1024 를 넘는 길끼리의 대조). Qwen3.5 처럼 이 도구의
토크나이저가 아직 모르는 어휘에.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "계약"))
import gguf읽기 as GR       # noqa: E402
import 토크나이저 as TK      # noqa: E402

질문들 = ["What is the capital of France? Answer in one word.", "대한민국의 수도는 어디야? 한 단어로 답해.",
         "Write a short poem about the sea."]
긴글 = ("In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley "
       "in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.")


def 틀(구조):
    """구조의 채팅 틀(37단계 — qwen3 는 채팅틀qwen3, 그 밖은 Qwen2.5 의 ChatML)."""
    return TK.채팅틀qwen3 if 구조 == "qwen3" else TK.채팅틀


def 라마틀(g, 질문):
    """39단계 — llama 구조의 채팅 틀(user 하나 · 생성 머리): 바이트 BPE(Llama 3)는 3.1 · 3.2 의 틀(날짜 고정), SentencePiece 는 Mistral v0.3."""
    m = g.meta
    bos = m["tokenizer.ggml.tokens"][m["tokenizer.ggml.bos_token_id"]]
    if m["tokenizer.ggml.model"] == "gpt2":
        return (bos + "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + 질문.strip() + "<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n")
    return bos + "[INST] " + 질문 + " [/INST]"


class 라마토크나이저:
    """39단계 — llama.cpp 의 llama-tokenize 로 나눈다(특수 토큰 알아봄, BOS 를 덧붙이지 않음)."""

    def __init__(self, 도구, gguf):
        self.도구, self.gguf = 도구, gguf

    def 나누기(self, 글):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "prompt.txt")          # llama.cpp 는 한글 경로를 열지 못한다
            with open(p, "wb") as f:
                f.write(글.encode("utf-8"))
            r = subprocess.run([self.도구, "-m", self.gguf, "-f", p, "--ids", "--log-disable", "--no-escape", "--no-bos"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        return json.loads(r.stdout.strip().splitlines()[-1])


def 재기입력(tk, 출력, 구조="qwen2"):
    """22d 의 입력(호스트/재기_7B.gl) — 18e 와 같은 꼴: 채팅 질문(셋째)의 채팅 틀, 문장 1 의 되풀이 8192."""
    채팅 = tk.나누기(틀(구조)([{"role": "user", "content": 질문들[2]}]))
    긴 = (tk.나누기(긴글) * 400)[:8192]
    with open(출력, "wb") as f:
        w = lambda v: f.write(int(v).to_bytes(8, "little", signed=True))
        for ids in (채팅, 긴):
            w(len(ids))
            for t in ids:
                w(t)
    print("재기 입력: 채팅", len(채팅), "· 긴 글", len(긴))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    gguf, 출력 = sys.argv[1:3]
    최대길이 = int(sys.argv[sys.argv.index("--최대길이") + 1]) if "--최대길이" in sys.argv else 4096
    로짓행 = int(sys.argv[sys.argv.index("--로짓행") + 1]) if "--로짓행" in sys.argv else 128
    if "--토큰" in sys.argv:                  # 0.12.0 12-3e-4 — 토크나이저 없이 JSON 의 토큰열
        토큰들 = json.load(open(sys.argv[sys.argv.index("--토큰") + 1], encoding="utf-8"))["토큰"]
        경우들 = [([int(t) for t in ids], 17, 0) for ids in 토큰들]
        긴 = (토큰들[0] * (1100 // len(토큰들[0]) + 1))[:1100]
        경우들.append(([int(t) for t in 긴], 17, 1000))
        쓰기(출력, 최대길이, 로짓행, 경우들)
        return
    g = GR.GGUF(gguf)
    구조 = g.meta.get("general.architecture", "qwen2")
    if 구조 == "llama":                        # 39단계
        tk = 라마토크나이저(sys.argv[sys.argv.index("--라마토큰") + 1], gguf)
        채팅 = lambda q: tk.나누기(라마틀(g, q))
    else:
        tk = TK.Qwen토크나이저(gguf=g)
        채팅 = lambda q: tk.나누기(틀(구조)([{"role": "user", "content": q}]))
    if "--재기" in sys.argv:
        if 구조 == "llama":
            재기 = [채팅(질문들[2]), (tk.나누기(긴글) * 400)[:8192]]
            with open(출력, "wb") as f:
                for ids in 재기:
                    f.write(len(ids).to_bytes(8, "little", signed=True))
                    for t in ids:
                        f.write(int(t).to_bytes(8, "little", signed=True))
            print("재기 입력: 채팅", len(재기[0]), "· 긴 글", len(재기[1]))
            return
        재기입력(tk, 출력, 구조)
        return
    경우들 = [(채팅(q), 17, 0) for q in 질문들]
    경우들.append(((tk.나누기(긴글) * 400)[:2100], 17, 2000))
    if "--무작위" in sys.argv:                 # 24단계 — 합성 모델: 정한 무작위 토큰열 셋(길이는 채팅 질문과 같게)과 2100 (보통 낱말 0 … 151642)
        import numpy as np
        r = np.random.default_rng(int(sys.argv[sys.argv.index("--무작위") + 1]))
        상한 = 151643
        if 구조 == "llama":                    # 39단계 — 어휘 앞쪽의 보통 토큰(종류 1)이 이어진 만큼(Llama 3 은 128000)
            종류 = g.meta["tokenizer.ggml.token_type"]
            상한 = next((i for i, t in enumerate(종류) if t != 1), len(종류))
            assert 상한 > 0, "어휘 앞쪽에 보통 토큰이 없다"
        경우들 = [([int(t) for t in r.integers(0, 상한, len(ids))], n, 잡기) for ids, n, 잡기 in 경우들]
    쓰기(출력, 최대길이, 로짓행, 경우들)


def 쓰기(출력, 최대길이, 로짓행, 경우들):
    with open(출력, "wb") as f:
        w = lambda v: f.write(int(v).to_bytes(8, "little", signed=True))
        w(최대길이), w(로짓행), w(len(경우들))
        for ids, n, 잡기 in 경우들:
            w(len(ids)), w(n), w(잡기)
            for t in ids:
                w(t)
    print("경우", [(len(ids), n, 잡기) for ids, n, 잡기 in 경우들])


if __name__ == "__main__":
    main()
