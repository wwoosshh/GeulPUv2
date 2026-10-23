#!/usr/bin/env python3
"""0.12.0 12-G2 — BPE 사전 처리 표(엔진/사전처리표.gl)를 llama.cpp 의 소스에서 만든다(MIT, Copyright (c) 2023-2026 The ggml authors):
  - src/llama-vocab.cpp 의 load(): 이름(tokenizer.ggml.pre) → 갈래(LLAMA_VOCAB_PRE_TYPE_…) · ignore_merges
  - 같은 파일의 llm_tokenizer_bpe: 갈래 → 정규식 목록(regex_exprs) · byte_encode
  - src/unicode.cpp 의 unicode_regex_split_custom: 손으로 짠 빠른 길이 있는 정규식의 글

  python -I 도구/사전처리표.py <llama-vocab.cpp> <unicode.cpp> <llama.cpp 의 판> 엔진/사전처리표.gl
"""
import re
import sys

vocab_src, unicode_src, 판, out = sys.argv[1:5]
V = open(vocab_src, encoding="utf-8").read()
U = open(unicode_src, encoding="utf-8").read()


def 글자들(t):
    """C++ 조각의 문자열 글들(주석 밖) — 이웃한 글은 잇는다"""
    out, i, n, 앞글 = [], 0, len(t), False
    while i < n:
        if t.startswith("//", i):
            i = t.find("\n", i)
            i = n if i < 0 else i
            continue
        if t.startswith("/*", i):
            i = t.index("*/", i) + 2
            continue
        c = t[i]
        if c == '"':
            i += 1
            s = []
            while t[i] != '"':
                if t[i] == "\\":
                    e = t[i + 1]
                    s.append({"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"', "'": "'", "0": "\0"}[e])
                    i += 2
                else:
                    s.append(t[i])
                    i += 1
            i += 1
            if 앞글:
                out[-1] += "".join(s)
            else:
                out.append("".join(s))
            앞글 = True
            continue
        if not c.isspace():
            앞글 = False
        i += 1
    return out


# ---- 갈래 → 정규식 목록 (llm_tokenizer_bpe) ----
a = V.index("llm_tokenizer_bpe(const llama_vocab & vocab) {")
b = V.index("std::vector<std::string> regex_exprs;", a)
sw = V[a:b]
갈래식 = {}            # 갈래 → (정규식 목록, 바이트 꼴)
라벨 = list(re.finditer(r"case LLAMA_VOCAB_PRE_TYPE_(\w+):|default:", sw))
모음 = []
for k, m in enumerate(라벨):
    이름 = m.group(1) or "DEFAULT_REGEX"
    모음.append(이름)
    끝 = 라벨[k + 1].start() if k + 1 < len(라벨) else len(sw)
    몸 = sw[m.end():끝]
    if "regex_exprs" not in 몸:
        continue                                   # 다음 라벨과 몸을 함께 쓴다
    s = 몸.index("regex_exprs = {")
    e = 몸.index("};", s)
    식 = 글자들(몸[s:e])
    바이트 = "byte_encode = false" not in 몸
    for g in 모음:
        갈래식[g] = (식, 바이트)
    모음 = []
assert "QWEN2" in 갈래식 and "QWEN35" in 갈래식 and "DEFAULT_REGEX" in 갈래식, sorted(갈래식)

# ---- 이름 → 갈래 · ignore_merges (load) ----
L0 = V.index("void llama_vocab::impl::load(")
lv = V[L0:]
이름표 = []            # (이름들, 갈래, 합침무시)
for m in re.finditer(r"\(\s*((?:tokenizer_pre\s*==\s*\"[^\"]+\"[\s|]*)+)\)\s*\{([^{}]*)\}", lv):
    이름들 = re.findall(r"tokenizer_pre\s*==\s*\"([^\"]+)\"", m.group(1))
    g = re.search(r"pre_type\s*=\s*LLAMA_VOCAB_PRE_TYPE_(\w+);", m.group(2))
    if not g:
        continue
    무시 = re.search(r"ignore_merges\s*=\s*true", m.group(2)) is not None
    이름표.append((이름들, g.group(1), 무시))
assert any("qwen35" in x[0] for x in 이름표)

# ---- 빠른 길 (unicode_regex_split_custom) ----
c0 = U.index("static std::vector<size_t> unicode_regex_split_custom(const std::string & text")
c1 = U.index("return bpe_offsets;", c0)
cu = U[c0:c1]
빠른 = []
for m in re.finditer(r"((?:regex_expr\s*==\s*\"(?:[^\"\\]|\\.)*\"\s*(?:\|\|\s*)?)+)\)\s*\{(.*?)bpe_offsets = unicode_regex_split_custom_(\w+)\(", cu, re.S):
    for 식 in 글자들(m.group(1)):
        빠른.append((식, m.group(3)))
빠른번호 = {"gpt2": 1, "llama3": 2, "qwen2": 3, "qwen35": 4, "k2_horizon": 5, "kimi_k2": 6, "afmoe": 7, "newlines": 8}
assert {f for _, f in 빠른} <= set(빠른번호), sorted({f for _, f in 빠른})


def 글(s):
    r = s.replace("\\", "\\\\").replace('"', '\\"').replace("{", "\\{").replace("}", "\\}")
    return '"' + r.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t").replace("\0", "\\0") + '"'


줄 = [f"""(* llama.cpp {판} 의 src/llama-vocab.cpp · src/unicode.cpp(MIT, Copyright (c) 2023-2026 The ggml authors —
   THIRD_PARTY_NOTICES.md)에서 도구/사전처리표.py 가
   만들었다. 고치지 말 것(다시 만든다). 0.12.0 12-G2 — BPE 의 사전 처리(문서/정규식.md): 이름(tokenizer.ggml.pre) → 갈래 · ignore_merges,
   갈래 → 정규식 목록(llm_tokenizer_bpe 의 regex_exprs) · 바이트 꼴(byte_encode), 손으로 짠 빠른 길이 있는 정규식(unicode_regex_split_custom —
   1 gpt2 · 2 llama3 · 3 qwen2 · 4 qwen35 · 5 k2-horizon · 6 kimi-k2 · 7 afmoe · 8 줄바꿈) *)

(* 이름의 번호(이름표의 차례 — 없으면 −1). "default" 와 빈 이름은 DEFAULT *)
[문자열 이름을 사전처리찾기]는 -> 정수 {{"""]
for k, (이름들, g, 무시) in enumerate(이름표):
    조건 = " 또는 ".join(f"문자열_같은가(이름, {글(x)})" for x in 이름들)
    줄.append(f"    ({조건})이면 {{ 반환 {k}. }}")
줄.append("    반환 -1.\n}\n")
줄.append("(* 번호의 갈래(llama.cpp 의 LLAMA_VOCAB_PRE_TYPE_…) *)\n[정수 번을 사전처리갈래]는 -> 문자열 {")
for k, (_, g, _) in enumerate(이름표):
    줄.append(f"    번 == {k}이면 {{ 반환 {글(g)}. }}")
줄.append("    반환 없음.\n}\n")
줄.append("(* 번호의 ignore_merges(조각 통째가 낱말이면 합치기 없이) *)\n[정수 번을 사전처리합침무시]는 -> 참거짓 {")
무시들 = [k for k, (_, _, 무시) in enumerate(이름표) if 무시]
for k in 무시들:
    줄.append(f"    번 == {k}이면 {{ 반환 참. }}")
줄.append("    반환 거짓.\n}\n")
줄.append("(* 번호의 갈래가 바이트 꼴(GPT-2 의 바이트 글자)인가 — 아니면 SentencePiece 꼴의 BPE(엔진이 받지 않는다) *)\n[정수 번을 사전처리바이트]는 -> 참거짓 {")
for k, (_, g, _) in enumerate(이름표):
    식, 바이트 = 갈래식.get(g, 갈래식["DEFAULT_REGEX"])
    if not 바이트:
        줄.append(f"    번 == {k}이면 {{ 반환 거짓. }}")
줄.append("    반환 참.\n}\n")
줄.append("(* 번호의 k 째 정규식(없으면 없음) — 갈래에 정규식 목록이 따로 없으면 llama.cpp 의 기본 목록 *)\n[정수 번을 정수 k를 사전처리정규식]는 -> 문자열 {")
for k, (_, g, _) in enumerate(이름표):
    식, _ = 갈래식.get(g, 갈래식["DEFAULT_REGEX"])
    for j, s in enumerate(식):
        줄.append(f"    (번 == {k} 그리고 k == {j})이면 {{ 반환 {글(s)}. }}")
줄.append("    반환 없음.\n}\n")
줄.append("(* 기본 갈래(이름 \"default\" · 빈 이름)의 k 째 정규식 *)\n[정수 k를 사전처리기본정규식]는 -> 문자열 {")
for j, s in enumerate(갈래식["DEFAULT_REGEX"][0]):
    줄.append(f"    k == {j}이면 {{ 반환 {글(s)}. }}")
줄.append("    반환 없음.\n}\n")
줄.append("(* 정규식의 빠른 길 번호(없으면 0) *)\n[문자열 식을 정규빠른길]는 -> 정수 {")
for s, f in 빠른:
    줄.append(f"    문자열_같은가(식, {글(s)})이면 {{ 반환 {빠른번호[f]}. }}")
줄.append("    반환 0.\n}\n")
open(out, "wb").write("\n".join(줄).encode("utf-8"))
print(f"이름 블록 {len(이름표)} · 이름 {sum(len(x[0]) for x in 이름표)} · 갈래 {len(갈래식)} · 빠른 길 {len(빠른)} → {out}")
