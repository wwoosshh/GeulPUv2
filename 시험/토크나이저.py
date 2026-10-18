"""Qwen2 토크나이저 — 바이트 BPE(GPT-2 와 같은 바이트 → 글자 표), 손으로 편 정규식, 특수 토큰, 채팅 틀(ChatML).

Hugging Face tokenizers(tokenizer.json)의 Qwen2 설정 그대로:
  1. 특수 토큰(added_tokens — <|im_start|> 등 22 개)을 원문에서 먼저 떼어 낸다(가장 긴 것부터).
  2. 나머지 조각은 NFC 정규화.
  3. 정규식 (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+
     로 자른다(대안을 차례로 — 처음 맞는 것, 대안마다 탐욕). \\s 는 유니코드 White_Space(파이썬의 isspace 와 다르다 — \\x1c~\\x1f 는 아니다).
  4. 조각의 UTF-8 바이트를 글자로 바꿔 BPE(merges.txt 의 차례 — 머리줄이 없다).
시험: 시험/토크나이저대조.py 가 HF tokenizers 와 같은 토큰인지 견준다.
"""
import json
import os
import unicodedata

# 유니코드 White_Space (Rust regex 의 \s)
공백 = set("\t\n\x0b\x0c\r \x85\xa0     　") | {chr(c) for c in range(0x2000, 0x200b)}


def _바이트글자():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _글자(ch):
    return unicodedata.category(ch)[0] == "L"


def _숫자(ch):
    return unicodedata.category(ch)[0] == "N"


def _앞자르기(t):
    """Qwen2 의 정규식을 손으로 편 것 — 조각들(원문의 부분 문자열, 이어 붙이면 원문)."""
    out, i, n = [], 0, len(t)
    while i < n:
        c = t[i]
        # 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if c == "'" and i + 1 < n:
            맞음 = None
            for s in ("re", "ve", "ll", "s", "t", "m", "d"):        # 둘째 글자부터 대소문자 무시
                if t[i + 1:i + 1 + len(s)].lower() == s:
                    맞음 = 1 + len(s)
                    break
            if 맞음:
                out.append(t[i:i + 맞음])
                i += 맞음
                continue
        # 2. [^\r\n\p{L}\p{N}]?\p{L}+
        j = i
        if c not in "\r\n" and not _글자(c) and not _숫자(c):
            j = i + 1
        if j < n and _글자(t[j]):
            k = j
            while k < n and _글자(t[k]):
                k += 1
            out.append(t[i:k])
            i = k
            continue
        # 3. \p{N}
        if _숫자(c):
            out.append(c)
            i += 1
            continue
        # 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        j = i + 1 if c == " " else i
        기타 = lambda ch: ch not in 공백 and not _글자(ch) and not _숫자(ch)
        if j < n and 기타(t[j]):
            k = j
            while k < n and 기타(t[k]):
                k += 1
            while k < n and t[k] in "\r\n":
                k += 1
            out.append(t[i:k])
            i = k
            continue
        # 5~7. 공백
        if c in 공백:
            k = i
            while k < n and t[k] in 공백:
                k += 1                               # 공백 줄 W = t[i:k]
            마지막줄 = max((p for p in range(i, k) if t[p] in "\r\n"), default=-1)
            if 마지막줄 >= 0:                        # 5. \s*[\r\n]+ — 마지막 줄바꿈까지
                out.append(t[i:마지막줄 + 1])
                i = 마지막줄 + 1
            elif k == n or k - i >= 2:               # 6. \s+(?!\S) — 끝이면 모두, 아니면 마지막 하나를 남긴다
                e = k if k == n else k - 1
                out.append(t[i:e])
                i = e
            else:                                    # 7. \s+
                out.append(t[i:k])
                i = k
            continue
        # (어느 대안에도 맞지 않는 글자 — 위에서 모두 덮인다)
        raise AssertionError(f"자를 수 없는 글자 {c!r}")
    return out


class Qwen토크나이저:
    def __init__(self, 폴더=None, gguf=None):
        """폴더: Hugging Face 모델 폴더(vocab.json · merges.txt · tokenizer_config.json), 또는 gguf: GGUF 파일(경로 또는 gguf읽기.GGUF —
        tokenizer.ggml.tokens · merges · token_type; 14단계에서 같은 낱말 · 병합 · 특수 토큰 22 개임을 확인함)."""
        if gguf is not None:
            import gguf읽기
            meta = (gguf if isinstance(gguf, gguf읽기.GGUF) else gguf읽기.GGUF(gguf)).meta
            assert meta["tokenizer.ggml.model"] == "gpt2" and meta["tokenizer.ggml.pre"] == "qwen2", "Qwen2 의 바이트 BPE 가 아니다"
            toks, 종류 = meta["tokenizer.ggml.tokens"], meta["tokenizer.ggml.token_type"]
            self.낱말 = {t: i for i, t in enumerate(toks) if 종류[i] == 1}
            병합 = meta["tokenizer.ggml.merges"]
            self.특수 = {toks[i]: i for i in range(len(toks)) if 종류[i] in (3, 4)}       # 제어 · 사용자 정의(added tokens)
        else:
            self.낱말 = json.load(open(os.path.join(폴더, "vocab.json"), encoding="utf-8"))
            병합 = [l for l in open(os.path.join(폴더, "merges.txt"), encoding="utf-8").read().split("\n") if l and not l.startswith("#version")]
            설정 = json.load(open(os.path.join(폴더, "tokenizer_config.json"), encoding="utf-8"))
            self.특수 = {v["content"]: int(k) for k, v in 설정["added_tokens_decoder"].items()}
        self.순위 = {}
        for l in 병합:
            a, b = l.split(" ")
            self.순위[(a, b)] = len(self.순위)
        self.특수차례 = sorted(self.특수, key=len, reverse=True)
        self.거꾸로 = {v: k for k, v in self.낱말.items()}
        self.거꾸로.update({v: k for k, v in self.특수.items()})
        self.바이트 = _바이트글자()
        self.바이트거꾸로 = {c: b for b, c in self.바이트.items()}
        self.끝들 = {self.특수["<|im_end|>"], self.특수["<|endoftext|>"]}
        self._캐시 = {}

    def _bpe(self, word):
        if word in self._캐시:
            return self._캐시[word]
        parts = list(word)
        while len(parts) > 1:
            best = min((self.순위.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(parts, parts[1:])))
            if best[0] == 1 << 30:
                break
            a, b = parts[best[1]], parts[best[1] + 1]
            merged, k = [], 0
            while k < len(parts):
                if k + 1 < len(parts) and parts[k] == a and parts[k + 1] == b:
                    merged.append(a + b)
                    k += 2
                else:
                    merged.append(parts[k])
                    k += 1
            parts = merged
        if len(self._캐시) < 100000:
            self._캐시[word] = parts
        return parts

    def _일반(self, text):
        ids = []
        for piece in _앞자르기(unicodedata.normalize("NFC", text)):
            word = "".join(self.바이트[b] for b in piece.encode("utf-8"))
            ids += [self.낱말[t] for t in self._bpe(word)]
        return ids

    def 나누기(self, text):
        """글 → 토큰 번호(특수 토큰은 원문에 그대로 쓰면 그 번호 하나)."""
        ids, i, 시작 = [], 0, 0
        while i < len(text):
            for s in self.특수차례:
                if text.startswith(s, i):
                    if 시작 < i:
                        ids += self._일반(text[시작:i])
                    ids.append(self.특수[s])
                    i += len(s)
                    시작 = i
                    break
            else:
                i += 1
        if 시작 < len(text):
            ids += self._일반(text[시작:])
        return ids

    def 잇기(self, ids):
        """토큰 번호 → 글(특수 토큰은 그 글자 그대로, 바이트가 끊긴 곳은 바꿈표)."""
        out, 바이트들 = [], bytearray()
        for i in ids:
            t = self.거꾸로[i]
            if i in self.특수.values():
                out.append(바이트들.decode("utf-8", errors="replace"))
                바이트들 = bytearray()
                out.append(t)
            else:
                바이트들 += bytes(self.바이트거꾸로[c] for c in t)
        out.append(바이트들.decode("utf-8", errors="replace"))
        return "".join(out)


기본시스템 = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def 채팅틀(메시지들, 이어쓰기=True):
    """ChatML(Qwen2.5 의 chat_template 의 도구 없는 갈래) — [{"role", "content"}] → 글. 첫 메시지가 system 이 아니면 기본 시스템 말을
    앞에 둔다. 이어쓰기: 끝에 '<|im_start|>assistant\\n'(add_generation_prompt)."""
    s = ""
    if not 메시지들 or 메시지들[0]["role"] != "system":
        s += f"<|im_start|>system\n{기본시스템}<|im_end|>\n"
    for m in 메시지들:
        s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    if 이어쓰기:
        s += "<|im_start|>assistant\n"
    return s


def 채팅틀qwen3(메시지들, 이어쓰기=True, 생각=True):
    """37단계 — Qwen3 의 chat_template(GGUF 의 tokenizer.chat_template — 도구 없는 갈래): 기본 시스템 말이 없다. 마지막 사용자 질문 앞의
    assistant 말은 생각(<think> … </think>)을 떼고 답만, 뒤의 것은 생각을 '<think>\\n…\\n</think>\\n\\n' 로 감싸 둔다. 이어쓰기에 생각 끔
    (enable_thinking=False)이면 '<think>\\n\\n</think>\\n\\n' 를 붙인다."""
    줄 = chr(10)
    s = ""
    if 메시지들 and 메시지들[0]["role"] == "system":
        s += f"<|im_start|>system\n{메시지들[0]['content']}<|im_end|>\n"
    마지막질문 = max([i for i, m in enumerate(메시지들) if m["role"] == "user"], default=len(메시지들) - 1)
    for i, m in enumerate(메시지들):
        if m["role"] == "user" or (m["role"] == "system" and i > 0):
            s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        elif m["role"] == "assistant":
            내용, 생각글 = m["content"], ""
            if "</think>" in 내용:
                생각글 = 내용.split("</think>")[0].rstrip(줄).split("<think>")[-1].lstrip(줄)
                내용 = 내용.split("</think>")[-1].lstrip(줄)
            if i > 마지막질문 and (i == len(메시지들) - 1 or 생각글):
                s += f"<|im_start|>assistant\n<think>\n{생각글.strip(줄)}\n</think>\n\n{내용.lstrip(줄)}"
            else:
                s += f"<|im_start|>assistant\n{내용}"
            s += "<|im_end|>\n"
    if 이어쓰기:
        s += "<|im_start|>assistant\n"
        if not 생각:
            s += "<think>\n\n</think>\n\n"
    return s
