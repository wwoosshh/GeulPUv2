#!/usr/bin/env python3
"""GeulPUv2 ②단계 — 도구(tools)를 낀 채팅 틀의 참조 글: GGUF 의 tokenizer.chat_template 을 jinja2 로 그린다(transformers 와 같은 환경 —
trim_blocks · lstrip_blocks · loopcontrols, tojson = json.dumps(ensure_ascii=False, 열쇠 차례 그대로)). jinja2 가 있는 곳(WSL Ubuntu)에서:

  python3 시험/도구틀참조.py <모델.gguf> 시험/도구틀경우.json <출력.json> [--생각끔]

메시지는 서버가 받는 OpenAI 꼴을 서버와 같은 규칙으로 고쳐 넘긴다(문서/도구.md): 호출의 arguments 글은 JSON 으로 풀어서(vLLM · llama.cpp
처럼 — Qwen3 틀은 글을 그대로 넣으므로 Qwen3 만 글 그대로), content 조각은 이어 붙인 글로, Qwen3 의 null content 는 빈 글로. Llama 3 의
날짜는 틀의 기본(26 Jul 2024 — 서버도 같다). 출력: {경우 id: 그린 글 | {"error": 까닭}}. 대조는 시험/도구틀.py 가 서버로 한다.
"""
import json
import mmap
import struct
import sys

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment


def gguf_meta(path, keys):
    f = open(path, "rb")
    b = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    assert b[:4] == b"GGUF"
    _, _, nkv = struct.unpack_from("<IQQ", b, 4)
    o = 24
    fm = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def s(o):
        n = struct.unpack_from("<Q", b, o)[0]
        return b[o + 8:o + 8 + n].decode("utf-8"), o + 8 + n

    def val(o, t):
        if t in fm:
            return struct.unpack_from(fm[t], b, o)[0], o + struct.calcsize(fm[t])
        if t == 8:
            return s(o)
        if t == 9:
            et, n = struct.unpack_from("<IQ", b, o)
            o += 12
            if et in fm:
                return None, o + n * struct.calcsize(fm[et])
            out = []
            for _ in range(n):
                v, o = val(o, et)
                out.append(v)
            return out, o
        raise ValueError(t)

    out = {}
    for _ in range(nkv):
        k, o = s(o)
        t = struct.unpack_from("<I", b, o)[0]
        v, o = val(o + 4, t)
        if k in keys:
            out[k] = v
    return out


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def raise_exception(msg):
    raise jinja2.exceptions.TemplateError(msg)


def 고치기(msgs, qwen3):
    out = []
    for m in msgs:
        m = dict(m)
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = "".join(p["text"] for p in c)
        if qwen3 and m.get("content") is None:
            m["content"] = ""
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                a = tc["function"].get("arguments")
                if isinstance(a, str) and not qwen3:
                    try:
                        tc["function"]["arguments"] = json.loads(a)
                    except ValueError:
                        pass
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def main():
    gguf, 경우파일, 출력 = sys.argv[1:4]
    생각끔 = "--생각끔" in sys.argv[4:]
    meta = gguf_meta(gguf, {"tokenizer.chat_template"})
    틀글 = meta["tokenizer.chat_template"]
    라마 = "<|start_header_id|>" in 틀글
    qwen3 = "enable_thinking" in 틀글
    bos = "<|begin_of_text|>" if 라마 else "<|endoftext|>"
    eos = "<|eot_id|>" if 라마 else "<|im_end|>"
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    틀 = env.from_string(틀글)
    경우 = json.load(open(경우파일, encoding="utf-8"))
    도구표 = 경우["도구"]
    out = {}
    for c in 경우["경우"]:
        if c.get("think_only") and not qwen3:
            continue
        kw = dict(messages=고치기(c["messages"], qwen3), add_generation_prompt=True, bos_token=bos, eos_token=eos)
        if c.get("tools") and c.get("tool_choice") != "none":
            kw["tools"] = [도구표[t] for t in c["tools"]]
        if qwen3 and 생각끔:
            kw["enable_thinking"] = False
        try:
            out[c["id"]] = 틀.render(**kw)
        except Exception as e:                      # Llama 의 호출 둘 등 — 서버도 400 이어야 한다
            out[c["id"]] = {"error": f"{type(e).__name__}: {e}"}
    json.dump(out, open(출력, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
    print(f"{len(out)} 경우 · {'llama' if 라마 else ('qwen3' if qwen3 else 'qwen')} · 생각끔 {생각끔} → {출력}")


if __name__ == "__main__":
    main()
