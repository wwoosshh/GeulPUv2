#!/usr/bin/env python3
"""GeulPUv2 ②단계 — 도구(tools)를 낀 채팅 틀의 참조 글: GGUF 의 tokenizer.chat_template 을 jinja2 로 그린다(transformers 와 같은 환경 —
trim_blocks · lstrip_blocks · loopcontrols, tojson = json.dumps(ensure_ascii=False, 열쇠 차례 그대로)). jinja2 가 있는 곳(WSL Ubuntu)에서:

  python3 시험/도구틀참조.py <모델.gguf> 시험/도구틀경우.json <출력.json> [--생각끔]

0.12.0 12-G1b — 서버가 이 틀을 jinja 해석기(엔진/채팅틀.gl)로 그대로 실행하므로, 메시지 다듬기도 서버와 같은 일반 규칙이다(모델마다의
갈래 없이): role 의 developer → system · ipython → tool, content 조각은 이어 붙인 글 · null · 없음은 빈 글, 호출의 arguments 글은 JSON 으로
풀어서(틀이 'arguments is string' 으로 글을 그대로 쓰면 글 그대로 — Qwen3), 틀이 system 말을 그리지 않으면(그려 보아 정한다 — Mistral
v0.3) 맨 앞 system 말을 첫 user 말 앞에 '<system>\n\n' 으로. 맥락: bos_token · eos_token(GGUF 의 번호의 글), date_string '26 Jul 2024',
--생각끔 이면 enable_thinking 거짓. tools 는 서버가 그 호출 꼴을 읽는 틀만(Qwen <tool_call> JSON · Llama 3 JSON · Qwen3.5 · Coder XML ·
MiniCPM XML — 아니면 서버는 400 이라 참조도 오류). 출력: {경우 id: 그린 글 | {"error": 까닭}}. 대조는 시험/채팅틀대조.py(GPU 없이 —
엔진/시험_채팅틀.gl) · 시험/도구틀.py(서버로).
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


def 내용글(m):
    c = m.get("content")
    if isinstance(c, list):
        return "".join(p["text"] for p in c)
    if c is None:
        return ""
    return c


def 고치기(msgs, 인자글, 시스템받음):
    """서버(엔진/채팅틀.gl 의 맥락메시지쓰기)와 같은 다듬기"""
    out = []
    ms = list(msgs)
    시스템 = None
    역 = {"developer": "system", "ipython": "tool"}
    if not 시스템받음 and ms and 역.get(ms[0].get("role"), ms[0].get("role")) == "system":
        시스템 = 내용글(ms[0])
        ms = ms[1:]
    for m in ms:
        m = dict(m)
        m["role"] = 역.get(m.get("role"), m.get("role"))
        c = 내용글(m)
        if 시스템 is not None and m["role"] == "user":
            c = 시스템 + "\n\n" + c
            시스템 = None
        m["content"] = c
        if isinstance(m.get("tool_calls"), list):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                f = tc.get("function") if isinstance(tc, dict) else None
                if isinstance(f, dict) and isinstance(f.get("arguments"), str) and not 인자글:
                    try:
                        f["arguments"] = json.loads(f["arguments"])
                    except ValueError:
                        pass
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def 도구꼴(틀글, 앞꼴, 낱말):
    """서버의 틀도구꼴 — 1 Qwen <tool_call> JSON · 2 Llama 3 JSON · 3 Qwen3.5 · Coder XML · 4 MiniCPM XML · 0 받지 않음"""
    낱 = set(낱말 or [])
    꼴 = 0
    if "<function=" in 틀글 and "<parameter=" in 틀글:
        꼴 = 3
    elif "<function name=" in 틀글 and "<param name=" in 틀글:
        꼴 = 4
    elif "<tool_call>" in 틀글 and "tojson" in 틀글:
        꼴 = 1
    elif 앞꼴 in ("llama-bpe", "llama3", "llama-v3") and "<|start_header_id|>" in 틀글 and "tojson" in 틀글:
        꼴 = 2
    if 꼴 in (1, 3) and not {"<tool_call>", "</tool_call>"} <= 낱:
        꼴 = 0
    if 꼴 == 4 and not {"<function", "</function>", "<param", "</param>"} <= 낱:
        꼴 = 0
    if "tools" not in 틀글:
        꼴 = 0
    return 꼴


def 참조들(gguf, 경우, 생각끔):
    """경우 파일(dict) 의 경우마다 참조 글 — {id: 글 | {"error": 까닭}}"""
    meta = gguf_meta(gguf, {"tokenizer.chat_template", "tokenizer.ggml.tokens", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                            "tokenizer.ggml.pre"})
    틀글 = meta["tokenizer.chat_template"]
    낱말 = meta.get("tokenizer.ggml.tokens")
    bos = 낱말[meta["tokenizer.ggml.bos_token_id"]] if 낱말 and meta.get("tokenizer.ggml.bos_token_id") is not None else ""
    eos = 낱말[meta["tokenizer.ggml.eos_token_id"]] if 낱말 and meta.get("tokenizer.ggml.eos_token_id") is not None else ""
    qwen3 = "enable_thinking" in 틀글
    꼴 = 도구꼴(틀글, meta.get("tokenizer.ggml.pre", ""), 낱말)
    인자글 = "arguments is string" in 틀글
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    틀 = env.from_string(틀글)
    기본 = dict(add_generation_prompt=True, bos_token=bos, eos_token=eos, date_string="26 Jul 2024")
    try:
        시스템받음 = "GEULPU_SYSTEM_PROBE" in 틀.render(messages=[{"role": "system", "content": "GEULPU_SYSTEM_PROBE"}, {"role": "user", "content": "hi"}],
                                                   **dict(기본, bos_token="", eos_token=""))
    except Exception:                               # noqa: BLE001 — system 을 받지 않는 틀(Mistral v0.3)
        시스템받음 = False
    도구표 = 경우["도구"]
    out = {}
    for c in 경우["경우"]:
        if c.get("think_only") and not qwen3:
            continue
        kw = dict(기본, messages=고치기(c["messages"], 인자글, 시스템받음))
        도구씀 = bool(c.get("tools")) and c.get("tool_choice") != "none"
        if 도구씀 and 꼴 == 0:                    # 서버는 그 호출 꼴을 읽지 못하는 틀의 tools 를 400 으로
            out[c["id"]] = {"error": "이 틀의 도구 호출 꼴은 서버가 받지 않는다"}
            continue
        if 도구씀:
            kw["tools"] = [도구표[t] for t in c["tools"]]
        if 생각끔:
            kw["enable_thinking"] = False
        for k, v in (c.get("chat_template_kwargs") or {}).items():
            if k not in ("enable_thinking", "date_string"):
                kw[k] = v
        try:
            out[c["id"]] = 틀.render(**kw)
        except Exception as e:                      # noqa: BLE001 — Llama 의 호출 둘 등 — 서버도 400 이어야 한다
            out[c["id"]] = {"error": f"{type(e).__name__}: {e}"}
    return out, 꼴, 시스템받음


def main():
    gguf, 경우파일, 출력 = sys.argv[1:4]
    생각끔 = "--생각끔" in sys.argv[4:]
    경우 = json.load(open(경우파일, encoding="utf-8"))
    out, 꼴, 시스템받음 = 참조들(gguf, 경우, 생각끔)
    json.dump(out, open(출력, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
    print(f"{len(out)} 경우 · 도구 꼴 {꼴} · system {'그림' if 시스템받음 else '첫 user 말에 붙임'} · 생각끔 {생각끔} → {출력}")


if __name__ == "__main__":
    main()
