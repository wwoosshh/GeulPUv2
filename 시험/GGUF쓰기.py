#!/usr/bin/env python3
"""모델 평가 2-1(문서/모델평가.md 3.8) — GGUF 의 메타데이터만 고쳐 다시 쓰기(앱/GGUF쓰기.gl)의 GPU 없는 대조(엔진/시험_GGUF쓰기.gl).

작은 GGUF 를 지어(키의 형 13 가지 · 글 배열 · 정수 배열 · 텐서 넷, 맞춤 32 · 64) 바꿀 키(글 · u32 · bool · 지우기 · 새 키)를 주고, 프로그램이 쓴
파일을 이 파일의 GGUF 읽기로 원래와 견준다:
  - 키: 바꾼 키는 새 값(파일에 있던 수 형은 그대로), 지운 키는 없고, 새 키는 끝에, 나머지는 바이트까지 같고 차례도 같다
  - 텐서 정보(이름 · 차원 · 형 · 자리)가 같고, 자료 구역이 바이트까지 같고, 자료 구역의 시작이 맞춤의 배수다
  - 틀린 고침(글 키를 수로, 배열 키를 바꾸기, 키 이름 257 바이트)은 종료 1 로 거절한다

    python -I -X utf8 시험/GGUF쓰기.py check <시험_GGUF쓰기.exe>          (빠른 점검 · CI)
    python -I -X utf8 시험/GGUF쓰기.py model <원래.gguf> <새.gguf>        (실제 파일 둘 — 키의 차이와 자료 구역이 같은가)
"""
import hashlib
import json
import os
import random
import struct
import subprocess
import sys
import tempfile

폭 = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
꼴 = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}


def 글(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def 값(t, v):
    if t == 8:
        return 글(v)
    if t == 9:
        et, xs = v
        return struct.pack("<IQ", et, len(xs)) + b"".join(값(et, x) for x in xs)
    return struct.pack(꼴[t], v)


def 짓기(키들, 텐서들, 맞춤):
    """키들: [(이름, 형, 값)], 텐서들: [(이름, 차원들, 형, 자료 바이트)] — GGUF v3 바이트"""
    머리 = b"GGUF" + struct.pack("<IQQ", 3, len(텐서들), len(키들))
    for k, t, v in 키들:
        머리 += 글(k) + struct.pack("<I", t) + 값(t, v)
    자리 = 0
    자료 = b""
    for 이름, 차원, t, d in 텐서들:
        머리 += 글(이름) + struct.pack("<I", len(차원)) + b"".join(struct.pack("<Q", x) for x in 차원) + struct.pack("<IQ", t, 자리)
        자료 += d
        while len(자료) % 맞춤:
            자료 += b"\0"
        자리 = len(자료)
    while len(머리) % 맞춤:
        머리 += b"\0"
    return 머리 + 자료


def 읽기(b):
    """(키들 [(이름, 형, 값 바이트)], 텐서 정보 바이트, 맞춤, 자료 시작)"""
    assert b[:4] == b"GGUF", "GGUF 가 아니다"
    판, 텐서수, 키수 = struct.unpack_from("<IQQ", b, 4)
    assert 판 == 3
    o = 24
    키들 = []
    맞춤 = 32

    def 건너뛰기(t, o):
        if t in 폭:
            return o + 폭[t]
        if t == 8:
            n, = struct.unpack_from("<Q", b, o)
            return o + 8 + n
        if t == 9:
            et, n = struct.unpack_from("<IQ", b, o)
            o += 12
            for _ in range(n):
                o = 건너뛰기(et, o)
            return o
        raise ValueError(f"형 {t}")

    for _ in range(키수):
        n, = struct.unpack_from("<Q", b, o)
        k = b[o + 8:o + 8 + n].decode("utf-8")
        o += 8 + n
        t, = struct.unpack_from("<I", b, o)
        o += 4
        e = 건너뛰기(t, o)
        if k == "general.alignment" and t == 4:
            맞춤, = struct.unpack_from("<I", b, o)
        키들.append((k, t, b[o:e]))
        o = e
    시작 = o
    for _ in range(텐서수):
        n, = struct.unpack_from("<Q", b, o)
        o += 8 + n
        nd, = struct.unpack_from("<I", b, o)
        o += 4 + 8 * nd + 4 + 8
    정보 = b[시작:o]
    자료시작 = (o + 맞춤 - 1) // 맞춤 * 맞춤
    return 키들, 정보, 맞춤, 자료시작


def 바이트값(t, v):
    return 값(t, v)


def 표본(맞춤, 씨앗):
    r = random.Random(씨앗)
    토큰 = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>", "안녕", "Ġthe", "ĊĊ", "é", "🙂"] + [f"t{i}" for i in range(54)]
    틀 = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    키들 = [("general.architecture", 8, "llama"), ("general.name", 8, "시험 모델 — GeulPU"), ("general.u8", 0, 200), ("general.i8", 1, -5),
          ("general.u16", 2, 60000), ("general.i16", 3, -300), ("general.f32", 6, 1.5), ("general.u64", 10, 1 << 40), ("general.i64", 11, -(1 << 40)),
          ("general.f64", 12, 2.25), ("general.flag", 7, 1), ("tokenizer.ggml.model", 8, "gpt2"), ("tokenizer.ggml.tokens", 9, (8, 토큰)),
          ("tokenizer.ggml.token_type", 9, (5, [3 if t.startswith("<") else 1 for t in 토큰])), ("tokenizer.ggml.bos_token_id", 4, 1),
          ("tokenizer.ggml.eos_token_id", 4, 2), ("tokenizer.ggml.add_bos_token", 7, 1), ("tokenizer.chat_template", 8, 틀),
          ("general.nested", 9, (9, [(4, [1, 2]), (4, [3])]))]
    if 맞춤 != 32:
        키들.insert(1, ("general.alignment", 4, 맞춤))
    텐서들 = [("token_embd.weight", [8, 64], 0, bytes(r.getrandbits(8) for _ in range(8 * 64 * 4))),
           ("blk.0.attn_q.weight", [32, 8], 8, bytes(r.getrandbits(8) for _ in range(8 * 34))),
           ("blk.0.attn_norm.weight", [8], 0, bytes(r.getrandbits(8) for _ in range(32))),
           ("output.weight", [8, 64], 1, bytes(r.getrandbits(8) for _ in range(8 * 64 * 2 + 7)))]
    return 키들, 텐서들


def 기대(키들, 고침):
    """고친 뒤의 키 [(이름, 형, 값 바이트)] — 바꾼 키는 그 자리(수 형은 파일의 것), 지운 키는 빼고, 새 키는 끝에"""
    형이름 = {"string": 8, "u32": 4, "i32": 5, "bool": 7, "u64": 10, "i64": 11}
    by = {c["key"]: c for c in 고침}
    out = []
    for k, t, b in 키들:
        c = by.get(k)
        if c is None:
            out.append((k, t, b))
        elif c["type"] != "remove":
            v = c["value"]
            if t == 8:
                out.append((k, 8, 값(8, v)))
            else:
                out.append((k, t, struct.pack(꼴[t], int(v))))
    있던 = {k for k, _, _ in 키들}
    for c in 고침:
        if c["key"] not in 있던 and c["type"] != "remove":
            t = 형이름[c["type"]]
            v = c["value"]
            out.append((c["key"], t, 값(t, v) if t == 8 else struct.pack(꼴[t], int(v))))
    return out


def 돌리기(exe, 원래, 고침, 폴더, 이름):
    p0 = os.path.join(폴더, 이름 + ".gguf")
    p1 = os.path.join(폴더, 이름 + "-fix.gguf")
    pj = os.path.join(폴더, 이름 + ".json")
    open(p0, "wb").write(원래)
    json.dump({"changes": 고침}, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    if os.path.exists(p1):
        os.remove(p1)
    r = subprocess.run([exe, p0, pj, p1], capture_output=True, timeout=60)
    return r.returncode, (r.stdout + r.stderr).decode("utf-8", "replace").strip(), (open(p1, "rb").read() if os.path.exists(p1) else None)


def 견주기(원래, 새것, 고침):
    키0, 정보0, 맞춤0, 자료0 = 읽기(원래)
    키1, 정보1, 맞춤1, 자료1 = 읽기(새것)
    틀림 = []
    if 키1 != 기대(키0, 고침):
        a, b = 기대(키0, 고침), 키1
        틀림.append(f"키가 기대와 다르다({len(a)} · {len(b)}): " + next((f"{x[0]} · {y[0]}" for x, y in zip(a, b) if x != y), "길이"))
    if 정보0 != 정보1:
        틀림.append("텐서 정보가 다르다")
    if 맞춤0 != 맞춤1 or 자료1 % 맞춤1:
        틀림.append(f"맞춤이 다르다({맞춤0} · {맞춤1} · 자료 시작 {자료1})")
    if 원래[자료0:] != 새것[자료1:]:
        틀림.append("자료 구역이 다르다")
    return 틀림


def check(exe):
    sys.stdout.reconfigure(encoding="utf-8")
    exe = os.path.abspath(exe)
    틀림 = []
    n = 0
    긴틀 = "{# 바꾼 템플릿 #}" + "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}" * 40
    경우 = [
        ("기본", 32, [{"key": "tokenizer.chat_template", "type": "string", "value": 긴틀}, {"key": "tokenizer.ggml.eos_token_id", "type": "u32", "value": 4},
                    {"key": "tokenizer.ggml.add_bos_token", "type": "bool", "value": False}, {"key": "general.name", "type": "remove"},
                    {"key": "tokenizer.ggml.eot_token_id", "type": "u32", "value": 4}, {"key": "general.새 키", "type": "string", "value": "값 ✓"}]),
        ("맞춤64", 64, [{"key": "tokenizer.chat_template", "type": "string", "value": "짧게"}, {"key": "general.u16", "type": "u32", "value": 7},
                      {"key": "general.i64", "type": "i64", "value": -9}]),
        ("그대로", 32, [{"key": "general.없는 키", "type": "remove"}]),
        ("수형유지", 32, [{"key": "general.u8", "type": "u64", "value": 77}, {"key": "general.flag", "type": "bool", "value": False}]),
    ]
    with tempfile.TemporaryDirectory() as 폴더:
        for 이름, 맞춤, 고침 in 경우:
            키들, 텐서들 = 표본(맞춤, n)
            원래 = 짓기(키들, 텐서들, 맞춤)
            rc, 말, 새것 = 돌리기(exe, 원래, 고침, 폴더, 이름)
            n += 1
            if rc != 0 or 새것 is None:
                틀림.append(f"{이름}: 프로그램 실패({rc}) {말[-200:]}")
                continue
            for t in 견주기(원래, 새것, 고침):
                틀림.append(f"{이름}: {t}")
        거절 = [("글을 수로", [{"key": "general.architecture", "type": "u32", "value": 1}]),
              ("배열 바꾸기", [{"key": "tokenizer.ggml.tokens", "type": "string", "value": "x"}]),
              ("긴 키", [{"key": "k" * 257, "type": "u32", "value": 1}])]
        for 이름, 고침 in 거절:
            키들, 텐서들 = 표본(32, n)
            rc, 말, 새것 = 돌리기(exe, 짓기(키들, 텐서들, 32), 고침, 폴더, "거절" + str(n))
            n += 1
            if rc != 1:
                틀림.append(f"거절 '{이름}': 종료 {rc}(1 이어야) {말[-120:]}")
    for t in 틀림[:20]:
        print("틀림 —", t)
    if 틀림:
        print(f"GGUF 다시 쓰기: 못 미침 {len(틀림)}")
        return 1
    print(f"GGUF 다시 쓰기: 통과 — 고침 {len(경우)} 경우(키 · 텐서 정보 · 자료 구역 · 맞춤) · 틀린 고침 {len(거절)} 경우 거절")
    return 0


def 해시(p, a):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        f.seek(a)
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def model(p0, p1):
    """실제 파일 둘 — 머리만 읽고(앞 64 MB 면 넉넉하다) 키의 차이를 보이고, 자료 구역의 SHA-256 이 같은가"""
    sys.stdout.reconfigure(encoding="utf-8")
    def 머리(p):
        with open(p, "rb") as f:
            b = f.read(1 << 26)
        return 읽기(b)
    키0, 정보0, _, 자료0 = 머리(p0)
    키1, 정보1, _, 자료1 = 머리(p1)
    d0, d1 = dict((k, (t, v)) for k, t, v in 키0), dict((k, (t, v)) for k, t, v in 키1)
    for k in sorted(set(d0) | set(d1)):
        if d0.get(k) != d1.get(k):
            print("바뀐 키:", k, "(지움)" if k not in d1 else "(새 키)" if k not in d0 else "")
    같은정보 = 정보0 == 정보1
    h0, h1 = 해시(p0, 자료0), 해시(p1, 자료1)
    print(f"텐서 정보 {'같음' if 같은정보 else '다름'} · 자료 구역 SHA-256 {h0[:16]} · {h1[:16]} {'같음' if h0 == h1 else '다름'}")
    return 0 if 같은정보 and h0 == h1 else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "check":
        sys.exit(check(sys.argv[2]))
    if len(sys.argv) >= 4 and sys.argv[1] == "model":
        sys.exit(model(sys.argv[2], sys.argv[3]))
    print(__doc__)
    sys.exit(2)
