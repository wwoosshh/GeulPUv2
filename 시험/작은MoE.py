#!/usr/bin/env python3
"""0.12.0 12-3g — 실제 MoE GGUF(qwen35moe)에서 앞 층 · 앞 전문가만 남긴 작은 GGUF(엔진 시험용 — VRAM 에 다 든다).

  python -I 시험/작은MoE.py <qwen35moe gguf> <출력.gguf> [--층 4] [--전문가 32]

- 메타데이터는 바이트 그대로 옮기고 qwen35moe.block_count · qwen35moe.expert_count · general.name 만 바꾼다(토크나이저 · 고름 수 그대로 —
  고름 수는 남긴 전문가 수 이하여야 한다).
- 텐서: 낱말표 · 출력 · output_norm 그대로, 층 0 … 층 − 1 의 텐서 그대로(날 바이트), 전문가 텐서(ffn_*_exps — [전문가][N][K])와
  라우터(ffn_gate_inp — [전문가][너비])는 앞 전문가 수만큼(맨 바깥 차원이 전문가라 앞 바이트를 자른다). 층의 차례(전체 어텐션 간격)는
  파일의 것 그대로라, 층 수는 간격의 배수로 두면 어텐션 층이 들어간다.
값은 실제 모델의 것이지만 층 · 전문가를 줄여 말이 되지 않는다 — 엔진 · CPU 참조의 비트 비교에만 쓴다.
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.normpath(os.path.join(HERE, "..")), "계약"))
import gguf읽기 as GR        # noqa: E402

정렬 = 32


def 문자열(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def 값건너기(b, o, t):
    고정 = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in 고정:
        return o + 고정[t]
    if t == 8:
        return o + 8 + struct.unpack_from("<Q", b, o)[0]
    if t == 9:
        et, n = struct.unpack_from("<IQ", b, o)
        o += 12
        if et in 고정:
            return o + 고정[et] * n
        for _ in range(n):
            o = 값건너기(b, o, et)
        return o
    raise ValueError(f"GGUF 값 타입 {t}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    위치 = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and not sys.argv[i - 1].startswith("--")]
    원본, 출력 = 위치[:2]
    층 = int(sys.argv[sys.argv.index("--층") + 1]) if "--층" in sys.argv else 4
    E2 = int(sys.argv[sys.argv.index("--전문가") + 1]) if "--전문가" in sys.argv else 32
    g = GR.GGUF(원본)
    assert g.meta.get("general.architecture") == "qwen35moe", g.meta.get("general.architecture")
    E = g.meta["qwen35moe.expert_count"]
    assert 1 <= E2 <= E and g.meta["qwen35moe.expert_used_count"] <= E2, (E, E2)
    assert 1 <= 층 <= g.meta["qwen35moe.block_count"]
    b = g.b
    _, nkv = struct.unpack_from("<QQ", b, 8)
    o = 24
    바꿈 = {"qwen35moe.block_count": struct.pack("<I", 층), "qwen35moe.expert_count": struct.pack("<I", E2),
          "general.name": 문자열(f"{g.meta.get('general.name', 'qwen35moe')} — 층 {층} · 전문가 {E2} (시험/작은MoE.py)")}
    kv = []
    for _ in range(nkv):
        n = struct.unpack_from("<Q", b, o)[0]
        k = bytes(b[o + 8:o + 8 + n]).decode("utf-8")
        o += 8 + n
        t = struct.unpack_from("<I", b, o)[0]
        e = 값건너기(b, o + 4, t)
        raw = 바꿈.pop(k) if k in 바꿈 else bytes(b[o + 4:e])
        kv.append((k, t, raw))
        o = e
    assert not 바꿈, 바꿈
    블록 = GR.블록
    텐서 = []                                               # (이름, 타입, dims, 원본 바이트 시작, 바이트 수)
    for 이름, (t, dims, off) in g.tensors.items():
        if 이름.startswith("blk."):
            l = int(이름.split(".")[1])
            if l >= 층:
                continue
        vb, bb = 블록[t]
        n = int(np.prod(dims)) // vb * bb
        dims = list(dims)
        if "_exps." in 이름 or 이름.endswith("ffn_gate_inp.weight"):
            assert dims[-1] == E, (이름, dims)
            n = n // E * E2
            dims[-1] = E2
        텐서.append((이름, t, dims, g.data0 + off, n))
    머리 = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(텐서), len(kv)))
    for k, t, raw in kv:
        머리 += 문자열(k) + struct.pack("<I", t) + raw
    off = 0
    for 이름, t, dims, _, n in 텐서:
        머리 += 문자열(이름) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", t, off)
        off += (n + 정렬 - 1) // 정렬 * 정렬
    머리 += b"\0" * ((-len(머리)) % 정렬)
    with open(출력, "wb") as f:
        f.write(머리)
        for _, _, _, s, n in 텐서:
            f.write(b[s:s + n])
            f.write(b"\0" * ((-n) % 정렬))
    print(f"{출력}: 층 {층} · 전문가 {E2} · 텐서 {len(텐서)} · {os.path.getsize(출력) / 2 ** 20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
