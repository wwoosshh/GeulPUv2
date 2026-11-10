#!/usr/bin/env python3
"""0.24.0 24-2 — 실제 deepseek2 GGUF(GLM-4.7-Flash · Kanana-2)에서 층 몇 개만 남긴 작은 GGUF(엔진 시험용 — 시험_7B 의 실행 경로 다섯 · CPU 참조의
비트, 구조 회귀). 시험/작은젬마.py 와 같은 꼴.

  python -I 시험/작은deepseek2.py <deepseek2 gguf> <출력.gguf> [--층 0,1,2]

- 메타데이터는 바이트 그대로 옮기고 deepseek2.block_count · deepseek2.leading_dense_block_count(남긴 층 가운데 dense 의 수 — dense 가 앞에 와야
  한다) · general.name 만 바꾼다(토크나이저 · 헤드 · MLA · 라우팅 그대로).
- 텐서: 낱말표 · output_norm · output 은 그대로, 남긴 층의 텐서는 새 번호(0, 1, …)로 날 바이트 그대로. 기본 층 0 · 1 · 2 는 dense 하나와 MoE
  둘(GLM-4.7-Flash 의 층 1 · 2 는 전문가 down 이 Q4_K · Q6_K 로 다르다 — 커널 표의 두 형식)이다.
값은 실제 모델의 것이지만 층을 줄여 말이 되지 않는다 — 엔진 · CPU 참조의 비트 비교에만 쓴다.
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.normpath(os.path.join(HERE, "..")), "계약"))
sys.path.insert(0, HERE)
import gguf읽기 as GR        # noqa: E402
from 작은젬마 import 값건너기, 문자열    # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    위치 = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and not sys.argv[i - 1].startswith("--층")]
    if len(위치) < 2:
        sys.exit("사용법: python -I 시험/작은deepseek2.py <deepseek2 gguf> <출력.gguf> [--층 0,1,2]")
    원본, 출력 = 위치[:2]
    층들 = [int(x) for x in sys.argv[sys.argv.index("--층") + 1].split(",")] if "--층" in sys.argv else [0, 1, 2]
    g = GR.GGUF(원본)
    assert g.meta.get("general.architecture") == "deepseek2", g.meta.get("general.architecture")
    L = g.meta["deepseek2.block_count"]
    dense = g.meta.get("deepseek2.leading_dense_block_count", 0)
    assert 층들 and len(set(층들)) == len(층들) and all(0 <= l < L for l in 층들), (층들, L)
    새dense = sum(1 for l in 층들 if l < dense)
    assert all(l < dense for l in 층들[:새dense]) and all(l >= dense for l in 층들[새dense:]), "dense 층이 앞에 와야 한다"
    번호 = {l: i for i, l in enumerate(층들)}
    정렬 = g.meta.get("general.alignment", 32)
    b = g.b
    _, nkv = struct.unpack_from("<QQ", b, 8)
    o = 24
    이름글 = f"{g.meta.get('general.name', 'deepseek2')} — 층 {','.join(map(str, 층들))} (시험/작은deepseek2.py)"
    kv = []
    for _ in range(nkv):
        n = struct.unpack_from("<Q", b, o)[0]
        k = bytes(b[o + 8:o + 8 + n]).decode("utf-8")
        o += 8 + n
        t = struct.unpack_from("<I", b, o)[0]
        e = 값건너기(b, o + 4, t)
        raw = bytes(b[o + 4:e])
        if k == "deepseek2.block_count":
            raw = struct.pack({4: "<I", 5: "<i", 10: "<Q", 11: "<q"}[t], len(층들))
        elif k == "deepseek2.leading_dense_block_count":
            raw = struct.pack({4: "<I", 5: "<i", 10: "<Q", 11: "<q"}[t], 새dense)
        elif k == "general.name":
            raw = 문자열(이름글)
        kv.append((k, t, raw))
        o = e
    텐서 = []
    for 이름, (t, dims, off) in g.tensors.items():
        새 = 이름
        if 이름.startswith("blk."):
            l = int(이름.split(".")[1])
            if l not in 번호:
                continue
            새 = f"blk.{번호[l]}." + ".".join(이름.split(".")[2:])
        vb, bb = GR.블록[t]
        텐서.append((새, t, list(dims), 이름, int(np.prod(dims)) // vb * bb))
    머리 = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(텐서), len(kv)))
    for k, t, raw in kv:
        머리 += 문자열(k) + struct.pack("<I", t) + raw
    off = 0
    for 새, t, dims, _, n in 텐서:
        머리 += 문자열(새) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", t, off)
        off += (n + 정렬 - 1) // 정렬 * 정렬
    머리 += b"\0" * ((-len(머리)) % 정렬)
    with open(출력, "wb") as f:
        f.write(머리)
        for 새, t, dims, 이름, n in 텐서:
            f.write(g.원(이름)[2].tobytes())
            f.write(b"\0" * ((-n) % 정렬))
    print(f"{출력}: 층 {층들} → {len(층들)} 층(dense {새dense}) · 텐서 {len(텐서)} · {os.path.getsize(출력) / 2 ** 20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
