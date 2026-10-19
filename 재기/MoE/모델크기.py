#!/usr/bin/env python3
"""MoE 재기 — GGUF 의 텐서 크기를 전문가 · 비전문가로 나눠 센다(바닥 셈의 바이트).

  python 모델크기.py <gguf> [<출력.json>]

텐서의 바이트는 데이터 자리의 차이로(형식 표 없이 — 정렬 패딩은 다음 텐서 앞까지 포함되나 32 바이트 이하).
"""
import json
import mmap
import re
import struct
import sys
from collections import defaultdict


def 읽기(path):
    f = open(path, "rb")
    b = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    assert b[:4] == b"GGUF"
    ver, nt, nkv = struct.unpack_from("<IQQ", b, 4)
    o = 24

    def s(o):
        n = struct.unpack_from("<Q", b, o)[0]
        return b[o + 8:o + 8 + n].decode("utf-8", "replace"), o + 8 + n

    고정 = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def val(o, t):
        if t in 고정:
            return struct.unpack_from(고정[t], b, o)[0], o + struct.calcsize(고정[t])
        if t == 8:
            return s(o)
        if t == 9:
            et, n = struct.unpack_from("<IQ", b, o)
            o += 12
            if et in 고정:
                sz = struct.calcsize(고정[et])
                return ("배열", et, n), o + n * sz
            out = []
            for _ in range(n):
                v, o = val(o, et)
                out.append(v)
            return out if n <= 64 else ("배열", et, n), o
        raise ValueError(t)

    meta = {}
    for _ in range(nkv):
        k, o = s(o)
        t = struct.unpack_from("<I", b, o)[0]
        v, o = val(o + 4, t)
        meta[k] = v
    텐서 = []
    for _ in range(nt):
        name, o = s(o)
        nd = struct.unpack_from("<I", b, o)[0]
        dims = struct.unpack_from("<%dQ" % nd, b, o + 4)
        o += 4 + 8 * nd
        typ, off = struct.unpack_from("<IQ", b, o)
        o += 12
        텐서.append([name, list(dims), typ, off])
    al = meta.get("general.alignment", 32)
    시작 = (o + al - 1) // al * al
    끝 = len(b)
    차례 = sorted(range(nt), key=lambda i: 텐서[i][3])
    for j, i in enumerate(차례):
        다음 = 텐서[차례[j + 1]][3] if j + 1 < nt else 끝 - 시작
        텐서[i].append(다음 - 텐서[i][3])
    return meta, 텐서


def 요약(meta, 텐서):
    arch = meta["general.architecture"]
    n_exp = meta.get(f"{arch}.expert_count")
    층 = defaultdict(lambda: {"전문가": 0, "공유": 0, "그밖": 0, "전문가텐서": {}})
    전체 = {"전문가": 0, "비전문가": 0, "임베딩": 0, "출력": 0}
    for name, dims, typ, off, nb in 텐서:
        m = re.match(r"blk\.(\d+)\.(.+)", name)
        if name.startswith("token_embd"):
            전체["임베딩"] += nb
            continue
        if m is None:
            전체["출력" if name.startswith("output") else "비전문가"] += nb
            continue
        il, rest = int(m.group(1)), m.group(2)
        if re.match(r"ffn_(gate|up|down|gate_up)_exps", rest):
            층[il]["전문가"] += nb
            층[il]["전문가텐서"][rest] = {"dims": dims, "type": typ, "bytes": nb}
            전체["전문가"] += nb
        elif "shexp" in rest:
            층[il]["공유"] += nb
            전체["비전문가"] += nb
        else:
            층[il]["그밖"] += nb
            전체["비전문가"] += nb
    return {"arch": arch, "n_expert": n_exp, "n_expert_used": meta.get(f"{arch}.expert_used_count"),
            "n_layer": meta.get(f"{arch}.block_count"),
            "meta": {k: v for k, v in meta.items() if k.startswith(arch + ".")},
            "전체": 전체, "층": {il: dict(v) for il, v in sorted(층.items())}}


if __name__ == "__main__":
    meta, 텐서 = 읽기(sys.argv[1])
    r = 요약(meta, 텐서)
    G = 1 << 30
    print("arch", r["arch"], "층", r["n_layer"], "전문가", r["n_expert"], "고름", r["n_expert_used"])
    for k, v in r["meta"].items():
        if not isinstance(v, tuple):
            print(" ", k, v)
    print({k: f"{v / G:.3f} GiB" for k, v in r["전체"].items()})
    전문가층 = [il for il, v in r["층"].items() if v["전문가"]]
    크기 = sorted({v["전문가"] // r["n_expert"] for il, v in r["층"].items() if v["전문가"]})
    print("전문가 하나(층마다 gate+up+down) 바이트:", 크기[:6], "…" if len(크기) > 6 else "", "층 수", len(전문가층))
    for il in 전문가층[:2]:
        print(" 층", il, {k: (v["dims"], v["type"], v["bytes"]) for k, v in r["층"][il]["전문가텐서"].items()},
              "공유", r["층"][il]["공유"], "그밖", r["층"][il]["그밖"])
    if len(sys.argv) > 2:
        json.dump(r, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
