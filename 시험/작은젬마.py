#!/usr/bin/env python3
"""0.13.0 12-3h-4 — 실제 Gemma 4 GGUF(gemma4)에서 층 몇 개만 남긴 작은 GGUF(엔진 시험용 — VRAM 에 다 든다). 0.24.0 — Gemma 3(gemma3)도:
기본은 층 0 … 5(새 번호의 규칙 l mod 6 < 5 가 원래 층의 종류와 같다 — 윈도 층 다섯 · 전체 어텐션 층 하나). 남긴 층의 종류가 새 번호의
규칙과 다르면(예: --층 0,5,6) attention.sliding_window_pattern 을 층마다의 불 배열로 적는다(계약 4.14 의 1 — 배열이면 그 값).

  python -I 시험/작은젬마.py <gemma4 gguf> <출력.gguf> [--층 0,5,6] [--Q8] [--W16 F16|BF16] [--헤드 128]

- 메타데이터는 바이트 그대로 옮기고 gemma4.block_count · 층마다의 배열(attention.head_count_kv · attention.sliding_window_pattern — 남긴
  층의 칸만) · general.name 만 바꾼다(토크나이저 · 헤드 · 창 · RoPE 그대로).
- 텐서: 낱말표 · output_norm · rope_freqs(있으면 출력도)는 그대로, 남긴 층의 텐서는 새 번호(0, 1, …)로 날 바이트 그대로. 기본 층 0 · 5 · 6 은
  12B 에서 윈도 · 전체 · 윈도 층이고, v · down 이 Q6_K 인 층(0 — qkv 의 구간 둘)과 모두 Q4_K 인 층(6 — 구간 하나)이 함께 든다.
- --Q8: 행렬(층의 attn_* · ffn_*, 낱말표 · 출력)을 Q8_0 로 다시 양자화한다 — k-양자화를 풀어(계약/정수계약.py 의 풀기K) 32 칸마다
  d = max|x| / 127, q = 반올림(x / d)(llama.cpp 의 quantize_row_q8_0_ref 와 같은 식). Q8_0 의 길(커널생성 gemma4큰 Q8_0)을 시험하는 파일이다.
- --W16 F16|BF16(0.24.0): 행렬(k-양자화 · Q8_0)을 풀어 F16(가까운 fp16) · BF16(fp32 의 위 16 비트 — 가까운 값)으로 적는다 — 16비트
  블록 가중치의 실행 경로(계약 1.3 — 커널생성 … W16)를 시험하는 파일이다. 값은 원본 파일의 F16 · BF16 과 다르다(양자화된 값을 푼 것).
- --헤드 128(0.24.0 — Gemma 3 의 헤드 256 파일만): 같은 텐서를 헤드 128 로 읽는 파일 — 헤드 수 · KV 헤드 수를 두 배로,
  attention.key_length · value_length(있으면 _swa · rope.dimension_count 도)를 128 로 적고, attn_q_norm · attn_k_norm 은 앞 128 값만
  남긴다(q · k · v · o 행렬의 모양은 그대로 맞는다). 헤드 128 의 실행 경로(27B 와 같은 헤드 크기)를 이 PC 에 드는 파일로 시험하려는 것이다.
값은 실제 모델의 것이지만 층을 줄여 말이 되지 않는다 — 엔진 · CPU 참조의 비트 비교에만 쓴다.
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.normpath(os.path.join(HERE, "..")), "계약"))
import gguf읽기 as GR        # noqa: E402
import 정수계약 as C         # noqa: E402

고정 = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
층배열끝 = (".attention.head_count_kv", ".attention.sliding_window_pattern")        # 구조 이름 뒤 — 층마다의 배열이면 남긴 층의 칸만
행렬 = ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")
K블록 = {GR.Q4_K: 144, GR.Q5_K: 176, GR.Q6_K: 210}


def 문자열(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def 값건너기(b, o, t):
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


def 배열자르기(b, o, 층들):
    """층마다의 배열(o 는 값의 시작 — 원소 타입 · 개수 · 칸)에서 남긴 층의 칸만"""
    et, n = struct.unpack_from("<IQ", b, o)
    w = 고정[et]
    return struct.pack("<IQ", et, len(층들)) + b"".join(bytes(b[o + 12 + w * l:o + 12 + w * (l + 1)]) for l in 층들)


def 풀기행렬(raw, t, K, r0, r1):
    """k-양자화 행렬의 행 [r0, r1) 을 짧은실수 [행][K] 로 — Q4_K · Q5_K 는 q · a − b(아래블록 32 칸), Q6_K 는 q · a(16 칸)"""
    줄 = K // 256 * K블록[t]
    형식, q, a, bm = C.풀기K(t, raw[r0 * 줄:r1 * 줄], r1 - r0, K)
    n = r1 - r0
    if 형식 == "Q6_K":
        return (q.astype(np.float32).reshape(n, K // 16, 16) * a[:, :, None]).reshape(n, K)
    return (q.astype(np.float32).reshape(n, K // 32, 32) * a[:, :, None] - bm[:, :, None]).reshape(n, K)


def Q8로(x):
    """[행][K] 짧은실수 → Q8_0 블록 바이트 [행][K/32][34](반실수 d · int8 32 개)"""
    R, K = x.shape
    xb = x.reshape(R, K // 32, 32).astype(np.float32)
    d = (np.abs(xb).max(-1) / np.float32(127)).astype(np.float32)
    역 = np.zeros_like(d)
    np.divide(np.float32(1), d, out=역, where=d != 0)
    v = xb * 역[:, :, None]
    q = np.clip(np.sign(v) * np.floor(np.abs(v) + np.float32(0.5)), -127, 127).astype(np.int8)       # roundf — 반은 0 에서 멀리
    out = np.empty((R, K // 32, 34), np.uint8)
    out[:, :, :2] = d.astype(np.float16)[:, :, None].view(np.uint8)
    out[:, :, 2:] = q.view(np.uint8)
    return out


def Q8풀기(raw, K, r0, r1):
    """Q8_0 행렬의 행 [r0, r1) 을 fp32 [행][K] 로 — 값 32 개의 블록마다 q · d(d 는 fp16)"""
    줄 = K // 32 * 34
    b = np.frombuffer(raw[r0 * 줄:r1 * 줄], np.uint8).reshape(r1 - r0, K // 32, 34)
    d = np.ascontiguousarray(b[:, :, :2]).view(np.float16).astype(np.float32)
    q = np.ascontiguousarray(b[:, :, 2:]).view(np.int8).astype(np.float32)
    return (q * d).reshape(r1 - r0, K)


def W16로(x, 형식):
    """[행][K] fp32 → F16(fp16 — 가까운 값) 또는 BF16(fp32 의 위 16 비트 — 가까운 값, 같으면 짝수)의 바이트"""
    x = np.ascontiguousarray(x, np.float32)
    if 형식 == "F16":
        h = x.astype(np.float16)
        assert np.isfinite(h).all(), "F16 에 들지 않는 값"
        return h.tobytes()
    u = x.view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    위치 = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] not in ("--층", "--헤드", "--W16")]
    원본, 출력 = 위치[:2]
    Q8 = "--Q8" in sys.argv
    W16 = sys.argv[sys.argv.index("--W16") + 1] if "--W16" in sys.argv else None      # 0.24.0 — 행렬을 F16 · BF16 으로(계약 1.3 의 실행 경로를 시험)
    assert W16 in (None, "F16", "BF16") and not (Q8 and W16), "--W16 뒤에는 F16 · BF16, --Q8 과 함께 쓰지 않는다"
    g = GR.GGUF(원본)
    구조 = g.meta.get("general.architecture")
    assert 구조 in ("gemma4", "gemma3"), 구조
    반헤드 = "--헤드" in sys.argv                            # 0.24.0 — 헤드 256 의 Gemma 3 파일을 헤드 128 로 읽는 파일
    if 반헤드:
        assert sys.argv[sys.argv.index("--헤드") + 1] == "128" and 구조 == "gemma3" and g.meta[구조 + ".attention.key_length"] == 256, \
            "--헤드 는 128, 헤드 256 의 Gemma 3 파일에만"
    정수꼴 = {4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
    헤드열쇠 = tuple(구조 + x for x in (".attention.key_length", ".attention.value_length", ".attention.key_length_swa",
                                    ".attention.value_length_swa", ".rope.dimension_count", ".rope.dimension_count_swa"))
    수열쇠 = tuple(구조 + x for x in (".attention.head_count", ".attention.head_count_kv"))
    기본층 = [0, 5, 6] if 구조 == "gemma4" else [0, 1, 2, 3, 4, 5]
    층들 = [int(x) for x in sys.argv[sys.argv.index("--층") + 1].split(",")] if "--층" in sys.argv else 기본층
    L = g.meta[구조 + ".block_count"]
    assert 층들 and len(set(층들)) == len(층들) and all(0 <= l < L for l in 층들), (층들, L)
    번호 = {l: i for i, l in enumerate(층들)}
    정렬 = g.meta.get("general.alignment", 32)
    b = g.b
    _, nkv = struct.unpack_from("<QQ", b, 8)
    o = 24
    층배열 = tuple(구조 + x for x in 층배열끝)
    윈도열쇠 = 구조 + ".attention.sliding_window_pattern"
    윈도배열 = None                                          # Gemma 3 — 남긴 층의 종류가 새 번호의 규칙과 다르면 층마다의 배열로 적는다
    규칙 = g.meta.get(윈도열쇠)
    if 구조 == "gemma3" and not isinstance(규칙, (list, tuple)):
        P = 6 if 규칙 is None else int(규칙)

        def 윈도층(l):
            return P == 0 or l % P < P - 1
        if [윈도층(l) for l in 층들] != [윈도층(i) for i in range(len(층들))]:
            윈도배열 = struct.pack("<IQ", 7, len(층들)) + bytes(int(윈도층(l)) for l in 층들)       # 불(타입 7)의 배열
    덧글 = " · Q8_0" if Q8 else (f" · {W16}" if W16 else "")
    이름글 = f"{g.meta.get('general.name', 구조)} — 층 {','.join(map(str, 층들))}{덧글} (시험/작은젬마.py)"
    kv = []
    for _ in range(nkv):
        n = struct.unpack_from("<Q", b, o)[0]
        k = bytes(b[o + 8:o + 8 + n]).decode("utf-8")
        o += 8 + n
        t = struct.unpack_from("<I", b, o)[0]
        e = 값건너기(b, o + 4, t)
        raw = bytes(b[o + 4:e])
        if k == 윈도열쇠 and 윈도배열 is not None:              # 정수 P 의 칸은 버린다 — 아래에서 배열로
            o = e
            continue
        if k == 구조 + ".block_count":
            raw = struct.pack({4: "<I", 5: "<i", 10: "<Q", 11: "<q"}[t], len(층들))
        elif k == "general.name":
            raw = 문자열(이름글)
        elif k in 층배열 and t == 9:
            raw = 배열자르기(b, o + 4, 층들)
            if 반헤드 and k == 구조 + ".attention.head_count_kv":       # 층마다의 KV 헤드 수 — 두 배로
                et, n배 = struct.unpack_from("<IQ", raw, 0)
                w = 고정[et]
                값들 = [int.from_bytes(raw[12 + w * i:12 + w * (i + 1)], "little") * 2 for i in range(n배)]
                raw = raw[:12] + b"".join(v.to_bytes(w, "little") for v in 값들)
        elif 반헤드 and k in 수열쇠 and t in 정수꼴:
            raw = struct.pack(정수꼴[t], struct.unpack(정수꼴[t], raw)[0] * 2)
        elif 반헤드 and k in 헤드열쇠 and t in 정수꼴:
            assert struct.unpack(정수꼴[t], raw)[0] == 256, (k, raw)
            raw = struct.pack(정수꼴[t], 128)
        elif k == "general.file_type" and Q8:
            raw = struct.pack({4: "<I", 5: "<i"}[t], 7)                 # LLAMA_FTYPE_MOSTLY_Q8_0
        elif k == "general.file_type" and W16:
            raw = struct.pack({4: "<I", 5: "<i"}[t], 1 if W16 == "F16" else 32)       # LLAMA_FTYPE_MOSTLY_F16 · MOSTLY_BF16
        kv.append((k, t, raw))
        o = e
    if 윈도배열 is not None:
        kv.append((윈도열쇠, 9, 윈도배열))
    텐서 = []                                               # (새 이름, 타입, dims, 원본 이름, 바이트 수, 다시 양자화)
    for 이름, (t, dims, off) in g.tensors.items():
        새 = 이름
        if 이름.startswith("blk."):
            l = int(이름.split(".")[1])
            if l not in 번호:
                continue
            새 = f"blk.{번호[l]}." + ".".join(이름.split(".")[2:])
        꼬리 = ".".join(이름.split(".")[2:]) if 이름.startswith("blk.") else 이름
        큰행렬 = 꼬리 in 행렬 or 이름 in ("token_embd.weight", "output.weight")
        다시 = Q8 and t in K블록 and 큰행렬
        if W16 and 큰행렬:                                    # 0.24.0 — k-양자화 · Q8_0 의 행렬을 풀어 F16 · BF16 으로
            assert t in K블록 or t == GR.Q8_0, f"{이름}: {W16} 로 바꾸지 않는 형식 {t}"
            다시 = W16
            n = dims[0] * dims[1] * 2
            t2 = GR.F16 if W16 == "F16" else GR.BF16
        elif 다시:
            n = dims[0] * dims[1] // 32 * 34
            t2 = GR.Q8_0
        else:
            assert not (Q8 and t in K블록), f"{이름}: Q8_0 로 바꾸지 않는 k-양자화 텐서"
            vb, bb = GR.블록[t]
            if 반헤드 and 꼬리 in ("attn_q_norm.weight", "attn_k_norm.weight"):      # 헤드마다의 정규화 가중치 — 앞 128 값만
                assert tuple(dims) == (256,) and (vb, bb) == (1, 4), (이름, dims, t)
                dims = (128,)
            n = int(np.prod(dims)) // vb * bb
            t2 = t
        텐서.append((새, t2, list(dims), 이름, n, 다시))
    머리 = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(텐서), len(kv)))
    for k, t, raw in kv:
        머리 += 문자열(k) + struct.pack("<I", t) + raw
    off = 0
    for 새, t2, dims, _, n, _ in 텐서:
        머리 += 문자열(새) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", t2, off)
        off += (n + 정렬 - 1) // 정렬 * 정렬
    머리 += b"\0" * ((-len(머리)) % 정렬)
    with open(출력, "wb") as f:
        f.write(머리)
        for 새, t2, dims, 이름, n, 다시 in 텐서:
            t, _, raw = g.원(이름)
            if 다시 in ("F16", "BF16"):
                K, N = dims[0], dims[1]
                for r0 in range(0, N, 4096):
                    r1 = min(N, r0 + 4096)
                    f.write(W16로(풀기행렬(raw, t, K, r0, r1) if t in K블록 else Q8풀기(raw, K, r0, r1), 다시))
            elif 다시:
                K, N = dims[0], dims[1]
                for r0 in range(0, N, 4096):
                    f.write(Q8로(풀기행렬(raw, t, K, r0, min(N, r0 + 4096))).tobytes())
            else:
                f.write(raw.tobytes()[:n])                    # --헤드 128 의 정규화 가중치는 앞 n 바이트만(그 밖은 전부)
            f.write(b"\0" * ((-n) % 정렬))
    print(f"{출력}: 층 {층들} → {len(층들)} 층{덧글} · 텐서 {len(텐서)} · {os.path.getsize(출력) / 2 ** 20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
