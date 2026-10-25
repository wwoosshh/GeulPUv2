#!/usr/bin/env python3
"""Qwen3.5 · 3.8(구조 qwen35)의 합성 GGUF — 실제 파일의 머리(메타데이터 · 텐서 목록 — 앞부분만 받은 것이어도 된다)에서 층만 줄이고 값은
무작위로 채운다. 모양 · 형식 · 토크나이저 · 템플릿은 그 파일의 것 그대로라, 이 PC 에 들지 않는 모델(Qwen3.8-27B)의 모양과 업로더마다의
형식 조합(텐서마다 다른 양자화 · MTP 층)을 엔진의 실행 경로 · CPU 참조로 비트까지 시험할 수 있다(값이 무작위라 말은 되지 않는다).

  python -I 시험/합성35.py <머리 또는 gguf> <출력.gguf> [--층 4] [--MTP] [--형식 원본|Q8_0] [--βα 원본|Q8_0] [--씨앗 20261025]

- --층 n: 층 0 … n − 1 만(전체 어텐션 간격의 배수로 두면 어텐션 층이 든다). block_count 는 n(+ MTP 층 수), attention.recurrent_layers 가
  있으면 앞 n 개(+ MTP 의 것).
- --MTP: 파일에 MTP 층(nextn_predict_layers)이 있으면 그 층(blk.<원래 층 수>.*)을 blk.<n>.* 로 남긴다 — 엔진이 그 층을 건너뛰는지 본다.
  없으면 nextn_predict_layers 를 0 으로.
- --형식 원본: 텐서마다 머리의 형식 그대로. Q8_0: 행렬(2 차원 — ssm_conv1d 말고)을 모두 Q8_0 으로(1 차원의 F32 는 그대로).
- --βα Q8_0: ssm_alpha · ssm_beta 만 Q8_0 으로(나머지는 --형식 대로).
- 값: Q8_0 · Q4_0 · k-양자화는 날 바이트를 무작위로 하되 척도(반실수 d · dmin)는 작은 양수(행렬마다 고른 범위)로, F32 는 텐서의 뜻에 맞춰
  (정규화 무게 1 ± 0.1, ssm_a = −exp(U(−1, 2)), ssm_dt.bias U(−2, 0), 합성곱 N(0, 0.3), 그 밖 N(0, 0.02)).
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.normpath(os.path.join(HERE, "..")), "계약"))
import gguf읽기 as GR        # noqa: E402

정렬 = 32
F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, BF16 = 0, 1, 2, 8, 12, 13, 14, 30
블록 = {F32: (1, 4), F16: (1, 2), BF16: (1, 2), Q4_0: (32, 18), Q8_0: (32, 34), Q4_K: (256, 144), Q5_K: (256, 176), Q6_K: (256, 210)}


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
            return o + n * 고정[et]
        for _ in range(n):
            o = 값건너기(b, o, et)
        return o
    raise ValueError(f"GGUF 값 타입 {t}")


def 반실수(r, n, lo, hi):
    return r.uniform(lo, hi, n).astype(np.float16).view(np.uint16)


def 무작위(r, t, n, 이름):
    """형식 t 의 값 n 개를 날 바이트로"""
    vb, bb = 블록[t]
    nb = n // vb
    if t == F32:
        if 이름.endswith("norm.weight"):
            v = 1.0 + r.normal(0, 0.1, n)
        elif 이름.endswith("ssm_a"):
            v = -np.exp(r.uniform(-1, 2, n))
        elif 이름.endswith("ssm_dt.bias"):
            v = r.uniform(-2, 0, n)
        elif "ssm_conv1d" in 이름:
            v = r.normal(0, 0.3, n)
        else:
            v = r.normal(0, 0.02, n)
        return v.astype(np.float32).tobytes()
    if t in (F16, BF16):
        v = r.normal(0, 0.02, n).astype(np.float32)
        return (v.astype(np.float16).view(np.uint16) if t == F16 else (v.view(np.uint32) >> 16).astype(np.uint16)).tobytes()
    raw = np.frombuffer(r.bytes(nb * bb), np.uint8).copy().reshape(nb, bb)
    if t == Q8_0:
        raw[:, 0:2] = 반실수(r, nb, 2e-4, 6e-4).view(np.uint8).reshape(nb, 2)
    elif t == Q4_0:
        raw[:, 0:2] = 반실수(r, nb, 1e-3, 3e-3).view(np.uint8).reshape(nb, 2)
    elif t in (Q4_K, Q5_K):
        raw[:, 0:2] = 반실수(r, nb, 1e-4, 3e-4).view(np.uint8).reshape(nb, 2)
        raw[:, 2:4] = 반실수(r, nb, 0, 2e-4).view(np.uint8).reshape(nb, 2)
    elif t == Q6_K:
        raw[:, 208:210] = 반실수(r, nb, 2e-5, 6e-5).view(np.uint8).reshape(nb, 2)
    else:
        raise ValueError(f"{이름}: 다루지 않는 형식 {t}")
    return raw.tobytes()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    위치 = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and not sys.argv[i - 1].startswith("--")]
    원본, 출력 = 위치[:2]
    층 = int(sys.argv[sys.argv.index("--층") + 1]) if "--층" in sys.argv else 4
    형식 = sys.argv[sys.argv.index("--형식") + 1] if "--형식" in sys.argv else "원본"
    βα = sys.argv[sys.argv.index("--βα") + 1] if "--βα" in sys.argv else "원본"
    씨앗 = int(sys.argv[sys.argv.index("--씨앗") + 1]) if "--씨앗" in sys.argv else 20261025
    MTP = "--MTP" in sys.argv
    g = GR.GGUF(원본)
    arch = g.meta.get("general.architecture")
    assert arch in ("qwen35", "qwen35moe"), arch
    n원 = g.meta[f"{arch}.block_count"]
    nn = g.meta.get(f"{arch}.nextn_predict_layers", 0)
    n주 = n원 - nn
    assert 1 <= 층 <= n주, (층, n주)
    남는MTP = nn if MTP else 0
    b = g.b
    _, nkv = struct.unpack_from("<QQ", b, 8)
    o = 24
    바꿈 = {f"{arch}.block_count": struct.pack("<I", 층 + 남는MTP),
          "general.name": 문자열(f"{g.meta.get('general.name', arch)} — 합성(층 {층}{' + MTP' if 남는MTP else ''} · 무작위 값 — 시험/합성35.py)")}
    if f"{arch}.nextn_predict_layers" in g.meta:
        바꿈[f"{arch}.nextn_predict_layers"] = struct.pack("<I", 남는MTP)
    rl = g.meta.get(f"{arch}.attention.recurrent_layers")
    if rl is not None:
        새 = list(rl[:층]) + list(rl[n주:n주 + 남는MTP])
        바꿈[f"{arch}.attention.recurrent_layers"] = struct.pack("<IQ", 7, len(새)) + bytes(int(bool(x)) for x in 새)
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
    assert not [k for k in 바꿈 if k != "general.name"], 바꿈
    if "general.name" in 바꿈:
        kv.append(("general.name", 8, 바꿈.pop("general.name")))
    텐서 = []                                               # (새 이름, 타입, dims)
    for 이름, (t, dims, off) in g.tensors.items():
        새이름 = 이름
        if 이름.startswith("blk."):
            l = int(이름.split(".")[1])
            if l >= 층 and not (남는MTP and n주 <= l < n주 + 남는MTP):
                continue
            if l >= n주:
                새이름 = f"blk.{층 + (l - n주)}." + 이름.split(".", 2)[2]
        행렬 = len(dims) >= 2 and "ssm_conv1d" not in 이름 and dims[0] % 32 == 0
        if 행렬 and (형식 == "Q8_0" or (βα == "Q8_0" and ("ssm_alpha" in 이름 or "ssm_beta" in 이름))):
            t = Q8_0
        텐서.append((새이름, t, list(dims)))
    r = np.random.default_rng(씨앗)
    머리 = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(텐서), len(kv)))
    for k, t, raw in kv:
        머리 += 문자열(k) + struct.pack("<I", t) + raw
    off = 0
    크기 = []
    for 이름, t, dims in 텐서:
        vb, bb = 블록[t]
        n = int(np.prod(dims))
        assert n % vb == 0, (이름, dims, t)
        크기.append(n // vb * bb)
        머리 += 문자열(이름) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", t, off)
        off += (크기[-1] + 정렬 - 1) // 정렬 * 정렬
    머리 += b"\0" * ((-len(머리)) % 정렬)
    with open(출력, "wb") as f:
        f.write(머리)
        for (이름, t, dims), n in zip(텐서, 크기):
            d = 무작위(r, t, int(np.prod(dims)), 이름)
            assert len(d) == n
            f.write(d)
            f.write(b"\0" * ((-n) % 정렬))
    종류 = {}
    for 이름, t, _ in 텐서:
        종류[GR.이름.get(t, str(t))] = 종류.get(GR.이름.get(t, str(t)), 0) + 1
    print(f"{출력}: {arch} · 층 {층}{' + MTP ' + str(남는MTP) if 남는MTP else ''} · 텐서 {len(텐서)} {종류} · {os.path.getsize(출력) / 2 ** 20:.0f} MiB",
          flush=True)


if __name__ == "__main__":
    main()
