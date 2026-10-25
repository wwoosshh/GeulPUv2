#!/usr/bin/env python3
"""#96 2단계 — 계약 5.4 의 J 파일(geulpu-jlens/1)을 읽고 쓴다(numpy 만 — torch · safetensors 꾸러미 없이).

safetensors 꼴: 앞 8 바이트가 머리 JSON 의 길이 N(부호 없는 64 비트 · 리틀 엔디언), 다음 N 바이트가 머리 JSON(UTF-8 — 뒤를 빈칸으로 채워
8 의 배수), 그 뒤가 자료. 머리의 텐서마다 dtype · shape · data_offsets(자료의 처음에서 [시작, 끝) 바이트 — 빈틈 없이 잇는다),
`__metadata__` 는 글 → 글. J 파일은 텐서 `J.<l>`(F16 · [D][D] · 행 우선 — 원천 층 l) 만 담고, metadata 에 format · d_model · n_layers 와
기록(fit · checked — JSON 글)을 둔다.
"""
import hashlib
import json
import struct

import numpy as np

형식 = "geulpu-jlens/1"


def 읽기(경로):
    """→ (J {층 l: [D][D] np.float16 — 읽기 전용}, 메타 {글: 글}). 꼴이 틀리면 ValueError."""
    with open(경로, "rb") as f:
        자료 = f.read()
    if len(자료) < 8:
        raise ValueError(f"{경로}: safetensors 파일이 아니다(8 바이트보다 짧다)")
    n = struct.unpack("<Q", 자료[:8])[0]
    if 8 + n > len(자료) or 자료[8:9] != b"{":
        raise ValueError(f"{경로}: safetensors 파일이 아니다(머리의 길이 {n})")
    머리 = json.loads(자료[8:8 + n].decode("utf-8"))
    메타 = 머리.pop("__metadata__", None) or {}
    if 메타.get("format") != 형식:
        raise ValueError(f"{경로}: J 파일이 아니다(format {메타.get('format')!r} — {형식} 이어야 한다)")
    D, L = int(메타["d_model"]), int(메타["n_layers"])
    바탕 = 8 + n
    J, 끝 = {}, 0
    for 이름, t in sorted(머리.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        if not 이름.startswith("J.") or not 이름[2:].isdigit():
            raise ValueError(f"{경로}: 모르는 텐서 {이름!r}")
        l = int(이름[2:])
        if not 0 <= l <= L - 2:
            raise ValueError(f"{경로}: 원천 층 {l} 이 0 … {L - 2} 밖이다")
        if t.get("dtype") != "F16" or list(t.get("shape", [])) != [D, D]:
            raise ValueError(f"{경로}: {이름} 이 F16 [{D}][{D}] 가 아니다({t.get('dtype')} {t.get('shape')})")
        a, b = t["data_offsets"]
        if a != 끝 or b - a != D * D * 2:
            raise ValueError(f"{경로}: {이름} 의 자리 [{a}, {b}) — 빈틈 없이 {D * D * 2} 바이트여야 한다")
        끝 = b
        J[l] = np.frombuffer(자료, "<f2", D * D, 바탕 + a).reshape(D, D)
    if 바탕 + 끝 != len(자료):
        raise ValueError(f"{경로}: 자료 뒤에 {len(자료) - 바탕 - 끝} 바이트가 남는다")
    return J, 메타


def 쓰기(경로, J, n_layers, 메타=None):
    """J {층 l: [D][D]}(반실수로 바꿔 쓴다) · 모델의 층 수 · 기록 {글: 글}(fit · checked 등 — JSON 은 글로) → J 파일."""
    층들 = sorted(J)
    D = int(np.asarray(J[층들[0]]).shape[0])
    메타 = {k: str(v) for k, v in (메타 or {}).items()}
    메타.update({"format": 형식, "d_model": str(D), "n_layers": str(int(n_layers))})
    머리, 자리 = {"__metadata__": 메타}, 0
    for l in 층들:
        assert 0 <= l <= n_layers - 2 and np.asarray(J[l]).shape == (D, D), (l, np.asarray(J[l]).shape)
        머리[f"J.{l}"] = {"dtype": "F16", "shape": [D, D], "data_offsets": [자리, 자리 + D * D * 2]}
        자리 += D * D * 2
    h = json.dumps(머리, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    h += b" " * (-len(h) % 8)
    with open(경로, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for l in 층들:
            f.write(np.ascontiguousarray(np.asarray(J[l], np.float16).astype("<f2")).tobytes())


def 해시(경로):
    """J 파일의 SHA-256(16진) — 응답 · 재현 파일에 적는 것."""
    h = hashlib.sha256()
    with open(경로, "rb") as f:
        for 덩이 in iter(lambda: f.read(1 << 24), b""):
            h.update(덩이)
    return h.hexdigest()
