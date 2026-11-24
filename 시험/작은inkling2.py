#!/usr/bin/env python3
"""0.25.0 25-4 — 실제 Inkling 2 GGUF(inkling2_loop)에서 층 몇 개만 남긴 작은 GGUF(엔진 시험용 — 엔진 · CPU 참조의 비트 비교).
기본은 원래 층 0 · 5 | 6 · 11 · 12 · 17 | 18 · 23 의 여덟 층(새 번호 0 … 7 — 슬라이딩 윈도 · 전체 어텐션 층이 번갈아): 반복 앞 둘, 반복 넷
(loop.start 2 · loop.end 6 · loop.count 3 — 원래 반복의 층이라 loop_msg_query · loop_*_ctrl_* 이 있다), 반복 뒤 둘. 블록 크기는 2 로 줄여
(attn_res.block_size — 블록 시작 0 · 2 · 4 · 6) 블록 상태 · 은행(앞 패스의 블록 상태 하나와 출력) · 메시지의 길이 모두 돈다.

  python -I 시험/작은inkling2.py <inkling2_loop gguf> <출력.gguf> [--층 0,5,6,11,12,17,18,23] [--반복 2,6,3] [--블록 2] [--전문가 16]

- 메타데이터는 바이트 그대로 옮기고 block_count · 층마다의 배열(attention.head_count · head_count_kv · sliding_window_pattern — 남긴 층의
  칸만) · loop.start · loop.end · loop.count · attn_res.block_size · general.name 만 바꾼다(토크나이저 · 헤드 · 창 · 전문가 그대로).
- 텐서: 낱말표 · 출력 · 끝의 정규화 · 질의는 그대로, 남긴 층의 텐서는 새 번호(0, 1, …)로 날 바이트 그대로. --전문가 E(기본 16 —
  파일과 CPU 참조를 줄인다)면 앞 E 개의 전문가만: ffn_gate_up_exps · ffn_down_exps · 라우터(ffn_gate_inp)의 앞 E 행 · exp_probs_b 의 앞 E 값,
  expert_count = E(expert_used_count 는 그대로 — E 이하여야 한다).
- 반복의 층(새 번호 [시작, 끝))은 원래 반복의 층이어야 한다(그 층에만 반복 제어의 텐서가 있다). 층 0 이 아닌 원래 층을 새 층 0 으로 두면
  그 층의 attn_res_query 는 쓰이지 않는다(새 층 0 은 질의가 없다 — 계약 4.15 의 3).
값은 실제 모델의 것이지만 층을 줄여 말이 되지 않는다 — 엔진 · CPU 참조의 비트 비교에만 쓴다.
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.normpath(os.path.join(HERE, "..")), "계약"))
import gguf읽기 as GR        # noqa: E402

고정 = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
층배열끝 = (".attention.head_count", ".attention.head_count_kv", ".attention.sliding_window_pattern")
정수꼴 = {4: "<I", 5: "<i", 10: "<Q", 11: "<q"}


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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    값깃발 = ("--층", "--반복", "--블록", "--전문가")
    위치 = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] not in 값깃발]
    원본, 출력 = 위치[:2]
    g = GR.GGUF(원본)
    구조 = g.meta.get("general.architecture")
    assert 구조 == "inkling2_loop", 구조
    층들 = [int(x) for x in sys.argv[sys.argv.index("--층") + 1].split(",")] if "--층" in sys.argv else [0, 5, 6, 11, 12, 17, 18, 23]
    시작, 끝, 패스 = [int(x) for x in sys.argv[sys.argv.index("--반복") + 1].split(",")] if "--반복" in sys.argv else (2, 6, 3)
    블록 = int(sys.argv[sys.argv.index("--블록") + 1]) if "--블록" in sys.argv else 2
    E0 = g.meta[구조 + ".expert_count"]
    E = int(sys.argv[sys.argv.index("--전문가") + 1]) if "--전문가" in sys.argv else 16
    assert g.meta[구조 + ".expert_used_count"] <= E <= E0, (E, E0)
    L = g.meta[구조 + ".block_count"]
    s0, e0 = g.meta[구조 + ".loop.start"], g.meta[구조 + ".loop.end"]
    assert 층들 == sorted(층들) and len(set(층들)) == len(층들) and all(0 <= l < L for l in 층들), (층들, L)
    assert 0 <= 시작 <= 끝 <= len(층들) and 패스 >= 1 and 블록 >= 1, (시작, 끝, 패스, 블록)
    assert all(s0 <= 층들[i] < e0 for i in range(시작, 끝)), f"반복의 새 층 {층들[시작:끝]} 은 원래 반복 [{s0}, {e0}) 의 층이어야 한다"
    번호 = {l: i for i, l in enumerate(층들)}
    정렬 = g.meta.get("general.alignment", 32)
    b = g.b
    _, nkv = struct.unpack_from("<QQ", b, 8)
    o = 24
    층배열 = tuple(구조 + x for x in 층배열끝)
    바꿈 = {구조 + ".block_count": len(층들), 구조 + ".loop.start": 시작, 구조 + ".loop.end": 끝, 구조 + ".loop.count": 패스,
          구조 + ".attn_res.block_size": 블록, 구조 + ".expert_count": E}
    이름글 = (f"{g.meta.get('general.name', 구조)} — 층 {','.join(map(str, 층들))} · 반복 [{시작}, {끝}) × {패스} · 블록 {블록} · 전문가 {E}"
           " (시험/작은inkling2.py)")
    kv = []
    for _ in range(nkv):
        n = struct.unpack_from("<Q", b, o)[0]
        k = bytes(b[o + 8:o + 8 + n]).decode("utf-8")
        o += 8 + n
        t = struct.unpack_from("<I", b, o)[0]
        e = 값건너기(b, o + 4, t)
        raw = bytes(b[o + 4:e])
        if k in 바꿈:
            raw = struct.pack(정수꼴[t], 바꿈[k])
        elif k == "general.name":
            raw = 문자열(이름글)
        elif k in 층배열 and t == 9:
            raw = 배열자르기(b, o + 4, 층들)
        kv.append((k, t, raw))
        o = e
    텐서 = []                                               # (새 이름, 타입, dims, 원본 이름, 바이트 수)
    for 이름, (t, dims, off) in g.tensors.items():
        새 = 이름
        if 이름.startswith("blk."):
            l = int(이름.split(".")[1])
            if l not in 번호:
                continue
            새 = f"blk.{번호[l]}." + ".".join(이름.split(".")[2:])
        vb, bb = GR.블록[t]
        dims = [int(d) for d in dims]
        꼬리 = ".".join(이름.split(".")[2:]) if 이름.startswith("blk.") else 이름
        if 꼬리 in ("ffn_gate_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_inp.weight", "exp_probs_b.bias"):
            assert dims[-1] == E0, (이름, dims)                   # 바깥 차원이 전문가 — 앞 E 개가 파일의 앞부분
            dims[-1] = E
        n = 1
        for d in dims:
            n *= d
        텐서.append((새, t, dims, 이름, n // vb * bb))
    머리 = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(텐서), len(kv)))
    for k, t, raw in kv:
        머리 += 문자열(k) + struct.pack("<I", t) + raw
    off = 0
    for 새, t2, dims, _, n in 텐서:
        머리 += 문자열(새) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", t2, off)
        off += (n + 정렬 - 1) // 정렬 * 정렬
    머리 += b"\0" * ((-len(머리)) % 정렬)
    with open(출력, "wb") as f:
        f.write(머리)
        for 새, t2, dims, 이름, n in 텐서:
            _, _, raw = g.원(이름)
            f.write(raw.tobytes()[:n])
            f.write(b"\0" * ((-n) % 정렬))
    print(f"{출력}: 층 {층들} → {len(층들)} 층 · 반복 [{시작}, {끝}) × {패스} · 블록 {블록} · 전문가 {E} · 텐서 {len(텐서)} · "
          f"{os.path.getsize(출력) / 2 ** 20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
