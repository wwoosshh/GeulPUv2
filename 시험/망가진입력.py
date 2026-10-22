#!/usr/bin/env python3
"""GeulPUv2 — 조작된 입력에서 엔진이 메모리 밖을 읽거나 쓰지 않는가(GHSA-4v72)를 GPU 없이 본다(빠른 점검 4f · 4g).

  python -I 시험/망가진입력.py gguf <시험_GGUF열기>
      작은 정상 GGUF 와 그것을 망가뜨린 것들(머리 · 개수 · 문자열 길이 · 배열 개수와 깊이 · 값 타입 · alignment · 차원 수와 차원 ·
      칸 수 넘침 · 텐서 자리와 자료 · 잘린 파일)로 엔진의 GGUF 열기를 돌린다. 정상은 종료 0 과 '열림', 망가진 것은 종료 1(gguf.gl 의
      오류) — 그 밖(접근 위반 · 망가졌는데 열림)이면 진다.
  python -I 시험/망가진입력.py 뽑기 <시험_뽑기>
      NaN · ±Inf 가 섞인 로짓 행을 매개변수 여러 벌로 뽑아(시험_뽑기 대조), 고른 번호가 모두 0 ~ V−1 이고 종료 0 인가.

마지막 줄 '… 모두 통과'(종료 0) 또는 '… 틀림 N'(종료 1)."""
import math
import os
import struct
import subprocess
import sys
import tempfile

F32, Q8_0 = 0, 8


def 글(x):
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def 메타(열쇠, 타입, 값):
    return 글(열쇠) + struct.pack("<I", 타입) + 값


def 배열(원소타입, 개수, 몸):
    return struct.pack("<I", 원소타입) + struct.pack("<q", 개수) + 몸


def 기본메타(정렬=32, 정렬타입=4):
    정렬값 = struct.pack("<I", 정렬) if 정렬타입 == 4 else 글(str(정렬))
    return [
        메타("general.architecture", 8, 글("qwen2")),
        메타("general.alignment", 정렬타입, 정렬값),
        메타("tokenizer.ggml.tokens", 9, 배열(8, 3, 글("a") + 글("b") + 글("c"))),
        메타("tokenizer.ggml.token_type", 9, 배열(5, 3, struct.pack("<3i", 1, 1, 1))),
        메타("test.nested", 9, 배열(9, 1, 배열(4, 2, struct.pack("<2I", 7, 8)))),
    ]


def 텐서정보(이름, 차원, 형식, 자리, 차원수=None):
    nd = len(차원) if 차원수 is None else 차원수
    return 글(이름) + struct.pack("<I", nd) + b"".join(struct.pack("<q", d) for d in 차원) + struct.pack("<I", 형식) + struct.pack("<q", 자리)


def 기본텐서():
    # a: F32 4 × 2 = 32 바이트(자리 0) · b: Q8_0 32 × 2 = 블록 2 × 34 = 68 바이트(자리 32)
    return [텐서정보("a", [4, 2], F32, 0), 텐서정보("b", [32, 2], Q8_0, 32)]


def 만들기(메타들=None, 텐서들=None, 메타수=None, 텐서수=None, 채움=32, 자료=None, 자르기=None, 마법=b"GGUF", 판=3):
    메타들 = 기본메타() if 메타들 is None else 메타들
    텐서들 = 기본텐서() if 텐서들 is None else 텐서들
    몸 = 마법 + struct.pack("<I", 판) + struct.pack("<q", len(텐서들) if 텐서수 is None else 텐서수)
    몸 += struct.pack("<q", len(메타들) if 메타수 is None else 메타수) + b"".join(메타들) + b"".join(텐서들)
    몸 += b"\0" * ((-len(몸)) % 채움)
    몸 += b"\x11" * 128 if 자료 is None else 자료
    return 몸 if 자르기 is None else 몸[:자르기]


def 깊은배열(깊이):
    몸 = 배열(4, 1, struct.pack("<I", 1))
    for _ in range(깊이 - 1):
        몸 = 배열(9, 1, 몸)
    return 몸


def 경우들():
    정상 = 만들기()
    잘린 = len(만들기(자료=b""))                  # 텐서 자료 바로 앞까지
    메 = 기본메타()
    yield "정상 파일", 정상, True
    yield "모르는 텐서 형식(엔진이 쓰지 않는다 — 시작만 본다)", 만들기(텐서들=기본텐서() + [텐서정보("c", [8], 99, 0)]), True
    yield "머리가 잘림(10 바이트)", 정상[:10], False
    yield "마법 수가 다름", 만들기(마법=b"GGUX"), False
    yield "판 2", 만들기(판=2), False
    yield "메타데이터 개수 2⁴⁰", 만들기(메타수=1 << 40), False
    yield "메타데이터 개수 음수", 만들기(메타수=-1), False
    yield "텐서 개수 2⁴⁰", 만들기(텐서수=1 << 40), False
    yield "열쇠 길이 음수(−16)", 만들기(메타들=[struct.pack("<q", -16) + b"x" * 16] + 메[1:]), False
    yield "열쇠 길이 2⁴⁰(파일 밖)", 만들기(메타들=[struct.pack("<q", 1 << 40) + b"general.architecture"] + 메[1:]), False
    yield "문자열 값 길이 파일 밖", 만들기(메타들=[메타("general.name", 8, struct.pack("<q", 1 << 40) + b"x")] + 메), False
    yield "문자열 값 길이 음수", 만들기(메타들=[메타("general.name", 8, struct.pack("<q", -8) + b"x")] + 메), False
    yield "u32 배열 개수 2⁴⁰", 만들기(메타들=메 + [메타("test.big", 9, 배열(4, 1 << 40, b""))]), False
    yield "배열 개수 음수", 만들기(메타들=메 + [메타("test.neg", 9, 배열(4, -1, b""))]), False
    yield "문자열 배열 개수 2⁴⁰", 만들기(메타들=메 + [메타("test.strs", 9, 배열(8, 1 << 40, 글("a")))]), False
    yield "배열 안 배열 깊이 6", 만들기(메타들=메 + [메타("test.deep", 9, 깊은배열(6))]), False
    yield "모르는 값 타입 13", 만들기(메타들=메 + [메타("test.t13", 13, b"\0" * 8)]), False
    yield "모르는 배열 원소 타입 13", 만들기(메타들=메 + [메타("test.e13", 9, 배열(13, 1, b"\0" * 8))]), False
    yield "alignment 0", 만들기(메타들=기본메타(정렬=0)), False
    yield "alignment 3", 만들기(메타들=기본메타(정렬=3)), False
    yield "alignment 가 문자열", 만들기(메타들=기본메타(정렬타입=8)), False
    yield "차원 수 0", 만들기(텐서들=[텐서정보("a", [], F32, 0)]), False
    yield "차원 수 5", 만들기(텐서들=[텐서정보("a", [4, 2, 1, 1, 1], F32, 0)]), False
    yield "차원 수 2³²−1", 만들기(텐서들=[텐서정보("a", [4, 2], F32, 0, 차원수=0xFFFFFFFF)]), False
    yield "차원 0", 만들기(텐서들=[텐서정보("a", [0, 2], F32, 0)]), False
    yield "차원 음수", 만들기(텐서들=[텐서정보("a", [4, -2], F32, 0)]), False
    yield "칸 수 넘침(2⁴⁰ × 2⁴⁰)", 만들기(텐서들=[텐서정보("a", [1 << 40, 1 << 40], F32, 0)]), False
    yield "텐서 자리 음수", 만들기(텐서들=[텐서정보("a", [4, 2], F32, -64)]), False
    yield "텐서 자리 파일 밖", 만들기(텐서들=[텐서정보("a", [4, 2], F32, 1 << 40)]), False
    yield "텐서 자료가 파일 밖까지(F32 1000 × 1000)", 만들기(텐서들=[텐서정보("a", [1000, 1000], F32, 0)]), False
    yield "텐서 자료 가운데서 잘림", 정상[:잘린 + 60], False
    yield "텐서 정보 가운데서 잘림", 정상[:잘린 - 40], False


def 시험_gguf(exe):
    틀림 = 0
    with tempfile.TemporaryDirectory() as 폴더:
        for k, (뜻, 몸, 열려야) in enumerate(경우들()):
            p = os.path.join(폴더, f"경우{k:02d}.gguf")
            with open(p, "wb") as f:
                f.write(몸)
            try:
                r = subprocess.run([exe, p], capture_output=True, timeout=60)
                코드, 나옴 = r.returncode, (r.stdout + r.stderr).decode("utf-8", "replace").strip().splitlines()
            except subprocess.TimeoutExpired:
                코드, 나옴 = None, ["60 초 안에 끝나지 않음"]
            끝줄 = 나옴[-1] if 나옴 else ""
            맞음 = (코드 == 0 and 끝줄.startswith("열림")) if 열려야 else 코드 == 1
            r = subprocess.CompletedProcess(exe, 코드)
            틀림 += 0 if 맞음 else 1
            print(f"{'통과' if 맞음 else '틀림'}  {뜻} — 종료 {r.returncode} · {끝줄[:90]}", flush=True)
        n = k + 1
    print(f"망가진 GGUF: {n} 경우 " + ("모두 통과" if not 틀림 else f"중 틀림 {틀림}"), flush=True)
    return 1 if 틀림 else 0


def 시험_뽑기(exe):
    V = 300
    행들 = [
        [math.nan] * V,
        [math.inf] * V,
        [-math.inf] * V,
        [math.nan] + [0.01 * j for j in range(1, V)],
        [0.01 * j for j in range(V - 1)] + [math.inf],
        [(-1.0) ** j * math.nan if j % 7 == 0 else 0.5 for j in range(V)],
        [0.003 * ((j * 37) % 101) for j in range(V)],          # 대조 — 유한한 행
    ]
    벌들 = [(0.0, 0, 1.0, 0.0), (0.7, 40, 0.9, 0.0), (1.0, 0, 1.0, 0.0), (0.8, 1, 1.0, 0.0), (1.5, 0, 0.5, 0.05), (0.6, 400, 0.95, 0.1)]
    몸 = struct.pack("<qqq", V, len(행들), len(벌들))
    for 행 in 행들:
        몸 += struct.pack(f"<{V}f", *행)
    for i, (T, K, P, Q) in enumerate(벌들):
        몸 += struct.pack("<dqddqq", T, K, P, Q, 12345 + i, 0)
    with tempfile.TemporaryDirectory() as 폴더:
        입력, 출력 = os.path.join(폴더, "입력.bin"), os.path.join(폴더, "출력.bin")
        with open(입력, "wb") as f:
            f.write(몸)
        r = subprocess.run([exe, "대조", 입력, 출력], capture_output=True, timeout=60)
        답 = list(struct.unpack(f"<{len(행들) * len(벌들)}q", open(출력, "rb").read())) if r.returncode == 0 and os.path.exists(출력) else []
    밖 = [x for x in 답 if not 0 <= x < V]
    맞음 = r.returncode == 0 and len(답) == len(행들) * len(벌들) and not 밖
    print(f"{'통과' if 맞음 else '틀림'}  행 {len(행들)}(NaN · ±Inf 섞임 여섯 · 유한 하나) × 벌 {len(벌들)} — 종료 {r.returncode} · 고른 번호 "
          f"{sorted(set(답))[:8]}{' …' if len(set(답)) > 8 else ''}" + (f" · 범위 밖 {밖[:4]}" if 밖 else ""), flush=True)
    print("유한하지 않은 로짓의 뽑기: " + ("모두 통과" if 맞음 else "틀림 1"), flush=True)
    return 0 if 맞음 else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 3 or sys.argv[1] not in ("gguf", "뽑기"):
        sys.exit(__doc__)
    sys.exit(시험_gguf(sys.argv[2]) if sys.argv[1] == "gguf" else 시험_뽑기(sys.argv[2]))
