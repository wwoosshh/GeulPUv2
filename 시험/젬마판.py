#!/usr/bin/env python3
"""0.12.0 12-3h-3 — Gemma 4 의 판(커널생성 gemma4큰 — 커널/생성젬마.gl)을 CPU 참조(계약/정수계약.py — 계약 4.12)와 비트까지:
머리정규화젬마D(q · k 무게 · v 무게 없이) → RoPEKV젬마D(헤드 전체 NEOX — 전체 층은 rope_freqs 표, q 자리 · 키 · 값 캐시 — 값은 0.13.0
12-3h-7 부터 키 32 개 블록마다 [차원][키]) → 타일어텐션젬마D(0.13.0 12-3h-7 — 텐서 코어) · 접기젬마D(배율 1, 창), 그리고 RMS더하기젬마(층
배율과 함께 · 없이) · 곱하기젬마 · GeGLU젬마(gate · up 이 32 열마다 번갈아) · 소프트캡젬마.
헤드 256(질의 16 · KV 8 — 슬라이딩 윈도 층)과 512(질의 16 · KV 1 — 전체 층), 같은 행들을 덩이 둘로(앞 덩이의 키 · 값을 캐시에) 돌린
어텐션, 그리고 묶음 둘에 걸친 창(행 1,100 — 창 130 · 1,024). 0.13.0 12-3h-6 — 생성(행 넷까지)의 조각어텐션젬마D · 조각접기젬마D 도
같은 참값과(행 1 · 2 · 4 를 끝 · 가운데 · 처음 · 타일 경계, 행 1,100 은 그룹 경계 · 창 경계에). 0.13.0 12-3h-8 — 합친 판(접기자리로 ·
조각접기자리로 · RMS더하기정수줄 · GeGLU자리로)은 두 판을 잇달아 띄운 것(접기 + 실수자리로 · RMS더하기 + RMS정규화정수줄 · GeGLU + 실수자리로)과
같은 입력에서 바이트까지(잔차 · 자리 · 정보 — 행 합 판의 칸까지). 0.13.0 12-3h-9 — 윈도 층의 고리 버퍼(캐시의 위치 p 는 칸 p mod 고리):
행들을 덩이씩 넣어 고리를 몇 번 돈 프롬프트 · 끝의 생성 · 고리에 남은 캐시, 그리고 고리가 아직 가진 창 안으로 되감아 다른 행으로 이은 것.
프롬프트 어텐션의 엔진 꼴(창이 있으면 묶음 칸 몇 개 — 묶음 g 의 칸 g mod 칸 수, 창이 없으면 행 64 개씩 나눠 띄움)도 같은 참값과.

  python -I 시험/젬마판.py <ptx> <출력.json> [--행 300] [--gemma3 [--shape 너비,확장폭,헤드 수,KV 헤드 수] [--헤드 128]]
  python -I 시험/젬마판.py <ptx> <출력.json> --층입력 <층 수> [--어휘 262144]

--층입력 <층 수>(0.24.0 의 24-5 — 커널생성 gemma4큰 K … --W16 --층입력 의 모듈, 계약 4.12 의 9): 층마다의 입력(PLE)의 커널만 본다 —
투영의 정규화(머리정규화젬마256 을 헤드 = 층 수 · KV 헤드 0 으로), 표를 읽어 더하기 p = y + e · c_p(층입력표_Q4_K · _Q5_K · _Q6_K — 무작위 표
[어휘 수][2n], 어휘 수는 모듈의 것 --어휘), PLE 블록(층GeGLU자리로젬마 = CPU 참조의 겔루(g, p_l) + 실수자리로). 그리고 KV 를 나눠 쓰는
층(계약 4.12 의 8)의 띄움 — 머리정규화젬마D · RoPEKV젬마D 를 KV 헤드 0 으로: q 의 8비트 자리 둘 · 블록 지수가 KV 헤드와 함께 띄운 것과 같고
버퍼의 k · v 열과 캐시는 그대로인가. 나머지 커널은 gemma4큰 K 의 모듈과 같은 소스라 다시 보지 않는다. 표 하나를 푸는 동안 메모리를 1.5 GB 쯤 쓴다.

--gemma3(0.24.0 — 커널생성 gemma3큰 2560 8 4 10240 의 모듈, 계약 4.14): 머리정규화젬마256 이 v 를 그대로 두고, 점수 스케일이 1/16 이고,
전체 어텐션 층도 헤드 256(RoPE 표는 θ 10⁶ · 선형 스케일링 8). 나머지 시험은 같다(헤드 512 의 커널은 이 모듈에 없다).
--헤드 128(27B 의 모듈 — 커널생성 gemma3큰 K 5376 32 16 21504 262208 --헤드 128, --shape 5376,21504,32,16): 헤드 128 의 커널을 같은
시험으로 보고, 점수 스케일은 27B 의 것 fp32(1/√(너비 ÷ 헤드 수)) = fp32(1/√168)(계약 4.14 의 4).

마지막 줄 "모두 0: True|False".
"""
import ctypes
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "계약"))
sys.path.insert(0, HERE)
import 쿠다드라이버 as G          # noqa: E402
import 시험장치               # noqa: E402
import 정수계약 as C        # noqa: E402

f32 = np.float32
u64, i64 = G.c_u64, G.c_i64
EPS = 1e-6
H = 16
창없음 = 1 << 40
V정규화 = True               # v 도 헤드마다 정규화(가중치 없이 — 계약 4.12). Gemma 3 의 모듈(--gemma3 — 계약 4.14 의 3)은 하지 않는다
눈금 = 1.0                   # 점수 스케일 — Gemma 4 는 1, Gemma 3 는 fp32(1/√256) = 1/16(계약 4.14 의 4)


def 비트(x):
    return struct.unpack("<I", struct.pack("<f", float(f32(x))))[0]


def 다른칸(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.dtype == f32 or b.dtype == f32:
        a, b = a.astype(f32).view(np.int32), b.astype(f32).view(np.int32)
    return int(np.count_nonzero(a != b))


class 장치:
    def __init__(self, ptx):
        self.dr = 시험장치.드라이버()
        self.mod = self.dr.모듈(open(ptx, "rb").read())
        self.fn = {}

    def 띄우기(self, 이름, grid, block, params):
        if 이름 not in self.fn:
            self.fn[이름] = self.dr.함수(self.mod, 이름)
        ps = [u64(p) if 참조_ else i64(p) for 참조_, p in params]
        args = (ctypes.c_void_p * len(ps))(*[ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in ps])
        rr = self.dr.cu.cuLaunchKernel(self.fn[이름], grid[0], grid[1], 1, block, 1, 1, 0, None, args, None)
        assert rr == 0, (이름, rr)

    def up(self, x):
        b = np.ascontiguousarray(x).tobytes()
        p = self.dr.할당(max(len(b), 4))
        self.dr.확인(self.dr.cu.cuMemcpyHtoD_v2(p, b, len(b)))
        return p

    def down(self, p, 모양, dt=f32):
        return self.dr.내리기(p, np.empty(모양, dt))


def 층시험(dv, D, HKV, θ, 인자, T, T1, 창들, r, 생성들=None):
    """헤드 D · KV 헤드 HKV 의 앞단(정규화 · RoPE · 캐시)과 어텐션 — 창마다, 한 번에 · 덩이 둘, 그리고 생성(행 넷까지 — 0.13.0 12-3h-6 의
    조각어텐션젬마D · 조각접기젬마D: 생성들 의 (첫 행, 행 수)마다 그 행들만 앞단을 다시 돌려 q 를 얻고 온 캐시로)."""
    if 생성들 is None:
        생성들 = [(T - 1, 1), (T - 4, 4), (T // 2, 2), (0, 1), (63, 2)]
    폭 = H * D + 2 * HKV * D + 1024                                  # 엔진의 버qkv 처럼 행이 넓다(넘는 칸은 그대로)
    최대길이 = (T + 1023) // 1024 * 1024
    묶음수 = 최대길이 // 1024
    nb = D // 32
    qkv = (r.standard_normal((T, 폭)) * 10.0 ** r.uniform(-0.5, 0.5, (1, 폭))).astype(f32)
    qw = (1.0 + 0.1 * r.standard_normal(D)).astype(f32)
    kw = (1.0 + 0.1 * r.standard_normal(D)).astype(f32)
    cos, sin = C.rope표(최대길이, θ, hd=D, 인자=인자)
    다 = {}
    # ---- CPU 참조 ----
    t0 = time.time()
    qn = qkv.copy()
    qn[:, :H * D] = C.머리정규화(qkv[:, :H * D], qw, EPS, D)
    kb, vb = H * D, H * D + HKV * D
    qn[:, kb:kb + HKV * D] = C.머리정규화(qkv[:, kb:kb + HKV * D], kw, EPS, D)
    if V정규화:
        qn[:, vb:vb + HKV * D] = C.머리정규화(qkv[:, vb:vb + HKV * D], np.ones(D, f32), EPS, D)
    위치 = np.arange(T)
    qh = [C.rope(qn[:, a * D:(a + 1) * D], cos, sin, 위치) for a in range(H)]
    kh = [C.rope(qn[:, kb + a * D:kb + (a + 1) * D], cos, sin, 위치) for a in range(HKV)]
    vh = [np.ascontiguousarray(qn[:, vb + a * D:vb + (a + 1) * D]) for a in range(HKV)]
    G_ = H // HKV
    o참 = {W: np.concatenate([C.어텐션(qh[a], kh[a // G_], vh[a // G_], 눈금=눈금, 창=(W if W < T else None)) for a in range(H)], 1)
          for W in 창들}
    print(f"  D {D}: CPU 참조 {time.time() - t0:.1f} 초", flush=True)

    def 앞단(pq, n, 시작, pqd, pqE, pkd, pkE, pvd, pvE):
        dv.띄우기(f"머리정규화젬마{D}", ((n * (H + HKV) + 3) // 4, 1), 128, [(1, pq), (1, pqw), (1, pkw), (0, n), (0, H), (0, HKV), (0, 폭)])
        dv.띄우기(f"RoPEKV젬마{D}", (((H + 2 * HKV) * nb + 31) // 32, n), 256,
                [(1, pq), (1, pqd), (1, pqE), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (1, pcos), (1, psin), (0, n), (0, H), (0, HKV), (0, 폭),
                 (0, 시작), (0, 최대길이)])                                       # 고리 = 최대길이(고리 없음 — 칸 = 위치)

    자리다름 = {"접기": 0, "조각": 0}

    def 자리대조(종류, n, po, 판이름, 칸, 덧):
        """0.13.0 12-3h-8 — 자리로 판(같은 상태 칸)의 o 입력 자리와 [po 에 실수자리로] 의 자리 · 정보를 바이트까지"""
        깊이 = H * D
        정보폭 = (n + 63) // 64 * 64
        nz, ni = 2 * n * 깊이 // 4, 2 * (깊이 // 32) * 정보폭
        za, ia, zb, ib = (dv.up(np.zeros(k, np.int32)) for k in (nz, ni, nz, ni))
        dv.띄우기(판이름, (H, n), D, [(1, 칸), (1, za), (1, ia)] + 덧)
        dv.띄우기("실수자리로", (-(-(깊이 // 32) // 32), n), 256, [(1, po), (1, zb), (1, ib), (0, n), (0, 깊이)])
        dv.dr.맞추기()
        자리다름[종류] += 다른칸(dv.down(za, (nz,), np.int32), dv.down(zb, (nz,), np.int32))
        자리다름[종류] += 다른칸(dv.down(ia, (ni,), np.int32), dv.down(ib, (ni,), np.int32))

    def 어텐션(n, 시작, pqd, pqE, pkd, pkE, pvd, pvE, po, W, 엔진꼴=False):
        """엔진꼴 이면 엔진(구조/gemma4/엔진.gl)의 띄움 — 0.13.0 12-3h-9: 창이 있으면 한 번에 · 묶음 칸 몇 개(묶음 g 의 칸 g mod 칸 수),
        창이 없으면 행 64 개씩 나눠 · 칸은 묶음 모두. 아니면 한 번에 · 칸은 묶음 모두. 접기자리로 는 같은 띄움들의 바이트를 실수자리로 와"""
        블록, 칸수 = n, 묶음수
        if 엔진꼴:
            if W < 창없음:
                칸수 = min(묶음수, (min(n, 1024) + W + 1022) // 1024 + 1)
            else:
                블록 = 64
        p묶 = dv.dr.할당(블록 * H * 칸수 * (D + 4) * 4)
        깊이 = H * D
        정보폭 = (n + 63) // 64 * 64
        nz, ni = 2 * n * 깊이 // 4, 2 * (깊이 // 32) * 정보폭
        za, ia, zb, ib = (dv.up(np.zeros(k, np.int32)) for k in (nz, ni, nz, ni))
        for r0 in range(0, n, 블록):
            nb_ = min(블록, n - r0)
            dv.띄우기(f"타일어텐션젬마{D}", (-(-nb_ * (H // HKV) // 16) * 묶음수, HKV), 128,       # 0.13.0 12-3h-7 — (질의 벡터 16 개 × 묶음, KV 헤드)
                    [(1, pqd), (1, pqE), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (1, p묶), (0, n), (0, H), (0, HKV), (0, 시작), (0, 최대길이),
                     (0, 최대길이), (0, W), (0, 비트(눈금)), (0, r0), (0, nb_), (0, 칸수)])
            덧 = [(0, n), (0, H), (0, 시작), (0, 최대길이), (0, W), (0, r0), (0, 칸수)]
            dv.띄우기(f"접기젬마{D}", (H, nb_), D, [(1, p묶), (1, po)] + 덧)
            dv.띄우기(f"접기자리로젬마{D}", (H, nb_), D, [(1, p묶), (1, za), (1, ia)] + 덧)
        dv.띄우기("실수자리로", (-(-(깊이 // 32) // 32), n), 256, [(1, po), (1, zb), (1, ib), (0, n), (0, 깊이)])
        dv.dr.맞추기()
        자리다름["접기"] += 다른칸(dv.down(za, (nz,), np.int32), dv.down(zb, (nz,), np.int32))
        자리다름["접기"] += 다른칸(dv.down(ia, (ni,), np.int32), dv.down(ib, (ni,), np.int32))

    def 생성어텐션(n, 시작, pqd, pqE, pkd, pkE, pvd, pvE, po, W, 걸음=None):
        """걸음 — 0.13.0 12-3h-9 의 슬롯걸음(블록이 타일 슬롯0 · 슬롯0 + 걸음 · … 을 차례로 — 없으면 블록 하나에 타일 하나)"""
        슬롯 = 최대길이 // 64 if W >= 최대길이 else (W + 2) // 64 + 2        # 구조/gemma4/엔진.gl 의 젬마슬롯수
        걸음 = min(슬롯, 걸음 or 슬롯)
        p조 = dv.dr.할당(n * H * 슬롯 * (D + 4) * 4)
        dv.띄우기(f"조각어텐션젬마{D}", (걸음 * HKV, n), 256,
                [(1, pqd), (1, pqE), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (1, p조), (0, n), (0, H), (0, HKV), (0, 시작), (0, 최대길이), (0, W),
                 (0, 비트(눈금)), (0, 슬롯), (0, 걸음)])
        dv.띄우기(f"조각접기젬마{D}", (H, n), D, [(1, p조), (1, po), (0, n), (0, H), (0, 시작), (0, 최대길이), (0, W), (0, 슬롯)])
        자리대조("조각", n, po, f"조각접기자리로젬마{D}", p조, [(0, n), (0, H), (0, 시작), (0, 최대길이), (0, W), (0, 슬롯)])

    pqw, pkw, pcos, psin = dv.up(qw), dv.up(kw), dv.up(cos), dv.up(sin)

    def 캐시():
        return (dv.dr.할당(2 * HKV * 최대길이 * D), dv.dr.할당(HKV * 최대길이 * nb * 8),
                dv.dr.할당(2 * HKV * 최대길이 * D), dv.dr.할당(HKV * 최대길이 * nb * 8))
    pq = dv.up(qkv)
    pqd, pqE = dv.dr.할당(2 * T * H * D), dv.dr.할당(T * H * nb * 8)
    pkd, pkE, pvd, pvE = 캐시()
    앞단(pq, T, 0, pqd, pqE, pkd, pkE, pvd, pvE)
    dv.dr.맞추기()
    qD, qE_ = C.자리로(np.concatenate(qh, 1), 2)
    kD, kE_ = zip(*[C.자리로(x, 2) for x in kh])
    vD, vE_ = zip(*[C.자리로(x, 2) for x in vh])
    qnG = dv.down(pq, (T, 폭))
    다[f"D{D} 정규화"] = 다른칸(qnG, qn)
    다[f"D{D} q 자리"] = 다른칸(dv.down(pqd, (2, T, H * D), np.int8), qD.astype(np.int8))
    다[f"D{D} q 지수"] = 다른칸(dv.down(pqE, (T, H * nb), np.int64), qE_)
    다[f"D{D} 키 자리"] = 다른칸(dv.down(pkd, (2, HKV, 최대길이, D), np.int8)[:, :, :T],
                             np.stack([np.stack([d[j] for d in kD]) for j in range(2)]).astype(np.int8))
    다[f"D{D} 키 지수"] = 다른칸(dv.down(pkE, (HKV, 최대길이, nb), np.int64)[:, :T], np.stack(kE_))
    값 = dv.down(pvd, (2, HKV, 최대길이 // 32, D, 32), np.int8).transpose(0, 1, 2, 4, 3).reshape(2, HKV, 최대길이, D)   # 키 32 개 블록마다 [차원][키]
    다[f"D{D} 값 자리"] = 다른칸(값[:, :, :T], np.stack([np.stack([d[j] for d in vD]) for j in range(2)]).astype(np.int8))
    다[f"D{D} 값 지수"] = 다른칸(dv.down(pvE, (HKV, 최대길이, nb), np.int64)[:, :T], np.stack(vE_))
    for W in 창들:
        po = dv.dr.할당(T * H * D * 4)
        어텐션(T, 0, pqd, pqE, pkd, pkE, pvd, pvE, po, W)
        dv.dr.맞추기()
        다[f"D{D} 어텐션 창 {W if W < 창없음 else '없음'} 한 번에"] = 다른칸(dv.down(po, (T, H * D)), o참[W])
        po = dv.dr.할당(T * H * D * 4)                        # 0.13.0 12-3h-9 — 엔진의 띄움(창이 있으면 묶음 칸 몇 개, 없으면 행 64 개씩)
        어텐션(T, 0, pqd, pqE, pkd, pkE, pvd, pvE, po, W, 엔진꼴=True)
        dv.dr.맞추기()
        다[f"D{D} 어텐션 창 {W if W < 창없음 else '없음'} 엔진 꼴(묶음 칸 · 행 나눔)"] = 다른칸(dv.down(po, (T, H * D)), o참[W])
        # 덩이 둘 — 앞 덩이의 키 · 값을 캐시에
        pkd2, pkE2, pvd2, pvE2 = 캐시()
        os_ = []
        for s0, s1 in ((0, T1), (T1, T)):
            n = s1 - s0
            pq2 = dv.up(qkv[s0:s1])
            pqd2, pqE2 = dv.dr.할당(2 * n * H * D), dv.dr.할당(n * H * nb * 8)
            앞단(pq2, n, s0, pqd2, pqE2, pkd2, pkE2, pvd2, pvE2)
            po2 = dv.dr.할당(n * H * D * 4)
            어텐션(n, s0, pqd2, pqE2, pkd2, pkE2, pvd2, pvE2, po2, W)
            dv.dr.맞추기()
            os_.append(dv.down(po2, (n, H * D)))
        다[f"D{D} 어텐션 창 {W if W < 창없음 else '없음'} 덩이 둘"] = 다른칸(np.concatenate(os_), o참[W])
        # 생성(행 넷까지) — 그 행들의 q(앞단을 다시 — 캐시에는 같은 값을 다시 쓴다)와 온 캐시
        합 = 0
        for s0, n in 생성들:
            pq3 = dv.up(qkv[s0:s0 + n])
            pqd3, pqE3 = dv.dr.할당(2 * n * H * D), dv.dr.할당(n * H * nb * 8)
            앞단(pq3, n, s0, pqd3, pqE3, pkd, pkE, pvd, pvE)
            for 걸음 in (None, 3):                    # 0.13.0 12-3h-9 — 블록 하나에 타일 하나 · 블록이 타일 여럿(슬롯걸음 3)
                po3 = dv.dr.할당(n * H * D * 4)
                생성어텐션(n, s0, pqd3, pqE3, pkd, pkE, pvd, pvE, po3, W, 걸음)
                dv.dr.맞추기()
                합 += 다른칸(dv.down(po3, (n, H * D)), o참[W][s0:s0 + n])
        다[f"D{D} 어텐션 창 {W if W < 창없음 else '없음'} 생성(행 넷까지) {len(생성들)} 경우 × 슬롯걸음 둘"] = 합
    다[f"D{D} 어텐션 접기자리로 = 접기 + 실수자리로(바이트)"] = 자리다름["접기"]
    다[f"D{D} 어텐션 조각접기자리로 = 조각접기 + 실수자리로(바이트)"] = 자리다름["조각"]
    return 다


def 고리시험(dv, D, HKV, θ, T, 고리, 덩이, W, r, 되감기):
    """0.13.0 12-3h-9 — 윈도 층의 고리 버퍼(키 · 값 · 지수 캐시의 위치 p 는 칸 p mod 고리 — 고리 ≥ 창 + 덩이 − 1): T 행을 덩이씩(앞단이 그
    덩이의 키 · 값을 고리에 쓰고 어텐션이 읽는다 — 엔진의 프롬프트 길) 넣어 고리를 몇 번 돈 뒤 끝의 생성(행 넷까지)과 고리에 남은 캐시,
    그리고 되감기 = (p, 새 행 수): 위치 p 로 되돌려(T − p ≤ 고리 − 창 + 1 — 고리가 아직 가진 창) 다른 행으로 이은 것."""
    폭 = H * D + 2 * HKV * D + 1024
    nb = D // 32
    p되, n새 = 되감기
    최대길이 = (max(T, p되 + n새) + 1023) // 1024 * 1024
    묶음수 = 최대길이 // 1024
    G_ = H // HKV

    def 행들(n):
        return (r.standard_normal((n, 폭)) * 10.0 ** r.uniform(-0.5, 0.5, (1, 폭))).astype(f32)
    qkvA = 행들(T)
    qkvB = np.concatenate([qkvA[:p되], 행들(n새)])
    qw = (1.0 + 0.1 * r.standard_normal(D)).astype(f32)
    kw = (1.0 + 0.1 * r.standard_normal(D)).astype(f32)
    cos, sin = C.rope표(최대길이, θ, hd=D, 인자=None)
    kb, vb = H * D, H * D + HKV * D

    def 참(qkv):
        qn = qkv.copy()
        qn[:, :H * D] = C.머리정규화(qkv[:, :H * D], qw, EPS, D)
        qn[:, kb:kb + HKV * D] = C.머리정규화(qkv[:, kb:kb + HKV * D], kw, EPS, D)
        if V정규화:
            qn[:, vb:vb + HKV * D] = C.머리정규화(qkv[:, vb:vb + HKV * D], np.ones(D, f32), EPS, D)
        위치 = np.arange(len(qkv))
        qh = [C.rope(qn[:, a * D:(a + 1) * D], cos, sin, 위치) for a in range(H)]
        kh = [C.rope(qn[:, kb + a * D:kb + (a + 1) * D], cos, sin, 위치) for a in range(HKV)]
        vh = [np.ascontiguousarray(qn[:, vb + a * D:vb + (a + 1) * D]) for a in range(HKV)]
        return kh, vh, np.concatenate([C.어텐션(qh[a], kh[a // G_], vh[a // G_], 눈금=눈금, 창=W) for a in range(H)], 1)
    t0 = time.time()
    khA, vhA, oA = 참(qkvA)
    oB = 참(qkvB)[2]
    print(f"  고리 {고리}(창 {W} · 덩이 {덩이}): CPU 참조 {time.time() - t0:.1f} 초", flush=True)
    pqw, pkw, pcos, psin = dv.up(qw), dv.up(kw), dv.up(cos), dv.up(sin)
    pkd, pkE = dv.dr.할당(2 * HKV * 고리 * D), dv.dr.할당(HKV * 고리 * nb * 8)
    pvd, pvE = dv.dr.할당(2 * HKV * 고리 * D), dv.dr.할당(HKV * 고리 * nb * 8)

    def 앞단(qkv, s0, n):
        pq = dv.up(qkv[s0:s0 + n])
        pqd, pqE = dv.dr.할당(2 * n * H * D), dv.dr.할당(n * H * nb * 8)
        dv.띄우기(f"머리정규화젬마{D}", ((n * (H + HKV) + 3) // 4, 1), 128, [(1, pq), (1, pqw), (1, pkw), (0, n), (0, H), (0, HKV), (0, 폭)])
        dv.띄우기(f"RoPEKV젬마{D}", (((H + 2 * HKV) * nb + 31) // 32, n), 256,
                [(1, pq), (1, pqd), (1, pqE), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (1, pcos), (1, psin), (0, n), (0, H), (0, HKV), (0, 폭),
                 (0, s0), (0, 고리)])
        return pqd, pqE

    def 프롬프트(qkv, o참, s0, s1):
        다름 = 0
        칸수 = min(묶음수, (min(덩이, 1024) + W + 1022) // 1024 + 1)      # 엔진의 윈도 층 — 묶음 g 의 칸 g mod 칸 수
        for a in range(s0, s1, 덩이):
            n = min(덩이, s1 - a)
            pqd, pqE = 앞단(qkv, a, n)
            p묶, po = dv.dr.할당(n * H * 칸수 * (D + 4) * 4), dv.dr.할당(n * H * D * 4)
            dv.띄우기(f"타일어텐션젬마{D}", (-(-n * G_ // 16) * 묶음수, HKV), 128,
                    [(1, pqd), (1, pqE), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (1, p묶), (0, n), (0, H), (0, HKV), (0, a), (0, 최대길이),
                     (0, 고리), (0, W), (0, 비트(눈금)), (0, 0), (0, n), (0, 칸수)])
            dv.띄우기(f"접기젬마{D}", (H, n), D, [(1, p묶), (1, po), (0, n), (0, H), (0, a), (0, 최대길이), (0, W), (0, 0), (0, 칸수)])
            dv.dr.맞추기()
            다름 += 다른칸(dv.down(po, (n, H * D)), o참[a:a + n])
        return 다름

    def 생성(qkv, o참, s0, n):
        pqd, pqE = 앞단(qkv, s0, n)            # 그 행들의 q(캐시에는 같은 값을 같은 칸에 다시 쓴다)
        슬롯 = (W + 2) // 64 + 2                # 구조/gemma4/엔진.gl 의 젬마슬롯수(창 < 최대길이)
        p조, po = dv.dr.할당(n * H * 슬롯 * (D + 4) * 4), dv.dr.할당(n * H * D * 4)
        걸음 = min(슬롯, 5)                        # 블록이 타일 여럿(엔진의 전체 층 꼴) — 고리의 칸에서
        dv.띄우기(f"조각어텐션젬마{D}", (걸음 * HKV, n), 256,
                [(1, pqd), (1, pqE), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (1, p조), (0, n), (0, H), (0, HKV), (0, s0), (0, 고리), (0, W),
                 (0, 비트(눈금)), (0, 슬롯), (0, 걸음)])
        dv.띄우기(f"조각접기젬마{D}", (H, n), D, [(1, p조), (1, po), (0, n), (0, H), (0, s0), (0, 최대길이), (0, W), (0, 슬롯)])
        dv.dr.맞추기()
        return 다른칸(dv.down(po, (n, H * D)), o참[s0:s0 + n])
    다 = {}
    이름 = f"고리 {고리}(창 {W} · 덩이 {덩이} · 행 {T})"
    다[f"{이름} 프롬프트"] = 프롬프트(qkvA, oA, 0, T)
    여유 = 고리 - W + 1                         # 고리가 아직 창을 가진 맨 앞 행 = T − 여유
    다[f"{이름} 생성(행 넷까지) 3 경우"] = sum(생성(qkvA, oA, s0, n) for s0, n in ((T - 1, 1), (T - 4, 4), (T - 여유, 2)))
    p0 = max(0, T - 고리)
    칸 = np.arange(p0, T) % 고리
    kD, kE_ = zip(*[C.자리로(x, 2) for x in khA])
    vD, vE_ = zip(*[C.자리로(x, 2) for x in vhA])
    키 = dv.down(pkd, (2, HKV, 고리, D), np.int8)[:, :, 칸]
    값 = dv.down(pvd, (2, HKV, 고리 // 32, D, 32), np.int8).transpose(0, 1, 2, 4, 3).reshape(2, HKV, 고리, D)[:, :, 칸]
    다[f"{이름} 고리에 남은 캐시(키 · 값 · 지수)"] = (
        다른칸(키, np.stack([np.stack([d[j] for d in kD]) for j in range(2)]).astype(np.int8)[:, :, p0:T])
        + 다른칸(dv.down(pkE, (HKV, 고리, nb), np.int64)[:, 칸], np.stack(kE_)[:, p0:T])
        + 다른칸(값, np.stack([np.stack([d[j] for d in vD]) for j in range(2)]).astype(np.int8)[:, :, p0:T])
        + 다른칸(dv.down(pvE, (HKV, 고리, nb), np.int64)[:, 칸], np.stack(vE_)[:, p0:T]))
    assert T - p되 <= 여유, (T, p되, 여유)
    다[f"{이름} 되감기 {T - p되} 토큰 → 새 행 {n새}"] = 프롬프트(qkvB, oB, p되, p되 + n새) + 생성(qkvB, oB, p되 + n새 - 1, 1)
    return 다


def 표날(t, N, KS, r):
    """무작위 k-양자화 표의 날 바이트(GGUF 의 블록 그대로 — 행 N · 슈퍼블록 KS): 값 · 척도의 바이트는 무작위, fp16 척도(d · dmin — Q6_K 는 d)는
    유한한 값으로."""
    bb = {12: 144, 13: 176, 14: 210}[t]
    bl = r.integers(0, 256, (N, KS, bb), dtype=np.uint8)
    for a in ((0, 2) if t in (12, 13) else (208,)):
        h = (r.uniform(1e-4, 5e-3, (N, KS)) * r.choice([-1.0, 1.0], (N, KS))).astype(np.float16)
        bl[:, :, a:a + 2] = np.ascontiguousarray(h).view(np.uint8).reshape(N, KS, 2)
    return bl.reshape(-1)


def 공유시험(dv, r, D, HH, HKV, θ, T=37):
    """0.24.0 의 24-5 — KV 를 나눠 쓰는 층(계약 4.12 의 8 — 자기 k · v 가 없다): 머리정규화젬마D · RoPEKV젬마D 를 KV 헤드 0 으로 띄우면 q 의
    8비트 자리 둘 · 블록 지수가 KV 헤드와 함께 띄운 것과 같고, 버퍼의 k · v 열과 캐시(키 · 값 · 지수)는 그대로인가."""
    폭 = HH * D + 2 * HKV * D
    nb = D // 32
    고리 = 64
    qkv = (r.standard_normal((T, 폭)) * 10.0 ** r.uniform(-0.5, 0.5, (1, 폭))).astype(f32)
    pqw = dv.up((1.0 + 0.1 * r.standard_normal(D)).astype(f32))
    pkw = dv.up((1.0 + 0.1 * r.standard_normal(D)).astype(f32))
    cos, sin = C.rope표(1024, θ, hd=D, 인자=None)
    pcos, psin = dv.up(cos), dv.up(sin)
    나온 = []
    for kv in (HKV, 0):
        pq = dv.up(qkv)
        pqd, pqE = dv.up(np.zeros(2 * T * HH * D, np.int8)), dv.up(np.zeros(T * HH * nb, np.int64))
        pkd, pvd = (dv.up(np.zeros(2 * HKV * 고리 * D, np.int8)) for _ in range(2))
        pkE, pvE = (dv.up(np.zeros(HKV * 고리 * nb, np.int64)) for _ in range(2))
        dv.띄우기(f"머리정규화젬마{D}", ((T * (HH + kv) + 3) // 4, 1), 128, [(1, pq), (1, pqw), (1, pkw), (0, T), (0, HH), (0, kv), (0, 폭)])
        dv.띄우기(f"RoPEKV젬마{D}", (((HH + 2 * kv) * nb + 31) // 32, T), 256,
                [(1, pq), (1, pqd), (1, pqE), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (1, pcos), (1, psin), (0, T), (0, HH), (0, kv), (0, 폭),
                 (0, 5), (0, 고리)])
        dv.dr.맞추기()
        나온.append((dv.down(pq, (T, 폭)), dv.down(pqd, (2 * T * HH * D,), np.int8), dv.down(pqE, (T * HH * nb,), np.int64),
                    [dv.down(p, (2 * HKV * 고리 * D,), np.int8) for p in (pkd, pvd)],
                    [dv.down(p, (HKV * 고리 * nb,), np.int64) for p in (pkE, pvE)]))
    (_, qdA, qEA, kvA, _), (버B, qdB, qEB, kvB, EB) = 나온
    assert sum(int(np.count_nonzero(x)) for x in kvA) > 0            # 비교하는 쪽(KV 헤드와 함께)은 캐시를 썼다
    다 = 다른칸(qdA, qdB) + 다른칸(qEA, qEB)                         # q 의 8비트 자리 둘 · 블록 지수
    다 += 다른칸(버B[:, HH * D:], qkv[:, HH * D:])                   # 버퍼의 k · v 열은 그대로
    다 += sum(int(np.count_nonzero(x)) for x in kvB + EB)            # 캐시는 그대로(0)
    return 다


def 층입력시험(dv, r, L, N, n=256, T=37):
    """0.24.0 의 24-5 — 층마다의 입력(PLE — 계약 4.12 의 9)의 커널(커널생성 gemma4큰 K … --W16 --층입력): 투영의 정규화(머리정규화젬마256 을
    헤드 = 층 수 · KV 헤드 0 으로)를 정수계약.머리정규화 와, 표를 읽어 더하는 커널(층입력표_Q4_K · _Q5_K · _Q6_K — p = y + e · c_p)을
    정수계약.층입력 과(표는 무작위 — 어휘 수 N 은 모듈의 것 · 폭은 층 둘의 2n 으로: 폭은 띄울 때 받는다), PLE 블록(층GeGLU자리로젬마 — g 와 p_l
    에서 proj 의 입력(8비트 자리 둘)까지)을 CPU 참조의 겔루(g, p_l)를 실수자리로 에 넣은 것과 바이트까지."""
    from 판26 import 엔진배치K
    결과 = {}
    u = (r.standard_normal((T, L * n)) * 10.0 ** r.uniform(-2, 2, (T, 1))).astype(f32)
    e = (r.standard_normal((T, L * n)) * 0.05).astype(f32)
    w = (1.0 + 0.2 * r.standard_normal(n)).astype(f32)
    wp = C.곱(w, C.층입력c_h)
    pu, pw = dv.up(u), dv.up(wp)
    dv.띄우기("머리정규화젬마256", (-(-(T * L) // 4), 1), 128, [(1, pu), (1, pw), (1, pw), (0, T), (0, L), (0, 0), (0, L * n)])
    dv.dr.맞추기()
    결과["층마다의 입력 — 정규화(머리정규화젬마256 을 헤드 = 층 수 · KV 헤드 0 으로)"] = 다른칸(dv.down(pu, (T, L * n)), C.머리정규화(u, wp, EPS, n))
    # 표를 읽어 더하기 — 형식마다: 폭 2n(층 둘)의 표 [N][2n], 토큰은 처음 · 끝 · 무작위. 정규화까지 커널로 한 뒤 표 커널
    KS, 폭 = 2, 2 * n
    토큰 = r.integers(0, N, T).astype(np.int64)
    토큰[:2] = [0, N - 1]
    p토큰 = dv.up(토큰)
    for t, 이름 in ((12, "Q4_K"), (13, "Q5_K"), (14, "Q6_K")):
        값, 척, 풀린 = 엔진배치K(t, 표날(t, N, KS, r), N, 폭)
        e2 = C.임베딩K(풀린, 토큰)
        del 풀린
        u2 = (r.standard_normal((T, 폭)) * 10.0 ** r.uniform(-2, 2, (T, 1))).astype(f32)
        pu2 = dv.up(u2)
        p값 = dv.up(np.frombuffer(값, np.uint8))
        p척 = dv.up(np.frombuffer(척, np.uint8))
        del 값, 척
        dv.띄우기("머리정규화젬마256", (-(-(T * 2) // 4), 1), 128, [(1, pu2), (1, pw), (1, pw), (0, T), (0, 2), (0, 0), (0, 폭)])
        dv.띄우기("층입력표_" + 이름, ((T * 폭 + 255) // 256, 1), 256,
                [(1, p토큰), (1, p값), (1, p척), (1, pu2), (0, T), (0, 폭), (0, 비트(C.층입력스케일(n)))])
        dv.dr.맞추기()
        결과[f"층마다의 입력 — p = y + e · c_p(층입력표_{이름} — 표 [{N}][{폭}])"] = 다른칸(dv.down(pu2, (T, 폭)), C.층입력(e2, u2, w, n))
    p = C.층입력(e, u, w, n)
    g = (r.standard_normal((T, n)) * 4).astype(f32)
    g[0, :8] = [0.0, -0.0, 1e-30, -1e-30, 30.0, -30.0, 1e4, -1e4]
    정보폭 = (T + 63) // 64 * 64
    nz, ni = 2 * T * n // 4, 2 * (n // 32) * 정보폭
    pg, pp = dv.up(g), dv.up(p)
    다 = 0
    for l in (0, L // 2, L - 1):
        h = C.겔루(g, p[:, l * n:(l + 1) * n])
        z1, i1, z2, i2 = (dv.up(np.zeros(k, np.int32)) for k in (nz, ni, nz, ni))
        dv.띄우기("실수자리로", (-(-(n // 32) // 32), T), 256, [(1, dv.up(h)), (1, z1), (1, i1), (0, T), (0, n)])
        dv.띄우기("층GeGLU자리로젬마", (-(-(n // 32) // 32), T), 256,
                [(1, pg), (1, pp), (1, z2), (1, i2), (0, T), (0, n), (0, L * n), (0, l * n)])
        dv.dr.맞추기()
        다 += (다른칸(dv.down(z1, (nz,), np.int32), dv.down(z2, (nz,), np.int32))
              + 다른칸(dv.down(i1, (ni,), np.int32), dv.down(i2, (ni,), np.int32)))
    결과["PLE 블록 — 층GeGLU자리로 = 겔루(g, p_l)(CPU 참조) + 실수자리로(바이트 — 층 처음 · 가운데 · 끝)"] = 다
    return 결과


def MoE시험(dv, r, D=2816, F=704, Fd=2112, E=128, k=8, M=150):
    """0.24.0 의 24-5 — 26B-A4B 의 MoE(계약 4.12 의 10 — 커널생성 gemma4큰 K 2816 16 8 2112 262144 --W16 --MoE K8 의 모듈): 공통의 MoE라우팅을
    로짓폭 = 전문가(공유 게이트의 열 없음)로 + MoE배율젬마 → MoE배치 → MoE모으기(h₃ 의 자리) → 전문가의 gate · up(Q4_K 의 그룹 판 "" — 실수
    [묶음 행][2F], gate · up 이 32 열마다 번갈아) → GeGLU자리로젬마(묶음 행 모두) → 전문가의 down(Q8_0 의 그룹 판 "") → MoE합둘젬마 를 차례로
    GPU 출력에서 이어 CPU 참조(정수계약 — 상위k · 전문가무게 · 전문가배율 · 행렬곱K블라스 · 겔루 · 행렬곱블라스 · 전문가합 · rms정규화 · 더)와
    비트까지. 그리고 dense 가지의 down(깊이 Fd)의 고전 Q8_0 타일 판(끝손질 없음) 모두 대 행렬곱블라스."""
    import 판MoE as PM
    from 머리128 import 자리바이트
    from 판23 import 깔기
    from 판26 import 자리행합, 정보판, 타일판들
    dr = dv.dr
    결과 = {}

    def 비우기(p, n):
        dr.확인(dr.cu.cuMemsetD8_v2(p, 0, n))

    def 영(n):
        p = dr.할당(n)
        비우기(p, n)
        return p

    # ---- 1. 라우팅(로짓폭 = 전문가) · 무게의 전문가 배율 ----
    l = np.ascontiguousarray(PM.로짓들(r, M, E)[:, :E])
    s = r.uniform(0.25, 4.0, E).astype(f32)
    참고른 = C.상위k(l, k)
    참무게 = C.전문가배율(C.전문가무게(C.소프트맥스전문가(l), 참고른), s, 참고른)
    게이트 = np.full(M, 7.0, f32)
    p로짓, p고른, p무게, p개수, p순번, p게이트 = dv.up(l), 영(M * k * 8), 영(M * k * 4), 영(E * 8), 영(M * k * 8), dv.up(게이트)
    dv.띄우기("MoE라우팅", ((M + 3) // 4, 1), 128, [(1, p로짓), (1, p고른), (1, p무게), (1, p개수), (1, p순번), (1, p게이트), (0, M), (0, E), (0, k), (0, E)])
    dv.띄우기("MoE배율젬마", ((M * k + 255) // 256, 1), 256, [(1, p무게), (1, p고른), (1, dv.up(s)), (0, M * k)])
    dr.맞추기()
    결과["라우팅(로짓폭 = 전문가 — 공유 게이트의 칸은 손대지 않는다) · 무게의 전문가 배율(MoE배율젬마)"] = (
        다른칸(dv.down(p고른, (M, k), np.int64), 참고른) + 다른칸(dv.down(p무게, (M, k)), 참무게) + 다른칸(dv.down(p게이트, (M,)), 게이트))
    # ---- 2. 배치 · 3. 모으기(공통 판 — 판MoE.py 가 따로 본다. 여기서는 이어서 쓴다) ----
    표폭 = 2 * E + 1 + 4 * (E + 1)
    p표 = 영(표폭 * 8)
    dv.띄우기("MoE배치", (1, 1), 32, [(1, p개수), (1, p표), (1, 영(E * 8)), (0, E)])
    dr.맞추기()
    표참, _ = PM.배치참조(참고른, E)
    글순번 = dv.down(p순번, (M, k), np.int64)
    pp = (표참[:E][참고른] + 글순번).reshape(-1)                      # (행, 순위) 평 번호 → 묶음 행
    묶행수 = M * k + 3 * E + 128
    묶폭 = (묶행수 + 63) // 64 * 64
    X = (r.standard_normal((M, D)) * 10.0 ** r.uniform(-1, 1, (M, 1))).astype(f32)
    바이트, M0 = 자리바이트(X, 2)
    p묶자, p묶정 = 영(2 * 묶행수 * D), 영(2 * (D // 32) * 묶폭 * 4)
    dv.띄우기("MoE모으기", (M * k, 1), 128, [(1, dv.up(바이트)), (1, dv.up(정보판(M0, 자리행합(X, 2), (M + 63) // 64 * 64))), (1, p고른),
                                         (1, p순번), (1, p표), (1, p묶자), (1, p묶정), (0, M), (0, 묶행수), (0, D), (0, k)])
    # ---- 4. 전문가: gate · up(Q4_K — 32 열마다 번갈아), down(Q8_0) ----
    guV, guS, gu표, gu_w = PM.전문가행렬(r, "Q4_K", E, 2 * F, D, True)
    dnV, dnS, dn표, dn_w = PM.전문가행렬(r, "Q8_0", E, D, F, False)
    gcol = np.concatenate([np.arange(64 * j, 64 * j + 32) for j in range(F // 32)])
    gu참, h참, y참 = np.zeros((M * k, 2 * F), f32), np.zeros((M * k, F), f32), np.zeros((M * k, D), f32)
    for e in range(E):
        ix = np.nonzero(참고른.reshape(-1) == e)[0]
        if len(ix):
            gu참[ix] = PM.행렬곱(X[ix // k], gu_w[e], 2 * F, 2)
            h참[ix] = C.겔루(gu참[ix][:, gcol], gu참[ix][:, gcol + 32])
            y참[ix] = PM.행렬곱(h참[ix], dn_w[e], D, 2)
    del gu_w, dn_w
    참자리, 참정보 = 자리바이트(h참, 2)
    pgV, pgS, pgT, pdV, pdS, pdT = (dv.up(np.frombuffer(b, np.uint8)) if isinstance(b, bytes) else dv.up(b) for b in (guV, guS, gu표, dnV, dnS, dn표))
    p치2F, p치D = dv.up(np.zeros(2 * F, f32)), dv.up(np.zeros(D, f32))
    다gu = 다down = 0
    판들 = [t for t in 타일판들]
    for 판, BM, BN, T, NT in 판들:
        pgu = 영(묶행수 * 2 * F * 4)
        dv.띄우기(f"{판}_그룹_Q4_K", ((M * k + BM - 1) // BM + E, (2 * F + BN - 1) // BN), T,
                [(1, p묶자), (1, p묶정), (1, pgV), (1, pgS), (1, p치2F), (1, pgu), (0, 묶행수), (0, D), (0, 2 * F), (0, (2 * F + 63) // 64 * 64),
                 (0, 0), (0, 0), (0, 2 * F), (1, p표), (1, pgT), (0, E)])
        dr.맞추기()
        글gu = dv.down(pgu, (묶행수, 2 * F))
        밖 = np.ones(묶행수, bool)
        밖[pp] = False
        다gu += 다른칸(글gu[pp], gu참) + int(np.count_nonzero(글gu[밖]))
    결과[f"전문가 gate · up — Q4_K 의 그룹 판 \"\" {len(판들)} 개(실수 [묶음 행][2F], 구간 밖의 행은 0 그대로)"] = 다gu
    p다자, p다정 = 영(2 * 묶행수 * F), 영(2 * (F // 32) * 묶폭 * 4)
    dv.띄우기("GeGLU자리로젬마", (-(-(F // 32) // 32), 묶행수), 256, [(1, pgu), (1, p다자), (1, p다정), (0, 묶행수), (0, F)])
    dr.맞추기()
    글자 = dv.down(p다자, (2, 묶행수, F), np.int8)
    글정 = dv.down(p다정, (2 * (F // 32), 묶폭), np.int32)
    결과["GeGLU자리로젬마(묶음 행 모두) = 겔루(g, u)(CPU 참조)의 down 입력 자리 · 정보"] = 다른칸(글자[:, pp, :], 참자리) + 다른칸(글정[:F // 32, pp], 참정보.T)
    for 판, BM, BN, T, NT in 판들:
        p출 = 영(묶행수 * D * 4)
        dv.띄우기(f"{판}_그룹_Q8_0", ((M * k + BM - 1) // BM + E, (D + BN - 1) // BN), T,
                [(1, p다자), (1, p다정), (1, pdV), (1, pdS), (1, p치D), (1, p출), (0, 묶행수), (0, F), (0, D), (0, (D + 63) // 64 * 64),
                 (0, 0), (0, 0), (1, p표), (1, pdT), (0, E)])
        dr.맞추기()
        다down += 다른칸(dv.down(p출, (묶행수, D))[pp], y참)
    결과[f"전문가 down — Q8_0 의 그룹 판 \"\" {len(판들)} 개(입력은 GeGLU자리로젬마 의 출력)"] = 다down
    # ---- 5. 합 · 정규화 둘 ----
    dd = (r.standard_normal((M, D)) * 10.0 ** r.uniform(-1, 1, (M, 1))).astype(f32)
    dd[0, :6] = [0.0, -0.0, 1e-30, -1e-30, 1e4, -1e4]
    w1 = (1.0 + 0.3 * r.standard_normal(D)).astype(f32)
    w2 = (1.0 + 0.3 * r.standard_normal(D)).astype(f32)
    o참 = C.전문가합(참무게, np.ascontiguousarray(y참.reshape(M, k, D).transpose(1, 0, 2)))
    f참 = C.더(C.rms정규화(dd, w1, EPS), C.rms정규화(o참, w2, EPS))
    pd = dv.up(dd)
    dv.띄우기("MoE합둘젬마", (M, 1), 256, [(1, p출), (1, p무게), (1, p고른), (1, p순번), (1, p표), (1, pd), (1, dv.up(w1)), (1, dv.up(w2)),
                                    (0, M), (0, D), (0, k)])
    dr.맞추기()
    결과["MoE합둘젬마 — f = RMSNorm(d; w1) + RMSNorm(o; w2), o = 순위 차례의 FMA 합(전문가 출력은 GPU 의 것)"] = 다른칸(dv.down(pd, (M, D)), f참)
    # ---- 6. dense 가지의 down — 고전 Q8_0 타일 판(끝손질 없음) ----
    q = r.integers(-127, 128, (D, Fd), dtype=np.int8)
    d = (np.abs(r.standard_normal((D, Fd // 32))) * 0.002 + 1e-5).astype(np.float16).astype(f32)
    값, 척 = 깔기(q, d, "Q8_0")
    Xd = (r.standard_normal((M, Fd)) * 10.0 ** r.uniform(-1, 1, (M, 1))).astype(f32)
    바이트d, M0d = 자리바이트(Xd, 2)
    pxd, pid = dv.up(바이트d), dv.up(정보판(M0d, 자리행합(Xd, 2), (M + 63) // 64 * 64))
    pv, ps = dv.up(np.frombuffer(값, np.uint8)), dv.up(np.frombuffer(척, np.uint8))
    참d = C.행렬곱블라스(Xd, q, d, np.zeros(D, f32), 자리수=2)
    다고전 = 0
    for 판, BM, BN, T, NT in 판들:
        p출 = 영(M * D * 4)
        dv.띄우기(판, ((M + BM - 1) // BM, (D + BN - 1) // BN), T,
                [(1, pxd), (1, pid), (1, pv), (1, ps), (1, p치D), (1, p출), (0, M), (0, Fd), (0, D), (0, (D + 63) // 64 * 64), (0, 0), (0, 0)])
        dr.맞추기()
        다고전 += 다른칸(dv.down(p출, (M, D)), 참d)
    결과[f"dense 가지의 down(깊이 {Fd}) — 고전 Q8_0 타일 판 \"\" {len(판들)} 개 대 행렬곱블라스"] = 다고전
    return 결과


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    global H, V정규화, 눈금
    ptx, 출력 = [a for a in sys.argv[1:] if not a.startswith("--")][:2]
    if "--MoE" in sys.argv:                                          # 0.24.0 의 24-5 — Gemma 4 26B-A4B 의 모듈: MoE 의 커널과 고전 Q8_0 판만
        dv, r = 장치(ptx), np.random.default_rng(26)
        결과 = {"경우": MoE시험(dv, r)}
        for k, v in 결과["경우"].items():
            print(f"  {k}: 불일치 {v}", flush=True)
        결과["모두 0"] = all(v == 0 for v in 결과["경우"].values())
        print("모두 0:", 결과["모두 0"], flush=True)
        json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return
    if "--층입력" in sys.argv:                                       # 0.24.0 의 24-5 — Gemma 4 E2B · E4B 의 모듈: 층마다의 입력의 커널만
        L = int(sys.argv[sys.argv.index("--층입력") + 1])
        N = int(sys.argv[sys.argv.index("--어휘") + 1]) if "--어휘" in sys.argv else 262144
        dv, r = 장치(ptx), np.random.default_rng(4)
        결과 = {"층 수": L, "어휘 수": N, "경우": 층입력시험(dv, r, L, N)}
        for D, θ in ((256, 1e4), (512, 1e6)):                          # KV 를 나눠 쓰는 층 — 질의 헤드 8 · KV 헤드 2(E4B 의 모양)
            결과["경우"][f"KV 를 나눠 쓰는 층 — 헤드 {D}: KV 헤드 0 으로 띄운 q 의 8비트 자리 둘 · 블록 지수 = KV 헤드와 함께 띄운 것, 버퍼의 k · v 열 · 캐시는 그대로"] = (
                공유시험(dv, r, D, 8, 2, θ))
        for k, v in 결과["경우"].items():
            print(f"  {k}: 불일치 {v}", flush=True)
        결과["모두 0"] = all(v == 0 for v in 결과["경우"].values())
        print("모두 0:", 결과["모두 0"], flush=True)
        json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return
    T = int(sys.argv[sys.argv.index("--행") + 1]) if "--행" in sys.argv else 300
    젬마3 = "--gemma3" in sys.argv
    # --shape <너비>,<확장폭>,<헤드 수>,<KV 헤드 수> — 모듈의 모양(--gemma3 의 기본은 4B: 2560,10240,8,4. 270M 은 640,2048,4,1)
    모양 = [int(x) for x in sys.argv[sys.argv.index("--shape") + 1].split(",")] if "--shape" in sys.argv else [2560, 10240, 8, 4]
    assert len(모양) == 4 and (젬마3 or "--shape" not in sys.argv), "--shape 은 --gemma3 과 함께, 값 넷"
    # --헤드 <크기> — Gemma 3 모듈의 헤드 크기(기본 256, 27B 는 128)
    헤드3 = int(sys.argv[sys.argv.index("--헤드") + 1]) if "--헤드" in sys.argv else 256
    assert 헤드3 in (128, 256) and (젬마3 or "--헤드" not in sys.argv), "--헤드 는 --gemma3 과 함께, 128 · 256"
    D긴 = 헤드3 if 젬마3 else 256                                    # 긴 프롬프트 · 링 버퍼 시험의 헤드 크기(그 모듈에 있는 커널)
    dv = 장치(ptx)
    r = np.random.default_rng(4)
    결과 = {"행": T, "경우": {}}
    if 젬마3:
        # ---- 0.24.0 — Gemma 3 의 모듈(커널생성 gemma3큰 — 계약 4.14): v 는 그대로 · 점수 스케일 1/16. 헤드 256 · 질의 8 · KV 4(4B 의 꼴)의
        #      슬라이딩 윈도 층(θ 10⁴)과 전체 어텐션 층(θ 10⁶ · 선형 스케일링 8 — RoPE 표의 일) ----
        #      헤드 128(27B)이면 점수 스케일은 fp32(1/√(너비 ÷ 헤드 수)) — llama.cpp 가 층이 62 개인 파일에 쓰는 값
        V정규화, 눈금, H = False, float(C.점수눈금(모양[0] // 모양[2] if 헤드3 == 128 else 헤드3)), 모양[2]
        결과["점수 스케일"] = 눈금
        결과["경우"].update(층시험(dv, 헤드3, 모양[3], 10000.0, None, T, T // 2, [창없음, 130], r))
        다 = 층시험(dv, 헤드3, 모양[3], 1e6, np.full(헤드3 // 2, 8.0, f32), T, T // 2, [창없음], r)
        결과["경우"].update({f"전체 층(선형 스케일링 8) {k}": v for k, v in 다.items()})
    else:
        인자 = np.where(np.arange(256) < 64, 1.0, 1e30).astype(f32)  # Gemma 4 12B 의 rope_freqs 꼴(앞 64 짝만 돈다)
        # ---- 슬라이딩 윈도 층(헤드 256 · KV 8)과 전체 층(헤드 512 · KV 1, rope_freqs) ----
        결과["경우"].update(층시험(dv, 256, 8, 10000.0, None, T, T // 2, [창없음, 130], r))
        결과["경우"].update(층시험(dv, 512, 1, 1e6, 인자, T, T // 2, [창없음], r))
    # ---- 묶음 둘에 걸친 창(행 1,100 — 질의 둘 · KV 하나로 줄여) ----
    H = 2
    다 = 층시험(dv, D긴, 1, 10000.0, None, 1100, 1030, [130, 1024, 창없음], r,
              생성들=[(1099, 1), (1022, 2), (1096, 4), (1030, 1), (129, 4), (1086, 4)])     # 그룹 경계 1,024 · 창 경계 · 슬롯 18 의 끝
    결과["경우"].update({f"긴 {k}": v for k, v in 다.items() if "어텐션" in k})
    # ---- 0.13.0 12-3h-9 — 윈도 층의 고리 버퍼: 엔진의 꼴(창 1,024 · 덩이 512 → 고리 1,536)과 덩이가 64 의 배수가 아닌 작은 꼴 ----
    결과["경우"].update(고리시험(dv, D긴, 1, 10000.0, 3200, 1536, 512, 1024, r, (3200 - 450, 700)))
    결과["경우"].update(고리시험(dv, D긴, 1, 10000.0, 1000, 256, 100, 130, r, (1000 - 120, 300)))
    H = 16
    # ---- 크기와 상관없는 판 ----
    D, F, n = (모양[0], 모양[1], 37) if 젬마3 else (3840, 15360, 37)     # 모듈의 너비 · 확장폭(Gemma 3 — --shape, 기본 4B · Gemma 4 12B)
    h = r.standard_normal((n, D)).astype(f32)
    y = (r.standard_normal((n, D)) * 3).astype(f32)
    w = (1.0 + 0.1 * r.standard_normal(D)).astype(f32)
    s = f32(0.7421875)
    for 배율 in (None, s):
        참 = C.더(h, C.rms정규화(y, w, EPS))
        if 배율 is not None:
            참 = C.곱(참, 배율)
        ph = dv.up(h)
        dv.띄우기("RMS더하기젬마", (n, 1), 256, [(1, ph), (1, dv.up(y)), (1, dv.up(w)), (0, 0 if 배율 is None else 비트(배율)), (0, n), (0, D)])
        dv.dr.맞추기()
        결과["경우"][f"RMS더하기{' · 층 배율' if 배율 is not None else ''}"] = 다른칸(dv.down(ph, (n, D)), 참)
    c_e = C.제곱근(f32(D))
    px = dv.up(h)
    dv.띄우기("곱하기젬마", ((n * D + 255) // 256, 1), 256, [(1, px), (0, 비트(c_e)), (0, n * D)])
    dv.dr.맞추기()
    결과["경우"]["곱하기(√너비)"] = 다른칸(dv.down(px, (n, D)), C.곱(h, c_e))
    g = (r.standard_normal((n, F)) * 4).astype(f32)
    u = r.standard_normal((n, F)).astype(f32)
    g[0, :8] = [0.0, -0.0, 1e-30, -1e-30, 30.0, -30.0, 1e4, -1e4]
    gu = np.empty((n, 2 * F), f32)
    for j in range(F // 32):
        gu[:, 64 * j:64 * j + 32] = g[:, 32 * j:32 * j + 32]
        gu[:, 64 * j + 32:64 * j + 64] = u[:, 32 * j:32 * j + 32]
    pgo = dv.dr.할당(n * F * 4)
    dv.띄우기("GeGLU젬마", ((n * F + 255) // 256, 1), 256, [(1, dv.up(gu)), (1, pgo), (0, n), (0, F)])
    dv.dr.맞추기()
    결과["경우"]["GeGLU"] = 다른칸(dv.down(pgo, (n, F)), C.겔루(g, u))
    z = (r.standard_normal((5, 262144)) * 20).astype(f32)
    z[0, :6] = [0.0, -0.0, 1e-20, 300.0, -300.0, 1e30]
    pz = dv.up(z)
    dv.띄우기("소프트캡젬마", ((z.size + 255) // 256, 1), 256, [(1, pz), (0, z.size), (0, 비트(60.0)), (0, 비트(2.0 / 30.0)), (0, 비트(-30.0))])
    dv.dr.맞추기()
    결과["경우"]["소프트캡"] = 다른칸(dv.down(pz, z.shape), C.소프트캡(z, 30.0))
    # ---- 0.13.0 12-3h-8 — 합친 판: 두 판을 잇달아 띄운 것과 바이트까지(잔차 · 자리 · 정보 — 행 합 판의 칸까지) ----
    정보폭 = (n + 63) // 64 * 64
    γ = (1.0 + 0.1 * r.standard_normal(D)).astype(f32)
    for 배율 in (None, s):
        sb = 0 if 배율 is None else 비트(배율)
        ph1, ph2 = dv.up(h), dv.up(h)
        nz, ni = 2 * n * D // 4, 2 * 2 * (D // 32) * 정보폭
        z1, i1, z2, i2 = (dv.up(np.zeros(k, np.int32)) for k in (nz, ni, nz, ni))
        py, pw, pg = dv.up(y), dv.up(w), dv.up(γ)
        dv.띄우기("RMS더하기젬마", (n, 1), 256, [(1, ph1), (1, py), (1, pw), (0, sb), (0, n), (0, D)])
        dv.띄우기("RMS정규화정수줄", (n, 1), 256, [(1, ph1), (1, pg), (1, z1), (1, i1), (0, n)])
        dv.띄우기("RMS더하기정수줄젬마", (n, 1), 256, [(1, ph2), (1, py), (1, pw), (0, sb), (1, pg), (1, z2), (1, i2), (0, n), (0, D)])
        dv.dr.맞추기()
        결과["경우"][f"RMS더하기정수줄{' · 층 배율' if 배율 is not None else ''} = RMS더하기 + RMS정규화정수줄(바이트)"] = (
            다른칸(dv.down(ph1, (n, D)), dv.down(ph2, (n, D))) + 다른칸(dv.down(z1, (nz,), np.int32), dv.down(z2, (nz,), np.int32))
            + 다른칸(dv.down(i1, (ni,), np.int32), dv.down(i2, (ni,), np.int32)))
    pgu, pmid = dv.up(gu), dv.dr.할당(n * F * 4)
    nz, ni = 2 * n * F // 4, 2 * (F // 32) * 정보폭
    z1, i1, z2, i2 = (dv.up(np.zeros(k, np.int32)) for k in (nz, ni, nz, ni))
    dv.띄우기("GeGLU젬마", ((n * F + 255) // 256, 1), 256, [(1, pgu), (1, pmid), (0, n), (0, F)])
    dv.띄우기("실수자리로", (-(-(F // 32) // 32), n), 256, [(1, pmid), (1, z1), (1, i1), (0, n), (0, F)])
    dv.띄우기("GeGLU자리로젬마", (-(-(F // 32) // 32), n), 256, [(1, pgu), (1, z2), (1, i2), (0, n), (0, F)])
    dv.dr.맞추기()
    결과["경우"]["GeGLU자리로 = GeGLU + 실수자리로(바이트)"] = (다른칸(dv.down(z1, (nz,), np.int32), dv.down(z2, (nz,), np.int32))
                                                       + 다른칸(dv.down(i1, (ni,), np.int32), dv.down(i2, (ni,), np.int32)))
    for k, v in 결과["경우"].items():
        print(f"  {k}: 다른 칸 {v}", flush=True)
    결과["모두 0"] = all(v == 0 for v in 결과["경우"].values())
    print("모두 0:", 결과["모두 0"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
