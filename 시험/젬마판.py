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

  python -I 시험/젬마판.py <ptx> <출력.json> [--행 300]

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
    qn[:, vb:vb + HKV * D] = C.머리정규화(qkv[:, vb:vb + HKV * D], np.ones(D, f32), EPS, D)
    위치 = np.arange(T)
    qh = [C.rope(qn[:, a * D:(a + 1) * D], cos, sin, 위치) for a in range(H)]
    kh = [C.rope(qn[:, kb + a * D:kb + (a + 1) * D], cos, sin, 위치) for a in range(HKV)]
    vh = [np.ascontiguousarray(qn[:, vb + a * D:vb + (a + 1) * D]) for a in range(HKV)]
    G_ = H // HKV
    o참 = {W: np.concatenate([C.어텐션(qh[a], kh[a // G_], vh[a // G_], 눈금=1.0, 창=(W if W < T else None)) for a in range(H)], 1)
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
                     (0, 최대길이), (0, W), (0, 비트(1.0)), (0, r0), (0, nb_), (0, 칸수)])
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
                 (0, 비트(1.0)), (0, 슬롯), (0, 걸음)])
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
        qn[:, vb:vb + HKV * D] = C.머리정규화(qkv[:, vb:vb + HKV * D], np.ones(D, f32), EPS, D)
        위치 = np.arange(len(qkv))
        qh = [C.rope(qn[:, a * D:(a + 1) * D], cos, sin, 위치) for a in range(H)]
        kh = [C.rope(qn[:, kb + a * D:kb + (a + 1) * D], cos, sin, 위치) for a in range(HKV)]
        vh = [np.ascontiguousarray(qn[:, vb + a * D:vb + (a + 1) * D]) for a in range(HKV)]
        return kh, vh, np.concatenate([C.어텐션(qh[a], kh[a // G_], vh[a // G_], 눈금=1.0, 창=W) for a in range(H)], 1)
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
                     (0, 고리), (0, W), (0, 비트(1.0)), (0, 0), (0, n), (0, 칸수)])
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
                 (0, 비트(1.0)), (0, 슬롯), (0, 걸음)])
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    global H
    ptx, 출력 = [a for a in sys.argv[1:] if not a.startswith("--")][:2]
    T = int(sys.argv[sys.argv.index("--행") + 1]) if "--행" in sys.argv else 300
    dv = 장치(ptx)
    r = np.random.default_rng(4)
    결과 = {"행": T, "경우": {}}
    인자 = np.where(np.arange(256) < 64, 1.0, 1e30).astype(f32)      # Gemma 4 12B 의 rope_freqs 꼴(앞 64 짝만 돈다)
    # ---- 슬라이딩 윈도 층(헤드 256 · KV 8)과 전체 층(헤드 512 · KV 1, rope_freqs) ----
    결과["경우"].update(층시험(dv, 256, 8, 10000.0, None, T, T // 2, [창없음, 130], r))
    결과["경우"].update(층시험(dv, 512, 1, 1e6, 인자, T, T // 2, [창없음], r))
    # ---- 묶음 둘에 걸친 창(행 1,100 — 질의 둘 · KV 하나로 줄여) ----
    H = 2
    다 = 층시험(dv, 256, 1, 10000.0, None, 1100, 1030, [130, 1024, 창없음], r,
              생성들=[(1099, 1), (1022, 2), (1096, 4), (1030, 1), (129, 4), (1086, 4)])     # 그룹 경계 1,024 · 창 경계 · 슬롯 18 의 끝
    결과["경우"].update({f"긴 {k}": v for k, v in 다.items() if "어텐션" in k})
    # ---- 0.13.0 12-3h-9 — 윈도 층의 고리 버퍼: 엔진의 꼴(창 1,024 · 덩이 512 → 고리 1,536)과 덩이가 64 의 배수가 아닌 작은 꼴 ----
    결과["경우"].update(고리시험(dv, 256, 1, 10000.0, 3200, 1536, 512, 1024, r, (3200 - 450, 700)))
    결과["경우"].update(고리시험(dv, 256, 1, 10000.0, 1000, 256, 100, 130, r, (1000 - 120, 300)))
    H = 16
    # ---- 크기와 상관없는 판 ----
    D, F, n = 3840, 15360, 37
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
