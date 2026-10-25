#!/usr/bin/env python3
"""0.12.0 12-3h-3 — Gemma 4 의 판(커널생성 gemma4큰 — 커널/생성젬마.gl)을 CPU 참조(계약/정수계약.py — 계약 4.12)와 비트까지:
머리정규화젬마D(q · k 무게 · v 무게 없이) → RoPEKV젬마D(헤드 전체 NEOX — 전체 층은 rope_freqs 표, q 자리 · 키 · 값 캐시) → 어텐션젬마D묶음 ·
접기젬마D(배율 1, 창), 그리고 RMS더하기젬마(층 배율과 함께 · 없이) · 곱하기젬마 · GeGLU젬마(gate · up 이 32 열마다 번갈아) · 소프트캡젬마.
헤드 256(질의 16 · KV 8 — 슬라이딩 윈도 층)과 512(질의 16 · KV 1 — 전체 층), 같은 행들을 덩이 둘로(앞 덩이의 키 · 값을 캐시에) 돌린
어텐션, 그리고 묶음 둘에 걸친 창(행 1,100 — 창 130 · 1,024). 0.13.0 12-3h-6 — 생성(행 넷까지)의 조각어텐션젬마D · 조각접기젬마D 도
같은 참값과(행 1 · 2 · 4 를 끝 · 가운데 · 처음 · 타일 경계, 행 1,100 은 그룹 경계 · 창 경계에).

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
                 (0, 시작), (0, 최대길이)])

    def 어텐션(n, 시작, pqd, pqE, pkd, pkE, pvd, pvE, po, W):
        p묶 = dv.dr.할당(n * H * 묶음수 * (D + 4) * 4)
        dv.띄우기(f"어텐션젬마{D}묶음", (H * 묶음수, n), D,
                [(1, pqd), (1, pqE), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (1, p묶), (0, n), (0, H), (0, HKV), (0, 시작), (0, 최대길이), (0, W),
                 (0, 비트(1.0))])
        dv.띄우기(f"접기젬마{D}", (H, n), D, [(1, p묶), (1, po), (0, n), (0, H), (0, 시작), (0, 최대길이), (0, W)])

    def 생성어텐션(n, 시작, pqd, pqE, pkd, pkE, pvd, pvE, po, W):
        슬롯 = 최대길이 // 64 if W >= 최대길이 else (W + 2) // 64 + 2        # 구조/gemma4/엔진.gl 의 젬마슬롯수
        p조 = dv.dr.할당(n * H * 슬롯 * (D + 4) * 4)
        dv.띄우기(f"조각어텐션젬마{D}", (슬롯 * HKV, n), 256,
                [(1, pqd), (1, pqE), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (1, p조), (0, n), (0, H), (0, HKV), (0, 시작), (0, 최대길이), (0, W),
                 (0, 비트(1.0)), (0, 슬롯)])
        dv.띄우기(f"조각접기젬마{D}", (H, n), D, [(1, p조), (1, po), (0, n), (0, H), (0, 시작), (0, 최대길이), (0, W), (0, 슬롯)])

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
    다[f"D{D} 값 자리"] = 다른칸(dv.down(pvd, (2, HKV, 최대길이, D), np.int8)[:, :, :T],
                             np.stack([np.stack([d[j] for d in vD]) for j in range(2)]).astype(np.int8))
    다[f"D{D} 값 지수"] = 다른칸(dv.down(pvE, (HKV, 최대길이, nb), np.int64)[:, :T], np.stack(vE_))
    for W in 창들:
        po = dv.dr.할당(T * H * D * 4)
        어텐션(T, 0, pqd, pqE, pkd, pkE, pvd, pvE, po, W)
        dv.dr.맞추기()
        다[f"D{D} 어텐션 창 {W if W < 창없음 else '없음'} 한 번에"] = 다른칸(dv.down(po, (T, H * D)), o참[W])
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
            po3 = dv.dr.할당(n * H * D * 4)
            생성어텐션(n, s0, pqd3, pqE3, pkd, pkE, pvd, pvE, po3, W)
            dv.dr.맞추기()
            합 += 다른칸(dv.down(po3, (n, H * D)), o참[W][s0:s0 + n])
        다[f"D{D} 어텐션 창 {W if W < 창없음 else '없음'} 생성(행 넷까지) {len(생성들)} 경우"] = 합
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
    다 = 층시험(dv, 256, 1, 10000.0, None, 1100, 1030, [130, 1024], r,
              생성들=[(1099, 1), (1022, 2), (1096, 4), (1030, 1), (129, 4), (1086, 4)])     # 그룹 경계 1,024 · 창 경계 · 슬롯 18 의 끝
    결과["경우"].update({f"긴 {k}": v for k, v in 다.items() if "어텐션" in k})
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
    for k, v in 결과["경우"].items():
        print(f"  {k}: 다른 칸 {v}", flush=True)
    결과["모두 0"] = all(v == 0 for v in 결과["경우"].values())
    print("모두 0:", 결과["모두 0"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
