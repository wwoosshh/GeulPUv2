#!/usr/bin/env python3
"""0.25.0 25-3 — Inkling 2 의 판(커널생성 inkling2큰 — 커널/생성inkling2.gl · 생성젬마.gl 의 잉클링판)을 CPU 참조(계약/정수계약.py — 계약 4.15)와
비트까지. 무작위 값으로
  1. 머리준비잉클링128 — q ← 클립(q ⊙ q클립; c) · τ, k ← 클립(k; c), v ← RMSNorm0(v)(타우씀 0 · 1 — τ 는 위치 9,000 부터라 1 이 아니다)
  2. 상대준비잉클링 — r ← 클립(r ⊙ r클립; c_r) · τ
  3. KV잉클링128 — RoPE 없이 q 자리 · 지수, 키 · 값 캐시(값은 키 32 개 블록마다 [차원][키])
  4. 타일어텐션잉클링128 · 접기잉클링128(프롬프트 — 한 번에 · 엔진 꼴 · 덩이 둘)과 조각어텐션잉클링128 · 조각접기잉클링128(생성 — 행 1 · 2 · 4,
     슬롯걸음 둘): 상대 위치 편향(β = 상대편향표 — 범위 밖은 0)과 싱크를 넣은 정수계약.어텐션 과(헤드마다)
  5. 게이트잉클링 — o · σ(g)
  6. 합성곱잉클링 · 합성곱상태잉클링 — 짧은합성곱(정적 탭 · 동적 탭, 처음부터 덩이 넷 — 행 1 · 2 의 덩이는 상태의 옛 행이 앞으로)
  7. 블록섞기잉클링 — 블록섞기(상태 1 · 4 · 5 · 7, 질의 1 ~ 4, 점수가 고른 것 · 퍼진 것 · 적힌지수가 0 으로 가는 것)
  8. 반복섞기잉클링 — 반복제어(그리고 0 · −0 · 포화 · 넘침 값, c = −0), RMS0둘정수줄잉클링 — [RMSNorm0(c) | RMSNorm0(m)] 의 실수자리로 와 바이트까지
  9. SITU잉클링 — SITU(β 4 · λ 25, λ 없이), SITU자리로잉클링 — SITU 참값의 실수자리로 와 바이트까지
 10. MoE라우팅시그(전문가 128 중 8 · 선택 편향 · 정규화 · × 8) — 시그모이드전문가, MoE합잉클링 — 전문가합, 소프트캡젬마(c 30) — 소프트캡
 11. 엔진의 작은 판 — 열복사잉클링 · 더하기잉클링 · 로짓가리기잉클링
헤드 24 · KV 8(Inkling-2-Mini 의 모양 — 그룹 셋)에 창 130 · 1,024 · 없음, 그리고 묶음 둘에 걸친 경우(행 1,100 — 헤드 셋 · KV 하나로 줄여).
q · k 헤드의 반은 세 배로 키워 클립이 걸리게 한다(1 · 4 — 걸린 헤드가 q · k · r 에 모두 있는지도 경우 하나로 센다).
CPU 참조의 헤드는 프로세스 여럿으로 나눠 셈한다(--일꾼, 기본 8).

  python -I 시험/inkling2판.py <ptx> <출력.json> [--행 300] [--일꾼 8]
마지막 줄 "모두 0: True|False".
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "계약"))
sys.path.insert(0, HERE)
import 정수계약 as C        # noqa: E402
from 젬마판 import 장치, 비트, 다른칸    # noqa: E402

f32 = np.float32
EPS = 1e-6
D = 128
R = 16
창없음 = 1 << 40
QK클립 = f32(22.627416610717773)       # Inkling-2-Mini 의 attention.qk_clip(fp32(16√2))
R클립 = f32(8.0)                        # attention.relative_clip
범위 = 1024                            # attention.relative_extent · sliding_window
바닥, 알파 = 8192, f32(0.10000000149011612)
눈금 = float(C.점수눈금(D))


def 머리참(args):
    """헤드 하나의 CPU 참조(프로세스 안에서) — 편향 · 싱크를 넣은 계약 어텐션"""
    qh, kh, vh, W, 표h, th = args
    return C.어텐션(qh, kh, vh, 눈금=눈금, 창=(W if W < 창없음 else None), 편향=표h, 범위=표h.shape[1], 싱크=th)


def 앞단참(qkv, H, HKV, qclip, τ):
    """머리준비잉클링128 의 참 — q · k · v [행][폭] 의 앞 (H + 2 HKV) · D 열"""
    qn = qkv.copy()
    kb, vb = H * D, H * D + HKV * D
    q = C.L2클립(C.곱(qkv[:, :kb], qclip.reshape(1, -1)), QK클립, D)
    if τ is not None:
        q = C.곱(q, τ[:, None])
    qn[:, :kb] = q
    qn[:, kb:vb] = C.L2클립(qkv[:, kb:vb], QK클립, D)
    qn[:, vb:vb + HKV * D] = C.머리정규화(qkv[:, vb:vb + HKV * D], np.ones(D, f32), EPS, D)
    return qn


def 상대참(rr, H, rclip, τ):
    x = C.L2클립(C.곱(rr, rclip.reshape(1, -1)), R클립, R)
    return C.곱(x, τ[:, None]) if τ is not None else x


def 클립키움(qkv, H, HKV):
    """q 헤드 · KV 헤드(키)의 짝수 번을 세 배로 — 노름이 c(22.6)를 넘어 클립이 걸린다(곱의 값은 CPU 참조와 GPU 가 같은 바이트를 받는다)"""
    for a in range(0, H, 2):
        qkv[:, a * D:(a + 1) * D] *= f32(3)
    for a in range(0, HKV, 2):
        qkv[:, (H + a) * D:(H + a + 1) * D] *= f32(3)


def 앞단시험(dv, r, H, HKV, T=37, 시작=9000):
    """1 · 2 — 머리준비잉클링128 · 상대준비잉클링 (타우씀 0 · 1)"""
    다 = {}
    폭 = H * D + 2 * HKV * D + 512
    qkv = (r.standard_normal((T, 폭)) * 10.0 ** r.uniform(-0.5, 0.5, (1, 폭))).astype(f32)
    클립키움(qkv, H, HKV)
    qclip = (1.0 + 0.2 * r.standard_normal(H * D)).astype(f32)
    rr = (r.standard_normal((T, H * R)) * 10.0 ** r.uniform(-0.5, 0.7, (1, H * R))).astype(f32)
    rclip = (1.0 + 0.2 * r.standard_normal(H * R)).astype(f32)
    τ표 = C.로그스케일(np.arange(시작 + T), 바닥, 알파)
    τ = τ표[시작:시작 + T]
    assert (τ != 1).all(), "τ 가 1 이 아닌 위치여야 한다"
    pqc, prc, pτ = dv.up(qclip), dv.up(rclip), dv.up(τ표)
    for 타우 in (0, 1):
        pq = dv.up(qkv)
        dv.띄우기("머리준비잉클링128", ((T * (H + HKV) + 3) // 4, 1), 128,
                [(1, pq), (1, pqc), (1, pτ), (0, T), (0, H), (0, HKV), (0, 폭), (0, 시작), (0, 타우), (0, 비트(QK클립))])
        pr = dv.up(rr)
        dv.띄우기("상대준비잉클링", ((T * H + 3) // 4, 1), 128,
                [(1, pr), (1, prc), (1, pτ), (0, T), (0, H), (0, 시작), (0, 타우), (0, 비트(R클립))])
        dv.dr.맞추기()
        참q = 앞단참(qkv, H, HKV, qclip, τ if 타우 else None)
        다[f"앞단 q · k · v (타우씀 {타우})"] = 다른칸(dv.down(pq, (T, 폭))[:, :(H + 2 * HKV) * D], 참q[:, :(H + 2 * HKV) * D])
        다[f"앞단 q · k · v 밖의 열 그대로 (타우씀 {타우})"] = 다른칸(dv.down(pq, (T, 폭))[:, (H + 2 * HKV) * D:], qkv[:, (H + 2 * HKV) * D:])
        다[f"상대 r (타우씀 {타우})"] = 다른칸(dv.down(pr, (T, H * R)), 상대참(rr, H, rclip, τ if 타우 else None))
    # 클립이 실제로 걸린 헤드가 있는가(시험의 범위) — q · k · r 마다 걸린 것과 걸리지 않은 것이 모두 있어야 한다
    def 노름(x, n):
        return np.sqrt((x.astype(np.float64).reshape(-1, n) ** 2).sum(1))
    수 = {"q": (노름(C.곱(qkv[:, :H * D], qclip.reshape(1, -1)), D), float(QK클립)),
         "k": (노름(qkv[:, H * D:(H + HKV) * D], D), float(QK클립)),
         "r": (노름(C.곱(rr, rclip.reshape(1, -1)), R), float(R클립))}
    빠짐 = 0
    for 이름, (nn, c) in 수.items():
        걸림 = int((nn > c).sum())
        print(f"  앞단: {이름} 헤드 {nn.size} 개 가운데 클립이 걸린 것 {걸림} 개", flush=True)
        빠짐 += int(걸림 == 0) + int(걸림 == nn.size)
    다["앞단 클립이 걸린 헤드 · 걸리지 않은 헤드가 q · k · r 에 모두 있다(빠진 종류 수)"] = 빠짐
    return 다


def 층시험(dv, r, H, HKV, T, T1, 창들, 일꾼, 생성들=None):
    """앞단(클립 · 정규화 — τ 는 1: 위치 < 8,192) → KV잉클링128 → 어텐션(편향 · 싱크) — 창마다 한 번에 · 엔진 꼴 · 덩이 둘, 그리고 생성"""
    if 생성들 is None:
        생성들 = [(T - 1, 1), (T - 4, 4), (T // 2, 2), (0, 1), (63, 2)]
    G_ = H // HKV
    폭 = H * D + 2 * HKV * D + 256
    최대길이 = (T + 1023) // 1024 * 1024
    묶음수 = 최대길이 // 1024
    nb = D // 32
    qkv = (r.standard_normal((T, 폭)) * 10.0 ** r.uniform(-0.5, 0.5, (1, 폭))).astype(f32)
    클립키움(qkv, H, HKV)
    qclip = (1.0 + 0.2 * r.standard_normal(H * D)).astype(f32)
    rr = (r.standard_normal((T, H * R)) * 10.0 ** r.uniform(-0.5, 0.7, (1, H * R))).astype(f32)
    rclip = (1.0 + 0.2 * r.standard_normal(H * R)).astype(f32)
    Wtab = (0.3 * r.standard_normal((범위, R))).astype(f32)
    싱크 = (2.0 * r.standard_normal((T, H))).astype(f32)
    τ표 = np.ones(최대길이, f32)
    다 = {}
    # ---- CPU 참조 ----
    t0 = time.time()
    qn = 앞단참(qkv, H, HKV, qclip, None)
    rn = 상대참(rr, H, rclip, None)
    표 = C.상대편향표(rn.reshape(T, H, R), Wtab)                     # [행][헤드][범위]
    kb, vb = H * D, H * D + HKV * D
    qh = [np.ascontiguousarray(qn[:, a * D:(a + 1) * D]) for a in range(H)]
    kh = [np.ascontiguousarray(qn[:, kb + a * D:kb + (a + 1) * D]) for a in range(HKV)]
    vh = [np.ascontiguousarray(qn[:, vb + a * D:vb + (a + 1) * D]) for a in range(HKV)]
    o참 = {}
    with ProcessPoolExecutor(일꾼) as ex:
        for W in 창들:
            일들 = [(qh[a], kh[a // G_], vh[a // G_], W, np.ascontiguousarray(표[:, a, :]), np.ascontiguousarray(싱크[:, a])) for a in range(H)]
            o참[W] = np.concatenate(list(ex.map(머리참, 일들)), 1)
    print(f"  헤드 {H} · KV {HKV} · 행 {T}: CPU 참조 {time.time() - t0:.1f} 초", flush=True)
    pqc, prc, pτ, pW = dv.up(qclip), dv.up(rclip), dv.up(τ표), dv.up(Wtab)

    def 앞단(pq, pr, n, 시작, pqd, pqE, pkd, pkE, pvd, pvE):
        dv.띄우기("머리준비잉클링128", ((n * (H + HKV) + 3) // 4, 1), 128,
                [(1, pq), (1, pqc), (1, pτ), (0, n), (0, H), (0, HKV), (0, 폭), (0, 시작), (0, 0), (0, 비트(QK클립))])
        dv.띄우기("상대준비잉클링", ((n * H + 3) // 4, 1), 128,
                [(1, pr), (1, prc), (1, pτ), (0, n), (0, H), (0, 시작), (0, 0), (0, 비트(R클립))])
        dv.띄우기("KV잉클링128", (((H + 2 * HKV) * nb + 31) // 32, n), 256,
                [(1, pq), (1, pqd), (1, pqE), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (0, n), (0, H), (0, HKV), (0, 폭), (0, 시작), (0, 최대길이)])

    def 어텐션(n, 시작, pqd, pqE, pkd, pkE, pvd, pvE, pr, pt, po, W, 엔진꼴=False):
        블록, 칸수 = n, 묶음수
        if 엔진꼴:
            if W < 창없음:
                칸수 = min(묶음수, (min(n, 1024) + W + 1022) // 1024 + 1)
            else:
                블록 = 64
        p묶 = dv.dr.할당(블록 * H * 칸수 * (D + 4) * 4)
        for r0 in range(0, n, 블록):
            nb_ = min(블록, n - r0)
            dv.띄우기("타일어텐션잉클링128", (-(-nb_ * G_ // 16) * 묶음수, HKV), 128,
                    [(1, pqd), (1, pqE), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (1, p묶), (1, pr), (1, pW), (0, n), (0, H), (0, HKV), (0, 시작),
                     (0, 최대길이), (0, 최대길이), (0, W), (0, 비트(눈금)), (0, 범위), (0, r0), (0, nb_), (0, 칸수)])
            dv.띄우기("접기잉클링128", (H, nb_), D,
                    [(1, p묶), (1, pt), (1, po), (0, n), (0, H), (0, 시작), (0, 최대길이), (0, W), (0, r0), (0, 칸수)])

    def 생성어텐션(n, 시작, pqd, pqE, pkd, pkE, pvd, pvE, pr, pt, po, W, 걸음=None):
        슬롯 = 최대길이 // 64 if W >= 최대길이 else (W + 2) // 64 + 2
        걸음 = min(슬롯, 걸음 or 슬롯)
        p조 = dv.dr.할당(n * H * 슬롯 * (D + 4) * 4)
        dv.띄우기("조각어텐션잉클링128", (걸음 * HKV, n), 256,
                [(1, pqd), (1, pqE), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (1, p조), (1, pr), (1, pW), (0, n), (0, H), (0, HKV), (0, 시작),
                 (0, 최대길이), (0, W), (0, 비트(눈금)), (0, 범위), (0, 슬롯), (0, 걸음)])
        dv.띄우기("조각접기잉클링128", (H, n), D, [(1, p조), (1, pt), (1, po), (0, n), (0, H), (0, 시작), (0, 최대길이), (0, W), (0, 슬롯)])

    def 캐시():
        return (dv.dr.할당(2 * HKV * 최대길이 * D), dv.dr.할당(HKV * 최대길이 * nb * 8),
                dv.dr.할당(2 * HKV * 최대길이 * D), dv.dr.할당(HKV * 최대길이 * nb * 8))
    pq, pr = dv.up(qkv), dv.up(rr)
    pqd, pqE = dv.dr.할당(2 * T * H * D), dv.dr.할당(T * H * nb * 8)
    pkd, pkE, pvd, pvE = 캐시()
    앞단(pq, pr, T, 0, pqd, pqE, pkd, pkE, pvd, pvE)
    dv.dr.맞추기()
    qD, qE_ = C.자리로(np.concatenate(qh, 1), 2)
    kD, kE_ = zip(*[C.자리로(x, 2) for x in kh])
    vD, vE_ = zip(*[C.자리로(x, 2) for x in vh])
    이름 = f"헤드 {H} · KV {HKV} · 행 {T}"
    다[f"{이름} 앞단"] = 다른칸(dv.down(pq, (T, 폭))[:, :(H + 2 * HKV) * D], qn[:, :(H + 2 * HKV) * D])
    다[f"{이름} 상대 r"] = 다른칸(dv.down(pr, (T, H * R)), rn)
    다[f"{이름} q 자리"] = 다른칸(dv.down(pqd, (2, T, H * D), np.int8), qD.astype(np.int8))
    다[f"{이름} q 지수"] = 다른칸(dv.down(pqE, (T, H * nb), np.int64), qE_)
    다[f"{이름} 키 자리"] = 다른칸(dv.down(pkd, (2, HKV, 최대길이, D), np.int8)[:, :, :T],
                             np.stack([np.stack([d[j] for d in kD]) for j in range(2)]).astype(np.int8))
    다[f"{이름} 키 지수"] = 다른칸(dv.down(pkE, (HKV, 최대길이, nb), np.int64)[:, :T], np.stack(kE_))
    값 = dv.down(pvd, (2, HKV, 최대길이 // 32, D, 32), np.int8).transpose(0, 1, 2, 4, 3).reshape(2, HKV, 최대길이, D)
    다[f"{이름} 값 자리"] = 다른칸(값[:, :, :T], np.stack([np.stack([d[j] for d in vD]) for j in range(2)]).astype(np.int8))
    다[f"{이름} 값 지수"] = 다른칸(dv.down(pvE, (HKV, 최대길이, nb), np.int64)[:, :T], np.stack(vE_))
    prn, pt = dv.up(rn), dv.up(싱크)                                  # 어텐션의 상대 r 은 앞단 뒤의 값(참과 같은 바이트 — 위에서 비교)
    for W in 창들:
        창글 = W if W < 창없음 else "없음"
        po = dv.dr.할당(T * H * D * 4)
        어텐션(T, 0, pqd, pqE, pkd, pkE, pvd, pvE, prn, pt, po, W)
        dv.dr.맞추기()
        다[f"{이름} 어텐션 창 {창글} 한 번에"] = 다른칸(dv.down(po, (T, H * D)), o참[W])
        po = dv.dr.할당(T * H * D * 4)
        어텐션(T, 0, pqd, pqE, pkd, pkE, pvd, pvE, prn, pt, po, W, 엔진꼴=True)
        dv.dr.맞추기()
        다[f"{이름} 어텐션 창 {창글} 엔진 꼴(묶음 칸 · 행 나눔)"] = 다른칸(dv.down(po, (T, H * D)), o참[W])
        # 덩이 둘 — 앞 덩이의 키 · 값을 캐시에(상대 r · 싱크는 그 덩이의 행)
        pkd2, pkE2, pvd2, pvE2 = 캐시()
        os_ = []
        for s0, s1 in ((0, T1), (T1, T)):
            n = s1 - s0
            pq2, pr2 = dv.up(qkv[s0:s1]), dv.up(rr[s0:s1])
            pqd2, pqE2 = dv.dr.할당(2 * n * H * D), dv.dr.할당(n * H * nb * 8)
            앞단(pq2, pr2, n, s0, pqd2, pqE2, pkd2, pkE2, pvd2, pvE2)
            po2 = dv.dr.할당(n * H * D * 4)
            어텐션(n, s0, pqd2, pqE2, pkd2, pkE2, pvd2, pvE2, dv.up(rn[s0:s1]), dv.up(싱크[s0:s1]), po2, W)
            dv.dr.맞추기()
            os_.append(dv.down(po2, (n, H * D)))
        다[f"{이름} 어텐션 창 {창글} 덩이 둘"] = 다른칸(np.concatenate(os_), o참[W])
        # 생성(행 넷까지) — 그 행들의 q(앞단을 다시 — 캐시에는 같은 값을 다시 쓴다)와 온 캐시
        합 = 0
        for s0, n in 생성들:
            pq3, pr3 = dv.up(qkv[s0:s0 + n]), dv.up(rr[s0:s0 + n])
            pqd3, pqE3 = dv.dr.할당(2 * n * H * D), dv.dr.할당(n * H * nb * 8)
            앞단(pq3, pr3, n, s0, pqd3, pqE3, pkd, pkE, pvd, pvE)
            for 걸음 in (None, 3):
                po3 = dv.dr.할당(n * H * D * 4)
                생성어텐션(n, s0, pqd3, pqE3, pkd, pkE, pvd, pvE, dv.up(rn[s0:s0 + n]), dv.up(싱크[s0:s0 + n]), po3, W, 걸음)
                dv.dr.맞추기()
                합 += 다른칸(dv.down(po3, (n, H * D)), o참[W][s0:s0 + n])
        다[f"{이름} 어텐션 창 {창글} 생성(행 넷까지) {len(생성들)} 경우 × 슬롯걸음 둘"] = 합
    return 다


def 게이트시험(dv, r, T=37, H=24):
    깊이 = H * D
    폭 = 깊이 + 64
    o = r.standard_normal((T, 깊이)).astype(f32)
    g = (r.standard_normal((T, 폭)) * 4.0).astype(f32)
    po, pg, pout = dv.up(o), dv.up(g), dv.dr.할당(T * 깊이 * 4)
    dv.띄우기("게이트잉클링", (-(-T * 깊이 // 256), 1), 256, [(1, po), (1, pg), (1, pout), (0, T), (0, 깊이), (0, 폭), (0, 32)])
    dv.dr.맞추기()
    return {"게이트 o · σ(g)": 다른칸(dv.down(pout, (T, 깊이)), C.게이트(o, g[:, 32:32 + 깊이]))}


def 합성곱시험(dv, r, T=37, n=4):
    """6 — 합성곱잉클링 · 합성곱상태잉클링: 정적 탭(채널 2,304 — 어텐션 · FFN 출력의 꼴)과 동적 탭(채널 1,024 — KV 헤드 8 · 헤드폭 128),
    입력 · 출력 · 동적 탭은 넓은 버퍼의 열(폭 · 바탕). 처음(상태 0)부터 덩이 넷(행 20 · 1 · 2 · 14)으로 이은 출력을 한 번에 셈한 짧은합성곱 과,
    덩이마다의 상태를 덩이마다 셈한 짧은합성곱 의 상태와"""
    다 = {}
    for 이름, Cn, 헤드, 동적 in (("정적", 2304, 1, False), ("동적", 1024, 8, True)):
        입력폭, 입력바탕, 출력폭, 출력바탕 = Cn + 96, 32, Cn + 64, 16
        동적폭, 동적바탕 = 헤드 * n + 24, 8
        y = (r.standard_normal((T, 입력폭)) * 10.0 ** r.uniform(-1, 1, (1, 입력폭))).astype(f32)
        y[5, 입력바탕:입력바탕 + 6] = [0.0, -0.0, 1e-30, -1e-30, 0.0, -0.0]
        w = (r.standard_normal((Cn, n)) * 0.5).astype(f32)
        a = (r.standard_normal((T, 동적폭)) * 0.3).astype(f32)
        yy = np.ascontiguousarray(y[:, 입력바탕:입력바탕 + Cn])
        aa = np.ascontiguousarray(a[:, 동적바탕:동적바탕 + 헤드 * n]) if 동적 else None
        참, _ = C.짧은합성곱(yy, w, None, aa, 헤드)
        c = C.역제곱근상수(n * (2 if 동적 else 1) + 1)
        pw, ps = dv.up(w), dv.up(np.zeros((n - 1, Cn), f32))
        글들, 밖, 상태다름, 상태 = [], 0, 0, None
        for s0, s1 in ((0, 20), (20, 21), (21, 23), (23, T)):
            m = s1 - s0
            py, po = dv.up(y[s0:s1]), dv.up(np.zeros((m, 출력폭), f32))
            dv.띄우기("합성곱잉클링", (-(-m * Cn // 256), 1), 256,
                    [(1, py), (1, po), (1, pw), (1, dv.up(a[s0:s1])), (1, ps), (0, m), (0, Cn), (0, 입력폭), (0, 입력바탕), (0, 출력폭), (0, 출력바탕),
                     (0, int(동적)), (0, 동적폭), (0, 동적바탕), (0, Cn // 헤드), (0, 비트(c))])
            dv.띄우기("합성곱상태잉클링", (-(-Cn // 256), 1), 256, [(1, py), (1, ps), (0, m), (0, Cn), (0, 입력폭), (0, 입력바탕)])
            dv.dr.맞추기()
            글 = dv.down(po, (m, 출력폭))
            글들.append(글[:, 출력바탕:출력바탕 + Cn])
            밖 += int(np.count_nonzero(글[:, :출력바탕])) + int(np.count_nonzero(글[:, 출력바탕 + Cn:]))
            _, 상태 = C.짧은합성곱(yy[s0:s1], w, 상태, aa[s0:s1] if 동적 else None, 헤드)
            상태다름 += 다른칸(dv.down(ps, (n - 1, Cn)), 상태)
        다[f"합성곱 {이름} 덩이 넷 — 출력"] = 다른칸(np.concatenate(글들), 참)
        다[f"합성곱 {이름} 덩이 넷 — 출력 밖의 열 그대로"] = 밖
        다[f"합성곱 {이름} 덩이마다 상태"] = 상태다름
    return 다


def 블록섞기시험(dv, r, T=37, Dw=2304):
    """7 — 블록섞기잉클링: 상태 수(마지막이 x) · 질의 수 · 질의의 크기(점수가 고른 것 · 퍼진 것 · 적힌지수가 0 으로 가는 것)마다 블록섞기 와.
    점수는 CPU 의 행렬곱16(Ŝ · Q — 1 절, c_D 를 곱하기 전)을, 상태는 RMSNorm0 한 Ŝ 를 넣는다(엔진은 같은 값을 W16 판 · RMS 판으로 낸다)"""
    다 = {}
    one = np.ones(Dw, f32)
    cD = C.역제곱근상수(Dw)
    for 상태수, nq, 크기 in ((1, 1, 1.0), (4, 2, 0.05), (7, 4, 2.0), (5, 3, 40.0)):
        S = [(r.standard_normal((T, Dw)) * 10.0 ** r.uniform(-1, 1, (T, 1))).astype(f32) for _ in range(상태수)]
        Qn, QE = C.가중치16((r.standard_normal((nq, Dw)) * 크기).astype(f32))
        Sh = np.stack([C.rms정규화(s, one, EPS) for s in S])
        z = np.stack([C.행렬곱16(s, Qn, QE, np.zeros(nq, f32)) for s in Sh])
        참 = C.블록섞기(S[:-1], S[-1], Qn, QE, EPS)                       # [행][질의][너비] — 판의 출력은 [질의][행][너비]
        po = dv.dr.할당(T * nq * Dw * 4)
        dv.띄우기("블록섞기잉클링", (T, nq), 256, [(1, dv.up(z)), (1, dv.up(Sh)), (1, po), (0, T), (0, 상태수), (0, nq), (0, Dw), (0, 비트(cD))])
        dv.dr.맞추기()
        다[f"블록 섞기 상태 {상태수} · 질의 {nq} · 질의 크기 {크기:g}"] = 다른칸(dv.down(po, (nq, T, Dw)), 참.transpose(1, 0, 2))
    return 다


def 반복섞기시험(dv, r, T=37, Dw=2304, 랭크=64):
    """8 — 반복섞기잉클링: 이득 · 잊기의 행렬곱 출력(CPU 의 행렬곱16 — 반복제어 와 같은 셈)을 넣어 반복제어 와, 그리고 g · f 에 0 · −0 · 포화 ·
    넘침 값, c 에 −0 을 넣은 경우를 계약의 식(탄 · 곱해더하기) 그대로와"""
    one = np.ones(Dw, f32)
    c = (r.standard_normal((T, Dw)) * 10.0 ** r.uniform(-1, 1, (T, 1))).astype(f32)
    m = (r.standard_normal((T, Dw)) * 10.0 ** r.uniform(-1, 1, (T, 1))).astype(f32)
    아래 = C.가중치16((r.standard_normal((랭크, 2 * Dw)) * 0.03).astype(f32))
    이득 = C.가중치16((r.standard_normal((Dw, 랭크)) * 0.15).astype(f32))
    잊기 = C.가중치16((r.standard_normal((Dw, 랭크)) * 0.15).astype(f32))
    a = C.행렬곱16(np.concatenate([C.rms정규화(c, one, EPS), C.rms정규화(m, one, EPS)], 1), 아래[0], 아래[1], np.zeros(랭크, f32))
    g = C.행렬곱16(a, 이득[0], 이득[1], np.zeros(Dw, f32))
    f = C.행렬곱16(a, 잊기[0], 잊기[1], np.zeros(Dw, f32))

    def 띄움(g_, f_, c_, m_):
        po = dv.dr.할당(T * Dw * 4)
        dv.띄우기("반복섞기잉클링", (-(-T * Dw // 256), 1), 256, [(1, dv.up(g_)), (1, dv.up(f_)), (1, dv.up(c_)), (1, dv.up(m_)), (1, po), (0, T * Dw)])
        dv.dr.맞추기()
        return dv.down(po, (T, Dw))
    다 = {"반복 섞기 = 반복제어": 다른칸(띄움(g, f, c, m), C.반복제어(c, m, 아래, 이득, 잊기, EPS))}
    # down 의 입력 자리 — RMS0둘정수줄잉클링 = [RMSNorm0(c) | RMSNorm0(m)](CPU 참조)을 실수자리로 에 넣은 것(자리 · 정보 바이트)
    정보폭 = (T + 63) // 64 * 64
    nz, ni = 2 * T * 2 * Dw // 4, 2 * (2 * Dw // 32) * 정보폭
    z1, i1, z2, i2 = (dv.up(np.zeros(k, np.int32)) for k in (nz, ni, nz, ni))
    cm = np.concatenate([C.rms정규화(c, one, EPS), C.rms정규화(m, one, EPS)], 1)
    dv.띄우기("실수자리로", (-(-(2 * Dw // 32) // 32), T), 256, [(1, dv.up(cm)), (1, z1), (1, i1), (0, T), (0, 2 * Dw)])
    dv.띄우기("RMS0둘정수줄잉클링", (T, 1), 256, [(1, dv.up(c)), (1, dv.up(m)), (1, z2), (1, i2), (0, T)])
    dv.dr.맞추기()
    다["RMS0둘정수줄 = [RMSNorm0(c) | RMSNorm0(m)] 의 실수자리로(자리 · 정보 바이트)"] = (
        다른칸(dv.down(z1, (nz,), np.int32), dv.down(z2, (nz,), np.int32)) + 다른칸(dv.down(i1, (ni,), np.int32), dv.down(i2, (ni,), np.int32)))
    특 = np.array([0.0, -0.0, 1e-30, -1e-30, 1e-8, -1e-8, 0.5, -0.5, 9.0, -9.0, 20.0, -20.0, 50.0, -50.0, 1e4, -1e4, 3e38, -3e38], f32)
    g2, f2, c2 = g.copy(), f.copy(), c.copy()
    g2[0, :특.size] = 특
    f2[1, :특.size] = 특
    g2[2, :특.size], f2[2, :특.size], c2[2, :특.size] = 특, 특[::-1], -0.0
    f2[3, :특.size], c2[3, :특.size] = 특, -0.0
    참2 = C.fma32(C.탄(g2), m, C.fma32(-C.탄(f2), c2, c2))
    다["반복 섞기 — 특별한 값(0 · −0 · 포화 · 넘침, c = −0)"] = 다른칸(띄움(g2, f2, c2, m), 참2)
    return 다


def SITU시험(dv, r, T=37, F=1152, 베타=4.0, 선베타=25.0):
    """9 — SITU잉클링(gate · up 이 32 열마다 번갈아 든 [행][2F])을 SITU 와(β · λ 는 Inkling-2-Mini 의 4 · 25, 그리고 λ 없이), SITU자리로잉클링 을
    SITU 의 참값을 실수자리로 에 넣은 것과 자리 · 정보의 바이트까지"""
    g = (r.standard_normal((T, F)) * 10.0 ** r.uniform(-1, 1.3, (T, 1))).astype(f32)
    u = (r.standard_normal((T, F)) * 10.0 ** r.uniform(-1, 1.7, (T, 1))).astype(f32)
    특 = np.array([0.0, -0.0, 1e-30, -1e-30, 3.0, -3.0, 9.0, -9.0, 40.0, -40.0, 200.0, -200.0, 1e4, -1e4, 3e38, -3e38], f32)
    g[0, :특.size] = 특
    u[1, :특.size] = 특
    g[2, :특.size], u[2, :특.size] = 특, 특[::-1]
    gu = np.empty((T, F // 32, 2, 32), f32)
    gu[:, :, 0, :] = g.reshape(T, F // 32, 32)
    gu[:, :, 1, :] = u.reshape(T, F // 32, 32)
    pgu = dv.up(gu.reshape(T, 2 * F))
    정보폭 = (T + 63) // 64 * 64
    nz, ni = 2 * T * F // 4, 2 * (F // 32) * 정보폭
    다 = {}
    for β, λ in ((베타, 선베타), (베타, 0.0)):
        상수 = [(0, 비트(2 * β)), (0, 비트(2.0 / β)), (0, 비트(-β)), (0, int(λ > 0)),
               (0, 비트(2 * λ if λ > 0 else 0.0)), (0, 비트(2.0 / λ if λ > 0 else 0.0)), (0, 비트(-λ if λ > 0 else 0.0))]
        참 = C.SITU(g, u, β, λ)
        ph = dv.dr.할당(T * F * 4)
        dv.띄우기("SITU잉클링", (-(-T * F // 256), 1), 256, [(1, pgu), (1, ph), (0, T), (0, F)] + 상수)
        z1, i1, z2, i2 = (dv.up(np.zeros(k, np.int32)) for k in (nz, ni, nz, ni))
        dv.띄우기("실수자리로", (-(-(F // 32) // 32), T), 256, [(1, dv.up(참)), (1, z1), (1, i1), (0, T), (0, F)])
        dv.띄우기("SITU자리로잉클링", (-(-(F // 32) // 32), T), 256, [(1, pgu), (1, z2), (1, i2), (0, T), (0, F)] + 상수)
        dv.dr.맞추기()
        다[f"SITU(β {β:g} · λ {λ:g})"] = 다른칸(dv.down(ph, (T, F)), 참)
        다[f"SITU자리로(β {β:g} · λ {λ:g}) = SITU 참값의 실수자리로(자리 · 정보 바이트)"] = (
            다른칸(dv.down(z1, (nz,), np.int32), dv.down(z2, (nz,), np.int32)) + 다른칸(dv.down(i1, (ni,), np.int32), dv.down(i2, (ni,), np.int32)))
    return 다


def MoE시험(dv, r, M=150, E=128, k=8, D=2304, 배율=8.0):
    """10 — MoE라우팅시그(Inkling-2-Mini 의 꼴 — 전문가 128 개 중 8 개, 선택 편향, 정규화 · × 8)를 시그모이드전문가 와(고른 번호 · 무게 · 개수 ·
    순번의 순열 — MLA판.py 의 6 과 같은 검사), MoE합잉클링 을 전문가합 과(전문가 출력은 무작위 — 구간은 호스트가 MoE배치 처럼 넷의 배수로 쌓아
    배치표에, 무게 · 고른 · 순번은 GPU 의 것), 소프트캡젬마(c = 30 — final_logit_softcapping)를 소프트캡 과"""
    l = (r.standard_normal((M, E)) * 2.0).astype(f32)
    b = (r.standard_normal(E) * 0.05).astype(f32)
    b[7] = b[3]
    for t in r.choice(M, M // 5, replace=False):
        l[t, 3] = l[t, 7] = f32(4.0 + r.random())                    # 같은 점수 둘(상위 k 안 — 작은 번호 먼저)
    l[:, 9] = f32(-50.0)                                              # 고르지 않는 전문가
    고참, w참 = C.시그모이드전문가(l, b, k, True, 배율)
    n참 = np.bincount(고참.reshape(-1), minlength=E)
    p고, p무, p개, p순, p게 = dv.dr.할당(M * k * 8), dv.dr.할당(M * k * 4), dv.up(np.zeros(E, np.int64)), dv.dr.할당(M * k * 8), dv.dr.할당(M * 4)
    dv.띄우기("MoE라우팅시그", ((M + 3) // 4, 1), 128, [(1, dv.up(l)), (1, dv.up(b)), (1, p고), (1, p무), (1, p개), (1, p순), (1, p게), (0, M), (0, E),
                                               (0, k), (0, E), (0, 1), (0, 비트(배율))])
    dv.dr.맞추기()
    글고 = dv.down(p고, (M, k), np.int64)
    글순 = dv.down(p순, (M, k), np.int64)
    글무 = dv.down(p무, (M, k))
    글개 = dv.down(p개, (E,), np.int64)
    순열 = sum(int(not np.array_equal(np.sort(글순[글고 == e]), np.arange(int((글고 == e).sum())))) for e in range(E))
    다 = {f"MoE라우팅시그(전문가 {E} · 고름 {k} · 정규화 · 배율 {배율:g}) 고른 · 무게 · 개수 · 순번":
         다른칸(글고, 고참) + 다른칸(글무, w참) + 다른칸(글개, n참) + 순열}
    시작 = np.zeros(E + 1, np.int64)
    시작[1:] = np.cumsum((글개 + 3) // 4 * 4)
    묶행수 = int(시작[E])
    y = (r.standard_normal((묶행수, D)) * 10.0 ** r.uniform(-1, 1, (묶행수, 1))).astype(f32)
    ys = np.stack([y[시작[글고[:, q]] + 글순[:, q]] for q in range(k)])                     # [순위][행][너비]
    po = dv.dr.할당(M * D * 4)
    dv.띄우기("MoE합잉클링", (-(-M * D // 256), 1), 256, [(1, dv.up(y)), (1, p무), (1, p고), (1, p순), (1, dv.up(시작)), (1, po), (0, M), (0, D), (0, k)])
    dv.dr.맞추기()
    다["MoE합잉클링 = 전문가합(무게 · 고른 · 순번은 GPU 의 라우팅)"] = 다른칸(dv.down(po, (M, D)), C.전문가합(글무, ys))
    z = (r.standard_normal((5, 201088)) * 20).astype(f32)
    z[0, :6] = [0.0, -0.0, 1e-20, 300.0, -300.0, 1e30]
    pz = dv.up(z)
    dv.띄우기("소프트캡젬마", ((z.size + 255) // 256, 1), 256, [(1, pz), (0, z.size), (0, 비트(60.0)), (0, 비트(2.0 / 30.0)), (0, 비트(-30.0))])
    dv.dr.맞추기()
    다["소프트캡젬마(c 30)"] = 다른칸(dv.down(pz, z.shape), C.소프트캡(z, 30.0))
    return 다


def 작은판시험(dv, r, T=37):
    """11 — 엔진(25-4)의 작은 판: 열복사잉클링(넓은 버퍼의 열 — 값 그대로, 밖의 칸은 그대로), 더하기잉클링(x + y — 더하기 하나),
    로짓가리기잉클링(패딩한 어휘의 칸만 −∞)"""
    다 = {}
    a = (r.standard_normal((T, 700)) * 3).astype(f32)
    a[0, 123:127] = [0.0, -0.0, 1e-40, -3e38]
    b = np.full((T, 500), 7.0, f32)
    pb = dv.up(b)
    dv.띄우기("열복사잉클링", (-(-T * 300 // 256), 1), 256, [(1, dv.up(a)), (1, pb), (0, T), (0, 300), (0, 700), (0, 123), (0, 500), (0, 77)])
    dv.dr.맞추기()
    참 = b.copy()
    참[:, 77:377] = a[:, 123:423]
    다["열복사(넓은 버퍼의 열 — 밖의 칸 그대로)"] = 다른칸(dv.down(pb, b.shape), 참)
    x = (r.standard_normal(T * 2304) * 10.0 ** r.uniform(-3, 3, T * 2304)).astype(f32)
    y = (r.standard_normal(T * 2304) * 10.0 ** r.uniform(-3, 3, T * 2304)).astype(f32)
    x[:4] = [0.0, -0.0, 3e38, -0.0]
    y[:4] = [-0.0, -0.0, 3e38, 0.0]
    px = dv.up(x)
    dv.띄우기("더하기잉클링", (-(-x.size // 256), 1), 256, [(1, px), (1, dv.up(y)), (0, x.size)])
    dv.dr.맞추기()
    with np.errstate(over="ignore"):
        다["더하기 x + y"] = 다른칸(dv.down(px, x.shape), C.더(x, y))
    V, 쓸 = 1000, 966
    z = (r.standard_normal((T, V)) * 20).astype(f32)
    pz = dv.up(z)
    dv.띄우기("로짓가리기잉클링", (-(-T * (V - 쓸) // 256), 1), 256, [(1, pz), (0, T), (0, V), (0, 쓸)])
    dv.dr.맞추기()
    참z = z.copy()
    참z[:, 쓸:] = -np.inf
    다["로짓 가리기(패딩 칸만 −∞)"] = 다른칸(dv.down(pz, z.shape), 참z)
    return 다


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ptx, 출력 = [a for a in sys.argv[1:] if not a.startswith("--")][:2]
    T = int(sys.argv[sys.argv.index("--행") + 1]) if "--행" in sys.argv else 300
    일꾼 = int(sys.argv[sys.argv.index("--일꾼") + 1]) if "--일꾼" in sys.argv else 8
    dv = 장치(ptx)
    r = np.random.default_rng(25)
    결과 = {"행": T, "경우": {}}
    t0 = time.time()
    assert dv.있나("잉클링표시"), "Inkling 2 의 모듈이 아니다(잉클링표시 가 없다)"
    결과["경우"].update(앞단시험(dv, r, 24, 8))
    결과["경우"].update(게이트시험(dv, r))
    결과["경우"].update(합성곱시험(dv, r))
    결과["경우"].update(블록섞기시험(dv, r))
    결과["경우"].update(반복섞기시험(dv, r))
    결과["경우"].update(SITU시험(dv, r))
    결과["경우"].update(MoE시험(dv, r))
    결과["경우"].update(작은판시험(dv, r))
    # Inkling-2-Mini 의 모양(헤드 24 · KV 8): 창 130 · 1,024(슬라이딩 윈도 층) · 없음(전체 층)
    결과["경우"].update(층시험(dv, r, 24, 8, T, T // 2, [130, 1024, 창없음], 일꾼))
    # 묶음 둘에 걸친 경우(행 1,100 — 헤드 셋 · KV 하나, 그룹 경계 1,024 · 창 경계 · 범위 밖의 거리)
    다 = 층시험(dv, r, 3, 1, 1100, 1030, [130, 1024, 창없음], 일꾼,
              생성들=[(1099, 1), (1022, 2), (1096, 4), (1030, 1), (129, 4), (1086, 4)])
    결과["경우"].update({f"긴 {k}": v for k, v in 다.items()})
    for k, v in 결과["경우"].items():
        print(f"  {k}: 불일치 {v}", flush=True)
    결과["모두 0"] = all(v == 0 for v in 결과["경우"].values())
    결과["초"] = round(time.time() - t0, 1)
    print(f"경우 {len(결과['경우'])} · {결과['초']} 초", flush=True)
    print("모두 0:", 결과["모두 0"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
