#!/usr/bin/env python3
"""0.25.0 25-3 — Inkling 2 의 판(커널생성 inkling2큰 — 커널/생성inkling2.gl · 생성젬마.gl 의 잉클링판)을 CPU 참조(계약/정수계약.py — 계약 4.15)와
비트까지. 무작위 값으로
  1. 머리준비잉클링128 — q ← 클립(q ⊙ q클립; c) · τ, k ← 클립(k; c), v ← RMSNorm0(v)(타우씀 0 · 1 — τ 는 위치 9,000 부터라 1 이 아니다)
  2. 상대준비잉클링 — r ← 클립(r ⊙ r클립; c_r) · τ
  3. KV잉클링128 — RoPE 없이 q 자리 · 지수, 키 · 값 캐시(값은 키 32 개 블록마다 [차원][키])
  4. 타일어텐션잉클링128 · 접기잉클링128(프롬프트 — 한 번에 · 엔진 꼴 · 덩이 둘)과 조각어텐션잉클링128 · 조각접기잉클링128(생성 — 행 1 · 2 · 4,
     슬롯걸음 둘): 상대 위치 편향(β = 상대편향표 — 범위 밖은 0)과 싱크를 넣은 정수계약.어텐션 과(헤드마다)
  5. 게이트잉클링 — o · σ(g)
헤드 24 · KV 8(Inkling-2-Mini 의 모양 — 그룹 셋)에 창 130 · 1,024 · 없음, 그리고 묶음 둘에 걸친 경우(행 1,100 — 헤드 셋 · KV 하나로 줄여).
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


def 앞단시험(dv, r, H, HKV, T=37, 시작=9000):
    """1 · 2 — 머리준비잉클링128 · 상대준비잉클링 (타우씀 0 · 1)"""
    다 = {}
    폭 = H * D + 2 * HKV * D + 512
    qkv = (r.standard_normal((T, 폭)) * 10.0 ** r.uniform(-0.5, 0.5, (1, 폭))).astype(f32)
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
    # 클립이 실제로 걸린 헤드가 있는가(시험의 범위)
    q0 = C.곱(qkv[:, :H * D], qclip.reshape(1, -1)).reshape(T * H, D)
    노름 = np.sqrt((q0.astype(np.float64) ** 2).sum(1))
    print(f"  앞단: q 헤드 {T * H} 개 가운데 클립이 걸린 것 {(노름 > float(QK클립)).sum()} 개", flush=True)
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
