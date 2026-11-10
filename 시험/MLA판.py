#!/usr/bin/env python3
"""0.24.0 24-2 — deepseek2 의 판(커널생성 deepseek2큰 — 커널/생성MLA.gl)을 CPU 참조(계약/정수계약.py — 계약 4.13)와 비트까지. 무작위 값으로
  1. RMS정규화정수줄MLA{q 압축} — q 압축의 RMSNorm 뒤 선형층의 자리 · 정보(머리128.자리바이트 의 배치)
  2. KV쓰기MLA — 키 캐시(c′ = RMSNorm · k_pe 의 NORM RoPE — 블록 18)의 자리 · 지수, 값 캐시(c′ — 블록 16, 키 32 개 블록마다 [차원][키])
  3. Q조립MLA — Q = [q_abs | q_pe 의 NORM RoPE] 의 자리 · 지수
  4. 타일어텐션MLA · 접기MLA(프롬프트 — 한 번에 · 행 64 개씩 · 덩이 둘)와 조각어텐션MLA · 조각접기MLA(생성 — 행 1 · 2 · 4, 슬롯걸음 둘)
     — 헤드마다 정수계약.어텐션(키 576 · 값 512 — 값은 키의 앞 512) 과. 행 1,100 은 그룹 둘(1,024 를 넘는다)
  5. 머리행렬곱MLA — attn_k_b 꼴(입력걸음 N + 64 · 깊이 N · 열 512)과 attn_v_b 꼴(입력걸음 512 · 깊이 512 · 열 V): 헤드마다 행렬곱블라스 와
  6. MoE라우팅시그 — 고른 번호 · 가중치(정규화 · 배율, 그리고 정규화 없이 · 배율 1) · 공유 게이트 1 · 개수 · 순번의 순열(같은 점수는 작은 번호 먼저)

  python -I 시험/MLA판.py <ptx> <출력.json> [--머리 20] [--nope 192] [--값 256] [--qlora 768] [--eps 1e-5] [--행 1100] [--배율 1.8]
마지막 줄 "모두 0: True|False".
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "계약"))
sys.path.insert(0, HERE)
import 정수계약 as C        # noqa: E402
from 머리128 import 자리바이트    # noqa: E402
from 판26 import 정보판, 자리행합    # noqa: E402
from 젬마판 import 장치, 비트, 다른칸    # noqa: E402

f32 = np.float32
창없음 = 1 << 40
값깃발 = ("--머리", "--nope", "--값", "--qlora", "--eps", "--행", "--배율")


def 인자값(n, a):
    return type(a)(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else a


def 무작위(r, 모양, 폭=1.0):
    """행마다 크기가 다른 값(10^±폭) — 0 인 행 하나"""
    x = (r.standard_normal(모양) * 10.0 ** r.uniform(-폭, 폭, (모양[0], 1))).astype(f32)
    x[min(3, 모양[0] - 1)] = 0
    return x


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    위치인자, i = [], 1
    while i < len(sys.argv):
        if sys.argv[i] in 값깃발:
            i += 2
            continue
        if not sys.argv[i].startswith("--"):
            위치인자.append(sys.argv[i])
        i += 1
    ptx, 출력 = 위치인자[:2]
    H, N, V, QA = 인자값("--머리", 20), 인자값("--nope", 192), 인자값("--값", 256), 인자값("--qlora", 768)
    eps, T, β = 인자값("--eps", 1e-5), 인자값("--행", 1100), 인자값("--배율", 1.8)
    S = N + 64                                                    # q 의 헤드 걸음(nope + rope)
    눈금 = f32(1.0 / np.sqrt(np.float64(S)))
    dv = 장치(ptx)
    r = np.random.default_rng(2402)
    다 = {}
    t0 = time.time()

    def 적기(이름, n):
        다[이름] = int(n)
        print(("다름! " if n else "") + f"{이름}: {n}", flush=True)

    # ---- 1. q 압축의 RMSNorm → 자리 ----
    if QA:
        M = 70
        X = 무작위(r, (M, QA))
        γ = (1.0 + 0.1 * r.standard_normal(QA)).astype(f32)
        바이트, M0 = 자리바이트(C.rms정규화(X, γ, eps), 2)
        폭 = (M + 63) // 64 * 64
        pz, pi = dv.up(np.zeros(2 * M * QA // 4, np.int32)), dv.up(np.zeros(2 * (QA // 32) * 폭, np.int32))
        dv.띄우기(f"RMS정규화정수줄MLA{QA}", (M, 1), 256, [(1, dv.up(X)), (1, dv.up(γ)), (1, pz), (1, pi), (0, M)])
        dv.dr.맞추기()
        적기("RMS정규화정수줄 자리", 다른칸(dv.down(pz, (2, M, QA), np.int8), 바이트))
        적기("RMS정규화정수줄 정보", 다른칸(dv.down(pi, (2 * (QA // 32), 폭), np.int32)[:QA // 32, :M], M0.T))

    # ---- 2 · 3. KV 쓰기 · Q 조립 ----
    L = (T + 1023) // 1024 * 1024
    cos, sin = C.rope표(L, 1e6, hd=64)
    kv = 무작위(r, (T, 576), 0.5)
    γkv = (1.0 + 0.1 * r.standard_normal(512)).astype(f32)
    q = 무작위(r, (T, H * S), 0.5)
    qabs = 무작위(r, (T, H * 512), 0.5)
    위치 = np.arange(T)
    c참 = C.rms정규화(kv[:, :512], γkv, eps)
    K참 = np.concatenate([c참, C.rope이웃(kv[:, 512:], cos, sin, 위치)], 1)
    Q참 = np.concatenate([np.concatenate([qabs[:, 512 * h:512 * h + 512], C.rope이웃(q[:, S * h + N:S * h + S], cos, sin, 위치)], 1)
                         for h in range(H)], 1)
    pcos, psin, pkv, pγ = dv.up(cos), dv.up(sin), dv.up(kv), dv.up(γkv)

    def 캐시():
        return dv.dr.할당(2 * L * 576), dv.dr.할당(L * 18 * 8), dv.dr.할당(2 * L * 512), dv.dr.할당(L * 16 * 8)

    def KV쓰기(p입력, n, 시작, 캐시들):
        pkd, pkE, pvd, pvE = 캐시들
        dv.띄우기("KV쓰기MLA", (n, 1), 256, [(1, p입력), (1, pγ), (1, pcos), (1, psin), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (0, n), (0, 시작), (0, L)])

    def Q조립(pab, pq_, n, 시작):
        pqd, pqE = dv.dr.할당(2 * n * H * 576), dv.dr.할당(n * H * 18 * 8)
        dv.띄우기("Q조립MLA", (H, n), 256, [(1, pab), (1, pq_), (1, pcos), (1, psin), (1, pqd), (1, pqE), (0, n), (0, H), (0, S), (0, 시작)])
        return pqd, pqE

    캐 = 캐시()
    KV쓰기(pkv, T, 0, 캐)
    pqd, pqE = Q조립(dv.up(qabs), dv.up(q), T, 0)
    dv.dr.맞추기()
    kD, kE = C.자리로(K참, 2)
    vD, vE = C.자리로(c참, 2)
    qD, qE = C.자리로(Q참, 2)
    pkd, pkE, pvd, pvE = 캐
    적기("KV쓰기 키 자리", 다른칸(dv.down(pkd, (2, L, 576), np.int8)[:, :T], kD.astype(np.int8)))
    적기("KV쓰기 키 지수", 다른칸(dv.down(pkE, (L, 18), np.int64)[:T], kE))
    값 = dv.down(pvd, (2, L // 32, 512, 32), np.int8).transpose(0, 1, 3, 2).reshape(2, L, 512)
    적기("KV쓰기 값 자리", 다른칸(값[:, :T], vD.astype(np.int8)))
    적기("KV쓰기 값 지수", 다른칸(dv.down(pvE, (L, 16), np.int64)[:T], vE))
    적기("값 = 키의 앞 512(CPU 참조)", 다른칸(kD[:, :, :512], vD) + 다른칸(kE[:, :16], vE))
    적기("Q조립 자리", 다른칸(dv.down(pqd, (2, T, H * 576), np.int8), qD.astype(np.int8)))
    적기("Q조립 지수", 다른칸(dv.down(pqE, (T, H * 18), np.int64), qE))

    # ---- 4. 어텐션 ----
    o참 = np.concatenate([C.어텐션(Q참[:, 576 * h:576 * h + 576], K참, c참, 눈금=눈금) for h in range(H)], 1)
    print(f"  CPU 참조(어텐션 헤드 {H}): {time.time() - t0:.0f} 초", flush=True)
    묶음수 = L // 1024

    def 프롬프트(n, 시작, pqd_, pqE_, 캐시들, po, 블록=None):
        pkd_, pkE_, pvd_, pvE_ = 캐시들
        블록 = 블록 or n
        p묶 = dv.dr.할당(블록 * H * 묶음수 * 516 * 4)
        for r0 in range(0, n, 블록):
            nb_ = min(블록, n - r0)
            dv.띄우기("타일어텐션MLA", (-(-nb_ * H // 16) * 묶음수, 1), 128,
                    [(1, pqd_), (1, pqE_), (1, pkd_), (1, pkE_), (1, pvd_), (1, pvE_), (1, p묶), (0, n), (0, H), (0, 1), (0, 시작), (0, L), (0, L),
                     (0, 창없음), (0, 비트(눈금)), (0, r0), (0, nb_), (0, 묶음수)])
            dv.띄우기("접기MLA", (H, nb_), 512, [(1, p묶), (1, po), (0, n), (0, H), (0, 시작), (0, L), (0, 창없음), (0, r0), (0, 묶음수)])

    po = dv.dr.할당(T * H * 512 * 4)
    프롬프트(T, 0, pqd, pqE, 캐, po)
    dv.dr.맞추기()
    적기("어텐션 프롬프트 한 번에", 다른칸(dv.down(po, (T, H * 512)), o참))
    po = dv.dr.할당(T * H * 512 * 4)
    프롬프트(T, 0, pqd, pqE, 캐, po, 블록=64)
    dv.dr.맞추기()
    적기("어텐션 프롬프트 행 64 개씩", 다른칸(dv.down(po, (T, H * 512)), o참))
    T1 = 600
    캐2 = 캐시()
    os_ = []
    for s0, s1 in ((0, T1), (T1, T)):
        n = s1 - s0
        KV쓰기(dv.up(kv[s0:s1]), n, s0, 캐2)
        pqd2, pqE2 = Q조립(dv.up(qabs[s0:s1]), dv.up(q[s0:s1]), n, s0)
        po2 = dv.dr.할당(n * H * 512 * 4)
        프롬프트(n, s0, pqd2, pqE2, 캐2, po2)
        dv.dr.맞추기()
        os_.append(dv.down(po2, (n, H * 512)))
    적기("어텐션 프롬프트 덩이 둘", 다른칸(np.concatenate(os_), o참))
    슬롯 = L // 64
    for 걸음 in (슬롯, 7):
        합 = 0
        for s0, n in ((T - 1, 1), (T - 4, 4), (T // 2, 2), (0, 1), (63, 2)):
            pqd3, pqE3 = Q조립(dv.up(qabs[s0:s0 + n]), dv.up(q[s0:s0 + n]), n, s0)
            p조 = dv.dr.할당(n * H * 슬롯 * 516 * 4)
            po3 = dv.dr.할당(n * H * 512 * 4)
            dv.띄우기("조각어텐션MLA", (걸음, n), 256, [(1, pqd3), (1, pqE3), (1, pkd), (1, pkE), (1, pvd), (1, pvE), (1, p조), (0, n), (0, H), (0, 1),
                                                    (0, s0), (0, L), (0, 창없음), (0, 비트(눈금)), (0, 슬롯), (0, 걸음)])
            dv.띄우기("조각접기MLA", (H, n), 512, [(1, p조), (1, po3), (0, n), (0, H), (0, s0), (0, L), (0, 창없음), (0, 슬롯)])
            dv.dr.맞추기()
            합 += 다른칸(dv.down(po3, (n, H * 512)), o참[s0:s0 + n])
        적기(f"어텐션 생성(행 1 · 2 · 4 — 슬롯걸음 {걸음})", 합)

    # ---- 5. 헤드마다의 행렬곱 ----
    for 이름, 걸음, 깊이, 열 in (("attn_k_b 꼴", S, N, 512), ("attn_v_b 꼴", 512, 512, V)):
        M = 37
        X = 무작위(r, (M, H * 걸음))
        바이트, M0 = 자리바이트(X, 2)
        폭 = (M + 63) // 64 * 64
        정보 = 정보판(M0, 자리행합(X, 2), 폭)
        wq = r.integers(-127, 128, (H, 열, 깊이), dtype=np.int8)
        wq[0, 0, :] = 0                                               # 0 인 행
        wd = (np.abs(r.standard_normal((H, 열, 깊이 // 32))) * 0.002 + 1e-5).astype(np.float16).astype(f32)
        참 = np.concatenate([C.행렬곱블라스(np.ascontiguousarray(X[:, 걸음 * h:걸음 * h + 깊이]), wq[h], wd[h], np.zeros(열, f32))
                            for h in range(H)], 1)
        po = dv.dr.할당(M * H * 열 * 4)
        dv.띄우기("머리행렬곱MLA", (H * ((열 + 31) // 32), M), 32,
                [(1, dv.up(바이트)), (1, dv.up(정보)), (1, dv.up(wq)), (1, dv.up(wd)), (1, po), (0, M), (0, H), (0, H * 걸음), (0, 걸음), (0, 깊이), (0, 열)])
        dv.dr.맞추기()
        적기(f"머리행렬곱 {이름}", 다른칸(dv.down(po, (M, H * 열)), 참))

    # ---- 6. 시그모이드 라우팅 ----
    M, E, k = 300, 64, 4
    l = (r.standard_normal((M, E)) * 2.0).astype(f32)
    b = (r.standard_normal(E) * 0.01).astype(f32)
    b[7] = b[3]
    for t in r.choice(M, M // 5, replace=False):
        l[t, 3] = l[t, 7] = f32(4.0 + r.random())                    # 같은 점수 둘(상위 k 안 — 작은 번호 먼저)
    l[:, 9] = f32(-50.0)                                              # 고르지 않는 전문가
    for 정규, 배 in ((1, β), (0, 1.0)):
        고참, w참 = C.시그모이드전문가(l, b, k, bool(정규), 배)
        n참 = np.bincount(고참.reshape(-1), minlength=E)
        p고, p무, p개, p순, p게 = dv.dr.할당(M * k * 8), dv.dr.할당(M * k * 4), dv.up(np.zeros(E, np.int64)), dv.dr.할당(M * k * 8), dv.dr.할당(M * 4)
        dv.띄우기("MoE라우팅시그", ((M + 3) // 4, 1), 128, [(1, dv.up(l)), (1, dv.up(b)), (1, p고), (1, p무), (1, p개), (1, p순), (1, p게), (0, M), (0, E),
                                                       (0, k), (0, E), (0, 정규), (0, 비트(배))])
        dv.dr.맞추기()
        글고 = dv.down(p고, (M, k), np.int64)
        글순 = dv.down(p순, (M, k), np.int64)
        순열 = sum(int(not np.array_equal(np.sort(글순[글고 == e]), np.arange(int((글고 == e).sum())))) for e in range(E))
        적기(f"MoE라우팅시그(정규화 {정규} · 배율 {배:g}) 고른 · 가중치 · 게이트 · 개수 · 순번",
             다른칸(글고, 고참) + 다른칸(dv.down(p무, (M, k)), w참) + 다른칸(dv.down(p게, (M,)), np.ones(M, f32)) +
             다른칸(dv.down(p개, (E,), np.int64), n참) + 순열)

    결과 = {"머리": H, "nope": N, "값": V, "q 압축": QA, "eps": eps, "행": T, "배율": β, "다른 칸": 다, "초": round(time.time() - t0)}
    결과["모두 0"] = all(v == 0 for v in 다.values())
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("모두 0:", 결과["모두 0"], flush=True)


if __name__ == "__main__":
    main()
