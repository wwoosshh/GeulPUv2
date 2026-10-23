#!/usr/bin/env python3
"""0.12.0 12-3g-2 — MoE 의 커널(커널생성 … --MoE Q8_0 | K — 커널/생성MoE.gl)을 CPU 참조(계약/정수계약.py — 계약 4.11)와 비트까지. 무작위 값으로
  1. MoE라우팅 — 로짓(같은 값 · −0 · +0 을 섞고, 행의 60 % 가 전문가 0 을 첫째로 — 타일이 여럿인 전문가, 고르지 않는 전문가 둘)의 고른 번호 ·
     무게 · 공유 게이트, 전문가마다 순번이 0 … 개수 − 1 의 순열
  2. MoE배치 — 배치표 = numpy 의 누적합, 행 수가 0 으로 돌아온다
  3. MoE모으기 — 묶음 행의 자리 · 정보(행 합을 쓰는 모듈이면 T 도)가 그 토큰 행의 것과 같다
  4. 그룹 타일 커널(커널 변형 모두 × 형식) — gate·up(_SwiGLU정수: 2F 열 → 자리 F)과 down(""): 전문가마다 행렬곱 참조(행렬곱블라스 · 행렬곱K블라스)와.
     전문가 구간 밖의 묶음 행(넷의 배수로 올린 틈)은 손대지 않는다(처음 값 0 그대로)
  5. MoE합 — 전문가합 → 공유 전문가 FMA → 잔차
  6. 이어서 — 1 … 5 를 GPU 출력으로 이어(형식 하나 · 커널 변형 하나) 층 하나의 MoE FFN 을 CPU 참조와
전문가 행렬은 무작위(Q8_0: 정수 · 반실수 척도, k-양자화: 날 바이트에 유한한 반실수 d · dmin) — 엔진과 같은 배치로 깔아 전문가마다 이어 둔다.

  python -I 시험/판MoE.py <ptx> <출력.json> --형식 Q8_0|K [--토큰 300] [--전문가 64] [--고름 8] [--너비 2048] [--확장 512] [--빠른] [--행합]
--행합 — 행 합을 쓰는 모듈(정밀 CUDA k-양자화 모듈 — 커널생성 --먼): 정보 뒤의 T 도 모으고 · 쓰고 · 읽는다.
마지막 줄 "모두 0: True|False".
"""
import ctypes
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "계약"))
sys.path.insert(0, HERE)
import 쿠다드라이버 as G          # noqa: E402
import 시험장치               # noqa: E402
import 정수계약 as C        # noqa: E402
from 머리128 import 자리바이트    # noqa: E402
from 판23 import 깔기         # noqa: E402
from 판26 import 엔진배치K, 자리행합, 정보판, 타일판들, 블록바이트    # noqa: E402

f32 = np.float32
u64, i64 = G.c_u64, G.c_i64
형식번호 = {"Q4_K": 12, "Q5_K": 13, "Q6_K": 14}
값깃발 = ("--형식", "--토큰", "--전문가", "--고름", "--너비", "--확장")


def 인자값(n, a):
    return type(a)(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else a


def 다른칸(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.dtype == f32:
        a, b = a.view(np.int32), np.asarray(b, f32).view(np.int32)
    return int(np.count_nonzero(a != b))


def 로짓들(r, M, E):
    """[M][E + 1] — 열 E 는 공유 게이트. 같은 값 · −0 · +0, 전문가 0 을 행의 60 % 가 첫째로, 전문가 5 · E − 1 은 고르지 않는다."""
    l = (r.standard_normal((M, E + 1)) * 2.0).astype(f32)
    for t in r.choice(M, M // 5, replace=False):
        a, b = r.choice(E, 2, replace=False)
        l[t, a] = l[t, b]                                       # 같은 값 — 작은 번호 먼저
    for t in r.choice(M, M // 10, replace=False):
        a, b = r.choice(E, 2, replace=False)
        l[t, :E] = np.minimum(l[t, :E], f32(-1.0))
        l[t, a], l[t, b] = f32(0.0), f32(-0.0)                  # −0 과 +0 이 가장 큰 값 — 같은 값
    앞 = r.random(M) < 0.6
    l[앞, 0] = l[앞, :E].max(1) + f32(1.0)
    l[:, 5] = f32(-100.0)
    l[:, E - 1] = f32(-100.0)
    return l


def 배치참조(고른, E):
    """MoE배치 의 배치표 — [시작 E + 1 | 끝 E | 타일 시작 4 × (E + 1)] (시작은 행 수를 넷의 배수로 올려 쌓는다)."""
    n = np.bincount(고른.reshape(-1), minlength=E).astype(np.int64)
    a = (n + 3) // 4 * 4
    시작 = np.concatenate([[0], np.cumsum(a)])
    표 = [시작, 시작[:E] + n]
    for bm in (16, 32, 64, 128):
        표.append(np.concatenate([[0], np.cumsum((n + bm - 1) // bm)]))
    return np.concatenate(표).astype(np.int64), n


def 무작위K(r, 형식, N, K):
    """k-양자화의 날 바이트 [N][K/256 슈퍼블록] — 무작위 바이트에 유한한 반실수 d(· dmin)."""
    bb = 블록바이트[형식번호[형식]]
    bl = r.integers(0, 256, (N, K // 256, bb), dtype=np.uint8)
    d = (np.abs(r.standard_normal((N, K // 256))) * 0.01 + 1e-4).astype(np.float16)
    if 형식 == "Q6_K":
        bl[:, :, 208:210] = d.view(np.uint8).reshape(N, K // 256, 2)
    else:
        bl[:, :, 0:2] = d.view(np.uint8).reshape(N, K // 256, 2)
        m = (np.abs(r.standard_normal((N, K // 256))) * 0.01).astype(np.float16)
        bl[:, :, 2:4] = m.view(np.uint8).reshape(N, K // 256, 2)
    return bl.reshape(-1)


def gu차례(F):
    """gate [F] · up [F] 행을 출력 32 열마다 번갈아(엔진과 같다 — 판26 의 gu)."""
    순 = np.empty(2 * F, np.int64)
    for j in range(F // 32):
        순[64 * j:64 * j + 32] = np.arange(32 * j, 32 * j + 32)
        순[64 * j + 32:64 * j + 64] = F + np.arange(32 * j, 32 * j + 32)
    return 순


def 전문가행렬(r, 형식, E, N, K, gu):
    """전문가 E 개의 (값 바이트 이음, 척 바이트 이음, 전문가표 [E][2] — 낱말 자리, 참조용 가중치 목록)."""
    값들, 척들, 표, ws = [], [], np.zeros((E, 2), np.int64), []
    vo = so = 0
    for e in range(E):
        if 형식 == "Q8_0":
            q = r.integers(-127, 128, (N, K), dtype=np.int8)
            d = (np.abs(r.standard_normal((N, K // 32))) * 0.002 + 1e-5).astype(np.float16).astype(f32)
            if gu:
                순 = gu차례(N // 2)
                q, d = np.ascontiguousarray(q[순]), np.ascontiguousarray(d[순])
            값, 척 = 깔기(q, d, "Q8_0")
            w = (q, d)
        else:
            raw = 무작위K(r, 형식, N, K)
            if gu:
                행바이트 = (K // 256) * 블록바이트[형식번호[형식]]
                raw = np.ascontiguousarray(raw.reshape(N, 행바이트)[gu차례(N // 2)]).reshape(-1)
            값, 척, w = 엔진배치K(형식번호[형식], raw, N, K)
        assert len(값) % 16 == 0 and len(척) % 16 == 0
        표[e] = (vo // 4, so // 4)
        vo += len(값)
        so += len(척)
        값들.append(값)
        척들.append(척)
        ws.append(w)
    return b"".join(값들), b"".join(척들), 표, ws


def 행렬곱(x, w, N, 자리수):
    치 = np.zeros(N, f32)
    if isinstance(w[0], str):
        return C.행렬곱K블라스(x, w, 치, 자리수=자리수)
    return C.행렬곱블라스(x, w[0], w[1], 치, 자리수=자리수)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    위치 = []
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a in 값깃발:
            i += 2
            continue
        if not a.startswith("--"):
            위치.append(a)
        i += 1
    ptx, 출력 = 위치[:2]
    묶음형식 = 인자값("--형식", "K")
    assert 묶음형식 in ("Q8_0", "K"), 묶음형식
    형식들 = ["Q8_0"] if 묶음형식 == "Q8_0" else ["Q4_K", "Q5_K", "Q6_K"]
    M, E, k = 인자값("--토큰", 300), 인자값("--전문가", 64), 인자값("--고름", 8)
    D, F = 인자값("--너비", 2048), 인자값("--확장", 512)
    자리수 = 1 if "--빠른" in sys.argv else 2
    행합 = "--행합" in sys.argv
    정보줄 = 2 if 행합 else 1
    dr = 시험장치.드라이버()
    mod = dr.모듈(open(ptx, "rb").read())
    fn = {}

    def 띄우기(이름, grid, block, params):
        if 이름 not in fn:
            fn[이름] = dr.함수(mod, 이름)
        ps = [u64(p) if 참조 else i64(p) for 참조, p in params]
        args = (ctypes.c_void_p * len(ps))(*[ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in ps])
        g = dr.cu.cuLaunchKernel(fn[이름], grid[0], grid[1], 1, block, 1, 1, 0, None, args, None)
        assert g == 0, (이름, g)

    def up(b):
        b = b if isinstance(b, bytes) else np.ascontiguousarray(b).tobytes()
        p = dr.할당(len(b))
        dr.확인(dr.cu.cuMemcpyHtoD_v2(p, b, len(b)))
        return p

    def 내려(p, 모양, dt):
        return dr.내리기(p, np.empty(모양, dt))

    def 비우기(p, n):
        dr.확인(dr.cu.cuMemsetD8_v2(p, 0, n))

    r = np.random.default_rng(4112)
    결과 = {"토큰": M, "전문가": E, "고름": k, "너비": D, "확장": F, "자리수": 자리수, "행합": 행합, "경우": []}
    모두 = []

    def 적기(이름, 다, **덧):
        결과["경우"].append(dict(이름=이름, **덧, **{"다른 칸": 다}))
        모두.append(sum(다.values()) == 0)
        print(("다름! " if sum(다.values()) else "") + f"{이름} {덧 if 덧 else ''}: {다}", flush=True)

    # ---- 1. 라우팅 ----
    l = 로짓들(r, M, E)
    참고른 = C.상위k(l[:, :E], k)
    참무게 = C.전문가무게(C.소프트맥스전문가(l[:, :E]), 참고른)
    참게이트 = C.시그모이드(l[:, E])
    p로짓, p고른, p무게 = up(l), dr.할당(M * k * 8), dr.할당(M * k * 4)
    p개수, p순번, p게이트 = dr.할당(E * 8), dr.할당(M * k * 8), dr.할당(M * 4)
    라우팅인자 = [(1, p로짓), (1, p고른), (1, p무게), (1, p개수), (1, p순번), (1, p게이트), (0, M), (0, E), (0, k), (0, E + 1)]
    띄우기("MoE라우팅", ((M + 3) // 4, 1), 128, 라우팅인자)
    dr.맞추기()
    글고른 = 내려(p고른, (M, k), np.int64)
    글순번 = 내려(p순번, (M, k), np.int64)
    표참, n참 = 배치참조(참고른, E)
    순열틀림 = 0
    for e in range(E):
        s = np.sort(글순번[글고른 == e])
        순열틀림 += int(not np.array_equal(s, np.arange(len(s))))
    적기("MoE라우팅", {"고른": 다른칸(글고른, 참고른), "무게": 다른칸(내려(p무게, (M, k), f32), 참무게),
                      "공유 게이트": 다른칸(내려(p게이트, (M,), f32), 참게이트), "개수": 다른칸(내려(p개수, (E,), np.int64), n참),
                      "순번 순열": 순열틀림}, 빈전문가=int((n참 == 0).sum()), 가장많은=int(n참.max()))

    # ---- 2. 배치 ----
    표폭 = 2 * E + 1 + 4 * (E + 1)
    p표 = dr.할당(표폭 * 8)
    띄우기("MoE배치", (1, 1), 32, [(1, p개수), (1, p표), (0, E)])
    dr.맞추기()
    적기("MoE배치", {"배치표": 다른칸(내려(p표, (표폭,), np.int64), 표참), "개수 0": int(np.count_nonzero(내려(p개수, (E,), np.int64)))})
    pp = (표참[:E][참고른] + 글순번).reshape(-1)              # (행, 순위) 평 번호 → 묶음 행(GPU 의 순번으로)
    묶행수 = M * k + 3 * E + 128
    묶폭 = (묶행수 + 63) // 64 * 64
    밖 = np.ones(묶행수, bool)
    밖[pp] = False

    # ---- 3. 모으기 ----
    X = (r.standard_normal((M, D)) * 10.0 ** r.uniform(-1, 1, (M, 1))).astype(f32)
    바이트, M0 = 자리바이트(X, 자리수)
    정보 = 정보판(M0, 자리행합(X, 자리수), (M + 63) // 64 * 64)    # [2 · 블록][폭] — 행 합을 쓰지 않는 모듈은 앞 반만 쓴다
    p자, p정 = up(바이트), up(정보)
    p묶자, p묶정 = dr.할당(자리수 * 묶행수 * D), dr.할당(2 * (D // 32) * 묶폭 * 4)
    모으기인자 = [(1, p자), (1, p정), (1, p고른), (1, p순번), (1, p표), (1, p묶자), (1, p묶정), (0, M), (0, 묶행수), (0, D), (0, k)]
    띄우기("MoE모으기", (M * k, 1), 128, 모으기인자)
    dr.맞추기()
    글묶자 = 내려(p묶자, (자리수, 묶행수, D), np.int8)
    글묶정 = 내려(p묶정, (2 * (D // 32), 묶폭), np.int32)
    tt = np.repeat(np.arange(M), k)
    적기("MoE모으기", {"자리": 다른칸(글묶자[:, pp, :], 바이트[:, tt, :]),
                      "정보": 다른칸(글묶정[:정보줄 * (D // 32), pp], 정보[:정보줄 * (D // 32), tt]),
                      "구간 밖": int(np.count_nonzero(글묶자[:, 밖, :])) + int(np.count_nonzero(글묶정[:, :묶행수][:, 밖]))})

    # ---- 4. 그룹 타일 커널 ----
    gcol = np.concatenate([np.arange(64 * j, 64 * j + 32) for j in range(F // 32)])
    누구 = [np.nonzero(참고른.reshape(-1) == e)[0] for e in range(E)]       # 전문가 e 의 (행, 순위) 평 번호
    Np2F, NpD = (2 * F + 63) // 64 * 64, (D + 63) // 64 * 64
    p치2F, p치D = up(np.zeros(2 * F, f32)), up(np.zeros(D, f32))
    Ys = r.standard_normal((M, D)).astype(f32)                            # 공유 전문가의 출력(MoE합 — 보통 행렬곱의 몫)
    X0 = (r.standard_normal((M, D)) * 3.0).astype(f32)                     # 잔차
    p공, p잔 = up(Ys), up(X0)
    for fi, 형식 in enumerate(형식들):
        K꼴 = 형식 != "Q8_0"
        guV, guS, gu표, gu_w = 전문가행렬(r, 형식, E, 2 * F, D, True)
        dnV, dnS, dn표, dn_w = 전문가행렬(r, 형식, E, D, F, False)
        h참 = np.zeros((M * k, F), f32)
        y참 = np.zeros((M * k, D), f32)
        for e in range(E):
            ix = 누구[e]
            if len(ix):
                ye = 행렬곱(X[ix // k], gu_w[e], 2 * F, 자리수)
                h참[ix] = C.swiglu(ye[:, gcol], ye[:, gcol + 32])
                y참[ix] = 행렬곱(h참[ix], dn_w[e], D, 자리수)
        참자리, 참정보 = 자리바이트(h참, 자리수)
        참T = 자리행합(h참, 자리수)
        묶h자 = np.zeros((자리수, 묶행수, F), np.int8)
        묶h자[:, pp, :] = 참자리
        묶h정 = np.zeros((2 * (F // 32), 묶폭), np.int32)
        묶h정[:F // 32, pp] = 참정보.T
        묶h정[F // 32:, pp] = 참T.T
        pgV, pgS, pgT = up(guV), up(guS), up(gu표)
        pdV, pdS, pdT = up(dnV), up(dnS), up(dn표)
        p묶h자, p묶h정 = up(묶h자), up(묶h정)
        p다자, p다정 = dr.할당(자리수 * 묶행수 * F), dr.할당(2 * (F // 32) * 묶폭 * 4)
        p출 = dr.할당(묶행수 * D * 4)

        def gu띄우기(kn, BM, BN, T, 자, 정):
            뒤 = [(0, 묶행수), (0, D), (0, 2 * F), (0, Np2F), (0, 0), (0, 0)] + ([(0, 2 * F)] if K꼴 else [])
            띄우기(kn, ((M * k + BM - 1) // BM + E, (2 * F + BN - 1) // BN), T,
                   [(1, 자), (1, 정), (1, pgV), (1, pgS), (1, p치2F), (1, p다정), (1, p다자)] + 뒤 + [(1, p표), (1, pgT), (0, E)])

        def down띄우기(kn, BM, BN, T, 자, 정):
            뒤 = [(0, 묶행수), (0, F), (0, D), (0, NpD), (0, 0), (0, 0)] + ([(0, D)] if K꼴 else [])
            띄우기(kn, ((M * k + BM - 1) // BM + E, (D + BN - 1) // BN), T,
                   [(1, 자), (1, 정), (1, pdV), (1, pdS), (1, p치D), (1, p출)] + 뒤 + [(1, p표), (1, pdT), (0, E)])

        for 판, BM, BN, T, NT in 타일판들:
            if NT == 8:
                kn = f"{판}_그룹_SwiGLU정수_{형식}"
                비우기(p다자, 자리수 * 묶행수 * F)
                비우기(p다정, 2 * (F // 32) * 묶폭 * 4)
                gu띄우기(kn, BM, BN, T, p묶자, p묶정)
                dr.맞추기()
                글자 = 내려(p다자, (자리수, 묶행수, F), np.int8)
                글정 = 내려(p다정, (2 * (F // 32), 묶폭), np.int32)
                다 = {"자리": 다른칸(글자[:, pp, :], 참자리), "정보": 다른칸(글정[:F // 32, pp], 참정보.T),
                     "구간 밖": int(np.count_nonzero(글자[:, 밖, :])) + int(np.count_nonzero(글정[:, :묶행수][:, 밖]))}
                if 행합:
                    다["행합"] = 다른칸(글정[F // 32:, pp], 참T.T)
                적기("gate·up", 다, 판=kn)
            kn = f"{판}_그룹_{형식}"
            비우기(p출, 묶행수 * D * 4)
            down띄우기(kn, BM, BN, T, p묶h자, p묶h정)
            dr.맞추기()
            글y = 내려(p출, (묶행수, D), f32)
            적기("down", {"출력": 다른칸(글y[pp], y참), "구간 밖": int(np.count_nonzero(글y[밖]))}, 판=kn)

        # ---- 5. 합 (형식마다 — 참조의 y 로) ----
        묶Y = np.zeros((묶행수, D), f32)
        묶Y[pp] = y참
        ys = np.ascontiguousarray(y참.reshape(M, k, D).transpose(1, 0, 2))
        참x = C.더(X0, C.fma32(np.broadcast_to(참게이트[:, None], (M, D)), Ys, C.전문가합(참무게, ys)))
        p묶Y = up(묶Y)
        dr.올리기(p잔, X0)
        합인자 = [(1, p묶Y), (1, p공), (1, p무게), (1, p게이트), (1, p고른), (1, p순번), (1, p표), (1, p잔), (0, M), (0, D), (0, k), (0, 1)]
        띄우기("MoE합", ((M * D + 255) // 256, 1), 256, 합인자)
        dr.맞추기()
        적기("MoE합", {"x": 다른칸(내려(p잔, (M, D), f32), 참x)}, 형식=형식)

        # ---- 6. 이어서(형식 하나 · 커널 변형 하나): 라우팅 → 배치 → 모으기 → gate·up → down → 합 을 GPU 출력으로 ----
        if fi == 0:
            판, BM, BN, T, NT = 타일판들[0]
            dr.올리기(p잔, X0)
            for 버, n in ((p고른, M * k * 8), (p무게, M * k * 4), (p순번, M * k * 8), (p게이트, M * 4), (p표, 표폭 * 8),
                         (p묶자, 자리수 * 묶행수 * D), (p묶정, 2 * (D // 32) * 묶폭 * 4), (p다자, 자리수 * 묶행수 * F),
                         (p다정, 2 * (F // 32) * 묶폭 * 4), (p출, 묶행수 * D * 4)):
                비우기(버, n)
            띄우기("MoE라우팅", ((M + 3) // 4, 1), 128, 라우팅인자)
            띄우기("MoE배치", (1, 1), 32, [(1, p개수), (1, p표), (0, E)])
            띄우기("MoE모으기", (M * k, 1), 128, 모으기인자)
            gu띄우기(f"{판}_그룹_SwiGLU정수_{형식}", BM, BN, T, p묶자, p묶정)
            down띄우기(f"{판}_그룹_{형식}", BM, BN, T, p다자, p다정)
            띄우기("MoE합", ((M * D + 255) // 256, 1), 256, [(1, p출)] + 합인자[1:])
            dr.맞추기()
            적기("이어서", {"x": 다른칸(내려(p잔, (M, D), f32), 참x), "개수 0": int(np.count_nonzero(내려(p개수, (E,), np.int64)))},
                 형식=형식, 판=판)
            # 다시 돌린 라우팅의 순번(원자 더하기의 차례)은 처음과 다를 수 있다 — 묶음 행 · 모으기 출력을 그것으로(다음 형식이 쓴다)
            글순번 = 내려(p순번, (M, k), np.int64)
            pp = (표참[:E][참고른] + 글순번).reshape(-1)
            밖 = np.ones(묶행수, bool)
            밖[pp] = False
        for 버 in (pgV, pgS, pgT, pdV, pdS, pdT, p묶h자, p묶h정, p다자, p다정, p출, p묶Y):
            dr.확인(dr.cu.cuMemFree_v2(u64(버)))
            dr.할당들.remove(버)

    결과["모두 0"] = all(모두)
    with open(출력, "w", encoding="utf-8") as f:
        json.dump(결과, f, ensure_ascii=False, indent=1)
    print("모두 0:", all(모두), flush=True)


if __name__ == "__main__":
    main()
