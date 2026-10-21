#!/usr/bin/env python3
"""6단계 — 정수 블록 계약의 CPU 참조 구현(numpy). 정수는 정확히(int64), 짧은실수 연산은 IEEE 754 그대로(numpy float32 의 RN),
곱해더하기는 반올림 한 번을 정확히 에뮬레이션한다. GPU 의 커널(3단계/커널생성.py 의 정수로 · 정수타일선형 · 생성메가의 정수 판)과
비트까지 같아야 한다(6a). 계약은 3단계/커널생성.py 의 설명 그대로:

  E = ⌊log₂ max|x|⌋ (최댓값 0 이면 −98, [−98, 100] 로), n = 가까운정수(x · 2^(29 − E)), 균형 256진 네 자리 D₀ … D₃,
  S_j = Σ D_j · q (정확), v_j = S_j · 2^(8j + E − 29) (정확), u = ((v₃ + v₂) + v₁) + v₀, y ← 곱해더하기(u, d, y) (블록 차례),
  출력 = y + 치우침.
8단계 — 자리 둘(자리수=2): n = 가까운정수(x · 2^(13 − E)), |n| ≤ 2¹⁴, D₀ ∈ [−128, 127], D₁ ∈ [−64, 64], v_j = S_j · 2^(8j + E − 13),
u = v₁ + v₀. 일반으로 자리수 m 의 고정소수점 비트는 P = 8m − 3.
31단계 — 짧은실수 연산은 규칙 R(비정규수 버림 — 아래 비정규수버림 의 설명)을 따른다.
"""
import numpy as np

f32 = np.float32

# ── 31단계 — 짧은실수 규칙 R(비정규수 버림 — 문서/31단계.md): 비정규수 입력은 부호를 지킨 0 으로 읽고, 결과는 가수 24 비트로(지수 제한
#    없이) 반올림한 크기가 2⁻¹²⁶ 보다 작으면 부호를 지킨 0. GPU 는 PTX 의 .ftz(모듈 소스의 (* 비정규수버림 *) 표시)가, Vulkan 은 비정규수
#    버림 모드가 이 규칙을 낸다(31d — NVIDIA Vulkan · Intel Vulkan · PTX .ftz 가 경계값까지 모두 이 규칙). numpy 는 IEEE(비정규수 그대로)라
#    짧은실수 연산을 아래 함수(곱 · 더 · 빼기 · 나눔 · 제곱근 · fma32)로 해서 같은 규칙을 낸다. 비정규수버림 = False 면 IEEE 그대로 —
#    표시가 없는 모듈(GPT-2 판 · 14단계의 파이썬판 Qwen2 — 3 ~ 16단계의 기록)과 견줄 때(시험/공통.py 가 끈다).
비정규수버림 = True
막정규 = f32(2.0 ** -126)
_문턱 = 2.0 ** -126 - 2.0 ** -151        # fp64: 정확한 값의 크기가 이보다 작으면 가수 24 비트로(지수 제한 없이) 반올림한 값이 2⁻¹²⁶ 아래


def 버림(x):
    """규칙 R 의 입력 읽기 — 비정규수를 부호를 지킨 0 으로(비정규수버림 이 거짓이면 그대로)."""
    x = np.asarray(x, f32)
    if not 비정규수버림:
        return x
    return np.where(np.abs(x) < 막정규, np.copysign(f32(0), x), x).astype(f32)


def 곱(a, b):
    """짧은실수 a · b (반올림 하나) — 규칙 R. fp64 의 곱은 정확하다(가수 24 + 24 비트)."""
    a, b = np.asarray(a, f32), np.asarray(b, f32)
    if not 비정규수버림:
        return (a * b).astype(f32)
    p = 버림(a).astype(np.float64) * 버림(b).astype(np.float64)
    return np.where(np.abs(p) < _문턱, np.copysign(0.0, p), p).astype(f32)


def 더(a, b):
    """짧은실수 a + b (반올림 하나) — 규칙 R. 정규수(또는 0) 둘의 합이 2⁻¹²⁶ 아래면 늘 정확해서, IEEE 결과의 비정규수를 버리면 된다."""
    a, b = np.asarray(a, f32), np.asarray(b, f32)
    if not 비정규수버림:
        return (a + b).astype(f32)
    return 버림((버림(a) + 버림(b)).astype(f32))


def 빼기(a, b):
    """짧은실수 a − b = 더(a, −b)."""
    return 더(a, -np.asarray(b, f32))


def 나눔(a, b):
    """짧은실수 a / b (바른 반올림) — 규칙 R. IEEE 결과가 비정규수면 0. IEEE 결과가 정확히 2⁻¹²⁶ 인데 정확한 몫이 그보다 작은 것은
    몫이 2⁻¹²⁶(1 − 2⁻²⁴) 일 때뿐이다(짧은실수 둘의 몫은 2⁻¹²⁶(1 − 2⁻²⁵) 과 2⁻¹²⁶ 사이에 오지 않는다) — 지수 제한 없이 반올림하면
    2⁻¹²⁶ 아래라 0."""
    a, b = np.asarray(a, f32), np.asarray(b, f32)
    if not 비정규수버림:
        return (a / b).astype(f32)
    a, b = 버림(a), 버림(b)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        r = (a / b).astype(f32)
    작음 = (np.abs(r) < 막정규) | ((np.abs(r) == 막정규) & (np.abs(a.astype(np.float64)) < 2.0 ** -126 * np.abs(b.astype(np.float64))))
    return np.where(작음, np.copysign(f32(0), r), r).astype(f32)


def 제곱근(x):
    """짧은실수 √x (바른 반올림) — 규칙 R 은 입력만(결과는 비정규수가 될 수 없다)."""
    return np.sqrt(버림(x)).astype(f32)


def fma32(a, b, c):
    """짧은실수 a · b + c 를 반올림 한 번으로(가까운 쪽, 같으면 짝수) — fp64 로 곱하고(정확) 더한 뒤, 그 합이 짧은실수 두 수의
    한가운데에 떨어졌을 때만 오차의 부호(TwoSum)로 바로잡는다. a · b 는 fp64 안에서 정확해야 한다(가수 합 53 비트 이하).
    규칙 R(31단계): 입력의 비정규수는 0, 정확한 값(= r + err)의 크기가 _문턱 보다 작으면 부호를 지킨 0."""
    a, b, c = (버림(v).astype(np.float64) for v in (a, b, c))
    p = a * b
    r = p + c
    bb = r - p
    err = (p - (r - bb)) + (c - bb)                       # p + c = r + err (정확)
    f = r.astype(f32)
    fd = f.astype(np.float64)
    lo = np.where(fd <= r, f, np.nextafter(f, f32(-np.inf)))
    hi = np.where(fd <= r, np.nextafter(f, f32(np.inf)), f)
    mid = (lo.astype(np.float64) + hi.astype(np.float64)) / 2
    tie = (r == mid) & (err != 0)
    g = np.where(tie, np.where(err > 0, hi, lo), f).astype(f32)
    if not 비정규수버림:
        return g
    ar = np.abs(r)
    작음 = (ar < _문턱) | ((ar == _문턱) & np.where(r > 0, err < 0, err > 0))
    return np.where(작음, np.copysign(f32(0), r).astype(f32), g).astype(f32)


def 자리로(x, 자리수=4):
    """x [행][깊이] (짧은실수) → (자리 [자리수][행][깊이] int64, E [행][블록]). 고정소수점 비트 P = 8·자리수 − 3 (넷 29, 둘 13)."""
    x = np.asarray(x, f32)
    M, K = x.shape
    P = 8 * 자리수 - 3
    xb = x.reshape(M, K // 32, 32)
    m = np.abs(xb).max(2)
    e = np.frexp(m.astype(np.float64))[1] - 1                 # m = 가수 · 2^e, 가수 ∈ [0.5, 1) → ⌊log₂ m⌋ = e − 1
    if 자리수 == 1:
        # GeulPUv2 빠른 계약 F1(문서/2단계.md): m · 2^(6 − E) < 127.5 이면 E 를 하나 줄여 최댓값을 64 ~ 127 에 둔다(|n| ≤ 127).
        # m · 2^(6 − e) 는 2 의 거듭제곱을 곱한 것이라 fp64 에서 정확하다. 자르기 [−98, 100] 는 그 뒤.
        e = e - (np.ldexp(m.astype(np.float64), 6 - e) < 127.5).astype(e.dtype)
    E = np.where(m > 0, np.clip(e, -98, 100), -98).astype(np.int64)
    s = np.ldexp(f32(1), (P - E)).astype(f32)
    n = np.rint(xb * s[:, :, None]).astype(np.int64)          # 짧은실수 곱(2 의 거듭제곱 — 정확) 뒤 가까운 정수(같으면 짝수)
    assert np.abs(n).max() <= (127 if 자리수 == 1 else 2 ** (P + 1))
    D = []
    for _ in range(자리수 - 1):
        d = ((n + 128) & 255) - 128
        D.append(d)
        n = (n - d) >> 8
    D.append(n)
    assert np.abs(n).max() <= (127 if 자리수 == 1 else 64)
    return np.stack(D).reshape(자리수, M, K), E


def 행렬곱(x, q, d, 치우침, 자리수=4):
    """정수 블록 계약의 출력 [행][열]. q [깊이][열] 정수(Q8_0 의 q 또는 Q4_0 의 니블 − 8), d [블록][열] (반실수 값), 치우침 [열].
    자리수 4 (6단계) 또는 2 (8단계) — u 는 맨 위 자리부터 차례로 더한다: ((v₃ + v₂) + v₁) + v₀ · v₁ + v₀."""
    D, E = 자리로(x, 자리수)
    P = 8 * 자리수 - 3
    M, K = D.shape[1:]
    KB = K // 32
    N = q.shape[1]
    y = np.zeros((M, N), f32)
    q = np.asarray(q, np.int64)
    d = np.asarray(d, f32)
    for b in range(KB):
        qb = q[32 * b:32 * b + 32]
        v = [np.ldexp((D[j][:, 32 * b:32 * b + 32] @ qb).astype(f32), (8 * j + E[:, b] - P)[:, None]).astype(f32)
             for j in range(자리수)]
        u = v[자리수 - 1]
        for j in range(자리수 - 2, -1, -1):
            u = 더(u, v[j])
        y = fma32(u, np.broadcast_to(d[b][None, :], u.shape), y)
    return 더(y, np.asarray(치우침, f32)[None, :])


# ── 9단계 — 정수 KV 계약: 어텐션의 CPU 참조 (docs/17 §7 "9단계" 의 계약 그대로) ──────────────────────────────────────────
# q · k · v 는 자리 둘 활성값과 같은 바꾸기(자리로(…, 2) — 블록 32 칸, |n| ≤ 2¹⁴). 점수 · 가중합은 자리 짝 넷의 정확한 정수 합을
# 짧은실수로 반올림 하나(numpy 의 int64 → float32 가 RN). 지수는 적힌지수(아래 — 모든 연산이 IEEE 짧은실수라 GPU 와 비트가 같다).
# 상수는 짧은실수로 정확히 표현되는 십진수로 적는다(커널생성.py 의 적힌지수식과 같은 글자).
지수상수 = {"log2e": "1.44269502162933349609375", "C1": "0.693359375", "C2": "0.000212194441701285541058",
         "c5": "0.000198756912141107022762", "c4": "0.001398199936375021934509", "c3": "0.008333452045917510986328",
         "c2": "0.041665796190500259399414", "c1": "0.166666656732559204101562", "c0": "0.5",
         "밑": "-88.0", "끝": "-87.3000030517578125"}
_k = {n: f32(float(v)) for n, v in 지수상수.items()}


def 적힌지수(x):
    """적힌지수(x), x ≤ 0 (짧은실수 배열) — docs/17 9단계 2: n = 가까운정수(max(x, −88) · log₂e) (정확한 곱을 마법수로 반올림), r = x − n·C₁ + n·C₂ (곱해더하기 둘),
    y = ((((c₅·r + c₄)·r + c₃)·r + c₂)·r + c₁)·r + c₀ (곱해더하기 사슬), y = y·r² + r (곱해더하기), y = y + 1, 결과 = y · 2ⁿ (지수 비트에 n 을
    더한다 — 정확), x < −87.3 이면 0. 가까운정수는 같으면 짝수(cvt.rni)."""
    x = np.asarray(x, f32)
    xc = np.maximum(x, _k["밑"]).astype(f32)
    nf = 빼기(fma32(xc, _k["log2e"], f32(12582912.0)), f32(12582912.0))   # 정확한 곱의 가까운 정수(마법수 1.5·2²³)
    n = nf.astype(np.int32)
    r = fma32(nf, -_k["C1"], xc)
    r = fma32(nf, _k["C2"], r)
    z = 곱(r, r)
    y = fma32(_k["c5"], r, _k["c4"])
    for c in ("c3", "c2", "c1", "c0"):
        y = fma32(y, r, _k[c])
    y = fma32(y, z, r)
    y = 더(y, f32(1))
    bits = y.view(np.int32) + (n.astype(np.int32) << 23)
    return np.where(x < _k["끝"], f32(0), bits.view(f32)).astype(f32)


def e자리(ep):
    """e′ [R][32] (≥ 0, 짧은실수) → 블록(키 32 개)마다 E = ⌊log₂ max⌋ (0 이면 −98), n = 가까운정수(e′ · 2^(13 − E)) — 정확한 곱을 마법수
    2²³ + 128 을 더하는 곱해더하기로 반올림(커널과 같은 식; n + 128 이 나온다). 돌려줌: n [R][32] int64, E [R][1]."""
    m = ep.max(1, keepdims=True)
    E = np.where(m > 0, np.clip(np.frexp(m.astype(np.float64))[1] - 1, -98, 100), -98).astype(np.int64)
    s = np.ldexp(f32(1), (13 - E)).astype(f32)
    me = fma32(ep, np.broadcast_to(s, ep.shape), f32(8388736.0)).view(np.int32).astype(np.int64) - 1258291200
    n = me - 128
    assert n.min() >= 0 and n.max() <= 2 ** 14
    return n, E


# GeulPUv2 2b단계 — 빠른 어텐션 F1A(문서/2b단계.md): 이 글에 든 이름(q · k · v)의 바꾸기를 F1 의 자리 하나(자리로(x, 1))로 하고, 그 값
# n·2^(E−5) 를 자리 둘의 (128·n)·2^((E+1)−13) 로 둔다 — 같은 수이고 |128·n| ≤ 16,256 < 2¹⁴ 라 자리 둘 계약의 식 · 차례가 그대로 선다
# (에뮬레이션 정리). "" 이면 9 · 22단계의 계약 그대로. 커널생성 --어텐션한자리 KV 는 "kv", QKV 는 "qkv".
# GeulPUv2 ③b-1 — "kv8" · "v8" 은 8비트 KV(F1A′)의 에뮬레이션: 키 · 값(v8 은 값만)을 팔비트(아래)로 바꿨다가 되돌린 값을 자리 둘로(문서/긴문맥.md).
어텐션한자리 = ""
# GeulPUv2 ③b-2 — 8비트 KV 캐시(계약 F1A′, 계약.md 2.1): 참이면 어텐션의 키 · 값을 팔비트KV(아래)의 (n, s)로 두고, fp32 스케일 s 를
# 2^(E − 13) 자리에 쓴다. 커널생성 --KV8.
KV8 = False


def 팔비트(x):
    """GeulPUv2 ③b-1 — 8비트 KV(F1A′)의 에뮬레이션. x [행][깊이] fp32 → 블록(값 32 개)마다 m = max|x|, s = m / 127(반올림 하나), s > 0 이면
    y = 가까운정수(x / s) · s(|가까운정수| ≤ 127), 아니면 0 — 모두 규칙 R. 커널(생성머리128.gl 의 큰팔비트)과 같은 식."""
    x = 버림(np.asarray(x, f32))
    M, K = x.shape
    xb = x.reshape(M, K // 32, 32)
    s = np.broadcast_to(나눔(np.abs(xb).max(2, keepdims=True), f32(127.0)), xb.shape)
    양 = s > 0
    n = np.where(양, np.rint(나눔(xb, np.where(양, s, f32(1.0)))), f32(0.0)).astype(f32)
    assert np.abs(n).max() <= 127
    return np.where(양, 곱(n, s), f32(0.0)).astype(f32).reshape(M, K)


def 팔비트KV(x):
    """GeulPUv2 ③b-2 — 8비트 KV 캐시의 저장(계약 F1A′). x [행][깊이] fp32 → (n [행][깊이] int64, s [행][깊이/32] fp32): 블록(값 32 개)마다
    s = m / 127(m = max|x|, 반올림 하나), s > 0 이면 n = 가까운정수(x / s)(|n| ≤ 127), 아니면 0 — 모두 규칙 R. n · s 는 팔비트 와 같은 값."""
    x = 버림(np.asarray(x, f32))
    M, K = x.shape
    xb = x.reshape(M, K // 32, 32)
    s = 나눔(np.abs(xb).max(2), f32(127.0))
    양 = s > 0
    n = np.where(양[:, :, None], np.rint(나눔(xb, np.where(양, s, f32(1.0))[:, :, None])), f32(0.0)).astype(np.int64)
    assert np.abs(n).max() <= 127
    return n.reshape(M, K), s


def 어텐션자리로(x, 이름):
    """어텐션의 q · k · v(이름) 바꾸기 — (자리 [2][행][깊이] int64, E [행][블록]). 이름이 어텐션한자리 에 있으면 F1A(위) · 8비트 에뮬레이션(③b-1)."""
    if 이름 not in 어텐션한자리:
        return 자리로(x, 2)
    if "8" in 어텐션한자리:
        return 자리로(팔비트(x), 2)
    D1, E1 = 자리로(x, 1)
    n = D1[0] * 128
    d0 = ((n + 128) & 255) - 128
    return np.stack([d0, (n - d0) >> 8]), E1 + 1


def 거듭제곱(E):
    """2^(E − 13) 을 짧은실수로(정확 — 비트 (E + 114) << 23)."""
    return ((np.asarray(E, np.int64) + 114) << 23).astype(np.int32).view(f32)


def 점수눈금(hd):
    """점수의 눈금 — 머리 64 는 0.125(정확), 머리 128 은 짧은실수(1/√128)(가까운 반올림 — 22단계 계약). 점수 합 뒤에 곱 한 번."""
    return f32(0.125) if hd == 64 else f32(1.0 / np.sqrt(np.float64(hd)))


def _조각몫(nq, nk, nv, Pq, Pk, Pv, Rr, 시작, n):
    """행 Rr(위치 — 모두 시작 이상)의 키 조각 [시작, 시작 + 64) 몫 (m, l, o) — 9단계 계약의 1~4(점수 · 최댓값 · 적힌지수 · 합 · e′ 자리 ·
    가중합). 헤드 크기 hd = nq 의 열 수(64 · 128 — 블록 hd/32 개를 b 오름차순으로, 22단계). 8비트 KV(③b-2 — 계약 F1A′)면 nk · nv 가 8비트
    정수이고 Pk · Pv 가 fp32 스케일이라, 그 곱(점수의 곱 · FMA, e′)마다 반올림 하나 — 식과 차례는 같다."""
    hd = nq.shape[1]
    keys = np.arange(시작, min(시작 + 64, n))
    valid = keys[None, :] <= Rr[:, None]                         # [R][keys]
    # 1. 점수
    s = None
    for b in range(hd // 32):
        sl = slice(32 * b, 32 * b + 32)
        u = (nq[Rr][:, sl] @ nk[keys][:, sl].T).astype(f32)        # 정확한 정수 합 → 반올림 하나
        부분 = 곱(u, Pq[Rr, b][:, None])                            # 정확
        s = 곱(부분, Pk[keys, b][None, :]) if b == 0 else fma32(부분, np.broadcast_to(Pk[keys, b][None, :], 부분.shape), s)
    s = 곱(s, 점수눈금(hd))
    s = np.where(valid, s, f32(-3.0e38)).astype(f32)
    # 2. 최댓값 · 지수
    m = s.max(1)
    e = np.where(valid, 적힌지수(빼기(s, m[:, None])), f32(0)).astype(f32)
    # 3. 합 — 2⁻²³ 눈금의 정수 합, 반올림 하나
    l = (np.rint((e * f32(8388608.0)).astype(f32)).astype(np.int64).sum(1).astype(f32) * f32(2.0 ** -23)).astype(f32)
    # 4. 가중합 — 값의 블록 b 마다 e′ = e · 2^(E_v − 13), 키 32 개 블록 j 마다 자리 둘
    o = np.zeros((len(Rr), hd), f32)
    for b in range(hd // 32):
        ep = 곱(e, Pv[keys, b][None, :])                            # 정확(2⁻¹²⁶ 아래로 작아지면 규칙 R 로 0) — 8비트 KV 는 반올림 하나
        uj = []
        for j in range(0, len(keys), 32):
            epj = np.zeros((len(Rr), 32), f32)
            epj[:, :min(32, len(keys) - j)] = ep[:, j:j + 32]
            ne, Ee = e자리(epj)                                     # [R][32] ∈ [0, 2¹⁴], [R][1]
            nvj = np.zeros((32, 32), np.int64)
            nvj[:min(32, len(keys) - j)] = nv[keys[j:j + 32]][:, 32 * b:32 * b + 32]
            uj.append(((ne @ nvj).astype(f32), 거듭제곱(Ee[:, 0])))
        if len(uj) == 1:
            uj.append((np.zeros_like(uj[0][0]), 거듭제곱(np.full(len(Rr), -98))))
        (u0, P0), (u1, P1) = uj
        o[:, 32 * b:32 * b + 32] = fma32(u1, P1[:, None], 곱(u0, P0[:, None]))
    return m, l, o


def _접기(M, L, O, m, l, o):
    """접기 한 걸음 — M′ = max(M, m), 가 = 적힌지수(M − M′), 나 = 적힌지수(m − M′), L′ = 곱해더하기(l, 나, L·가), O′ = 곱해더하기(o, 나, O·가)."""
    M2 = np.maximum(M, m)
    가 = 적힌지수(빼기(M, M2))
    나 = 적힌지수(빼기(m, M2))
    return M2, fma32(l, 나, 곱(L, 가)), fma32(o, 나[:, None], 곱(O, 가[:, None]))


def 어텐션(q, k, v, 조각=64, 묶음=16, 첫행=0):
    """머리 하나의 어텐션(원인 가림, × 점수눈금(hd) — 머리 64 는 0.125) — 정수 KV 계약의 참조. q · k · v [n][hd] 짧은실수(hd 64 · 128) →
    출력 [n − 첫행][hd] 짧은실수(O / L).
    프롬프트 판(mma)과 생성 판(dp4a) 모두와 비트까지 같아야 한다(9a).
    접기(13단계 계약): 키 조각(위치 64)을 묶음(조각 `묶음` 개 = 위치 1024 — 묶음 g 는 위치 [1024g, 1024g + 1024))으로 나눠, 묶음 안에서
    조각을 차례로 접어 묶음의 (M_g, L_g, O_g) 를 만들고(묶음의 첫 조각이 처음 값), 묶음들을 차례로 같은 식으로 접는다(첫 묶음이 처음 값).
    위치 1024 까지는 묶음이 하나라 9~12단계의 한 층 접기와 같다. 묶음=None 이면 한 층(9~12단계 계약 — 견주기용).
    첫행: 이 행부터만 계산한다(긴 문맥의 짧은 덩이 대조 — 앞 행은 건너뛴다; 값은 모든 행을 계산할 때의 그 행과 같다)."""
    if F2:                                                          # GeulPUv2 빠른 어텐션 F2(시험용)
        return 어텐션F2(q, k, v, 조각, 묶음, 첫행)
    q, k, v = (np.asarray(x, f32) for x in (q, k, v))
    n = q.shape[0]
    (Dq, Eq), (Dk, Ek), (Dv, Ev) = 어텐션자리로(q, "q"), 어텐션자리로(k, "k"), 어텐션자리로(v, "v")   # 2b단계 — F1A 면 에뮬레이션
    nq, nk, nv = (D[1] * 256 + D[0] for D in (Dq, Dk, Dv))          # [n][hd] int64
    Pq, Pk, Pv = 거듭제곱(Eq), 거듭제곱(Ek), 거듭제곱(Ev)              # [n][hd/32]
    if KV8:                                                          # ③b-2 — 키 · 값은 8비트 정수, fp32 스케일이 2^(E − 13) 자리에(계약 F1A′)
        (nk, Pk), (nv, Pv) = 팔비트KV(k), 팔비트KV(v)
    hd = q.shape[1]
    R_ = np.arange(첫행, n)
    M = np.zeros(len(R_), f32)
    L = np.zeros(len(R_), f32)
    O = np.zeros((len(R_), hd), f32)
    폭 = 조각 * 묶음 if 묶음 else n + 조각                              # 묶음 하나의 위치 수
    for g0 in range(0, n, 폭):
        selg = R_ >= g0                                              # 이 묶음에 유효한 키가 있는 행
        Mg = np.zeros(int(selg.sum()), f32)
        Lg = np.zeros(int(selg.sum()), f32)
        Og = np.zeros((int(selg.sum()), hd), f32)
        for 시작 in range(g0, min(g0 + 폭, n), 조각):
            sel = R_ >= 시작
            m, l, o = _조각몫(nq, nk, nv, Pq, Pk, Pv, R_[sel], 시작, n)
            idx = np.nonzero(sel[selg])[0]
            if 시작 == g0:                                          # 묶음의 첫 조각 — 처음 값
                Mg[idx], Lg[idx], Og[idx] = m, l, o
            else:
                Mg[idx], Lg[idx], Og[idx] = _접기(Mg[idx], Lg[idx], Og[idx], m, l, o)
        if g0 == 0:                                                  # 첫 묶음 — 처음 값
            M[selg], L[selg], O[selg] = Mg, Lg, Og
        else:
            M[selg], L[selg], O[selg] = _접기(M[selg], L[selg], O[selg], Mg, Lg, Og)
    return 나눔(O, L[:, None])


# ── GeulPUv2 빠른 어텐션 F2(문서/빠른어텐션.md — F2-1 의 시험용 계약, 아직 계약.md 에 없다): 커널생성 --f2proto 와 같은 비트 —
#    1 V128(값 스케일을 헤드의 값 128 개마다 하나), 2 P(확률을 타일마다 한 지수), 4 P7(확률 7비트 — F1 활성값의 규칙, 없으면 16비트),
#    8 X2(밑이 2 인 지수 함수, 점수 눈금 c · log₂e), 16 Q1E(q 의 지수를 헤드마다 하나). 지금은 27 · 31 만(KV8 과 함께) ──────────────
F2 = 0
지수2상수 = ("0.001333355787210166454315185546875", "0.009618128649890422821044921875", "0.0555041097104549407958984375",
            "0.2402265071868896484375", "0.693147182464599609375")
_k2 = [f32(float(v)) for v in 지수2상수]
점수눈금2 = f32(0.1275174319744110107421875)                # fp32(fp32(1/√128) · fp32(log₂e)) — 0x3E0293EE


def 지수2(x):
    """F2 X2 — 2^x (x ≤ 0): xc = max(x, −126), t = xc + 1.5·2²³(가까운 정수로 반올림), r = xc − (t − 1.5·2²³)(정확 — r ∈ [−½, ½]),
    5차 다항식(곱해더하기 체인 — 지수2상수, 끝은 + 1), 결과의 비트 = p 의 비트 + (t 의 비트 ≪ 23)(2³² 의 나머지 — 지수에 n 을 더한다)."""
    x = np.asarray(x, f32)
    xc = np.maximum(x, f32(-126.0)).astype(f32)
    t = 더(xc, f32(12582912.0))
    r = 빼기(xc, 빼기(t, f32(12582912.0)))
    p = fma32(_k2[0], r, _k2[1])
    for c in _k2[2:]:
        p = fma32(p, r, c)
    p = fma32(p, r, f32(1.0))
    bits = (p.view(np.int32).astype(np.int64) + (t.view(np.int32).astype(np.int64) << 23)) & 0xFFFFFFFF
    return bits.astype(np.uint32).view(f32)


def 자리로머리(x, 자리수=2):
    """F2 Q1E — 자리로 와 같은 식이고 지수는 행마다 하나(값 hd 개의 최댓값). (자리 [자리수][행][hd] int64, E [행] int64)."""
    x = np.asarray(x, f32)
    P = 8 * 자리수 - 3
    m = np.abs(x).max(1)
    e = np.frexp(m.astype(np.float64))[1] - 1
    E = np.where(m > 0, np.clip(e, -98, 100), -98).astype(np.int64)
    s = np.ldexp(f32(1), (P - E)).astype(f32)
    n = np.rint(x * s[:, None]).astype(np.int64)
    assert np.abs(n).max() <= 2 ** (P + 1)
    D = []
    for _ in range(자리수 - 1):
        d = ((n + 128) & 255) - 128
        D.append(d)
        n = (n - d) >> 8
    D.append(n)
    return np.stack(D), E


def 팔비트KV머리(x):
    """F2 V128 — 팔비트KV 의 식을 값 hd 개(헤드 하나)마다 스케일 하나로. (n [행][hd] int64, s [행] fp32)."""
    x = 버림(np.asarray(x, f32))
    s = 나눔(np.abs(x).max(1), f32(127.0))
    양 = s > 0
    n = np.where(양[:, None], np.rint(나눔(x, np.where(양, s, f32(1.0))[:, None])), f32(0.0)).astype(np.int64)
    assert np.abs(n).max() <= 127
    return n, s


def _지수P(m, 일곱):
    """F2 P — 행마다 타일의 e′ 최댓값 m → (E, s). 일곱(P7): F1 규칙(⌊log₂ m⌋, m · 2^(6 − E) < 127.5 면 E − 1, 최댓값이 64 ~ 127),
    s = 2^(5 − E). 아니면 자리 둘의 규칙 s = 2^(13 − E). 자르기 [−98, 100], m = 0 이면 −98."""
    m64 = m.astype(np.float64)
    e = np.frexp(m64)[1] - 1
    if 일곱:
        e = e - (np.ldexp(m64, 6 - e) < 127.5).astype(e.dtype)
    E = np.where(m > 0, np.clip(e, -98, 100), -98).astype(np.int64)
    return E, np.ldexp(f32(1), ((5 if 일곱 else 13) - E)).astype(f32)


def _조각몫F2(nq, nk, nv, Eq, Pk, sv, Rr, 시작, n):
    """F2 — 행 Rr 의 키 타일 [시작, 시작 + 64) 몫 (m, l, U, Pe): 점수 z = 블록 넷의 체인(첫 블록은 곱, 그다음 FMA — u_b 는 정확한 정수 합의
    fp32), s = z · qc(qc = fp32(2^(E_q − 13) · c′)), 최댓값 m, e = 지수2(s − m)(가린 키는 0), l 은 F1A′ 와 같은 정수 합, e′ = e · s_v(키마다
    하나), 행마다 e′ 최댓값의 지수 E 로 n = 가까운정수(e′ · 2^(P − E)), U_d = fp32(Σ_t n_t · v_t,d)(P7 은 |·| < 2²⁰ 이라 정확), Pe = 2^(E − P)."""
    hd = nq.shape[1]
    keys = np.arange(시작, min(시작 + 64, n))
    valid = keys[None, :] <= Rr[:, None]
    z = None
    for b in range(hd // 32):
        sl = slice(32 * b, 32 * b + 32)
        u = (nq[Rr][:, sl] @ nk[keys][:, sl].T).astype(f32)
        kb = np.broadcast_to(Pk[keys, b][None, :], u.shape)
        z = 곱(u, kb) if b == 0 else fma32(u, kb, z)
    qc = 곱(거듭제곱(Eq[Rr]), 점수눈금2)
    s = 곱(z, qc[:, None])
    s = np.where(valid, s, f32(-3.0e38)).astype(f32)
    m = s.max(1)
    e = np.where(valid, 지수2(빼기(s, m[:, None])), f32(0)).astype(f32)
    l = (np.rint((e * f32(8388608.0)).astype(f32)).astype(np.int64).sum(1).astype(f32) * f32(2.0 ** -23)).astype(f32)
    ep = 곱(e, sv[keys][None, :])
    일곱 = bool(F2 & 4)
    E, sP = _지수P(ep.max(1), 일곱)
    ne = np.rint(ep.astype(np.float64) * sP.astype(np.float64)[:, None]).astype(np.int64)
    assert ne.min() >= 0 and ne.max() <= (127 if 일곱 else 2 ** 14)
    U = (ne @ nv[keys]).astype(f32)
    return m, l, U, (거듭제곱(E + 8) if 일곱 else 거듭제곱(E))


def _접기2(M, L, O, m, l, o):
    """F2 — 그룹 병합(두 층의 둘째): _접기 와 같은 식, 지수는 지수2."""
    M2 = np.maximum(M, m)
    가 = 지수2(빼기(M, M2))
    나 = 지수2(빼기(m, M2))
    return M2, fma32(l, 나, 곱(L, 가)), fma32(o, 나[:, None], 곱(O, 가[:, None]))


def 어텐션F2(q, k, v, 조각=64, 묶음=16, 첫행=0):
    """F2 의 어텐션(머리 하나 — 머리 128, 8비트 KV) — 어텐션 과 같은 꼴. 그룹 안의 타일 병합은 커널의 식 그대로:
    첫 타일 O = fma(U, Pe, +0), 그다음 M′ = max(M, m), 가 = 지수2(M − M′), 나 = 지수2(m − M′), L′ = fma(l, 나, L · 가),
    O′ = fma(U, fp32(Pe · 나), O · 가). 그룹 사이는 _접기2. 출력 O / L."""
    assert F2 in (27, 31), F2
    q, k, v = (np.asarray(x, f32) for x in (q, k, v))
    n, hd = q.shape
    Dq, Eq = 자리로머리(q, 2)
    nq = Dq[1] * 256 + Dq[0]
    nk, Pk = 팔비트KV(k)
    nv, sv = 팔비트KV머리(v)
    R_ = np.arange(첫행, n)
    M = np.zeros(len(R_), f32)
    L = np.zeros(len(R_), f32)
    O = np.zeros((len(R_), hd), f32)
    폭 = 조각 * 묶음
    for g0 in range(0, n, 폭):
        selg = R_ >= g0
        Mg = np.zeros(int(selg.sum()), f32)
        Lg = np.zeros(int(selg.sum()), f32)
        Og = np.zeros((int(selg.sum()), hd), f32)
        for 시작 in range(g0, min(g0 + 폭, n), 조각):
            sel = R_ >= 시작
            m, l, U, Pe = _조각몫F2(nq, nk, nv, Eq, Pk, sv, R_[sel], 시작, n)
            idx = np.nonzero(sel[selg])[0]
            if 시작 == g0:
                Mg[idx], Lg[idx], Og[idx] = m, l, fma32(U, Pe[:, None], f32(0))
            else:
                M2 = np.maximum(Mg[idx], m)
                가 = 지수2(빼기(Mg[idx], M2))
                나 = 지수2(빼기(m, M2))
                Lg[idx] = fma32(l, 나, 곱(Lg[idx], 가))
                Og[idx] = fma32(U, 곱(Pe, 나)[:, None], 곱(Og[idx], 가[:, None]))
                Mg[idx] = M2
        if g0 == 0:
            M[selg], L[selg], O[selg] = Mg, Lg, Og
        else:
            M[selg], L[selg], O[selg] = _접기2(M[selg], L[selg], O[selg], Mg, Lg, Og)
    return 나눔(O, L[:, None])


def 낱말차례(바이트):
    """[…][32] 바이트(블록 하나) → [0 4 1 5 2 6 3 7] 낱말 차례의 [...][32] 바이트 (정수로 · 타일 판의 A 조각 배치)."""
    b = np.asarray(바이트).reshape(*np.asarray(바이트).shape[:-1], 8, 4)
    return b[..., [0, 4, 1, 5, 2, 6, 3, 7], :].reshape(np.asarray(바이트).shape)


def 키캐시(k, 최대길이):
    """머리 하나의 키 캐시 바이트 — [자리 2][최대길이][hd 바이트](위치 n 뒤는 0) 와 지수 [최대길이][hd/32] (E + 114, uint8). k [n][hd]
    (hd 64 · 128 — 22단계)."""
    D, E = 어텐션자리로(np.asarray(k, f32), "k")
    n, hd = D.shape[1], D.shape[2]
    out = np.zeros((2, 최대길이, hd), np.int8)
    out[:, :n] = 낱말차례(D.reshape(2, n, hd // 32, 32)).reshape(2, n, hd)
    ex = np.zeros((최대길이, hd // 32), np.uint8)
    ex[:n] = E + 114
    return out, ex


def 값캐시(v, 최대길이):
    """머리 하나의 값 캐시 바이트 — [키 블록 최대길이/32][자리 2][d hd][낱말 8][4 바이트]: 낱말 2c + h 의 바이트 i 는 키 16h + 2c + i (i < 2),
    16h + 8 + 2c + (i − 2) (i ≥ 2) 의 d 번째 자리. 지수 [최대길이][hd/32] (int8). v [n][hd] (hd 64 · 128 — 22단계)."""
    D, E = 어텐션자리로(np.asarray(v, f32), "v")
    n, hd = D.shape[1], D.shape[2]
    키 = np.zeros(32, np.int64)
    for h in range(2):
        for c in range(4):
            키[(2 * c + h) * 4:(2 * c + h) * 4 + 4] = [16 * h + 2 * c, 16 * h + 2 * c + 1, 16 * h + 8 + 2 * c, 16 * h + 8 + 2 * c + 1]
    Dp = np.zeros((2, 최대길이, hd), np.int64)
    Dp[:, :n] = D
    blk = Dp.reshape(2, 최대길이 // 32, 32, hd)[:, :, 키, :]                  # [자리][블록][자리 32 (낱말 차례)][d]
    out = np.ascontiguousarray(blk.transpose(1, 0, 3, 2)).astype(np.int8)    # [블록][자리][d][32 바이트]
    ex = np.zeros((최대길이, hd // 32), np.uint8)
    ex[:n] = E + 114
    return out, ex


def 키캐시KV8(k, 최대길이):
    """GeulPUv2 ③b-2 — 8비트 KV(계약 F1A′)의 키 캐시(헤드 하나): 바이트 [최대길이][hd](블록마다 32비트 워드 차례 — 키캐시 의 자리 하나, 위치
    n 뒤는 0)와 스케일 [최대길이][hd/32] fp32. k [n][hd]."""
    n8, s = 팔비트KV(np.asarray(k, f32))
    n, hd = n8.shape
    out = np.zeros((최대길이, hd), np.int8)
    out[:n] = 낱말차례(n8.reshape(n, hd // 32, 32)).reshape(n, hd)
    sc = np.zeros((최대길이, hd // 32), f32)
    sc[:n] = s
    return out, sc


def 값캐시KV8(v, 최대길이):
    """GeulPUv2 ③b-2 — 8비트 KV 의 값 캐시(헤드 하나): 바이트 [키 블록 최대길이/32][d hd][32 바이트](키의 차례는 값캐시 와 같다 — 그 자리
    하나)와 스케일 [최대길이][hd/32] fp32. v [n][hd]."""
    n8, s = 팔비트KV(np.asarray(v, f32))
    n, hd = n8.shape
    키 = np.zeros(32, np.int64)
    for h in range(2):
        for c in range(4):
            키[(2 * c + h) * 4:(2 * c + h) * 4 + 4] = [16 * h + 2 * c, 16 * h + 2 * c + 1, 16 * h + 8 + 2 * c, 16 * h + 8 + 2 * c + 1]
    Dp = np.zeros((최대길이, hd), np.int64)
    Dp[:n] = n8
    blk = Dp.reshape(최대길이 // 32, 32, hd)[:, 키, :]                         # [블록][키 32 (낱말 차례)][d]
    out = np.ascontiguousarray(blk.transpose(0, 2, 1)).astype(np.int8)      # [블록][d][32 바이트]
    sc = np.zeros((최대길이, hd // 32), f32)
    sc[:n] = s
    return out, sc


# ── 14단계 — Llama 계열 블록(Qwen2): RMSNorm · RoPE · SwiGLU · GQA (문서/14단계.md 의 계약) ──────────────────────────────────

def rms정규화(x, w, eps=1e-6):
    """RMSNorm (행마다, 너비 D) — 가닥 t (0 … 255)의 부분합 p_t = Σ_{c ≡ t (mod 256), c 오름차순} x_c · x_c (곱 · 합마다 반올림), p 를 128 부터
    반씩 접어(p_t ← p_t + p_{t+접기}) 합 S, r = 1 / 제곱근(S / D + eps) (IEEE 가까운 반올림, eps 는 짧은실수), y_c = (x_c · r) · w_c (반올림 둘).
    x [행][D] → y [행][D] (짧은실수)."""
    x = np.asarray(x, f32)
    R, D = x.shape
    p = np.zeros((R, 256), f32)
    with np.errstate(over="ignore"):                               # 넘치면 IEEE 대로 무한(GPU 와 같다)
        for c0 in range(0, D, 256):
            b = x[:, c0:c0 + 256]
            n = b.shape[1]
            p[:, :n] = 더(p[:, :n], 곱(b, b))
    접기 = 128
    while 접기 > 0:
        p[:, :접기] = 더(p[:, :접기], p[:, 접기:2 * 접기])
        접기 //= 2
    평균 = 나눔(p[:, 0], f32(D))
    r = 나눔(f32(1), 제곱근(더(평균, f32(eps))))
    return 곱(곱(x, r[:, None]), np.asarray(w, f32)[None, :])


def _십진파이():
    """π 를 지금 십진 문맥의 자리로(파이썬 decimal 문서의 recipe — 급수가 더 바뀌지 않을 때까지)."""
    import decimal
    c = decimal.getcontext()
    c.prec += 2
    three = decimal.Decimal(3)
    lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
    while s != lasts:
        lasts = s
        n, na = n + na, na + 8
        d, da = d + da, da + 32
        t = (t * n) / d
        s += t
    c.prec -= 2
    return +s


def _십진cos_sin(x):
    """(cos x, sin x) 를 지금 십진 문맥의 자리로 — x 를 2π 로 줄인 뒤 테일러 급수(decimal 문서의 recipe)."""
    import decimal
    c = decimal.getcontext()
    c.prec += 10
    두파이 = 2 * _십진파이()
    r = x - 두파이 * (x / 두파이).to_integral_value(rounding=decimal.ROUND_HALF_EVEN)

    def 급수(첫, 처음):
        i, lasts, s, fact, num, sign = 첫, 0, 처음, 1, 처음, 1
        while s != lasts:
            lasts = s
            i += 2
            fact *= i * (i - 1)
            num *= r * r
            sign *= -1
            s += num / fact * sign
        return s
    co, si = 급수(0, decimal.Decimal(1)), 급수(1, r)
    c.prec -= 10
    return +co, +si


def _바른반올림32(값, 정밀식):
    """fp64 값들(배열)을 짧은실수로 — 값이 짧은실수 두 수의 한가운데에 아주 가까우면(fp64 의 오차로 반올림이 갈릴 수 있으면) 정밀식(i) 으로
    다시 구해(파이썬 decimal 80 자리) 바르게 반올림한다. 그래서 결과는 참값의 바른 반올림(libm 의 fp64 오차와 상관없다)."""
    import decimal
    v = np.asarray(값, np.float64)
    f = v.astype(f32)
    fd = f.astype(np.float64)
    아래 = np.where(fd <= v, f, np.nextafter(f, f32(-np.inf))).astype(np.float64)
    위 = np.where(fd <= v, np.nextafter(f, f32(np.inf)), f).astype(np.float64)
    가운데 = (아래 + 위) / 2
    위험 = np.abs(v - 가운데) <= 8 * np.spacing(np.abs(v))
    if 위험.any():
        out = f.reshape(-1).copy()
        lo32, hi32 = 아래.reshape(-1).astype(f32), 위.reshape(-1).astype(f32)
        with decimal.localcontext() as c:
            c.prec = 80
            for i in np.flatnonzero(위험.reshape(-1)):
                x = 정밀식(int(i))
                m = (decimal.Decimal(float(lo32[i])) + decimal.Decimal(float(hi32[i]))) / 2
                if x == m:                                       # 정확히 한가운데 — 가수의 끝 비트가 0 인 쪽
                    out[i] = lo32[i] if int(lo32[i:i + 1].view(np.int32)[0]) % 2 == 0 else hi32[i]
                else:
                    out[i] = lo32[i] if x < m else hi32[i]
        f = out.reshape(f.shape)
    return f


def 얀인자(hd, theta, 배, 원래=32768):
    """GeulPUv2 ③a — YaRN(계약 4.8): 짝 i 의 주파수 곱 g [hd/2] (fp64)와 표의 크기 m (fp64).
    경계(β) = hd · ln(L / (β · 2π)) / (2 · ln θ) — 아래 = ⌊경계(32)⌋, 위 = ⌈경계(1)⌉ 를 [0, hd − 1] 로(같으면 위 += 0.001),
    r_i = clamp((i − 아래) / (위 − 아래), 0, 1) (fp64 나누기), g_i = r_i / s + (1 − r_i) (fp64 연산 셋 — 이 차례),
    m = 0.1 · ln s + 1 (ln s 는 fp64 로 바르게 — decimal 60 자리, 곱 · 더하기는 fp64). 경계가 정수에 너무 가까우면 멈춘다."""
    import decimal
    import math
    with decimal.localcontext() as c:
        c.prec = 60
        두파이 = 2 * _십진파이()
        로그밑 = decimal.Decimal(repr(float(theta))).ln()

        def 경계(바퀴):
            return decimal.Decimal(hd) * (decimal.Decimal(원래) / (decimal.Decimal(바퀴) * 두파이)).ln() / (2 * 로그밑)
        e아래, e위 = 경계(32), 경계(1)
        assert abs(e아래 - e아래.to_integral_value()) > decimal.Decimal("1e-6") and abs(e위 - e위.to_integral_value()) > decimal.Decimal("1e-6"), \
            "YaRN 경계가 정수에 너무 가깝다"
        아래 = float(math.floor(e아래))
        위 = float(math.ceil(e위))
        lns = float(decimal.Decimal(배).ln())
    아래 = max(아래, 0.0)
    위 = min(위, float(hd - 1))
    if 위 == 아래:
        위 += 0.001
    s = float(배)
    g = np.empty(hd // 2, np.float64)
    for i in range(hd // 2):
        r = (float(i) - 아래) / (위 - 아래)
        r = min(max(r, 0.0), 1.0)
        g[i] = r / s + (1.0 - r)
    m = 0.1 * lns + 1.0
    return g, m


def rope표(최대길이, theta=1e6, hd=64, 인자=None, 얀=None):
    """RoPE 의 cos · sin 표 [위치][hd/2] (짧은실수) — θ_i = θ^(−2i/hd) 를 fp64 로 바르게 반올림(decimal 60 자리의 거듭제곱 → fp64), 각 = 위치 · θ_i
    (fp64 곱 — IEEE 반올림 하나), cos(각) · sin(각) 을 짧은실수로 바르게 반올림(정확한 값의 반올림 하나 — 플랫폼의 수학 함수와 상관없다).
    표준 라이브러리(decimal)만 쓴다 — 14단계의 처음 판(mpmath 50 자리)과 같은 표임을 위치 32768 까지 확인했다.
    39단계 — 인자(llama 의 `rope_freqs` — 짧은실수 hd/2 칸, 계약 4.7): θ′_i = θ_i / f_i(fp64 나눗셈 — IEEE 반올림 하나) 로 바꾼 뒤 그대로.
    GeulPUv2 ③a — 얀 = (배, 원래 길이)(계약 4.8): θ′_i = θ_i · g_i (fp64 곱 하나), 표 = 짧은실수(m · cos(각)) · 짧은실수(m · sin(각)) 를 바르게
    (m · cos 의 정확한 곱의 반올림 하나 — 경계에 가까우면 decimal 80 자리의 m · cos)."""
    import decimal
    with decimal.localcontext() as c:
        c.prec = 60
        inv = np.array([float(decimal.Decimal(repr(float(theta))) ** (decimal.Decimal(-2 * k) / decimal.Decimal(hd))) for k in range(hd // 2)],
                       np.float64)
    if 인자 is not None:
        f = np.asarray(인자, np.float32).astype(np.float64)
        assert f.shape == inv.shape and (f > 0).all(), "RoPE 늘이기 인자는 hd/2 칸의 양수여야 한다"
        inv = inv / f
    m = None
    if 얀 is not None:
        g, m = 얀인자(hd, theta, 얀[0], 얀[1])
        inv = inv * g
    각 = np.arange(최대길이, dtype=np.float64)[:, None] * inv[None, :]
    평 = 각.reshape(-1)
    if m is None:
        c_ = _바른반올림32(np.cos(각), lambda j: _십진cos_sin(decimal.Decimal(float(평[j])))[0])
        s_ = _바른반올림32(np.sin(각), lambda j: _십진cos_sin(decimal.Decimal(float(평[j])))[1])
        return c_, s_
    md = decimal.Decimal(m)                        # m 은 fp64 — Decimal 로 정확히
    c_ = _바른반올림32(m * np.cos(각), lambda j: md * _십진cos_sin(decimal.Decimal(float(평[j])))[0])
    s_ = _바른반올림32(m * np.sin(각), lambda j: md * _십진cos_sin(decimal.Decimal(float(평[j])))[1])
    return c_, s_


def rope(x, cos, sin, 위치):
    """RoPE(NeoX — 반쪽 돌리기) 를 머리 [n][hd] 에(hd 64 · 128, 반 = hd/2): i < 반 에서 x′_i = 곱해더하기(x_i, cos, −(x_{i+반} · sin)),
    x′_{i+반} = 곱해더하기(x_{i+반}, cos, x_i · sin) — cos · sin 은 표의 [위치][i] (표의 폭 = 반)."""
    x = np.asarray(x, f32)
    반 = x.shape[1] // 2
    c, s = cos[위치], sin[위치]                                      # [n][반]
    assert c.shape[1] == 반, "RoPE 표의 폭이 머리 크기의 반과 다르다"
    a, b = x[:, :반], x[:, 반:]
    return np.concatenate([fma32(a, c, -곱(b, s)), fma32(b, c, 곱(a, s))], 1).astype(f32)


def 머리정규화(x, w, eps=1e-6, hd=128):
    """37단계 — 머리마다 Q · K 정규화(Qwen3 — 계약 4.6): x [n][머리 수 · hd] 의 머리마다(행 하나의 hd 칸) rms정규화 를 D = hd 로 —
    가닥 t < 128 에 칸 하나씩, 128 접기는 +0 이라 p_t = x_t · x_t 를 64 부터 반씩 접은 S, r = 1 / 제곱근(S / hd + eps), y = (x · r) · w.
    w 는 hd 칸(머리끼리 같은 무게). RoPE 앞에 q · k 에 — v 는 하지 않는다."""
    x = np.asarray(x, f32)
    n, A = x.shape
    assert A % hd == 0, (A, hd)
    return rms정규화(x.reshape(n * (A // hd), hd), w, eps).reshape(n, A)


def swiglu(g, u):
    """SwiGLU 의 h = silu(g) · u — e = 적힌지수(0 − |g|), s = g ≥ 0 이면 g / (1 + e), 아니면 (g · e) / (1 + e), h = s · u (연산마다 IEEE 반올림)."""
    g, u = np.asarray(g, f32), np.asarray(u, f32)
    e = 적힌지수(빼기(f32(0), np.abs(g)))
    d = 더(f32(1), e)
    s = np.where(버림(g) >= 0, 나눔(g, d), 나눔(곱(g, e), d)).astype(f32)
    return 곱(s, u)


def 행렬곱빠른(x, q, d, 치우침):
    """행렬곱(…, 자리수=2) 와 같은 값 — 블록 합 S_j 를 fp64 행렬곱으로(|S| ≤ 32 · 128 · 128 < 2⁵³ 라 정확한 정수) 구한다(큰 모양의 CPU 참조용)."""
    D, E = 자리로(x, 2)
    M, K = D.shape[1:]
    KB = K // 32
    N = q.shape[1]
    y = np.zeros((M, N), f32)
    qf = np.asarray(q, np.float64)
    d = np.asarray(d, f32)
    Df = D.astype(np.float64)
    for b in range(KB):
        qb = qf[32 * b:32 * b + 32]
        v = [np.ldexp((Df[j][:, 32 * b:32 * b + 32] @ qb).astype(f32), (8 * j + E[:, b] - 13)[:, None]).astype(f32) for j in range(2)]
        u = 더(v[1], v[0])
        y = fma32(u, np.broadcast_to(d[b][None, :], u.shape), y)
    return 더(y, np.asarray(치우침, f32)[None, :])


def 행렬곱블라스(x, qNK, dNK, 치우침, 일꾼=8, 칸한도=1 << 25, 자리수=2):
    """행렬곱(…, 자리수=2) 와 같은 값(22단계 — 7B 의 CPU 참조). 가중치는 GGUF 의 배치 그대로: qNK [N][K] int8, dNK [N][K/32] 짧은실수.
    블록 합 S_j 를 짧은실수 행렬곱(BLAS)으로 — |D_j| ≤ 128, |q| ≤ 128 이라 |S_j| ≤ 32 · 128 · 128 = 2¹⁹ 이고 부분합도 정수로 2²⁴ 아래라 어느
    순서로 더해도 정확하다. 블록들을 한 번의 묶음 행렬곱으로, 열을 덩이로 나눠 일꾼(스레드 — numpy 가 GIL 을 놓는다)에게. v_j = S_j · 2^(8j + E − 13)
    (정확), u = v₁ + v₀ (반올림 하나), y ← 곱해더하기(u, d, y) 를 블록 차례로 — 행렬곱 과 같은 식.
    33단계 — 자리수 1(같은 최적화의 대조 V1): u = v₀ = S₀ · 2^(E − 5) (정확), 나머지는 같다."""
    from concurrent.futures import ThreadPoolExecutor
    D, E = 자리로(np.asarray(x, f32), 자리수)
    P = 8 * 자리수 - 3
    M, K = D.shape[1:]
    KB = K // 32
    N = qNK.shape[0]
    Df = D.astype(f32).reshape(자리수, M, KB, 32).transpose(0, 2, 1, 3).copy()      # [자리수][KB][M][32]
    배 = [np.ldexp(f32(1), (8 * j + E - P)).astype(f32).T[:, :, None] for j in range(자리수)]   # [KB][M][1]
    덩이 = max(32, min(4096, 칸한도 // max(1, KB * M)) // 32 * 32)
    y = np.zeros((M, N), f32)

    def 열(n0):
        n1 = min(N, n0 + 덩이)
        qb = np.ascontiguousarray(np.asarray(qNK[n0:n1], f32).reshape(n1 - n0, KB, 32).transpose(1, 2, 0))   # [KB][32][n]
        if 자리수 == 1:
            u = (np.matmul(Df[0], qb) * 배[0]).astype(f32)
        else:
            u = (np.matmul(Df[1], qb) * 배[1] + np.matmul(Df[0], qb) * 배[0]).astype(f32)                   # [KB][M][n]
        db = np.asarray(dNK[n0:n1], f32).T                                                                     # [KB][n]
        yy = np.zeros((M, n1 - n0), f32)
        for b in range(KB):
            yy = fma32(u[b], np.broadcast_to(db[b][None, :], yy.shape), yy)
        y[:, n0:n1] = yy
    with ThreadPoolExecutor(일꾼) as ex:
        list(ex.map(열, range(0, N, 덩이)))
    return 더(y, np.asarray(치우침, f32)[None, :])


def qwen2_앞으로(W, 설정, ids, cos, sin, 첫위치=0, 캐시=None, 묶음=16, 끝행=None):
    """정수 계약의 Qwen2 앞으로 — 토큰 ids(위치 첫위치 …)의 모든 행 로짓 [n][어휘] (짧은실수). W: 이름 → 정수 가중치 (q [K][N] 정수, d [KB][N]
    짧은실수) 또는 짧은실수 배열 — embed(q [어휘][D], d [어휘][D/32] — 행 = 토큰), head(q [D][어휘], d [D/32][어휘]), 층 l 의 wq · wk · wv · wo ·
    wg · wu · wd (q [K][N], d [K/32][N]) 와 bq · bk · bv,
    n1 · n2 · norm. 캐시(사전, 없으면 새로): 층마다 k · v (RoPE 뒤 짧은실수 [위치][KV머리][hd]) — 이어 계산(대화)을 위해 첫위치 앞의 키 · 값을 담는다.
    머리 크기 hd = D / 머리 수 (cos · sin 표의 폭은 hd/2 — rope표(…, hd=hd)).
    끝행: 로짓을 마지막 끝행 행만(긴 열의 CPU 시간 — 행마다의 값은 같다)."""
    NH, NKV, NL = 설정["머리"], 설정["KV머리"], 설정["층"]
    eps = 설정["eps"]
    ids = np.asarray(ids)
    n = len(ids)
    eq, ed = W["embed"]                                           # q [어휘][D] 정수, d [어휘][D/32] 짧은실수
    x = 곱(eq[ids].astype(f32), np.repeat(ed[ids], 32, axis=1))                # 임베딩: q · d (정확)
    위치 = np.arange(첫위치, 첫위치 + n)
    캐시 = {} if 캐시 is None else 캐시
    G = NH // NKV
    hd = x.shape[1] // NH                                          # 머리 크기(0.5B 64, 7B 128)
    for l in range(NL):
        h = rms정규화(x, W[f"{l}.n1"], eps)
        q = 행렬곱빠른(h, *W[f"{l}.wq"], W[f"{l}.bq"])
        k = 행렬곱빠른(h, *W[f"{l}.wk"], W[f"{l}.bk"])
        v = 행렬곱빠른(h, *W[f"{l}.wv"], W[f"{l}.bv"])
        q = np.stack([rope(q[:, hd * a:hd * a + hd], cos, sin, 위치) for a in range(NH)], 1)       # [n][NH][hd]
        k = np.stack([rope(k[:, hd * a:hd * a + hd], cos, sin, 위치) for a in range(NKV)], 1)
        v = v.reshape(n, NKV, hd)
        if l in 캐시:
            k = np.concatenate([캐시[l][0], k], 0)
            v = np.concatenate([캐시[l][1], v], 0)
        캐시[l] = (k, v)
        y = np.concatenate([어텐션(np.concatenate([np.zeros((첫위치, hd), f32), q[:, a]], 0), k[:, a // G], v[:, a // G], 묶음=묶음, 첫행=첫위치)
                            for a in range(NH)], 1)
        x = 더(x, 행렬곱빠른(y, *W[f"{l}.wo"], np.zeros(y.shape[1], f32)))
        h = rms정규화(x, W[f"{l}.n2"], eps)
        g = 행렬곱빠른(h, *W[f"{l}.wg"], np.zeros(W[f"{l}.wg"][0].shape[1], f32))
        u = 행렬곱빠른(h, *W[f"{l}.wu"], np.zeros(W[f"{l}.wu"][0].shape[1], f32))
        f = swiglu(g, u)
        x = 더(x, 행렬곱빠른(f, *W[f"{l}.wd"], np.zeros(x.shape[1], f32)))
    h = rms정규화(x if 끝행 is None else x[-끝행:], W["norm"], eps)
    hq, hd_ = W["head"]
    return 행렬곱빠른(h, hq, hd_, np.zeros(hq.shape[1], f32))


# ── 26단계 — k-양자화(Q4_K · Q5_K · Q6_K): 파일의 정수 · 척도 그대로 (문서/계약.md 1절) ──────────────────────────────────────────
# 가중치 하나 = (형식, q [N][K] int8, a [N][K/L] 짧은실수, b [N][K/32] 짧은실수 또는 None). L 은 아래블록의 칸(Q4_K · Q5_K 32, Q6_K 16).
# Q4_K · Q5_K: 슈퍼블록 256 = 아래블록 여덟, 반실수 d · m, 아래블록마다 6비트 s_j · t_j (llama.cpp 의 get_scale_min_k4), q ∈ [0, 15] · [0, 31],
#   a_j = d · s_j, b_j = m · t_j. Q6_K: 아래블록 열여섯, 반실수 d, 부호 있는 8비트 s_k, q = 6비트 − 32 ∈ [−32, 31], a_k = d · s_k.
# a · b 는 반실수(가수 11 비트) × 작은 정수(8 비트 안)라 짧은실수로 정확하다.

def _척최소K(sc):
    """Q4_K · Q5_K 의 척도 바이트 12 개 [..., 12] → (s [..., 8], t [..., 8]) — get_scale_min_k4 와 같다."""
    sc = np.asarray(sc).astype(np.int64)
    s = np.empty(sc.shape[:-1] + (8,), np.int64)
    t = np.empty_like(s)
    s[..., :4] = sc[..., 0:4] & 63
    t[..., :4] = sc[..., 4:8] & 63
    s[..., 4:] = (sc[..., 8:12] & 15) | ((sc[..., 0:4] >> 6) << 4)
    t[..., 4:] = (sc[..., 8:12] >> 4) | ((sc[..., 4:8] >> 6) << 4)
    return s, t


def 풀기K(t, raw, N, K):
    """GGUF k-양자화 텐서(타입 12 Q4_K · 13 Q5_K · 14 Q6_K, 날 바이트 — 행 N 개, 행마다 깊이 K) → (형식, q, a, b) (위의 꼴)."""
    KS = K // 256
    if t in (12, 13):
        bl = np.asarray(raw).reshape(N, KS, 144 if t == 12 else 176)
        d = np.ascontiguousarray(bl[:, :, 0:2]).view(np.float16).astype(f32)[..., 0]
        m = np.ascontiguousarray(bl[:, :, 2:4]).view(np.float16).astype(f32)[..., 0]
        s, tt = _척최소K(bl[:, :, 4:16])
        a = (d[..., None] * s.astype(f32)).astype(f32).reshape(N, K // 32)
        b = (m[..., None] * tt.astype(f32)).astype(f32).reshape(N, K // 32)
        if t == 12:
            qs = bl[:, :, 16:144].reshape(N, KS, 4, 32)                       # 덩이 c (값 64 개)마다 바이트 32 개
            q = np.stack([qs & 15, qs >> 4], 3)                               # 아래블록 2c (아래 니블) · 2c + 1 (위 니블)
        else:
            qh = bl[:, :, 16:48][:, :, None, :]                               # 높은 비트: 덩이 c 의 아래블록 2c 는 비트 2c, 2c + 1 은 비트 2c + 1
            qs = bl[:, :, 48:176].reshape(N, KS, 4, 32)
            c = np.arange(4)[None, None, :, None]
            q = np.stack([(qs & 15) + (((qh >> (2 * c)) & 1) << 4), (qs >> 4) + (((qh >> (2 * c + 1)) & 1) << 4)], 3)
        return ("Q4_K" if t == 12 else "Q5_K"), q.reshape(N, K).astype(np.int8), a, b
    assert t == 14, t
    bl = np.asarray(raw).reshape(N, KS, 210)
    ql = bl[:, :, 0:128].reshape(N, KS, 2, 64)                                # 반쪽 n (값 128 개)마다 낮은 4비트 64 바이트
    qh = bl[:, :, 128:192].reshape(N, KS, 2, 32)                              # 높은 2비트 32 바이트
    sc = np.ascontiguousarray(bl[:, :, 192:208]).view(np.int8)
    d = np.ascontiguousarray(bl[:, :, 208:210]).view(np.float16).astype(f32)[..., 0]
    lo, hi = ql[..., :32], ql[..., 32:]
    q = np.stack([(lo & 15) | (((qh >> 0) & 3) << 4), (hi & 15) | (((qh >> 2) & 3) << 4),
                  (lo >> 4) | (((qh >> 4) & 3) << 4), (hi >> 4) | (((qh >> 6) & 3) << 4)], 3).astype(np.int16) - 32
    a = (d[..., None] * sc.astype(f32)).astype(f32).reshape(N, K // 16)      # 아래블록 k (값 16k … 16k + 15) 의 척도 sc[k]
    return "Q6_K", q.reshape(N, K).astype(np.int8), a, None


def 행렬곱K(x, w, 치우침, 자리수=2):
    """정수 블록 계약의 k-양자화 출력 [행][열] — 문서/계약.md 1절 그대로 칸마다(느린 참조 — 행렬곱K블라스 의 자체 시험용).
    Q4_K · Q5_K: 아래블록 j(= 활성값 블록 j)마다 S_j = Σ n · q, T_j = Σ n (정확), u_j = S_j · 2^(E − 13), w_j = T_j · 2^(E − 13) (정확),
    y ← 곱해더하기(u_j, a_j, y), y ← 곱해더하기(−w_j, b_j, y) (j 차례, 0 에서 — 사슬 하나), 출력 = y + 치우침.
    Q6_K: 아래블록 k(16 칸)마다 S_k = Σ n · q, u_k = S_k · 2^(E_⌊k/2⌋ − 13), y ← 곱해더하기(u_k, a_k, y), 출력 = y + 치우침.
    GeulPUv2 빠른 계약 F1(자리수 1): n 은 한 자리(|n| ≤ 127), 2^(E − 13) 대신 2^(E − 5) — 나머지 식 · 차례는 같다."""
    형식, q, a, b = w
    D, E = 자리로(np.asarray(x, f32), 자리수)
    n = (D[0] if 자리수 == 1 else D[1] * 256 + D[0]).astype(np.int64)     # |n| ≤ 2¹⁴ (자리 하나 127)
    P = 8 * 자리수 - 3
    M, K = n.shape
    N = q.shape[0]
    L = 16 if 형식 == "Q6_K" else 32
    qi = np.asarray(q, np.int64)
    y = np.zeros((M, N), f32)
    for j in range(K // L):
        S = n[:, L * j:L * j + L] @ qi[:, L * j:L * j + L].T                  # 정확한 정수 [M][N]
        e = E[:, (L * j) // 32][:, None] - P
        u = np.ldexp(S.astype(np.float64), e).astype(f32)                    # 정확(|S| < 2²⁴)
        y = fma32(u, np.broadcast_to(np.asarray(a, f32)[:, j][None, :], u.shape), y)
        if b is not None:
            T = n[:, 32 * j:32 * j + 32].sum(1)[:, None]
            w_ = np.broadcast_to(np.ldexp(T.astype(np.float64), e).astype(f32), u.shape)
            y = fma32(-w_, np.broadcast_to(np.asarray(b, f32)[:, j][None, :], u.shape), y)
    return 더(y, np.asarray(치우침, f32)[None, :])


def 행렬곱K블라스(x, w, 치우침, 일꾼=8, 칸한도=1 << 25, 자리수=2):
    """행렬곱K 와 같은 값(큰 모양의 CPU 참조). 블록 합 S 를 자리마다 짧은실수 행렬곱(BLAS)으로 — 자리 하나의 부분합이 정수로 2²⁴ 아래
    (Q5_K: 128 · 31 · 32)라 어느 순서로 더해도 정확하다 — S = 256 · S₁ + S₀ 는 fp64 로(정확). 열을 덩이로 나눠 일꾼에게.
    GeulPUv2 빠른 계약 F1(자리수 1): S = S₀, 척 2^(E − 5)."""
    from concurrent.futures import ThreadPoolExecutor
    형식, q, a, b = w
    D, E = 자리로(np.asarray(x, f32), 자리수)
    P = 8 * 자리수 - 3
    M, K = D.shape[1:]
    N = q.shape[0]
    L = 16 if 형식 == "Q6_K" else 32
    KL = K // L
    Df = D.astype(f32).reshape(자리수, M, KL, L).transpose(0, 2, 1, 3).copy()        # [자리수][KL][M][L]
    배 = np.ldexp(np.float64(1), (E - P)).T[:, :, None]                               # [KB][M][1]
    if L == 16:
        배 = np.repeat(배, 2, axis=0)                                                  # 아래블록 k 는 활성값 블록 ⌊k/2⌋
    wT = None
    if b is not None:
        n = (D[0] if 자리수 == 1 else D[1] * 256 + D[0]).astype(np.int64)
        wT = (n.reshape(M, KL, L).sum(2).T[:, :, None].astype(np.float64) * 배).astype(f32)     # [KL][M][1] (정확)
    덩이 = max(32, min(4096, 칸한도 // max(1, KL * M)) // 32 * 32)
    y = np.zeros((M, N), f32)

    def 열(n0):
        n1 = min(N, n0 + 덩이)
        qb = np.ascontiguousarray(np.asarray(q[n0:n1], f32).reshape(n1 - n0, KL, L).transpose(1, 2, 0))   # [KL][L][n]
        if 자리수 == 1:
            S = np.matmul(Df[0], qb).astype(np.float64)
        else:
            S = np.matmul(Df[1], qb).astype(np.float64) * 256 + np.matmul(Df[0], qb).astype(np.float64)
        u = (S * 배).astype(f32)                                                                           # [KL][M][n]
        ab = np.asarray(a[n0:n1], f32).T
        bb = None if b is None else np.asarray(b[n0:n1], f32).T
        yy = np.zeros((M, n1 - n0), f32)
        for j in range(KL):
            yy = fma32(u[j], np.broadcast_to(ab[j][None, :], yy.shape), yy)
            if bb is not None:
                yy = fma32(np.broadcast_to(-wT[j], yy.shape), np.broadcast_to(bb[j][None, :], yy.shape), yy)
        y[:, n0:n1] = yy
    with ThreadPoolExecutor(일꾼) as ex:
        list(ex.map(열, range(0, N, 덩이)))
    return 더(y, np.asarray(치우침, f32)[None, :])


def 임베딩K(w, ids):
    """낱말표가 k-양자화일 때의 임베딩 [토큰][D] — Q4_K · Q5_K: a_j · q − b_j (곱은 정확, 뺄셈에서 반올림 하나), Q6_K: a_k · q (정확)."""
    형식, q, a, b = w
    ids = np.asarray(ids)
    qq = q[ids].astype(f32)
    if 형식 == "Q6_K":
        return 곱(qq, np.repeat(a[ids], 16, axis=1))
    return 빼기(곱(qq, np.repeat(a[ids], 32, axis=1)), np.repeat(b[ids], 32, axis=1))
