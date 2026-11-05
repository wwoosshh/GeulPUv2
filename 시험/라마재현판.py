#!/usr/bin/env python3
"""llama.cpp 실행의 비트 재현(문서/라마재현.md) — 재현 모듈(커널생성 라마재현)의 판을 llama.cpp 의 기록과 비트까지.

기록은 llama.cpp b12008 의 공식 DLL 을 그대로 부르고 그래프 노드를 cb_eval 로 적은 것이다(노드마다 입력과 출력). 판마다 기록된 입력을
넣고 기록된 출력과 견준다. 판정 1-1(생성 1걸음, 32층 — 시험 RMS · 행렬곱 · 원소 · 합성곱 · GDN · RoPE · 캐시 · FA · 임베딩)과
판정 2-1(프롬프트 ubatch, 0층 · 3층 — 시험 RMS · 원소 · MMQ · 행렬곱여럿 · 합성곱여럿 · RoPE여럿 · 캐시여럿 · FA여럿 · GDN여럿)의 시험이다.

  python -I 시험/라마재현판.py <라마재현.ptx> <기록 폴더> <모델.gguf> [시험 이름 …]
마지막 줄 "모두 0: True|False".
"""
import ctypes
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "계약"))
sys.path.insert(0, HERE)
import gguf읽기             # noqa: E402
import 쿠다드라이버 as G      # noqa: E402

f32 = np.float32
u64, i64 = G.c_u64, G.c_i64
결과 = []


class 기록:
    """llama.cpp 의 노드 기록: index.json(노드 · 텐서 — ne · nb · 파일) + 텐서마다의 날 바이트."""

    def __init__(self, d):
        self.d = d
        ix = json.load(open(os.path.join(d, "index.json"), encoding="utf-8"))
        self.노드들, self.텐서 = ix["nodes"], ix["tensors"]
        # 층 일부만 적은 기록(프롬프트 덩어리 — meta.json 의 layers): 그 층의 노드만 본다. 다른 층의 노드는 입력이 그 층이라 함께
        # 적혔을 뿐이고, 하네스가 텐서를 주소로 한 번만 저장해 그 노드의 다른 입력은 주소를 다시 쓴 앞 텐서의 값일 수 있다
        meta = os.path.join(os.path.dirname(os.path.normpath(d)), "meta.json")
        층 = json.load(open(meta, encoding="utf-8")).get("layers") if os.path.exists(meta) else None
        if 층 and len(층) < 32:
            def 안(n):
                m = re.search(r"-(\d+)$", n["name"])
                return (int(m.group(1)) in 층) if m else not n["name"].startswith("result_")
            self.노드들 = [n for n in self.노드들 if 안(n)]

    def 배열(self, rec):
        dt = {"f32": np.float32, "f16": np.float16, "i32": np.int32, "i64": np.int64}[rec["type"]]
        buf = np.fromfile(os.path.join(self.d, rec["file"]), dtype=np.uint8)
        ne, nb = rec["ne"], rec["nb"]
        return np.lib.stride_tricks.as_strided(buf.view(dt), shape=(ne[3], ne[2], ne[1], ne[0]),
                                               strides=(nb[3], nb[2], nb[1], nb[0])).copy()

    def 노드(self, name, nth=0):
        n = [x for x in self.노드들 if x["name"] == name][nth]
        return n, self.텐서[n["ptr"]]

    def 입력(self, n, i):
        return self.배열(self.텐서[n["srcs"][i]["ptr"]])

    def 연산(self, op):
        return [(n, self.텐서[n["ptr"]]) for n in self.노드들 if self.텐서[n["ptr"]]["op"] == op]


def 견주기(이름, got, want):
    got = np.ascontiguousarray(got, f32).reshape(-1)
    want = np.ascontiguousarray(want, f32).reshape(-1)
    assert got.size == want.size, (이름, got.size, want.size)
    bad = int(np.count_nonzero(got.view(np.uint32) != want.view(np.uint32)))
    결과.append((이름, got.size, bad))
    덧 = ""
    if bad:
        i = int(np.flatnonzero(got.view(np.uint32) != want.view(np.uint32))[0])
        덧 = f" (처음 {i}: {got[i]!r} · 기록 {want[i]!r})"
    print(f"{이름:40s} 값 {got.size:8d} · 불일치 {bad}{덧}", flush=True)


class 판:
    def __init__(self, ptx):
        self.dr = G.드라이버()
        self.ptx = open(ptx, "rb").read()
        self.mod = self.dr.모듈(self.ptx)

    def 띄우기(self, 이름, grid, block, params):
        self.fn = getattr(self, "fn", {})
        if 이름 not in self.fn:
            self.fn[이름] = self.dr.함수(self.mod, 이름)
        fn = self.fn[이름]
        ps = [u64(p) if 참조_ else i64(p) for 참조_, p in params]
        args = (ctypes.c_void_p * len(ps))(*[ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in ps])
        r = self.dr.cu.cuLaunchKernel(fn, grid[0], grid[1], 1, block, 1, 1, 0, None, args, None)
        if r != 0:
            raise RuntimeError(f"{이름}: cuLaunchKernel {r}")

    def 올림(self, x):
        x = np.ascontiguousarray(x)
        p = self.dr.할당(max(x.nbytes, 4))
        self.dr.올리기(p, x)
        return p

    def 받음(self, p, 모양, dt=f32):
        a = np.zeros(모양, dt)
        self.dr.맞추기()
        return self.dr.내리기(p, a)

    def 놓기(self):
        """잡은 장치 메모리만 돌려준다(모듈은 그대로 — 함수 핸들을 다시 쓴다)."""
        self.dr.맞추기()
        for p in self.dr.할당들:
            self.dr.확인(self.dr.cu.cuMemFree_v2(u64(p)))
        self.dr.할당들 = []


class 판V:
    """3단계 — 같은 판을 Vulkan 에서(판실행V · SPIR-V 묶음): --벌칸 <판실행V.exe> [--장치 <이름의 일부>] [--작업 <빈칸 없는 폴더>]"""

    def __init__(self, exe, spvs, 폴더, 장치):
        from 판실행 import 판실행
        self.d = 판실행(exe, spvs, 폴더, 장치=장치)
        print(f"Vulkan: {self.d.장치}", flush=True)

    def 띄우기(self, 이름, grid, block, params):
        self.d.띄우기(이름, (grid[0], grid[1], 1), (block, 1, 1), [p for _, p in params])

    def 올림(self, x):
        x = np.ascontiguousarray(x)
        p = self.d.할당(max(x.nbytes, 4))
        self.d.올리기(p, x)
        return p

    def 받음(self, p, 모양, dt=f32):
        a = np.zeros(모양, dt)
        return self.d.내리기(p, a)

    def 놓기(self):
        self.d.놓기()


def 가중블록(t, raw, R):
    """GGUF 의 블록을 장치 배치로: Q4_K · Q5_K 는 그대로, Q6_K 는 블록(210 바이트) 뒤에 2 바이트, Q8_0 은 d 뒤에 2 바이트(값 낱말을 4 바이트 경계에)."""
    if t == gguf읽기.Q6_K:
        b = raw.reshape(-1, 210)
        return np.concatenate([b, np.zeros((b.shape[0], 2), np.uint8)], 1).reshape(-1)
    if t == gguf읽기.Q8_0:
        b = raw.reshape(-1, 34)
        return np.concatenate([b[:, :2], np.zeros((b.shape[0], 2), np.uint8), b[:, 2:]], 1).reshape(-1)
    return np.ascontiguousarray(raw)


def 비트(x):
    return int(np.array([x], f32).view(np.int32)[0])


def RMS판(열):
    """13단계 — 엔진/라마재현몸.gl 의 RMS 와 같은 규칙: n4 = 열 / 4 가 ≤ 32 → 블록 32, ≤ 128 → 128, ≤ 256 → 256,
    512 < n4 ≤ 1024 → 1024(3584 · 3072 · 4096 은 1024, 1024 는 256)."""
    n4 = 열 // 4
    if n4 <= 32:
        return "라마RMS32", 32
    if n4 <= 128:
        return "라마RMS128", 128
    if n4 <= 256:
        return "라마RMS256", 256
    if n4 <= 512:
        return "라마RMS512", 512                                     # 14단계 — 폭 1536
    if n4 <= 1024:
        return "라마RMS1024", 1024
    raise NotImplementedError(f"RMSNorm 폭 {열} 의 판이 없다")


def 시험_RMS(P, R, M):
    """RMS_NORM 노드 모두(블록은 엔진의 규칙 — RMS판), 그리고 그 출력에 가중치를 곱하는 MUL 노드(엮은 판)."""
    출력표 = {}
    for n, rec in R.연산("RMS_NORM"):
        x = R.입력(n, 0)
        열 = x.shape[-1]
        이름, B = RMS판(열)
        행 = x.size // 열
        eps = int(np.array(rec["op_params"][:1], np.int32)[0])
        px, py = P.올림(x.astype(f32)), P.올림(np.zeros(x.size, f32))
        P.띄우기(이름, (행, 1), B, [(1, px), (1, px), (1, py), (0, 열), (0, 열), (0, 0), (0, eps)])
        got = P.받음(py, x.size)
        견주기(f"RMS_NORM {열}×{행} ({n['name']})", got, R.배열(rec))
        출력표[n["ptr"]] = (x, 이름, B, 열, 행, eps)
    for n, rec in R.연산("MUL"):
        s0, s1 = n["srcs"][0], n["srcs"][1]
        if s0.get("weight") or not s1.get("weight") or s0.get("ptr") not in 출력표:
            continue
        x, 이름, B, 열, 행, eps = 출력표[s0["ptr"]]
        w = M.풀기(s1["name"]).astype(f32).reshape(-1)
        px, pw, py = P.올림(x.astype(f32)), P.올림(w), P.올림(np.zeros(x.size, f32))
        P.띄우기(이름, (행, 1), B, [(1, px), (1, pw), (1, py), (0, 열), (0, 열), (0, 1), (0, eps)])
        견주기(f"RMS_NORM · MUL {열}×{행} ({n['name']})", P.받음(py, x.size), R.배열(rec))


def 행렬벡터(P, M, x, wname):
    """q8_1 양자화(라마양자화q8_1) + MMVQ — llama.cpp 의 MUL_MAT(열 1)과 같은 셈. x [K] → [R]."""
    t, dims, raw = M.원(wname)
    K, R = dims
    판형식 = {gguf읽기.Q4_K: ("라마MMVQ_Q4_K", K // 256), gguf읽기.Q5_K: ("라마MMVQ_Q5_K", K // 256),
            gguf읽기.Q6_K: ("라마MMVQ_Q6_K", K // 256), gguf읽기.Q8_0: ("라마MMVQ_Q8_0", K // 32)}
    이름, nb = 판형식[t]
    px = P.올림(np.ascontiguousarray(x, f32))
    pq, pd = P.올림(np.zeros(K // 4, np.int32)), P.올림(np.zeros(K // 32, np.float16))
    P.띄우기("라마양자화q8_1", ((K + 255) // 256, 1), 256, [(1, px), (1, pq), (1, pd), (0, K)])
    pw, py = P.올림(가중블록(t, raw, R)), P.올림(np.zeros(R, f32))
    P.띄우기(이름, (R, 1), 128, [(1, pw), (1, pq), (1, pd), (1, py), (0, nb)])
    return P.받음(py, R), gguf읽기.이름[t]


def 시험_행렬곱(P, R, M):
    """MUL_MAT 노드 모두(생성 1걸음 — 열 하나): 가중치 Q4_K · Q5_K · Q6_K · Q8_0, 출력층(Q6_K 248,320 행)까지."""
    for n, rec in R.연산("MUL_MAT"):
        s0 = n["srcs"][0]
        if not s0.get("weight"):
            continue
        x = R.입력(n, 1).reshape(-1)
        got, 형식 = 행렬벡터(P, M, x, s0["name"])
        견주기(f"MUL_MAT {형식} {got.size}×{x.size} ({n['name']})", got, R.배열(rec))
        P.놓기()


def 입력값(R, M, n, i):
    """노드의 입력 i — 가중치면 GGUF 에서(짧은실수), 아니면 기록에서."""
    s = n["srcs"][i]
    if s.get("weight"):
        return M.풀기(s["name"]).astype(f32).reshape(-1)
    return R.입력(n, i).astype(f32).reshape(-1)


def 원소(P, a, b, 종류, s=0, t=0):
    a = np.ascontiguousarray(a, f32).reshape(-1)
    b = np.ascontiguousarray(a if b is None else b, f32).reshape(-1)
    pa, pb, py = P.올림(a), P.올림(b), P.올림(np.zeros(a.size, f32))
    P.띄우기("라마원소", ((a.size + 255) // 256, 1), 256, [(1, pa), (1, pb), (1, py), (0, a.size), (0, b.size), (0, 종류), (0, s), (0, t)])
    return P.받음(py, a.size)


def 시험_원소(P, R, M):
    """SILU · SIGMOID · SOFTPLUS · SWIGLU · SCALE 과 ADD · MUL(RMS_NORM 에 엮이지 않은 것) 노드 모두."""
    rms = {n["ptr"] for n, _ in R.연산("RMS_NORM")}
    for op, 종류 in (("SILU", 2), ("SIGMOID", 3), ("SOFTPLUS", 4), ("SWIGLU", 5), ("SCALE", 6), ("ADD", 1), ("MUL", 0)):
        for n, rec in R.연산(op):
            if op == "MUL" and n["srcs"][0].get("ptr") in rms:
                continue                                                 # 시험_RMS 가 본다
            a = 입력값(R, M, n, 0)
            b = 입력값(R, M, n, 1) if len(n["srcs"]) > 1 else None
            s = t = 0
            if op == "SCALE":
                s, t = rec["op_params"][0], rec["op_params"][1]
            견주기(f"{op} {a.size} ({n['name']})", 원소(P, a, b, 종류, s, t), R.배열(rec))
        P.놓기()


def 시험_합성곱(P, R, M):
    """SSM_CONV(합성곱 상태 셋 + 새 값 하나 = CONCAT) 노드와, 민 상태(새 입력 끝 셋)."""
    for n, rec in R.연산("SSM_CONV"):
        cin = R.입력(n, 0).reshape(-1, 4).astype(f32)                    # [채널, 4]
        w = M.풀기(n["srcs"][1]["name"]).astype(f32).reshape(-1)
        C = cin.shape[0]
        ps, px, pw, py = P.올림(cin[:, :3].copy()), P.올림(cin[:, 3].copy()), P.올림(w), P.올림(np.zeros(C, f32))
        P.띄우기("라마합성곱", ((C + 255) // 256, 1), 256, [(1, ps), (1, px), (1, pw), (1, py), (0, C)])
        견주기(f"SSM_CONV {C} ({n['name']})", P.받음(py, C), R.배열(rec))
        견주기(f"합성곱 상태 밀기 {C * 3} ({n['name']})", P.받음(ps, C * 3), cin[:, 1:4])
        P.놓기()


def 시험_GDN(P, R, M):
    """GATED_DELTA_NET(한 걸음) 노드: 출력 [v머리][128] 과 새 상태 [v머리][128][128]."""
    눈금 = 비트((f32(1) / np.sqrt(f32(128))).astype(f32))
    for n, rec in R.연산("GATED_DELTA_NET"):
        q, k, v = (R.입력(n, i).astype(f32).reshape(-1) for i in range(3))
        g, beta = R.입력(n, 3).astype(f32).reshape(-1), R.입력(n, 4).astype(f32).reshape(-1)
        S = R.입력(n, 5).astype(f32).reshape(-1)
        H = g.size
        pq, pk, pv, pg, pb, pS, po = (P.올림(a) for a in (q, k, v, g, beta, S, np.zeros(H * 128, f32)))
        P.띄우기("라마GDN한걸음", (8, H), 128, [(1, pq), (1, pk), (1, pv), (1, pg), (1, pb), (1, pS), (1, po), (0, q.size // 128), (0, 눈금)])
        out = R.배열(rec).reshape(-1)
        견주기(f"GATED_DELTA_NET 출력 {H * 128} ({n['name']})", P.받음(po, H * 128), out[:H * 128])
        견주기(f"GATED_DELTA_NET 새 상태 {S.size} ({n['name']})", P.받음(pS, S.size), out[H * 128:H * 128 + S.size])
        P.놓기()


def 로프인자(M, D):
    """11단계 — RoPE 의 freq_factors: GGUF 의 rope_freqs.weight(F32, 머리 차원 / 2 칸 — llama3), 없으면 1."""
    try:
        _, _, raw = M.원("rope_freqs.weight")
        return np.frombuffer(bytes(raw), f32)[:D // 2].copy()
    except (KeyError, ValueError):
        return np.ones(D // 2, f32)


def 시험_RoPE(P, R, M):
    """ROPE(IMRoPE) 노드: q(머리 16) · k(머리 4), 위치 · θ 눈금(호스트의 powf — float64 의 거듭제곱을 짧은실수로) · 진동수 눈금 · attn."""
    for n, rec in R.연산("ROPE"):
        x = R.입력(n, 0).astype(f32).reshape(-1)
        pos = int(R.입력(n, 1).reshape(-1)[0])
        op = rec["op_params"]
        fb = np.array([op[5]], np.int32).view(f32)[0]
        ts = f32(np.power(np.float64(fb), -2.0 / op[1]))
        D = rec["ne"][0]                                                # 10단계 — 머리 크기(256 · 128)
        H = x.size // D
        px, py = P.올림(x), P.올림(np.zeros(x.size, f32))
        if op[2] == 0:                                                  # 11단계 — 보통 RoPE(llama) · freq_factors
            pf = P.올림(로프인자(M, D))
            P.띄우기(f"라마RoPE보통{D}", (H, 1), D, [(1, px), (1, pf), (1, py), (0, op[1]), (0, pos), (0, 비트(ts)), (0, op[6]), (0, op[8])])
        else:
            P.띄우기("라마RoPE" if D == 256 else f"라마RoPE{D}", (H, 1), D, [(1, px), (1, py), (0, op[1]), (0, pos), (0, 비트(ts)), (0, op[6]), (0, op[8])])
        견주기(f"ROPE {H}×{D} 위치 {pos} ({n['name']})", P.받음(py, x.size), R.배열(rec))
        P.놓기()


def 시험_캐시(P, R, M):
    """SET_ROWS(KV 캐시 쓰기, f16): 새 행의 반실수가 기록된 캐시의 그 행과 같은가."""
    for n, rec in R.연산("SET_ROWS"):
        x = R.입력(n, 0).astype(f32).reshape(-1)
        행 = int(R.입력(n, 1).reshape(-1)[0])
        캐시 = R.배열(rec).reshape(rec["ne"][1], rec["ne"][0])
        px, py = P.올림(x), P.올림(np.zeros(x.size, np.float16))
        P.띄우기("라마반실수쓰기", ((x.size + 255) // 256, 1), 256, [(1, px), (1, py), (0, x.size), (0, 0)])
        got = P.받음(py, x.size, np.float16)
        견주기(f"SET_ROWS f16 {x.size} 행 {행} ({n['name']})", got.astype(f32), 캐시[행].astype(f32))
        P.놓기()


def FA벡터점유(D):
    """llama.cpp 의 flash_attn_ext_vec<D, 1, F16, F16> 의 SM 당 블록 수(sm_89 SASS 의 레지스터 — D 256 약 200 → 2, D 128 약 168 → 3).
    XR_FA_OCC 로 바꿔 볼 수 있다."""
    return int(os.environ.get("XR_FA_OCC", "0")) or (2 if D == 256 else 3)


def 병렬블록(nkv, H=16, D=256):
    """flash attention vec 의 병렬 블록 수 — 엔진/라마재현몸.gl 의 병렬블록 과 같은 규칙(ntiles_KV = ⌈nkv / D⌉(launch_fattn 의
    nbatch_fa = D), 블록 = q 머리 × P, SM 당 점유 블록, P 는 min(점유, ntiles_KV) 부터 파도 효율이 가장 좋은 첫 값 — 95 % 이상이면 파도가
    느는 값에서 멈춘다)."""
    occ = FA벡터점유(D)
    nt = (nkv + D - 1) // D
    고른 = min(occ, nt)
    최고, 최고파도, 멈춤 = 0, 0, False
    for pt in range(고른, nt + 1):
        nbt = H * pt
        nw = (nbt + occ * NSM - 1) // (occ * NSM)
        eff = 100 * nbt // (nw * occ * NSM)
        if 최고 >= 95 and nw > 최고파도:
            멈춤 = True
        if not 멈춤 and eff > 최고:
            최고파도, 최고, 고른 = nw, eff, pt
    return 고른


def 시험_FA(P, R, M):
    """FLASH_ATTN_EXT(생성 — vec, 병렬 블록 P(kv 2560 → 7, 1280 → 5) + 합치기) 노드."""
    for n, rec in R.연산("FLASH_ATTN_EXT"):
        q = R.입력(n, 0).astype(f32)[0]                                 # [머리, 1, D]
        if q.shape[1] != 1:
            continue
        Kc = np.ascontiguousarray(np.transpose(R.입력(n, 1)[0], (1, 0, 2)))   # [kv수, kv머리, D] f16
        Vc = np.ascontiguousarray(np.transpose(R.입력(n, 2)[0], (1, 0, 2)))
        가림 = R.입력(n, 3).reshape(-1).astype(np.float16)
        nkv, H, D = Kc.shape[0], q.shape[0], q.shape[2]
        병렬 = 병렬블록(nkv, H, D)
        눈금 = rec["op_params"][0]
        sfx = "" if D == 256 else str(D)
        pq, pK, pV, pm = P.올림(q.reshape(-1)), P.올림(Kc.reshape(-1)), P.올림(Vc.reshape(-1)), P.올림(가림)
        p부분, p메타, po = P.올림(np.zeros(병렬 * H * D, f32)), P.올림(np.zeros(병렬 * H * 2, f32)), P.올림(np.zeros(H * D, f32))
        P.띄우기(f"라마FA벡터{sfx}", (병렬, H), 128, [(1, pq), (1, pK), (1, pV), (1, pm), (1, p부분), (1, p메타), (0, nkv), (0, 병렬),
                                                (0, H // Kc.shape[1]), (0, 눈금), (0, H)])
        P.띄우기(f"라마FA합치기{sfx}", (H, 1), D, [(1, p부분), (1, p메타), (1, po), (0, 병렬), (0, H)])
        견주기(f"FLASH_ATTN_EXT vec {H}×{D} kv {nkv} 병렬 {병렬} ({n['name']})", P.받음(po, H * D), R.배열(rec))
        P.놓기()


NSM = 60                                                        # RTX 4070 Ti — llama.cpp 가 일을 나눌 때 쓰는 SM 수
QK = {gguf읽기.Q4_K: 256, gguf읽기.Q5_K: 256, gguf읽기.Q6_K: 256, gguf읽기.Q8_0: 32}
MMQ판 = {gguf읽기.Q4_K: "라마MMQ_Q4_K", gguf읽기.Q5_K: "라마MMQ_Q5_K", gguf읽기.Q6_K: "라마MMQ_Q6_K", gguf읽기.Q8_0: "라마MMQ_Q8_0"}


def MMQ계획(N, K, R, 형식):
    """llama.cpp b12008 의 MMQ 일 나누기(sm_89): J = 열 타일이 가장 적은 첫 폭 — 설정표에 있는 폭만(8 · 16 · 24 · 32 · 40 · 48 · 64 · 80 ·
    96 · 112 · 128, 행이 128 의 배수가 아니면 8 · 16 · 32 · 64 · 128),
    타일 효율 ≥ 90 % 면 블록 = 타일, 아니면 stream-k 블록 min(SM, 타일 × ⌈K/256⌉). 블록 경계는 타일 안에서 256/qk 블록의 배수로 내린다.
    돌려주는 것: J, ntx, 타일 수, 조각 표 [(행 타일, 열 타일, k0, k1)], 타일 표 [(첫 조각, 조각 수)]."""
    qk = QK[형식]
    폭들 = [8, 16, 32, 64, 128] if R % 128 != 0 else [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128]
    J, 최소 = 0, 1 << 30
    for j in 폭들:
        if 최소 <= 1:
            break
        ntx = (N + j - 1) // j
        if ntx < 최소:
            J, 최소 = j, ntx
    nty, ntx = (R + 127) // 128, (N + J - 1) // J
    ntiles = ntx * nty
    nwaves = (ntiles + NSM - 1) // NSM
    grid = ntiles if 100 * ntiles // (NSM * nwaves) >= 90 else min(NSM, ntiles * ((K + 255) // 256))
    bpn, bpi = K // qk, 256 // qk
    W = ntiles * bpn

    def 시작(b):
        k = b * W // grid
        return k - (k % bpn) % bpi
    표, 타일표 = [], [[0, 0] for _ in range(ntiles)]
    for b in range(grid):
        k0, k1 = 시작(b), 시작(b + 1)
        while k0 < k1:
            t = k0 // bpn
            끝 = min(k1, (t + 1) * bpn)
            if 타일표[t][1] == 0:
                타일표[t][0] = len(표)
            타일표[t][1] += 1
            표.append([t // ntx, t % ntx, k0 - t * bpn, 끝 - t * bpn])
            k0 = 끝
    return J, ntx, ntiles, np.array(표, np.int32), np.array(타일표, np.int32)


def 행렬곱MMQ(P, M, x, wname):
    """quantize_mmq_q8_1 + MMQ(stream-k 조각) + 합치기 — llama.cpp 의 MUL_MAT(열 9 개 이상). x [N, K] → [N, R]."""
    t, dims, raw = M.원(wname)
    K, R = dims
    N = x.shape[0]
    J, ntx, ntiles, 표, 타일표 = MMQ계획(N, K, R, t)
    ds4 = int(t in (gguf읽기.Q4_K, gguf읽기.Q5_K))
    px = P.올림(np.ascontiguousarray(x, f32).reshape(-1))
    pq, pd, ps = P.올림(np.zeros(N * K // 4, np.int32)), P.올림(np.zeros(N * K // 32, f32)), P.올림(np.zeros(N * K // 32, f32))
    P.띄우기("라마MMQ양자화", ((N * K // 4 + 255) // 256, 1), 256, [(1, px), (1, pq), (1, pd), (1, ps), (0, N * K // 4), (0, ds4)])
    pw = P.올림(가중블록(t, raw, R))
    p조각 = P.올림(np.zeros(len(표) * 16384, f32))
    p표, p타일 = P.올림(표.reshape(-1)), P.올림(타일표.reshape(-1))
    py = P.올림(np.zeros(N * R, f32))
    P.띄우기(MMQ판[t], (len(표), 1), 256, [(1, pw), (1, pq), (1, pd), (1, ps), (1, p조각), (1, p표), (0, N), (0, R), (0, K), (0, J)])
    P.띄우기("라마MMQ합치기", (ntiles, 1), 256, [(1, p조각), (1, p타일), (1, py), (0, N), (0, R), (0, J), (0, ntx)])
    return P.받음(py, (N, R)), f"{gguf읽기.이름[t]} J {J} 조각 {len(표)}/타일 {ntiles}"


def 시험_MMQ(P, R, M):
    """MUL_MAT 노드 가운데 열이 9 개 이상인 것(프롬프트 ubatch — MMQ). 열 8 개 이하는 열마다 MMVQ(시험 행렬곱여럿)."""
    for n, rec in R.연산("MUL_MAT"):
        s0 = n["srcs"][0]
        if not s0.get("weight"):
            continue
        x = R.입력(n, 1)
        x = x.reshape(-1, x.shape[-1])
        if x.shape[0] <= 8:
            continue
        got, 덧 = 행렬곱MMQ(P, M, x, s0["name"])
        견주기(f"MUL_MAT {x.shape[0]}×{x.shape[1]} {덧} ({n['name']})", got, R.배열(rec))
        P.놓기()


def 시험_행렬곱여럿(P, R, M):
    """MUL_MAT 노드 가운데 열이 2 … 8 개인 것 — llama.cpp 는 MMVQ(열 수만큼)로, 열마다의 셈은 생성 걸음과 같다."""
    for n, rec in R.연산("MUL_MAT"):
        s0 = n["srcs"][0]
        if not s0.get("weight"):
            continue
        x = R.입력(n, 1)
        x = x.reshape(-1, x.shape[-1])
        if not 1 < x.shape[0] <= 8:
            continue
        열들 = []
        for c in range(x.shape[0]):
            y, 형식 = 행렬벡터(P, M, x[c], s0["name"])
            열들.append(y)
            P.놓기()
        견주기(f"MUL_MAT 열 {x.shape[0]} MMVQ {형식} ({n['name']})", np.stack(열들), R.배열(rec))


def 시험_합성곱여럿(P, R, M):
    """SSM_CONV(토큰 n 개 — 입력은 [채널][3 + n]) 노드와, 민 상태(끝 셋)."""
    for n, rec in R.연산("SSM_CONV"):
        cin = R.입력(n, 0).astype(f32)
        cin = cin.reshape(-1, cin.shape[-1])                              # [채널, 3 + n]
        C, T = cin.shape[0], cin.shape[1] - 3
        w = M.풀기(n["srcs"][1]["name"]).astype(f32).reshape(-1)
        x = np.ascontiguousarray(cin[:, 3:].T)                            # [토큰][채널]
        ps, px, pw, py = P.올림(cin[:, :3].copy()), P.올림(x), P.올림(w), P.올림(np.zeros(C * T, f32))
        P.띄우기("라마합성곱여럿", ((C * T + 255) // 256, 1), 256, [(1, ps), (1, px), (1, pw), (1, py), (0, C), (0, T)])
        P.띄우기("라마합성곱상태", ((C + 255) // 256, 1), 256, [(1, ps), (1, px), (0, C), (0, T)])
        견주기(f"SSM_CONV {C}×{T} ({n['name']})", P.받음(py, C * T), R.배열(rec))
        견주기(f"합성곱 상태 밀기 {C * 3} ({n['name']})", P.받음(ps, C * 3), cin[:, -3:])
        P.놓기()


def 시험_RoPE여럿(P, R, M):
    """ROPE(IMRoPE, 토큰 n 개 — 위치가 이어진 것): 토큰마다 위치 = 처음 + 토큰."""
    for n, rec in R.연산("ROPE"):
        x = R.입력(n, 0).astype(f32)
        T, H = x.shape[1], x.shape[2]
        pos = R.입력(n, 1).reshape(-1)[:T].astype(np.int64)
        assert np.array_equal(pos, pos[0] + np.arange(T)), "위치가 이어지지 않는다"
        op = rec["op_params"]
        fb = np.array([op[5]], np.int32).view(f32)[0]
        ts = f32(np.power(np.float64(fb), -2.0 / op[1]))
        D = x.shape[3]
        px, py = P.올림(x.reshape(-1)), P.올림(np.zeros(x.size, f32))
        if op[2] == 0:                                                  # 11단계 — 보통 RoPE(llama) · freq_factors
            pf = P.올림(로프인자(M, D))
            P.띄우기(f"라마RoPE보통여럿{D}", (T * H, 1), D, [(1, px), (1, pf), (1, py), (0, op[1]), (0, int(pos[0])), (0, H), (0, 비트(ts)),
                                                         (0, op[6]), (0, op[8])])
        else:
            P.띄우기("라마RoPE여럿" if D == 256 else f"라마RoPE여럿{D}", (T * H, 1), D, [(1, px), (1, py), (0, op[1]), (0, int(pos[0])), (0, H), (0, 비트(ts)), (0, op[6]), (0, op[8])])
        견주기(f"ROPE {T}×{H}×{D} 위치 {pos[0]}… ({n['name']})", P.받음(py, x.size), R.배열(rec))
        P.놓기()


def 시험_캐시여럿(P, R, M):
    """SET_ROWS(토큰 n 개 — 이어진 행): 새 행들의 반실수가 기록된 캐시의 그 행들과 같은가."""
    for n, rec in R.연산("SET_ROWS"):
        x = R.입력(n, 0).astype(f32).reshape(-1)
        행 = R.입력(n, 1).reshape(-1).astype(np.int64)
        assert np.array_equal(행, 행[0] + np.arange(행.size)), "행이 이어지지 않는다"
        캐시 = R.배열(rec).reshape(rec["ne"][1], rec["ne"][0])
        px, py = P.올림(x), P.올림(np.zeros(x.size, np.float16))
        P.띄우기("라마반실수쓰기", ((x.size + 255) // 256, 1), 256, [(1, px), (1, py), (0, x.size), (0, 0)])
        got = P.받음(py, x.size, np.float16)
        견주기(f"SET_ROWS f16 {x.size} 행 {행[0]}… ({n['name']})", got.astype(f32), 캐시[행[0]:행[-1] + 1].astype(f32))
        P.놓기()


# 10단계 — fattn-mma-f16.cuh 의 sm_8x 설정표에서 (머리 차원 D, 열) → (KV 묶음 nbatch_fa, SM 당 블록). 블록 수는
# cudaOccupancyMaxActiveBlocksPerMultiprocessor 와 같게(공유 메모리 — 2 단 KV + 가림 · 합치기 판, 블록마다 1 KiB 예약 — 와 레지스터의
# 작은 쪽: CPU 참조 fa_mma.py 의 occupancy). D 256 의 열 32 는 옮기지 않았다(판 8192 칸 — 공유판 32 KB 를 넘는다).
FA설정 = {(128, 8): (128, 1), (128, 16): (64, 2), (128, 32): (64, 2), (128, 64): (64, 2),
         (256, 8): (64, 1), (256, 16): (64, 1), (256, 64): (32, 2)}


def FA계획(T, nkv, Hk=4, D=256, gqa=4):
    """flash attention mma(fattn-mma-f16, GQA 4 → ncols2 4)의 설정과 stream-k 블록 수(sm_89 점유율): 토큰 2 → 열 8, 3 · 4 → 열 16,
    5 … 8 → 열 32, 9 개 이상 → 열 64(KV 묶음 · 점유는 FA설정). 블록 = min(점유 × SM, KV 묶음 × 타일), 타일 수의 배수로 내려 잃는 몫이
    5 % 이하면 내린 값. 블록 b 는 [b·W/블록, (b+1)·W/블록)(W = 타일 × KV 묶음). 돌려주는 것: 열, iterj, 타일 수, 조각 표, 타일 표."""
    n2 = 8 if gqa > 4 else 4 if gqa > 2 else 2 if gqa > 1 else 1      # 12단계 — ncols2(GQA 비)
    열 = 8 if T <= 8 // n2 else 16 if T <= 16 // n2 else 32 if T <= 32 // n2 else 64
    if D == 256 and (n2, gqa) != (4, 4):
        raise NotImplementedError(f"flash attention mma 의 D 256 · GQA {gqa} 는 옮기지 않았다")
    if (D, 열) not in FA설정:
        raise NotImplementedError(f"flash attention mma 의 D {D} · 토큰 {T}(열 {열}) 설정은 옮기지 않았다")
    nb, occ = FA설정[(D, 열)]
    n1 = 열 // n2
    iterj = (T + n1 - 1) // n1
    ntiles = iterj * Hk
    raw = min(occ * NSM, ((nkv + nb - 1) // nb) * ntiles)
    rounded = raw // ntiles * ntiles
    loss = 100 * (raw - rounded) // raw if rounded > 0 else 100
    grid = rounded if loss <= 5 else raw
    iter_k = nkv // nb
    W = ntiles * iter_k
    표, 타일표 = [], [[0, 0] for _ in range(ntiles)]
    for b in range(grid):
        k0, k1 = b * W // grid, (b + 1) * W // grid
        while k0 < k1:
            t = k0 // iter_k
            끝 = min(k1, (t + 1) * iter_k)
            if 타일표[t][1] == 0:
                타일표[t][0] = len(표)
            타일표[t][1] += 1
            표.append([t, k0 - t * iter_k, 끝 - t * iter_k])
            k0 = 끝
    return 열, iterj, ntiles, np.array(표, np.int32), np.array(타일표, np.int32), n2


def 시험_FA여럿(P, R, M):
    """FLASH_ATTN_EXT(토큰 n 개 — mma) 노드: 설정 · stream-k 조각 · 합치기까지."""
    for n, rec in R.연산("FLASH_ATTN_EXT"):
        q = R.입력(n, 0).astype(f32)[0]                                  # [머리, 토큰, D]
        T, D = q.shape[1], q.shape[2]
        if T == 1:
            continue
        q = np.ascontiguousarray(np.transpose(q, (1, 0, 2)))              # [토큰][머리][D]
        Kc = np.ascontiguousarray(np.transpose(R.입력(n, 1)[0], (1, 0, 2)))  # [kv수, kv머리, D] f16
        Vc = np.ascontiguousarray(np.transpose(R.입력(n, 2)[0], (1, 0, 2)))
        nkv = Kc.shape[0]
        가림 = R.입력(n, 3).astype(np.float16).reshape(-1, nkv)[:T]
        gqa = q.shape[1] // Kc.shape[1]
        열, iterj, ntiles, 표, 타일표, n2 = FA계획(T, nkv, Kc.shape[1], D, gqa)
        pq, pK, pV, pm = P.올림(q.reshape(-1)), P.올림(Kc.reshape(-1)), P.올림(Vc.reshape(-1)), P.올림(np.ascontiguousarray(가림))
        px, pmm, pr = P.올림(np.zeros(len(표) * 열 * D, f32)), P.올림(np.zeros(len(표) * 열, f32)), P.올림(np.zeros(len(표) * 열, f32))
        H = q.shape[1]
        p표, p타일, po = P.올림(표.reshape(-1)), P.올림(타일표.reshape(-1)), P.올림(np.zeros(T * H * D, f32))
        sfx, sfx2 = ("", "") if D == 256 else (f"_{D}", str(D))
        gq = [] if D == 256 else [(0, n2), (0, gqa)]                      # 12단계 — D 128 판의 GQA 인자
        P.띄우기(f"라마FA_MMA{열}{sfx}", (len(표), 1), 128, [(1, pq), (1, pK), (1, pV), (1, pm), (1, px), (1, pmm), (1, pr), (1, p표),
                                                             (0, T), (0, nkv), (0, iterj), (0, rec["op_params"][0]), (0, H)] + gq)
        P.띄우기(f"라마FA_MMA합치기{sfx2}", (ntiles, 1), D, [(1, px), (1, pmm), (1, pr), (1, p타일), (1, po), (0, T), (0, iterj), (0, 열),
                                                        (0, H)] + gq)
        견주기(f"FLASH_ATTN_EXT mma D {D} 열 {열} GQA {gqa}(n2 {n2}) 토큰 {T} kv {nkv} 조각 {len(표)}/타일 {ntiles} ({n['name']})", P.받음(po, T * H * D),
             R.배열(rec))
        P.놓기()


def 시험_GDN여럿(P, R, M):
    """GATED_DELTA_NET(토큰 n 개): 64 개 이상이면 덩어리 판(라마GDN덩어리), 적으면 한 걸음 판을 토큰 차례로. 출력과 새 상태."""
    눈금 = 비트((f32(1) / np.sqrt(f32(128))).astype(f32))
    for n, rec in R.연산("GATED_DELTA_NET"):
        q, k, v = (R.입력(n, i).astype(f32) for i in range(3))
        T = q.shape[1]
        if T == 1:
            continue
        q, k, v = (np.ascontiguousarray(a.reshape(-1)) for a in (q, k, v))
        g, beta = R.입력(n, 3).astype(f32).reshape(-1), R.입력(n, 4).astype(f32).reshape(-1)
        S = R.입력(n, 5).astype(f32).reshape(-1)
        H = g.size // T
        pq, pk, pv, pg, pb, pS, po = (P.올림(a) for a in (q, k, v, g, beta, S, np.zeros(T * H * 128, f32)))
        if T >= 64:
            P.띄우기("라마GDN덩어리", (H, 1), 256, [(1, pq), (1, pk), (1, pv), (1, pg), (1, pb), (1, pS), (1, po), (0, T), (0, H * 128), (0, 눈금),
                                                (0, H), (0, q.size // T // 128)])
            판이름 = "덩어리"
        else:
            hk = q.size // T // 128
            for t in range(T):
                P.띄우기("라마GDN한걸음", (8, H), 128, [(1, pq + 4 * t * hk * 128), (1, pk + 4 * t * hk * 128), (1, pv + 4 * t * H * 128),
                                                      (1, pg + 4 * t * H), (1, pb + 4 * t * H), (1, pS), (1, po + 4 * t * H * 128), (0, hk), (0, 눈금)])
            판이름 = "한 걸음 × 토큰"
        out = R.배열(rec).reshape(-1)
        견주기(f"GATED_DELTA_NET {판이름} 출력 {T}×{H}×128 ({n['name']})", P.받음(po, T * H * 128), out[:T * H * 128])
        견주기(f"GATED_DELTA_NET {판이름} 새 상태 {S.size} ({n['name']})", P.받음(pS, S.size), out[T * H * 128:T * H * 128 + S.size])
        P.놓기()


def 시험_임베딩(P, R, M):
    """토큰 임베딩: llama.cpp 는 CPU 에서 get_rows 를 하고 그 결과(model.input_embed)를 0 층 RMS_NORM 의 입력으로 올린다 — 그것과 견준다.
    토큰 번호는 --토큰 N (그 걸음에 넣은 토큰)."""
    if "--토큰" not in sys.argv:
        print("임베딩: --토큰 이 없어 건너뜀")
        return
    tok = int(sys.argv[sys.argv.index("--토큰") + 1])
    n, _ = R.노드("norm-0")
    t, dims, raw = M.원("token_embd.weight")
    K = dims[0]
    pw, py = P.올림(np.ascontiguousarray(raw)), P.올림(np.zeros(K, f32))
    P.띄우기("라마임베딩Q4_K", ((K + 255) // 256, 1), 256, [(1, pw), (1, py), (0, tok), (0, K // 256)])
    견주기(f"get_rows {gguf읽기.이름[t]} 토큰 {tok} (model.input_embed)", P.받음(py, K), R.입력(n, 0))
    P.놓기()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    값깃발 = ("--토큰", "--벌칸", "--장치", "--작업")
    인자 = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] not in 값깃발]
    ptx, 폴더, 모델 = 인자[:3]
    고른 = set(인자[3:])

    def 값(깃발, 기본=None):
        return sys.argv[sys.argv.index(깃발) + 1] if 깃발 in sys.argv else 기본
    if "--벌칸" in sys.argv:      # 3단계: ptx 자리에 SPIR-V 묶음(.spvs)
        P = 판V(값("--벌칸"), ptx, 값("--작업", r"E:\geulpu-work\xreplay\s3\tmp"), 값("--장치", "NVIDIA"))
    else:
        P = 판(ptx)
    R, M = 기록(폴더), gguf읽기.GGUF(모델)
    시험들 = {"RMS": 시험_RMS, "행렬곱": 시험_행렬곱, "원소": 시험_원소, "합성곱": 시험_합성곱, "GDN": 시험_GDN,
            "RoPE": 시험_RoPE, "캐시": 시험_캐시, "FA": 시험_FA, "임베딩": 시험_임베딩,
            "MMQ": 시험_MMQ, "행렬곱여럿": 시험_행렬곱여럿, "합성곱여럿": 시험_합성곱여럿, "RoPE여럿": 시험_RoPE여럿,
            "캐시여럿": 시험_캐시여럿, "FA여럿": 시험_FA여럿, "GDN여럿": 시험_GDN여럿}
    for 이름, f in 시험들.items():
        if not 고른 or 이름 in 고른:
            f(P, R, M)
    전체 = sum(r[1] for r in 결과)
    나쁨 = sum(r[2] for r in 결과)
    print(f"합계: 값 {전체} · 불일치 {나쁨}")
    print(f"모두 0: {나쁨 == 0}")
    return 0 if 나쁨 == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
