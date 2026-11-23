#!/usr/bin/env python3
"""동시 요청의 줄 판(정수줄여럿 — 행 2 … 8 의 생성 걸음, 2026-11-23)을 CPU 참조와 비트까지 — Qwen3.5 k-양자화 파일의 실제 가중치와
무작위 활성값, 행 1 … 8.
  k-양자화 정수줄여럿{끝손질}_{형식}: 정수계약.행렬곱K블라스 — 판26.py 와 같은 배치 · k-양자화 타일 판의 인자(행수 · 깊이 · 열수 · 척도폭 ·
    행시작 · 열시작 · 출력폭). 끝손질 "" · _잔차 · _SwiGLU정수(다음 자리 · 정보, --행합 이면 행 합도).
  고전 Q8_0 정수줄여럿{끝손질}: 정수계약.행렬곱블라스 — 판23.py 와 같은 배치 · 고전 타일 판의 인자. 파일의 β · α(4,096 × 32)와, 열 · 깊이의
    가장자리를 보려고 만든 Q8_0 행렬(열 1,000 — 32 의 배수가 아님 · 깊이 4,192 — 판 하나에 블록 셋만 남음).
모듈에 없는 판(공유 48 KB 에 들지 않는 형식 · 끝손질)은 건너뛰고 '없음' 으로 적는다.

  python -I 시험/판여럿.py <ptx> <Qwen3.5 k-양자화 gguf> <출력.json> [--빠른] [--행합] [--극단] [--스레드 N — 기본 256(엔진의 줄여럿스레드)]
"""
import ctypes
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
for _d in ("계약",):
    sys.path.insert(0, os.path.join(ROOT, _d))
sys.path.insert(0, HERE)
import 쿠다드라이버 as G          # noqa: E402
import 시험장치               # noqa: E402
import gguf읽기 as GR       # noqa: E402
import 정수계약 as C        # noqa: E402
from 머리128 import 자리바이트    # noqa: E402
from 판23 import 극단활성값, 극단척도, 양자읽기, 깔기    # noqa: E402
from 판26 import 엔진배치K, 극단날, 자리행합, 정보판, 다른칸    # noqa: E402

f32 = np.float32
u64, i64 = G.c_u64, G.c_i64

# (이름, 텐서들(열로 잇는다), 끝손질) — Qwen3.5-9B Q4_K_M: GDN 층 0(qkv Q5_K · z Q4_K · ssm_out Q5_K · down Q6_K) · 어텐션 층 3(q·k Q4_K ·
# v Q6_K · o Q4_K) · 층 7(down Q4_K) · gate·up(Q4_K — 출력 32 열마다 번갈아, 엔진과 같다)
K행렬들 = [("GDN qkv", ["blk.0.attn_qkv.weight"], ""), ("z", ["blk.0.attn_gate.weight"], ""),
          ("q·k", ["blk.3.attn_q.weight", "blk.3.attn_k.weight"], ""), ("v", ["blk.3.attn_v.weight"], ""),
          ("ssm_out", ["blk.0.ssm_out.weight"], "_잔차"), ("o", ["blk.3.attn_output.weight"], "_잔차"),
          ("down0", ["blk.0.ffn_down.weight"], "_잔차"), ("down7", ["blk.7.ffn_down.weight"], "_잔차"),
          ("gu", ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"], "_SwiGLU정수")]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    극단 = "--극단" in sys.argv
    자리수 = 1 if "--빠른" in sys.argv else 2
    행합 = "--행합" in sys.argv
    T = 256
    if "--스레드" in sys.argv:
        k = sys.argv.index("--스레드")
        T = int(sys.argv[k + 1])
        del sys.argv[k:k + 2]
    인자 = [a for a in sys.argv[1:] if a not in ("--극단", "--빠른", "--행합")]
    ptx, 파일, 출력 = 인자
    dr = 시험장치.드라이버()
    mod = dr.모듈(open(ptx, "rb").read())
    fn = {}

    def 있음(이름):
        f = ctypes.c_void_p()
        return dr.cu.cuModuleGetFunction(ctypes.byref(f), mod, ("_G" + 이름.encode("utf-8").hex()).encode()) == 0

    def 함수(이름):
        if 이름 not in fn:
            fn[이름] = dr.함수(mod, 이름)
        return fn[이름]

    def 띄우기(이름, grid, block, params):
        ps = [u64(p) if 참조 else i64(p) for 참조, p in params]
        args = (ctypes.c_void_p * len(ps))(*[ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in ps])
        r = dr.cu.cuLaunchKernel(함수(이름), grid[0], grid[1], 1, block, 1, 1, 0, None, args, None)
        assert r == 0, (이름, r)

    def up(b):
        b = b if isinstance(b, bytes) else np.ascontiguousarray(b).tobytes()
        p = dr.할당(len(b))
        dr.확인(dr.cu.cuMemcpyHtoD_v2(p, b, len(b)))
        return p

    def 놓기():
        dr.맞추기()
        for p in dr.할당들:
            dr.확인(dr.cu.cuMemFree_v2(u64(p)))
        dr.할당들 = []

    r = np.random.default_rng(2026_11_23)
    결과 = {"극단": 극단, "자리수": 자리수, "행합": 행합, "경우": [], "없음": []}
    g = GR.GGUF(파일)

    def 한경우(한, 다):
        결과["경우"].append(dict(한, **{"다른 칸": 다}))
        if sum(다.values()):
            print("다름!", 한, 다, flush=True)

    # ---- k-양자화 ----
    for 이름, 텐서들, e in K행렬들:
        원들 = [g.원(n) for n in 텐서들]
        ts = {t for t, _, _ in 원들}
        assert len(ts) == 1, (이름, ts)
        t = ts.pop()
        K = 원들[0][1][0]
        Ns = [dims[1] for _, dims, _ in 원들]
        N = sum(Ns)
        raw = np.concatenate([np.asarray(x) for _, _, x in 원들])
        if e == "_SwiGLU정수":
            F = Ns[0]
            from 판26 import 블록바이트
            rows = raw.reshape(N, K // 256 * 블록바이트[t])
            순 = np.empty(2 * F, np.int64)
            for j in range(F // 32):
                순[64 * j:64 * j + 32] = np.arange(32 * j, 32 * j + 32)
                순[64 * j + 32:64 * j + 64] = F + np.arange(32 * j, 32 * j + 32)
            raw = np.ascontiguousarray(rows[순]).reshape(-1)
        if 극단:
            raw = 극단날(t, raw, N, K, r)
        값, 척, w = 엔진배치K(t, raw, N, K)
        형식 = w[0]
        kn = f"정수줄여럿{e}_{형식}"
        if not 있음(kn):
            결과["없음"].append(kn)
            print("없음:", kn, flush=True)
            continue
        p값, p척 = up(값), up(척)
        p치 = up(np.zeros(N, f32))
        치 = np.zeros(N, f32)
        X8 = 극단활성값(r, 8, K) if 극단 else (r.standard_normal((8, K)) * 10.0 ** r.uniform(-1, 1, (8, 1))).astype(f32)
        y8 = C.행렬곱K블라스(X8, w, 치, 자리수=자리수)
        Np = (N + 63) // 64 * 64
        W = 64 if e == "_SwiGLU정수" else 32
        잔8 = (r.standard_normal((8, N)) * 3.0).astype(f32)
        for m in range(1, 9):
            바이트m, M0m = 자리바이트(X8[:m], 자리수)
            p자m, p정m = up(바이트m), up(정보판(M0m, 자리행합(X8[:m], 자리수), 64))
            앞 = [(1, p자m), (1, p정m), (1, p값), (1, p척), (1, p치)]
            뒤 = [(0, m), (0, K), (0, N), (0, Np), (0, 0), (0, 0), (0, N)]
            grid = ((N + W - 1) // W, 1)
            if e == "_SwiGLU정수":
                gcol = np.concatenate([np.arange(64 * j, 64 * j + 32) for j in range(N // 64)])
                h8 = C.swiglu(y8[:m][:, gcol], y8[:m][:, gcol + 32])
                참자리, 참정보 = 자리바이트(h8, 자리수)
                p다자, p다정 = dr.할당(자리수 * m * (N // 2)), dr.할당(2 * (N // 64) * 64 * 4)
                띄우기(kn, grid, T, 앞 + [(1, p다정), (1, p다자)] + 뒤)
                dr.맞추기()
                글정 = dr.내리기(p다정, np.empty((2 * (N // 64), 64), np.int32))
                다 = {"자리": 다른칸(dr.내리기(p다자, np.empty((자리수, m, N // 2), np.int8)), 참자리),
                     "정보": 다른칸(글정[:N // 64, :m].T, 참정보)}
                if 행합:
                    다["행합"] = 다른칸(글정[N // 64:, :m].T, 자리행합(h8, 자리수))
            else:
                p출 = dr.할당(m * N * 4)
                if e == "_잔차":
                    dr.확인(dr.cu.cuMemcpyHtoD_v2(p출, np.ascontiguousarray(잔8[:m]).ctypes.data, m * N * 4))
                    띄우기(kn, grid, T, 앞 + [(1, p출), (1, p출)] + 뒤)
                    참 = C.더(잔8[:m], y8[:m])
                else:
                    띄우기(kn, grid, T, 앞 + [(1, p출)] + 뒤)
                    참 = y8[:m]
                dr.맞추기()
                다 = {"출력": 다른칸(dr.내리기(p출, np.empty((m, N), f32)), 참)}
            한경우({"가중치": 이름, "형식": 형식, "판": kn, "행": m, "K": int(K), "N": int(N)}, 다)
        print(f"{이름} ({형식}) {kn} 행 1 … 8 끝 — 다른 칸 {sum(sum(c['다른 칸'].values()) for c in 결과['경우'] if c['판'] == kn)}", flush=True)
        놓기()

    # ---- 고전 Q8_0 ----
    Q8들 = []
    for 이름, n in (("β", "blk.0.ssm_beta.weight"), ("α", "blk.0.ssm_alpha.weight")):
        q, d, 형식 = 양자읽기(g, n)
        assert 형식 == "Q8_0", (n, 형식)
        Q8들.append((이름, q, d))
    Nx, Kx = 1000, 4192                                         # 가장자리: 열이 32 의 배수가 아니고, 마지막 판에 블록 셋
    Q8들.append(("만든 Q8_0", r.integers(-127, T, (Nx, Kx)).astype(np.int8),
                (r.standard_normal((Nx, Kx // 32)) * 0.01).astype(np.float16).astype(f32)))
    for 이름, q, d in Q8들:
        if 극단:
            d = 극단척도(r, d)
        N, K = q.shape
        값, 척 = 깔기(q, d, "Q8_0")
        p값, p척 = up(값), up(척)
        치 = np.zeros(N, f32)
        p치 = up(치)
        X8 = 극단활성값(r, 8, K) if 극단 else (r.standard_normal((8, K)) * 10.0 ** r.uniform(-1, 1, (8, 1))).astype(f32)
        y8 = C.행렬곱블라스(X8, q, d, 치, 자리수=자리수)
        Np = (N + 63) // 64 * 64
        잔8 = (r.standard_normal((8, N)) * 3.0).astype(f32)
        for e in ("", "_잔차"):
            kn = f"정수줄여럿{e}"
            if not 있음(kn):
                결과["없음"].append(kn)
                print("없음:", kn, flush=True)
                continue
            for m in range(1, 9):
                바이트m, M0m = 자리바이트(X8[:m], 자리수)
                p자m, p정m = up(바이트m), up(정보판(M0m, 자리행합(X8[:m], 자리수), 64))
                앞 = [(1, p자m), (1, p정m), (1, p값), (1, p척), (1, p치)]
                뒤 = [(0, m), (0, K), (0, N), (0, Np), (0, 0), (0, 0)]
                grid = ((N + 31) // 32, 1)
                p출 = dr.할당(m * N * 4)
                if e == "_잔차":
                    dr.확인(dr.cu.cuMemcpyHtoD_v2(p출, np.ascontiguousarray(잔8[:m]).ctypes.data, m * N * 4))
                    띄우기(kn, grid, T, 앞 + [(1, p출), (1, p출)] + 뒤)
                    참 = C.더(잔8[:m], y8[:m])
                else:
                    띄우기(kn, grid, T, 앞 + [(1, p출)] + 뒤)
                    참 = y8[:m]
                dr.맞추기()
                다 = {"출력": 다른칸(dr.내리기(p출, np.empty((m, N), f32)), 참)}
                한경우({"가중치": 이름, "형식": "Q8_0", "판": kn, "행": m, "K": int(K), "N": int(N)}, 다)
            print(f"{이름} (Q8_0) {kn} 행 1 … 8 끝", flush=True)
        놓기()
    결과["모두 0"] = all(v == 0 for c in 결과["경우"] for v in c["다른 칸"].values())
    print("경우", len(결과["경우"]), "· 모두 0:", 결과["모두 0"], "· 없음:", 결과["없음"], flush=True)
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
