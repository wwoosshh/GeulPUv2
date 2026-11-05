#!/usr/bin/env python3
"""llama.cpp 재현 4단계(문서/라마재현.md) — GPU 없는 시험: llama-server 의 로그 확률 흉내(엔진/라마로그확률.gl)가 파이썬 참조와 비트까지
같은가. 참조는 서버 코드(server-common.cpp get_token_probabilities · server-task.cpp logarithm)를 그대로 옮긴 것 — MSVC STL 의
partial_sort 자리 옮김, 같은 C 런타임의 expf · logf(윈도 ucrtbase · 리눅스 libm, ctypes).

  python -I -X utf8 시험/라마영수증.py <시험_라마로그확률 실행 파일>
      정한 무작위 로짓(같은 값 · 아주 작은 값 · 모두 같은 줄 · n_probs 1 · 5 · 20)으로 글 쪽과 참조를 견준다(빠른 점검 4u).
  python -I -X utf8 시험/라마영수증.py <실행 파일> --서버 <로짓.bin> <서버 응답.json>
      하네스의 로짓(llama.cpp 가 낸 것)으로 셈한 글 쪽 값을 llama-server 가 실제로 돌려준 로그 확률(completion_probabilities)과 견준다.
마지막 줄 "모두 같음: True|False".
"""
import ctypes
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

F32 = np.float32
_c = ctypes.CDLL("ucrtbase.dll" if os.name == "nt" else "libm.so.6")
for _n in ("expf", "logf"):
    getattr(_c, _n).restype = ctypes.c_float
    getattr(_c, _n).argtypes = [ctypes.c_float]


def _구멍(a, hole, bottom, val):
    """MSVC STL _Pop_heap_hole_by_index — lt(x, y) = x.logit > y.logit"""
    top = idx = hole
    max_nl = (bottom - 1) >> 1
    while idx < max_nl:
        idx = 2 * idx + 2
        if a[idx][0] > a[idx - 1][0]:
            idx -= 1
        a[hole] = a[idx]
        hole = idx
    if idx == max_nl and bottom % 2 == 0:
        a[hole] = a[bottom - 1]
        hole = bottom - 1
    idx = (hole - 1) >> 1
    while top < hole and a[idx][0] > val[0]:
        a[hole] = a[idx]
        hole = idx
        idx = (hole - 1) >> 1
    a[hole] = val


def 부분정렬(a, mid):
    """std::partial_sort(a, a + mid, a + n, 큰 차례) — MSVC STL"""
    if mid == 0:
        return
    hole = mid >> 1
    while hole > 0:
        hole -= 1
        _구멍(a, hole, mid, a[hole])
    for nxt in range(mid, len(a)):
        if a[nxt][0] > a[0][0]:
            val = a[nxt]
            a[nxt] = a[0]
            _구멍(a, 0, mid, val)
    last = mid
    while last >= 2:
        last -= 1
        val = a[last]
        a[last] = a[0]
        _구멍(a, 0, last, val)


def 로그(p):
    return F32(np.finfo(F32).min) if p == 0 else F32(_c.logf(float(p)))


def 참조(로짓, n, 고른):
    a = [(F32(v), i) for i, v in enumerate(로짓)]
    부분정렬(a, min(n, len(a)))
    mx = a[0][0]
    ps = np.empty(len(a), F32)
    cum = F32(0)
    for i, (l, _) in enumerate(a):
        p = F32(_c.expf(float(F32(l - mx))))
        ps[i] = p
        cum = F32(cum + p)
    ps = (ps / cum).astype(F32)
    상위 = [(a[k][1], int(로그(ps[k]).view(np.uint32))) for k in range(min(n, len(a)))]
    j = next((i for i, (_, t) in enumerate(a) if t == 고른), None)
    return (int(로그(ps[j]).view(np.uint32)) if j is not None else -1), 상위


def 글쪽(exe, 행들, n, 고른들):
    V, R = 행들.shape[1], 행들.shape[0]
    with tempfile.TemporaryDirectory() as td:
        i, o = os.path.join(td, "in.bin"), os.path.join(td, "out.bin")
        with open(i, "wb") as f:
            f.write(np.array([V, R, n], np.int64).tobytes())
            f.write(np.ascontiguousarray(행들, F32).tobytes())
            f.write(np.array(고른들, np.int64).tobytes())
        r = subprocess.run([exe, i, o], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"시험_라마로그확률 실패({r.returncode}): {r.stdout}{r.stderr}")
        d = np.fromfile(o, np.int64).reshape(R, 1 + 2 * n)
    return [(int(x[0]) & 0xffffffff if x[0] != -1 else -1, [(int(x[1 + 2 * k]), int(x[2 + 2 * k]) & 0xffffffff) for k in range(n)])
            for x in d]


def 정한로짓(V, 씨):
    g = np.random.default_rng(씨)
    줄 = [g.normal(0, 4, V), np.round(g.normal(0, 3, V) * 2) / 2, g.normal(0, 4, V), np.zeros(V), g.normal(0, 1, V)]
    줄[2][: V // 3] = -1e30                                        # exp 가 0 이 되는 값 — 로그 확률은 짧은실수의 가장 작은 값
    줄[4][7] = 200.0                                               # 하나만 크면 나머지 확률이 모두 0
    return np.array(줄, F32)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    exe = os.path.abspath(sys.argv[1])
    나쁨 = 전체 = 0
    if "--서버" in sys.argv:
        k = sys.argv.index("--서버")
        run = json.load(open(sys.argv[k + 2], encoding="utf-8"))
        cps = run["response"]["completion_probabilities"]
        n = len(cps[0]["top_logprobs"])
        행들 = np.fromfile(sys.argv[k + 1], F32).reshape(-1, 248320)[: len(cps)]
        고른들 = [c["id"] for c in cps]
        for r, (b, 상위) in enumerate(글쪽(exe, 행들, n, 고른들)):
            c = cps[r]
            서버 = [(e["id"], int(F32(e["logprob"]).view(np.uint32))) for e in c["top_logprobs"]]
            d = sum(1 for g, s in zip(상위, 서버) if g != s) + (0 if b == int(F32(c["logprob"]).view(np.uint32)) else 1)
            전체 += n + 1
            나쁨 += d
            if d and 나쁨 < 40:
                print(f"걸음 {r}: 다름 {d} — 글 {상위[:3]} · 서버 {서버[:3]}")
        print(f"서버 응답과: 걸음 {len(cps)} · 값 {전체} · 다름 {나쁨}")
    else:
        for V, 씨, n in ((2000, 1, 20), (2000, 2, 5), (2000, 3, 1), (20011, 4, 20)):
            행들 = 정한로짓(V, 씨)
            고른들 = [int(np.argmax(행들[0])), 5, 0, 3, 7]
            got = 글쪽(exe, 행들, n, 고른들)
            for r in range(len(행들)):
                want = 참조(행들[r], n, 고른들[r])
                d = (0 if got[r][0] == want[0] else 1) + sum(1 for g, w in zip(got[r][1], want[1]) if g != w)
                전체 += n + 1
                나쁨 += d
                if d:
                    print(f"V {V} 씨 {씨} n {n} 줄 {r}: 다름 {d} — 글 {got[r][1][:3]} · 참조 {want[1][:3]}")
        print(f"정한 로짓: 값 {전체} · 다름 {나쁨}")
    print(f"모두 같음: {나쁨 == 0}")
    return 0 if 나쁨 == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
