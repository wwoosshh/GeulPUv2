#!/usr/bin/env python3
"""MoE 재기 — 이 PC 의 대역폭: PCIe(고정 메모리 → GPU, 전문가 크기 덩이와 큰 덩이), GPU 안 복사, CPU 의 RAM 읽기.

  python 대역폭.py [전문가 바이트=1769472]

CUDA 드라이버(nvcuda.dll)만 쓴다. 시간은 cuEvent 로(복사는 같은 흐름에 차례로).
"""
import ctypes
import os
import sys
import threading
import time

import numpy as np

EXP = int(sys.argv[1]) if len(sys.argv) > 1 else 1769472
cu = ctypes.WinDLL("nvcuda.dll")
c_u64 = ctypes.c_uint64


def 확인(r):
    if r != 0:
        raise RuntimeError(f"CUDA 오류 {r}")


확인(cu.cuInit(0))
ctx = ctypes.c_void_p()
확인(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), 0))
확인(cu.cuCtxSetCurrent(ctx))
cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(c_u64), ctypes.c_size_t]
cu.cuMemHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
cu.cuMemcpyHtoDAsync_v2.argtypes = [c_u64, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
cu.cuMemcpyDtoDAsync_v2.argtypes = [c_u64, c_u64, ctypes.c_size_t, ctypes.c_void_p]
cu.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
cu.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
cu.cuEventSynchronize.argtypes = [ctypes.c_void_p]
cu.cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]

N = 1 << 30
d1, d2 = c_u64(), c_u64()
확인(cu.cuMemAlloc_v2(ctypes.byref(d1), N))
확인(cu.cuMemAlloc_v2(ctypes.byref(d2), N))
h = ctypes.c_void_p()
확인(cu.cuMemHostAlloc(ctypes.byref(h), N, 0))
ctypes.memset(h, 1, N)
pageable = np.ones(N, np.uint8)
e0, e1 = ctypes.c_void_p(), ctypes.c_void_p()
확인(cu.cuEventCreate(ctypes.byref(e0), 0))
확인(cu.cuEventCreate(ctypes.byref(e1), 0))


def 재기(f, 바이트, 번=3):
    best = 1e9
    for _ in range(번):
        확인(cu.cuEventRecord(e0, None))
        f()
        확인(cu.cuEventRecord(e1, None))
        확인(cu.cuEventSynchronize(e1))
        ms = ctypes.c_float()
        확인(cu.cuEventElapsedTime(ctypes.byref(ms), e0, e1))
        best = min(best, ms.value)
    return 바이트 / (best / 1e3) / 1e9, best


k = N // EXP


def 전문가들(src):
    for i in range(k):
        확인(cu.cuMemcpyHtoDAsync_v2(c_u64(d1.value + i * EXP), ctypes.c_void_p(src + i * EXP), EXP, None))


print(f"PCIe 고정 · 전문가 크기 덩이 {EXP} B × {k}: {재기(lambda: 전문가들(h.value), k * EXP)[0]:.2f} GB/s")
print(f"PCIe 고정 · 1 GiB 한 번: {재기(lambda: 확인(cu.cuMemcpyHtoDAsync_v2(d1, h, N, None)), N)[0]:.2f} GB/s")
print(f"PCIe 페이지 · 전문가 크기 덩이: {재기(lambda: 전문가들(pageable.ctypes.data), k * EXP, 2)[0]:.2f} GB/s")
gb, ms = 재기(lambda: 확인(cu.cuMemcpyDtoDAsync_v2(d2, d1, N, None)), 2 * N)
print(f"GPU 안 복사(읽기 + 쓰기) 1 GiB: {gb:.1f} GB/s")

# CPU 의 RAM 읽기 — 스레드 여럿이 큰 배열의 다른 부분을 더한다(numpy 는 더하는 동안 GIL 을 놓는다)
a = np.ones((8 << 30) // 8, np.float64)          # 8 GiB
for 스레드 in (1, 8, 16, 20):
    조각 = np.array_split(a, 스레드)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        ts = [threading.Thread(target=np.add.reduce, args=(c,)) for c in 조각]
        [t.start() for t in ts]
        [t.join() for t in ts]
        best = min(best, time.perf_counter() - t0)
    print(f"RAM 읽기 · 스레드 {스레드}: {a.nbytes / best / 1e9:.1f} GB/s")
