#!/usr/bin/env python3
"""생성의 행렬곱(문서/생성행렬곱.md G0) — k-양자화 1행 GEMV 커널(`정수줄하나{에필로그}_{형식}`)을 하나씩 모양마다 잰다.

  python -I -X utf8 재기/줄하나재기.py cut <모듈.gl> <출력.gl>
      커널생성이 낸 모듈 소스에서 머리(외부 선언 · 표시)와 정수줄하나 커널(Q4_K · Q5_K · Q6_K · 있으면 IQ4_XS × 에필로그 셋)만 떼어 낸다 —
      PTX 뒷단으로 작은 모듈을 만들어 JIT 를 짧게(커널의 PTX 는 온 모듈의 것과 같다).
  python -I -X utf8 재기/줄하나재기.py run <모듈.ptx> [--case 형식:에필로그:KxN[:W[:T]] …] [--path dram|l2|both]
                                       [--launches 100] [--repeats 7] [--digits 1|2] [--peak-gbps G] [--json 출력.json]
      에필로그는 none · residual · swiglu. W 는 블록이 맡는 열(기본 32, swiglu 64), T 는 블록의 스레드(기본: PTX 의 .maxntid).
      --case 를 주지 않으면 Llama-3.1-8B 의 모양 여섯. 실행 --launches 개를 CUDA 그래프 하나로 잡아(생성 스텝과 같이 틈이 작다)
      그래프를 거듭 띄워 잰다 — 실행 하나의 µs 는 반복 --repeats 번의 중앙값.
      - dram: 같은 모양의 행렬을 L2 의 세 배 넘게 여러 벌 두고 실행마다 다음 벌을 — 가중치를 DRAM 에서 읽는다.
      - l2: 행렬 한 벌을 거듭 — L2 의 60 % 안에 드는 모양만. 대역폭이 SM 쪽에 묶인 커널의 빠르기를 본다.
      가중치 · 스케일은 정한 씨앗의 무작위 바이트, 활성값의 자리는 무작위 정수, 매직 넘버는 지수 0 근처의 블록마다 값(시간은 값에 따라
      갈리지 않는다). SM 클럭은 잴 때 NVML 로 읽은 중앙값(못 읽으면 장치의 최대 클럭).
"""
import argparse
import ctypes
import json
import os
import re
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "시험"))
import 쿠다드라이버 as G     # noqa: E402

u64, i64 = G.c_u64, G.c_i64
형식들 = ("Q4_K", "Q5_K", "Q6_K")
에필로그들 = {"none": "", "residual": "_잔차", "swiglu": "_SwiGLU정수"}
높은바이트 = {"Q4_K": 0, "Q5_K": 4, "Q6_K": 8}           # 블록(32 칸) · 열마다 값 16 바이트 뒤의 높은 비트
파일바이트 = {"Q4_K": 144, "Q5_K": 176, "Q6_K": 210}     # 256 칸마다 — 커널이 읽는 바이트(엔진 배치도 같다)
기본모양 = ["Q4_K:none:4096x6144", "Q4_K:residual:4096x4096", "Q4_K:swiglu:4096x28672", "Q4_K:residual:14336x4096",
           "Q6_K:residual:14336x4096", "Q6_K:none:4096x1024"]


def cut(원본, 출력):
    줄들 = open(원본, encoding="utf-8").read().split("\n")
    정의 = re.compile(r"^\[.*\]는( -> .*)? \{$")

    def 주석위(i):                       # 정의 줄 i 바로 위에 붙은 주석 줄들의 첫 줄
        while i > 0 and 줄들[i - 1].startswith("(*"):
            i -= 1
        return i

    첫 = next(i for i, s in enumerate(줄들) if 정의.match(s))
    머리 = 줄들[:주석위(첫)]
    몸 = []
    for 형식 in 형식들 + ("IQ4_XS",):
        for e in 에필로그들.values():
            이름 = f"정수줄하나{e}_{형식}"
            i = next((i for i, s in enumerate(줄들) if 정의.match(s) and s.endswith(f" {이름}]는 {{")), None)
            if i is None:
                continue
            j = next(j for j in range(i + 1, len(줄들)) if 줄들[j] == "}")
            몸 += 줄들[주석위(i):j + 1] + [""]
    with open(출력, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(머리 + 몸))
    print(f"{출력}: 머리 {len(머리)} 줄 · 커널 {sum(1 for s in 몸 if 정의.match(s))} 개 ({len(몸)} 줄)")


class 재기:
    def __init__(self, ptx):
        self.dr = G.드라이버()
        cu = self.dr.cu
        P = ctypes.POINTER
        cu.cuEventCreate.argtypes = [P(ctypes.c_void_p), ctypes.c_uint]
        cu.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cu.cuEventSynchronize.argtypes = [ctypes.c_void_p]
        cu.cuEventElapsedTime.argtypes = [P(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
        cu.cuStreamCreate.argtypes = [P(ctypes.c_void_p), ctypes.c_uint]
        cu.cuStreamBeginCapture_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
        cu.cuStreamEndCapture.argtypes = [ctypes.c_void_p, P(ctypes.c_void_p)]
        cu.cuGraphInstantiateWithFlags.argtypes = [P(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_ulonglong]
        cu.cuGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cu.cuGraphExecDestroy.argtypes = [ctypes.c_void_p]
        cu.cuGraphDestroy.argtypes = [ctypes.c_void_p]
        cu.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
        cu.cuDeviceGetAttribute.argtypes = [P(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        cu.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        self.cu = cu
        self.ptx = open(ptx, "rb").read()
        t0 = time.time()
        self.mod = self.dr.모듈(self.ptx)
        self.JIT초 = time.time() - t0
        self.스트림 = ctypes.c_void_p()
        self.dr.확인(cu.cuStreamCreate(ctypes.byref(self.스트림), 1))       # CU_STREAM_NON_BLOCKING
        self.시작, self.끝 = ctypes.c_void_p(), ctypes.c_void_p()
        self.dr.확인(cu.cuEventCreate(ctypes.byref(self.시작), 0))
        self.dr.확인(cu.cuEventCreate(ctypes.byref(self.끝), 0))
        이름 = ctypes.create_string_buffer(256)
        self.dr.확인(cu.cuDeviceGetName(이름, 256, 0))
        self.장치 = 이름.value.decode()
        self.SM = self.속성(16)                 # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
        self.최대클럭 = self.속성(13) / 1000    # CLOCK_RATE (kHz) → MHz
        self.L2 = self.속성(38)                 # L2_CACHE_SIZE
        self.클럭읽기 = nvml클럭()

    def 속성(self, a):
        v = ctypes.c_int()
        self.dr.확인(self.cu.cuDeviceGetAttribute(ctypes.byref(v), a, 0))
        return v.value

    def 스레드수(self, 이름):
        """PTX 의 .entry 에 붙은 .maxntid(생성기의 실행한도)"""
        hx = ("_G" + 이름.encode("utf-8").hex()).encode()
        i = self.ptx.index(b".entry " + hx + b"(")
        m = re.compile(rb"\.maxntid\s+(\d+)").search(self.ptx, i, i + 4000)
        return int(m.group(1)) if m else 128

    def 그래프(self, fn, grid, T, 인자들):
        """인자 묶음마다 실행 하나 — 그래프 하나로 잡는다"""
        cu, s = self.cu, self.스트림
        self.dr.확인(cu.cuStreamBeginCapture_v2(s, 0))             # CU_STREAM_CAPTURE_MODE_GLOBAL
        보관 = []
        for ps in 인자들:
            arr = (ctypes.c_void_p * len(ps))(*[ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in ps])
            보관.append(arr)
            self.dr.확인(cu.cuLaunchKernel(fn, grid, 1, 1, T, 1, 1, 0, s, arr, None))
        g, ex = ctypes.c_void_p(), ctypes.c_void_p()
        self.dr.확인(cu.cuStreamEndCapture(s, ctypes.byref(g)))
        self.dr.확인(cu.cuGraphInstantiateWithFlags(ctypes.byref(ex), g, 0))
        return g, ex

    def 띄워재기(self, ex, 번):
        cu, s = self.cu, self.스트림
        self.dr.확인(cu.cuEventRecord(self.시작, s))
        for _ in range(번):
            self.dr.확인(cu.cuGraphLaunch(ex, s))
        self.dr.확인(cu.cuEventRecord(self.끝, s))
        self.dr.확인(cu.cuEventSynchronize(self.끝))
        ms = ctypes.c_float()
        self.dr.확인(cu.cuEventElapsedTime(ctypes.byref(ms), self.시작, self.끝))
        return ms.value


def nvml클럭():
    """SM 클럭(MHz)을 읽는 함수 — NVML 이 없으면 None"""
    이름들 = (["nvml.dll", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"] if os.name == "nt"
            else ["libnvidia-ml.so.1"])
    for 이름 in 이름들:
        try:
            lib = ctypes.WinDLL(이름) if os.name == "nt" else ctypes.CDLL(이름)
        except OSError:
            continue
        h = ctypes.c_void_p()
        if lib.nvmlInit_v2() != 0 or lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) != 0:
            return None

        def 읽기():
            c = ctypes.c_uint()
            return c.value if lib.nvmlDeviceGetClockInfo(h, 1, ctypes.byref(c)) == 0 else None    # NVML_CLOCK_SM
        return 읽기
    return None


def 모양풀기(글):
    p = 글.split(":")
    형식, 에필로그 = p[0], p[1]
    assert 형식 in 형식들 and 에필로그 in 에필로그들, 글
    K, N = (int(x) for x in p[2].lower().split("x"))
    W = int(p[3]) if len(p) > 3 else (64 if 에필로그 == "swiglu" else 32)
    T = int(p[4]) if len(p) > 4 else None
    assert K % 256 == 0 and N % W == 0, 글
    return 형식, 에필로그, K, N, W, T


def 잴것(m, 형식, 에필로그, K, N, W, T, 길, a, rng):
    dr = m.dr
    e = 에필로그들[에필로그]
    이름 = f"정수줄하나{e}_{형식}"
    fn = dr.함수(m.mod, 이름)
    T = T or m.스레드수(이름)
    KB, KS = K // 32, K // 256
    Np = (N + 63) // 64 * 64
    값바이트 = KB * N * (16 + 높은바이트[형식])
    척바이트 = KS * N * 16 + (KS * Np * 2 if 형식 == "Q6_K" else 0)
    읽는바이트 = K * N * 파일바이트[형식] // 256
    if 길 == "l2":
        if 값바이트 + 척바이트 > 0.6 * m.L2:
            return None
        벌 = 1
    else:
        벌 = max(2, -(-3 * m.L2 // (값바이트 + 척바이트)))
    # 활성값: 자리 [자리수][K] (블록마다 낱말 차례 — 값은 무작위라 차례는 상관없다) · 정보 [블록][64] 의 0 열에 M0 (뒤 절반은 행 합)
    자리 = rng.integers(-100, 101, a.digits * K, dtype=np.int8)
    E = rng.integers(-4, 5, KB)
    정보 = np.zeros((2 * KB, 64), np.int32)
    정보[:KB, 0] = ((153 - 8 * a.digits + E) << 23) + 4194304
    p자, p정 = dr.할당(자리.nbytes), dr.할당(정보.nbytes)
    dr.올리기(p자, 자리)
    dr.올리기(p정, 정보)
    p치 = dr.할당(N * 4)
    벌들 = []
    for _ in range(벌):
        p값, p척 = dr.할당(값바이트), dr.할당(척바이트)
        dr.올리기(p값, rng.integers(0, 256, 값바이트, dtype=np.uint8))
        척 = rng.integers(0, 256, 척바이트, dtype=np.uint8)
        척.view(np.uint16)[:KS * N * 8].reshape(KS * N, 8)[:, :2] = np.float16(0.01).view(np.uint16)  # d · dmin 은 유한한 값
        dr.올리기(p척, 척)
        벌들.append((p값, p척))
    if 에필로그 == "swiglu":
        p다정, p다자 = dr.할당(2 * (N // 64) * 64 * 4), dr.할당(a.digits * N // 2)
        뒤 = lambda: [u64(p다정), u64(p다자)]             # noqa: E731
    elif 에필로그 == "residual":
        p잔, p출 = dr.할당(N * 4), dr.할당(N * 4)
        뒤 = lambda: [u64(p잔), u64(p출)]               # noqa: E731
    else:
        p출 = dr.할당(N * 4)
        뒤 = lambda: [u64(p출)]                         # noqa: E731
    인자들 = [[u64(p자), u64(p정), u64(벌들[k % 벌][0]), u64(벌들[k % 벌][1]), u64(p치)] + 뒤() + [i64(1), i64(K), i64(N)]
            for k in range(a.launches)]
    grid = N // W
    g, ex = m.그래프(fn, grid, T, 인자들)
    t0 = time.time()
    while time.time() - t0 < 0.3:                      # 데우기 — 클럭이 오르고, L2 길은 행렬이 L2 에 오른다
        m.띄워재기(ex, 4)
    한번 = m.띄워재기(ex, 1)
    번 = max(1, int(0.05 / max(한번 / 1000, 1e-6)))     # 반복 하나가 50 ms 남짓
    클럭들, 멈춤 = [], threading.Event()

    def 표본():
        while not 멈춤.is_set():
            c = m.클럭읽기()
            if c:
                클럭들.append(c)
            time.sleep(0.005)
    th = None
    if m.클럭읽기:
        th = threading.Thread(target=표본, daemon=True)
        th.start()
    µs들 = [m.띄워재기(ex, 번) * 1000 / (번 * a.launches) for _ in range(a.repeats)]
    멈춤.set()
    if th:
        th.join()
    m.cu.cuGraphExecDestroy(ex)
    m.cu.cuGraphDestroy(g)
    dr.맞추기()
    for p in dr.할당들:
        dr.확인(dr.cu.cuMemFree_v2(u64(p)))
    dr.할당들 = []
    µs = float(np.median(µs들))
    클럭 = float(np.median(클럭들)) if 클럭들 else m.최대클럭
    GBs = 읽는바이트 / µs / 1000
    return dict(format=형식, epilogue=에필로그, K=K, N=N, W=W, T=T, path=길, copies=벌, bytes=읽는바이트, us=µs,
                us_min=float(min(µs들)), gbps=GBs, clock_mhz=클럭, clock_sampled=bool(클럭들),
                bytes_per_clk_sm=읽는바이트 / (µs * 클럭) / m.SM, grid=grid, blocks_per_sm=grid / m.SM)


def run(a):
    m = 재기(a.ptx)
    print(f"장치 {m.장치} · SM {m.SM} · 최대 클럭 {m.최대클럭:.0f} MHz · L2 {m.L2 / 2**20:.0f} MiB · "
          f"NVML {'있음' if m.클럭읽기 else '없음'} · JIT {m.JIT초:.1f} 초", flush=True)
    rng = np.random.default_rng(1)
    길들 = ["dram", "l2"] if a.path == "both" else [a.path]
    결과 = []
    for 글 in a.case or 기본모양:
        형식, 에필로그, K, N, W, T = 모양풀기(글)
        for 길 in 길들:
            r = 잴것(m, 형식, 에필로그, K, N, W, T, 길, a, rng)
            if r is None:
                print(f"{형식:5} {에필로그:8} {K}x{N:<6} {길:4}  — L2 의 60 % 를 넘어 건너뜀", flush=True)
                continue
            몫 = f" ({r['gbps'] / a.peak_gbps * 100:4.1f} %)" if a.peak_gbps else ""
            print(f"{형식:5} {에필로그:8} {K}x{N:<6} {길:4} {r['bytes'] / 1e6:7.2f} MB × {r['copies']:2} 벌  "
                  f"{r['us']:8.2f} µs  {r['gbps']:7.1f} GB/s{몫}  {r['bytes_per_clk_sm']:5.2f} B/클럭/SM "
                  f"@{r['clock_mhz']:.0f} MHz{'' if r['clock_sampled'] else '(최대)'}  그리드 {r['grid']} "
                  f"({r['blocks_per_sm']:.1f}/SM) · {r['T']} 스레드", flush=True)
            결과.append(r)
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(dict(device=m.장치, sm=m.SM, max_clock_mhz=m.최대클럭, l2=m.L2, ptx=os.path.basename(a.ptx),
                           launches=a.launches, repeats=a.repeats, digits=a.digits, results=결과), f, ensure_ascii=False, indent=1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cut")
    c.add_argument("module_gl")
    c.add_argument("out_gl")
    r = sub.add_parser("run")
    r.add_argument("ptx")
    r.add_argument("--case", action="append")
    r.add_argument("--path", choices=["dram", "l2", "both"], default="both")
    r.add_argument("--launches", type=int, default=100)
    r.add_argument("--repeats", type=int, default=7)
    r.add_argument("--digits", type=int, choices=[1, 2], default=1)
    r.add_argument("--peak-gbps", type=float)
    r.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "cut":
        cut(a.module_gl, a.out_gl)
    else:
        run(a)


if __name__ == "__main__":
    main()
