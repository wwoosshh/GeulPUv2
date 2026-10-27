#!/usr/bin/env python3
"""텐서 코어 최고치 재기(#140 의 첫 측정) — 손으로 쓴 PTX 를 드라이버로 불러 한 명령의 처리량(TOPS · TFLOPS)을 잰다.

  python3 mmabench.py [--json 결과.json]

  mma_s8    mma.sync m16n8k32 s8·s8→s32 (sm_80 — GeulPU 의 int8 행렬곱이 쓰는 명령)
  mma_f16   mma.sync m16n8k16 f16·f16→f32
  mma_bf16  mma.sync m16n8k16 bf16·bf16→f32
  wgmma_s8  wgmma.mma_async m64n128k32 s8·s8→s32 (sm_90a — Hopper 만; 다른 GPU 에서는 건너뛴다)

워프(또는 워프그룹)마다 서로 기대지 않는 누산기 여러 벌에 같은 명령을 되풀이한다 — 자료는 아무 값이나(처리량만 본다).
CUDA 드라이버 API 만 쓴다(libcuda.so.1 · nvcuda.dll). numpy 없음.
"""
import argparse
import ctypes
import json
import os
import sys
import time

cu = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")


def 확인(r, 무엇):
    if r != 0:
        이름 = ctypes.c_char_p()
        cu.cuGetErrorName(r, ctypes.byref(이름))
        raise RuntimeError(f"{무엇}: CUDA {r} {이름.value.decode() if 이름.value else ''}")


def 장치열기():
    확인(cu.cuInit(0), "cuInit")
    d = ctypes.c_int()
    확인(cu.cuDeviceGet(ctypes.byref(d), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    확인(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, d), "cuCtxCreate")

    def 속성(n):
        v = ctypes.c_int()
        확인(cu.cuDeviceGetAttribute(ctypes.byref(v), n, d), "cuDeviceGetAttribute")
        return v.value
    이름 = ctypes.create_string_buffer(256)
    cu.cuDeviceGetName(이름, 256, d)
    return {"name": 이름.value.decode("utf-8", "replace"), "sm": 속성(16), "clock_khz": 속성(13),
            "cc": f"{속성(75)}.{속성(76)}"}


def 모듈(ptx):
    로그 = ctypes.create_string_buffer(16384)
    옵션 = (ctypes.c_int * 2)(5, 6)             # CU_JIT_ERROR_LOG_BUFFER · _SIZE_BYTES
    값 = (ctypes.c_void_p * 2)(ctypes.cast(로그, ctypes.c_void_p), ctypes.c_void_p(16384))
    m = ctypes.c_void_p()
    r = cu.cuModuleLoadDataEx(ctypes.byref(m), ptx.encode("ascii") + b"\0", 2, 옵션, 값)
    if r != 0:
        raise RuntimeError(f"cuModuleLoadDataEx {r}: {로그.value.decode('utf-8', 'replace')}")
    return m


def 함수(m, 이름):
    f = ctypes.c_void_p()
    확인(cu.cuModuleGetFunction(ctypes.byref(f), m, 이름.encode()), "cuModuleGetFunction")
    return f


def 재기(f, 격자, 블록, 반복, 출력):
    p1 = ctypes.c_uint64(출력)
    p2 = ctypes.c_uint32(반복)
    인자 = (ctypes.c_void_p * 2)(ctypes.cast(ctypes.byref(p1), ctypes.c_void_p), ctypes.cast(ctypes.byref(p2), ctypes.c_void_p))
    e0, e1 = ctypes.c_void_p(), ctypes.c_void_p()
    확인(cu.cuEventCreate(ctypes.byref(e0), 0), "cuEventCreate")
    확인(cu.cuEventCreate(ctypes.byref(e1), 0), "cuEventCreate")
    확인(cu.cuLaunchKernel(f, 격자, 1, 1, 블록, 1, 1, 0, None, 인자, None), "cuLaunchKernel(데우기)")
    확인(cu.cuCtxSynchronize(), "cuCtxSynchronize(데우기)")
    시간들 = []
    for _ in range(3):
        확인(cu.cuEventRecord(e0, None), "cuEventRecord")
        확인(cu.cuLaunchKernel(f, 격자, 1, 1, 블록, 1, 1, 0, None, 인자, None), "cuLaunchKernel")
        확인(cu.cuEventRecord(e1, None), "cuEventRecord")
        확인(cu.cuEventSynchronize(e1), "cuEventSynchronize")
        ms = ctypes.c_float()
        확인(cu.cuEventElapsedTime(ctypes.byref(ms), e0, e1), "cuEventElapsedTime")
        시간들.append(ms.value)
    시간들.sort()
    return 시간들[1]


def 머리(대상, 판="7.4"):
    return [f".version {판}", f".target {대상}", ".address_size 64"]


def 시작줄(이름):
    return [f".visible .entry {이름}(.param .u64 pout, .param .u32 piters)", "{",
            "  .reg .u32 %r<8>;", "  .reg .u64 %rd<8>;", "  .reg .pred %p<4>;",
            "  ld.param.u64 %rd1, [pout];", "  ld.param.u32 %r1, [piters];",
            "  mov.u32 %r2, %tid.x;", "  mov.u32 %r3, %ctaid.x;", "  mov.u32 %r4, %ntid.x;", "  mad.lo.u32 %r5, %r3, %r4, %r2;"]


def 끝줄(누산, 형):
    L = []
    for k in range(1, len(누산)):
        L.append(f"  add.{형} {누산[0]}, {누산[0]}, {누산[k]};")
    L += ["  mul.wide.u32 %rd2, %r5, 4;", "  add.u64 %rd3, %rd1, %rd2;", f"  st.global.{'s32' if 형 == 's32' else 'f32'} [%rd3], {누산[0]};",
          "  ret;", "}"]
    return L


def mma커널(이름, 종류, 벌):
    """mma.sync — 워프마다 누산기 '벌' 벌(4 칸씩). 종류: s8 · f16 · bf16."""
    if 종류 == "s8":
        명령 = "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
        누형, 더하기형, 영 = ".s32", "s32", "0"
        일 = 2 * 16 * 8 * 32
    else:
        명령 = f"mma.sync.aligned.m16n8k16.row.col.f32.{종류}.{종류}.f32"
        누형, 더하기형, 영 = ".f32", "f32", "0f00000000"
        일 = 2 * 16 * 8 * 16
    L = 머리("sm_80") + 시작줄(이름)
    L += ["  .reg .b32 %a<4>;", "  .reg .b32 %b<2>;", f"  .reg {누형} %c<{4 * 벌}>;"]
    L += ["  xor.b32 %a0, %r5, 0x01010101;", "  xor.b32 %a1, %r5, 0x00020002;", "  xor.b32 %a2, %r5, 0x00010001;",
          "  xor.b32 %a3, %r5, 0x00030003;", "  xor.b32 %b0, %r5, 0x00010001;", "  xor.b32 %b1, %r5, 0x00020002;"]
    if 종류 != "s8":
        L += [f"  and.b32 %{x}, %{x}, 0x00ff00ff;" for x in ("a0", "a1", "a2", "a3", "b0", "b1")]    # 작은 수(비정규) — 넘침 없이
    L += [f"  mov{누형} %c{k}, {영};" for k in range(4 * 벌)]
    L += ["LOOP:"]
    for g in range(벌):
        c = ",".join(f"%c{4 * g + i}" for i in range(4))
        L.append(f"  {명령} {{{c}}}, {{%a0,%a1,%a2,%a3}}, {{%b0,%b1}}, {{{c}}};")
    L += ["  sub.u32 %r1, %r1, 1;", "  setp.ne.u32 %p1, %r1, 0;", "  @%p1 bra LOOP;"]
    L += 끝줄([f"%c{k}" for k in range(4 * 벌)], 더하기형)
    return "\n".join(L) + "\n", 일 * 벌


def wgmma커널(이름, 번):
    """wgmma.mma_async m64n128k32 s8 — 워프그룹(128 스레드)마다 한 번에 '번' 번, 공유 메모리의 A · B(서술자, 틀 없음)."""
    L = 머리("sm_90a", "8.0") + 시작줄(이름)
    L += ["  .reg .s32 %d<64>;", "  .reg .u32 %s<4>;", "  .reg .u64 %da;", "  .reg .u64 %db;", "  .reg .u64 %t<4>;",
          "  .shared .align 1024 .b8 smem[16384];"]
    # 공유 메모리를 채운다(스레드마다 128 바이트)
    L += ["  mov.u32 %s0, smem;", "  shl.b32 %s1, %r2, 7;", "  add.u32 %s1, %s0, %s1;", "  mov.u32 %s2, 0;",
          "FILL:", "  st.shared.u32 [%s1], %r5;", "  add.u32 %s1, %s1, 4;", "  add.u32 %s2, %s2, 1;",
          "  setp.lt.u32 %p2, %s2, 32;", "  @%p2 bra FILL;", "  bar.sync 0;"]
    # 서술자: 시작 주소 >> 4 (14 비트) | LBO 128 B >> 4 = 8 (비트 16–29) | SBO 256 B >> 4 = 16 (비트 32–45), 틀 없음
    L += ["  cvt.u64.u32 %t0, %s0;", "  shr.u64 %t0, %t0, 4;", "  and.b64 %t0, %t0, 16383;",
          "  or.b64 %da, %t0, 68719476736;", "  or.b64 %da, %da, 524288;",                   # 16 << 32 · 8 << 16
          "  add.u64 %t1, %t0, 256;", "  or.b64 %db, %t1, 68719476736;", "  or.b64 %db, %db, 524288;"]   # B 는 4 KB 뒤(4096 >> 4 = 256)
    L += [f"  mov.s32 %d{k}, 0;" for k in range(64)]
    L += ["  setp.ne.u32 %p3, %r4, 0;"]                                                       # 늘 참 — D 에 더한다
    L += ["LOOP:", "  wgmma.fence.sync.aligned;"]
    d = ",".join(f"%d{k}" for k in range(64))
    for _ in range(번):
        L.append(f"  wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 {{{d}}}, %da, %db, %p3;")
    L += ["  wgmma.commit_group.sync.aligned;", "  wgmma.wait_group.sync.aligned 0;",
          "  sub.u32 %r1, %r1, 1;", "  setp.ne.u32 %p1, %r1, 0;", "  @%p1 bra LOOP;"]
    L += 끝줄([f"%d{k}" for k in range(64)], "s32")
    return "\n".join(L) + "\n", 2 * 64 * 128 * 32 * 번


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--ptx", default="", help="PTX 를 이 폴더에 쓰기만 하고 끝(ptxas 확인용)")
    a = ap.parse_args()
    커널들 = [("mma_s8", *mma커널("mma_s8", "s8", 8), 4), ("mma_f16", *mma커널("mma_f16", "f16", 8), 4),
             ("mma_bf16", *mma커널("mma_bf16", "bf16", 8), 4), ("wgmma_s8", *wgmma커널("wgmma_s8", 8), 1)]
    if a.ptx:
        os.makedirs(a.ptx, exist_ok=True)
        for 이름, ptx, _, _ in 커널들:
            open(os.path.join(a.ptx, 이름 + ".ptx"), "w", newline="\n").write(ptx)
        return
    장치 = 장치열기()
    print(f"장치: {장치['name']} · SM {장치['sm']} · cc {장치['cc']} · 최대 클럭 {장치['clock_khz'] / 1000:.0f} MHz", flush=True)
    출력 = ctypes.c_uint64()
    확인(cu.cuMemAlloc_v2(ctypes.byref(출력), 64 << 20), "cuMemAlloc")
    결과 = {"device": 장치, "kernels": {}}
    for 이름, ptx, 일, 워프그룹 in 커널들:
        if 이름.startswith("wgmma") and not 장치["cc"].startswith("9."):
            print(f"  {이름}: 건너뜀(sm_90 아님)", flush=True)
            continue
        try:
            f = 함수(모듈(ptx), 이름)
            블록 = 128
            격자 = 장치["sm"] * (4 if 워프그룹 == 4 else 2)
            단위 = 블록 // 32 if 워프그룹 == 4 else 1                # 블록마다 일하는 단위(워프 · 워프그룹)
            반복 = 2000
            ms = 재기(f, 격자, 블록, 반복, 출력.value)
            반복 = max(1, int(반복 * 200.0 / max(ms, 0.01)))          # 200 ms 쯤으로
            ms = 재기(f, 격자, 블록, 반복, 출력.value)
            총 = 격자 * 단위 * 반복 * 일
            T = 총 / (ms / 1000.0) / 1e12
            결과["kernels"][이름] = {"ms": ms, "iters": 반복, "grid": 격자, "block": 블록, "ops": 총, "tops": T,
                                   "per_sm_per_clock_at_max": 총 / (ms / 1000.0) / 장치["sm"] / (장치["clock_khz"] * 1000.0)}
            print(f"  {이름}: {T:8.1f} T(OP|FLOP)/s  ({ms:.1f} ms · 반복 {반복} · 격자 {격자} × {블록})", flush=True)
        except RuntimeError as e:
            결과["kernels"][이름] = {"error": str(e)[:2000]}
            print(f"  {이름}: 실패 — {str(e)[:400]}", flush=True)
            if 이름.startswith("wgmma"):
                break
    if a.json:
        open(a.json, "w", encoding="utf-8").write(json.dumps(결과, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
