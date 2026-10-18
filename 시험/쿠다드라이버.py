"""시험의 CUDA 드라이버 — GeulPU 의 엔진/글gpt2.py(연구용 파이썬 엔진)에서 드라이버만 떼어 왔다(GeulPUv2 2단계). nvcuda.dll 만 쓴다.
커널 단위 비트 시험(판23 · 판26 · 머리128 · 줄128)이 글 생성기가 낸 PTX 모듈을 올려 커널을 띄우는 데 쓴다."""
import ctypes

import numpy as np

c_u64, c_i64, c_uint = ctypes.c_uint64, ctypes.c_int64, ctypes.c_uint


class 드라이버:
    def __init__(self):
        cu = ctypes.WinDLL("nvcuda.dll")
        cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(c_u64), ctypes.c_size_t]
        cu.cuMemcpyHtoD_v2.argtypes = [c_u64, ctypes.c_void_p, ctypes.c_size_t]
        cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, c_u64, ctypes.c_size_t]
        cu.cuMemsetD8_v2.argtypes = [c_u64, ctypes.c_ubyte, ctypes.c_size_t]
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [c_uint] * 7 + [ctypes.c_void_p] * 3
        cu.cuModuleGetGlobal_v2.argtypes = [ctypes.POINTER(c_u64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p,
                                            ctypes.c_char_p]
        cu.cuMemFree_v2.argtypes = [c_u64]
        cu.cuModuleUnload.argtypes = [ctypes.c_void_p]
        self.cu = cu
        self.할당들, self.모듈들 = [], []
        ctx = ctypes.c_void_p()
        self.확인(cu.cuInit(0))
        self.확인(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), 0))
        self.확인(cu.cuCtxSetCurrent(ctx))

    @staticmethod
    def 확인(r):
        if r != 0:
            raise RuntimeError(f"CUDA 드라이버 오류 {r}")

    def 모듈(self, ptx):
        mod = ctypes.c_void_p()
        self.확인(self.cu.cuModuleLoadData(ctypes.byref(mod), ptx + b"\0"))
        self.모듈들.append(mod)
        return mod

    def 함수(self, mod, name):
        fn = ctypes.c_void_p()
        self.확인(self.cu.cuModuleGetFunction(ctypes.byref(fn), mod, ("_G" + name.encode("utf-8").hex()).encode()))
        return fn

    def 할당(self, nbytes):
        p = c_u64()
        self.확인(self.cu.cuMemAlloc_v2(ctypes.byref(p), nbytes))
        self.확인(self.cu.cuMemsetD8_v2(p.value, 0, nbytes))
        self.할당들.append(p.value)
        return p.value

    def 놓기(self):
        """이 드라이버로 잡은 메모리와 모듈을 돌려준다(한 실행에서 모델을 여럿 만들 때)."""
        self.맞추기()
        for p in self.할당들:
            self.확인(self.cu.cuMemFree_v2(c_u64(p)))
        for m in self.모듈들:
            self.확인(self.cu.cuModuleUnload(m))
        self.할당들, self.모듈들 = [], []

    def 올리기(self, dptr, arr):
        arr = np.ascontiguousarray(arr)
        self.확인(self.cu.cuMemcpyHtoD_v2(dptr, arr.ctypes.data, arr.nbytes))

    def 내리기(self, dptr, arr):
        self.확인(self.cu.cuMemcpyDtoH_v2(arr.ctypes.data, dptr, arr.nbytes))
        return arr

    def 맞추기(self):
        self.확인(self.cu.cuCtxSynchronize())
