"""Native AVX2 integer dot + exact native FP post, bounded worker chunks."""
import ctypes,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from native_post import ROOT,C,NativePost

class CpuLinear:
    def __init__(self,workers=4):
        self.post=NativePost()
        self.lib=ctypes.CDLL(str(ROOT/'빌드/npu/next-stage/cpu_dot.dll'))
        self.lib.geul_cpu_dot.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4
        self.lib.geul_cpu_dot.restype=None
        self.pool=ThreadPoolExecutor(max_workers=workers)
        self.profile={}
    def __call__(self,x,q,scales,bias,자리수=2,**kwargs):
        t=time.perf_counter()
        x=np.asarray(x,np.float32); q=np.ascontiguousarray(q,np.int8)
        m,k=x.shape; n=q.shape[0]; kb=k//32
        assert q.shape[1]==k and k%32==0 and 자리수 in (1,2)
        digits,e=C.자리로(x,자리수)
        digits=np.ascontiguousarray(digits,np.int8)
        packing=time.perf_counter()-t
        def chunk(lo):
            hi=min(lo+1024,n); w=np.ascontiguousarray(q[lo:hi])
            sums=np.empty((m,kb,hi-lo),np.int32)
            self.lib.geul_cpu_dot(digits.ctypes.data,w.ctypes.data,sums.ctypes.data,m,hi-lo,kb,자리수)
            return self.post(sums,e,scales[lo:hi],자리수)
        y=np.concatenate(list(self.pool.map(chunk,range(0,n,1024))),axis=1)
        self.profile={'packing_seconds':packing,'cpu_dot_and_post_seconds':time.perf_counter()-t-packing}
        return C.더(y,np.asarray(bias,np.float32)[None,:])
    def close(self): self.pool.shutdown()
