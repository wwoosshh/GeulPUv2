import ctypes
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent))
from exact_linear import C

class NativePost:
    def __init__(self):
        self.lib=ctypes.CDLL(str(ROOT/'빌드/npu/next-stage/post.dll'))
        pointer=ctypes.c_void_p
        self.lib.geul_fma.argtypes=[pointer]*4+[ctypes.c_int]*2
        self.lib.geul_fma.restype=None
        self.lib.geul_finish.argtypes=[pointer]*4+[ctypes.c_int]*6
        self.lib.geul_finish.restype=None
    def fma(self,a,b,c):
        a,b,c=[np.ascontiguousarray(v,np.float32) for v in (a,b,c)]
        assert a.shape==b.shape==c.shape
        out=np.empty_like(a)
        self.lib.geul_fma(*(v.ctypes.data for v in (a,b,c,out)),a.size,int(C.비정규수버림))
        return out
    def __call__(self,s,e,d,digits):
        s=np.ascontiguousarray(s,np.int32)
        e=np.ascontiguousarray(e,np.int64)
        d=np.ascontiguousarray(d,np.float32)
        m,kb,n=s.shape
        if e.shape!=(m,kb) or d.shape!=(n,kb): raise ValueError('post shape mismatch')
        out=np.empty((m,n),np.float32)
        self.lib.geul_finish(*(v.ctypes.data for v in (s,e,d,out)),m,n,kb,n,8*digits-3,int(C.비정규수버림))
        return out
