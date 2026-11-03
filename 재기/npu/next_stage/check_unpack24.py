import ctypes
import json
import numpy as np
from native_post import ROOT,HERE

lib=ctypes.CDLL(str(ROOT/'빌드/npu/next-stage/unpack24.dll'))
lib.unpack24.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
lib.unpack24.restype=None
rng=np.random.default_rng(615)
extremes=np.array([-2147483648,2147483647,-8388609,8388608,-16777216,16777216,-134742016],np.int32)
for count in range(8):
    original=rng.integers(-8388608,8388608,128,dtype=np.int32)
    original[:count]=extremes[:count]
    original[-4:]=[-8388608,8388607,-1,0]
    raw=np.zeros(104,np.int32)
    raw[0]=count
    for i in range(count):
        raw[i+1]=(i<<8)|(int(original[i])>>24&255)
    data=bytearray()
    for value in original:
        data.extend((int(value)&0xffffff).to_bytes(3,'little'))
    raw[8:]=np.frombuffer(data,dtype=np.int32)
    out=np.empty(128,np.int32)
    lib.unpack24(raw.ctypes.data,out.ctypes.data,1)
    assert np.array_equal(original,out)
(HERE/'results/unpack24-boundaries.json').write_text(json.dumps(dict(complete=True,cases=8,values=1024)),encoding='utf8',newline='\n')
print('SIMD signed-boundary cases passed: 8')
