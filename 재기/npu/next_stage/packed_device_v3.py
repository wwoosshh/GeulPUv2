import time
import ctypes
from native_post import ROOT
import numpy as np
from reuse_device import ReuseDevice
from packed_design_v3 import design

class PackedDevice(ReuseDevice):
    output_words=104
    kernel=staticmethod(design)
    def __init__(self,frames=4096):
        super().__init__(frames)
        self.lib=ctypes.CDLL(str(ROOT/'빌드/npu/next-stage/unpack24.dll'))
        self.lib.unpack24.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
        self.lib.unpack24.restype=None
        self.replay=None
        self.reset_metrics()
    def reset_metrics(self):
        self.metrics=dict(corrected_tiles=0,packed_calls=0,overflow_tiles=0,replay_calls=0,output_bytes=0,decode_seconds=0.,replay_seconds=0.)
    def decode_result(self,raw,activations,weights):
        start=time.perf_counter()
        bad=np.flatnonzero((raw[:,0]<0)|(raw[:,0]>7))
        values=np.empty((len(raw),4,32),np.int32)
        self.lib.unpack24(raw.ctypes.data,values.ctypes.data,len(raw))
        self.metrics['corrected_tiles']+=int((raw[:,0]>0).sum())
        self.metrics['decode_seconds']+=time.perf_counter()-start
        self.metrics['packed_calls']+=1
        self.metrics['overflow_tiles']+=len(bad)
        self.metrics['output_bytes']+=self.frames*104*4
        if len(bad):
            start=time.perf_counter()
            if self.replay is None:
                from multi_device import MultiDevice
                self.replay=MultiDevice(4,frames=64)
            for lo in range(0,len(bad),64):
                indices=bad[lo:lo+64]
                weight_index=(indices//16)*4+indices%4
                values[indices]=self.replay(np.ascontiguousarray(activations[indices]),np.ascontiguousarray(weights[weight_index]))
                self.metrics['replay_calls']+=1
                self.metrics['output_bytes']+=64*128*4
            self.metrics['replay_seconds']+=time.perf_counter()-start
        return values
