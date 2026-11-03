import time
import numpy as np
from reuse_device import ReuseDevice
from packed_design import design

class PackedDevice(ReuseDevice):
    output_words=104
    kernel=staticmethod(design)
    def __init__(self,frames=4096):
        super().__init__(frames)
        self.replay=None
        self.reset_metrics()
    def reset_metrics(self):
        self.metrics=dict(packed_calls=0,overflow_tiles=0,replay_calls=0,output_bytes=0,decode_seconds=0.,replay_seconds=0.)
    def decode_result(self,raw,activations,weights):
        start=time.perf_counter()
        bad=np.flatnonzero(raw[:,0])
        words=raw[:,8:].view(np.uint32).reshape(-1,32,3)
        unpacked=np.empty((len(raw),32,4),np.uint32)
        unpacked[:,:,0]=words[:,:,0]&0xffffff
        unpacked[:,:,1]=((words[:,:,0]>>24)|(words[:,:,1]<<8))&0xffffff
        unpacked[:,:,2]=((words[:,:,1]>>16)|(words[:,:,2]<<16))&0xffffff
        unpacked[:,:,3]=words[:,:,2]>>8
        values=((unpacked^0x800000).astype(np.int32)-0x800000).reshape(-1,4,32)
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
