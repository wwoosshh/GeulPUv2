import time
import numpy as np
from retained_device import RetainedDevice

class LeanReuseDevice(RetainedDevice):
    def grouped(self,inputs,weights,groups):
        stamp=time.perf_counter()
        kb=inputs.shape[1]
        span=len(weights)
        ps=(span+3)//4*4
        pg=(groups+3)//4*4
        if pg==groups:
            source=inputs
        else:
            source=np.zeros((pg,kb,2,4,32),np.int8)
            source[:groups]=inputs
        if ps==span:
            bank=weights
        else:
            bank=np.zeros((ps,32,32),np.int8)
            bank[:span]=weights
        output=np.empty((pg,ps,4,32),np.int32)
        self.stages['allocate_pad']+=time.perf_counter()-stamp
        count=pg//4*ps
        batch=self.frames//4
        self.last_calls=0
        for start in range(0,count,batch):
            stamp=time.perf_counter()
            stop=min(start+batch,count)
            tasks=np.arange(start,stop).reshape(-1,4)
            group=(tasks//ps)[:,None,:]*4+np.arange(4)[None,:,None]
            wi=(tasks%ps)[:,None,:]
            activations=source[group,wi%kb].reshape(-1,8,32)
            packed=bank[(tasks%ps).ravel()]
            self.ah[activations.size:].fill(0)
            self.bh[packed.size:].fill(0)
            self.ah[:activations.size]=activations.ravel()
            self.bh[:packed.size]=packed.ravel()
            self.stages['pack_copy']+=time.perf_counter()-stamp
            stamp=time.perf_counter()
            self.a._sync_to_device()
            self.b._sync_to_device()
            self.stages['upload_sync']+=time.perf_counter()-stamp
            stamp=time.perf_counter()
            result=self.kernel(self.a,self.b,self.c,frames=self.frames)
            self.stages['dispatch_wait']+=time.perf_counter()-stamp
            self.stages['runtime_wait_ns']+=result[1].npu_time
            stamp=time.perf_counter()
            view=self.c.numpy()
            self.stages['download_sync']+=time.perf_counter()-stamp
            stamp=time.perf_counter()
            raw=view.reshape(-1,self.output_words)[:len(activations)]
            self.stages['output_copy']+=time.perf_counter()-stamp
            stamp=time.perf_counter()
            values=self.decode_result(raw,activations,packed)
            self.stages['decode_replay']+=time.perf_counter()-stamp
            stamp=time.perf_counter()
            output[group,wi]=values.reshape(len(tasks),4,4,4,32)
            self.stages['scatter']+=time.perf_counter()-stamp
            self.last_calls+=1
        stamp=time.perf_counter()
        result=np.ascontiguousarray(output[:groups,:span]).reshape(-1,4,32)
        self.stages['final_copy']+=time.perf_counter()-stamp
        return result
