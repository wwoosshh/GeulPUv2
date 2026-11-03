"""New stage-level diagnostic, not a rerun of historical benchmark scripts."""
import ctypes
import json
import time
import traceback
from collections import defaultdict
from contextlib import ExitStack,closing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from native_post import HERE,ROOT
from reuse_device import ReuseDevice
from packed_device_v3 import PackedDevice
from joined_prefill import JoinedPrefill
from cpu_linear import CpuLinear
from parallel_post import ParallelPost
from cached_qwen import R

class Measured:
    def read_output(self,view,count):
        return view.reshape(-1,self.output_words)[:count].copy()

    def reset_stages(self):
        self.stages=defaultdict(float)
    def grouped(self,inputs,weights,groups):
        stamp=time.perf_counter()
        kb=inputs.shape[1]
        span=len(weights)
        ps=(span+3)//4*4
        pg=(groups+3)//4*4
        source=np.zeros((pg,kb,2,4,32),np.int8)
        source[:groups]=inputs
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
            self.ah.fill(0)
            self.bh.fill(0)
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
            raw=self.read_output(view,len(activations))
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

class Plain(Measured,ReuseDevice):pass
class Packed(Measured,PackedDevice):pass

def main():
    report=dict(complete=False,runs=[])
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        with ExitStack() as stack:
            cpu=stack.enter_context(closing(CpuLinear()))
            post=stack.enter_context(closing(ParallelPost(4)))
            pool=stack.enter_context(ThreadPoolExecutor(1))
            devices={'plain':Plain(4096),'packed_v3':Packed(4096)}
            backends={n:JoinedPrefill(v,finalizer=post,finalizer_executor=pool) for n,v in devices.items()}
            g=R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
            q,s=R.가중(g,'blk.0.ffn_up.weight')[1:3]
            x=np.random.default_rng(619).standard_normal((256,q.shape[1])).astype(np.float32)
            bias=np.zeros(len(q),np.float32)
            expected=cpu(x,q,s,bias)
            for name,b in backends.items():
                devices[name].reset_stages()
                assert np.array_equal(expected.view(np.uint32),b(x,q,s,bias).view(np.uint32))
            for name in ('plain','packed_v3','packed_v3','plain','plain','packed_v3'):
                device=devices[name]
                device.reset_stages()
                if hasattr(device,'reset_metrics'):device.reset_metrics()
                stamp=time.perf_counter()
                actual=backends[name](x,q,s,bias)
                elapsed=time.perf_counter()-stamp
                assert np.array_equal(expected.view(np.uint32),actual.view(np.uint32))
                report['runs'].append(dict(mode=name,seconds=elapsed,stages=dict(device.stages),profile=backends[name].profile.copy(),metrics=getattr(device,'metrics',{}).copy()))
                print(name,round(elapsed,4),flush=True)
            report['complete']=True
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        (HERE/'results/transfer-diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf8',newline='\n')

if __name__=='__main__':main()
