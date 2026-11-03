import ctypes
import hashlib
import json
import time
import traceback
from contextlib import ExitStack,closing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from native_post import HERE,ROOT
from output_view_device import OutputViewDevice
from joined_prefill import JoinedPrefill
from parallel_post import ParallelPost
from cpu_linear import CpuLinear
from cached_qwen import CachedQwen,R

def main():
    report=dict(complete=False,checks=0,runs=[],model=[])
    def check(a,b):
        assert np.array_equal(a.view(np.uint32),b.view(np.uint32))
        report['checks']+=1
    def save():
        (HERE/'results/output-view.json').write_text(json.dumps(report,indent=2),encoding='utf8',newline='\n')
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        with ExitStack() as stack:
            cpu=stack.enter_context(closing(CpuLinear()))
            post=stack.enter_context(closing(ParallelPost(4)))
            pool=stack.enter_context(ThreadPoolExecutor(1))
            devices={'output_view':OutputViewDevice()}
            backends={n:JoinedPrefill(v,finalizer=post,finalizer_executor=pool) for n,v in devices.items()}
            rng=np.random.default_rng(620)
            for m,n,k in ((5,55,96),(17,97,160),(65,513,1024)):
                x=rng.standard_normal((m,k)).astype(np.float32)
                q=rng.integers(-128,128,(n,k),dtype=np.int8)
                s=rng.uniform(-.01,.01,(n,k//32)).astype(np.float32)
                bias=rng.standard_normal(n).astype(np.float32)
                expected=cpu(x,q,s,bias)
                for name,b in backends.items():
                    first=b(x,q,s,bias)
                    snapshot=first.copy()
                    check(expected,first)
                    devices[name].clear_handle()
                    changed=np.ascontiguousarray(-x)
                    check(cpu(changed,q,s,bias),b(changed,q,s,bias))
                    check(snapshot,first)
                    check(expected,b(x,q,s,bias))
            print('tail, ownership and handle reload checks passed',flush=True)
            g=R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
            q,s=R.가중(g,'blk.0.ffn_up.weight')[1:3]
            x=np.random.default_rng(619).standard_normal((256,q.shape[1])).astype(np.float32)
            bias=np.zeros(len(q),np.float32)
            expected=cpu(x,q,s,bias)
            for b in backends.values():check(expected,b(x,q,s,bias))
            for name in ('output_view',)*3:
                dev=devices[name]
                dev.reset_stages()
                loads=dev.handle_loads
                stamp=time.perf_counter()
                result=backends[name](x,q,s,bias)
                seconds=time.perf_counter()-stamp
                check(expected,result)
                assert dev.handle_loads==loads,'unexpected reload'
                report['runs'].append(dict(mode=name,seconds=seconds,stages=dict(dev.stages),profile=backends[name].profile.copy(),handle_loads_during_run=dev.handle_loads-loads))
                save()
                print(name,round(seconds,4),flush=True)
            def route(x,*args,**kwargs):
                return (cpu if len(x)==1 else backends['output_view'])(x,*args,**kwargs)
            model=CachedQwen(g,route,quant_cache=True)
            prompt=np.random.default_rng(407).integers(100,30000,256).tolist()[:16]
            first=model.step(prompt)
            second=model.step([int(first.argmax())])
            hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in (first,second)]
            ref=next(r for r in json.loads((HERE/'results/phase-bench.json').read_text())['runs'] if r['length']==16 and r['backend']=='cpu')
            assert hashes==[ref['prefill_hash'],ref['decode_hash']]
            report['model']=hashes
            report['complete']=True
            print('model bits matched',flush=True)
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        save()

if __name__=='__main__':main()
