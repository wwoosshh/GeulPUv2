import ctypes
import hashlib
import json
import time
import traceback
from contextlib import ExitStack,closing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from native_post import HERE,ROOT
from build_pack24 import build

def main():
    report=dict(complete=False,integer_checks=0,linear_checks=0,runs=[],model=[])
    def save():
        (HERE/'results/packed-partials.json').write_text(json.dumps(report,indent=2),encoding='utf8',newline='\n')
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        build()
        from packed_device import PackedDevice
        from joined_prefill import JoinedPrefill
        from cpu_linear import CpuLinear
        from parallel_post import ParallelPost
        from cached_qwen import CachedQwen,R
        device=PackedDevice()
        rng=np.random.default_rng(902)
        # Check signs, forced overflow replay, group/weight padding and mixed tiles.
        for fill in (0,127,-128,None):
            inputs=np.full((5,3,2,4,32),fill or 0,np.int8)
            weights=np.full((7,32,32),127,np.int8)
            if fill is None:
                inputs=rng.integers(-128,128,inputs.shape,dtype=np.int8)
                weights=rng.integers(-128,128,weights.shape,dtype=np.int8)
            digits=inputs[:,np.arange(7)%3].astype(np.int64)
            expected=np.einsum('gsrk,snk->gsrn',digits[:,:,0]+256*digits[:,:,1],weights.astype(np.int64)).astype(np.int32)
            actual=device.grouped(inputs,weights,5).reshape(expected.shape)
            assert np.array_equal(actual,expected),'packed integer mismatch'
            report['integer_checks']+=1
        print('integer replay checks passed',flush=True)
        with ExitStack() as stack:
            cpu=stack.enter_context(closing(CpuLinear()))
            post=stack.enter_context(closing(ParallelPost(4)))
            pool=stack.enter_context(ThreadPoolExecutor(1))
            backend=JoinedPrefill(device,finalizer=post,finalizer_executor=pool)
            for m,n,k in ((5,55,96),(17,97,160)):
                x=rng.standard_normal((m,k)).astype(np.float32)
                q=rng.integers(-128,128,(n,k),dtype=np.int8)
                s=rng.uniform(-.01,.01,(n,k//32)).astype(np.float32)
                bias=rng.standard_normal(n).astype(np.float32)
                assert np.array_equal(cpu(x,q,s,bias).view(np.uint32),backend(x,q,s,bias).view(np.uint32))
                report['linear_checks']+=1
            g=R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
            q,s=R.가중(g,'blk.0.ffn_up.weight')[1:3]
            bias=np.zeros(len(q),np.float32)
            for m in (64,256):
                x=rng.standard_normal((m,q.shape[1])).astype(np.float32)
                expected=cpu(x,q,s,bias)
                assert np.array_equal(expected.view(np.uint32),backend(x,q,s,bias).view(np.uint32))
                report['linear_checks']+=1
                for repeat in range(3):
                    device.reset_metrics()
                    start=time.perf_counter()
                    actual=backend(x,q,s,bias)
                    elapsed=time.perf_counter()-start
                    assert np.array_equal(expected.view(np.uint32),actual.view(np.uint32))
                    report['linear_checks']+=1
                    report['runs'].append(dict(rows=m,seconds=elapsed,metrics=device.metrics.copy(),profile=backend.profile.copy()))
                    save()
                print('packed timing done',m,flush=True)
            def route(x,*args,**kwargs):
                return (cpu if len(x)==1 else backend)(x,*args,**kwargs)
            model=CachedQwen(g,route,quant_cache=True)
            prompt=np.random.default_rng(407).integers(100,30000,256).tolist()[:16]
            first=model.step(prompt)
            second=model.step([int(first.argmax())])
            hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in (first,second)]
            ref=next(r for r in json.loads((HERE/'results/phase-bench.json').read_text())['runs'] if r['length']==16 and r['backend']=='cpu')
            assert hashes==[ref['prefill_hash'],ref['decode_hash']]
            report['model']=hashes
            report['complete']=True
            print('packed model bits matched',flush=True)
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        save()

if __name__=='__main__':main()
