"""Correctness only. Does not rerun completed performance benchmarks."""
import ctypes
import hashlib
import json
import traceback
from contextlib import ExitStack,closing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from native_post import HERE,ROOT
from adaptive_reuse import AdaptiveReuseDevice,select_capacity
from joined_prefill import JoinedPrefill
from parallel_post import ParallelPost
from cpu_linear import CpuLinear
from cached_qwen import CachedQwen,R


def main():
    report=dict(complete=False,policy_checks=0,tile_checks=0,linear_checks=0,model_hashes=[],selections={})
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        cases=((4,508,2048),(4,512,2048),(4,516,4096),(16,512,4096),
               (20,204,4096),(20,208,2048),(20,408,8192),(20,412,2048))
        for groups,span,expected in cases:
            assert select_capacity(groups,span)==expected
            report['policy_checks']+=1
        for args in ((0,1),(1,0),(-1,3)):
            try:
                select_capacity(*args)
            except ValueError:
                report['policy_checks']+=1
            else:
                raise AssertionError('invalid size accepted')
        device=AdaptiveReuseDevice()
        rng=np.random.default_rng(1204)
        for groups,span,_ in cases:
            inputs=rng.integers(-128,128,(groups,3,2,4,32),dtype=np.int8)
            weights=rng.integers(-128,128,(span,32,32),dtype=np.int8)
            digits=inputs[:,np.arange(span)%3].astype(np.int64)
            combined=digits[:,:,0]+256*digits[:,:,1]
            expected=np.einsum('gsrk,snk->gsrn',combined,weights.astype(np.int64)).astype(np.int32)
            actual=device.grouped(inputs,weights,groups).reshape(expected.shape)
            assert np.array_equal(actual,expected),'integer partial sums differ'
            padded=((groups+3)//4*4)*((span+3)//4*4)
            assert device.last_calls==(padded+device.frames-1)//device.frames
            report['tile_checks']+=1
        with ExitStack() as stack:
            cpu=stack.enter_context(closing(CpuLinear()))
            post=stack.enter_context(closing(ParallelPost(4)))
            pool=stack.enter_context(ThreadPoolExecutor(1))
            backend=JoinedPrefill(device,finalizer=post,finalizer_executor=pool)
            for m,n,k in ((1,33,64),(17,97,160),(64,257,1024),(65,513,1024),(129,1025,1024)):
                x=rng.standard_normal((m,k)).astype(np.float32)
                q=rng.integers(-128,128,(n,k),dtype=np.int8)
                scales=rng.uniform(-.01,.01,(n,k//32)).astype(np.float32)
                bias=rng.standard_normal(n).astype(np.float32)
                expected=cpu(x,q,scales,bias)
                actual=backend(x,q,scales,bias)
                assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32)),'float bits differ'
                report['linear_checks']+=1
            print('policy, tile, linear checks passed',flush=True)
            def route(x,*args,**kwargs):
                return (cpu if len(x)==1 else backend)(x,*args,**kwargs)
            g=R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
            model=CachedQwen(g,route,quant_cache=True)
            prompt=np.random.default_rng(407).integers(100,30000,256).tolist()[:64]
            first=model.step(prompt)
            second=model.step([int(first.argmax())])
            hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in (first,second)]
            ref=next(r for r in json.loads((HERE/'results/phase-bench.json').read_text())['runs'] if r['length']==64 and r['backend']=='cpu')
            assert hashes==[ref['prefill_hash'],ref['decode_hash']]
            report['model_hashes']=hashes
        report['selections']=device.selections
        report['complete']=True
        print('adaptive model output matched',flush=True)
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        (HERE/'results/adaptive-reuse.json').write_text(json.dumps(report,indent=2),encoding='utf8',newline='\n')


if __name__=='__main__':
    main()
