import hashlib
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack,closing
from statistics import median
import numpy as np
from native_post import HERE,ROOT
from multi_device import MultiDevice
from reuse_device import ReuseDevice
from parallel_post import ParallelPost
from joined_prefill import JoinedPrefill
from cpu_linear import CpuLinear
from cached_qwen import CachedQwen,R


def main():
    report=dict(complete=False,checks=0,linear=[],model=[])
    path=HERE/'results/reuse-overlap.json'
    def save():
        path.write_text(json.dumps(report,indent=2),encoding='utf8')
    def check(a,b):
        if not np.array_equal(a.view(np.uint32),b.view(np.uint32)):
            raise AssertionError('output bits differ')
        report['checks']+=1
    try:
        with ExitStack() as stack:
            post=stack.enter_context(closing(ParallelPost(4)))
            cpu=stack.enter_context(closing(CpuLinear()))
            coordinator=stack.enter_context(ThreadPoolExecutor(1))
            original=MultiDevice(4)
            reuse=ReuseDevice()
            modes={
                'baseline':JoinedPrefill(original,finalizer=post),
                'overlap':JoinedPrefill(original,finalizer=post,finalizer_executor=coordinator),
                'reuse':JoinedPrefill(reuse,finalizer=post),
                'combined':JoinedPrefill(reuse,finalizer=post,finalizer_executor=coordinator),
            }
            rng=np.random.default_rng(902)
            for m,n,k in ((5,55,96),(17,97,160),(1,33,64)):
                x=rng.standard_normal((m,k)).astype(np.float32)
                q=rng.integers(-128,128,(n,k),dtype=np.int8)
                scales=rng.uniform(-0.01,0.01,(n,k//32)).astype(np.float32)
                bias=rng.standard_normal(n).astype(np.float32)
                expected=cpu(x,q,scales,bias)
                for name,backend in modes.items():
                    check(expected,backend(x,q,scales,bias))
                    save()
                print('edge shape passed',m,n,k,flush=True)
            g=R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
            weight=R.가중(g,'blk.0.ffn_up.weight')
            q,scales=weight[1:3]
            bias=np.zeros(len(q),np.float32)
            for m in (16,64,256):
                x=rng.standard_normal((m,q.shape[1])).astype(np.float32)
                expected=cpu(x,q,scales,bias)
                for backend in modes.values():
                    check(expected,backend(x,q,scales,bias))
                for repeat in range(4):
                    order=list(modes) if repeat%2==0 else list(reversed(modes))
                    for name in order:
                        backend=modes[name]
                        start=time.perf_counter()
                        actual=backend(x,q,scales,bias)
                        elapsed=time.perf_counter()-start
                        check(expected,actual)
                        calls=backend.submissions
                        reuse_factor=4 if name in ('reuse','combined') else 1
                        report['linear'].append(dict(rows=m,mode=name,seconds=elapsed,
                            calls=calls,input_a_bytes=calls*2048*256,
                            weight_bytes=calls*2048//reuse_factor*1024,output_bytes=calls*2048*128*4,
                            profile=backend.profile.copy()))
                        save()
                print('linear complete',m,flush=True)
            scores={name:median(r['seconds'] for r in report['linear'] if r['rows']==256 and r['mode']==name) for name in modes}
            winner=min(scores,key=scores.get)
            report['winner']=winner
            prompt=np.random.default_rng(407).integers(100,30000,256).tolist()[:16]
            ref=next(r for r in json.loads((HERE/'results/phase-bench.json').read_text())['runs'] if r['length']==16 and r['backend']=='cpu')
            model=CachedQwen(g,cpu,quant_cache=True)
            def route(x,*args,**kwargs):
                return (cpu if len(x)==1 else modes[selected])(x,*args,**kwargs)
            model.backend=route
            for selected in set(('baseline',winner)):
                model.reset()
                model.step(prompt[:3])
            for selected in ('baseline',winner,winner,'baseline'):
                model.reset()
                start=time.perf_counter()
                first=model.step(prompt)
                token=int(first.argmax())
                stamp=time.perf_counter()
                second=model.step([token])
                end=time.perf_counter()
                hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in (first,second)]
                if hashes!=[ref['prefill_hash'],ref['decode_hash']]:
                    raise AssertionError('model logits differ')
                report['model'].append(dict(mode=selected,prefill=stamp-start,decode=end-stamp,hashes=hashes))
                save()
                print('model',selected,round(stamp-start,3),flush=True)
            report['complete']=True
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        save()


if __name__=='__main__':
    main()
