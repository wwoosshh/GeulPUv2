import hashlib
import json
import time
import traceback
from contextlib import ExitStack, closing
from statistics import median
import numpy as np
from native_post import HERE, ROOT, NativePost, C
from parallel_post import ParallelPost
from multi_device import MultiDevice
from joined_prefill import JoinedPrefill
from cpu_linear import CpuLinear
from cached_qwen import CachedQwen, R


def main():
    report = dict(complete=False, checks=0, post=[], linear=[], model=[])
    path = HERE / 'results/parallel-post.json'
    original_ftz = C.비정규수버림

    def save():
        path.write_text(json.dumps(report,indent=2),encoding='utf8')

    def check(a,b):
        if not np.array_equal(a.view(np.uint32),b.view(np.uint32)):
            raise AssertionError('output bits differ')
        report['checks'] += 1

    try:
        with ExitStack() as stack:
            posts = {w: stack.enter_context(closing(ParallelPost(w))) for w in (1,2,4)}
            reference_post = NativePost()
            rng = np.random.default_rng(725)
            # Strided inputs, narrow outputs, uneven partitions, both FTZ modes.
            for shape in ((1,3,1),(5,7,55),(16,32,256),(64,32,1024)):
                m,kb,n=shape
                sums=rng.integers(-(1<<25),1<<25,(m,kb,n*2),dtype=np.int32)[:,:,::2]
                exponents=rng.integers(-98,101,(m,kb),dtype=np.int64)
                scales=rng.uniform(-1,1,(n,kb)).astype(np.float16).astype(np.float32)
                for ftz in (False,True):
                    C.비정규수버림=ftz
                    for digits in (1,2):
                        expected=reference_post(sums,exponents,scales,digits)
                        for post in posts.values():
                            check(expected,post(sums,exponents,scales,digits))
            C.비정규수버림=original_ftz
            for repeat in range(6):
                for workers in ((1,2,4) if repeat%2==0 else (4,2,1)):
                    start=time.perf_counter()
                    actual=posts[workers](sums,exponents,scales,2)
                    elapsed=time.perf_counter()-start
                    check(reference_post(sums,exponents,scales,2),actual)
                    report['post'].append(dict(workers=workers,seconds=elapsed))
            save()
            print('CPU post checks complete',flush=True)
            device=MultiDevice(4)
            backends={w:JoinedPrefill(device,finalizer=p) for w,p in posts.items()}
            cpu=stack.enter_context(closing(CpuLinear()))
            g=R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
            weight=R.가중(g,'blk.0.ffn_up.weight')
            q,scales=weight[1:3]
            bias=np.zeros(len(q),np.float32)
            for m in (16,64,256):
                x=rng.standard_normal((m,q.shape[1])).astype(np.float32)
                expected=cpu(x,q,scales,bias)
                for backend in backends.values():
                    check(expected,backend(x,q,scales,bias))
                for repeat in range(4):
                    for workers in ((1,2,4) if repeat%2==0 else (4,2,1)):
                        backend=backends[workers]
                        start=time.perf_counter()
                        actual=backend(x,q,scales,bias)
                        elapsed=time.perf_counter()-start
                        check(expected,actual)
                        report['linear'].append(dict(rows=m,workers=workers,seconds=elapsed,profile=backend.profile.copy()))
                        save()
                print('linear rows',m,'complete',flush=True)
            scores={w:median(r['seconds'] for r in report['linear'] if r['rows']==256 and r['workers']==w) for w in posts}
            winner=min(scores,key=scores.get)
            report['selected_post_workers']=winner
            prompt=np.random.default_rng(407).integers(100,30000,256).tolist()[:16]
            reference=next(r for r in json.loads((HERE/'results/phase-bench.json').read_text())['runs'] if r['length']==16 and r['backend']=='cpu')
            model=CachedQwen(g,cpu,quant_cache=True)
            def route(x,*args,**kwargs):
                return (cpu if len(x)==1 else backends[selected])(x,*args,**kwargs)
            model.backend=route
            for selected in set((1,winner)):
                model.reset()
                model.step(prompt[:3])
            for selected in (1,winner,winner,1):
                model.reset()
                start=time.perf_counter()
                logits=model.step(prompt)
                token=int(logits.argmax())
                first=time.perf_counter()
                decoded=model.step([token])
                end=time.perf_counter()
                hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in (logits,decoded)]
                if hashes!=[reference['prefill_hash'],reference['decode_hash']]:
                    raise AssertionError('model logits differ')
                report['model'].append(dict(workers=selected,prefill=first-start,decode=end-first,hashes=hashes))
                save()
                print('model',selected,round(first-start,3),flush=True)
            report['complete']=True
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        C.비정규수버림=original_ftz
        save()


if __name__=='__main__':
    main()
