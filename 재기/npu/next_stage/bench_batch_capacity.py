"""Measure only new submission capacities; retain prior baseline artifacts."""
import hashlib
import json
import time
import traceback
import ctypes
from contextlib import ExitStack, closing
from concurrent.futures import ThreadPoolExecutor
from statistics import median
import numpy as np
from native_post import HERE, ROOT
from reuse_device import ReuseDevice
from joined_prefill import JoinedPrefill
from parallel_post import ParallelPost
from cpu_linear import CpuLinear
from cached_qwen import CachedQwen, R


def main():
    report = dict(complete=False, checks=0, linear=[], model=[], baseline_source='reuse-overlap.json and long-overlap.json; not rerun')
    def save():
        (HERE/'results/batch-capacity.json').write_text(json.dumps(report,indent=2),encoding='utf8',newline='\n')
    def check(a,b):
        if not np.array_equal(a.view(np.uint32),b.view(np.uint32)):
            raise AssertionError('output bits differ')
        report['checks'] += 1
    try:
        report['idle_sleep_inhibited']=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        with ExitStack() as stack:
            cpu=stack.enter_context(closing(CpuLinear()))
            post=stack.enter_context(closing(ParallelPost(4)))
            coordinator=stack.enter_context(ThreadPoolExecutor(1))
            modes={f:JoinedPrefill(ReuseDevice(f),finalizer=post,finalizer_executor=coordinator) for f in (4096,8192)}
            rng=np.random.default_rng(902)
            for m,n,k in ((5,55,96),(17,97,160)):
                x=rng.standard_normal((m,k)).astype(np.float32)
                q=rng.integers(-128,128,(n,k),dtype=np.int8)
                scales=rng.uniform(-0.01,0.01,(n,k//32)).astype(np.float32)
                bias=rng.standard_normal(n).astype(np.float32)
                expected=cpu(x,q,scales,bias)
                for backend in modes.values():
                    check(expected,backend(x,q,scales,bias))
            print('new capacity edge checks passed',flush=True)
            g=R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
            q,scales=R.가중(g,'blk.0.ffn_up.weight')[1:3]
            bias=np.zeros(len(q),np.float32)
            for m in (64,256):
                x=rng.standard_normal((m,q.shape[1])).astype(np.float32)
                expected=cpu(x,q,scales,bias)
                for backend in modes.values():
                    check(expected,backend(x,q,scales,bias))
                for repeat in range(3):
                    for frames in ((4096,8192) if repeat%2==0 else (8192,4096)):
                        backend=modes[frames]
                        start=time.perf_counter()
                        actual=backend(x,q,scales,bias)
                        seconds=time.perf_counter()-start
                        check(expected,actual)
                        report['linear'].append(dict(rows=m,frames=frames,seconds=seconds,calls=backend.submissions,
                            output_bytes=backend.submissions*frames*128*4,profile=backend.profile.copy()))
                        save()
                print('linear done',m,flush=True)
            winner=min(modes,key=lambda f:median(r['seconds'] for r in report['linear'] if r['rows']==256 and r['frames']==f))
            report['winner']=winner
            def route(x,*args,**kwargs):
                return (cpu if len(x)==1 else modes[winner])(x,*args,**kwargs)
            model=CachedQwen(g,route,quant_cache=True,context_capacity=257)
            prompt=np.random.default_rng(407).integers(100,30000,256).tolist()[:64]
            model.step(prompt[:3])
            model.reset()
            print('new model path start',winner,flush=True)
            start=time.perf_counter()
            first=model.step(prompt)
            stamp=time.perf_counter()
            second=model.step([int(first.argmax())])
            end=time.perf_counter()
            hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in (first,second)]
            ref=next(r for r in json.loads((HERE/'results/phase-bench.json').read_text())['runs'] if r['length']==64 and r['backend']=='cpu')
            assert hashes==[ref['prefill_hash'],ref['decode_hash']], 'model output differs'
            report['model'].append(dict(length=64,frames=winner,prefill=stamp-start,decode=end-stamp,hashes=hashes))
            report['complete']=True
            print('model passed',round(stamp-start,3),flush=True)
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        save()


if __name__=='__main__':
    main()
