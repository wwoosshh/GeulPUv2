"""Compare packing and validate full decoder on identical saved prompt prefixes."""
import json,time
from math import ceil
from contextlib import closing
import numpy as np
from phase_bench import HERE,ROOT,CachedQwen,R,Device,NativePost,CpuLinear,checked,digest
from joined_prefill import JoinedPrefill

def main():
    result=HERE/'results/joined-prefill.json'
    reference=json.loads((HERE/'results/phase-bench.json').read_text())
    rows=[]
    def save(row):
        rows.append(row)
        result.write_text(json.dumps(rows,indent=2),encoding='utf8')
        print(row,flush=True)
    g=R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
    with closing(CpuLinear()) as cpu:
        npu=JoinedPrefill(Device('gl',2048),finalizer=NativePost())
        model=CachedQwen(g,cpu,quant_cache=True,context_capacity=257)
        rng=np.random.default_rng(407)
        prompt=rng.integers(100,30000,256).tolist()
        w=model.layers[0][1]['ffn_up']
        for m in (16,64,256,5):
            x=rng.standard_normal((m,w[1].shape[1])).astype(np.float32)
            q,scale=w[1:3]
            if m==5:
                q,scale=q[:55],scale[:55]
            bias=np.zeros(len(q),np.float32)
            expected=cpu(x,q,scale,bias)
            kb=q.shape[1]//32
            old_width=max(32,min(4096,((16<<20)//(2*m*kb*4))//32*32))
            old_calls=ceil(m/4)*sum(ceil(ceil(min(old_width,len(q)-lo)/32)*kb/2048)
                                      for lo in range(0,len(q),old_width))
            for i in range(3):
                t=time.perf_counter()
                out=npu(x,q,scale,bias)
                elapsed=time.perf_counter()-t
                checked(expected,out)
                save(dict(rows=m,outputs=len(q),depth=q.shape[1],repeat=i,seconds=elapsed,baseline_calls_by_shape=old_calls,calls=npu.submissions,
                          useful_frames=npu.useful_frames,submitted_frames=npu.submitted_frames,profile=npu.profile.copy()))
        # Run an integrated case if the packed path beat the previous NPU layer.
        prior=[r['seconds'] for r in reference['linear'] if r['length']==256 and r['backend']=='npu']
        if np.mean([r['seconds'] for r in rows if r['rows']==256 and r['repeat']>0]) < np.mean(prior):
            model.backend=lambda x,*a,**kw: (cpu if len(x)==1 else npu)(x,*a,**kw)
            model.step(prompt[:3])
            model.reset()
            t=time.perf_counter()
            first=model.step(prompt)
            tok=int(first.argmax())
            stamp=time.perf_counter()
            second=model.step([tok])
            end=time.perf_counter()
            ref=next(r for r in reference['runs'] if r['backend']=='cpu' and r['length']==256)
            assert (digest(first),digest(second))==(ref['prefill_hash'],ref['decode_hash'])
            save(dict(full_length=256,prefill=stamp-t,next_token=end-stamp,first_hash=digest(first),next_hash=digest(second)))
if __name__=='__main__':
    main()
