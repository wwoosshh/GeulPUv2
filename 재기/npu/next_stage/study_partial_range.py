import json
from contextlib import closing
import numpy as np
from native_post import HERE,ROOT
from reuse_device import ReuseDevice
from joined_prefill import JoinedPrefill
from parallel_post import ParallelPost
from cpu_linear import CpuLinear
from cached_qwen import R

class Capture(ReuseDevice):
    def __init__(self):
        super().__init__(4096)
        self.stats=[]
    def grouped(self,*args):
        out=super().grouped(*args)
        values=out.reshape(-1,128)
        self.stats.append(dict(values=out.size,tiles=len(values),minimum=int(out.min()),maximum=int(out.max()),
            int16_values=int(((out>=-32768)&(out<=32767)).sum()),
            int24_values=int(((out>=-8388608)&(out<=8388607)).sum()),
            int16_tiles=int(((values>=-32768)&(values<=32767)).all(axis=1).sum()),
            int24_tiles=int(((values>=-8388608)&(values<=8388607)).all(axis=1).sum())))
        return out

def main():
    device=Capture()
    g=R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
    report=dict(complete=False,cases=[])
    rng=np.random.default_rng(902)
    with closing(ParallelPost(4)) as post,closing(CpuLinear()) as cpu:
        backend=JoinedPrefill(device,finalizer=post)
        for key,k in (('ffn_up',1024),('ffn_down',3072),('attn_q',1024)):
            q,scales=R.가중(g,'blk.0.'+key+'.weight')[1:3]
            x=rng.standard_normal((64,k)).astype(np.float32)
            bias=np.zeros(len(q),np.float32)
            device.stats=[]
            actual=backend(x,q,scales,bias)
            expected=cpu(x,q,scales,bias)
            assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32))
            report['cases'].append(dict(name=key,shape=[64,len(q),k],chunks=device.stats))
    report['complete']=True
    (HERE/'results/partial-range.json').write_text(json.dumps(report,indent=2),encoding='utf8',newline='\n')
    print(json.dumps({key:sum(c[key] for r in report['cases'] for c in r['chunks']) for key in ('values','tiles','int16_values','int24_values','int16_tiles','int24_tiles')}),flush=True)

if __name__=='__main__':main()
