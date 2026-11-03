import time,json,hashlib,statistics
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from native_post import HERE,ROOT,NativePost
from cached_qwen import CachedQwen,R
from gl_aie.runtime import Device,PlannedLinear
from cpu_linear import CpuLinear

g=R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
expected=json.loads((HERE/'results/viability-cpu.json').read_text())[0]
report={'rounds':[],'frame_ms':{}}
def save():
    (HERE/'results/viability-compare.json').write_text(json.dumps(report,indent=2),encoding='utf8')
devices={size:Device('gl_row1',size) for size in (512,1024,2048)}
rng=np.random.default_rng(349)
for length in (128,512,1024,2048):
    a=rng.integers(-128,128,(length,8,32),dtype=np.int8)
    b=rng.integers(-128,128,(length,32,32),dtype=np.int8)
    ref=np.einsum('fk,fnk->fn',a[:,0].astype(np.int32)+256*a[:,4].astype(np.int32),b.astype(np.int32))
    for size,device in devices.items():
        if size<length: continue
        timings=[]
        for i in range(5):
            t=time.perf_counter()
            out=device(a,b)
            timings.append(1000*(time.perf_counter()-t))
            assert np.array_equal(ref,out[:,0])
        report['frame_ms'][f'{length}/{size}']=statistics.median(timings[1:])
    save()
class Adaptive:
    rows=1
    fused=True
    frames=2048
    def __call__(self,a,b):
        return devices[min(s for s in devices if s>=len(a))](a,b)
with ThreadPoolExecutor(1) as pool:
    old=PlannedLinear(Device('gl',2048),finalizer=NativePost(),finalizer_executor=pool)
    opt=PlannedLinear(Adaptive(),finalizer=NativePost(),finalizer_executor=pool)
    cpu=CpuLinear()
    for name in ['baseline','optimized','cpu_cached']:
        def linear(x,*args,**kwargs):
            fn=cpu if name=='cpu_cached' else opt if name=='optimized' and len(x)==1 else old
            return fn(x,*args,**kwargs)
        model=CachedQwen(g,linear,quant_cache=name!='baseline')
        for rep in range(2):
            feed=[1,2,3]
            model.reset()
            for step in range(4):
                t=time.perf_counter()
                p=time.process_time()
                logits=model.step(feed)
                elapsed=time.perf_counter()-t
                used=time.process_time()-p
                digest=hashlib.sha256(logits.tobytes()).hexdigest()
                assert digest==expected[step]['hash'],(name,rep,step,digest)
                feed=[int(logits.argmax())]
                row=dict(name=name,repeat=rep,step=step,wall=elapsed,cpu=used,hash=digest,token=feed[0])
                report['rounds'].append(row)
                save()
                print(row,flush=True)
        del model
    cpu.close()
