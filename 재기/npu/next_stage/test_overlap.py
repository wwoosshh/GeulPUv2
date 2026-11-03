"""Two independent device buffer slots; host-observed concurrency, not DMA proof."""
from concurrent.futures import ThreadPoolExecutor
import time,json,statistics,traceback
import numpy as np
from native_post import HERE,NativePost,C
from gl_aie.runtime import Device,PlannedLinear
from exact_linear import difference

rng=np.random.default_rng(3102)
report={'passed':False,'scope':'serial versus two independent XRT buffer slots; hardware DMA overlap not directly measured'}
try:
    devices=[Device('gl',frames=2048),Device('gl',frames=2048)]
    jobs=[]
    for count in (2048,7,2048,33,2048,1,2048,2047):
        a=rng.integers(-128,128,(count,8,32),dtype=np.int8)
        b=rng.integers(-128,128,(count,32,32),dtype=np.int8)
        dot=a.astype(np.int64)@b.astype(np.int64).transpose(0,2,1)
        jobs.append((a,b,dot[:,:4]+256*dot[:,4:]))
    for dev in devices: np.testing.assert_array_equal(dev(*jobs[0][:2]),jobs[0][2])
    samples={'serial':[],'two_slots':[]}
    checks=0
    with ThreadPoolExecutor(max_workers=2) as pool:
        for repeat in range(8):
            order=['serial','two_slots'] if repeat%2==0 else ['two_slots','serial']
            for mode in order:
                start=time.perf_counter()
                outputs=[]
                for base in range(0,len(jobs),2):
                    if mode=='serial': outputs.extend(devices[i](*jobs[base+i][:2]) for i in range(2))
                    else:
                        futures=[pool.submit(devices[i],*jobs[base+i][:2]) for i in range(2)]
                        outputs.extend(f.result() for f in futures)
                elapsed=time.perf_counter()-start
                for actual,job in zip(outputs,jobs): np.testing.assert_array_equal(actual,job[2]);checks+=1
                samples[mode].append(elapsed)
    report.update(passed=True,checks=checks,samples=samples,median_seconds={k:statistics.median(v) for k,v in samples.items()})
    report['post_pipeline']=[]
    with ThreadPoolExecutor(max_workers=1) as executor:
        plain=PlannedLinear(devices[0],finalizer=NativePost())
        pipe=PlannedLinear(devices[0],finalizer=NativePost(),finalizer_executor=executor)
        for m,n,k in [(3,8193,1024),(16,8193,1024)]:
            x=rng.uniform(-1,1,(m,k)).astype(np.float32)
            q=rng.integers(-128,128,(n,k),dtype=np.int8)
            d=rng.uniform(-.1,.1,(n,k//32)).astype(np.float32)
            bias=rng.uniform(-.1,.1,n).astype(np.float32)
            ref=C.행렬곱블라스(x,q,d,bias,자리수=2)
            samples2={'serial':[],'post_pipeline':[]}
            for fn in (plain,pipe):
                assert difference(ref,fn(x,q,d,bias))['different_bits_elements']==0
            for repeat in range(6):
                order=[('serial',plain),('post_pipeline',pipe)]
                if repeat%2:order.reverse()
                for name,fn in order:
                    start=time.perf_counter();actual=fn(x,q,d,bias);elapsed=time.perf_counter()-start
                    assert difference(ref,actual)['different_bits_elements']==0
                    samples2[name].append(elapsed)
            report['post_pipeline'].append({'shape':[m,n,k],'samples':samples2,'median_seconds':{key:statistics.median(v) for key,v in samples2.items()}})
except Exception as exc:
    report.update(passed=False,error=repr(exc),traceback=traceback.format_exc())
finally:
    (HERE/'results/overlap.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
