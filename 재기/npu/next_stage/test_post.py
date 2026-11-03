import json,time,statistics
import numpy as np
from native_post import NativePost,ROOT,HERE,C
from exact_linear import difference

native=NativePost()
rng=np.random.default_rng(20261031)
out=HERE/'results'
out.mkdir(exist_ok=True)
report={'passed':False,'fma_elements':0,'fixtures':[]}
try:
    # Independent random finite bit patterns and FP32 cancellation/tiny boundaries.
    raw=rng.integers(0,2**32,(3,200000),dtype=np.uint32)
    raw[(raw&0x7f800000)==0x7f800000]=0
    a,b,c=raw.view(np.float32)
    edges=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x00800001,0x3f800000,0x3f800001,0x3f7fffff,0xbf800000,0x7f7fffff],np.uint32).view(np.float32)
    aa,bb,cc=np.meshgrid(edges,edges,edges,indexing='ij')
    a=np.concatenate([a,aa.ravel()]);b=np.concatenate([b,bb.ravel()]);c=np.concatenate([c,cc.ravel()])
    saved=C.비정규수버림
    for ftz in (True,False):
        C.비정규수버림=ftz
        with np.errstate(all='ignore'):
            ref=C.fma32(a,b,c);actual=native.fma(a,b,c)
        delta=difference(ref,actual)
        # Finite inputs can overflow to infinity; they are still compared by bits.
        assert delta['different_bits_elements']==0,delta
        report['fma_elements']+=a.size
    C.비정규수버림=saved
    for case in range(3):
        f=np.load(ROOT/f'빌드/matrix/case{case}.npz')
        m,k=f['x'].shape;n=f['q'].shape[0];kb=k//32
        s=np.fromfile(ROOT/f'빌드/matrix/case{case}.expected.bin',np.int32).reshape(2,m,kb,n)
        t=s[0]+256*s[1]
        def reference():
            u=np.ldexp(t.astype(np.float32),(f['e']-13)[:,:,None])
            y=np.zeros((m,n),np.float32)
            for block in range(kb): y=C.fma32(u[:,block],np.broadcast_to(f['d'][:,block],y.shape),y)
            return y
        samples={'numpy':[],'native':[]}
        ref=reference()
        for repeat in range(12):
            pairs=[('numpy',reference),('native',lambda:native(t,f['e'],f['d'],2))]
            if repeat%2:pairs.reverse()
            for name,fn in pairs:
                start=time.perf_counter();actual=fn();elapsed=time.perf_counter()-start
                assert difference(ref,actual)['different_bits_elements']==0
                if repeat: samples[name].append(elapsed)
        delta=difference(f['reference'],C.더(native(t,f['e'],f['d'],2),f['bias'][None,:]))
        assert delta['different_bits_elements']==0
        row={'shape':[m,n,k],'median_seconds':{n:statistics.median(v) for n,v in samples.items()},'samples':samples,**delta}
        report['fixtures'].append(row)
        print(row['shape'],row['median_seconds'],flush=True)
    report['passed']=True
except Exception as exc:
    report['error']=repr(exc)
    raise
finally:
    (out/'post.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
