"""Actual NPU comparison: same device graph, buffers, transfer and finalization."""
import json
import statistics
import time
import traceback
from pathlib import Path
import numpy as np
from runtime import Device,PlannedLinear,ROOT
from exact_linear import C,difference

HERE=Path(__file__).parent
RESULTS=HERE/'results'
RESULTS.mkdir(exist_ok=True)
names=['gl','c','c_unrolled','aie_cpp']
rng=np.random.default_rng(20261031)
report={'passed':False,'kernel_checks':0,'linear_checks':0,'microbenchmarks':[],'matrices':[],
        'scope':'Actual NPU, same single-core tile graph; host transport and FP32 finalization shared',
        'build':json.loads((HERE/'build-results.json').read_text(encoding='utf-8'))}

def check(reference,actual):
    delta=difference(reference,actual)
    assert delta['different_bits_elements']==0,delta
    report['linear_checks']+=1

try:
    for frames in (128,2048):
        devices={name:Device(name,frames) for name in names}
        first={}
        # Full and partial dispatches; changed buffers and signed extremes.
        for repeat,count in enumerate((1,frames,7,frames-1)):
            a=rng.integers(-128,128,(count,8,32),dtype=np.int8)
            b=rng.integers(-128,128,(count,32,32),dtype=np.int8)
            if repeat==0: a.fill(-128); b.fill(-128)
            dots=a.astype(np.int64)@b.astype(np.int64).transpose(0,2,1)
            expected=dots[:,:4]+256*dots[:,4:]
            for name,device in devices.items():
                start=time.perf_counter()
                actual=device(a,b)
                elapsed=time.perf_counter()-start
                if repeat==0: first[name]=elapsed
                np.testing.assert_array_equal(actual,expected)
                report['kernel_checks']+=1
        a=rng.integers(-128,128,(frames,8,32),dtype=np.int8)
        b=rng.integers(-128,128,(frames,32,32),dtype=np.int8)
        dots=a.astype(np.int64)@b.astype(np.int64).transpose(0,2,1)
        expected=dots[:,:4]+256*dots[:,4:]
        samples={name:[] for name in names}
        for repeat in range(12):
            order=names[repeat%len(names):]+names[:repeat%len(names)]
            if repeat%2: order=order[::-1]
            for name in order:
                start=time.perf_counter()
                actual=devices[name](a,b)
                samples[name].append(time.perf_counter()-start)
                np.testing.assert_array_equal(actual,expected)
                report['kernel_checks']+=1
        row={'frames':frames,'first_call_seconds_including_jit':first,'samples_seconds':samples,
             'median_seconds':{name:statistics.median(s) for name,s in samples.items()},
             'io_buffer_bytes':{'input':frames*1280,'output':frames*512}}
        report['microbenchmarks'].append(row)
        print('MICRO',json.dumps(row),flush=True)
    # Keep the 2048-frame instances for matched complete linear pipelines.
    linears={name:PlannedLinear(device) for name,device in devices.items()}
    for m,n,k in [(1,7,32),(5,35,96),(9,65,128),(1,4103,32)]:
        x=rng.uniform(-1,1,(m,k)).astype(np.float32)
        q=rng.integers(-128,128,(n,k),dtype=np.int8)
        d=rng.uniform(-1,1,(n,k//32)).astype(np.float32)
        bias=rng.uniform(-1,1,n).astype(np.float32)
        for exponent in (-98,0,100):
            xx,dd=np.ldexp(x,exponent),np.ldexp(d,-max(0,exponent))
            ref=C.행렬곱(xx,q.T,dd.T,bias,자리수=2)
            for linear in linears.values(): check(ref,linear(xx,q,dd,bias))
    for case in range(3):
        fixture=np.load(ROOT/f'빌드/matrix/case{case}.npz')
        args=[fixture[key] for key in ('x','q','d','bias')]
        for linear in linears.values(): check(fixture['reference'],linear(*args))
        samples={name:[] for name in names}
        profiles={name:[] for name in names}
        for repeat in range(12):
            order=names[repeat%len(names):]+names[:repeat%len(names)]
            if repeat%2: order=order[::-1]
            for name in order:
                start=time.perf_counter()
                actual=linears[name](*args)
                samples[name].append(time.perf_counter()-start)
                profiles[name].append(linears[name].profile.copy())
                check(fixture['reference'],actual)
        row={'shape_M_N_K':[args[0].shape[0],args[1].shape[0],args[0].shape[1]],'samples_seconds':samples,
             'median_seconds':{name:statistics.median(s) for name,s in samples.items()},'profiles':profiles}
        report['matrices'].append(row)
        print('MATRIX',json.dumps({key:value for key,value in row.items() if key not in ('samples_seconds','profiles')}),flush=True)
    report['passed']=True
except Exception as exc:
    report['error']=repr(exc)
    report['traceback']=traceback.format_exc()
    raise
finally:
    (RESULTS/'comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
