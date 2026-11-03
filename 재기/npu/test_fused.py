"""Actual NPU correctness, buffer refresh, tails, cache and paired latency tests."""
import json
import time
from pathlib import Path
import numpy as np
from exact_linear import C, ROOT, difference
from tiled_kernel import NpuTiles
from fused_kernel import NpuFusedTiles
from planned_linear import PlannedLinear

rng = np.random.default_rng(20261030)
report = dict(scope="Actual NPU digit fusion versus original planned backend",passed=False,kernel_checks=0,linear_checks=0,benchmarks=[])
destination = Path(__file__).parent/'results/2026-10-30-fused.json'

def check(ref,actual):
    delta = difference(ref,actual)
    assert delta['different_bits_elements']==0,delta
    report['linear_checks'] += 1

def timed(fn,*args):
    start=time.perf_counter()
    out=fn(*args)
    return out,time.perf_counter()-start

try:
    old_device,new_device=NpuTiles(frames=2048),NpuFusedTiles(frames=2048)
    old,new=PlannedLinear(old_device),PlannedLinear(new_device)
    # Changing retained host views must flush every call, including short tails.
    for i,count in enumerate([1,2048,7,2047,1,33,2,2048]):
        a=rng.integers(-128,128,(count,8,32),dtype=np.int8)
        b=rng.integers(-128,128,(count,32,32),dtype=np.int8)
        if i<2:
            a.fill(-128 if i==0 else 127)
            b.fill(-128)
        dots=a.astype(np.int64)@b.astype(np.int64).transpose(0,2,1)
        expected=dots[:,:4]+256*dots[:,4:]
        np.testing.assert_array_equal(new_device(a,b),expected)
        np.testing.assert_array_equal(old_device(a,b),dots)
        report['kernel_checks']+=2
    print('NPU integer extremes, tails and buffer refresh: PASS',flush=True)
    for m,n,k in [(1,7,32),(4,32,64),(5,35,96),(9,65,128),(3,1057,96),(1,4103,32)]:
        x=rng.uniform(-1,1,(m,k)).astype(np.float32)
        q=rng.integers(-128,128,(n,k),dtype=np.int8)
        d=rng.uniform(-1,1,(n,k//32)).astype(np.float32)
        bias=rng.uniform(-1,1,n).astype(np.float32)
        for digits in (1,2):
            for exponent in (-98,0,100):
                xx,dd=np.ldexp(x,exponent),np.ldexp(d,-max(0,exponent))
                ref=C.행렬곱(xx,q.T,dd.T,bias,자리수=digits)
                check(ref,new(xx,q,dd,bias,자리수=digits))
                check(ref,old(xx,q,dd,bias,자리수=digits))
        misses=new.misses
        q[0,0]=np.int8(int(q[0,0])^1)
        check(C.행렬곱(x,q.T,d.T,bias,자리수=2),new(x,q,d,bias))
        assert new.misses==misses+1
        print(f'linear shape {m,n,k}: PASS',flush=True)
    # Explicit zero, signed zero, tiny values, cancellation and Q4-range weights.
    for value in (0.,-0.,np.nextafter(np.float32(0),np.float32(1)),np.finfo(np.float32).tiny):
        x=np.full((5,64),value,np.float32)
        x[:,1::2]=-x[:,1::2]
        q=rng.integers(-8,8,(35,64),dtype=np.int8)
        d=np.full((35,2),2**-100,np.float32)
        bias=np.zeros(35,np.float32)
        check(C.행렬곱(x,q.T,d.T,bias,자리수=2),new(x,q,d,bias))
    # Finite FP32 does not imply representability by the clamped contract.
    # Preserve reference rejection instead of silently clamping/wrapping digits.
    report['rejections']=[]
    for value in (np.finfo(np.float32).max,-np.finfo(np.float32).max):
        x=np.full((5,64),value,np.float32)
        for name,fn in [('reference',lambda: C.행렬곱(x,q.T,d.T,bias,자리수=2)),('fused',lambda: new(x,q,d,bias))]:
            try:
                fn()
            except AssertionError:
                report['rejections'].append(dict(backend=name,input=float(value),reason='outside digit contract range'))
            else:
                raise AssertionError('out-of-contract input unexpectedly accepted')
    small=PlannedLinear(new_device,budget=4096)
    for value in (1,2,3):
        small.prepare(np.full((32,64),value,np.int8))
        assert small.bytes<=4096
    assert len(small.plans)==2
    for case in range(3):
        fixture=np.load(ROOT/f'빌드/matrix/case{case}.npz')
        args=[fixture[key] for key in ('x','q','d','bias')]
        # Both kernels compiled and buffers initialized above; warm weight plans.
        for fn in (old,new):
            check(fixture['reference'],fn(*args))
        samples={'planned':[],'fused':[]}
        profiles={'planned':[],'fused':[]}
        calls={}
        for repeat in range(6):
            order=[('planned',old),('fused',new)]
            if repeat%2: order.reverse()
            for name,fn in order:
                before=fn.device.calls
                actual,elapsed=timed(fn,*args)
                check(fixture['reference'],actual)
                samples[name].append(elapsed)
                profiles[name].append(fn.profile.copy())
                calls[name]=fn.device.calls-before
        med={key:float(np.median(value)) for key,value in samples.items()}
        row=dict(shape_M_N_K=[args[0].shape[0],args[1].shape[0],args[0].shape[1]],samples_seconds=samples,median_seconds=med,speedup=med['planned']/med['fused'],profiles=profiles,dispatches=calls,
                 allocated_output_bytes_per_call={'planned':old_device.c.numpy().nbytes,'fused':new_device.c.numpy().nbytes})
        report['benchmarks'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('profiles','samples_seconds')}),flush=True)
    report['passed']=True
except Exception as exc:
    report['error']=repr(exc)
    raise
finally:
    destination.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Results: {destination}',flush=True)
