"""Validate a proposed lowering rule, not a new NPU performance result.

For the two-digit Q8/Q4 contract, exact T=256*S1+S0 can be converted to
FP32 once and scaled. Do not apply this to arbitrary floating-point sums.
"""
import json
from pathlib import Path
import numpy as np
from exact_linear import C, ROOT, difference

rng=np.random.default_rng(29103)
s0=rng.integers(-524288,524289,8192,dtype=np.int32)
s1=rng.integers(-262144,262145,8192,dtype=np.int32)
# Include cancellation, zeros, extrema, and values around FP32 integer ties.
edge0=np.array([0,1,-1,3,-3,127,-128,524288,-524288],np.int32)
edge1=np.array([0,1,-1,65536,-65536,131072,-131072,262144,-262144],np.int32)
a,b=np.meshgrid(edge0,edge1)
s0=np.concatenate([s0,a.ravel()])
s1=np.concatenate([s1,b.ravel()])
t64=s1.astype(np.int64)*256+s0
assert np.max(np.abs(t64))<2**31
t=t64.astype(np.int32)
count=0
for exponent in range(-98,101):
    old=C.더(np.ldexp(s1.astype(np.float32),exponent-5),np.ldexp(s0.astype(np.float32),exponent-13))
    new=np.ldexp(t.astype(np.float32),exponent-13)
    delta=difference(old,new)
    assert delta['different_bits_elements']==0,(exponent,delta)
    count+=old.size
fixtures=[]
for case in range(3):
    path=ROOT/f'빌드/matrix/case{case}.npz'
    data=np.load(path)
    m,k=data['x'].shape
    n=data['q'].shape[0]
    sums=np.fromfile(ROOT/f'빌드/matrix/case{case}.expected.bin',np.int32).reshape(2,m,k//32,n)
    t=sums[1].astype(np.int64)*256+sums[0].astype(np.int64)
    assert np.max(np.abs(t))<=2**26
    u=np.ldexp(t.astype(np.int32).astype(np.float32),(data['e']-13)[:,:,None])
    y=np.zeros((m,n),np.float32)
    for block in range(k//32):
        y=C.fma32(u[:,block],np.broadcast_to(data['d'][:,block],y.shape),y)
    y=C.더(y,data['bias'][None,:])
    delta=difference(data['reference'],y)
    assert delta['different_bits_elements']==0
    fixtures.append(dict(shape_M_N_K=[m,n,k],**delta))
report=dict(passed=True,scope='CPU verification of exact two-digit integer fusion; not an NPU implementation or speed measurement',exponent_cases=199,partial_sum_pairs_checked=count,linear_fixtures=fixtures)
destination=Path(__file__).parent/'results/2026-10-29-digit-fusion.json'
destination.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
