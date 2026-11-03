import numpy as np
from planned_linear import PlannedLinear
from tiled_kernel import NpuTiles
from exact_linear import C,difference

rng = np.random.default_rng(20261029)
device = NpuTiles(frames=2048)
aa = rng.integers(-128,128,(1,8,32),dtype=np.int8)
bb = rng.integers(-128,128,(1,32,32),dtype=np.int8)
np.testing.assert_array_equal(device(aa,bb),aa.astype(np.int32)@bb.astype(np.int32).transpose(0,2,1))
linear = PlannedLinear(device)
for m,n,k in [(1,7,32),(5,35,96),(3,1057,96),(1,4103,32)]:
    x = rng.uniform(-1,1,(m,k)).astype(np.float32)
    q = rng.integers(-128,128,(n,k),dtype=np.int8)
    scales = rng.uniform(-1,1,(n,k//32)).astype(np.float32)
    bias = np.zeros(n,np.float32)
    for digits in (1,2):
        for exponent in (-98,0,100):
            xx,dd = np.ldexp(x,exponent),np.ldexp(scales,-max(0,exponent))
            expected = C.행렬곱(xx,q.T,dd.T,bias,자리수=digits)
            actual = linear(xx,q,dd,bias,자리수=digits)
            delta = difference(expected,actual)
            assert delta['different_bits_elements']==0,(m,n,k,digits,exponent,delta,expected,actual)
    misses = linear.misses
    q[0,0] = np.int8(int(q[0,0])^1)
    actual = linear(x,q,scales,bias)
    assert linear.misses==misses+1
    assert difference(C.행렬곱(x,q.T,scales.T,bias,자리수=2),actual)['different_bits_elements']==0
    print(f"planned shape {m,n,k}, precision and cache mutation: PASS",flush=True)
small = PlannedLinear(device,budget=4096)
for value in (1,2,3):
    small.prepare(np.full((32,64),value,np.int8))
    assert small.bytes<=4096
assert len(small.plans)==2
print(f"PASS: cache eviction; dispatches={device.calls}; hits={linear.hits}; misses={linear.misses}",flush=True)
