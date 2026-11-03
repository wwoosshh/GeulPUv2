import numpy as np
from tiled_kernel import NpuTiles
from tiled_linear import TiledLinear
from exact_linear import C, difference

rng = np.random.default_rng(812)
device = NpuTiles()
for frames in (128,1,17,128):
    a = rng.integers(-128,128,(frames,8,32),dtype=np.int8)
    b = rng.integers(-128,128,(frames,32,32),dtype=np.int8)
    a[0,0] = -128
    b[0,0] = -128
    expected = a.astype(np.int64) @ b.astype(np.int64).transpose(0,2,1)
    for _ in range(2):
        np.testing.assert_array_equal(device(a,b),expected)
print("tile changing input/extreme/padding: PASS",flush=True)
linear = TiledLinear(device)
for m,n,k in [(1,9,32),(3,35,96),(5,67,160),(2,1031,96)]:
    x = rng.uniform(-1,1,size=(m,k)).astype(np.float32)
    q = rng.integers(-128,128,(n,k),dtype=np.int8)
    d = rng.normal(size=(n,k//32)).astype(np.float32)
    bias = rng.normal(size=n).astype(np.float32)
    for digits in (1,2):
        for exponent in (-98,0,100):
            xx = np.ldexp(x,exponent)
            dd = np.ldexp(d,-max(0,exponent))
            ref = C.행렬곱(xx,q.T,dd.T,bias,자리수=digits)
            actual = linear(xx,q,dd,bias,자리수=digits)
            delta = difference(ref,actual)
            assert delta['different_bits_elements']==0,delta
    print(f"linear m={m} n={n} k={k}: PASS",flush=True)
print(f"all passed, NPU dispatches={device.calls}",flush=True)
