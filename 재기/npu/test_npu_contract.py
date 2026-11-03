"""Actual-device boundary tests. Run separately from CPU unit tests."""
import numpy as np
from exact_kernel import NpuDot32, CAPACITY
from exact_linear import ExactLinear, C, difference

dot = NpuDot32()
rng = np.random.default_rng(20261028)
for size in (CAPACITY, 17, 1024, CAPACITY):
    a = rng.integers(-128,128,(size,32),dtype=np.int8)
    b = rng.integers(-128,128,a.shape,dtype=np.int8)
    a[0] = -128
    b[0] = -128
    expected = (a.astype(np.int64)*b.astype(np.int64)).sum(axis=1)
    for repeat in range(3):
        np.testing.assert_array_equal(dot(a,b),expected)
    print(f"dot32 changing inputs tasks={size} repeats=3 exact=True",flush=True)
linear = ExactLinear(dot)
for exponent in (-98, -30, 0, 30, 100):
    x = np.ldexp(rng.uniform(-1,1,(2,96)).astype(np.float32),exponent)
    x[0,:32] = 0
    x[0,0] = -0.0
    q = rng.integers(-128,128,(9,96),dtype=np.int8)
    scales = np.ldexp(rng.uniform(-1,1,(9,3)).astype(np.float32),-max(0,exponent))
    bias = np.zeros(9,np.float32)
    for digits in (1,2):
        ref = C.행렬곱(x,q.T,scales.T,bias,자리수=digits)
        actual = linear(x,q,scales,bias,자리수=digits)
        delta = difference(ref,actual)
        assert delta["different_bits_elements"] == 0, delta
    print(f"linear exponent={exponent} both contracts exact=True",flush=True)
print(f"PASS actual NPU dispatches={dot.calls}",flush=True)
