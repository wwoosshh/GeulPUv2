import unittest
import numpy as np
from exact_linear import ExactLinear, C, difference

class ContractTest(unittest.TestCase):
    def test_finalization(self):
        # CPU integer oracle tests host arithmetic, not NPU execution.
        dot = lambda a, b: (a.astype(np.int64)*b.astype(np.int64)).sum(axis=1).astype(np.int32)
        rng = np.random.default_rng(42)
        for m, n, k in [(1, 19, 32), (3, 7, 96), (2, 17, 128)]:
            x = rng.normal(size=(m,k)).astype(np.float32)
            x[0,:32] = 0
            q = rng.integers(-128,128,(n,k),dtype=np.int8)
            scales = rng.normal(size=(n,k//32)).astype(np.float32)
            bias = rng.normal(size=n).astype(np.float32)
            for digits in (1, 2):
                ref = C.행렬곱블라스(x,q,scales,bias,자리수=digits)
                got = ExactLinear(dot, 37)(x,q,scales,bias,자리수=digits)
                self.assertEqual(difference(ref,got)["different_bits_elements"],0)

if __name__ == "__main__":
    unittest.main()
