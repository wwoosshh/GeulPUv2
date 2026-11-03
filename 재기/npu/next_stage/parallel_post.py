"""Parallelize independent output columns, preserving each FMA chain."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from native_post import NativePost, C


class ParallelPost(NativePost):
    def __init__(self, workers=4):
        if workers not in (1, 2, 4):
            raise ValueError("workers must be 1, 2 or 4")
        super().__init__()
        self.workers = workers
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def __call__(self, sums, exponents, scales, digits):
        sums = np.ascontiguousarray(sums, np.int32)
        exponents = np.ascontiguousarray(exponents, np.int64)
        scales = np.ascontiguousarray(scales, np.float32)
        m, kb, n = sums.shape
        if exponents.shape != (m, kb) or scales.shape != (n, kb):
            raise ValueError("post shape mismatch")
        if digits not in (1, 2) or n == 0:
            raise ValueError("invalid precision or empty output")
        ftz = int(C.비정규수버림)
        width = (n + self.workers - 1) // self.workers

        def finish(lo):
            count = min(width, n-lo)
            output = np.empty((m, count), np.float32)
            # Input retains its original full row stride. Workers only read it.
            self.lib.geul_finish(sums.ctypes.data + lo*4, exponents.ctypes.data,
                                 scales.ctypes.data + lo*kb*4, output.ctypes.data,
                                 m, count, kb, n, 8*digits-3, ftz)
            return output

        if self.workers == 1:
            return finish(0)
        return np.concatenate(list(self.pool.map(finish, range(0,n,width))), axis=1)

    def close(self):
        self.pool.shutdown(wait=True)
