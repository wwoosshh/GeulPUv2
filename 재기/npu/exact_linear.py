"""Contract Q8/Q4 linear: NPU integer sums, CPU ordered FP32 finalization."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "계약"))
import 정수계약 as C

class ExactLinear:
    def __init__(self, dot, capacity=16384):
        self.dot, self.capacity = dot, capacity

    def __call__(self, x, q, scales, bias, 일꾼=8, 칸한도=1 << 25, 자리수=2):
        x, q = np.asarray(x, np.float32), np.asarray(q)
        if q.dtype != np.int8 or q.ndim != 2 or x.ndim != 2 or x.shape[1] != q.shape[1] or x.shape[1] % 32:
            raise ValueError("requires FP32 [M,K] and int8 [N,K], K divisible by 32")
        if 자리수 not in (1, 2) or not np.isfinite(x).all():
            raise ValueError("only finite contract inputs and one/two digits supported")
        digits, exponents = C.자리로(x, 자리수)
        m, k = x.shape
        n, kb = q.shape[0], k // 32
        scales = np.asarray(scales, np.float32)
        if scales.shape != (n, kb) or not np.isfinite(scales).all():
            raise ValueError("invalid scales")
        digits = digits.astype(np.int8).reshape(자리수, m, kb, 32)
        weights = q.reshape(n, kb, 32)
        y = np.zeros((m, n), np.float32)
        # Bound temporary storage; preserve ascending block FMA order for every output.
        width = max(1, min(n, self.capacity // max(1, m * kb * 자리수)))
        for lo in range(0, n, width):
            hi = min(n, lo + width)
            shape = (자리수, m, kb, hi-lo)
            sums = np.empty(np.prod(shape), np.int32)
            for start in range(0, sums.size, self.capacity):
                stop = min(sums.size, start + self.capacity)
                j, row, block, col = np.unravel_index(np.arange(start, stop), shape)
                sums[start:stop] = self.dot(digits[j, row, block], weights[lo+col, block])
            sums = sums.reshape(shape)
            parts = [np.ldexp(sums[j].astype(np.float32),
                     (8*j + exponents - (8*자리수-3))[:, :, None]).astype(np.float32)
                     for j in range(자리수)]
            u = (parts[1] + parts[0]).astype(np.float32) if 자리수 == 2 else parts[0]
            out = np.zeros((m, hi-lo), np.float32)
            for block in range(kb):
                out = C.fma32(u[:, block], np.broadcast_to(scales[lo:hi, block], out.shape), out)
            y[:, lo:hi] = out
        return C.더(y, np.asarray(bias, np.float32)[None, :])

def difference(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    aa, bb = a.view(np.uint32), b.view(np.uint32)
    mask = aa != bb
    result = {"elements": a.size, "different_bits_elements": int(mask.sum()),
              "max_abs": float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)), initial=0))}
    if mask.any():
        index = tuple(np.argwhere(mask)[0])
        result["first"] = {"index": list(map(int, index)), "reference_hex": f"{aa[index]:08x}", "actual_hex": f"{bb[index]:08x}"}
    return result
