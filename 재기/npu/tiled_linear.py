import time
import numpy as np
from exact_linear import C

class TiledLinear:
    def __init__(self, device):
        self.device = device
        self.profile = {}

    def __call__(self,x,q,scales,bias,일꾼=8,칸한도=1<<25,자리수=2):
        from tiled_kernel import ROWS, COLS, FRAMES
        x,q,scales = np.asarray(x,np.float32),np.asarray(q),np.asarray(scales,np.float32)
        if x.ndim != 2 or q.ndim != 2 or q.dtype != np.int8 or x.shape[1] != q.shape[1] or x.shape[1]%32 or 자리수 not in (1,2):
            raise ValueError("invalid contract input")
        m,k = x.shape
        n,kb = q.shape[0],k//32
        if scales.shape != (n,kb) or not np.isfinite(x).all() or not np.isfinite(scales).all():
            raise ValueError("invalid scales or nonfinite input")
        t = time.perf_counter()
        digits,e = C.자리로(x,자리수)
        digits = digits.astype(np.int8).reshape(자리수*m,kb,32)
        weights = q.reshape(n,kb,32)
        y = np.zeros((m,n),np.float32)
        packing = time.perf_counter()-t
        device_time = finish_time = 0.0
        # Larger host chunks amortize ordered FMA calls without merging blocks.
        for lo in range(0,n,1024):
            hi = min(n,lo+1024)
            nc = hi-lo
            tiles = (nc+COLS-1)//COLS
            t = time.perf_counter()
            padded = np.zeros((tiles*COLS,kb,32),np.int8)
            padded[:nc] = weights[lo:hi]
            packed = padded.reshape(tiles,COLS,kb,32).transpose(0,2,1,3).reshape(tiles*kb,COLS,32)
            sums = np.empty((자리수*m,kb,tiles*COLS),np.int32)
            packing += time.perf_counter()-t
            for row in range(0,자리수*m,ROWS):
                nr = min(ROWS,자리수*m-row)
                for start in range(0,tiles*kb,FRAMES):
                    stop = min(tiles*kb,start+FRAMES)
                    t = time.perf_counter()
                    indices = np.arange(start,stop)
                    blocks = indices%kb
                    aa = np.zeros((stop-start,ROWS,32),np.int8)
                    aa[:,:nr] = digits[row:row+nr,blocks,:].transpose(1,0,2)
                    packing += time.perf_counter()-t
                    t = time.perf_counter()
                    result = self.device(aa,packed[start:stop])
                    device_time += time.perf_counter()-t
                    t = time.perf_counter()
                    for f,index in enumerate(indices):
                        col,block = divmod(int(index),kb)
                        sums[row:row+nr,block,col*COLS:(col+1)*COLS] = result[f,:nr]
                    packing += time.perf_counter()-t
            t = time.perf_counter()
            sums = sums[:,:,:nc].reshape(자리수,m,kb,nc)
            parts = [np.ldexp(sums[j].astype(np.float32),(8*j+e-(8*자리수-3))[:,:,None]).astype(np.float32) for j in range(자리수)]
            u = (parts[1]+parts[0]).astype(np.float32) if 자리수 == 2 else parts[0]
            yy = np.zeros((m,nc),np.float32)
            for block in range(kb):
                yy = C.fma32(u[:,block],np.broadcast_to(scales[lo:hi,block],yy.shape),yy)
            y[:,lo:hi] = yy
            finish_time += time.perf_counter()-t
        self.profile = dict(packing_seconds=packing,device_and_transfer_seconds=device_time,finalization_seconds=finish_time)
        return C.더(y,np.asarray(bias,np.float32)[None,:])
