"""Bounded immutable host weight plans, batched dispatch, ordered finalization.

Plans are content-addressed, so in-place weight changes cannot reuse stale packing.
This cache is host memory, not device-resident weights.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import time
import numpy as np
from exact_linear import C

@dataclass(frozen=True)
class WeightPlan:
    n: int
    kb: int
    packed: np.ndarray

class PlannedLinear:
    def __init__(self,device,budget=768*1024*1024,finalizer=None,finalizer_executor=None):
        self.device = device
        self.finalizer = finalizer
        self.finalizer_executor = finalizer_executor
        self.budget = budget
        self.plans = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0
        self.profile = {}

    def prepare(self,q):
        q = np.ascontiguousarray(q)
        key = (q.shape,hashlib.sha256(memoryview(q)).digest())
        if key in self.plans:
            self.hits += 1
            self.plans.move_to_end(key)
            return self.plans[key]
        self.misses += 1
        n,k = q.shape
        kb = k//32
        padded = np.zeros(((n+31)//32*32,kb,32),np.int8)
        padded[:n] = q.reshape(n,kb,32)
        packed = np.ascontiguousarray(padded.reshape(-1,32,kb,32).transpose(0,2,1,3)).reshape(-1,32,32)
        packed.flags.writeable = False
        plan = WeightPlan(n,kb,packed)
        while self.plans and self.bytes+packed.nbytes > self.budget:
            _,old = self.plans.popitem(last=False)
            self.bytes -= old.packed.nbytes
        if packed.nbytes <= self.budget:
            self.plans[key] = plan
            self.bytes += packed.nbytes
        return plan

    def __call__(self,x,q,scales,bias,일꾼=8,칸한도=1<<25,자리수=2):
        t = time.perf_counter()
        x,q,scales = np.asarray(x,np.float32),np.asarray(q),np.asarray(scales,np.float32)
        if x.ndim!=2 or q.ndim!=2 or q.dtype!=np.int8 or x.shape[1]!=q.shape[1] or x.shape[1]%32 or 자리수 not in (1,2):
            raise ValueError("invalid contract input")
        m,k = x.shape
        n,kb = q.shape[0],k//32
        if scales.shape!=(n,kb) or not np.isfinite(x).all() or not np.isfinite(scales).all():
            raise ValueError("invalid scales or input")
        plan = self.prepare(q)
        digits,e = C.자리로(x,자리수)
        digits = digits.astype(np.int8).reshape(자리수*m,kb,32)
        fused = getattr(self.device,"fused",False)
        rows = getattr(self.device,"rows",4) if fused else 8
        total_rows = m if fused else 자리수*m
        y = np.zeros((m,n),np.float32)
        packing = time.perf_counter()-t
        device_time = finish_time = 0.0
        pending = []
        def finish_chunk(sums,e,d,digits):
            start = time.perf_counter()
            value = self.finalizer(sums,e,d,digits)
            return value,time.perf_counter()-start
        # Bound intermediate storage independently of the full model/vocabulary.
        width = max(32,min(4096,(16*1024*1024//max(1,자리수*m*kb*4))//32*32))
        for lo in range(0,n,width):
            hi = min(n,lo+width)
            nc = hi-lo
            tiles = (nc+31)//32
            packed = plan.packed[lo//32*kb:(lo//32+tiles)*kb]
            sums = np.empty((total_rows,kb,tiles*32),np.int32)
            for row in range(0,total_rows,rows):
                nr = min(rows,total_rows-row)
                result = np.empty((tiles*kb,rows,32),np.int32)
                for start in range(0,tiles*kb,self.device.frames):
                    stop = min(tiles*kb,start+self.device.frames)
                    t = time.perf_counter()
                    blocks = np.arange(start,stop)%kb
                    aa = np.zeros((stop-start,8,32),np.int8)
                    aa[:,:nr] = digits[row:row+nr,blocks,:].transpose(1,0,2)
                    if fused and 자리수==2:
                        aa[:,4:4+nr] = digits[m+row:m+row+nr,blocks,:].transpose(1,0,2)
                    packing += time.perf_counter()-t
                    t = time.perf_counter()
                    result[start:stop] = self.device(aa,packed[start:stop])
                    device_time += time.perf_counter()-t
                t = time.perf_counter()
                sums[row:row+nr] = result.reshape(tiles,kb,rows,32)[:,:,:nr,:].transpose(2,1,0,3).reshape(nr,kb,tiles*32)
                packing += time.perf_counter()-t
            t = time.perf_counter()
            if fused and self.finalizer is not None:
                if self.finalizer_executor is None:
                    y[:,lo:hi] = self.finalizer(sums[:,:,:nc],e,scales[lo:hi],자리수)
                    finish_time += time.perf_counter()-t
                else:
                    # Future owns this chunk's sums; subsequent chunks allocate new
                    # storage. Bound retained chunks to two, and never reuse them.
                    pending.append((lo,hi,self.finalizer_executor.submit(finish_chunk,sums[:,:,:nc],e,scales[lo:hi],자리수)))
                    if len(pending)>=2:
                        left,right,future=pending.pop(0)
                        y[:,left:right],seconds=future.result()
                        finish_time+=seconds
                continue
            if fused:
                u = np.ldexp(sums[:,:,:nc].astype(np.float32),(e-(8*자리수-3))[:,:,None]).astype(np.float32)
            else:
                sums = sums[:,:,:nc].reshape(자리수,m,kb,nc)
                parts = [np.ldexp(sums[j].astype(np.float32),(8*j+e-(8*자리수-3))[:,:,None]).astype(np.float32) for j in range(자리수)]
                u = (parts[1]+parts[0]).astype(np.float32) if 자리수==2 else parts[0]
            yy = np.zeros((m,nc),np.float32)
            for block in range(kb):
                yy = C.fma32(u[:,block],np.broadcast_to(scales[lo:hi,block],yy.shape),yy)
            y[:,lo:hi] = yy
            finish_time += time.perf_counter()-t
        for left,right,future in pending:
            y[:,left:right],seconds=future.result()
            finish_time+=seconds
        self.profile = dict(packing_seconds=packing,device_and_transfer_seconds=device_time,finalization_seconds=finish_time)
        return C.더(y,np.asarray(bias,np.float32)[None,:])
