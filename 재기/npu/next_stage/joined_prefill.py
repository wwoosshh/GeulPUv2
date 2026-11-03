"""Research prefill linear. Pack adjacent four-row groups into one submission."""
import time
import numpy as np
from native_post import C
from planned_linear import PlannedLinear


class JoinedPrefill(PlannedLinear):
    def __call__(self, x, q, scales, bias, 자리수=2):
        started=time.perf_counter()
        m,k=x.shape
        n=q.shape[0]
        if m==1:
            return super().__call__(x,q,scales,bias,자리수=자리수)
        if k%32 or q.shape[1]!=k or q.dtype!=np.int8 or 자리수!=2:
            raise ValueError("expected two-digit Q8 aligned prefill")
        kb=k//32
        if scales.shape!=(n,kb) or bias.shape!=(n,):
            raise ValueError("weight scale or bias shape mismatch")
        if not np.isfinite(x).all() or not np.isfinite(scales).all():
            raise ValueError('finite input and scales required')
        plan=self.prepare(q)
        digits,e=C.자리로(x,자리수)
        groups=(m+3)//4
        # [group, depth block, digit, row, 32] matches the eight-row GL ABI.
        padded=np.zeros((2,groups*4,kb,32),np.int8)
        padded[:,:m]=digits.reshape(2,m,kb,32)
        inputs=padded.reshape(2,groups,4,kb,32).transpose(1,3,0,2,4)
        out=np.empty((m,n),np.float32)
        tiles_per_chunk=max(1,min(128,(8<<20)//(groups*4*kb*32*4)))
        device_seconds=0.0
        post_seconds=0.0
        wait_seconds=0.0
        pending=[]
        def finish(sums, scales):
            stamp=time.perf_counter()
            value=self.finalizer(sums,e,scales,자리수)
            return value,time.perf_counter()-stamp
        self.submissions=0
        self.useful_frames=0
        self.submitted_frames=0
        for first in range(0,(n+31)//32,tiles_per_chunk):
            lo=first*32
            hi=min(n,lo+tiles_per_chunk*32)
            tiles=(hi-lo+31)//32
            span=tiles*kb
            total=groups*span
            result=np.empty((total,4,32),np.int32)
            if hasattr(self.device, 'grouped'):
                stamp=time.perf_counter()
                result=self.device.grouped(inputs,plan.packed[first*kb:first*kb+span],groups)
                device_seconds+=time.perf_counter()-stamp
                self.submissions+=self.device.last_calls
                self.useful_frames+=total
                self.submitted_frames+=self.device.last_calls*self.device.frames
            for start in range(0,total if not hasattr(self.device, 'grouped') else 0,self.device.frames):
                stop=min(start+self.device.frames,total)
                index=np.arange(start,stop)
                weight_index=index%span
                aa=inputs[index//span,weight_index%kb].reshape(-1,8,32)
                bb=plan.packed[first*kb+weight_index]
                stamp=time.perf_counter()
                result[start:stop]=self.device(aa,bb)
                device_seconds+=time.perf_counter()-stamp
                self.submissions+=1
                self.useful_frames+=stop-start
                self.submitted_frames+=self.device.frames
            sums=result.reshape(groups,tiles,kb,4,32).transpose(0,3,2,1,4).reshape(groups*4,kb,tiles*32)
            stamp=time.perf_counter()
            if self.finalizer_executor is None:
                out[:,lo:hi],seconds=finish(sums[:m,:,:hi-lo],scales[lo:hi])
                post_seconds+=seconds
            else:
                pending.append((lo,hi,self.finalizer_executor.submit(finish,sums[:m,:,:hi-lo],scales[lo:hi])))
                if len(pending)>=2:
                    left,right,future=pending.pop(0)
                    out[:,left:right],seconds=future.result()
                    post_seconds+=seconds
            wait_seconds+=time.perf_counter()-stamp
        for left,right,future in pending:
            stamp=time.perf_counter()
            out[:,left:right],seconds=future.result()
            post_seconds+=seconds
            wait_seconds+=time.perf_counter()-stamp
        self.profile=dict(device_and_transfer_seconds=device_seconds,
                          finalization_seconds=post_seconds,
                          packing_seconds=time.perf_counter()-started-device_seconds-wait_seconds,
                          post_wait_seconds=wait_seconds)
        return C.더(out,bias[None,:])
