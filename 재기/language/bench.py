import array
import sys
import time

N=1048576
use_numpy=sys.argv[1]=='numpy'
if use_numpy:
    import numpy as np
    a=np.arange(N,dtype=np.int64)&65535
    b=np.zeros_like(a)
else:
    a=array.array('q',(i&65535 for i in range(N)))
    b=array.array('q',[0])*N

def process(mode):
    if use_numpy:
        if mode==0: np.copyto(b,a)
        elif mode==1:
            np.multiply(a,3,out=b); np.add(b,7,out=b); np.bitwise_and(b,65535,out=b)
        elif mode==2: np.copyto(b.reshape(-1,32,32),a.reshape(-1,32,32).transpose(0,2,1))
        else: return int(np.sum(a,dtype=np.int64))
    else:
        if mode==3:
            total=0
            for i in range(N): total+=a[i]
            return total
        if mode==2:
            for t in range(0,N,1024):
                for r in range(32):
                    for c in range(32): b[t+c*32+r]=a[t+r*32+c]
        else:
            for i in range(N):
                if mode==0: b[i]=a[i]
                else: b[i]=(a[i]*3+7)&65535
    return 0

for mode in range(4):
    for rep in range(8):
        start=time.perf_counter_ns()
        total=process(mode)
        elapsed=(time.perf_counter_ns()-start)/1e6
        errors=0
        if mode==3: errors=int(total!=34359214080)
        else:
            for i,value in enumerate(b):
                expected=i&65535
                if mode==1: expected=(expected*3+7)&65535
                if mode==2: expected=((i//1024)*1024+(i%32)*32+(i%1024)//32)&65535
                errors+=int(value!=expected)
                total+=int(value)
        print(mode,rep,elapsed,total,errors,flush=True)
        assert errors==0
