import numpy as np
from native_post import ROOT,C
rng=np.random.default_rng(331)
a,b,c=rng.uniform(-3,3,(3,256)).astype(np.float32)
a[0],b[0],c[0]=np.float32(1+2**-23),np.float32(1-2**-23),-1
a[1],b[1],c[1]=np.finfo(np.float32).tiny,.5,0
directory=ROOT/'빌드/npu/next-stage'
for name,array in [('a',a),('b',b),('c',c),('expected',C.fma32(a,b,c))]: array.tofile(directory/f'host-{name}.bin')
