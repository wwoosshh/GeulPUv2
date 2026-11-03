"""Alternating same-session packing comparison after model measurement."""
import json,time
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from phase_bench import HERE,ROOT,R,Device,PlannedLinear,NativePost,CpuLinear,checked
from joined_prefill import JoinedPrefill

g=R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
w=R.가중(g,'blk.0.ffn_up.weight')
rng=np.random.default_rng(967)
x=rng.standard_normal((256,w[1].shape[1])).astype(np.float32)
bias=np.zeros(w[1].shape[0],np.float32)
rows=[]
with closing(CpuLinear()) as cpu,ThreadPoolExecutor(1) as pool:
    device=Device('gl',2048)
    old=PlannedLinear(device,finalizer=NativePost(),finalizer_executor=pool)
    new=JoinedPrefill(device,finalizer=NativePost())
    ref=cpu(x,w[1],w[2],bias)
    for name,backend in [('old_warm',old),('new_warm',new),('cpu',cpu),('old',old),('new',new),('new',new),('old',old),('cpu',cpu)]:
        t=time.perf_counter()
        result=backend(x,w[1],w[2],bias)
        elapsed=time.perf_counter()-t
        checked(ref,result)
        row=dict(backend=name,seconds=elapsed,profile=backend.profile.copy())
        rows.append(row)
        print(row,flush=True)
        (HERE/'results/joined-paired.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
