import sys,json,time,statistics
import numpy as np
from native_post import HERE,ROOT,NativePost
from gl_aie.runtime import Device,PlannedLinear
from exact_linear import difference
import trace_model

device=Device('gl',frames=2048)
old=PlannedLinear(device)
new=PlannedLinear(device,finalizer=NativePost())
rows=[]
for case in range(3):
    f=np.load(ROOT/f'빌드/matrix/case{case}.npz')
    args=[f[k] for k in ('x','q','d','bias')]
    for fn in (old,new): fn(*args)
    samples={'numpy':[],'native':[]};profiles={'numpy':[],'native':[]}
    for repeat in range(8):
        order=[('numpy',old),('native',new)]
        if repeat%2:order.reverse()
        for name,fn in order:
            start=time.perf_counter();actual=fn(*args);elapsed=time.perf_counter()-start
            assert difference(f['reference'],actual)['different_bits_elements']==0
            samples[name].append(elapsed);profiles[name].append(fn.profile.copy())
    rows.append({'case':case,'samples':samples,'median_seconds':{k:statistics.median(v) for k,v in samples.items()},'profiles':profiles})
    print(rows[-1]['median_seconds'],flush=True)
(HERE/'results/linear.json').write_text(json.dumps({'passed':True,'rows':rows},indent=2)+'\n',encoding='utf-8')
sys.argv=['trace_model.py',str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'),
          '--backend','gl-native-post','--tokens','1,2,3','--repeats','2',
          '--output',str(HERE/'results/model-native-post.jsonl')]
trace_model.main()
