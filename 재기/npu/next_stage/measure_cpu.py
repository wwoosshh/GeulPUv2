import json,time,hashlib
from cpu_linear import CpuLinear
from native_post import ROOT,HERE
from cached_qwen import CachedQwen,R
backend=CpuLinear()
g=R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
model=CachedQwen(g,backend)
rounds=[]
for rep in range(2):
    model.reset()
    feed=[1,2,3]
    rows=[]
    for step in range(4):
        start=time.perf_counter()
        proc=time.process_time()
        logits=model.step(feed)
        elapsed=time.perf_counter()-start
        cpu=time.process_time()-proc
        feed=[int(logits.argmax())]
        row=dict(seconds=elapsed,cpu_seconds=cpu,token=feed[0],hash=hashlib.sha256(logits.tobytes()).hexdigest())
        rows.append(row)
        print(rep,step,row,flush=True)
    rounds.append(rows)
    (HERE/'results/viability-cpu.json').write_text(json.dumps(rounds,indent=2),encoding='utf-8')
backend.close()
