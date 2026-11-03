import json,time,traceback,hashlib,sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from native_post import HERE,ROOT,C,NativePost
from cached_qwen import CachedQwen,R
from gl_aie.runtime import Device,PlannedLinear
from exact_linear import difference

report={'passed':False,'scope':'Qwen3 Q8_0 cached FP32 K/V greedy decode prototype; GL NPU linears + native CPU post; CPU reference attention','rounds':[]}
PROFILE='--profile' in sys.argv
stem='generation-profile' if PROFILE else 'generation'
executor=None
try:
    g=R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
    prompt=[1,2,3]
    prefix=prompt.copy();expected=[];tokens=[]
    # Independent full-prefix reference before timing cached execution.
    for step in range(3 if PROFILE else 4):
        logits,_=R.앞으로(g,[prefix],말=False,끝행만=True)
        expected.append(logits)
        token=int(logits.argmax(axis=1)[0]);tokens.append(token);prefix.append(token)
    device=Device('gl',frames=2048)
    executor=ThreadPoolExecutor(max_workers=1)
    linear=PlannedLinear(device,finalizer=NativePost(),finalizer_executor=executor)
    model=CachedQwen(g,linear)
    for repeat in range(2):
        model.reset();feed=prompt.copy();times=[];produced=[];profiles=[]
        for step in range(3 if PROFILE else 4):
            start=time.perf_counter();actual=model.step(feed);times.append(time.perf_counter()-start)
            profiles.append(model.last_profile.copy())
            delta=difference(expected[step],actual)
            if delta['different_bits_elements']:
                np.savez(HERE/'results/generation-mismatch.npz',reference=expected[step],actual=actual)
                raise AssertionError(delta)
            token=int(actual.argmax(axis=1)[0]);produced.append(token);feed=[token]
        assert produced==tokens
        row={'repeat':repeat,'prefill_seconds':times[0],'decode_seconds':times[1:],
             'decode_tokens_per_second':(len(times)-1)/sum(times[1:]),'generated_token_ids':produced,'kv_bytes':model.kv_bytes,'profiles':profiles,
             'last_logits_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'different_bits_elements':0}
        report['rounds'].append(row);print(json.dumps(row),flush=True)
    report['passed']=True
    report['limitations']=['Fixed token IDs, no tokenizer/chat template or EOS stopping.','Only 3 decode steps per round, no reliable p95 or long-context inference.','FP32 K/V projections are cached, but CPU reference attention re-quantizes history each step.','GL device kernel is integrated through Python prototype; production GL engine ABI is not implemented.']
except Exception as exc:
    report.update(error=repr(exc),traceback=traceback.format_exc());raise
finally:
    if executor is not None: executor.shutdown(wait=True)
    (HERE/f'results/{stem}.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
