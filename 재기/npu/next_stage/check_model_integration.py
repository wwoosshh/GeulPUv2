"""Stateful end-to-end correctness; no performance measurements."""
import ctypes
import hashlib
import json
import traceback
from contextlib import ExitStack,closing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from native_post import HERE,ROOT
from adaptive_output_view import AdaptiveOutputViewDevice
from joined_prefill import JoinedPrefill
from parallel_post import ParallelPost
from cpu_linear import CpuLinear
from cached_qwen import CachedQwen,R

def bits(a):return hashlib.sha256(a.tobytes()).hexdigest()
def kv_hash(model):
    h=hashlib.sha256()
    for pair in model.cache:
        for heads in pair:
            for a in heads:h.update(a.tobytes())
    return h.hexdigest()

def main():
    report=dict(complete=False,checks=0,reset_checks=0,scenarios=[])
    def save():
        (HERE/'results/model-integration.json').write_text(json.dumps(report,indent=2),encoding='utf8',newline='\n')
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        with ExitStack() as stack:
            cpu=stack.enter_context(closing(CpuLinear()))
            post=stack.enter_context(closing(ParallelPost(4)))
            pool=stack.enter_context(ThreadPoolExecutor(1))
            device=AdaptiveOutputViewDevice()
            npu=JoinedPrefill(device,finalizer=post,finalizer_executor=pool)
            route_counts=dict(cpu=0,npu=0)
            def route(x,*args,**kwargs):
                key='cpu' if len(x)==1 else 'npu'
                route_counts[key]+=1
                return (cpu if key=='cpu' else npu)(x,*args,**kwargs)
            g=R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
            model=CachedQwen(g,cpu,quant_cache=True,context_capacity=80)
            tokens=np.random.default_rng(734).integers(100,30000,65).tolist()
            references={}
            def reset(backend):
                model.backend=backend
                model.reset()
                assert model.pos==0 and all(v is None for v in model.cache)
                assert all(not head for layer in model.quantized for head in layer)
                report['reset_checks']+=1
            def run(chunks,expected=None):
                offset=0
                steps=[]
                saved=[]
                for index in range(len(chunks)+3):
                    if index<len(chunks):
                        block=tokens[offset:offset+chunks[index]]
                        offset+=chunks[index]
                    else:
                        block=[steps[-1]['token']]
                    logits=model.step(block)
                    entry=dict(input_tokens=block,position=model.pos,logit_hash=bits(logits),kv_hash=kv_hash(model),token=int(logits.argmax()),kv_bytes=model.kv_bytes)
                    if expected is not None:
                        reference,reference_logits=expected[index]
                        assert entry==reference,f'state mismatch at step {index}'
                        assert np.array_equal(logits.view(np.uint32),reference_logits.view(np.uint32))
                        report['checks']+=1
                    for old,digest in saved:assert bits(old)==digest,'prior logits overwritten'
                    saved.append((logits,entry['logit_hash']))
                    steps.append(entry)
                return list(zip(steps,[a.copy() for a,_ in saved]))
            for label,chunks in (('prefill17',[17]),('prefill65',[65]),('chunked17',[7,10]),('reset_repeat17',[17])):
                key=tuple(chunks)
                if key not in references:
                    reset(cpu)
                    references[key]=run(chunks)
                    print('CPU reference ready',label,flush=True)
                reset(route)
                verified=run(chunks,references[key])
                report['scenarios'].append(dict(name=label,chunks=chunks,steps=[r for r,_ in verified]))
                save()
                print('integration passed',label,flush=True)
            report['selections']=device.selections
            report['handle_loads']={str(k):v.handle_loads for k,v in device.devices.items()}
            assert all(v==1 for v in report['handle_loads'].values())
            report['routes']=route_counts
            report['complete']=True
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        save()

if __name__=='__main__':main()
