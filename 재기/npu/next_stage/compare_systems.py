"""Isolated-worker end-to-end resource comparison under the exact contract."""
import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack,closing
from pathlib import Path
import numpy as np
from native_post import HERE,ROOT
from system_metrics import Resources

def worker(mode,length):
    from concurrent.futures import ThreadPoolExecutor
    from cpu_linear import CpuLinear
    from parallel_post import ParallelPost
    from joined_prefill import JoinedPrefill
    from adaptive_output_view import AdaptiveOutputViewDevice
    from cached_qwen import CachedQwen,R
    path=HERE/f'results/system-{mode}-{length}.json'
    report=dict(complete=False,mode=mode,length=length,output_tokens=4,runs=[],pid=os.getpid(),logical_cpus=os.cpu_count())
    try:
        with ExitStack() as stack:
            cpu=stack.enter_context(closing(CpuLinear()))
            if mode=='hybrid':
                post=stack.enter_context(closing(ParallelPost(4)))
                pool=stack.enter_context(ThreadPoolExecutor(1))
                device=AdaptiveOutputViewDevice()
                npu=JoinedPrefill(device,finalizer=post,finalizer_executor=pool)
                # Warm all existing graph capacities without timing compilation.
                for frames in (2048,4096,8192):
                    device._device(frames).grouped(np.zeros((4,1,2,4,32),np.int8),np.zeros((4,32,32),np.int8),4)
                def route(x,*args,**kwargs):return (cpu if len(x)==1 else npu)(x,*args,**kwargs)
            else:route=cpu
            g=R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
            model=CachedQwen(g,route,quant_cache=True,context_capacity=260)
            prompt=np.random.default_rng(407).integers(100,30000,256).tolist()[:length]
            warm=model.step(prompt[:3])
            model.step([int(warm.argmax())])
            for repeat in range(2):
                model.reset()
                hashes=[]
                tokens=[]
                with Resources() as resources:
                    stamp=time.perf_counter()
                    logits=model.step(prompt)
                    first=time.perf_counter()-stamp
                    prefill_cpu=time.process_time()-resources.cpu
                    hashes.append(hashlib.sha256(logits.tobytes()).hexdigest())
                    tokens.append(int(logits.argmax()))
                    print(mode,length,repeat,'prefill done',round(first,3),flush=True)
                    decode=[]
                    for _ in range(3):
                        stamp=time.perf_counter()
                        logits=model.step([tokens[-1]])
                        decode.append(time.perf_counter()-stamp)
                        hashes.append(hashlib.sha256(logits.tobytes()).hexdigest())
                        tokens.append(int(logits.argmax()))
                report['runs'].append(dict(repeat=repeat,prefill_seconds=first,prefill_cpu_seconds=prefill_cpu,
                    decode_seconds=decode,hashes=hashes,tokens=tokens,resources=resources.result))
                path.write_text(json.dumps(report,indent=2),encoding='utf8',newline='\n')
                print(mode,length,repeat,'request done',round(resources.result['wall_seconds'],3),flush=True)
            report['complete']=True
            if mode=='hybrid':report['selections']=device.selections
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:path.write_text(json.dumps(report,indent=2),encoding='utf8',newline='\n')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',choices=('cpu','hybrid'))
    parser.add_argument('--length',type=int)
    args=parser.parse_args()
    if args.worker:return worker(args.worker,args.length)
    report=dict(complete=False,pairs=[],energy_measurement='unavailable: amd-smi LoadLibraryA failure; CPU time is not energy',
        throughput_scope='sequential back-to-back requests; not concurrent or saturated serving throughput')
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for length in (16,64,256):
            order=('cpu','hybrid') if length!=64 else ('hybrid','cpu')
            for mode in order:
                print('isolated worker start',mode,length,flush=True)
                subprocess.run([sys.executable,'-X','utf8',__file__,'--worker',mode,'--length',str(length)],check=True)
            a=json.loads((HERE/f'results/system-cpu-{length}.json').read_text())
            b=json.loads((HERE/f'results/system-hybrid-{length}.json').read_text())
            assert a['complete'] and b['complete']
            assert all(r['hashes']==a['runs'][0]['hashes'] for r in a['runs']+b['runs'])
            report['pairs'].append(dict(length=length,all_logits_match=True))
            print('paired hashes matched',length,flush=True)
        report['complete']=True
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        (HERE/'results/system-comparison.json').write_text(json.dumps(report,indent=2),encoding='utf8',newline='\n')

if __name__=='__main__':main()
