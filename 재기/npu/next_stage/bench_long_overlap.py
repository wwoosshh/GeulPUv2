"""Longer full-model paired measurements; preserve individual layer samples."""
import hashlib
import json
import time
import traceback
from contextlib import ExitStack, closing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from native_post import HERE, ROOT
from multi_device import MultiDevice
from reuse_device import ReuseDevice
from parallel_post import ParallelPost
from joined_prefill import JoinedPrefill
from cpu_linear import CpuLinear
from cached_qwen import CachedQwen, R


def main():
    report = dict(complete=False, runs=[], scope='Qwen3-0.6B-Q8_0, NPU4/post4, CPU output and decode')
    path = HERE/'results/long-overlap.json'
    def save():
        path.write_text(json.dumps(report, indent=2), encoding='utf8', newline='\n')
    try:
        refs = json.loads((HERE/'results/phase-bench.json').read_text())['runs']
        with ExitStack() as stack:
            cpu = stack.enter_context(closing(CpuLinear()))
            post = stack.enter_context(closing(ParallelPost(4)))
            executor = stack.enter_context(ThreadPoolExecutor(1))
            modes = {'baseline': JoinedPrefill(MultiDevice(4), finalizer=post),
                     'combined': JoinedPrefill(ReuseDevice(), finalizer=post, finalizer_executor=executor)}
            records = []
            selected = 'baseline'
            def route(x, q, *args, **kwargs):
                backend = cpu if len(x) == 1 else modes[selected]
                start = time.perf_counter()
                result = backend(x, q, *args, **kwargs)
                seconds = time.perf_counter()-start
                index = len(records)
                names = ('qkv', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down')
                records.append(dict(layer=index//5, operation=names[index%5] if index < len(model.layers)*5 else 'output',
                    m=len(x), n=q.shape[0], k=q.shape[1], seconds=seconds,
                    backend='cpu' if backend is cpu else selected,
                    profile=dict(getattr(backend, 'profile', {}))))
                return result
            g = R.GR.GGUF(str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'))
            model = CachedQwen(g, route, quant_cache=True, context_capacity=257)
            prompt = np.random.default_rng(407).integers(100, 30000, 256).tolist()
            for selected in modes:
                records.clear()
                model.reset()
                model.step(prompt[:3])
            for length in (64, 256):
                ref = next(r for r in refs if r['length']==length and r['backend']=='cpu')
                for selected in ('baseline','combined','combined','baseline','baseline','combined'):
                    records.clear()
                    model.reset()
                    print('start',length,selected,flush=True)
                    start = time.perf_counter()
                    first = model.step(prompt[:length])
                    elapsed = time.perf_counter()-start
                    prefill_records = list(records)
                    profile = dict(model.last_profile)
                    records.clear()
                    start = time.perf_counter()
                    second = model.step([int(first.argmax())])
                    decode = time.perf_counter()-start
                    hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in (first,second)]
                    if hashes != [ref['prefill_hash'],ref['decode_hash']]:
                        raise AssertionError(f'logit mismatch: {length} {selected} {hashes}')
                    report['runs'].append(dict(length=length, mode=selected, prefill=elapsed, decode=decode,
                        hashes=hashes, profile=profile, linears=prefill_records))
                    save()
                    print('passed',length,selected,round(elapsed,3),flush=True)
            report['complete'] = True
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
