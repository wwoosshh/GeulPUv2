"""Summarize completed full-model runs; verify stored logits across backends."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('baseline',type=Path)
p.add_argument('optimized',type=Path)
p.add_argument('--output',type=Path,required=True)
args = p.parse_args()
summaries = []
all_rows = []
for path in (args.baseline,args.optimized):
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    assert rows[-1]['event']=='complete' and rows[-1]['passed']
    assert all(r.get('different_bits_elements',0)==0 for r in rows)
    all_rows.append(rows)
    linear = [r for r in rows if r['event']=='linear']
    logits = [r for r in rows if r['event']=='logits']
    assert len(logits)>=2
    reference = np.load(path.with_suffix('.baseline-logits.npy'))
    for r in logits:
        actual = np.load(path.with_suffix(f".repeat-{r['repeat']}-logits.npy"))
        assert np.array_equal(reference.view(np.uint32),actual.view(np.uint32))
    summaries.append(dict(backend=rows[0]['backend'],tokens=rows[0]['tokens'],
        linear_calls=len(linear),elements=sum(r['elements'] for r in linear),
        npu_dispatches=rows[-1]['npu_dispatches'],logit_hash=logits[0]['sha256'],
        plan_cache_hits=rows[-1].get('plan_cache_hits'),plan_cache_misses=rows[-1].get('plan_cache_misses'),
        host_plan_bytes=rows[-1].get('host_plan_bytes'),
        repetitions=[dict(repeat=i,cpu_seconds=sum(r['cpu_seconds'] for r in linear if r['repeat']==i),
            hybrid_seconds=sum(r['hybrid_seconds'] for r in linear if r['repeat']==i),
            profile={k:(sum(r.get('profile',{}).get(k,0) for r in linear if r['repeat']==i)
                        if any(k in r.get('profile',{}) for r in linear if r['repeat']==i) else None)
                     for k in ['packing_seconds','device_and_transfer_seconds','finalization_seconds']})
            for i in sorted({r['repeat'] for r in linear})]))
assert all_rows[0][0]['tokens']==all_rows[1][0]['tokens']
assert all_rows[0][0]['model_sha256']==all_rows[1][0]['model_sha256']
hashes = [{(r['repeat'],r['call']):r['sha256'] for r in rows if r['event']=='linear'} for rows in all_rows]
assert hashes[0]==hashes[1]
a = np.load(args.baseline.with_suffix('.repeat-1-logits.npy'))
b = np.load(args.optimized.with_suffix('.repeat-1-logits.npy'))
assert np.array_equal(a.view(np.uint32),b.view(np.uint32))
old,new = [s['repetitions'][1]['hybrid_seconds'] for s in summaries]
report = dict(passed=True,all_linear_hashes_match=True,saved_logits_bit_equal=True,
              warm_speedup=old/new,warm_seconds_reduction=1-new/old,runs=summaries)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
