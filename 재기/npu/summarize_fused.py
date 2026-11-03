"""Summarize actual-device tests and completed model traces without extrapolation."""
import hashlib
import json
from pathlib import Path
import statistics

here=Path(__file__).parent
results=here/'results'
bench=json.loads((results/'2026-10-30-fused.json').read_text(encoding='utf-8'))
report={'passed':bench['passed'],'kernel_checks':bench['kernel_checks'],'linear_checks':bench['linear_checks'],'matrix':[],'models':{}}
for row in bench['benchmarks']:
    profile={name:{key:statistics.median(p[key] for p in entries) for key in entries[0]} for name,entries in row['profiles'].items()}
    report['matrix'].append({key:row[key] for key in ('shape_M_N_K','median_seconds','speedup','dispatches','allocated_output_bytes_per_call')}|{'median_profiles':profile})
for backend in ('planned','fused','fused-length1','fused-length16'):
    path=results/f'2026-10-30-model-{backend}.jsonl'
    if not path.exists(): continue
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    done=rows[-1].get('event')=='complete' and rows[-1].get('passed') is True
    report['passed'] &= done
    linears=[r for r in rows if r['event']=='linear']
    residuals=[r for r in rows if r['event']=='residual']
    logits=[r for r in rows if r['event']=='logits']
    report['models'][backend]={'complete':done,'tokens':rows[0]['tokens'],'linear_checks':len(linears),'residual_checks':len(residuals),'logit_checks':len(logits),
        'different_bits_elements':sum(r.get('different_bits_elements',0) for r in rows),
        'logits_sha256':[r['sha256'] for r in logits],
        'repeat_differences':[r['repeat_difference'] for r in logits],
        'linear_seconds_by_repeat':{str(i):sum(r['hybrid_seconds'] for r in linears if r['repeat']==i) for i in sorted({r['repeat'] for r in linears})},
        'profile_seconds_by_repeat':{str(i):{key:sum(r['profile'][key] for r in linears if r['repeat']==i) for key in linears[0]['profile']} for i in sorted({r['repeat'] for r in linears})},
        'completion':rows[-1]}
if all(name in report['models'] for name in ('planned','fused')):
    report['same_model_logits_across_backends']=report['models']['planned']['logits_sha256']==report['models']['fused']['logits_sha256']
    report['passed'] &= report['same_model_logits_across_backends']
report['source_sha256']={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in ('fused_kernel.py','tiled_kernel.py','planned_linear.py','test_fused.py','trace_model.py')}
report['limitations']=['NPU kernel is AIE C++; GL frontend/lowering is not implemented.','Model timing sums instrumented linear calls, not generation TPS or uninstrumented end-to-end latency.','Matrix timing uses six alternating warm samples per backend, one session.','Output byte counts are allocated XRT buffer sizes, not hardware bus counters.']
(results/'2026-10-30-summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
