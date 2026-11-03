import hashlib
import json
import statistics
from pathlib import Path

HERE=Path(__file__).parent
RESULTS=HERE/'results'
bench=json.loads((RESULTS/'comparison.json').read_text(encoding='utf-8'))
report={'benchmark_passed':bench['passed'],'kernel_checks':bench['kernel_checks'],'linear_checks':bench['linear_checks'],
        'frontend_commit':'0899bd753e1626e31c3094f5787d0f46ad4f7cc4',
        'text_equal_to_handwritten_aie_cpp':bench['build']['gl_vs_aie_cpp_text_equal'],
        'microbenchmarks':[],'matrices':[],'models':{}}
for kind in ('microbenchmarks','matrices'):
    for row in bench[kind]:
        med=row['median_seconds']
        report[kind].append({key:row[key] for key in ('frames','shape_M_N_K','median_seconds') if key in row}|
            {'c_over_gl':med['c']/med['gl'],'c_unrolled_over_gl':med['c_unrolled']/med['gl'],
             'aie_cpp_over_gl':med['aie_cpp']/med['gl'],
             'sample_range_seconds':{name:[min(v),max(v)] for name,v in row['samples_seconds'].items()}})
for name in ('gl','c_unrolled'):
    path=RESULTS/f'model-{name}.jsonl'
    if not path.exists(): continue
    events=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    linears=[e for e in events if e['event']=='linear']
    residuals=[e for e in events if e['event']=='residual']
    logits=[e for e in events if e['event']=='logits']
    report['models'][name]={'complete':events[-1]['event']=='complete' and events[-1].get('passed'),
         'linear_checks':len(linears),'residual_checks':len(residuals),'logit_checks':len(logits),
         'different_bits_elements':sum(e.get('different_bits_elements',0) for e in events),
         'logit_sha256':[e['sha256'] for e in logits],
         'linear_seconds_by_repeat':{str(r):sum(e['hybrid_seconds'] for e in linears if e['repeat']==r) for r in (0,1)}}
report['all_model_checks_complete']=all(report['models'].get(name,{}).get('complete',False) for name in ('gl','c_unrolled'))
if report['all_model_checks_complete']:
    report['same_model_logits']=report['models']['gl']['logit_sha256']==report['models']['c_unrolled']['logit_sha256']
report['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.gl','.py','.c','.cc','.ps1')}
report['limitations']=['Restricted GL backend, not general-purpose GL NPU support.',
    'GL uses explicit dot32 AIE intrinsic; portable C variants rely on scalar loops and compiler optimization.',
    'Handwritten AIE C++ is the equivalent intrinsic-enabled control and has identical kernel text bytes.',
    'Runtime and ordered FP32 finalization are shared Python/NumPy, outside language comparison.',
    'Measured warm host-visible latency includes transfers/synchronization; not device cycles.',
    'Model sums are instrumented linear calls; not autoregressive generation TPS.']
(RESULTS/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
