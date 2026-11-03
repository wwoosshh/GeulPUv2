import json,hashlib
from pathlib import Path
HERE=Path(__file__).parent
RESULTS=HERE/'results'
report={}
for key in ('post','linear','device','device-vector','overlap','host','generation','generation-profile'):
    p=RESULTS/f'{key}.json'
    if p.exists(): report[key]=json.loads(p.read_text(encoding='utf-8'))
p=RESULTS/'model-native-post.jsonl'
if p.exists():
    events=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]
    ls=[r for r in events if r['event']=='linear']
    report['model']={'complete':events[-1]['event']=='complete' and events[-1].get('passed'),
        'linear_checks':len(ls),'residual_checks':sum(r['event']=='residual' for r in events),
        'logit_checks':sum(r['event']=='logits' for r in events),
        'different_bits_elements':sum(r.get('different_bits_elements',0) for r in events),
        'linear_seconds':{str(i):sum(r['hybrid_seconds'] for r in ls if r['repeat']==i) for i in (0,1)},
        'finalization_seconds':{str(i):sum(r['profile']['finalization_seconds'] for r in ls if r['repeat']==i) for i in (0,1)}}
report['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.py','.c','.gl','.ps1')}
for path in (HERE.parent/'planned_linear.py',HERE.parents[2]/'빌드/npu/next-stage/post.dll'):
    report['source_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
(RESULTS/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
small={
    'native_post':report.get('post',{}).get('passed'),
    'native_linear':report.get('linear',{}).get('passed'),
    'model':report.get('model'),
    'device_scalar_compatible':report.get('device',{}).get('compatible'),
    'device_vector_difference':report.get('device-vector',{}).get('difference'),
    'overlap':report.get('overlap',{}).get('median_seconds'),
    'post_pipeline':[r['median_seconds'] for r in report.get('overlap',{}).get('post_pipeline',[])],
    'gl_host':report.get('host',{}).get('passed'),
    'generation':report.get('generation')}
print(json.dumps(small,indent=2))
