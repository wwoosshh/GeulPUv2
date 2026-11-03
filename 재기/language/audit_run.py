import hashlib,json,statistics,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=root/'빌드/language'
names=['geul','geul-opt','native-current','native-opt']
samples=[]
for r in range(4):
    for name in names[r:]+names[:r]:
        try:
            result=subprocess.run([str(out/(name+'.exe'))],capture_output=True,text=True,check=True,timeout=60)
        except (OSError,subprocess.SubprocessError) as exc:
            (Path(__file__).parent/'audit-blocked.json').write_text(json.dumps(dict(complete=False,backend=name,error=str(exc),partial_samples=samples),indent=2)+'\n',encoding='utf-8')
            raise
        (out/f'audit-{r}-{name}.txt').write_text(result.stdout,encoding='utf-8')
        lines=result.stdout.strip().splitlines()
        assert len(lines)==32
        for line in lines:
            mode,rep,ms,total,errors=line.split()
            assert int(total)==34359214080 and int(errors)==0
            samples.append(dict(round=r,backend=name,mode=int(mode),rep=int(rep),ms=float(ms)))
        print(r,name,'PASS',flush=True)
hashes={name:hashlib.sha256((out/(name+'.exe')).read_bytes()).hexdigest() for name in names+['geul-self','geul-opt-self']}
assert hashes['geul']==hashes['geul-self']
assert hashes['geul-opt']==hashes['geul-opt-self']
summary={}
for mode,task in enumerate(['copy','affine_mask','tile_transpose','sum']):
    summary[task]={}
    for name in names:
        rows=[s for s in samples if s['backend']==name and s['mode']==mode and s['rep']>0]
        values=[s['ms'] for s in rows]
        summary[task][name]=dict(median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values),round_medians=[statistics.median(s['ms'] for s in rows if s['round']==r) for r in range(4)])
report=dict(passed=True,selfhost_byte_equal=True,samples_per_task_backend=28,executable_sha256=hashes,summary=summary,samples=samples)
(Path(__file__).parent/'audit-results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
