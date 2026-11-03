"""Rotated process order; one warmup + seven measured samples per task/round."""
import hashlib
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'빌드/language'
commands={
    'geul':[str(OUT/'geul.exe')],
    'native_c':[str(OUT/'native.exe')],
    'python':[sys.executable,str(Path(__file__).with_name('bench.py')),'python'],
    'numpy':[sys.executable,str(Path(__file__).with_name('bench.py')),'numpy'],
}
names=list(commands)
samples=[]
for round_index in range(4):
    order=names[round_index:]+names[:round_index]
    for name in order:
        print(f'round={round_index} backend={name}',flush=True)
        start=time.perf_counter()
        p=subprocess.run(commands[name],cwd=ROOT,capture_output=True,text=True,check=True,timeout=180)
        (OUT/f'{name}-{round_index}.txt').write_text(p.stdout,encoding='utf-8')
        lines=p.stdout.strip().splitlines()
        assert len(lines)==32,(name,p.stdout,p.stderr)
        for line in lines:
            mode,rep,elapsed,total,errors=line.split()
            assert int(errors)==0
            assert int(total)==34359214080,(name,mode,total)
            samples.append(dict(backend=name,round=round_index,mode=int(mode),rep=int(rep),milliseconds=float(elapsed)))
        print(f'finished {name} including validation {time.perf_counter()-start:.2f}s',flush=True)
result={
    'python':sys.version,'platform':platform.platform(),
    'elements':1048576,'dtype':'signed int64','bytes_per_array':8388608,
    'rounds':4,'samples_per_task_backend':28,'warmup_per_round':1,
    'geul_commit':subprocess.check_output(['git','-C',str(ROOT/'geul-lang'),'rev-parse','HEAD'],text=True).strip(),
    'c_flags':'MSVC /O2 /W4, default x64 ISA; no /arch:AVX2; no LTO',
    'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('bench.*')},
    'samples':samples,'summary':{},
}
for mode,task in enumerate(['copy','affine_mask','tile_transpose','sum']):
    stats={}
    for name in names:
        values=[s['milliseconds'] for s in samples if s['mode']==mode and s['backend']==name and s['rep']>0]
        stats[name]={'median_ms':statistics.median(values),'min_ms':min(values),'max_ms':max(values),
                     'round_medians_ms':[statistics.median([s['milliseconds'] for s in samples if s['mode']==mode and s['backend']==name and s['rep']>0 and s['round']==r]) for r in range(4)]}
    stats['ratios']={name+'_time_divided_by_geul':stats[name]['median_ms']/stats['geul']['median_ms'] for name in names if name!='geul'}
    result['summary'][task]=stats
path=Path(__file__).parent/'results.json'
path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result['summary'],indent=2),flush=True)
