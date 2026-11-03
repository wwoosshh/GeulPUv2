import hashlib,json,statistics,subprocess,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]
out=root/'빌드/matrix'
sys.path.insert(0,str(root/'계약'))
import 정수계약 as C
shapes=json.loads((out/'shapes.json').read_text())
names=['gl','c','c-avx2']
samples=[]
checks=[]
for round_index in range(3):
    for name in names[round_index:]+names[:round_index]:
        print(round_index,name,'running',flush=True)
        p=subprocess.run([str(out/f'{name}.exe')],cwd=out,capture_output=True,text=True,check=True,timeout=300)
        (out/f'{name}-{round_index}.txt').write_text(p.stdout,encoding='utf-8')
        lines=p.stdout.strip().splitlines()
        assert len(lines)==15,(p.stdout,p.stderr)
        for line in lines:
            case,rep,ms,errors=line.split()
            assert int(errors)==0
            samples.append(dict(round=round_index,backend=name,case=int(case),rep=int(rep),ms=float(ms)))
        for case,(m,n,k) in enumerate(shapes):
            raw=(out/f'case{case}.out.bin').read_bytes()
            expected=(out/f'case{case}.expected.bin').read_bytes()
            assert raw==expected,(name,case)
            sums=np.frombuffer(raw,np.int32).reshape(2,m,k//32,n)
            data=np.load(out/f'case{case}.npz')
            e,d,bias=data['e'],data['d'],data['bias']
            parts=[np.ldexp(sums[j].astype(np.float32),(8*j+e-13)[:,:,None]).astype(np.float32) for j in range(2)]
            u=C.더(parts[1],parts[0])
            y=np.zeros((m,n),np.float32)
            for b in range(k//32): y=C.fma32(u[:,b],np.broadcast_to(d[:,b],y.shape),y)
            y=C.더(y,bias[None,:])
            assert np.array_equal(y.view(np.uint32),data['reference'].view(np.uint32)),(name,case,'FP32')
            checks.append(dict(round=round_index,backend=name,case=case,integer_bit_equal=True,final_fp32_bit_equal=True,sha256=hashlib.sha256(raw).hexdigest()))
        print(round_index,name,'PASS integers + full contract linear output',flush=True)
summary={}
for case,shape in enumerate(shapes):
    summary[str(case)]={'shape_M_N_K':shape,'backends':{}}
    for name in names:
        rows=[s for s in samples if s['case']==case and s['backend']==name and s['rep']>0]
        ms=[s['ms'] for s in rows]
        summary[str(case)]['backends'][name]=dict(median_ms=statistics.median(ms),min_ms=min(ms),max_ms=max(ms),round_medians=[statistics.median(s['ms'] for s in rows if s['round']==r) for r in range(3)])
result=dict(passed=True,scope='CPU INT8 dot32 block matrices, two activation digits; FP32 finalization verified outside timer',samples=samples,checks=checks,summary=summary,
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('matrix.*')},binary_sha256={name:hashlib.sha256((out/f'{name}.exe').read_bytes()).hexdigest() for name in names})
(Path(__file__).parent/'matrix-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
