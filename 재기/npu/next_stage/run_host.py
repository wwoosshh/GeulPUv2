import subprocess,json
from native_post import ROOT,HERE
directory=ROOT/'빌드/npu/next-stage'
report={'passed':False,'scope':'Compiled GL x86 host calls native post.dll through its FFI, 256 FP32 bit comparisons'}
try:
    p=subprocess.run([str(directory/'host.exe')],cwd=directory,capture_output=True,text=True,check=False)
    report.update(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,passed=p.returncode==0)
    if p.returncode: raise RuntimeError(report)
except Exception as exc:
    report['error']=repr(exc)
    raise
finally:
    (HERE/'results/host.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))
