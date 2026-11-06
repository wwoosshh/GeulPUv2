import json,sys,copy,subprocess,tempfile,os,py_compile,types
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
for f in ('additional_pilot.py','additional_runner.py','compare_arms.py'):py_compile.compile(f,doraise=True)
import additional_pilot as ap, additional_runner as ar, compare_arms as ca
ON=sys.argv[1]
# 1. arg parsing requires --speculative
r=subprocess.run([sys.executable,'additional_pilot.py','--url','x','--profile','q8','--gpu-label','g','--data','d','--models-dir','m','--requests','r','--out','o'],capture_output=True,text=True);assert r.returncode==2 and 'speculative' in r.stderr
r=subprocess.run([sys.executable,'additional_runner.py','--root','x','--code','c','--profile','q8','--gpu-label','g'],capture_output=True,text=True);assert r.returncode==2 and 'speculative' in r.stderr
src=open('additional_pilot.py').read();rs=open('additional_runner.py').read()
assert "settings['speculative']=(a.speculative=='on')" in src and "sv.get('speculative') is (a.speculative=='on')" in src and "'--speculative',a.speculative" in rs
# 2. fail-closed load(): drive run() with a fake sp: OFF requested but server reports speculative True must fail with explicit error
import io
def outcome(spec_arm,effective,tok_len=8192,hdr_ok=True):
 saved={}
 res={}
 orig=ap.sp.save
 ap.sp.save=lambda r,o:res.update(copy.deepcopy(r))
 orig_hdr=ap.server_header
 ap.server_header=lambda:({'slots':2,'context':8192,'line':'fake'} if hdr_ok else {'slots':4,'context':8192,'line':'fake'})
 try:
  kit=None
  ap.sp.state=lambda url:{'app':{'version':'0.21.0','commit':ap.COMMIT},'job':{'state':'idle'},'device':{'name':'x'},'settings':res.get('_cur',{})}
  ap.sp.find_model=lambda *a,**k:'mid';ap.sp.model_hash_ok=lambda *a,**k:True
  cur={}
  def lm(url,mid,s):
   cur.update(s);res['_cur']=dict(s);return {'precision':'fast','layer':'cuda','template':'qwen3','speculative':effective,'kv8':False,'fast_attention':False}
  ap.sp.load_model=lm
  def fake_api(url,path,body=None,**k):
   if path=='/tokenize':return {'tokens':[1],'count':1,'max_model_len':tok_len}
   raise ap.sp.PilotError('STOP_AFTER_LOAD')
  ap.sp.api=fake_api
  ap.sp.chat=lambda *a,**k:(_ for _ in ()).throw(ap.sp.PilotError('STOP_AFTER_LOAD'))
  a=types.SimpleNamespace(profile='q8',gpu_label='g',url='u',requests='requests.json',data='d',models_dir='m',out='o',smoke=True,speculative=spec_arm,slots8_only=False)
  ap.run(a)
 finally:ap.sp.save=orig;ap.server_header=orig_hdr
 return res['summary'].get('error',''),res.get('settings'),res['load_evidence'][0]['status']
e,s,st=outcome('off',True);assert 'effective speculative path differs' in e and st=='FAILED',(e,st)
e,s,st=outcome('on',False);assert 'effective speculative path differs' in e and st=='FAILED',(e,st)
e,s,st=outcome('off',False);assert 'STOP_AFTER_LOAD' in e or 'tokenize' in e.lower() or 'PilotError' in e,e   # passes load, stops later in fake api
e,s,st=outcome('on',True);assert st=='VERIFIED'
e,s,st=outcome('off',False,8064);assert 'max_model_len' in e and st=='FAILED',(e,st)
e,s,st=outcome('off',False,8192,False);assert 'server log header' in e and st=='FAILED',(e,st)
print('load assertions ok: OFF/ON mismatches fail closed, matches verify; context-shrink and header mismatch fail closed')
# 2b. v6 overlap gate: max_concurrent pure-function boundaries
assert ap.max_concurrent([])==0
assert ap.max_concurrent([(0,10),(1,9),(2,9),(3,9),(4,9),(5,9),(6,9),(7,9)])==8  # 8 overlapping -> gate passes
assert ap.max_concurrent([(0,10),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8)])==3  # chained, not 8-way
assert ap.max_concurrent([(0,1),(2,3),(4,5)])==1
assert ap.max_concurrent([(0,10),(0,10),(5,6)])==3
assert ap.max_concurrent([(0,5),(5,9)])==2  # touching spans count as concurrent
assert ap.max_concurrent([(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)])==7  # 7 is not 8 -> gate would refuse
# 2c. v6 source assertions: M6 module, effective-context proof, runner fit gate
src=open('additional_pilot.py').read();rs=open('additional_runner.py').read()
for needle in ("'--slots8-only'",'max_model_len','server_header','M6-slots8','max_concurrent','slots_used','text_sha256_diffs','state_sampler'):assert needle in src,needle
# 2d. v8 MIG measured-fit gate (slices: nvidia-smi memory unreadable; own CUDA context mem_get_info)
assert (2*36*1024*2*8192 + 2*36*32*8192*1)//1048576==1170==ar.PER_SLOT_MIB
assert ar.SLOT_STEP==6 and 2+ar.SLOT_STEP==8   # slots 2 -> 8 is +6, not +4 (4 only applied to the RTX slots<=4 anchor)
need=6*1170+2000;assert need==9020
d=ar.slots8_decision(9020.0);assert d['decision']=='PASS' and d['required_free_mib']==9020
assert ar.slots8_decision(9019.9)['decision']=='REFUSE'      # marginal -> refuse
assert ar.slots8_decision(None)['decision']=='REFUSE'        # unavailable -> refuse
assert ar.slots8_decision(0)['decision']=='REFUSE'
# H100 MIG probe numbers: F0 47188; plausible F2min 47188-~13000 passes, A100 40054-~13000 passes, 9000 free refuses
assert ar.slots8_decision(47188-13500)['decision']=='PASS' and ar.slots8_decision(40054-13500)['decision']=='PASS' and ar.slots8_decision(9000)['decision']=='REFUSE'
# probe unavailable fails closed (no libcuda here, or init failure): constructing must raise, never fall back
try:ar.CudaProbe();raise SystemExit('probe unexpectedly available in CPU test box')
except (OSError,RuntimeError):pass
rs=open('additional_runner.py').read();src=open('additional_pilot.py').read()
assert "cuCtxSetCurrent(self.ctx)" in rs and rs.index("cuCtxSetCurrent(self.ctx)!=0")<rs.index("cuMemGetInfo_v2") and "lacks cuCtxSetCurrent" in rs
for needle in ('measured_cuda_free','F0_free_mib','F2_samples_min_mib','slots8_gate.json','device_uuid',"phase=='full'",'slots8 gate refuses before any 8-slot load','no CUDA free samples during smoke','CudaProbe()'):assert needle in rs,needle
for needle in ('slots8_gate.json','measured gate file missing','F2_samples_min_mib',"6*profile['per_slot_mib']+2000"):assert needle in src,needle
for banned in ('13853','slice-mib','nominal','refusing marginal fit','m0=snapshot()'):assert banned not in rs,banned   # no RTX anchor / nominal fallback
# ordering: smoke never loads 8 slots; 8-slot load only after gate require
assert "load(dict(settings,slots=(8 if" not in src
i=src.index("require(os.path.exists(gate)");j=src.index("load(dict(settings,slots=8))");assert i<j
assert src.count("load(dict(settings,slots=8))")==1
# runner: gate evaluated between smoke and full, in the loop, before launching full
k=rs.index("phase=='full'");l=rs.index("cmd=[os.path.join(venv,'python3'),os.path.join(a.code,'additional_pilot.py')");assert k<l
import threading
class FakeL:
 def __init__(s):s.cur=None;s.calls=[]
 def cuCtxSetCurrent(s,ctx):s.calls.append('set');s.cur=ctx;return 0 if getattr(s,'ok',True) else 1
 def cuMemGetInfo_v2(s,f,t):
  assert s.cur is not None,'no current ctx';s.calls.append('mem');f._obj.value=1<<30;t._obj.value=2<<30;return 0
import ctypes
p=ar.CudaProbe.__new__(ar.CudaProbe);p.c=ctypes;p.L=FakeL();p.ctx=ctypes.c_void_p(1)
out=[];th=threading.Thread(target=lambda:out.append(p.mem()));th.start();th.join();assert out==[(1024.0,2048.0)] and p.L.calls==['set','mem'],(out,p.L.calls)
p.L.ok=False
try:p.mem();raise SystemExit('cuCtxSetCurrent failure must fail closed')
except RuntimeError:pass
print('thread-probe ok: mem() sets current ctx before cuMemGetInfo, fails closed')
print('v8 gates ok: 6-slot arithmetic, boundary/None refuse, probe fail-closed, no RTX/nominal fallback, 2-slot smoke before gate before 8-slot load')
# 3. compare_arms
tmp=tempfile.mkdtemp();on=json.load(open(ON));on['arm']={'speculative':'on'}
p1=tmp+'/on.json';json.dump(on,open(p1,'w'))
import io,contextlib
with contextlib.redirect_stdout(io.StringIO()):ca.main(p1,p1,tmp+'/self.json')
d=json.load(open(tmp+'/self.json'))['summary'];assert d['different']==0 and d['identical']==d['rows']==97-0 or d['identical']==d['rows'],d
mut=copy.deepcopy(on);h=mut['outputs']['B0']['gen-4k']['logit_hashes'];h[100]=h[100]+'x'
mut['outputs']['M5-slots4']['math-1']['tokens_sha256']='z'
json.dump(mut,open(tmp+'/mut.json','w'))
with contextlib.redirect_stdout(io.StringIO()):ca.main(tmp+'/mut.json',p1,tmp+'/m.json')
m=json.load(open(tmp+'/m.json'));rs_={ (r['phase'],r['id']):r for r in m['rows']}
assert rs_[('B0','gen-4k')]['first_diff_step']==100 and rs_[('M5-slots4','math-1')]['same'] is False and m['summary']['different']==2
print('compare_arms ok: identical self-compare',d['rows'],'rows; 2 injected diffs detected at exact step')
print('ALL CPU TESTS PASS')

# v9 Llama profile-grounded tests
P=ap.PROFILES['llama']
assert (2*32*1024*2*8192+2*32*32*8192*1)//1048576==1040==P['per_slot_mib'] and ap.PROFILES['q8']['per_slot_mib']==1170==ar.PER_SLOT_MIB
need=6*P['per_slot_mib']+2000;assert need==8240
assert ar.slots8_decision(8240.0,per_slot=1040)['decision']=='PASS' and ar.slots8_decision(8239.9,per_slot=1040)['required_free_mib']==8240 and ar.slots8_decision(8239.9,per_slot=1040)['decision']=='REFUSE'
assert ar.slots8_decision(8240.0,per_slot=1170)['decision']=='REFUSE'   # q8 constant would not pass here: profile must select
assert (P['bos_id'],P['eos_id'],P['architecture'])==(128000,128009,'llama') and P['template_sha256']=='e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65'
assert ap.PROFILES['q8']['bos_id'] is None and ap.PROFILES['q8']['eos_id']==151645 and ap.PROFILES['hybrid']['eos_id']==248046 and ap.PROFILES['hybrid']['per_slot_mib'] is None
rs=open('additional_runner.py').read();src=open('additional_pilot.py').read()
for needle in ("profile['eos_id']","profile['per_slot_mib']","unexpected native BOS id","no profile-grounded per-slot KV size"):assert needle in rs,needle
for needle in ("native BOS pin failed","BOS_PIN[0]=profile['bos_id']","profile['per_slot_mib']"):assert needle in src,needle
assert "6*1170" not in src and "151645" not in rs and "248046" not in rs
# BOS pin behaviour with fake sp
class FS:
 PilotError=Exception
 @staticmethod
 def tokenize_request(rq):return rq
 @staticmethod
 def sha256_ints(x):return 'h'+str(list(x))
 def api(b,p,body,timeout=0):return {'tokens':FS.ids,'count':len(FS.ids)}
 def chat(b,rq,s):return {'n_prompt':len(FS.ids),'prompt_tokens_sha256':'h'+str(list(FS.ids))}
old=ap.sp;ap.sp=FS;ap.BOS_PIN[0]=128000
try:
 FS.ids=[128000,5,6];ap.parity('b',{'id':'x'})
 for bad in ([5,6],[128000,128000,6]):
  FS.ids=bad
  try:ap.parity('b',{'id':'x'});raise SystemExit('BOS pin did not fail '+str(bad))
  except Exception as e:assert 'BOS pin' in str(e),e
finally:ap.sp=old;ap.BOS_PIN[0]=None
rs=open('additional_runner.py').read()
assert "(('smoke',) if a.smoke_only else ('smoke','full'))" in rs and "'SMOKE_COMPLETE' if a.smoke_only else 'COMPLETE'" in rs and "incompatible with --slots8" in rs
assert rs.index("a.smoke_only and a.slots8")<rs.index("for phase in")
r=subprocess.run([sys.executable,'additional_runner.py','--help'],capture_output=True,text=True);assert '--smoke-only' in r.stdout
# the smoke-only run still passes through the same cleanup/guard path (single 'finally'), and never creates the full child
assert rs.count("cmd=[os.path.join(venv,'python3')")==1
assert "return 0 if report['status'] in ('COMPLETE','SMOKE_COMPLETE') else 2" in rs and "report['status']=='COMPLETE' else 2" not in rs
assert "if a.slots8 or a.profile=='llama':" in rs and "if a.slots8 and not profile['per_slot_mib']" in rs
assert rs.index("if a.slots8 or a.profile=='llama':")<rs.index("cuda=CudaProbe()")<rs.index("model=os.path.join(root,'models',profile['file'])")  # probe + F0 before model/engine start
assert "own-CUDA preflight: F0 free" in rs and "own CUDA guard produced no samples; refusing unguarded run" in rs
assert rs.index("refusing unguarded run")<rs.index("for phase in")   # before any pilot child
assert rs.count("cuda=CudaProbe()")==1 and "if a.slots8 and phase=='full'" in rs   # slots8 gate untouched
assert "if free<2000:" in rs   # 2000 MiB kill guard unchanged
# CudaProbe failure is fatal (no fallback to nvidia-smi) for llama ordinary runs
try:ar.CudaProbe();raise SystemExit('probe unexpectedly available')
except (OSError,RuntimeError):pass
src=open('additional_pilot.py').read()
assert "time.sleep(profile['m3_stagger'])" in src and "n>=14 and len(others)==15" in src   # threshold unchanged
assert ap.PROFILES['q8']['m3_stagger']==1.0 and ap.PROFILES['hybrid']['m3_stagger']==1.0 and ap.PROFILES['llama']['m3_stagger']==0.5
assert 1.5+14*0.5<10.9   # last dispatch 8.5 s after resident start < measured llama resident 10.9 s
print('v9 llama cpu tests OK')
