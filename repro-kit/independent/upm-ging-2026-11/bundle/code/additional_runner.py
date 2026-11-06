#!/usr/bin/env python3
"""One selected model, one bounded owned app, fail closed, preserve all old data."""
import argparse,datetime,hashlib,json,os,signal,socket,struct,subprocess,sys,threading,time,urllib.request
from additional_pilot import PROFILES

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def gguf_meta(path):
 def val(f,t):
  if t==8:return f.read(struct.unpack('<Q',f.read(8))[0]).decode('utf8')
  if t==9:
   typ,n=struct.unpack('<IQ',f.read(12))
   for _ in range(n):val(f,typ)
   return None
  fm={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t]
  return struct.unpack('<'+fm,f.read(struct.calcsize(fm)))[0]
 out={}
 with open(path,'rb') as f:
  magic,ver,nt,nk=struct.unpack('<4sIQQ',f.read(24))
  if magic!=b'GGUF' or ver!=3:raise RuntimeError('invalid GGUF header')
  for _ in range(nk):
   key=val(f,8);v=val(f,struct.unpack('<I',f.read(4))[0])
   if key in ('general.architecture','tokenizer.chat_template','tokenizer.ggml.add_bos_token','tokenizer.ggml.bos_token_id','tokenizer.ggml.eos_token_id'):out[key]=v
 return out

class CudaProbe:
 """Own CUDA context via the driver API: measured free/total MiB on the current device (MIG slice aware; reflects foreign use). Fail closed if unavailable."""
 def __init__(self):
  import ctypes;self.c=ctypes;self.L=ctypes.CDLL('libcuda.so.1')
  if self.L.cuInit(0)!=0:raise RuntimeError('cuInit failed')
  d=ctypes.c_int()
  if self.L.cuDeviceGet(ctypes.byref(d),0)!=0:raise RuntimeError('cuDeviceGet failed')
  nm=ctypes.create_string_buffer(128);self.L.cuDeviceGetName(nm,128,d);self.name=nm.value.decode()
  u=ctypes.create_string_buffer(16)
  if self.L.cuDeviceGetUuid(u,d)!=0:raise RuntimeError('cuDeviceGetUuid failed')
  self.uuid=u.raw.hex();self.ctx=ctypes.c_void_p()
  if not hasattr(self.L,'cuCtxSetCurrent'):raise RuntimeError('libcuda lacks cuCtxSetCurrent')
  if self.L.cuCtxCreate_v2(ctypes.byref(self.ctx),0,d)!=0:raise RuntimeError('cuCtxCreate failed')
 def mem(self):
  # CUDA contexts are per-thread: the watchdog thread must make the probe context current first (fail closed if the symbol or call is unavailable)
  if self.L.cuCtxSetCurrent(self.ctx)!=0:raise RuntimeError('cuCtxSetCurrent failed')
  f=self.c.c_size_t();t=self.c.c_size_t()
  if self.L.cuMemGetInfo_v2(self.c.byref(f),self.c.byref(t))!=0:raise RuntimeError('cuMemGetInfo failed')
  return f.value/1048576,t.value/1048576
 def close(self):
  try:self.L.cuCtxDestroy_v2(self.ctx)
  except Exception:pass

PER_SLOT_MIB=1170;SLOT_STEP=6;GUARD_MIB=2000
def slots8_decision(f2min,per_slot=PER_SLOT_MIB,step=SLOT_STEP,guard=GUARD_MIB):
 """slots 2 -> 8 adds 6 slots of preallocated KV. Required free (minimum seen with slots=2 resident) = 6*1170+2000 = 9020 MiB."""
 need=step*per_slot+guard
 return {'required_free_mib':need,'f2min_mib':f2min,'decision':'PASS' if f2min is not None and f2min>=need else 'REFUSE'}

class RunnerCancelled(BaseException):pass

def terminate_owned(proc):
 """Stop an owned new-session process group, including TERM-resistant descendants."""
 if proc is None:return
 try:os.killpg(proc.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 try:proc.wait(timeout=30)
 except subprocess.TimeoutExpired:pass
 # A group may outlive its leader. Escalate the owned group even if leader exited.
 try:os.killpg(proc.pid,signal.SIGKILL)
 except ProcessLookupError:pass
 proc.wait(timeout=10)

def main(a):
 root=a.root;profile=PROFILES[a.profile];started=datetime.datetime.now(datetime.timezone.utc);deadline=time.monotonic()+21600
 if a.pin:
  cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[:max(1,len(cpus)//2)])
 tag=a.profile+'-spec'+a.speculative+('-slots8' if a.slots8 else '')+'-'+started.strftime('%Y%m%dT%H%M%S')
 run=os.path.join(root,'additional-'+tag);os.mkdir(run)
 data=os.path.join(run,'data');os.mkdir(data);home=os.path.join(run,'home');os.mkdir(home)
 venv=os.path.join(root,'venv','bin');env={'HOME':home,'GEULPU_HOME':data,'PATH':'/usr/local/bin:/usr/bin:/bin','LANG':'C.UTF-8'}
 report={'gpu_label':a.gpu_label,'speculative_arm':a.speculative,'profile':a.profile,'start_utc':started.isoformat(),'hard_stop_utc':(started+datetime.timedelta(hours=6)).isoformat(),'run':run,'status':'INCOMPLETE','cpu_affinity':sorted(os.sched_getaffinity(0)),'memguard':{'available':False,'tripped':False}}
 def write():
  with open(os.path.join(run,'runner.json'),'w') as f:json.dump(report,f,indent=1)
 def snapshot():
  r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=30)
  fields=[x.strip() for x in r.stdout.strip().split(',')]
  return [int(x) for x in fields] if len(fields)==2 and all(x.isdigit() for x in fields) else None
 def processes():return subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=30).stdout
 app=None;child=None;saved={};stop=threading.Event();guard=None;port=None;cuda=None;samples=[];phase_now={'p':'preflight'}
 previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGHUP)}
 def cancelled(sig,frame):
  # Ignore repeats during finally; external TERM/HUP must enter owned cleanup.
  for signum in previous_handlers:signal.signal(signum,signal.SIG_IGN)
  report['cancel_signal']=sig
  raise RunnerCancelled('external signal '+str(sig))
 for sig in previous_handlers:signal.signal(sig,cancelled)
 def api(path,body=None):
  req=urllib.request.Request('http://127.0.0.1:'+str(port)+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=30) as r:return json.load(r)
 try:
  if a.slots8 or a.profile=='llama':
   if a.slots8 and not profile['per_slot_mib']:raise RuntimeError('slots8: no profile-grounded per-slot KV size')
   # Measured fit, fail closed, no declared-capacity or other-GPU fallback: own CUDA context reads free/total on THIS slice.
   cuda=CudaProbe();f0,t0=cuda.mem()
   report['measured_cuda_free']={'device_name':cuda.name,'device_uuid':cuda.uuid,'total_mib':t0,'F0_free_mib':f0,'per_slot_mib':profile['per_slot_mib'],'slot_increment':SLOT_STEP,'guard_mib':GUARD_MIB,'note':'free includes this probe context and any foreign use; unmeasured-by-nvidia-smi on MIG','F2_samples_min_mib':None,'decision':None}
   if f0<18000:raise RuntimeError('own-CUDA preflight: F0 free %.0f MiB below 18000 MiB'%f0)
  model=os.path.join(root,'models',profile['file'])
  if os.path.getsize(model)!=profile['size'] or sha(model)!=profile['sha256']:raise RuntimeError('model bytes/hash mismatch')
  meta=gguf_meta(model);template=meta.pop('tokenizer.chat_template');meta['template_sha256']=hashlib.sha256(template.encode()).hexdigest()
  if meta['general.architecture']!=profile['architecture'] or meta['template_sha256']!=profile['template_sha256']:raise RuntimeError('native metadata/template mismatch')
  expected_bos=False
  if meta.get('tokenizer.ggml.bos_token_id')!=profile['bos_id'] and profile['bos_id'] is not None:raise RuntimeError('unexpected native BOS id')
  if meta.get('tokenizer.ggml.add_bos_token',False)!=expected_bos:raise RuntimeError('unexpected BOS insertion flag')
  if meta.get('tokenizer.ggml.eos_token_id')!=profile['eos_id']:raise RuntimeError('unexpected native EOS')
  report['metadata']=meta
  report['code_sha256']={name:sha(os.path.join(a.code,name)) for name in ('additional_pilot.py','additional_runner.py','core_pilot.py','stress_pilot.py','m4_followup.py','requests.json')}
  wheel=os.path.join(root,'wheelcheck','geulpu-0.21.0-py3-none-manylinux_2_34_x86_64.whl')
  if sha(wheel)!='e71335f5328c159d420a90563c5d2a9b566162694044d14f6e9e95f1cd5f64a7':raise RuntimeError('wheel hash mismatch')
  subprocess.run([os.path.join(venv,'pip'),'install','--force-reinstall','--no-deps',wheel],env=env,capture_output=True,check=True,timeout=600)
  report['preflight_processes']=processes();mem=snapshot();report['preflight_memory']=mem
  if mem and mem[1]-mem[0]<(18000 if a.profile in ('q8','llama') else 12000):raise RuntimeError('insufficient safe preflight GPU headroom')
  for cand in (44744,44844):
   with socket.socket() as s:
    if s.connect_ex(('127.0.0.1',cand))!=0:port=cand;break
  if port is None:raise RuntimeError('no unused test port')
  report['port']=port;write()
  app=subprocess.Popen([os.path.join(venv,'geulpu'),'--data',data,'--port',str(port),'--no-open'],env=env,start_new_session=True,stdout=open(os.path.join(run,'app.log'),'w'),stderr=subprocess.STDOUT)
  report['app_pid']=app.pid
  def watchdog():
   while not stop.wait(1 if cuda else 5):
    try:
     if cuda:
      fr,tt=cuda.mem();samples.append((time.time(),phase_now['p'],fr))
      m=[int(tt-fr),int(tt)]
     else:m=snapshot()
     if m:
      report['memguard']['available']=True;free=m[1]-m[0];report['memguard']['minfree_mib']=min(free,report['memguard'].get('minfree_mib',free))
      if free<2000:
       report['memguard']['tripped']=True;terminate_owned(child);terminate_owned(app);return
     if time.monotonic()>deadline:
      report['watchdog_tripped']=True;terminate_owned(child);terminate_owned(app);return
    except Exception as e:report['memguard']['last_error']=repr(e)
  guard=threading.Thread(target=watchdog,daemon=True);guard.start()
  ready=False
  for _ in range(80):
   if app.poll() is not None:raise RuntimeError('app exited during device check')
   try:
    state=api('/api/state')
    if (state.get('device') or {}).get('checked'):saved=dict(state.get('settings') or {});ready=True;break
   except Exception:pass
   time.sleep(3)
  if not ready:raise RuntimeError('device check timeout')
  if cuda is not None and not report['memguard']['available']:raise RuntimeError('own CUDA guard produced no samples; refusing unguarded run')
  if a.smoke_only and a.slots8:raise RuntimeError('--smoke-only is incompatible with --slots8')
  for phase in (('smoke',) if a.smoke_only else ('smoke','full')):
   phase_now['p']=phase
   if a.slots8 and phase=='full':
    sm=[x[2] for x in samples if x[1]=='smoke']
    if not sm:raise RuntimeError('slots8: no CUDA free samples during smoke')
    dec=slots8_decision(min(sm),per_slot=profile['per_slot_mib']);report['measured_cuda_free'].update({'F2_samples_min_mib':min(sm),'footprint2_mib':report['measured_cuda_free']['F0_free_mib']-min(sm),'decision':dec['decision'],'required_free_mib':dec['required_free_mib'],'smoke_sample_count':len(sm)});write()
    json.dump(dict(report['measured_cuda_free']),open(os.path.join(run,'slots8_gate.json'),'w'))
    if dec['decision']!='PASS':raise RuntimeError('slots8 gate refuses before any 8-slot load: min free %.0f MiB < %d MiB'%(min(sm),dec['required_free_mib']))
   cmd=[os.path.join(venv,'python3'),os.path.join(a.code,'additional_pilot.py'),'--url','http://127.0.0.1:'+str(port),'--profile',a.profile,'--gpu-label',a.gpu_label,'--data',data,'--models-dir',os.path.join(root,'models'),'--requests',os.path.join(a.code,'requests.json'),'--speculative',a.speculative,'--out',os.path.join(run,phase+'.json')]
   if phase=='smoke':cmd+=['--smoke']
   if a.slots8:cmd+=['--slots8-only']
   timeout=min(2400 if phase=='smoke' else 21600,max(1,deadline-time.monotonic()))
   child=subprocess.Popen(cmd,env=env,start_new_session=True,stdout=open(os.path.join(run,phase+'.log'),'w'),stderr=subprocess.STDOUT)
   try:rc=child.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    terminate_owned(child);raise RuntimeError(phase+' deadline')
   report[phase+'_exit']=rc;write()
   if rc!=0:raise RuntimeError(phase+' failed, no retry')
  if report['memguard']['tripped']:raise RuntimeError('memory guard tripped')
  report['status']='SMOKE_COMPLETE' if a.smoke_only else 'COMPLETE'
 except BaseException as e:report['error']=repr(e)
 finally:
  for sig in previous_handlers:signal.signal(sig,signal.SIG_IGN)
  stop.set()
  if guard:guard.join(timeout=100)
  try:terminate_owned(child);report['child_dead']=child is None or child.poll() is not None
  except Exception as e:report['child_cleanup_error']=repr(e);report['status']='INCOMPLETE'
  if app:
   try:
    api('/api/server/stop',{})
    if saved:api('/api/settings',{k:saved[k] for k in ('precision','kv8_cache','fast_attention','slots','context','speculative') if k in saved})
   except Exception as e:report['restore_error']=repr(e)
   try:terminate_owned(app);report['app_dead']=app.poll() is not None
   except Exception as e:report['app_cleanup_error']=repr(e);report['status']='INCOMPLETE'
  if port:
   with socket.socket() as s:report['port_closed']=s.connect_ex(('127.0.0.1',port))!=0
   if not report['port_closed']:report['status']='INCOMPLETE';report['cleanup_error']='owned service port still open'
  try:report['postflight_processes']=processes()
  except Exception as e:report['postflight_error']=repr(e)
  try:
   if cuda:report.setdefault('measured_cuda_free',{})['F_end_free_mib']=cuda.mem()[0];cuda.close()
  except Exception as e:report['probe_close_error']=repr(e)
  report['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write();print(json.dumps(report),flush=True)
  for sig,handler in previous_handlers.items():signal.signal(sig,handler)
 return 0 if report['status'] in ('COMPLETE','SMOKE_COMPLETE') else 2
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--code',required=True);ap.add_argument('--profile',choices=PROFILES,required=True);ap.add_argument('--gpu-label',required=True);ap.add_argument('--pin',action='store_true');ap.add_argument('--speculative',choices=('on','off'),required=True);ap.add_argument('--slots8',action='store_true');ap.add_argument('--smoke-only',action='store_true');raise SystemExit(main(ap.parse_args()))
