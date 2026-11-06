#!/usr/bin/env python3
"""Model-native fast-mode suites. One attempt, no adaptive generation retries.
Own cold references per model/GPU; no Q4-vs-Q8 or default-mode contract claim.
"""
import argparse,copy,datetime,hashlib,json,os,re,time,traceback,concurrent.futures,threading
import core_pilot as cp
import stress_pilot as sp
import m4_followup as mf

PROFILES={
 'hybrid':{'key':'qwen3.5-9b-q4_k_m','file':'Qwen3.5-9B-Q4_K_M.gguf','name':'Qwen3.5-9B-Q4_K_M','size':5680522464,'sha256':'03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8','architecture':'qwen35','ui_template':'qwen3','per_slot_mib':None,'m3_stagger':1.0,'bos_id':None,'eos_id':248046,'template_sha256':'7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67'},
 'llama':{'key':'llama-3.1-8b-instruct-q8_0','file':'Meta-Llama-3.1-8B-Instruct-Q8_0.gguf','name':'Meta-Llama-3.1-8B-Instruct-Q8_0','size':8540775840,'sha256':'9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283','architecture':'llama','ui_template':'llama3','per_slot_mib':1040,'m3_stagger':0.5,'bos_id':128000,'eos_id':128009,'template_sha256':'e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65'},
 'q8':{'key':'qwen3-8b-q8_0','file':'Qwen3-8B-Q8_0.gguf','name':'Qwen3-8B-Q8_0','size':8709518112,'sha256':'408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6','architecture':'qwen3','ui_template':'qwen3','per_slot_mib':1170,'m3_stagger':1.0,'bos_id':None,'eos_id':151645,'template_sha256':'57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361'}}
COMMIT='0efe2a5824ea202f5c02f98afb0620647966e884'

BOS_PIN=[None]
def parity(base,rq):
 t=sp.api(base,'/tokenize',sp.tokenize_request(rq),timeout=120)
 ids=t.get('tokens'); r=sp.chat(base,dict(rq,max_tokens=1),False)
 good=isinstance(ids,list) and t.get('count')==len(ids)==r['n_prompt'] and sp.sha256_ints(ids)==r['prompt_tokens_sha256']
 ev={'count':t.get('count'),'chat_count':r['n_prompt'],'tokenize_sha256':sp.sha256_ints(ids) if isinstance(ids,list) else None,'chat_sha256':r['prompt_tokens_sha256'],'first_token_ids':ids[:8] if isinstance(ids,list) else None,'last_token_ids':ids[-12:] if isinstance(ids,list) else None,'status':'EXACT' if good else 'FAILED'}
 if good and BOS_PIN[0] is not None and not (ids[0]==BOS_PIN[0] and (len(ids)<2 or ids[1]!=BOS_PIN[0])):raise sp.PilotError('native BOS pin failed: first ids '+str(ids[:3])+' expected exactly one '+str(BOS_PIN[0]))
 if not good:raise sp.PilotError('strict per-request prompt ID parity failed '+rq['id']+' '+str(ev))
 return ev


def max_concurrent(spans):
 """Maximum simultaneously in-flight requests from (t0,t1) wall spans.
 A request is in flight over [t0,t1]; at equal timestamps starts count before ends."""
 pts=[]
 for t0,t1 in spans:pts.append((t0,1));pts.append((t1,-1))
 pts.sort(key=lambda p:(p[0],-p[1]))
 c=m=0
 for _,d in pts:c+=d;m=max(m,c)
 return m

def server_header():
 """Newest server log's startup header: effective slots and context per slot
 (the app writes '슬롯 N · 슬롯마다 문맥 L' with the engine's effective max length
 AFTER any fit-driven reduction, so this detects silent context shrink)."""
 f=sp._server_log_file()
 if not f:return None
 try:
  for line in open(f,encoding='utf8',errors='replace'):
   m=re.search(r'슬롯 (\d+) · 슬롯마다 문맥 (\d+)',line)
   if m:return {'slots':int(m.group(1)),'context':int(m.group(2)),'line':line.strip()}
 except OSError:pass
 return None

class state_sampler:
 """Non-gating background /api/state samples during a phase (1 s cadence)."""
 def __init__(self,base):
  self.base=base;self.samples=[];self._stop=threading.Event()
  self._t=threading.Thread(target=self._loop,daemon=True);self._t.start()
 def _loop(self):
  while not self._stop.is_set():
   try:
    st=sp.state(self.base);sv=st.get('server') or {}
    self.samples.append({'t':round(time.time(),2),'server':{k:sv.get(k) for k in ('state','slots','seconds','busy','active','queued','running') if k in sv},'job':(st.get('job') or {}).get('state')})
   except Exception as e:self.samples.append({'t':round(time.time(),2),'error':repr(e)})
   self._stop.wait(1.0)
 def stop(self):
  self._stop.set();self._t.join(timeout=5)

def run(a):
 result={'suite':'model-native-v1','profile':a.profile,'gpu_label':a.gpu_label,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modules':{},'outputs':{},'instrumentation':{},'runs':[],'evidence':{},'summary':{'status':'INCOMPLETE'}}
 sp.DATA_DIR=a.data; refs={};requests={}; profile=PROFILES[a.profile];BOS_PIN[0]=profile['bos_id']
 def checkpoint():sp.save(result,a.out)
 def require(good,why):
  if not good:raise sp.PilotError(why)
 def gate_comparisons():
  if any(not r['same'] for r in result['runs']):raise sp.PilotError('candidate divergence; stop after module, no subsequent modules')
 def record(label,items,parallel=1,cache=False,stagger=0):
  rec=cp.Recorder(a.url);rec.run(items,parallel=parallel,cache_prompt=cache,stagger=stagger)
  rec.record(result,label,refs);checkpoint();gate_comparisons();return rec
 def load(settings):
  # /api/state.server.template is a UI family, not GGUF general.architecture.
  # Pinned app/상태.gl509-520 maps thinking-toggle templates to qwen3.
  ev={'load_request':{'settings':copy.deepcopy(settings),'start':{'model':mid}},
      'expected':{'architecture':profile['architecture'],'ui_template':profile['ui_template'],'precision':'fast','layer':'cuda'},'status':'PENDING'}
  result.setdefault('load_evidence',[]).append(ev);checkpoint()
  try:
   ev['before_state']=sp.state(a.url);checkpoint()
   sv=sp.load_model(a.url,mid,settings);ev['returned_server']=copy.deepcopy(sv);checkpoint()
   live=sp.state(a.url);ev['after_state']=copy.deepcopy(live);checkpoint()
   require(all((live.get('settings') or {}).get(k)==v for k,v in settings.items()),'effective requested settings mismatch')
   require(sv.get('precision')=='fast' and sv.get('layer')=='cuda','effective precision/backend not fast CUDA')
   require(sv.get('template')==profile['ui_template'],'effective UI template family mismatch')
   if a.profile=='hybrid':require(not sv.get('speculative'),'hybrid speculative path unexpectedly enabled')
   # v5: explicit fail-closed effective-path assertions for every load.
   require(settings.get('speculative') is (a.speculative=='on'),'requested speculative setting differs from arm')
   require(sv.get('speculative') is (a.speculative=='on'),'effective speculative path differs from arm: '+repr(sv.get('speculative')))
   require((live.get('settings') or {}).get('speculative') is (a.speculative=='on'),'live speculative setting differs from arm')
   require(sv.get('kv8') is False and sv.get('fast_attention') is False,'KV8/fast attention unexpectedly engaged')
   # v6: the pinned engine silently REDUCES context length to fit device memory
   # (engine creation shrink loop at the pinned commit), so effective context is
   # proven, not assumed: /tokenize max_model_len (the server returns the engine's
   # effective max length) plus the new server log's startup header.
   tok=sp.api(a.url,'/tokenize',{'messages':[{'role':'user','content':'hi'}],'chat_template_kwargs':{'enable_thinking':False}},timeout=60)
   ev['max_model_len']=tok.get('max_model_len')
   require(tok.get('max_model_len')==settings.get('context'),'effective max_model_len differs from requested context (engine shrank context to fit): '+repr(tok.get('max_model_len')))
   hdr=server_header();ev['server_log_header']=hdr
   require(hdr and hdr['slots']==settings.get('slots') and hdr['context']==settings.get('context'),'server log header slots/context mismatch: '+repr(hdr))
   result.setdefault('paths',[]).append({'settings':settings,'effective':{k:sv.get(k) for k in sp.PATH_FIELDS}})
   ev['status']='VERIFIED';checkpoint();return sv
  except BaseException as e:
   ev['status']='FAILED';ev['error']=repr(e)
   try:ev['failure_state']=sp.state(a.url)
   except Exception as state_error:ev['failure_state_error']=repr(state_error)
   checkpoint();raise

 try:
  kit=json.load(open(a.requests));settings=copy.deepcopy(kit['configs']['default']);settings['speculative']=(a.speculative=='on')
  result['arm']={'speculative':a.speculative}
  st=sp.state(a.url);result['initial_state']=copy.deepcopy(st);checkpoint();require((st.get('app') or {}).get('version')=='0.21.0','version mismatch');require((st.get('app') or {}).get('commit')==COMMIT,'commit mismatch')
  require((st.get('job') or {}).get('state')!='running','app busy')
  result['environment']={'version':(st.get('app') or {}).get('version'),'commit':COMMIT,'device':(st.get('device') or {}).get('name')}
  result['profile_spec']=profile;result['manifest_sha256']=hashlib.sha256(open(a.requests,'rb').read()).hexdigest()
  mid=sp.find_model(a.url,profile,[a.models_dir]);sp.model_hash_ok(a.url,mid,profile);result['model_id']=mid
  load(settings);result['settings']=settings
  reqs=copy.deepcopy(kit['requests']+sp.PILOT_REQUESTS)
  # Explicit non-thinking for constructed prompts; supplied manifest kwargs stay unchanged.
  for rq in reqs:rq.setdefault('chat_template_kwargs',{'enable_thinking':False});sp.semantic_request(rq)
  thinking={'id':'native-thinking','kind':'bounded native thinking sampled conformance, not quality','messages':[{'role':'user','content':'Compute 17 times 19 and explain briefly.'}],'chat_template_kwargs':{'enable_thinking':True},'temperature':0.6,'top_p':0.95,'seed':7,'max_tokens':128}
  reqs.append(thinking)
  require(len({r['id'] for r in reqs})==len(reqs),'duplicate request IDs')
  # Smoke is model-specific: every shape strict ID parity, cold bypass evidence,
  # repeat identity and actual requested 4-slot allocation before long GPU work.
  result['evidence']['shape_parity']={}
  for rq in reqs:
   result['evidence']['shape_parity'][rq['id']]=parity(a.url,rq);checkpoint()
  anchor=next(r for r in reqs if r['id']=='math-2');smoke=[]
  for _ in range(2):
   mark=sp.log_marker(a.url);r=sp.chat(a.url,anchor,False);c=sp.new_counters(a.url,mark)
   require(c and not any(x['cache'] or x['store'] or x['other_slot'] for x in c),'cold bypass unverified')
   smoke.append(r)
  require(sp.compare(smoke[0],smoke[1])['same'],'smoke repeat identity failed')
  mark=sp.log_marker(a.url);warm=[sp.chat(a.url,anchor,None) for _ in range(2)];counts=sp.new_counters(a.url,mark)
  require(all(sp.compare(r,smoke[0])['same'] for r in warm),'smoke default repeat differs')
  if a.profile=='q8':require(any(x['cache'] or x['store'] or x['other_slot'] for x in counts),'q8 caching semantics unverifiable')
  result['evidence']['smoke']={'cold_repeat':'IDENTICAL','default_repeat':'IDENTICAL','default_repeat_counters':counts,'cache_requirement':'measured, hybrid state semantics only' if a.profile=='hybrid' else 'positive hit required'}
  load(dict(settings,slots=4)) if not a.slots8_only else None;load(settings)  # slots8 mode: smoke stays at the reviewed slots=2; 8-slot load only after measured gate
  result['modules']['smoke']={'status':'DONE'};checkpoint()
  if a.smoke:
   result['summary']={'status':'SMOKE-COMPLETE'};checkpoint();return 0
  # Size each new model against native measured tokenizer and preserve exact requests.
  core=[]
  for spec in cp.CORE_SPECS:
   if a.profile=='hybrid' and spec['id'].startswith('evict-'):continue
   if spec['sized']:
    def count(messages):
     t=sp.api(a.url,'/tokenize',{'messages':messages,'chat_template_kwargs':{'enable_thinking':False}},timeout=120)
     require(isinstance(t.get('tokens'),list) and t.get('count')==len(t['tokens']),'sizing token IDs missing')
     return t['count']
    rq,n,c=cp.auto_size(count,spec);result['evidence']['sizing:'+rq['id']]={'entries':n,'tokens':c,'bounds':list(cp.SIZED_BOUNDS[rq['id']])}
   else:rq={k:copy.deepcopy(spec[k]) for k in ('id','kind','messages','max_tokens','temperature','seed')}
   rq['chat_template_kwargs']={'enable_thinking':False};parity(a.url,rq);core.append(rq)
  requests={r['id']:r for r in reqs+core};result['requests']=requests;checkpoint()
  for rq in requests.values():
   mark=sp.log_marker(a.url);r=sp.chat(a.url,rq,False);c=sp.new_counters(a.url,mark)
   require(c and not any(x['cache'] or x['store'] or x['other_slot'] for x in c),'B0 cold evidence missing '+rq['id'])
   refs[rq['id']]=r;result['outputs'].setdefault('B0',{})[rq['id']]=r;result['evidence']['B0:'+rq['id']]={'cold':True,'counters':c};checkpoint()
  require(len(refs)==(20 if a.profile=='hybrid' else 23),'B0 reference count incomplete')
  result['modules']['B0']={'status':'DONE','references':len(refs)}
  for rid in ('ctx-long','ctx-max'):
   lo,hi,mt=cp.BOUNDS[rid];cp.check_bounds(refs[rid],lo,hi,mt,rid)
  result['modules']['M2']={'status':'DONE'};checkpoint()
  if a.slots8_only:
   # M6: the engine's documented ceiling — 8 concurrent chats (user-guide at the
   # pinned commit). Same 23 requests: solo B0 refs (slots=2) vs one 8-way wave
   # (slots=8). Heavy prompts dispatch first so all 8 slots stay occupied for tens
   # of seconds; dispatch order does not change per-request semantics. Identity
   # gate: prompt token IDs + generated token IDs + every per-step logit hash vs
   # own cold refs. Serialized text hashes are recorded informationally (known
   # within-arm tool-text nondeterminism).
   heavy=['gen-4k','ctx-max','ctx-long','evict-a','evict-b','evict-c','gen-1k','prefix-a']
   allreqs=[requests[r] for r in heavy]+[r for r in requests.values() if r['id'] not in heavy]
   require(len(allreqs)==23 and len({r['id'] for r in allreqs})==23,'M6 request set incomplete')
   gate=os.path.join(os.path.dirname(os.path.abspath(a.out)),'slots8_gate.json')
   require(os.path.exists(gate),'slots8 measured gate file missing (fail closed before any 8-slot load)')
   g=json.load(open(gate));require(g.get('decision')=='PASS' and g.get('F2_samples_min_mib') is not None and profile['per_slot_mib'] and g.get('per_slot_mib')==profile['per_slot_mib'] and g['F2_samples_min_mib']>=6*profile['per_slot_mib']+2000 and g.get('device_uuid'),'slots8 measured gate not PASS')
   result['evidence']['slots8_gate']=g
   load(dict(settings,slots=8))
   try:
    rec=cp.Recorder(a.url);sampler=state_sampler(a.url)
    rec.run(allreqs,parallel=8,cache_prompt=False);sampler.stop()
    rec.record(result,'M6-slots8',refs);checkpoint();gate_comparisons()
    mc=max_concurrent([(x['t0'],x['t1']) for x in rec.recs.values()])
    slots_used=sorted({c['slot'] for c in rec.counters})
    textdiff=sorted(rid for rid,x in rec.recs.items() if x['res']['text_sha256']!=refs[rid]['text_sha256'])
    result['evidence']['M6-slots8'].update({'max_concurrent':mc,'slots_used':slots_used,'requests':len(allreqs),'counter_lines':len(rec.counters),'text_sha256_diffs':textdiff,'text_note':'informational only; identity gate is prompt IDs + generated token IDs + per-step logit hashes','state_samples':sampler.samples})
    checkpoint()
    require(len(rec.counters)>=len(allreqs),'M6 server counter lines incomplete: '+str(len(rec.counters)))
    require(mc==8,'8-way wall overlap unproven: max_concurrent='+str(mc))
    require(len(slots_used)==8,'8 distinct server slots unproven: '+repr(slots_used))
    result['modules']['M6-slots8']={'status':'DONE','requests':len(allreqs),'parallel':8,'max_concurrent':mc,'slots_used':slots_used}
   finally:load(settings)
  else:
   # Resident overlap measured, no timer-based claim of device scheduler overlap.
   resident=requests['gen-4k'];others=[r for r in reqs if r['id'] not in ('gen-1k','native-thinking')]
   rec=cp.Recorder(a.url)
   with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
    fr=ex.submit(rec.work,resident,None);time.sleep(1.5)
    futures=[]
    for rq in others:futures.append(ex.submit(rec.work,rq,None));time.sleep(profile['m3_stagger'])
    fr.result()
    for f in futures:f.result()
   rec.counters=sp.new_counters(a.url,rec.mark);rec.record(result,'M3-churn',refs);gate_comparisons()
   rr=rec.recs['gen-4k'];n=sum(x['t0']<rr['t1'] and x['t1']>rr['t0'] for k,x in rec.recs.items() if k!='gen-4k')
   result['evidence']['M3-churn'].update({'overlap_count':n,'overlap_total':len(others),'resident_seconds':rr['t1']-rr['t0']});checkpoint()
   require(n>=14 and len(others)==15,'M3 required14/15 wall-span overlap unmet');result['modules']['M3']={'status':'DONE'}
   if a.profile=='hybrid':
    result['modules']['M4-SSD']={'status':'N/A','reason':'qwen35 context store unsupported in pinned engine'}
    result['modules']['speculation']={'status':'N/A','reason':'qwen35 speculative decoding unsupported'}
   
    for i,rid in enumerate(('ctx-long','math-2','ctx-long')):record('hybrid-state-repeat-'+str(i),[requests[rid]],cache=None)
    result['modules']['hybrid-state-repeat']={'status':'DONE','scope':'repeat correctness after intervening request; no SSD restoration claim'}
   else:
    ev={'records':{},'counters':{}}
    require(len({refs[r]['n_prompt'] for r in ('evict-a','evict-b','evict-c')})==3,'M4 ambiguous log attribution')
    for rid in ('evict-a','evict-b','evict-c'):cp.check_bounds(refs[rid],*cp.EVICT_BOUNDS,64,rid)
    rc=mf.execute(a.url,requests,refs,ev);result['evidence']['M4-barrier']=ev
    result['outputs']['M4-barrier']={k:v['res'] for k,v in ev.get('records',{}).items() if 'res' in v}
    for k,v in ev.get('comparisons',{}).items():result['runs'].append(dict(phase='M4-barrier',id=k,**v))
    checkpoint()
    require(rc==0,'M4 eviction proof or comparisons failed');result['modules']['M4-SSD']={'status':'DONE'}
   load(dict(settings,slots=4))
   try:record('M5-slots4',reqs,parallel=4,cache=False);result['modules']['M5']={'status':'DONE','requests':len(reqs),'parallel':4}
   finally:load(settings)
  if a.slots8_only:require(all(result['modules'].get(m,{}).get('status')=='DONE' for m in ('smoke','B0','M2','M6-slots8')),'required module missing')
  else:
   require(all(result['modules'].get(m,{}).get('status')=='DONE' for m in ('smoke','B0','M2','M3','M5')),'required module missing')
   require(result['modules'].get('M4-SSD',{}).get('status')==('N/A' if a.profile=='hybrid' else 'DONE'),'M4 scoped coverage missing')
  result['summary']={'status':'COMPLETE','comparisons':len(result['runs']),'all_identical':all(r['same'] for r in result['runs']),'N/A':[k for k,v in result['modules'].items() if v['status']=='N/A']};checkpoint();return 0
 except Exception as e:
  result['summary']={'status':'INCOMPLETE','error':repr(e)};result['traceback']=traceback.format_exc();checkpoint();print(result['summary'],flush=True);return 2
 finally:checkpoint()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--url',required=True);ap.add_argument('--profile',choices=PROFILES,required=True);ap.add_argument('--gpu-label',required=True);ap.add_argument('--data',required=True);ap.add_argument('--models-dir',required=True);ap.add_argument('--requests',required=True);ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--speculative',choices=('on','off'),required=True);ap.add_argument('--slots8-only',action='store_true')
 raise SystemExit(run(ap.parse_args()))
