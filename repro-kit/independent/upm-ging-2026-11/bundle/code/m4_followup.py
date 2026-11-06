#!/usr/bin/env python3
"""One bounded M4 follow-up: unchanged inputs/settings; barrier B+C pressure.
Historical core JSON is read-only and remains INCOMPLETE. No adaptive retry.
"""
import argparse,concurrent.futures,datetime,hashlib,json,os,threading,time,traceback
import core_pilot as cp
import stress_pilot as sp

def digest(path):
 return hashlib.sha256(open(path,'rb').read()).hexdigest()

def classify(line,fp):
 if line is None:return 'EVIDENCE-MISSING (no attributable log line)'
 if line['cache']+line['other_slot']>=fp-64:
  return 'EVIDENCE-MISSING (re-request served from memory; eviction did not happen or KV larger than assumed)'
 if line['store']>0:return 'RESTORED-FROM-STORE (evicted from KV, restored from SSD store)'
 return 'RECOMPUTED-COLD (evicted, no store reuse)'

def execute(base,requests,refs,result):
 """All calls single-attempt; real 3-party barrier; attributable phase logs."""
 records={};counters={};lock=threading.Lock();barrier=threading.Barrier(3,timeout=30)
 result['records']=records;result['counters']=counters
 def call(label,rq,sync=False):
  admission=time.time()
  if sync:barrier.wait()
  t0=time.time()
  with lock:order=len(records);records[label]={'admission':admission,'t0':t0,'dispatch_order':order}
  r=sp.chat(base,rq,None);t1=time.time()
  with lock:records[label].update({'t1':t1,'res':r,'rid':rq['id']})
  return r
 mark=sp.log_marker(base);call('A',requests['evict-a']);counters['A']=sp.new_counters(base,mark)
 mark=sp.log_marker(base)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  b=pool.submit(call,'B',requests['evict-b'],True)
  c=pool.submit(call,'C',requests['evict-c'],True)
  result['barrier_release']=time.time();barrier.wait()
  b.result();c.result()
 counters['pressure']=sp.new_counters(base,mark)
 mark=sp.log_marker(base);call('A_prime',requests['evict-a']);counters['A_prime']=sp.new_counters(base,mark)
 result['records']=records;result['counters']=counters
 def attributable(lines,rq):
  hits=[l for l in lines if l['prompt']==refs[rq['id']]['n_prompt']]
  return hits[0] if len(hits)==1 else None
 al=attributable(counters['A'],requests['evict-a'])
 bl=attributable(counters['pressure'],requests['evict-b'])
 cl=attributable(counters['pressure'],requests['evict-c'])
 apl=attributable(counters['A_prime'],requests['evict-a'])
 overlap=max(0,min(records['B']['t1'],records['C']['t1'])-max(records['B']['t0'],records['C']['t0']))
 diffslots=bl is not None and cl is not None and bl['slot']!=cl['slot']
 replaces=al is not None and diffslots and al['slot'] in (bl['slot'],cl['slot'])
 comparisons={k:sp.compare(v['res'],refs[v['rid']]) for k,v in records.items()}
 label=classify(apl,refs['evict-a']['n_prompt'])
 result.update({'attribution':{'A':al,'B':bl,'C':cl,'A_prime':apl},
                'pressure_overlap_seconds':overlap,'pressure_distinct_slots':diffslots,
                'seed_slot_covered':replaces,'comparisons':comparisons,
                'A_prime_classification':label})
 proof=overlap>0 and diffslots and replaces and not label.startswith('EVIDENCE-MISSING')
 if not all(v['same'] for v in comparisons.values()):
  result['summary']={'status':'CANDIDATE-DIVERGENCES','eviction_proven':proof};return 1
 result['summary']={'status':'COMPLETE' if proof else 'INCOMPLETE','eviction_proven':proof}
 return 0 if proof else 2

def run(a):
 result={'kind':'M4-followup-barrier-v1','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'historical_run_unchanged':True,'provenance':{},'summary':{'status':'INCOMPLETE'}}
 try:
  old=json.load(open(a.source));kit=json.load(open(a.requests))
  refs={r['id']:r for r in json.load(open(a.references))}
  result['provenance']={'source_sha256':digest(a.source),'references_sha256':digest(a.references),
                        'manifest_file_sha256':digest(a.requests),'source_environment':old['environment'],
                        'source_manifest_sha256':old['manifest_sha256'],'source_summary':old.get('summary')}
  # Same full manifest identity contract as cleared core, no schema/spec changes.
  manifest=hashlib.sha256(open(a.requests,'rb').read()+sp.REQUEST_SCHEMA_VERSION.encode()
    +json.dumps(cp.CORE_SPECS,sort_keys=True,ensure_ascii=False).encode()
    +json.dumps(sp.PILOT_REQUESTS,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
  if manifest!=old['manifest_sha256']:raise sp.PilotError('source manifest identity mismatch')
  st=sp.state(a.url)
  for key,live in [('geulpu_version',(st.get('app') or {}).get('version')),
                   ('geulpu_commit',(st.get('app') or {}).get('commit'))]:
   if old['environment'].get(key)!=live:raise sp.PilotError('source '+key+' mismatch')
  if (st.get('job') or {}).get('state')=='running':raise sp.PilotError('app busy')
  settings=kit['configs']['default']
  if old['settings']!=settings or settings.get('slots')!=2:raise sp.PilotError('source settings mismatch')
  spec=next(m for m in kit['models'] if m['key']=='qwen3-8b-q4_k_m')
  mid=sp.find_model(a.url,spec,a.models_dir);sp.model_hash_ok(a.url,mid,spec)
  if mid!=old['model_id']:raise sp.PilotError('source model id mismatch')
  sv=sp.load_model(a.url,mid,settings);live=sp.state(a.url)
  if any((live.get('settings') or {}).get(k)!=v for k,v in settings.items()):
   raise sp.PilotError('effective settings mismatch')
  path={k:sv.get(k) for k in sp.PATH_FIELDS}
  if path!=old['path']:raise sp.PilotError('effective execution path mismatch')
  ident={'version':old['environment']['geulpu_version'],'commit':old['environment']['geulpu_commit'],'model':mid}
  kw=old['evidence']['preflight']['winning']['kwargs'];requests={}
  for rid in ('evict-a','evict-b','evict-c'):
   specd=next(s for s in cp.CORE_SPECS if s['id']==rid)
   n=old['evidence']['preflight'][rid]['entries']
   rq={'id':rid,'kind':specd['kind'],'messages':[{'role':'user','content':cp.long_prompt(specd['tag'],n,specd['question'],specd['num_off'])}],
       'max_tokens':specd['max_tokens'],'temperature':0,'seed':cp.SEED}
   if kw is not None:rq['chat_template_kwargs']=dict(kw)
   if old['evidence']['B0:'+rid].get('cold') is not True:raise sp.PilotError('not cold source '+rid)
   fp=cp.fingerprint(rq,settings,ident)
   if fp!=old['b0_fingerprints'][rid] or fp!=refs[rid].get('fingerprint'):raise sp.PilotError('fingerprint mismatch '+rid)
   tok=sp.api(a.url,'/tokenize',sp.tokenize_request(rq),timeout=120)
   tids=tok.get('tokens')
   if not isinstance(tids,list) or tok.get('count')!=len(tids) or sp.sha256_ints(tids)!=refs[rid]['prompt_tokens_sha256']:
    raise sp.PilotError('strict prompt parity failed '+rid)
   requests[rid]=rq
  if len({refs[r]['n_prompt'] for r in requests})!=3:raise sp.PilotError('ambiguous prompt-length attribution')
  result.update({'settings':settings,'path':path,'requests':requests})
  result['carry_provenance_only']={m:old['modules'][m] for m in ('B0','M3','M5')
       if (old.get('modules',{}).get(m) or {}).get('status')=='DONE' and
       (m!='M3' or old['evidence']['M3-churn'].get('overlap_target_14_of_15')=='MET')}
  sp.DATA_DIR=a.data
  rc=execute(a.url,requests,refs,result)
 except Exception as e:
  result['summary']={'status':'INCOMPLETE','error':repr(e)};result['traceback']=traceback.format_exc();rc=2
 finally:
  with open(a.out,'w') as f:json.dump(result,f,ensure_ascii=False,indent=1)
 print(json.dumps(result['summary']),flush=True);return rc

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--url',required=True);ap.add_argument('--source',required=True)
 ap.add_argument('--references',required=True);ap.add_argument('--requests',required=True)
 ap.add_argument('--models-dir',action='append',default=[]);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
 raise SystemExit(run(ap.parse_args()))
