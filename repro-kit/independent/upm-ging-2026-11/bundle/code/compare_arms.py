#!/usr/bin/env python3
"""Offline cross-arm comparison: spec-OFF full.json vs completed spec-ON full.json.
Empirical only. Never asserts equality; reports every identity/difference per (phase,id).
Within-arm repeat correctness (own cold refs) is read from each arm's own summary."""
import json,sys
def cmp(res,base):
 if res['prompt_tokens_sha256']!=base['prompt_tokens_sha256']:return {'same':False,'why':'prompt tokens differ'}
 a,b=res['logit_hashes'],base['logit_hashes']
 for i in range(min(len(a),len(b))):
  if a[i]!=b[i]:return {'same':False,'why':'logits differ from step %d'%i,'first_diff_step':i}
 if res['tokens_sha256']!=base['tokens_sha256'] or len(a)!=len(b) or res['n']!=base['n']:return {'same':False,'why':'generated tokens differ'}
 return {'same':True}
def main(off_path,on_path,out):
 off=json.load(open(off_path));on=json.load(open(on_path));rows=[]
 for ph in ('B0','M3-churn','M4-barrier','M5-slots4','M6-slots8'):
  for rid in sorted(set(off['outputs'].get(ph,{}))|set(on['outputs'].get(ph,{}))):
   x=off['outputs'].get(ph,{}).get(rid);y=on['outputs'].get(ph,{}).get(rid)
   r={'phase':ph,'id':rid,'n_off':x and x['n'],'n_on':y and y['n']}
   r.update(cmp(x,y) if x and y else {'same':None,'why':'missing in one arm'});rows.append(r)
 # Cross-phase: an arm's M6-slots8 wave vs the other arm's B0 solo refs
 # (both cold, cache_prompt=false, same requests) — informational, id-matched.
 for arm,other,tag in ((off,on,'off'),(on,off,'on')):
  for rid in sorted(arm['outputs'].get('M6-slots8',{})):
   x=arm['outputs']['M6-slots8'][rid];y=other['outputs'].get('B0',{}).get(rid)
   r={'phase':tag+'-M6-slots8-vs-other-B0','id':rid,'n_off':x['n'],'n_on':y and y['n']}
   r.update(cmp(x,y) if y else {'same':None,'why':'no other-arm B0 ref'});rows.append(r)
 s={'rows':len(rows),'identical':sum(r['same'] is True for r in rows),'different':sum(r['same'] is False for r in rows),'missing':sum(r['same'] is None for r in rows),
    'off_arm':{'speculative':(off.get('arm') or {}).get('speculative'),'status':off['summary'].get('status'),'within_arm_all_identical':off['summary'].get('all_identical')},
    'on_arm_status':on['summary'].get('status'),'on_arm_within_all_identical':on['summary'].get('all_identical'),
    'note':'empirical cross-arm result only; not a guarantee, not a threshold'}
 json.dump({'summary':s,'rows':rows},open(out,'w'),indent=1);print(json.dumps(s));return 0
if __name__=='__main__':raise SystemExit(main(*sys.argv[1:4]))
