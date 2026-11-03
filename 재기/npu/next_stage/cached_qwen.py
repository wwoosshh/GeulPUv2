"""Qwen3 Q8/Q4 experimental cached decoder; GL NPU linears, CPU attention.

Caches FP32 RoPE K/V. The reference attention still re-quantizes cached K/V.
This is an executable integration prototype, not a production GL-engine port.
"""
import sys
import time
import numpy as np
from native_post import ROOT,C
sys.path.insert(0,str(ROOT/'시험'))
import 참조7B as R

class CachedQwen:
    def __init__(self,g,linear,quant_cache=False,context_capacity=256):
        self.g,self.backend=g,linear
        self.quant_cache=quant_cache
        self.last_profile={}
        s=R.설정(g)
        if s['구조']!='qwen3': raise ValueError('Qwen3 only')
        self.s=s
        self.embed=R.가중(g,'token_embd.weight')
        if self.embed[0] not in ('Q8_0','Q4_0'): raise ValueError('quantized embeddings required')
        self.d=self.embed[1].shape[1]
        self.nh,self.nkv=s['머리'],s['KV머리']
        self.hd=int(s['머리크기'] or self.d//self.nh)
        if self.hd%32 or int(s['돌림'] or self.hd)!=self.hd: raise ValueError('full aligned RoPE heads required')
        self.a,self.kvw=self.nh*self.hd,self.nkv*self.hd
        self.scale=np.float32(1/np.sqrt(np.float64(self.hd)))
        self.context_capacity=context_capacity
        self.cos,self.sin=C.rope표(context_capacity,s['θ'],hd=self.hd)
        self.layers=[]
        for i in range(s['층']):
            p=f'blk.{i}.'
            ws=[R.가중(g,p+f'attn_{key}.weight') for key in ('q','k','v')]
            qkv=(np.concatenate([w[1] for w in ws]),np.concatenate([w[2] for w in ws]),
                 np.concatenate([R.치우침(g,p+f'attn_{key}.bias',w[1].shape[0]) for key,w in zip(('q','k','v'),ws)]))
            other={key:R.가중(g,p+key+'.weight') for key in ('attn_output','ffn_gate','ffn_up','ffn_down')}
            if any(w[0] not in ('Q8_0','Q4_0') for w in ws+list(other.values())): raise ValueError('Q8/Q4 linear required')
            norms={key:g.풀기(p+key+'.weight') for key in ('attn_norm','attn_q_norm','attn_k_norm','ffn_norm')}
            self.layers.append((qkv,other,norms))
        self.output=R.가중(g,'output.weight' if 'output.weight' in g.tensors else 'token_embd.weight')
        self.norm=g.풀기('output_norm.weight')
        self.reset()

    def reset(self):
        self.pos=0
        self.cache=[None]*len(self.layers)
        self.quantized=[[{} for _ in range(self.nkv)] for _ in self.layers]

    def mul(self,x,w):
        return self.linear(x,w[1],w[2],np.zeros(w[1].shape[0],np.float32),자리수=2)

    def linear(self,*args,**kwargs):
        start=time.perf_counter()
        out=self.backend(*args,**kwargs)
        self.last_profile['linear_seconds']+=time.perf_counter()-start
        self.last_profile['linear_calls']+=1
        for key,value in getattr(self.backend,'profile',{}).items():
            self.last_profile[key]=self.last_profile.get(key,0)+value
        return out

    def step(self,tokens):
        self.last_profile={'linear_seconds':0.,'attention_seconds':0.,'linear_calls':0}
        if self.pos+len(tokens)>self.context_capacity: raise ValueError('prototype context capacity exceeded')
        eq,ed=self.embed[1:3]
        x=(eq[tokens].astype(np.float32)*np.repeat(ed[tokens],32,axis=1)).astype(np.float32)
        positions=np.arange(self.pos,self.pos+len(tokens))
        for i,(qkv,weights,norms) in enumerate(self.layers):
            h=C.rms정규화(x,norms['attn_norm'],self.s['eps'])
            r=self.linear(h,*qkv,자리수=2)
            r=np.concatenate([C.머리정규화(r[:,:self.a],norms['attn_q_norm'],self.s['eps'],self.hd),
                              C.머리정규화(r[:,self.a:self.a+self.kvw],norms['attn_k_norm'],self.s['eps'],self.hd),r[:,self.a+self.kvw:]],axis=1)
            q=[C.rope(r[:,a*self.hd:(a+1)*self.hd],self.cos,self.sin,positions) for a in range(self.nh)]
            k=[C.rope(r[:,self.a+a*self.hd:self.a+(a+1)*self.hd],self.cos,self.sin,positions) for a in range(self.nkv)]
            v=[np.ascontiguousarray(r[:,self.a+self.kvw+a*self.hd:self.a+self.kvw+(a+1)*self.hd]) for a in range(self.nkv)]
            if self.cache[i] is not None:
                oldk,oldv=self.cache[i]
                k=[np.concatenate([old,new]) for old,new in zip(oldk,k)]
                v=[np.concatenate([old,new]) for old,new in zip(oldv,v)]
            self.cache[i]=(k,v)
            heads=[]
            attention_start=time.perf_counter()
            for a in range(self.nh):
                qp=np.concatenate([np.zeros((self.pos,self.hd),np.float32),q[a]])
                hidx=a//(self.nh//self.nkv)
                if self.quant_cache:
                    from short_attention import attention
                    heads.append(attention(qp,k[hidx],v[hidx],self.quantized[i][hidx],self.pos,self.scale))
                else:
                    heads.append(C.어텐션(qp,k[hidx],v[hidx],첫행=self.pos,눈금=self.scale))
            self.last_profile['attention_seconds']+=time.perf_counter()-attention_start
            y=np.concatenate(heads,axis=1)
            x=(x+self.mul(y,weights['attn_output'])).astype(np.float32)
            h=C.rms정규화(x,norms['ffn_norm'],self.s['eps'])
            f=C.swiglu(self.mul(h,weights['ffn_gate']),self.mul(h,weights['ffn_up']))
            x=(x+self.mul(f,weights['ffn_down'])).astype(np.float32)
        self.pos+=len(tokens)
        h=C.rms정규화(x[-1:],self.norm,self.s['eps'])
        return self.mul(h,self.output)

    @property
    def kv_bytes(self):
        return sum(a.nbytes for pair in self.cache if pair is not None for heads in pair for a in heads)
