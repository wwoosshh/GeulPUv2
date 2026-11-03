import numpy as np
from native_post import C

def quant(x,name):
    d,e=C.어텐션자리로(x,name)
    return d[1]*256+d[0],C.거듭제곱(e)

def attention(q,k,v,cache,first,scale):
    n=len(q)
    if n>64 or C.F2 or C.KV8:
        return C.어텐션(q,k,v,첫행=first,눈금=scale)
    old=cache.get('length',0)
    if old<n:
        for label,data in [('k',k),('v',v)]:
            new=quant(data[old:],label)
            cache[label]=tuple(np.concatenate([x,y]) for x,y in zip(cache[label],new)) if old else new
        cache['length']=n
    nq,pq=quant(q,'q')
    nk,pk=cache['k']
    nv,pv=cache['v']
    _,l,o=C._조각몫(nq,nk,nv,pq,pk,pv,np.arange(first,n),0,n,scale)
    return C.나눔(o,l[:,None])
