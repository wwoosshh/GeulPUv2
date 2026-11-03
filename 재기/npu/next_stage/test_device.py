"""Probe scalar FP32 FMA semantics before attempting device finalization."""
import json,traceback,os
import numpy as np
import aie.iron as iron
from aie.iron import In,Out,ExternalFunction,ObjectFifo,Worker,Runtime,Program
from native_post import HERE,C
from exact_linear import difference
from aie.utils.config import cxx_header_path

SOURCE='''#include <stdint.h>
extern "C" void fp_probe(float *a,float *out) {
  for(int i=0;i<256;i++) out[i]=__builtin_fmaf(a[i],a[256+i],a[512+i]);
}
'''
VECTOR=os.environ.get('GEUL_FP_VECTOR')=='1'
if VECTOR:
    SOURCE='''#define NOCPP
#include <aie_api/aie.hpp>
extern "C" void fp_probe(float *a,float *out) {
  for(int i=0;i<256;i+=16) {
    auto x=aie::load_v<16>(a+i);
    auto y=aie::load_v<16>(a+256+i);
    auto z=aie::load_v<16>(a+512+i);
    aie::accum<accfloat,16> acc(z);
    auto result=aie::mac(acc,x,y);
    aie::store_v(out+i,result.to_vector<float>());
  }
}
'''
@iron.jit
def probe(a:In,b:Out):
    at=np.ndarray[(768,),np.dtype[np.float32]]
    bt=np.ndarray[(256,),np.dtype[np.float32]]
    fa,fb=ObjectFifo(at,name='a'),ObjectFifo(bt,name='b')
    kernel=ExternalFunction('fp_probe',source_string=SOURCE,arg_types=[at,bt],include_dirs=[cxx_header_path()])
    def body(a,b,k):
        aa,bb=a.acquire(1),b.acquire(1)
        k(aa,bb)
        a.release(1);b.release(1)
    worker=Worker(body,fn_args=[fa.cons(),fb.prod(),kernel])
    runtime=Runtime()
    with runtime.sequence(at,bt) as (a,b):
        runtime.start(worker);runtime.fill(fa.prod(),a);runtime.drain(fb.cons(),b,wait=True)
    return Program(iron.get_current_device(),runtime).resolve_program()

report={'scope':'AIE vector MAC' if VECTOR else 'AIE scalar FP32 FMA','compatible':False}
stem='device-vector' if VECTOR else 'device'
try:
    rng=np.random.default_rng(311)
    values=rng.uniform(-2,2,(3,256)).astype(np.float32)
    values[:,0]=[np.float32(1+2**-23),np.float32(1-2**-23),-1]
    values[:,1]=[np.finfo(np.float32).tiny,0.5,0]
    a=iron.zeros((768,),dtype=np.float32,device='npu')
    b=iron.zeros((256,),dtype=np.float32,device='npu')
    a.numpy()[:]=values.ravel();a._sync_to_device()
    probe(a,b)
    actual=b.numpy().copy();ref=C.fma32(*values)
    report['difference']=difference(ref,actual)
    report['compatible']=report['difference']['different_bits_elements']==0
    np.savez(HERE/f'results/{stem}-probe.npz',input=values,reference=ref,actual=actual)
except Exception as exc:
    report['error']=repr(exc);report['traceback']=traceback.format_exc()
finally:
    (HERE/f'results/{stem}.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
