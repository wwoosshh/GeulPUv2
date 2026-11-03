"""Hold one weight tile in each NPU core while processing four row groups."""
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from native_post import HERE, ROOT

WORKERS=4
REUSE=4
OBJECT=ROOT/'빌드/npu/gl-aie/gl.o'


@iron.jit
@iron.compileconfig(object_files=[OBJECT])
def design(a: In,b: In,c: Out,*,frames: CompileTime[int]):
    at=np.ndarray[(256,),np.dtype[np.int8]]
    bt=np.ndarray[(1024,),np.dtype[np.int8]]
    ct=np.ndarray[(128,),np.dtype[np.int32]]
    af=np.ndarray[(frames*256,),np.dtype[np.int8]]
    bf=np.ndarray[(frames//REUSE*1024,),np.dtype[np.int8]]
    cf=np.ndarray[(frames*128,),np.dtype[np.int32]]
    fa=ObjectFifo(np.ndarray[(WORKERS*256,),np.dtype[np.int8]],name='a')
    fb=ObjectFifo(np.ndarray[(WORKERS*1024,),np.dtype[np.int8]],name='b')
    fc=ObjectFifo(np.ndarray[(WORKERS*128,),np.dtype[np.int32]],name='c')
    aa=fa.cons().split([i*256 for i in range(WORKERS)],obj_types=[at]*WORKERS,names=[f'a{i}' for i in range(WORKERS)])
    bb=fb.cons().split([i*1024 for i in range(WORKERS)],obj_types=[bt]*WORKERS,names=[f'b{i}' for i in range(WORKERS)])
    cc=fc.prod().join([i*128 for i in range(WORKERS)],obj_types=[ct]*WORKERS,names=[f'c{i}' for i in range(WORKERS)])
    kernel=Kernel('geul_tile',str(OBJECT),[at,bt,ct])
    def body(a,b,c,kernel):
        for _ in range_(frames//WORKERS//REUSE):
            weight=b.acquire(1)
            for _ in range_(REUSE):
                activation=a.acquire(1)
                result=c.acquire(1)
                kernel(activation,weight,result)
                a.release(1)
                c.release(1)
            b.release(1)
    cores=[Worker(body,fn_args=[aa[i].cons(),bb[i].cons(),cc[i].prod(),kernel]) for i in range(WORKERS)]
    runtime=Runtime()
    with runtime.sequence(af,bf,cf) as (a,b,c):
        runtime.start(*cores)
        runtime.fill(fa.prod(),a)
        runtime.fill(fb.prod(),b)
        runtime.drain(fc.cons(),c,wait=True)
    module=Program(iron.get_current_device(),runtime).resolve_program()
    (HERE/f'results/reuse4-{frames}.mlir').write_text(str(module),encoding='utf8')
    return module


class ReuseDevice:
    output_words=128
    kernel=staticmethod(design)
    frames=2048
    rows=4
    fused=True
    def __init__(self, frames=2048):
        if frames < WORKERS*REUSE or frames % (WORKERS*REUSE):
            raise ValueError('frame capacity must be a positive multiple of 16')
        self.frames=frames
        self.a=iron.zeros((self.frames*256,),dtype=np.int8,device='npu')
        self.b=iron.zeros((self.frames//REUSE*1024,),dtype=np.int8,device='npu')
        self.c=iron.zeros((self.frames*self.output_words,),dtype=np.int32,device='npu')
        self.ah,self.bh=self.a.numpy(),self.b.numpy()
        self.last_calls=0
        self.fallback=None

    def __call__(self,a,b):
        # Single-row callers keep the established four-worker transport.
        if self.fallback is None:
            from multi_device import MultiDevice
            self.fallback=MultiDevice(4)
        return self.fallback(a,b)

    def decode_result(self,raw,activations,weights):
        return raw.reshape(-1,4,32)

    def grouped(self,inputs,weights,groups):
        kb=inputs.shape[1]
        span=len(weights)
        padded_span=(span+WORKERS-1)//WORKERS*WORKERS
        padded_groups=(groups+REUSE-1)//REUSE*REUSE
        source=np.zeros((padded_groups,kb,2,4,32),np.int8)
        source[:groups]=inputs
        bank=np.zeros((padded_span,32,32),np.int8)
        bank[:span]=weights
        output=np.empty((padded_groups,padded_span,4,32),np.int32)
        count=padded_groups//REUSE*padded_span
        batch=self.frames//REUSE
        self.last_calls=0
        for start in range(0,count,batch):
            stop=min(start+batch,count)
            tasks=np.arange(start,stop).reshape(-1,WORKERS)
            group=(tasks//padded_span)[:,None,:]*REUSE+np.arange(REUSE)[None,:,None]
            weight_index=(tasks%padded_span)[:,None,:]
            activations=source[group,weight_index%kb].reshape(-1,8,32)
            packed_weights=bank[(tasks%padded_span).ravel()]
            self.ah.fill(0)
            self.bh.fill(0)
            self.ah[:activations.size]=activations.ravel()
            self.bh[:packed_weights.size]=packed_weights.ravel()
            self.a._sync_to_device()
            self.b._sync_to_device()
            self.kernel(self.a,self.b,self.c,frames=self.frames)
            raw=self.c.numpy().reshape(-1,self.output_words)[:len(activations)].copy()
            values=self.decode_result(raw,activations,packed_weights)
            output[group,weight_index]=values.reshape(len(tasks),REUSE,WORKERS,4,32)
            self.last_calls+=1
        return np.ascontiguousarray(output[:groups,:span]).reshape(-1,4,32)
