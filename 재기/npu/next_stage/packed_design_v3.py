"""Hold one weight tile in each NPU core while processing four row groups."""
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from native_post import HERE, ROOT

WORKERS=4
REUSE=4
OBJECT=ROOT/'빌드/npu/gl-aie/gl-packed-v3.o'


@iron.jit
@iron.compileconfig(object_files=[OBJECT])
def design(a: In,b: In,c: Out,*,frames: CompileTime[int]):
    at=np.ndarray[(256,),np.dtype[np.int8]]
    bt=np.ndarray[(1024,),np.dtype[np.int8]]
    ct=np.ndarray[(104,),np.dtype[np.int32]]
    af=np.ndarray[(frames*256,),np.dtype[np.int8]]
    bf=np.ndarray[(frames//REUSE*1024,),np.dtype[np.int8]]
    cf=np.ndarray[(frames*104,),np.dtype[np.int32]]
    fa=ObjectFifo(np.ndarray[(WORKERS*256,),np.dtype[np.int8]],name='a')
    fb=ObjectFifo(np.ndarray[(WORKERS*1024,),np.dtype[np.int8]],name='b')
    fc=ObjectFifo(np.ndarray[(WORKERS*104,),np.dtype[np.int32]],name='c')
    aa=fa.cons().split([i*256 for i in range(WORKERS)],obj_types=[at]*WORKERS,names=[f'a{i}' for i in range(WORKERS)])
    bb=fb.cons().split([i*1024 for i in range(WORKERS)],obj_types=[bt]*WORKERS,names=[f'b{i}' for i in range(WORKERS)])
    cc=fc.prod().join([i*104 for i in range(WORKERS)],obj_types=[ct]*WORKERS,names=[f'c{i}' for i in range(WORKERS)])
    kernel=Kernel('geul_tile_packed',str(OBJECT),[at,bt,ct])
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
    (HERE/f'results/packed24-v3-{frames}.mlir').write_text(str(module),encoding='utf8')
    return module


