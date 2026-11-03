"""Identical NPU transport for separately compiled GL, C, and AIE C++ objects."""
import sys
from pathlib import Path
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_

HERE=Path(__file__).parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent))
from planned_linear import PlannedLinear

def make_kernel(kind):
    obj=ROOT/f'빌드/npu/gl-aie/{kind}.o'
    if kind not in ('gl','gl_row1','c','c_unrolled','aie_cpp') or not obj.exists(): raise ValueError('build object first')
    output_size=32 if kind=='gl_row1' else 128
    @iron.jit
    @iron.compileconfig(object_files=[obj])
    def design(a: In,b: In,c: Out,*,frames: CompileTime[int]):
        at=np.ndarray[(256,),np.dtype[np.int8]]
        bt=np.ndarray[(1024,),np.dtype[np.int8]]
        ct=np.ndarray[(output_size,),np.dtype[np.int32]]
        af=np.ndarray[(frames*256,),np.dtype[np.int8]]
        bf=np.ndarray[(frames*1024,),np.dtype[np.int8]]
        cf=np.ndarray[(frames*output_size,),np.dtype[np.int32]]
        fa,fb,fc=ObjectFifo(at,name='a'),ObjectFifo(bt,name='b'),ObjectFifo(ct,name='c')
        kernel=Kernel('geul_tile',str(obj),[at,bt,ct])
        def body(a,b,c,kernel):
            for _ in range_(frames):
                aa,bb,cc=a.acquire(1),b.acquire(1),c.acquire(1)
                kernel(aa,bb,cc)
                a.release(1)
                b.release(1)
                c.release(1)
        worker=Worker(body,fn_args=[fa.cons(),fb.cons(),fc.prod(),kernel])
        runtime=Runtime()
        with runtime.sequence(af,bf,cf) as (a,b,c):
            runtime.start(worker)
            runtime.fill(fa.prod(),a)
            runtime.fill(fb.prod(),b)
            runtime.drain(fc.cons(),c,wait=True)
        return Program(iron.get_current_device(),runtime).resolve_program()
    return design

class Device:
    fused=True
    def __init__(self,kind,frames=2048):
        self.kind,self.frames=kind,frames
        self.rows=1 if kind=='gl_row1' else 4
        self.kernel=make_kernel(kind)
        self.a=iron.zeros((frames*256,),dtype=np.int8,device='npu')
        self.b=iron.zeros((frames*1024,),dtype=np.int8,device='npu')
        self.c=iron.zeros((frames*self.rows*32,),dtype=np.int32,device='npu')
        self.ah,self.bh=self.a.numpy(),self.b.numpy()
        self.calls=0

    def __call__(self,a,b):
        frames=len(a)
        if a.dtype!=np.int8 or b.dtype!=np.int8 or a.shape!=(frames,8,32) or b.shape!=(frames,32,32) or not 0<frames<=self.frames:
            raise ValueError('invalid tiles')
        self.ah.fill(0)
        self.bh.fill(0)
        self.ah[:a.size]=a.ravel()
        self.bh[:b.size]=b.ravel()
        self.a._sync_to_device()
        self.b._sync_to_device()
        self.kernel(self.a,self.b,self.c,frames=self.frames)
        self.calls+=1
        return self.c.numpy().reshape(self.frames,self.rows,32)[:frames].copy()
