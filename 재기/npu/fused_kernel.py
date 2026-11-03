"""Exact digit fusion on AIE: input rows 0:4 low, 4:8 high; four outputs."""
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.config import cxx_header_path

SOURCE = r'''
#define NOCPP
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void geul_fused(int8_t *a, int8_t *b, int32_t *c) {
  for (int r=0; r<4; ++r) {
    auto lo = aie::load_v<32>(a+r*32);
    auto hi = aie::load_v<32>(a+(r+4)*32);
    for (int n=0; n<32; ++n) {
      auto w = aie::load_v<32>(b+n*32);
      auto p0 = aie::mul(lo,w);
      auto p1 = aie::mul(hi,w);
      int32_t s0 = aie::reduce_add(p0.to_vector<int32_t>(0));
      int32_t s1 = aie::reduce_add(p1.to_vector<int32_t>(0));
      c[r*32+n] = s0 + 256*s1;
    }
  }
}
'''

@iron.jit
def fused_kernel(a: In, b: In, c: Out, *, frames: CompileTime[int]):
    at = np.ndarray[(256,), np.dtype[np.int8]]
    bt = np.ndarray[(1024,), np.dtype[np.int8]]
    ct = np.ndarray[(128,), np.dtype[np.int32]]
    af = np.ndarray[(frames*256,), np.dtype[np.int8]]
    bf = np.ndarray[(frames*1024,), np.dtype[np.int8]]
    cf = np.ndarray[(frames*128,), np.dtype[np.int32]]
    fa, fb, fc = ObjectFifo(at,name="a"), ObjectFifo(bt,name="b"), ObjectFifo(ct,name="c")
    kernel = ExternalFunction("geul_fused",source_string=SOURCE,arg_types=[at,bt,ct],include_dirs=[cxx_header_path()])
    def body(a,b,c,kernel):
        for _ in range_(frames):
            aa,bb,cc = a.acquire(1),b.acquire(1),c.acquire(1)
            kernel(aa,bb,cc)
            a.release(1)
            b.release(1)
            c.release(1)
    worker = Worker(body,fn_args=[fa.cons(),fb.cons(),fc.prod(),kernel])
    runtime = Runtime()
    with runtime.sequence(af,bf,cf) as (a,b,c):
        runtime.start(worker)
        runtime.fill(fa.prod(),a)
        runtime.fill(fb.prod(),b)
        runtime.drain(fc.cons(),c,wait=True)
    return Program(iron.get_current_device(),runtime).resolve_program()

class NpuFusedTiles:
    fused = True
    def __init__(self, frames=2048):
        if frames < 1:
            raise ValueError("frames must be positive")
        self.frames = frames
        self.a = iron.zeros((frames*256,),dtype=np.int8,device="npu")
        self.b = iron.zeros((frames*1024,),dtype=np.int8,device="npu")
        self.c = iron.zeros((frames*128,),dtype=np.int32,device="npu")
        self.ah,self.bh = self.a.numpy(),self.b.numpy()
        self.calls = 0

    def __call__(self,a,b):
        frames = len(a)
        if a.dtype!=np.int8 or b.dtype!=np.int8 or a.shape!=(frames,8,32) or b.shape!=(frames,32,32) or not 0<frames<=self.frames:
            raise ValueError("invalid tile shapes or types")
        self.ah.fill(0)
        self.bh.fill(0)
        self.ah[:a.size] = a.ravel()
        self.bh[:b.size] = b.ravel()
        self.a._sync_to_device()
        self.b._sync_to_device()
        fused_kernel(self.a,self.b,self.c,frames=self.frames)
        self.calls += 1
        return self.c.numpy().reshape(self.frames,4,32)[:frames].copy()
