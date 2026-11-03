"""Reuse eight activation rows and 32 weight rows per exact dot32 tile."""
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.config import cxx_header_path

ROWS, COLS, FRAMES = 8, 32, 128
SOURCE = r'''
#define NOCPP
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void geul_tile(int8_t *a, int8_t *b, int32_t *c) {
  for (int r=0; r<8; ++r) {
    auto x = aie::load_v<32>(a+r*32);
    for (int n=0; n<32; ++n) {
      auto y = aie::load_v<32>(b+n*32);
      auto p = aie::mul(x,y);
      c[r*32+n] = aie::reduce_add(p.to_vector<int32_t>(0));
    }
  }
}
'''

@iron.jit
def tiled_kernel(a: In, b: In, c: Out, *, frames: CompileTime[int]):
    at = np.ndarray[(ROWS*32,), np.dtype[np.int8]]
    bt = np.ndarray[(COLS*32,), np.dtype[np.int8]]
    ct = np.ndarray[(ROWS*COLS,), np.dtype[np.int32]]
    af = np.ndarray[(frames*ROWS*32,), np.dtype[np.int8]]
    bf = np.ndarray[(frames*COLS*32,), np.dtype[np.int8]]
    cf = np.ndarray[(frames*ROWS*COLS,), np.dtype[np.int32]]
    fa, fb, fc = ObjectFifo(at,name="a"), ObjectFifo(bt,name="b"), ObjectFifo(ct,name="c")
    kernel = ExternalFunction("geul_tile",source_string=SOURCE,arg_types=[at,bt,ct],include_dirs=[cxx_header_path()])
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

class NpuTiles:
    def __init__(self, frames=FRAMES):
        if frames < 1:
            raise ValueError("frames must be positive")
        self.frames = frames
        self.a = iron.zeros((frames*ROWS*32,),dtype=np.int8,device="npu")
        self.b = iron.zeros((frames*COLS*32,),dtype=np.int8,device="npu")
        self.c = iron.zeros((frames*ROWS*COLS,),dtype=np.int32,device="npu")
        # Retain host views: numpy() would synchronize old device contents each call.
        self.ah, self.bh = self.a.numpy(), self.b.numpy()
        self.calls = 0

    def __call__(self,a,b):
        frames = len(a)
        if a.dtype != np.int8 or b.dtype != np.int8 or a.shape != (frames,ROWS,32) or b.shape != (frames,COLS,32) or not 0 < frames <= self.frames:
            raise ValueError("invalid tile shapes or types")
        self.ah.fill(0)
        self.bh.fill(0)
        self.ah[:a.size] = a.ravel()
        self.bh[:b.size] = b.ravel()
        # A retained NumPy view does not mark an XRT tensor dirty. Runtime.to()
        # skips synchronization when its device is already 'npu'. Flush writes.
        self.a._sync_to_device()
        self.b._sync_to_device()
        tiled_kernel(self.a,self.b,self.c,frames=self.frames)
        self.calls += 1
        return self.c.numpy().reshape(self.frames,ROWS,COLS)[:frames].copy()
