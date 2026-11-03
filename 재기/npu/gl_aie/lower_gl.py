"""Experimental typed GL -> AIE C++ backend for a static integer kernel subset.

Uses upstream GL lexer/parser/sema; no regex translation or prewritten body.
Rejects unsupported nodes, unknown calls, range overflow and out-of-buffer access.
Not a general GL compiler. Buffer ABI: a:int8[256], b:int8[1024], dst:int32[128].
"""
import json
import sys
from dataclasses import fields
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'geul-lang/ref'))
from geul.driver import load_program
from geul import ast as A, sema, types as T

HEADER='''#define NOCPP
#include <stdint.h>
#include <aie_api/aie.hpp>
static inline int32_t dot_loaded(aie::vector<int8_t,32> x, aie::vector<int8_t,32> w) {
    auto p = aie::mul(x,w);
    return aie::reduce_add(p.to_vector<int32_t>(0));
}
'''

class BackendError(ValueError): pass

class Emitter:
    def __init__(self):
        self.ranges={}
        self.residues={}
        self.buffers={'a':256,'b':1024,'dst':128}
        self.proofs=[]
        self.loads={}
        self.pending=[]
        self.load_count=0

    def interval(self,e):
        if isinstance(e,A.IntLit): return (e.value,e.value)
        if isinstance(e,A.Name) and e.name in self.ranges: return self.ranges[e.name]
        if isinstance(e,A.Binary) and e.op in ('+','-','*'):
            a,b=self.interval(e.left),self.interval(e.right)
            values=[{'+':lambda:x+y,'-':lambda:x-y,'*':lambda:x*y}[e.op]() for x in a for y in b]
            return min(values),max(values)
        if isinstance(e,(A.Call,A.SOVCall)):
            self.validate_call(e)
            return (-524288,524288)
        raise BackendError(f'No interval proof for {type(e).__name__}')

    def bounded(self,e):
        lo,hi=self.interval(e)
        if lo < -(2**31) or hi >= 2**31: raise BackendError('INT32 overflow not excluded')
        # Check intermediates too; cancellation must not hide an overflow.
        if isinstance(e,A.Binary):
            self.bounded(e.left)
            self.bounded(e.right)
        return lo,hi

    def residue(self,e):
        if isinstance(e,A.IntLit): return e.value%32
        if isinstance(e,A.Name): return self.residues.get(e.name)
        if isinstance(e,A.Binary):
            a,b=self.residue(e.left),self.residue(e.right)
            if e.op=='*' and (a==0 or b==0): return 0
            if a is not None and b is not None:
                if e.op=='+': return (a+b)%32
                if e.op=='-': return (a-b)%32
                if e.op=='*': return (a*b)%32
        return None

    def access(self,base,index,width,write=False):
        if not isinstance(base,A.Name) or base.name not in self.buffers: raise BackendError('unknown buffer')
        if write != (base.name=='dst'): raise BackendError('buffer direction violation')
        lo,hi=self.bounded(index)
        if lo<0 or hi+width>self.buffers[base.name]: raise BackendError('buffer bounds not proven')
        if width==32 and self.residue(index)!=0: raise BackendError('32-byte offset alignment not proven')
        self.proofs.append(dict(buffer=base.name,min_offset=lo,max_offset=hi,width=width,write=write))

    def validate_call(self,e):
        if e.callee_sym.link_name!='dot32': raise BackendError('unsupported intrinsic')
        args=e.resolved_args
        if len(args)!=4: raise BackendError('dot32 arity')
        self.access(args[0],args[1],32)
        self.access(args[2],args[3],32)
        return args

    def load(self,base,index):
        self.access(base,index,32)
        key=(self.expr(base),self.expr(index))
        if key not in self.loads:
            name=f'vec{self.load_count}'
            self.load_count+=1
            self.loads[key]=name
            self.pending.append(f'auto {name} = aie::load_v<32>({key[0]} + {key[1]});')
        return self.loads[key]

    def call(self,e):
        a,ai,b,bi=self.validate_call(e)
        return f'dot_loaded({self.load(a,ai)}, {self.load(b,bi)})'

    def nodes(self,node):
        if isinstance(node,A.Node):
            yield node
            for f in fields(node): yield from self.nodes(getattr(node,f.name))
        elif isinstance(node,(list,tuple)):
            for item in node: yield from self.nodes(item)

    def hoist(self,body):
        # Input buffers are read-only and runtime allocates disjoint output.
        # Only hoist loads whose offset variables are already available here.
        for node in self.nodes(body):
            if isinstance(node,(A.Call,A.SOVCall)) and node.callee_sym.link_name=='dot32':
                args=node.resolved_args
                for base,index in ((args[0],args[1]),(args[2],args[3])):
                    deps={n.name for n in self.nodes(index) if isinstance(n,A.Name)}
                    if deps <= self.ranges.keys(): self.load(base,index)

    def expr(self,e):
        if isinstance(e,A.IntLit): return str(e.value)
        if isinstance(e,A.Name):
            if e.name not in self.ranges and e.name not in self.buffers: raise BackendError('unknown name')
            return e.name
        if isinstance(e,A.Binary) and e.op in ('+','-','*','<'):
            return f'({self.expr(e.left)} {e.op} {self.expr(e.right)})'
        if isinstance(e,(A.Call,A.SOVCall)): return self.call(e)
        if isinstance(e,A.Index): return f'{self.expr(e.base)}[{self.expr(e.index)}]'
        raise BackendError(f'unsupported expression {type(e).__name__}')

    def block(self,block,indent=1):
        lines=[]
        pad='    '*indent
        for s in block.stmts:
            if isinstance(s,A.VarDecl):
                if s.static or s.const or not isinstance(s.sym.type,T.IntType) or s.sym.type.bits!=32 or not s.sym.type.signed: raise BackendError('only local signed INT32 supported')
                if s.name in self.ranges or s.name in self.buffers: raise BackendError('shadowing not supported')
                self.ranges[s.name]=self.bounded(s.init)
                self.residues[s.name]=self.residue(s.init)
                value=self.expr(s.init)
                lines.extend(pad+p for p in self.pending)
                self.pending.clear()
                lines.append(pad+f'int32_t {s.name} = {value};')
            elif isinstance(s,A.For):
                init,cond,step=s.init,s.cond,s.step
                if not (isinstance(init,A.VarDecl) and isinstance(init.init,A.IntLit) and init.init.value==0 and isinstance(init.sym.type,T.IntType) and init.sym.type.bits==32 and init.sym.type.signed): raise BackendError('static INT32 loop required')
                name=init.name
                if name in self.ranges or name in self.buffers: raise BackendError('loop shadowing unsupported')
                if not (isinstance(cond,A.Binary) and cond.op=='<' and isinstance(cond.left,A.Name) and cond.left.name==name and isinstance(cond.right,A.IntLit) and 0<cond.right.value<=1024): raise BackendError('bounded static loop required')
                if not (isinstance(step,A.Assign) and step.op=='=' and isinstance(step.target,A.Name) and step.target.name==name and isinstance(step.value,A.Binary) and step.value.op=='+' and isinstance(step.value.left,A.Name) and step.value.left.name==name and isinstance(step.value.right,A.IntLit) and step.value.right.value==1): raise BackendError('unit stride required')
                outer=self.ranges.copy()
                residues=self.residues.copy()
                self.hoist(s.body)
                lines.extend(pad+p for p in self.pending)
                self.pending.clear()
                loads=self.loads.copy()
                self.ranges[name]=(0,cond.right.value-1)
                lines.append(pad+f'for (int32_t {name}=0; {self.expr(cond)}; {name}={self.expr(step.value)}) {{')
                lines+=self.block(s.body,indent+1)
                lines.append(pad+'}')
                self.ranges=outer
                self.residues=residues
                self.loads=loads
            elif isinstance(s,A.Assign) and s.op=='=' and isinstance(s.target,A.Index):
                self.access(s.target.base,s.target.index,1,write=True)
                self.bounded(s.value)
                target,value=self.expr(s.target),self.expr(s.value)
                lines.extend(pad+p for p in self.pending)
                self.pending.clear()
                lines.append(pad+target+' = '+value+';')
            else: raise BackendError(f'unsupported statement {type(s).__name__}')
        return lines

def compile_gl(source):
    program=load_program(str(source),str(ROOT/'geul-lang/표준'),auto_std=False)
    unit=sema.analyze(program,fragment=True)
    if len(unit.functions)!=1 or len(unit.externs)!=1 or unit.globals: raise BackendError('one kernel and one intrinsic required')
    fn=unit.functions[0]
    if fn.type.ret is not None and not fn.type.ret.is_void(): raise BackendError('kernel must return void')
    expected=[('a',8),('b',8),('dst',32)]
    for p,(name,bits) in zip(fn.params,expected,strict=True):
        if p.name!=name or not isinstance(p.rtype,T.PtrType) or not isinstance(p.rtype.target,T.IntType) or p.rtype.target.bits!=bits or not p.rtype.target.signed: raise BackendError('kernel ABI mismatch')
    intrinsic=unit.externs[0]
    # Explicit signature validation, independent of symbol spelling.
    if intrinsic.link_name!='dot32' or len(intrinsic.params)!=4: raise BackendError('intrinsic ABI mismatch')
    sig=[str(p.rtype) for p in intrinsic.params]
    if sig!=['작은정수 참조','중간정수','작은정수 참조','중간정수'] or str(intrinsic.type.ret)!='중간정수': raise BackendError('intrinsic types mismatch')
    emitter=Emitter()
    body=emitter.block(fn.decl.body)
    code=HEADER+'extern "C" void geul_tile(int8_t *a, int8_t *b, int32_t *dst) {\n'+'\n'.join(body)+'\n}\n'
    return code,dict(frontend='upstream GL parser + sema',backend='restricted typed AST to AIE C++',optimizations=['loop-invariant input vector load hoisting','common vector load reuse','exact two-digit integer fusion'],buffer_access_proofs=emitter.proofs,contract='signed INT8 dot32 exact INT32; S0+256*S1; no floating-point transformation')

if __name__=='__main__':
    here=Path(__file__).parent
    code,proof=compile_gl(here/'tile.gl')
    (here/'generated.cc').write_text(code,encoding='utf-8')
    (here/'lowering.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print('GL type check, static bounds and INT32 range: PASS')
