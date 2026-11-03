"""Build all comparison kernels using the same Peano version, target and -O2."""
import hashlib
import json
import os
import subprocess
import struct
import time
from pathlib import Path
from lower_gl import compile_gl,ROOT
from aie.utils.config import cxx_header_path

HERE=Path(__file__).parent
OUT=ROOT/'빌드/npu/gl-aie'
OUT.mkdir(parents=True,exist_ok=True)
generated,proof=compile_gl(HERE/'tile.gl')
(HERE/'generated.cc').write_text(generated,encoding='utf-8')
(HERE/'lowering.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
bin_dir=Path(os.environ['PEANO_INSTALL_DIR'])/'bin'
report={'target':'aie2p-none-unknown-elf','optimization':'-O2','kernels':{},'commands':[]}
report['compiler']=subprocess.run([str(bin_dir/'clang.exe'),'--version'],check=True,capture_output=True,text=True).stdout

def elf_text(path):
    data=path.read_bytes()
    assert data[:4]==b'\x7fELF' and data[5]==1,'expected little-endian ELF'
    if data[4]==1:
        shoff=struct.unpack_from('<I',data,32)[0]
        step,count,names=struct.unpack_from('<HHH',data,46)
        def section(i):
            base=shoff+i*step
            return (struct.unpack_from('<I',data,base)[0],*struct.unpack_from('<II',data,base+16))
    elif data[4]==2:
        shoff=struct.unpack_from('<Q',data,40)[0]
        step,count,names=struct.unpack_from('<HHH',data,58)
        def section(i):
            base=shoff+i*step
            return (struct.unpack_from('<I',data,base)[0],*struct.unpack_from('<QQ',data,base+24))
    else: raise ValueError('unknown ELF class')
    _,off,size=section(names)
    strings=data[off:off+size]
    chunks=[]
    for i in range(count):
        name,off,size=section(i)
        label=strings[name:].split(b'\0',1)[0]
        if label==b'.text' or label.startswith(b'.text.'):
            chunks.append(data[off:off+size])
    result=b''.join(chunks)
    assert result,'no machine text found'
    return result
for name,source,language in [('gl','generated.cc','c++'),('c','native.c','c'),('c_unrolled','native_unrolled.c','c'),('aie_cpp','native_aie.cc','c++')]:
    obj=OUT/f'{name}.o'
    compiler=bin_dir/('clang.exe' if language=='c' else 'clang++.exe')
    cmd=[str(compiler),'--target=aie2p-none-unknown-elf','-O2','-DNDEBUG','-x',language,
         '-std=c11' if language=='c' else '-std=c++20','-D__AIE_API_AIE_ADF_HPP__',
         '-Wno-attributes','-Wno-macro-redefined','-I',cxx_header_path(),'-c',str(HERE/source),'-o',str(obj)]
    start=time.perf_counter()
    p=subprocess.run(cmd,capture_output=True,text=True)
    report['commands'].append(cmd)
    (OUT/f'{name}.compile.log').write_text(p.stdout+p.stderr,encoding='utf-8')
    if p.returncode: raise RuntimeError(p.stderr)
    seconds=time.perf_counter()-start
    dis=subprocess.run([str(bin_dir/'llvm-objdump.exe'),'-d',str(obj)],capture_output=True,text=True,check=True)
    (OUT/f'{name}.asm').write_text(dis.stdout,encoding='utf-8')
    text_path=OUT/f'{name}.text.bin'
    text_path.write_bytes(elf_text(obj))
    report['kernels'][name]={'source':source,'source_sha256':hashlib.sha256((HERE/source).read_bytes()).hexdigest(),
        'object':str(obj),'object_sha256':hashlib.sha256(obj.read_bytes()).hexdigest(),
        'text_sha256':hashlib.sha256(text_path.read_bytes()).hexdigest(),'text_bytes':text_path.stat().st_size,'compile_seconds':seconds}
report['gl_source_sha256']=hashlib.sha256((HERE/'tile.gl').read_bytes()).hexdigest()
report['gl_vs_aie_cpp_text_equal']=(OUT/'gl.text.bin').read_bytes()==(OUT/'aie_cpp.text.bin').read_bytes()
(HERE/'build-results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
