import os
import subprocess
from pathlib import Path
from native_post import HERE,ROOT
from aie.utils.config import cxx_header_path

def build():
    out=ROOT/'빌드/npu/gl-aie'
    tools=Path(os.environ['PEANO_INSTALL_DIR'])/'bin'
    obj=out/'gl-packed-v3.o'
    subprocess.run([str(tools/'clang++.exe'),'--target=aie2p-none-unknown-elf','-O2','-DNDEBUG','-std=c++20','-D__AIE_API_AIE_ADF_HPP__','-Wno-attributes','-Wno-macro-redefined','-I',cxx_header_path(),'-c',str(HERE/'pack24_fused.cc'),'-o',str(obj)],check=True)
    asm=subprocess.run([str(tools/'llvm-objdump.exe'),'-d',str(obj)],capture_output=True,text=True,check=True).stdout
    (HERE/'results/pack24-fused.asm').write_text(asm,encoding='utf8',newline='\n')
    if not (ROOT/'빌드/npu/next-stage/unpack24.dll').exists():
        raise RuntimeError('build SIMD unpack24.dll with build_pack24_v2 first')

if __name__=='__main__':build()
