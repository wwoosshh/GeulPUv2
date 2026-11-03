import os
import subprocess
from pathlib import Path
from native_post import HERE,ROOT

def build():
    out=ROOT/'빌드/npu/gl-aie'
    tools=Path(os.environ['PEANO_INSTALL_DIR'])/'bin'
    subprocess.run([str(tools/'clang++.exe'),'--target=aie2p-none-unknown-elf','-O2','-std=c++20','-c',str(HERE/'pack24_v2.cc'),'-o',str(out/'pack24-v2-wrapper.o')],check=True)
    subprocess.run([str(tools/'clang++.exe'),'--target=aie2p-none-unknown-elf','-nostdlib','-Wl,-r',str(out/'gl.o'),str(out/'pack24-v2-wrapper.o'),'-o',str(out/'gl-packed-v2.o')],check=True)
    host=ROOT/'빌드/npu/next-stage'
    subprocess.run(['cl','/nologo','/LD','/O2','/MD',str(HERE/'unpack24.c'),'/Fo:'+str(host/'unpack24.obj'),'/link','/OUT:'+str(host/'unpack24.dll'),'/IMPLIB:'+str(host/'unpack24.lib')],check=True)

if __name__=='__main__':build()
