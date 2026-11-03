import os
import subprocess
from pathlib import Path
from native_post import HERE,ROOT

def build():
    out=ROOT/'빌드/npu/gl-aie'
    tools=Path(os.environ['PEANO_INSTALL_DIR'])/'bin'
    subprocess.run([str(tools/'clang++.exe'),'--target=aie2p-none-unknown-elf','-O2','-std=c++20','-c',str(HERE/'pack24.cc'),'-o',str(out/'pack24-wrapper.o')],check=True)
    subprocess.run([str(tools/'clang++.exe'),'--target=aie2p-none-unknown-elf','-nostdlib','-Wl,-r',str(out/'gl.o'),str(out/'pack24-wrapper.o'),'-o',str(out/'gl-packed.o')],check=True)

if __name__=='__main__':build()
