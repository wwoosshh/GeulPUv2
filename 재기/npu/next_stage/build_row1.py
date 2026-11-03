import json,os,sys,subprocess
from pathlib import Path
from native_post import HERE,ROOT
sys.path.insert(0,str(HERE.parent/'gl_aie'))
from lower_gl import compile_gl
from aie.utils.config import cxx_header_path
p=HERE/'tile_row1.gl'
p.write_text((HERE.parent/'gl_aie/tile.gl').read_text(encoding='utf8').replace('r < 4','r < 1'),encoding='utf8',newline='\n')
code,proof=compile_gl(p)
assert all(x['max_offset']+x['width']<=32 for x in proof['buffer_access_proofs'] if x['write'])
out=ROOT/'빌드/npu/gl-aie'
(out/'row1.cc').write_text(code,encoding='utf8')
subprocess.run([str(Path(os.environ['PEANO_INSTALL_DIR'])/'bin/clang++.exe'),'--target=aie2p-none-unknown-elf','-O2','-DNDEBUG','-std=c++20','-D__AIE_API_AIE_ADF_HPP__','-Wno-attributes','-Wno-macro-redefined','-I',cxx_header_path(),'-c',str(out/'row1.cc'),'-o',str(out/'gl_row1.o')],check=True)
