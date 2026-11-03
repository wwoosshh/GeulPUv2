import json
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'geul-lang/ref'))
from geul import driver,codegen
original=codegen.Asm
captured=[]
def capture(*args,**kwargs):
    asm=original(*args,**kwargs)
    captured.append(asm)
    return asm
codegen.Asm=capture
report={}
for name in ('bench','bench_opt'):
    driver.compile_file(str(root/f'재기/language/{name}.gl'),str(root/f'빌드/language/{name}-inspect.exe'),std_dir=str(root/'geul-lang/표준'))
    labels=captured[-1].labels
    report[name]={key:hex(0x140001000+labels[key]) for key in ('처리','시작하기')}
(root/'빌드/language/function-addresses.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
