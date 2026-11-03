"""Generate matching source optimizations without changing either compiler."""
from pathlib import Path
import re
p=Path(__file__).parent
gl=['[정수 참조 원본에서 정수 참조 대상에 정수 길이로 정수 모드를 처리]는 -> 정수 {']
c=['__declspec(noinline) int64_t process(const int64_t *a, int64_t *b, int64_t n, int mode) {']
gl += ['모드 == 3이면 {', '정수 s0 = 0. 정수 s1 = 0. 정수 s2 = 0. 정수 s3 = 0. 정수 i = 0.', '(i + 3 < 길이) 동안 {']
c += ['if(mode==3) { int64_t s0=0,s1=0,s2=0,s3=0,i=0; for(;i+3<n;i+=4) {']
for j in range(4):
    gl += [f's{j} += 원본[i + {j}].']
    c += [f's{j} += a[i+{j}];']
gl += ['i += 4. }', '(i < 길이) 동안 { s0 += 원본[i]. i += 1. }', '반환 ((s0 + s1) + s2) + s3. }']
c += ['} for(;i<n;++i) s0+=a[i]; return ((s0+s1)+s2)+s3; }']
gl += ['모드 == 2이면 { 반복 (정수 t = 0: t < 길이: t += 1024) { 반복 (정수 c = 0: c < 32: c += 1) { 반복 (정수 r = 0: r < 32: r += 1) { 대상[t + c * 32 + r] = 원본[t + r * 32 + c]. } } } 반환 0. }']
c += ['if(mode==2) { for(int64_t t=0;t<n;t+=1024) for(int64_t c=0;c<32;++c) for(int64_t r=0;r<32;++r) b[t+c*32+r]=a[t+r*32+c]; return 0; }']
for mode in (0,1):
    gl += [f'모드 == {mode}이면 {{ 정수 i = 0. (i + 3 < 길이) 동안 {{']
    c += [f'if(mode=={mode}) {{ int64_t i=0; for(;i+3<n;i+=4) {{']
    for j in range(4):
        gv=f'원본[i + {j}]'
        cv=f'a[i+{j}]'
        if mode==1: gv=f'({gv} * 3 + 7) & 65535'; cv=f'({cv}*3+7)&65535'
        gl += [f'대상[i + {j}] = {gv}.']
        c += [f'b[i+{j}]={cv};']
    gv='원본[i]' if mode==0 else '(원본[i] * 3 + 7) & 65535'
    cv='a[i]' if mode==0 else '(a[i]*3+7)&65535'
    gl += [f'i += 4. }} (i < 길이) 동안 {{ 대상[i] = {gv}. i += 1. }} 반환 0. }}']
    c += [f'}} for(;i<n;++i) b[i]={cv}; return 0; }}']
gl += ['반환 0. }']
c += ['return 0; }']
original=(p/'bench.gl').read_text(encoding='utf-8')
body=re.sub(r'\b(s[0-3]|i) \+= ([^.]+)\.',lambda m:f'{m[1]} = {m[1]} + {m[2]}.','\n'.join(gl))
(p/'bench_opt.gl').write_text('포함 "표준/시간.gl"\n'+body+'\n\n'+original[original.index('[시작하기]'):],encoding='utf-8',newline='\n')
original=(p/'bench.c').read_text()
start=original.index('__declspec(noinline)')
end=original.index('double now_ms')
(p/'bench_opt.c').write_text(original[:start]+'\n'.join(c)+'\n'+original[end:],encoding='utf-8',newline='\n')
