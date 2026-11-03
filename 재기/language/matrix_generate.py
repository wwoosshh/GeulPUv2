"""Matched INT8 dot32 matrix kernels, real GeulPU digit/scaling contract inputs."""
import json,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'계약'))
import 정수계약 as C
out=root/'빌드/matrix'
out.mkdir(parents=True,exist_ok=True)
shapes=[(1,1024,1024),(32,256,1024),(16,1024,3072)]
rng=np.random.default_rng(2901)
for case,(m,n,k) in enumerate(shapes):
    x=rng.uniform(-2,2,(m,k)).astype(np.float32)
    q=rng.integers(-128,128,(n,k),dtype=np.int8)
    d=rng.uniform(-.1,.1,(n,k//32)).astype(np.float16).astype(np.float32)
    bias=rng.uniform(-.1,.1,n).astype(np.float32)
    digits,e=C.자리로(x,2)
    a=digits.astype(np.int8).reshape(2*m,k)
    b=np.ascontiguousarray(q.reshape(n,k//32,32).transpose(1,0,2))
    expected=np.einsum('rbk,bnk->rbn',a.astype(np.int32).reshape(2*m,k//32,32),b.astype(np.int32),optimize=True)
    for name,arr in [('a',a),('b',b),('expected',expected)]: arr.tofile(out/f'case{case}.{name}.bin')
    np.savez(out/f'case{case}.npz',x=x,q=q,d=d,bias=bias,e=e,reference=C.행렬곱블라스(x,q,d,bias,자리수=2))
(out/'shapes.json').write_text(json.dumps(shapes))
gl='''포함 "표준/시간.gl"
외부 "ucrtbase.dll" "fopen" [문자열 경로를 문자열 모드로 열기]는 -> 공허 참조.
외부 "ucrtbase.dll" "fread" [공허 참조 버퍼에 정수 크기로 정수 개수를 공허 참조 파일에서 읽기]는 -> 정수.
외부 "ucrtbase.dll" "fwrite" [공허 참조 버퍼에서 정수 크기로 정수 개수를 공허 참조 파일에 기록]는 -> 정수.
외부 "ucrtbase.dll" "fclose" [공허 참조 파일을 닫기]는 -> 중간정수.
[문자열 경로를 공허 참조 자료에 정수 바이트로 적재]는 -> 정수 {
공허 참조 f = 열기(경로 를, "rb" 로).
f == 없음이면 { 반환 1. }
정수 count = 읽기(자료 에, 1 로, 바이트 를, f 에서).
닫기(f). count != 바이트이면 { 반환 2. } 반환 0.
}
[작은정수 참조 a에서 작은정수 참조 b로 중간정수 참조 dst에 정수 rows를 정수 n와 정수 k 행렬]는 {
정수 blocks = k / 32.
반복 (정수 r = 0: r < rows: r = r + 1) {
반복 (정수 block = 0: block < blocks: block = block + 1) {
정수 ai = r * k + block * 32.
반복 (정수 col = 0: col < n: col = col + 1) {
정수 bi = (block * n + col) * 32.
중간정수 s0 = 0. 중간정수 s1 = 0. 중간정수 s2 = 0. 중간정수 s3 = 0.
반복 (정수 t = 0: t < 32: t = t + 4) {
'''
c='''#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
__declspec(noinline) void matrix(const int8_t *a,const int8_t *b,int32_t *dst,int64_t rows,int64_t n,int64_t k) {
int64_t blocks=k/32;
for(int64_t r=0;r<rows;++r) for(int64_t block=0;block<blocks;++block) {
int64_t ai=r*k+block*32;
for(int64_t col=0;col<n;++col) {
int64_t bi=(block*n+col)*32;
int32_t s0=0,s1=0,s2=0,s3=0;
for(int64_t t=0;t<32;t+=4) {
'''
for j in range(4):
    gl+=f's{j} = s{j} + (a[ai + t + {j}] 으로 중간정수) * (b[bi + t + {j}] 으로 중간정수).\n'
    c+=f's{j} += (int32_t)a[ai+t+{j}] * (int32_t)b[bi+t+{j}];\n'
gl+='''} dst[(r * blocks + block) * n + col] = ((s0 + s1) + s2) + s3.
} } } }
[시작하기]는 -> 정수 {
'''
c+='''} dst[(r*blocks+block)*n+col]=((s0+s1)+s2)+s3;
} } }
double now_ms(void) { LARGE_INTEGER f,v; QueryPerformanceFrequency(&f);QueryPerformanceCounter(&v);return (double)v.QuadPart*1000.0/f.QuadPart; }
int load(const char *path,void *ptr,size_t bytes) { FILE *f=fopen(path,"rb");if(!f)return 1;size_t got=fread(ptr,1,bytes,f);fclose(f);return got!=bytes; }
int main(void) {
'''
for case,(m,n,k) in enumerate(shapes):
    total=2*m*(k//32)*n
    gl+=f'''{{
작은정수 참조 a = 할당({2*m*k})으로 작은정수 참조.
작은정수 참조 b = 할당({n*k})으로 작은정수 참조.
중간정수 참조 dst = 할당({total*4})으로 중간정수 참조.
중간정수 참조 expected = 할당({total*4})으로 중간정수 참조.
(적재("case{case}.a.bin" 를, a 에, {2*m*k} 로) != 0 또는 적재("case{case}.b.bin" 를, b 에, {n*k} 로) != 0 또는 적재("case{case}.expected.bin" 를, expected 에, {total*4} 로) != 0)이면 {{ 반환 2. }}
반복 (정수 rep = 0: rep < 5: rep = rep + 1) {{
실수 start = 정밀시각_밀리초().
행렬(a 에서, b 로, dst 에, {2*m} 를, {n} 와, {k}).
실수 elapsed = 정밀시각_밀리초() - start.
정수 errors = 0.
반복 (정수 i = 0: i < {total}: i = i + 1) {{ dst[i] != expected[i]이면 {{ errors = errors + 1. }} }}
"{case} {{rep}} {{elapsed}} {{errors}}\\n"을 쓰다.
errors != 0이면 {{ 반환 3. }}
}}
공허 참조 f = 열기("case{case}.out.bin" 를, "wb" 로).
f == 없음이면 {{ 반환 4. }}
정수 wrote = 기록(dst 에서, 4 로, {total} 를, f 에).
닫기(f). wrote != {total}이면 {{ 반환 5. }}
해제(a). 해제(b). 해제(dst). 해제(expected).
}}
'''
    c+=f'''{{
int8_t *a=malloc({2*m*k}),*b=malloc({n*k}); int32_t *dst=malloc({total*4}),*expected=malloc({total*4});
if(!a||!b||!dst||!expected)return 2;
if(load("case{case}.a.bin",a,{2*m*k})||load("case{case}.b.bin",b,{n*k})||load("case{case}.expected.bin",expected,{total*4}))return 2;
for(int rep=0;rep<5;++rep) {{ double start=now_ms(); matrix(a,b,dst,{2*m},{n},{k});double elapsed=now_ms()-start;int64_t errors=0;
for(int64_t i=0;i<{total};++i)if(dst[i]!=expected[i])++errors;
printf("{case} %d %.9f %lld\\n",rep,elapsed,(long long)errors);if(errors)return 3;
}}
FILE *f=fopen("case{case}.out.bin","wb");if(!f)return 4;size_t wrote=fwrite(dst,4,{total},f);fclose(f);if(wrote!={total})return 5;
free(a);free(b);free(dst);free(expected);
}}
'''
gl+='반환 0. }\n'
c+='return 0; }\n'
here=Path(__file__).parent
(here/'matrix.gl').write_text(gl,encoding='utf-8')
(here/'matrix.c').write_text(c,encoding='utf-8')
print('generated',shapes,flush=True)
