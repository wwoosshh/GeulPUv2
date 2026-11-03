#include <windows.h>
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
s0 += (int32_t)a[ai+t+0] * (int32_t)b[bi+t+0];
s1 += (int32_t)a[ai+t+1] * (int32_t)b[bi+t+1];
s2 += (int32_t)a[ai+t+2] * (int32_t)b[bi+t+2];
s3 += (int32_t)a[ai+t+3] * (int32_t)b[bi+t+3];
} dst[(r*blocks+block)*n+col]=((s0+s1)+s2)+s3;
} } }
double now_ms(void) { LARGE_INTEGER f,v; QueryPerformanceFrequency(&f);QueryPerformanceCounter(&v);return (double)v.QuadPart*1000.0/f.QuadPart; }
int load(const char *path,void *ptr,size_t bytes) { FILE *f=fopen(path,"rb");if(!f)return 1;size_t got=fread(ptr,1,bytes,f);fclose(f);return got!=bytes; }
int main(void) {
{
int8_t *a=malloc(2048),*b=malloc(1048576); int32_t *dst=malloc(262144),*expected=malloc(262144);
if(!a||!b||!dst||!expected)return 2;
if(load("case0.a.bin",a,2048)||load("case0.b.bin",b,1048576)||load("case0.expected.bin",expected,262144))return 2;
for(int rep=0;rep<5;++rep) { double start=now_ms(); matrix(a,b,dst,2,1024,1024);double elapsed=now_ms()-start;int64_t errors=0;
for(int64_t i=0;i<65536;++i)if(dst[i]!=expected[i])++errors;
printf("0 %d %.9f %lld\n",rep,elapsed,(long long)errors);if(errors)return 3;
}
FILE *f=fopen("case0.out.bin","wb");if(!f)return 4;size_t wrote=fwrite(dst,4,65536,f);fclose(f);if(wrote!=65536)return 5;
free(a);free(b);free(dst);free(expected);
}
{
int8_t *a=malloc(65536),*b=malloc(262144); int32_t *dst=malloc(2097152),*expected=malloc(2097152);
if(!a||!b||!dst||!expected)return 2;
if(load("case1.a.bin",a,65536)||load("case1.b.bin",b,262144)||load("case1.expected.bin",expected,2097152))return 2;
for(int rep=0;rep<5;++rep) { double start=now_ms(); matrix(a,b,dst,64,256,1024);double elapsed=now_ms()-start;int64_t errors=0;
for(int64_t i=0;i<524288;++i)if(dst[i]!=expected[i])++errors;
printf("1 %d %.9f %lld\n",rep,elapsed,(long long)errors);if(errors)return 3;
}
FILE *f=fopen("case1.out.bin","wb");if(!f)return 4;size_t wrote=fwrite(dst,4,524288,f);fclose(f);if(wrote!=524288)return 5;
free(a);free(b);free(dst);free(expected);
}
{
int8_t *a=malloc(98304),*b=malloc(3145728); int32_t *dst=malloc(12582912),*expected=malloc(12582912);
if(!a||!b||!dst||!expected)return 2;
if(load("case2.a.bin",a,98304)||load("case2.b.bin",b,3145728)||load("case2.expected.bin",expected,12582912))return 2;
for(int rep=0;rep<5;++rep) { double start=now_ms(); matrix(a,b,dst,32,1024,3072);double elapsed=now_ms()-start;int64_t errors=0;
for(int64_t i=0;i<3145728;++i)if(dst[i]!=expected[i])++errors;
printf("2 %d %.9f %lld\n",rep,elapsed,(long long)errors);if(errors)return 3;
}
FILE *f=fopen("case2.out.bin","wb");if(!f)return 4;size_t wrote=fwrite(dst,4,3145728,f);fclose(f);if(wrote!=3145728)return 5;
free(a);free(b);free(dst);free(expected);
}
return 0; }
