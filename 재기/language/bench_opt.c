#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__declspec(noinline) int64_t process(const int64_t *a, int64_t *b, int64_t n, int mode) {
if(mode==3) { int64_t s0=0,s1=0,s2=0,s3=0,i=0; for(;i+3<n;i+=4) {
s0 += a[i+0];
s1 += a[i+1];
s2 += a[i+2];
s3 += a[i+3];
} for(;i<n;++i) s0+=a[i]; return ((s0+s1)+s2)+s3; }
if(mode==2) { for(int64_t t=0;t<n;t+=1024) for(int64_t c=0;c<32;++c) for(int64_t r=0;r<32;++r) b[t+c*32+r]=a[t+r*32+c]; return 0; }
if(mode==0) { int64_t i=0; for(;i+3<n;i+=4) {
b[i+0]=a[i+0];
b[i+1]=a[i+1];
b[i+2]=a[i+2];
b[i+3]=a[i+3];
} for(;i<n;++i) b[i]=a[i]; return 0; }
if(mode==1) { int64_t i=0; for(;i+3<n;i+=4) {
b[i+0]=(a[i+0]*3+7)&65535;
b[i+1]=(a[i+1]*3+7)&65535;
b[i+2]=(a[i+2]*3+7)&65535;
b[i+3]=(a[i+3]*3+7)&65535;
} for(;i<n;++i) b[i]=(a[i]*3+7)&65535; return 0; }
return 0; }
double now_ms(void) {
    LARGE_INTEGER value,freq;
    QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&value);
    return (double)value.QuadPart*1000.0/(double)freq.QuadPart;
}
int main(void) {
    const int64_t n=1048576;
    int64_t *a=malloc(n*8),*b=malloc(n*8);
    if (!a || !b) return 2;
    for (int64_t i=0;i<n;++i) {a[i]=i&65535; b[i]=0;}
    for (int mode=0;mode<4;++mode) for (int rep=0;rep<8;++rep) {
        double start=now_ms();
        int64_t total=process(a,b,n,mode);
        double elapsed=now_ms()-start;
        int64_t errors=0;
        if (mode==3) { if(total!=34359214080LL) errors=1; }
        else for (int64_t i=0;i<n;++i) {
            int64_t expected=i&65535;
            if(mode==1) expected=(expected*3+7)&65535;
            if(mode==2) expected=((i/1024)*1024+(i%32)*32+(i%1024)/32)&65535;
            if(b[i]!=expected) ++errors;
            total+=b[i];
        }
        printf("%d %d %.9f %lld %lld\n",mode,rep,elapsed,(long long)total,(long long)errors);
        if(errors) return 1;
    }
    free(a); free(b); return 0;
}
