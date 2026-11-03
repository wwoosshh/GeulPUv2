#include <stdint.h>
#include <math.h>
#include <string.h>
#include <xmmintrin.h>
#define API __declspec(dllexport)
static uint32_t bits(float x) { uint32_t v; memcpy(&v,&x,4); return v; }
static float value(uint32_t v) { float x; memcpy(&x,&v,4); return x; }
static float flush(float x) { uint32_t v=bits(x); return (v&0x7fffffff)<0x00800000 ? value(v&0x80000000):x; }
static float adjacent(float x,int up) {
    uint32_t v=bits(x);
    if ((v&0x7fffffff)==0) return value(up?1:0x80000001);
    return value(v+(((v>>31)!=0)^up ? 1 : -1));
}
/* Match the reference's exact FP64 product + TwoSum correction. */
static float rule(float aa,float bb,float cc,int ftz) {
    if(ftz) { aa=flush(aa); bb=flush(bb); cc=flush(cc); }
    double a=aa,b=bb,c=cc,p=a*b,r=p+c;
    double z=r-p,err=(p-(r-z))+(c-z);
    float f=(float)r;
    if(isfinite(r) && isfinite(f)) {
        double fd=f;
        float lo=fd<=r?f:adjacent(f,0),hi=fd<=r?adjacent(f,1):f;
        double mid=((double)lo+(double)hi)/2.0;
        if(r==mid && err!=0) f=err>0?hi:lo;
    }
    if(ftz) {
        const double edge=0x1p-126-0x1p-151;
        double ar=fabs(r);
        if(ar<edge || (ar==edge && (r>0?err<0:err>0)))
            return value(signbit(r)?0x80000000:0);
    }
    return f;
}
API void geul_fma(const float *a,const float *b,const float *c,float *out,int n,int ftz) {
    unsigned saved=_mm_getcsr(); _mm_setcsr(saved & ~0xe040u);
    for(int i=0;i<n;i++) out[i]=rule(a[i],b[i],c[i],ftz);
    _mm_setcsr(saved);
}
API void geul_finish(const int32_t *s,const int64_t *e,const float *d,float *out,
                     int m,int n,int kb,int stride,int precision,int ftz) {
    unsigned saved=_mm_getcsr(); _mm_setcsr(saved & ~0xe040u);
    for(int r=0;r<m;r++) for(int col=0;col<n;col++) {
        float y=0;
        for(int b=0;b<kb;b++) {
            float u=ldexpf((float)s[((int64_t)r*kb+b)*stride+col],(int)e[r*kb+b]-precision);
            y=rule(u,d[col*kb+b],y,ftz);
        }
        out[r*n+col]=y;
    }
    _mm_setcsr(saved);
}
