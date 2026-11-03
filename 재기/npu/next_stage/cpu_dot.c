#include <stdint.h>
#include <immintrin.h>
/* Exact signed INT8 dot32: no saturating int16 intermediate. */
static int32_t dot32(const int8_t *a,const int8_t *b) {
    __m256i sum=_mm256_setzero_si256();
    for(int j=0;j<32;j+=16) {
        __m256i x=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a+j)));
        __m256i y=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b+j)));
        sum=_mm256_add_epi32(sum,_mm256_madd_epi16(x,y));
    }
    int32_t v[8]; _mm256_storeu_si256((__m256i*)v,sum);
    return v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[6]+v[7];
}
__declspec(dllexport) void geul_cpu_dot(const int8_t *a,const int8_t *q,int32_t *s,int m,int n,int kb,int digits) {
    for(int r=0;r<m;r++) for(int b=0;b<kb;b++) for(int c=0;c<n;c++) {
        const int8_t *w=q+((int64_t)c*kb+b)*32;
        int32_t lo=dot32(a+((int64_t)r*kb+b)*32,w);
        int32_t hi=digits==2?dot32(a+((int64_t)(m+r)*kb+b)*32,w):0;
        s[((int64_t)r*kb+b)*n+c]=lo+256*hi;
    }
}
