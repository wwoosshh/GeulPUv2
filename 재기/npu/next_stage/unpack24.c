#include <stdint.h>
#include <string.h>
#include <tmmintrin.h>
__declspec(dllexport) void unpack24(const int32_t *src,int32_t *dst,int tiles) {
    const __m128i mask=_mm_setr_epi8(0,1,2,-128,3,4,5,-128,6,7,8,-128,9,10,11,-128);
    for(int t=0;t<tiles;t++) {
        const int32_t *s=src+t*104;
        int32_t *d=dst+t*128;
        const char *bytes=(const char*)(s+8);
        for(int i=0;i<32;i++) {
            int32_t high;
            memcpy(&high,bytes+12*i+8,4);
            __m128i v=_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(bytes+12*i)),_mm_cvtsi32_si128(high));
            v=_mm_shuffle_epi8(v,mask);
            v=_mm_srai_epi32(_mm_slli_epi32(v,8),8);
            _mm_storeu_si128((__m128i*)(d+i*4),v);
        }
        if(s[0]>=0 && s[0]<=7) {
            for(int j=0;j<s[0];j++) {
                uint32_t entry=(uint32_t)s[j+1];
                unsigned index=(entry>>8)&127;
                d[index]=(int32_t)(((uint32_t)d[index]&0xffffff)|((entry&255)<<24));
            }
        }
    }
}
