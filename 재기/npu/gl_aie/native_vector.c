#include <stdint.h>
typedef int8_t i8x32 __attribute__((ext_vector_type(32)));
typedef int32_t i32x32 __attribute__((ext_vector_type(32)));
static inline int32_t dot32(int8_t *a, int32_t ai, int8_t *b, int32_t bi) {
    i8x32 x, w;
    __builtin_memcpy(&x,a+ai,32);
    __builtin_memcpy(&w,b+bi,32);
    i32x32 xx=__builtin_convertvector(x,i32x32);
    i32x32 ww=__builtin_convertvector(w,i32x32);
    return __builtin_reduce_add(xx*ww);
}
void geul_tile(int8_t *a, int8_t *b, int32_t *dst) {
    for (int32_t r=0; r<4; ++r) {
        int32_t ai=r*32, ah=(r+4)*32;
        for (int32_t n=0; n<32; ++n) {
            int32_t bi=n*32;
            int32_t s0=dot32(a,ai,b,bi);
            int32_t s1=dot32(a,ah,b,bi);
            dst[r*32+n]=s0+256*s1;
        }
    }
}
