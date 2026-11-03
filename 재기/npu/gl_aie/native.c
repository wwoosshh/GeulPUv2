#include <stdint.h>
static inline int32_t dot32(int8_t *a, int32_t ai, int8_t *b, int32_t bi) {
    int32_t sum = 0;
    for (int32_t k=0; k<32; ++k)
        sum += (int32_t)a[ai+k] * (int32_t)b[bi+k];
    return sum;
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
