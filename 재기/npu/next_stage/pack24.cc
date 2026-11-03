#include <stdint.h>
extern "C" void geul_tile(int8_t*,int8_t*,int32_t*);
extern "C" void geul_tile_packed(int8_t *a,int8_t *b,int32_t *out) {
    alignas(32) int32_t values[128];
    geul_tile(a,b,values);
    uint32_t overflow=0;
    for (int i=0;i<128;i++)
        overflow |= (values[i] < -8388608 || values[i] > 8388607);
    for (int i=0;i<8;i++) out[i]=0;
    out[0]=overflow;
    for (int i=0;i<32;i++) {
        uint32_t a0=(uint32_t)values[4*i]&0xffffff;
        uint32_t a1=(uint32_t)values[4*i+1]&0xffffff;
        uint32_t a2=(uint32_t)values[4*i+2]&0xffffff;
        uint32_t a3=(uint32_t)values[4*i+3]&0xffffff;
        out[8+3*i]=(int32_t)(a0|(a1<<24));
        out[9+3*i]=(int32_t)((a1>>8)|(a2<<16));
        out[10+3*i]=(int32_t)((a2>>16)|(a3<<8));
    }
}
