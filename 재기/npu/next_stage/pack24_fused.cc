#define NOCPP
#include <stdint.h>
#include <aie_api/aie.hpp>
// Same INT8 dot and digit combination as the GL-lowered tile.
static inline int32_t dot_loaded(aie::vector<int8_t,32> x,aie::vector<int8_t,32> w) {
    auto p=aie::mul(x,w);
    return aie::reduce_add(p.to_vector<int32_t>(0));
}
extern "C" void geul_tile_packed(int8_t *a,int8_t *b,int32_t *out) {
    for(int i=0;i<8;i++) out[i]=0;
    int count=0;
    for(int r=0;r<4;r++) {
        auto low=aie::load_v<32>(a+r*32);
        auto high=aie::load_v<32>(a+(r+4)*32);
        for(int n=0;n<32;n+=4) {
            uint32_t v[4];
            for(int j=0;j<4;j++) {
                auto weight=aie::load_v<32>(b+(n+j)*32);
                int32_t value=dot_loaded(low,weight)+256*dot_loaded(high,weight);
                if(value < -8388608 || value > 8388607) {
                    if(count<7) out[1+count]=((r*32+n+j)<<8)|((uint32_t)value>>24);
                    count++;
                }
                v[j]=(uint32_t)value&0xffffff;
            }
            int offset=8+(r*8+n/4)*3;
            out[offset]=(int32_t)(v[0]|(v[1]<<24));
            out[offset+1]=(int32_t)((v[1]>>8)|(v[2]<<16));
            out[offset+2]=(int32_t)((v[2]>>16)|(v[3]<<8));
        }
    }
    out[0]=count<=7 ? count : -1;
}
