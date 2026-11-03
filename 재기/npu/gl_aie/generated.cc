#define NOCPP
#include <stdint.h>
#include <aie_api/aie.hpp>
static inline int32_t dot_loaded(aie::vector<int8_t,32> x, aie::vector<int8_t,32> w) {
    auto p = aie::mul(x,w);
    return aie::reduce_add(p.to_vector<int32_t>(0));
}
extern "C" void geul_tile(int8_t *a, int8_t *b, int32_t *dst) {
    for (int32_t r=0; (r < 4); r=(r + 1)) {
        int32_t ai = (r * 32);
        int32_t ah = ((r + 4) * 32);
        auto vec0 = aie::load_v<32>(a + ai);
        auto vec1 = aie::load_v<32>(a + ah);
        for (int32_t n=0; (n < 32); n=(n + 1)) {
            int32_t bi = (n * 32);
            auto vec2 = aie::load_v<32>(b + bi);
            int32_t s0 = dot_loaded(vec0, vec2);
            int32_t s1 = dot_loaded(vec1, vec2);
            dst[((r * 32) + n)] = (s0 + (256 * s1));
        }
    }
}
