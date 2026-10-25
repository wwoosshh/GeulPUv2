# The contract — what is lost, and only where

*English version of [`문서/계약.md`](계약.md). Terms follow the [glossary](../docs/glossary.md).*

Every computation in GeulPU writes its precision into the source. The GPU is left no discretion — not the rounding mode, not the
order of additions, not approximate instructions — so under the same contract the GPU result is **bit-identical** to the CPU
reference implementation ([`계약/정수계약.py`](정수계약.py)), and identical whichever way it is computed (all at once, in
pieces, one token at a time, as a conversation, batched, or in the megakernel). Loss happens only at the places written below.

## 0. The PTX contract — what the compiler promises

- float32 additions, subtractions and multiplications are emitted with an explicit rounding mode (`.rn`), so the driver cannot fuse a
  multiply and an add into an FMA. A fused multiply-add exists only where the source writes `곱해더하기` (fma).
- No approximate instructions (`ex2.approx` …). exp is a written polynomial ("written exp", below).
- **float32 rule R — subnormal flush** (stage 31 — [`문서/31단계.md`](https://github.com/wwoosshh/GeulPU/blob/main/문서/31단계.md)). Every float32 operation below follows this
  rule: multiplication, addition, subtraction, fma, division and square root — everywhere this document says "IEEE rounding".
  - A subnormal input (magnitude above 0 and below 2⁻¹²⁶) is read as a zero of the same sign.
  - The result is rounded to a 24-bit significand with an unbounded exponent (round to nearest, ties to even). If its magnitude is
    below 2⁻¹²⁶, it becomes a zero of the same sign.
  - fma does not round the product.

  A module's source declares the rule with `(* 비정규수버림 *)` ("flush subnormals"), and the PTX backend then adds `.ftz` to every
  float32 operation. It adds `.ftz` only when the source declares it, which keeps PTX contract P2. Three paths were checked to follow
  exactly this rule down to the boundary values:
  - NVIDIA's Vulkan driver, which cannot preserve subnormals;
  - Intel's Vulkan driver, in its default and flush modes;
  - PTX `.ftz`.

  That is what lets CUDA and Vulkan produce the same bits. Real model outputs are unchanged by the switch (checked in stage 31).
- A float → int conversion that overflows sets the module's error cell — there are no silent values. Tests require the error cell
  to stay 0.

## 1. The integer block contract — linear layers (two digits)

Weights are the model file's integers and block scales exactly as stored (Q8_0 · Q4_0 here; k-quants in 1.1). An activation row x
is split into blocks of 32 values, and for each block:

1. E = ⌊log₂ max_k |x_k|⌋, clamped to [−98, 100] (an all-zero block uses −98).
2. n_k = round-to-nearest-even(x_k · 2^(13 − E)), |n| ≤ 2¹⁴, written as two balanced base-256 digits: n = 256·D₁ + D₀ with
   D₀ ∈ [−128, 127], D₁ ∈ [−64, 64].
3. S_j = Σ_k D_j,k · q_k,n — an **exact** integer sum on integer tensor cores (prefill) or with dp4a (decode); exact sums do not depend on
   the order of addition. For Q4_0, q is the nibble − 8.
4. v_j = S_j · 2^(8j + E − 13) (exact), u = v₁ + v₀ (one rounding), and y ← fma(u, d_b,n, y) over blocks b in ascending order; the
   output is y + bias.

The activation grid is 2⁻¹⁴ of the block maximum — 128 times finer than llama.cpp's re-quantization of activations to 8 bits (2⁻⁷).
(A four-digit variant with a 2⁻²⁹ grid is also part of the contract and was used for GPT-2 in the early stages.)

### 1.1 k-quants (stage 26)

Weights are the file's integers and scales as stored. Activations are the same 32-value blocks with two digits as above.

- **Q4_K · Q5_K** (super-block of 256 = eight sub-blocks of 32; half-precision d and m; per sub-block 6-bit s_j and t_j unpacked as
  llama.cpp's `get_scale_min_k4`; q ∈ [0, 15] and [0, 31]): a_j = d · s_j and b_j = m · t_j (both exact in float32). For activation
  block j (= sub-block j): S_j = Σ_k n_k · q_k and T_j = Σ_k n_k (exact integers, |S| < 2²⁴), u_j = S_j · 2^(E − 13) and
  w_j = T_j · 2^(E − 13) (exact). One chain in ascending j: **y ← fma(u_j, a_j, y), y ← fma(−w_j, b_j, y)**; output = y + bias.
- **Q6_K** (super-block of 256 = sixteen sub-blocks of 16; half-precision d; signed 8-bit s_k; q ∈ [0, 63]): a_k = d · s_k (exact). For
  each sub-block k, S_k = Σ n · (q − 32) over its 16 values (exact, |S| ≤ 2²³), u = S_k · 2^(E − 13), and **y ← fma(u, a_k, y)** in
  ascending k.
- **IQ4_XS** (GeulPUv2 0.11.0, step 1-6c — super-block of 256 = eight sub-blocks of 32; half-precision d; a 6-bit ls_j per sub-block (a nibble
  of scales_l plus two bits of scales_h); a 4-bit index per value into the codebook c ∈ {−127, −104, −83, −65, −49, −35, −22, −10, 1, 13,
  25, 38, 53, 69, 89, 113} — unpacked as llama.cpp's dequantize_row_iq4_xs): a_j = d · (ls_j − 32) (exact). Because |c| ≤ 127 the
  formula is section 1's Q8_0 one with q = c and block scale a_j (S_j per digit, one rounding in u = v₁ + v₀ — a single-digit sum of
  n · c can exceed 2²⁴).
- **Embedding** from a k-quant token table: Q4_K · Q5_K give a_j · q − b_j (the product is exact; one rounding at the subtraction),
  Q6_K gives a_k · (q − 32) (exact), IQ4_XS gives a_j · c (exact).
- Each matrix may use its own format (Q4_K_M mixes Q4_K and Q6_K, for example); Q8_0 · Q4_0 models keep the formula of section 1.

How the kernels reach these values is free as long as the bits are the same. Since stage 27:

- Q4_0 and Q4_K tiles get u with a **magic-number accumulator**. Let M = 1.5 · 2²³ · 2^(E−13) (its bit pattern is already in the
  block info). The digit-1 integer MMA's sum s₁ gives C = 256·s₁ + bits(M) (Q4_K: bits(M) − 8T), which is the start value of the
  digit-0 integer MMA's accumulator. Read as a float, the result is M + S·2^(E−13) (Q4_K: M + (S − 8T)·2^(E−13)) — exact, because
  |S| ≤ 2²² (Q4_0) and |S − 8T| ≤ 2²² (Q4_K) keep it inside M's binade — and subtracting M (Q4_K: M − 8T·2^(E−13)) gives u
  exactly. The per-element int → float conversion and scale multiplication disappear. Q8_0 · Q5_K · Q6_K sums can exceed 2²², so
  they keep the plain path.
- Q6_K tiles use one m16n8k16 integer MMA per 16-value sub-block with B = 4(q − 32) and the scale a/4: S′ = 4S and
  fma(4u, a/4, y) = fma(u, a, y) exactly.
- (GeulPUv2 0.11.0, step 1-1 — CUDA modules) Two-digit Q4_K · Q5_K tiles no longer add the row sum T with integer MMAs: the kernel
  that produced the linear layer's input also writes T per block after the info words (the row-sum plane), and the tile
  reads it. T is an exact integer, so the value is the same. Two-digit Q8_0 tiles compute the epilogue as
  fma(float(T) · P, d, y): u = float(T) · P is exact (|u| ≥ 2⁻¹¹¹ when T ≠ 0), and the product inside the fma is the same
  real number as in the stage-31 form fma(float(T) · 2⁻²⁴, (P · 2²⁴) · d, y), so the bits are the same.

Neither changes a single output bit — the stage-27 tests compare against the same CPU reference.

### 1.2 Fast contract F1 — one-digit activations in linear layers (GeulPUv2 stage 2, [`문서/2단계.md`](../문서/2단계.md))

Fast-mode modules (`커널생성 … --빠른`) keep the activations of the linear layers — the tile, row and row-tile kernels, the logit kernel,
and the conversions that feed them (convert, RMS spread, SwiGLU and the merge's epilogue) — as **one digit**. Attention (RoPE's q, the
KV cache and the e inside attention) keeps the two digits of section 2.

1. E = ⌊log₂ max_k |x_k|⌋, and **if max · 2^(6 − E) < 127.5 then E ← E − 1** (on the float's bits: 23-bit mantissa < 0x7F0000). Then
   clamp E to [−98, 100].
2. n_k = round-to-nearest-even(x_k · 2^(5 − E)), |n| ≤ 127 — one digit (D₀ = n). The block maximum lands in 64 … 127.
3. S = Σ_k n_k · q_k,n (exact integer sum — |S| < 2²² in every format), u = S · 2^(E − 5) (exact); after that the formulas and order of
   sections 1 and 1.1 (for Q4_K · Q5_K, w = T · 2^(E − 5)).

- Domain: block maximum |x| < 2¹⁰¹ (above it |n| would exceed 127 — two digits have the same property above 2¹⁰¹, and real
  activations never get there).
- The step is (0.996, 1.992] times llama.cpp's block maximum / 127 (its 8-bit activations) — about 2.15 times the error variance. The
  exact mode's step is 1/128 – 1/256 of that.
- Determinism, bit equality with the CPU reference and path invariance stay (exact integer sums in a fixed order). The CPU reference
  is `정수계약.자리로 · 행렬곱블라스 · 행렬곱K · 행렬곱K블라스` with `자리수=1`.
- How the kernels get there: with one digit every format keeps |S| < 2²², so the magic-number accumulator above also serves Q8_0,
  Q5_K (centred by 8 like Q4_K — |S − 8T| ≤ 32·127·23) and Q6_K (per half — |4S| ≤ 16·127·128). The output bits are the same (the
  kernel tests compare them with the CPU reference).

### 1.3 16-bit block weights — F16 · BF16 · F32 (0.12.0 12-3b, [`문서/블록조합.md`](../문서/블록조합.md))

Unquantized weights (F16, BF16, F32) are converted at load time into the 16-bit format of the activations in section 1 (two 8-bit
digits). **This conversion is the only loss.**

1. Each weight row (one output) is split along the depth into 32-value blocks: E_w = ⌊log₂ max|w|⌋ (clamped to [−98, 100]; −98 when the
   maximum is 0), n_w = round-to-nearest-even(w · 2^(13 − E_w)), |n_w| ≤ 2¹⁴. A depth that is not a multiple of 32 is padded with zeros
   (zeros change neither the block maximum nor the sums). F16 values of at least 1/8 of the block maximum and BF16 values of at least
   1/64 of it are unchanged; only smaller values are rounded to the step 2^(E_w − 13).
2. Exact mode: for each block b, T = Σ_k n_x,k · n_w,k (activations n_x = 256·D₁ + D₀ — an exact integer sum, |T| ≤ 2³³),
   u = float32(T) (one rounding), part = u · 2^(E_x,b − 13) (a power-of-two product — exact, rule R), y ← fma(part, 2^(E_w,b − 13), y) in
   ascending b, output = y + bias.
3. Fast mode (section 1.2 — one-digit activations n_x = D₀): |T| < 2²⁶, part = u · 2^(E_x,b − 5); everything else is the same.
4. Embeddings (token → activation) use the file's values as they are (F16 and BF16 → float32 is exact). When the output weights are
   tied to the embeddings, the logit matmul uses the format of item 1.
5. Integer sums do not depend on order, so tensor-core matmuls (four digit pairs — the exact sum 2¹⁶ S₁₁ + 2⁸(S₁₀ + S₀₁) + S₀₀) and dp4a
   give the same bits. The CPU reference is `정수계약.가중치16 · 행렬곱16`.

## 2. The integer KV contract — attention

The q · k · v of the attention projection are converted with the same rule as section 1 (32-value blocks along the head dimension,
two digits and an exponent byte), and keys and values are cached in that form. For one query at position p, attention proceeds over
chunks of 64 keys:

1. Score s_t = 0.125 × Σ_b (u_b,t × 2^(E_q,b − 13)) × 2^(E_k,t,b − 13) (head size 64; head 128 below), where u_b,t is the exact integer sum over the four digit pairs
   (2¹⁶S₁₁ + 2⁸(S₁₀ + S₀₁) + S₀₀) rounded once to float32. Masked keys (t > p) get −3 × 10³⁸.
2. m = max s_t and e_t = written_exp(s_t − m). **Written exp**(x): n = round(x · log₂e), r = x − n·C₁ − n·C₂ (two fmas),
   p = 1 + r + r²·(c₀ + r·(c₁ + … + r·c₅)) (Cephes expf coefficients as an fma chain), result = p · 2ⁿ, and 0 for x < −87.3. Every
   operation is an IEEE float32 operation, so the CPU produces the same bits.
3. l = float32(Σ_t round(e_t · 2²³)) × 2⁻²³ — summed exactly as integers, then rounded once.
4. Output o_d: per 32-key block j, e′_t = e_t × 2^(E_v − 13) is written as two digits, u_j,d = float32(Σ_t n_e,t · n_v,t,d) (exact sum,
   one rounding), and o_d = u₀,d × 2^(E_e,0 − 13) + u₁,d × 2^(E_e,1 − 13) (one rounding).
5. Key tiles are merged as in section 3. The output O / L goes straight into two digits for the output projection.

Because integer sums do not depend on their order, prefill (tensor-core MMA) and decode (dp4a) produce the same bits — this follows
directly from the contract.

### 2.1 Fast KV contract F1A′ — 8-bit KV (GeulPUv2 ③b-2, [`문서/긴문맥.md`](../문서/긴문맥.md))

With the fast mode's 8-bit KV cache setting on, keys and values are cached as 8-bit integers with a float32 scale, and attention
multiplies those integers directly. The 16-bit format (two 8-bit digits) of q and e′, the max m, e_t, the sum l and the two-level merge
(section 3) are as in section 2.

1. Storage: per block of 32 values along the head dimension, s = max|x| / 127 (float32 division, one rounding). If s > 0, n is the
   integer nearest to the float32 quotient x / s (one rounding; ties to even) and |n| ≤ 127; if s = 0, n = 0. The cache holds n (int8)
   and s (float32).
2. Score: per block b, u_b,t = float32(Σ n_q · n_k) — the exact integer sum of the 32 products of q's 16-bit-format integer
   n_q = 256·D₁ + D₀ and the key's n (|sum| < 2²⁶), converted to float32 (one rounding). With a_b,t = u_b,t × 2^(E_q,b − 13) (a power-of-two
   product), z ← a_0,t × s_k,t,0, then z ← fma(a_b,t, s_k,t,b, z) for b = 1, 2, … in order, and s_t = z × c (c = 0.125 for head size 64,
   the c of section 4 for head size 128). Masked keys (t > p) get −3 × 10³⁸.
3. Output o_d (d in value block b): e′_t = e_t × s_v,t,b (one rounding) is converted per 32-key block j to the 16-bit format (integer
   n_e,t, exponent E_e,j) as in section 2 step 4; u_j,d = float32(Σ_t n_e,t · n_v,t,d) (exact integer sum, |sum| < 2²⁶, one rounding),
   and o_d = u₀,d × 2^(E_e,0 − 13) + u₁,d × 2^(E_e,1 − 13) (one rounding).

All float32 operations follow rule R. The only difference from section 2 is that the float32 scale s takes the place of the power-of-two
factor 2^(E − 13) of keys and values; the order of operations is the same. Because keys and values have one digit, the tensor-core MMAs
drop from four digit pairs to two. With power-of-two scales and values of ordinary size, the result is bit-identical to the ③b-1
emulation (the dequantized value n · s written in the 16-bit format). The two differ only in tiny values: an all-zero block has s = 0, so
its contribution is exactly 0, while the 16-bit format keeps it as a value with exponent −98, and the 16-bit format's exponent does not go
below −98. [`시험/계약자체.py`](../시험/계약자체.py) checks the storage, the match with the emulation, and a computation that follows this
section's text row by row.

### 2.2 Fast attention contract F2 (GeulPUv2, [`문서/빠른어텐션.md`](../문서/빠른어텐션.md))

The fast mode's KV cache setting "8-bit · fast attention" (kernel generator `--빠른 --F2` — head size 128). Keys are as in 2.1 (int8 and a
float32 scale per 32 values); what differs is the scale of q and of the values, the unit of the scores and of the exponential, and the
probabilities of the weighted sum.

1. **q**: the 16-bit format (two 8-bit digits) with one exponent E_q per head (128 values) — E_q = ⌊log₂ max_d |q_d|⌋ (clamped to
   [−98, 100]; −98 if the maximum is 0), n_q,d = nearest integer of q_d · 2^(13 − E_q).
2. **Values**: int8 with one float32 scale per head (128 values) — s_v = max_d |v_d| / 127 (float32 division, one rounding); if s_v > 0,
   n is the integer nearest to the float32 quotient v / s_v (ties to even), else 0. The cache's scale slots [position][4] hold the same
   value four times (the layout of 2.1).
3. **Score** (log₂ units): per block b, u_b,t = float32(Σ n_q · n_k) (the exact integer sum of 2.1, one rounding); z ← u_0,t × s_k,t,0,
   then z ← fma(u_b,t, s_k,t,b, z) for b = 1, 2, 3 in order, and s_t = z × q_c with q_c = 2^(E_q − 13) × c′ (exact),
   c′ = float32(float32(1/√128) × float32(log₂e)) = 0x3E0293EE. Masked keys (t > p) get −3 × 10³⁸.
4. **exp2** (x ≤ 0): x_c = max(x, −126), t = x_c + 1.5 · 2²³ (rounded to the nearest integer), r = x_c − (t − 1.5 · 2²³) (exact — |r| ≤ ½),
   p = the fma chain (((((c₅ · r + c₄) · r + c₃) · r + c₂) · r + c₁) · r + 1) with c_k = float32((ln 2)^k / k!) (c₁ … c₅ = 0x3F317218 ·
   0x3E75FDF0 · 0x3D635847 · 0x3C1D955B · 0x3AAEC3FF), and the result's bits = p's bits + (t's bits ≪ 23) (mod 2³² — adds n to the
   exponent).
5. **Tile** (64 keys): m = max_t s_t, e_t = exp2(s_t − m) (0 for masked keys), l as in section 2 step 3 (integer sum).
   e′_t = e_t × s_v,t (one rounding). Per row, from the tile's largest e′: E = ⌊log₂ max_t e′_t⌋ (the two-digit rule — [−98, 100], −98 if
   0), n_t = nearest integer of e′_t · 2^(13 − E) ∈ [0, 2¹⁴], U_d = float32(Σ_t n_t · n_v,t,d) (the exact integer sum over the tile's 64
   keys, one rounding), P = 2^(E − 13).
6. **Merge** (the two levels of section 3 — the exponential is exp2): the group's first tile gives (M, L, O) = (m, l, fma(U, P, +0)); for
   each following tile M′ = max(M, m), α = exp2(M − M′), β = exp2(m − M′), L ← fma(l, β, L · α), O ← fma(U, float32(P · β), O · α), M ← M′.
   Groups merge with section 3's formulas using exp2. The output is O / L.

All float32 operations follow rule R; determinism, bit identity with the CPU reference and path invariance hold as before. The operations
differ from 2.1, so the output bits differ. The four value blocks share one set of e′ digits (2.1 makes one per block), the weighted sum
converts integers to float32 once per 64 keys (2.1: per 32) and the exponential is shorter — prompt attention runs about a third fewer
instructions. The CPU reference is `어텐션F2` in [`계약/정수계약.py`](정수계약.py) (`F2 = True`).

**F2B** (0.11.0, step 1-4 — fast attention for the qwen3 architecture, kernel generator `--빠른 --F2B`): only q in steps 1 and 3 differs
from F2 — q keeps one exponent per block of 32 values (E_q,b) as in 2.1 (the two digits of 1.2), and the score uses, per block b,
w_b,t = u_b,t × 2^(E_q,b − 13) (exact), z ← w_0,t × s_k,t,0, then z ← fma(w_b,t, s_k,t,b, z), and s_t = z × c′ (c′ = 0x3E0293EE as above).
Values, exp2, the tile and the merge are those of F2. After Qwen3's per-head normalization (4.6) the blocks of q differ in size, and with
one exponent per head (Q1E) the top-1 agreement of quality bar B″ fell short ([`문서/빠른어텐션.md`](../문서/빠른어텐션.md), step 1-4).
The CPU reference sets `F2 = True` and `F2B = True`.

## 3. Two-level merge (stage 13)

Key tiles (64 positions each) are grouped (16 tiles = 1024 positions; group g covers positions [1024g, 1024g + 1024)):

1. Within a group, tiles are merged in order into a group state (M_g, L_g, O_g) (the group's first tile is the initial value).
2. Groups are merged in order g = 0, 1, … with the same formula (the first group is the initial value).

The merge (both levels): M′ = max(M, m), α = written_exp(M − M′), β = written_exp(m − M′), L ← fma(l, β, L·α), O ← fma(o, β, O·α).
Group boundaries depend only on positions, so every way of computing gives the same bits. Up to position 1024 there is one group and
this equals the one-level merge of stages 9–12. Keeping only group states in global memory lets one (row, head) be split across blocks —
the reason short prompt chunks at long contexts became fast. Accuracy is unchanged (median error against fp64, new / old: 1.0003× and 0.9994×).

## 4. Llama-family blocks (stage 14 — Qwen2)

1. **RMSNorm** (one row of width D): lane t (0 … 255) sums p_t = Σ_{c ≡ t (mod 256)} x_c · x_c (ascending c, rounding at every product
   and sum); the 256 partial sums are reduced in halves from 128 down to S; r = 1 / sqrt(S / D + eps) (IEEE round-to-nearest);
   y_c = (x_c · r) · w_c. When the next operation is a linear layer, y goes straight into two digits.
   eps is a module constant and must equal the GGUF's `layer_norm_rms_epsilon` — for Qwen2 either float32(10⁻⁶) = 0x358637BD (modules
   qwen2… — Qwen2.5, R1-Distill-Qwen-7B) or float32(10⁻⁵) = 0x3727C5AC (modules qwen2e5… — R1-Distill-Qwen-14B, `커널생성 qwen2e5큰`);
   other values are refused (#16).
2. **RoPE** (NeoX, rotate-half): the cos · sin table [position][32] holds correctly rounded float32 values (θ_i = θ^(−2i/64) correctly
   rounded in fp64, angle = position · θ_i as an fp64 product, cos and sin correctly rounded to float32 — independent of the platform's
   math library; computed with Python's standard `decimal`). x′_i = fma(x_i, cos, −(x_{i+32} · sin)), x′_{i+32} = fma(x_{i+32}, cos,
   x_i · sin), applied to q and k after the bias. Then q goes to two digits and k · v to the integer KV cache.
3. **SwiGLU**: e = written_exp(0 − |g|); s = g / (1 + e) if g ≥ 0, else (g · e) / (1 + e); h = s · u (IEEE rounding at each operation).
   h goes into two digits for the down projection.
4. **GQA**: query head h reads the cache of KV head ⌊h / (query heads / KV heads)⌋ — the attention contract (sections 2 · 3) is unchanged.
5. Linear layers without a bias (o · gate · up · down · logits) add a bias of 0. The embedding is the token table's q · d (exact); there
   is no position embedding.
6. **Per-head Q · K normalization** (stage 37 — Qwen3): after the q · k · v linear layers (models without biases add 0, as in 5) and
   before RoPE, every q head and every k head (one head of one row = 128 values x_0 … x_127) gets the RMSNorm of 1 with D = 128. Each
   lane t < 128 holds one value and the stride-128 add is +0, so it reads: p_t = x_t · x_t, reduced in halves with strides h = 64, 32, …, 1 down to S
   (p_t ← p_t + p_{t+h}); r = 1 / sqrt(S / 128 + eps); y_t = (x_t · r) · w_t. The weights w are the layer's `attn_q_norm` ·
   `attn_k_norm` (128 float32 values shared by all heads); eps is the model's (it must equal the generator's constant). v is not
   normalized. RoPE (head size 128, item 2 below) and the integer KV cache follow unchanged. The kernel gives lane l the values
   d = l + 32·i (i = 0 … 3), so the stride-64 step is two adds (p_l + p_{l+64}, p_{l+32} + p_{l+96}), the stride-32 step adds those two, and xor
   shuffles 16 · 8 · 4 · 2 · 1 finish the reduction — bit-identical to the order above (checked over 4,096 heads in the contract self-test).
7. **The llama architecture** (stage 39 — Llama 3.1 · 3.2 · Mistral v0.3): the block is items 1 – 5 unchanged (no biases — 0 is
   added), head size 128 (below). Three things differ:
   1. **q · k row restore**: the GGUF converter (the permute in llama.cpp's `convert_hf_to_gguf.py`) stores the original row
      h · 64 + j of each head (128 rows; h ∈ {0, 1}, j < 64) at file row 2j + h. When the weights are loaded, new row j (< 64) of each
      head takes file row 2j and new row 64 + j takes file row 2j + 1 (q per query head, k per KV head — the quantized rows are moved
      as they are; every row is quantized on its own, so the values are unchanged; biases, if any, move with them). RoPE then uses
      item 2 of head size 128 below (pairs (i, i + 64)) unchanged, and the KV cache holds the channels in the original (HF) order.
   2. **RoPE scaling factors**: if the GGUF has `rope_freqs.weight` (64 positive float32 values f_i), θ′_i = θ_i / f_i (one fp64
      division, correctly rounded), angle = position · θ′_i (fp64 product), cos and sin correctly rounded to float32 (decimal).
      Without it, θ′_i = θ_i. A `rope.scaling.type` other than none is refused.
   3. **eps**: RMSNorm eps = 10⁻⁵ (the module constant float32(10⁻⁵) = 0x3727C5AC — it must equal the GGUF's value). The formula
      is item 1 unchanged. GeulPUv2 0.12.0, step 12-1b — a llama model with eps 10⁻⁶ (MiniCPM5) uses float32(10⁻⁶) = 0x358637BD (modules
      llamae6… — `커널생성 llamae6큰`; the marker kernel eps6표시 at the end of the module lets the engine check it against the file).

8. **YaRN position scaling** (GeulPUv2 ③a — Qwen2.5 7B and larger · Qwen3, [문서/긴문맥.md](../문서/긴문맥.md), Korean): on only
   when the context (max length) exceeds the GGUF's position count (`context_length` — Qwen2.5 32,768 · Qwen3 40,960); the factor s
   is the smallest of 2 · 4 that covers the context (original length L = 32,768, context limit 131,072). Only for architecture qwen2
   (width 3,584 or more) · qwen3 (a GGUF whose `rope.scaling.type` is present and not none is refused for every architecture — #17). When on, only the RoPE table changes
   (kernels unchanged):
   1. **Bounds**: c(β) = hd · ln(L / (β · 2π)) / (2 · ln θ), computed in double-double and correctly rounded to fp64 — low = ⌊c(32)⌋,
      high = ⌈c(1)⌉, clamped to [0, hd − 1] (if equal, high += 0.001). Within 10⁻⁶ of an integer it stops. (θ = 10⁶ · hd = 128 ·
      L = 32,768 give low 23 · high 40 — pairs 24 … 39 are mixed.)
   2. **Frequencies**: r_i = (i − low) / (high − low) (one fp64 division) clamped to [0, 1], g_i = r_i / s + (1 − r_i) (fp64 division,
      subtraction and addition — in this order), θ′_i = θ_i · g_i (one fp64 product — θ_i as in item 2 of head size 128). Pairs with
      r_i of 0 or 1 get g_i = 1 or 1/s exactly.
   3. **Magnitude**: m = 0.1 · ln s + 1 (ln s correctly rounded to fp64, one fp64 product and one fp64 sum — 1.0693147180559945 and
      1.138629436111989 for s = 2 and 4). The cos and sin tables hold float32(m · cos(position · θ′_i)) and float32(m · sin(...)) — the
      angle is one fp64 product, and the exact product m · cos is correctly rounded once (q and k are both scaled by m — the attention
      temperature). This is the formula of transformers' `_compute_yarn_parameters` (Qwen's config `rope_scaling: {type: yarn, factor,
      original_max_position_embeddings}`) and llama.cpp's rope_yarn, pinned down in fp64.
   4. A server with YaRN on puts `YaRN s×L` into the prompt cache's chunk names (the KV bytes differ). Within the position count the
      table and the KV are byte-for-byte unchanged.
9. **Output gate** (0.12.0 12-3e — the full-attention layers of Qwen3.5, [문서/블록조합.md](../문서/블록조합.md), Korean): in the output
   of the q linear (section 1), the first h entries of each head are q and the next h are the gate. q and k get the per-head norm of
   item 6 (D = h — one element per strand, so halving starts at a stride of h/2), RoPE is item 2 of "Head size and RoPE dimensions"
   below (Qwen3.5: the first 64 dimensions, θ = 10⁷), and the score scale is c = fp32(1/√h) (1/16 for h = 256 — exact). The attention
   output o_d = O_d / L (one division) is multiplied by σ(gate_d) (one product) — σ(x) = 1 / (1 + e) if x ≥ 0, else e / (1 + e), with
   e = written-exp(0 − |x|) (IEEE rounding per operation). Then the o linear's input (two digits — one digit in the fast contract F1,
   1.2). The 8-bit KV (2.1) and fast attention F2 (2.2) are not used in these layers.
10. **Linear attention GDN** (0.12.0 12-3e — the other layers of Qwen3.5 — gated delta net):
   1. **Projections**: `attn_qkv` → [q (H_k · 128) | k (H_k · 128) | v (H_v · 128)], `attn_gate` → z (H_v · 128), `ssm_beta` → b (H_v),
      `ssm_alpha` → a (H_v) — the integer block contract of section 1 (each matrix in its own format).
   2. **Convolution** (per channel, kernel 4 — weight entry 0 is the oldest token): y = w₀ · x_(t−3) (product),
      y ← fma(w₁, x_(t−2), y), y ← fma(w₂, x_(t−1), y), y ← fma(w₃, x_t, y). Rows before t are the sequence's convolution state (its last
      three rows — zero at the start). Then SiLU — the s of item 3 (e = written-exp(0 − |y|), y / (1 + e) if y ≥ 0, else (y · e) / (1 + e)).
   3. **L2 normalisation** (per 128-wide head of q and k): S by the strands and halving of item 1 (p_t = x_t · x_t, from a stride of 64),
      r = 1 / max(sqrt(S), eps), y = x · r (one rounding each — eps is the model's RMSNorm eps). q is then multiplied by
      c_k = fp32(1/√128) = 0x3DB504F3 (one product).
   4. **Gate values** (per v head): β = σ(b) (the σ of item 9), u = a + dt (`ssm_dt.bias` — one addition), sp = u + L if u > 0, else L
      (L = log1p(written-exp(0 − |u|)) — softplus), g = sp · A (`ssm_a` — one product), γ = written-exp(g). log1p(x) for 0 ≤ x ≤ 1 is fdlibm's
      `s_log1pf.c` reduced to that range: u = 1 + x, c = x − (u − 1), k = 1 and m = u · 0.5 if u ≥ fp32(√2), else k = 0 and m = u,
      f = m − 1, s = f / (2 + f), z = s · s, hfsq = (0.5 · f) · f, t = fma(Lp₇, z, Lp₆) → … → fma(t, z, Lp₁), R = z · t,
      corr = (k ? ln2_lo : 0) + c / u, log1p = (k ? ln2_hi : 0) − ((hfsq − (s · (hfsq + R) + corr)) − f) — IEEE rounding per operation.
      Constants (fp32): Lp₁ … Lp₇ = 0x3F2AAAAB · 0x3ECCCCCD · 0x3E924925 · 0x3E638E29 · 0x3E3A3325 · 0x3E1CD04F · 0x3E178897,
      ln2_hi = 0x3F317180, ln2_lo = 0x3717F7D1, fp32(√2) = 0x3FB504F3 (largest error 0.85 ulp over 2.5 million points of [0, 1]).
   5. **Recurrence** (v head j uses the q and k of k head j mod H_k — the whole head list repeated; tokens in order; state S[i][j] in
      fp32, 128 × 128, zero at the start): S_ij ← γ · S_ij (product); kv_j = for each of four row blocks (i ∈ [32b, 32b + 32))
      P_b = S_(32b)j · k_(32b) (product), then P_b ← fma(S_ij, k_i, P_b) for ascending i, kv_j = (P₀ + P₁) + (P₂ + P₃);
      δ_j = (v_j − kv_j) · β (one subtraction, one product); S_ij ← fma(k_i, δ_j, S_ij); o_j = Σ_i S_ij · q_i in the same order as kv_j.
      Every execution path (prompt, generation, table step) uses this order for every token.
   6. **Output**: o gets the RMSNorm of item 6 per head (D = 128, `ssm_norm` — the same weights for every head), times SiLU(z) (one
      product — SiLU as in step 2), then the input of `ssm_out` (two digits — one in F1).
   A layer is x ← x + mix(RMSNorm(x; `attn_norm`)), x ← x + SwiGLU(RMSNorm(x; `post_attention_norm`)) — mix is item 9 in the layers with
   (l + 1) % `full_attention_interval` = 0 and item 10 in the others.
11. **MoE FFN** (0.12.0 12-3g — Qwen3.5 / 3.6 MoE, architecture `qwen35moe`, [`문서/블록조합.md`](../문서/블록조합.md)): in the FFN slot of a
   layer, with h = RMSNorm(x; `post_attention_norm`) (item 1 — two digits when it feeds a linear layer):
   1. **Router**: l_e = `ffn_gate_inp`_e · h (experts e = 0 … E − 1), l_s = `ffn_gate_inp_shexp` · h — the integer block contract of
      section 1 (the file's F32 weights converted to the 16-bit blocks of 1.3 — the router and the shared gate can be one matrix with E + 1
      columns: each column is computed on its own, so the values are the same). 12-3g-1 compared it with an fp32 fma chain over the F32
      weights as they are: the block contract is closer to the math reference (exact integer sums per block — 64 roundings instead of
      2,048; the 12-3g-1 results in `문서/0.12.0.md`).
   2. **Softmax**: m = max_e l_e, e_i = written-exp(l_i − m), strand t (0 … 31) sums p_t = Σ_{i ≡ t (mod 32), ascending i} e_i (starting from
      0), then halving from 16 (p_t ← p_t + p_{t+16}, …) gives S — the same shape as the fold of item 1; p_i = e_i / S (one division).
   3. **Top k** (k = `expert_used_count`): descending l, equal values (−0 = +0) by the smaller index first — rank r = 0 … k − 1 picks
      expert e_r.
   4. **Weights**: s = p_(e_0) + … + p_(e_(k−1)) (additions in rank order), w_r = p_(e_r) / max(s, 2⁻¹⁴) (one division — 2⁻¹⁴ is llama.cpp's
      floor 6.103515625e-5).
   5. **Experts**: y_r = down_(e_r)(SwiGLU(gate_(e_r) · h, up_(e_r) · h)) — the integer block contract of section 1 (each expert matrix in its
      own format) and the SwiGLU of item 3 (two digits into down). A row's y_r depends only on that row and that expert's weights — the
      same bits however rows are grouped per expert.
   6. **Sum**: o = w_0 · y_0 (one product), then o ← fma(w_r, y_r, o) for r = 1 … k − 1 in order.
   7. **Shared expert**: y_s = down_s(SwiGLU(gate_s · h, up_s · h)) (items 1 and 3), g = σ(l_s) (the σ of item 9), o ← fma(g, y_s, o).
   8. **Residual**: x ← x + o (one addition).
   This is llama.cpp's `build_moe_ffn` (softmax → top k → weights divided by their sum → sum of expert outputs × weights, shared expert ×
   sigmoid gate) with its order fixed. The layers are items 9 and 10 as before; only the FFN is this one.
12. **Gemma 4** (0.12.0 12-3h — architecture `gemma4`, dense: 12B · 31B, [`문서/블록조합.md`](../문서/블록조합.md)): sliding-window layers
   (`attention.sliding_window_pattern` true) and full-attention layers have different shapes — head size `key_length_swa` · `key_length`
   (256 · 512 in the 12B), KV heads `attention.head_count_kv`[l] (8 · 1), RoPE θ `rope.freq_base_swa` · `rope.freq_base`. The other blocks are
   items 1 – 5 as they are (no biases — 0 is added).
   1. **Input**: x = embedding (the vocabulary's values as they are — a · q − b for k-quants, stage 26) · c_e (one product),
      c_e = sqrt(fp32(width)) (IEEE correctly rounded — 0x4277DEF6 for width 3,840).
   2. **Attention**: h = RMSNorm(x; `attn_norm`) (item 1), q = `attn_q` · h, k₀ = `attn_k` · h, v₀ = `attn_v` · h — layers without `attn_v` (the
      12B's full layers) use v₀ = k₀. q and k₀ get item 6's per-head normalization (D = head size — with item 1's 256 strands a 512 head
      gives each strand two elements), weights `attn_q_norm` · `attn_k_norm`. v₀ gets the same per-head normalization with **weight 1**
      (y = x · r — multiplying by 1 changes nothing). RoPE covers the whole head (n_rot = head size — item 2 of "Head size and RoPE
      dimensions", pairs (i, i + h/2)); full layers divide by `rope_freqs.weight` (h/2 entries): θ′_i = θ_i / f_i (the rule of 7.2 — in the
      Gemma 4 12B only the first 64 pairs have 1, the rest 10³⁰: the angle is tiny, cos is 1 and sin a tiny fp32 number). The score scale
      is **c = 1** (the one product of sections 2 and 3 is exact — no 1/√h). GQA is item 4. Sliding-window layers also mask keys with
      (row position − key position) ≥ W (`attention.sliding_window` — 1,024 in the 12B) on top of the causal mask: per row, tiles (64 keys)
      and groups (16 tiles) with no key the row sees are skipped, and the first tile and first group with a key the row sees are its
      initial values (the merge of section 3 — without a window these are always the first tile and group, so the earlier contract is
      unchanged). The output goes into `attn_output` (two digits).
   3. **After attention**: x ← x + RMSNorm(o; `post_attention_norm`) (item 1 — one addition).
   4. **GeGLU**: h₂ = RMSNorm(x; `ffn_norm`), g = `ffn_gate` · h₂, u = `ffn_up` · h₂, s = g · g, t = s · g, y = fma(c_b, t, c_a · g),
      h = (g · σ(y)) · u — σ as in item 9, c_a = fp32(2 √(2/π)) = 0x3FCC422A, c_b = fp32(2 √(2/π) · 0.044715) = 0x3D922279 (IEEE rounding at each
      operation). This writes the tanh approximation of GELU, ½ g (1 + tanh(√(2/π)(g + 0.044715 g³))), as g · σ(2 √(2/π)(g + 0.044715 g³))
      (the same function). h goes into `ffn_down` (two digits), x ← x + RMSNorm(`ffn_down` · h; `post_ffw_norm`).
   5. **Layer scale**: if `layer_output_scale.weight` (one fp32 s_l per layer) exists, x ← x · s_l (one product per element).
   6. **End**: RMSNorm(x; `output_norm`), logits z = output (`output.weight` — `token_embd.weight` without it) · x (section 1), softcap
      (when `final_logit_softcapping` = c is present): y = z · fp32(2/c), result = fma(2c, σ(y), −c) — c · tanh(z/c) = 2c · σ(2z/c) − c (2c and −c
      must be exact in fp32 — c = 30 in the 12B, fp32(2/30) = 0x3D888889).
   The layers are x ← x + RMSNorm(attention(RMSNorm(x; `attn_norm`)); `post_attention_norm`), x ← x + RMSNorm(GeGLU(RMSNorm(x; `ffn_norm`));
   `post_ffw_norm`), x ← x · s_l. This is llama.cpp's `src/models/gemma4.cpp` (master 7894e8b) with its order fixed (PLE, KV sharing and
   MoE — E2B · E4B · 26B-A4B — are not covered yet).

### Head size 128 (stage 22 — Qwen2.5-7B and larger)

The head-64 contract is unchanged. Three things differ at head size 128:

1. **Score scale**: the score sum chains the four 32-value blocks in ascending b with fmas (block 0 is the initial value — the same
   form as the two blocks of head 64), then multiplies once: s ← s × c with c = float32(1/√128) = 0x3DB504F3 =
   0.088388346135616302490234375 (round-to-nearest). Head 64's 0.125 is a power of two and can be applied anywhere; c is not, so its
   place is fixed.
2. **RoPE**: the table is [position][64], θ_i = θ^(−2i/128) (i = 0 … 63, correctly rounded in fp64), rotation pairs (i, i + 64). The
   fma forms are those of head 64.
3. **Output o_d** uses the formula of section 2.4 for each of the four 32-value blocks of values (d = 0 … 127). Written exp, the sum l
   and the two-level merge are unchanged. A group or tile state is (M, L, two spare, O[128]) = 132 values.

### Head size and RoPE dimensions (0.12.0 12-3b — the general path, [`문서/블록조합.md`](../문서/블록조합.md))

1. **Head size h** (32 ≤ h ≤ 256): sections 2 and 3 run on the head padded with zeros to h′ = ⌈h/32⌉ · 32, and the padded outputs are
   dropped — zeros change neither the block maxima nor the integer sums, so the bits are the same. The score scale is c = float32(1/√h)
   of the real head size (the same place as item 1 of head size 128; exact when h is a power of 4). The engine (0.12.0 12-3c-2) pads
   heads of h ≤ 128 with zeros all the way to 128 and runs the head-128 kernels — the blocks beyond h′ add zero to the score chain and the
   weighted sum (a zero of either sign is the same input to the exponential), so the bits are the same.
2. **RoPE dimensions** n_rot (GGUF `rope.dimension_count` — h when absent): only the first n_rot dimensions rotate — table cos · sin
   [position][n_rot/2], θ_i = θ^(−2i/n_rot) (correctly rounded), pairs (i, i + n_rot/2), the fma forms of item 2 of head size 128. The
   other dimensions pass through.
3. **Width, query width and FFN** that are not multiples of 32 pad the activation blocks with zeros (the same values). RMSNorm divides
   by the real width (section 4.1).

## 5. Sampling (stage 28 — the server)

Non-greedy generation picks the next token from one row of logits z (the V float32 values the engine produced) by this contract. All
arithmetic is float64, rounded once per operation in the written order, with no fused multiply-add. The chosen token depends only on
(that row of logits, the parameters, the seed, the step). The logits do not depend on the batch, the slot or how the prompt was fed
(sections 1–4), so sampling does not either. The geul-language version is [`호스트/뽑기.gl`](../엔진/뽑기.gl); the Python version is
[`시험/뽑기.py`](../시험/뽑기.py).

Inputs: temperature T (≥ 0), top_k K (integer ≥ 0, 0 = off), top_p P (0 < P ≤ 1), min_p Q (0 ≤ Q < 1), seed s (u64), step i (0 for the
first generated token of a request).

1. If T = 0 or K = 1, pick the smallest index whose z is the maximum (the same as the device argmax).
2. Order the indices by (z descending, then index ascending — −0 and +0 are equal): j₁, j₂, ….
3. a_k = (f64(z_{j_k}) − f64(z_{j₁})) / T. The candidates are the prefix with a_k ≥ −40 (length n_F). This is a declared loss:
   probabilities below e⁻⁴⁰ ≈ 4 × 10⁻¹⁸ of the largest are set to 0.
4. n_K = min(K, n_F) if K > 0, else n_F.
5. w_k = exp_w(a_k) for k ≤ n_K (w₁ = 1), and c_k = c_{k−1} + w_k (c₀ = 0, in order).
6. top-p: if P < 1, n_P is the smallest k with c_k ≥ P × c_{n_K}; else n_P = n_K.
7. min-p: n_Q is the number of k ≤ n_K with w_k ≥ Q (n_K if Q = 0).
8. n = min(n_P, n_Q) (≥ 1).
9. u = (splitmix64(s, i) ≫ 11) × 2⁻⁵³ and t = u × c_n. Pick j_k for the smallest k ≤ n with c_k > t, or j_n if none (rounding).

- **splitmix64(s, i)**: start from x = s + (i + 1) · 0x9E3779B97F4A7C15 (mod 2⁶⁴), then x ← (x ⊕ (x ≫ 30)) · 0xBF58476D1CE4E5B9,
  x ← (x ⊕ (x ≫ 27)) · 0x94D049BB133111EB, x ← x ⊕ (x ≫ 31), with logical shifts. This is the (i + 1)-th output of SplitMix64 seeded with s.
- **exp_w(a)**, a ∈ [−40, 0]:
  - k = ⌊a × 0x3FF71547652B82FE + 0.5⌋
  - r = (a − k × 0x3FE62E42FEE00000) − k × 0x3DEA39EF35793C76 (float64 bit patterns: 1/ln 2, and fdlibm's high and low parts of ln 2)
  - p ← 1/13!, then p ← p × r + 1/n! for n = 12 … 0, where 1/n! = f64(1) / f64(n!)
  - the value is p × 2^k (2^k built from its bits, so the multiply is exact)

Defaults when a request omits a field are those of llama-server: T 0.8, K 40, P 0.95, Q 0.05. Without a seed, the server draws one and
returns it in `geulpu.seed`, so the response can be reproduced. The order differs from llama.cpp, which applies top-k, top-p and min-p
to the temperature-1 distribution and the temperature last. This contract applies the temperature first (the Hugging Face order).

### 5.2 Log-probabilities (stage 38 — the server, [`문서/38단계.md`](https://github.com/wwoosshh/GeulPU/blob/main/문서/38단계.md))

Log-probabilities (`logprobs`, `top_logprobs` and the scores of `echo`) are computed from one row of logits z (the V float32 values the
engine produced) by this contract. All arithmetic is float64, rounded once per operation in the written order, with no fused
multiply-add. The distribution is **the model's** (temperature 1, no filtering) — not the sampling distribution of section 5. The
geul-language version is [`호스트/로그확률.gl`](../엔진/로그확률.gl); the Python version is [`계약/로그확률.py`](로그확률.py).
Because the logits do not depend on the batch, the slot or the path (generation step or prompt path), neither do the log-probabilities:
the values received while generating are bit-identical to the values from scoring the same text with `echo`.

1. j₁ = the smallest index whose z is the maximum, m = f64(z_{j₁}). a_j = f64(z_j) − m.
2. w_j = exp_w(a_j) (section 5's exp) for a_j ≥ −40, else 0. This is a declared loss (terms below e⁻⁴⁰ of the largest leave the sum).
3. Split the vocabulary into 16 ranges: B = ⌈V / 16⌉, range r = [rB, min(V, (r + 1)B)). In each range, starting from 0.0 and in index
   order, S_r ← S_r + w_j and T_r ← T_r + w_j · a_j (the product is rounded once). S = (…((S₀ + S₁) + S₂) + …) + S₁₅, and T likewise.
   The ranges exist so the work can be split across worker threads — the order inside and across ranges is fixed, so the result does
   not depend on the number of threads.
4. L = log_w(S), **logprob_j = a_j − L**, entropy **H = L − T / S**.
5. Top k (≤ 20): the first k indices in (z descending, then index ascending — −0 and +0 are equal) order, with their logprobs.

- **log_w(x)** follows fdlibm's `e_log.c` step by step (S is a normal number between 1 and V): the exponent k and the mantissa are taken
  from the bits, the mantissa is brought into [√2/2, √2) and f = xn − 1; |f| < 2⁻²⁰ has its own short path; otherwise s = f / (2 + f),
  z = s·s, w = z·z, t1 = w·(Lg2 + w·(Lg4 + w·Lg6)), t2 = z·(Lg1 + w·(Lg3 + w·(Lg5 + w·Lg7))), R = t2 + t1, and the two fdlibm forms
  combine f, s, R and k·ln2 (high and low parts). Constants as float64 bits: Lg1 0x3FE5555555555593, Lg2 0x3FD999999997FA04,
  Lg3 0x3FD2492494229359, Lg4 0x3FCC71C51D8E78AF, Lg5 0x3FC7466496CB03DE, Lg6 0x3FC39A09D078C69F, Lg7 0x3FC2F112DF3E5244,
  1/3 = 0x3FD5555555555555, ln 2 high and low as in section 5. The contract self-test (quick check) keeps it within 1 ulp of the
  correctly rounded ln (6,021 values — at most 1 ulp).

Responses: chat uses the OpenAI form `logprobs.content` (token, logprob, bytes, top_logprobs); completions use `logprobs` (tokens,
token_logprobs, top_logprobs, text_offset — with echo the prompt tokens come first, null before `geulpu.echo_from`). The `geulpu`
object carries the ids and values of the top k (`top_ids`, `top_logprobs` — the completions dictionary form merges tokens with the
same text), the entropy (`entropy`, `prompt_entropy`), the prompt tokens (`prompt_tokens`) and the logit hashes of the echo rows
(`prompt_logit_hashes`). Numbers are written with 17 significant digits, so the float64 values come back unchanged.

- **Logit hash** (`logit_hashes`, `prompt_logit_hashes`): for each 32-bit value w of a logit row (V float32 values),
  h ← (h ^ w) × 1099511628211 (mod 2⁶⁴), starting from h = 0xcbf29ce484222325 (FNV-1a 64). Written as 16 hex digits.
- **Fingerprints and reproduction files** (research panel — a record format, not part of the arithmetic contract): token fingerprint
  = SHA-256(`geulpu-fp/1\nmodel <model file SHA-256>\nprompt <ids,…>\ntokens <ids,…>\n`), logit fingerprint = SHA-256(`<token
  fingerprint>\n<per-step logit hashes,…>\n`) — the device and backend are not included. A capsule (`geulpu-capsule/1`) holds the request
  (prompt tokens, temperature, top-k, top-p, min-p, seed as the raw integer, `step_offset`, `max_tokens`), the result (tokens,
  log-probabilities, entropy, top k, logit hashes), the model (name, file, size, SHA-256) and where it was made (`made_on` — the device and
  backend are a record written by the maker; the computation mode `precision` · `kv8` · `fast_attention` and the YaRN factor `yarn` (#23 —
  section 4.8, 0 = none) choose the contract the check uses). [`도구/repro_check.py`](../도구/repro_check.py) checks it without a GPU, with this
  contract's CPU reference (sections 1–5): the same tokens, logit hashes, log-probabilities and fingerprints.

### 5.3 Layer lens — the logit lens (GeulPUv2 #96, [`문서/층렌즈.md`](../문서/층렌즈.md))

The layer lens reads the residual stream of one prompt row at every layer with the final logit computation (the logit lens — the
next-token distribution the model would give if it stopped at that layer). It is defined for the blocks of section 4 (qwen2, qwen3,
llama); architectures with other blocks do not accept it until their own section defines it.

1. x_l(r): the output of block l (0 … L − 1) for row r — the D float32 values after both residual additions (attention and FFN), that
   is, the input of block l + 1. It is the engine's residual row as is.
2. z⁽ˡ⁾(r) = the logit linear layer (section 1 — in the output weight's format, bias 0) of RMSNorm(x_l(r); `output_norm`) (section 4,
   item 1) — the same formula and the same module contract (exact or fast) as the final logits. So z⁽ᴸ⁻¹⁾(r) is bit-identical to row
   r's final logits.
3. For each layer, section 5.2 as is: the logprob of the target (the token r + 1 that row r scores), the entropy, the top k and the
   logit hash, and the target's rank (its position, from 1, in the order of 5.2 item 5 — z descending, then index ascending).

Responses: on a completions (`/v1/completions`) echo (`max_tokens` may be 0; `logprobs` is required), asking for `geulpu.layer_lens`
(`true` or `{"top_k": k}` — k from 0 to 20, by default the `logprobs` k) reads row r = `echo_from` − 1 (the row that scores token
`echo_from`; `echo_from` must be at least 1 and at most the prompt length − 1) at all L layers, and `geulpu.layer_lens` carries `row`,
`token`, `layers` and, per layer, `token_logprobs`, `entropy`, `top_ids`, `top_logprobs`, `logit_hashes` and `ranks` (layers 0 … L − 1
in order; numbers with 17 significant digits as in 5.2). A model that does not accept it (other blocks, or split across GPUs) gets 400.
[`도구/repro_check.py`](../도구/repro_check.py) `--layer-lens` computes the layer lens for steps of a capsule without a GPU, with this
section's CPU reference.

### 5.4 Jacobian lens (GeulPUv2 #96 phase 2, [`문서/층렌즈.md`](../문서/층렌즈.md))

The Jacobian lens moves the x_l of 5.3 into the final layer's basis with a D × D matrix J_l fitted outside, then reads it with the
formula of 5.3 (Anthropic 2026 — J_l is the mean, over the positions p of the prompts, of Σ_{p′ ≥ p} ∂x_{L−1}(p′)/∂x_l(p) in the original
model, fitted by jlens). J_l is not a value this contract computes but an input (the J file); this section fixes how to compute with
it. So the same J file, the same model file and the same module contract give the same bits on every GPU and in every run. The blocks are
those of 5.3.

1. **J file** (`geulpu-jlens/1`): a safetensors file — for each source layer l (the fitted ones among 0 … L − 2) a tensor `J.<l>` (F16,
   shape [D][D], row-major — row i is element i of u). In `__metadata__`, `format` is `geulpu-jlens/1`, `d_model` is D and `n_layers`
   is L — the file is refused when either differs from the model. The fit record (`fit` — tool, commit, original model, prompts,
   parameters) and the checked GGUF files (`checked` — file SHA-256 and agreement) are JSON strings and take no part in the computation.
2. u_l(r) = J_l x_l(r): for each element i, a double s ← +0, then for j = 0, 1, …, D − 1 in order s ← s + J_l[i][j] · x_l(r)_j. The
   halves of J are taken as they are (subnormals too); x is read like an input of rule R in section 0 (a subnormal reads as a zero of the
   same sign — the engine's x_l has none). The product of a half and a float is exact in double; each addition rounds to the nearest
   double (ties to even) — no fused multiply-add. u_l(r)_i = float(s) — like a result of rule R, one round to nearest at 24 significand
   bits (with no exponent limit), and a magnitude below 2⁻¹²⁶ becomes a zero of the same sign. Elements i are independent, so splitting
   them over threads gives the same bits.
3. z_J⁽ˡ⁾(r) = item 2 of 5.3 applied to u_l(r) instead of x_l(r) — the logit linear layer of RMSNorm(u_l(r); `output_norm`), under the
   same module contract (exact or fast). The last layer L − 1 and layers missing from the J file are not multiplied by J (z⁽ˡ⁾ of 5.3 as
   is — the last layer is bit-identical to the final logits).
4. For each layer, section 5.2 as is (the same as item 3 of 5.3).

Responses: asking for `geulpu.layer_lens` as `{"jacobian": true}` (it can go with `top_k`) in the request of 5.3 reads the same row r
with this section as well when the server was started with `--jlens <J file>`, and `geulpu.jacobian_lens` carries the same shape as
`layer_lens` (`row`, `token`, `layers` and the per-layer values) plus `jlens_sha256` (the J file's SHA-256) and `source_layers` (the
number of source layers). A server started without a J file answers 400. The engine downloads row r's x_l at every layer (it splits the
plan per layer — the kernels are unchanged), computes item 2 on the host, and reads x_l and u_l in one go (2L rows) with the two kernels
of the final logits — rows are independent, so the `layer_lens` of this response is bit-identical to a response that did not ask for J.
The app passes `<name>.jlens.safetensors` next to the model file to the server.

[`도구/jlens_fit.py`](../도구/jlens_fit.py) fits a J file with jlens on the original (Hugging Face) model.
[`도구/repro_check.py`](../도구/repro_check.py) `--jlens <J file>` computes the Jacobian lens with this section's CPU reference for the
steps of `--layer-lens`.

## 6. What the tests check ([`시험/`](../시험))

| Test | What |
|---|---|
| Matmul | every linear kernel of the module (tensor-core tiles · dp4a row kernels) against the CPU reference — real layer weights, random and extreme activations |
| Attention kernels | KV cache bytes, prompt-attention output digits and info, decode O/L against the CPU reference |
| Paths | the same token sequence all at once · one at a time · prompt + one at a time · two pieces · megakernel · three sentences batched · 64 tokens at a time · repeated — logits compared bit for bit |
| Cache bytes | whether the KV cache filled by those paths equals the one filled one token at a time |
| Conversation | chunks of different lengths fed and generated in turns equal one-shot processing, down to tokens and cache |
| Long context | the new kernels against the CPU reference (two heads), against each other (all heads); megakernel vs separate kernels; split vs one-shot feeding |
| 7B — head 128 ([`시험/머리128.py`](../시험/머리128.py) · [`시험/줄128.py`](../시험/줄128.py) · [`시험/참조7B.py`](../시험/참조7B.py) · [`엔진/시험_7B.gl`](../엔진/시험_7B.gl)) | RoPEKV · group kernels · group merge · tile kernels · tile merge (2100 random rows · real values of 7B layer 0, five paths) against the CPU reference; row kernels (five weights × rows 1 … 4 × three epilogues); all rows of logits for three chat prompts + 16 generated tokens against the CPU reference's full forward pass; the five paths (decode graph · all at once · in four · one row at a time · again) — logits and KV cache |
| Qwen2 ([`시험/qwen2.py`](https://github.com/wwoosshh/GeulPU/blob/main/시험/qwen2.py)) | every linear kernel (row-major · column-major · row · batched-row) · RMSNorm · RoPE+KV · SwiGLU conversions · GQA attention (five kernels + the decode kernel) against the CPU reference; paths (all at once · one at a time · two pieces · split feeding · small chunks · conversation) — logits and cache; all rows of logits for three chat prompts, and the last row beyond position 1024, against the CPU reference's full forward pass; repeatability of the terminal runner; GPT-2 module source unchanged |
| Qwen2 decode megakernel ([`시험/메가15.py`](https://github.com/wwoosshh/GeulPU/blob/main/시험/메가15.py)) | megakernel vs the stage-14 decode graph — logits and tokens at each of 64 steps for three chat prompts and contexts 1024 · 2100 · 8192, then the KV cache bytes; three turns of the runner's conversation (stopping inside the kernel at the end token) — tokens and cache; GPT-2 module source unchanged |
| k-quants (stage 26 — [`시험/판26.py`](../시험/판26.py)) | every k-quant kernel (243 cases: 11 tile shapes × epilogues × formats, row kernels, embeddings) on real weights of three GGUF files; full forward passes of Q4_K_M · Q5_K_M · Q6_K against the CPU reference |
| Fast contract F1 (GeulPUv2 stage 2 — [`시험/판23.py`](../시험/판23.py) · [`시험/판26.py`](../시험/판26.py) · [`시험/참조7B.py`](../시험/참조7B.py) · [`시험/품질.py`](../시험/품질.py)) | every linear kernel of the `--빠른` modules (Q8_0 · Q4_0 282 cases, Q4_K_M · Q5_K_M 81 cases — real 7B layer-0 weights, random and extreme activations) against the CPU reference with one digit; the five paths and every row's logits of three chat prompts against the CPU reference's full forward pass; the hidden loss relative to the exact mode (KL · top-1 agreement) compared with llama.cpp (criterion B′) |
| 8-bit KV F1A′ (GeulPUv2 ③b-2 — [`시험/계약자체.py`](../시험/계약자체.py) · [`시험/머리128.py`](../시험/머리128.py) `--KV8`) | the CPU reference itself (storage against exact rational arithmetic, the match with the ③b-1 emulation, a row-by-row transcription of section 2.1); RoPEKV, prompt attention, decode attention (tile kernel and its four-warp form) and the merges of the `--빠른 --KV8` module — 2100 random rows, extreme values, real values of 7B layer 0, long context up to 32,768, every path — KV cache (int8 and float32 scales) and outputs against the CPU reference; the engine's five paths and every row's logits of three chat prompts against the CPU reference's full forward pass (`시험/참조7B.py --KV8`), the paths at long context up to 131,072, and the 8-bit KV prompt-cache chunks (`시험/저장소CPU.py` · `엔진/시험_저장소`) |
| Qwen3 (stage 37 — [`시험/머리정규화37.py`](https://github.com/wwoosshh/GeulPU/blob/main/시험/머리정규화37.py) · [`시험/참조7B.py`](../시험/참조7B.py) · [`시험/참조생성.py`](https://github.com/wwoosshh/GeulPU/blob/main/시험/참조생성.py)) | the head-norm kernel (rows 1 · 4 · 64 · 2048 × the weights of every layer of 8B, random and extreme values) against the CPU reference — v untouched; every row's logits of three chat prompts + 16 generated tokens for 1.7B · 8B (Q8_0 · Q4_K_M) · two synthetic 14B-shape models against the CPU reference's full forward pass; five-path, Vulkan and shared32-emulation summaries equal to CUDA; greedy generation with the CPU reference (not a criterion — compared with llama.cpp) |
| Query width ≠ hidden size (GeulPUv2 0.11.0, step 1-6a — [`시험/머리128.py`](../시험/머리128.py) · [`시험/참조7B.py`](../시험/참조7B.py) · [`엔진/시험_7B.gl`](../엔진/시험_7B.gl) · [`엔진/시험_추측.gl`](../엔진/시험_추측.gl) · [`엔진/시험_긴문맥.gl`](../엔진/시험_긴문맥.gl)) | Qwen3-0.6B (hidden 1,024, query width 2,048 — q, the attention output and the input of the output projection are wider than the hidden size), its Q8_0 file and a local Q4_K_M requantization: the RoPEKV, group, merge and generation kernel variants (2,100 random rows, 600 extreme rows, three sets of real layer-0 values, long contexts up to 32,768 — exact, 8-bit KV and F2) against the CPU reference; every row's logits of three chat prompts + 16 generated tokens for the exact, fast, 8-bit KV and F2 modules (Q8_0) and the exact and 8-bit KV modules (k-quant) against the CPU reference's full forward pass, and the five paths; table steps of 2–8 rows of one slot against one row at a time; the four long-context paths up to position 32,760 |
| IQ4_XS (GeulPUv2 0.11.0, step 1-6c — [`시험/계약자체.py`](../시험/계약자체.py) · [`시험/판26.py`](../시험/판26.py) · [`시험/참조7B.py`](../시험/참조7B.py) · [`엔진/시험_7B.gl`](../엔진/시험_7B.gl)) | the CPU reference itself (values unpacked one by one from raw blocks — the formula of llama.cpp's dequantize_row_iq4_xs — and slow against BLAS); every IQ4_XS kernel variant of the KI module (row kernel, 11 tile variants, 3 row-tile variants, three epilogues, two embeddings, logits) with the real weights of two IQ4_XS files (Qwen3-0.6B requantized — one mixing in Q5_K · Q6_K, one with an IQ4_XS token table and output) and random and extreme activations, exact and fast modules, against the CPU reference; every row's logits of a whole model (three chat prompts + 16 generated tokens) against the CPU reference and the five paths, table steps of 2–8 rows, long context |
| General path — 0.12.0 12-3b ([`시험/계약자체.py`](../시험/계약자체.py) · [`시험/참조7B.py`](../시험/참조7B.py)) | the 16-bit block weight conversion against exact rational arithmetic, `행렬곱16` against the contract's sentences transcribed in rational arithmetic (exact and fast, depth padded with zeros), bit equality of attention with a zero-padded head; the CPU reference (F16 · BF16 · F32, head 64, width 896) against llama.cpp's next-token distribution |
| 16-bit block weight kernels — 0.12.0 12-3c-1 ([`시험/판16.py`](../시험/판16.py)) | every kernel variant of the `--W16` modules (exact and fast) — 11 tile kernels × two epilogues, the one-row kernel × three epilogues, `SwiGLU정수로`, the F16 · BF16 · F32 embeddings and their decode-graph variants (subnormal values included) — with random weights (sizes varying by row and block, zero blocks, values 2⁻²⁰ of the block maximum), random and extreme activations and extreme weights (blocks whose exponent is clipped at −98), up to the edges of 100 rows, 1,000 columns and depth 1,100 (zero-padded), against the CPU reference |
| MoE — 0.12.0 12-3g-1 ([`시험/계약자체.py`](../시험/계약자체.py) · [`시험/참조35.py`](../시험/참조35.py) · [`시험/qwen35수학.py`](../시험/qwen35수학.py)) | top-k ties (−0 and +0 too), the softmax (fold over 32 strands), weights, sum and the router's fma chain against scalar computations copied from the contract text, one row's result independent of the other rows (the same bits however rows are picked and grouped); the CPU reference (Qwen3.6-35B-A3B Q4_K_M) against the math reference (fp64) and llama.cpp — next-token distributions and chosen experts |
| MoE kernels — 0.12.0 12-3g-2 ([`시험/판MoE.py`](../시험/판MoE.py)) | the `--MoE Q8_0 · K` modules: `MoE라우팅` (ties, −0 and +0, experts spanning several tiles, empty experts), `MoE배치`, `MoE모으기`, every grouped tile kernel (11 variants × epilogues "" and SwiGLU정수 × Q8_0, Q4_K, Q5_K, Q6_K — 64 random experts; rows outside an expert's range stay untouched) and `MoE합`, then the MoE FFN chained through the GPU outputs, against the CPU reference |
| Gemma 4 — 0.12.0 12-3h-1 ([`시험/계약자체.py`](../시험/계약자체.py) · [`시험/참조gemma4.py`](../시험/참조gemma4.py) · [`시험/gemma4수학.py`](../시험/gemma4수학.py)) | the bits of the constants c_a, c_b and fp32(2/30); GeGLU and the softcap against scalar computations copied from the contract text, and their error bounds against the exact values; the sliding window (a window ≥ the length gives the same bits as attention without a window, keys outside the window cannot change the result, every row equals a hand fold of its tiles and groups); the CPU reference (gemma-4-12b-it Q4_K_M) against the math reference (fp64) and llama.cpp — next-token distributions on chat answers and on text |
| Log-probabilities (stage 38 — [`시험/로그확률38.py`](../시험/로그확률38.py) · [`시험/로그확률서버38.py`](../시험/로그확률서버38.py)) | the geul version (many rows at once and one at a time) against the Python reference — target logprob, entropy and top 20 of synthetic and real logit rows; the reference computed from the logit rows the server recorded (generation steps and echo rows) against the response values, and the rows' FNV hashes; the same tokens scored with echo (cache on and off) against the values from generation; concurrent against one at a time; CUDA against Vulkan |
| Layer lens (#96 — [`시험/렌즈.py`](../시험/렌즈.py) · [`도구/repro_check.py`](../도구/repro_check.py) `--layer-lens`) | the server's per-layer (all L layers) logit hash, target logprob, entropy, top k and rank against the CPU reference (5.3); the last layer against the echo values of the same response; the echo of the lens path (the plan path, also for four rows or fewer) against the echo of the whole answer without the lens; CUDA against Vulkan (exact); refusals (no echo, echo_from at the end, top_k 21, wrong type); inside the CPU reference, the last layer against the final logits |
| Jacobian lens (#96 phase 2 — [`시험/계약자체.py`](../시험/계약자체.py) · [`시험/렌즈.py`](../시험/렌즈.py) `--jlens` · [`도구/repro_check.py`](../도구/repro_check.py) `--jlens`) | the server's per-layer jacobian_lens (logit hash, 5.2 values, rank) against the CPU reference (5.4 — the same J file), the layer_lens of a response that asked for J against one that did not (bits), the last layer against the logit lens's last layer, CUDA against Vulkan (exact), refusal (wrong type of jacobian); J곱 (item 2 of 5.4) against the contract's sentences carried out element by element — the ordered double sum and the rule-R rounding of the exact rational (subnormal halves, −0, 65504, a cancellation whose result depends on the order, values around the 2⁻¹²⁶ threshold), one row at a time against all rows at once, u = x (bits) when J is the identity; the J file (safetensors): write and read, a file built by hand from the specification, refusal of wrong files; in repro_check, the last layer against the final logits |
| Quick check (CI) | without a GPU: the CPU reference against exact rational arithmetic and against an independent per-value dequantizer; byte-identical regeneration of every module; register spills in SASS; the RoPE table (item 8 — unscaled and YaRN factors 2 · 4 up to position 131,072 — [`엔진/시험_로프표.gl`](../엔진/시험_로프표.gl) against the CPU reference in [`시험/로프표.py`](../시험/로프표.py), byte for byte) — see [reproduce.md](https://github.com/wwoosshh/GeulPU/blob/main/docs/reproduce.md) |
