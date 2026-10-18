# The contract — what is lost, and only where

*English version of [`문서/계약.md`](../문서/계약.md). Terms follow the [glossary](glossary.md).*

Every computation in GeulPU writes its precision into the source. The GPU is left no discretion — not the rounding mode, not the
order of additions, not approximate instructions — so under the same contract the GPU result is **bit-identical** to the CPU
reference implementation ([`계약/정수계약.py`](../계약/정수계약.py)), and identical whichever way it is computed (all at once, in
pieces, one token at a time, as a conversation, batched, or in the megakernel). Loss happens only at the places written below.

## 0. The PTX contract — what the compiler promises

- float32 additions, subtractions and multiplications are emitted with an explicit rounding mode (`.rn`), so the driver cannot fuse a
  multiply and an add into an FMA. A fused multiply-add exists only where the source writes `곱해더하기` (fma).
- No approximate instructions (`ex2.approx` …). exp is a written polynomial ("written exp", below).
- **float32 rule R — subnormal flush** (stage 31 — [`문서/31단계.md`](../문서/31단계.md)). Every float32 operation below follows this
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
- **Embedding** from a k-quant token table: Q4_K · Q5_K give a_j · q − b_j (the product is exact; one rounding at the subtraction),
  Q6_K gives a_k · (q − 32) (exact).
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

Neither changes a single output bit — the stage-27 tests compare against the same CPU reference.

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
5. Chunks are folded as in section 3. The output O / L goes straight into two digits for the output projection.

Because integer sums do not depend on their order, prefill (tensor-core MMA) and decode (dp4a) produce the same bits — this follows
directly from the contract.

## 3. Two-level fold (stage 13)

Chunks are grouped (16 chunks = 1024 positions; group g covers positions [1024g, 1024g + 1024)):

1. Within a group, chunks are folded in order into a group state (M_g, L_g, O_g) (the group's first chunk is the initial value).
2. Groups are folded in order g = 0, 1, … with the same formula (the first group is the initial value).

The fold (both levels): M′ = max(M, m), α = written_exp(M − M′), β = written_exp(m − M′), L ← fma(l, β, L·α), O ← fma(o, β, O·α).
Group boundaries depend only on positions, so every way of computing gives the same bits. Up to position 1024 there is one group and
this equals the one-level fold of stages 9–12. Keeping only group states in global memory lets one (row, head) be split across blocks —
the reason short chunks at long contexts became fast. Accuracy is unchanged (median error against fp64, new / old: 1.0003× and 0.9994×).

## 4. Llama-family blocks (stage 14 — Qwen2)

1. **RMSNorm** (one row of width D): lane t (0 … 255) sums p_t = Σ_{c ≡ t (mod 256)} x_c · x_c (ascending c, rounding at every product
   and sum); the 256 partial sums are folded in halves from 128 down to S; r = 1 / sqrt(S / D + eps) (IEEE round-to-nearest);
   y_c = (x_c · r) · w_c. When the next operation is a linear layer, y goes straight into two digits.
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
   lane t < 128 holds one value and the 128-fold adds +0, so it reads: p_t = x_t · x_t, folded in halves from 64 down to S
   (p_t ← p_t + p_{t+fold}); r = 1 / sqrt(S / 128 + eps); y_t = (x_t · r) · w_t. The weights w are the layer's `attn_q_norm` ·
   `attn_k_norm` (128 float32 values shared by all heads); eps is the model's (it must equal the generator's constant). v is not
   normalized. RoPE (head size 128, item 2 below) and the integer KV cache follow unchanged. The kernel gives lane l the values
   d = l + 32·i (i = 0 … 3), so the 64-fold is two adds (p_l + p_{l+64}, p_{l+32} + p_{l+96}), the 32-fold adds those two, and xor
   shuffles 16 · 8 · 4 · 2 · 1 finish the fold — bit-identical to the order above (checked over 4,096 heads in the contract self-test).
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
      is item 1 unchanged.

### Head size 128 (stage 22 — Qwen2.5-7B and larger)

The head-64 contract is unchanged. Three things differ at head size 128:

1. **Score scale**: the score sum chains the four 32-value blocks in ascending b with fmas (block 0 is the initial value — the same
   form as the two blocks of head 64), then multiplies once: s ← s × c with c = float32(1/√128) = 0x3DB504F3 =
   0.088388346135616302490234375 (round-to-nearest). Head 64's 0.125 is a power of two and can be applied anywhere; c is not, so its
   place is fixed.
2. **RoPE**: the table is [position][64], θ_i = θ^(−2i/128) (i = 0 … 63, correctly rounded in fp64), rotation pairs (i, i + 64). The
   fma forms are those of head 64.
3. **Output o_d** uses the formula of section 2.4 for each of the four 32-value blocks of values (d = 0 … 127). Written exp, the sum l
   and the two-level fold are unchanged. A group or chunk state is (M, L, two spare, O[128]) = 132 values.

## 5. Sampling (stage 28 — the server)

Non-greedy generation picks the next token from one row of logits z (the V float32 values the engine produced) by this contract. All
arithmetic is float64, rounded once per operation in the written order, with no fused multiply-add. The chosen token depends only on
(that row of logits, the parameters, the seed, the step). The logits do not depend on the batch, the slot or how the prompt was fed
(sections 1–4), so sampling does not either. The geul-language version is [`호스트/뽑기.gl`](../호스트/뽑기.gl); the Python version is
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

### 5.2 Log-probabilities (stage 38 — the server, [`문서/38단계.md`](../문서/38단계.md))

Log-probabilities (`logprobs`, `top_logprobs` and the scores of `echo`) are computed from one row of logits z (the V float32 values the
engine produced) by this contract. All arithmetic is float64, rounded once per operation in the written order, with no fused
multiply-add. The distribution is **the model's** (temperature 1, no filtering) — not the sampling distribution of section 5. The
geul-language version is [`호스트/로그확률.gl`](../호스트/로그확률.gl); the Python version is [`계약/로그확률.py`](../계약/로그확률.py).
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
- **Fingerprints and reproduction capsules** (research panel — a record format, not part of the arithmetic contract): token fingerprint
  = SHA-256(`geulpu-fp/1\nmodel <model file SHA-256>\nprompt <ids,…>\ntokens <ids,…>\n`), logit fingerprint = SHA-256(`<token
  fingerprint>\n<per-step logit hashes,…>\n`) — the device and backend are not included. A capsule (`geulpu-capsule/1`) holds the request
  (prompt tokens, temperature, top-k, top-p, min-p, seed as the raw integer, `step_offset`, `max_tokens`), the result (tokens,
  log-probabilities, entropy, top k, logit hashes), the model (name, file, size, SHA-256) and where it was made (device, backend — a
  record written by the maker, not used in checking). [`도구/캡슐확인.py`](../도구/캡슐확인.py) checks it without a GPU, with this
  contract's CPU reference (sections 1–5): the same tokens, logit hashes, log-probabilities and fingerprints.

## 6. What the tests check ([`시험/`](../시험/))

| Test | What |
|---|---|
| Matmul | every linear kernel of the module (tensor-core tiles · dp4a row kernels) against the CPU reference — real layer weights, random and extreme activations |
| Attention kernels | KV cache bytes, prompt-attention output digits and info, decode O/L against the CPU reference |
| Paths | the same token sequence all at once · one at a time · prompt + one at a time · two pieces · megakernel · three sentences batched · 64 tokens at a time · repeated — logits compared bit for bit |
| Cache bytes | whether the KV cache filled by those paths equals the one filled one token at a time |
| Conversation | chunks of different lengths fed and generated in turns equal one-shot processing, down to tokens and cache |
| Long context | the new kernels against the CPU reference (two heads), against each other (all heads); megakernel vs separate kernels; split vs one-shot feeding |
| 7B — head 128 | RoPEKV · group kernels · group fold · chunk kernels · chunk fold (2100 random rows · real values of 7B layer 0, five paths) against the CPU reference; row kernels (five weights × rows 1 … 4 × three epilogues); all rows of logits for three chat prompts + 16 generated tokens against the CPU reference's full forward pass; the five paths (decode graph · all at once · in four · one row at a time · again) — logits and KV cache |
| Qwen2 ([`시험/qwen2.py`](../시험/qwen2.py)) | every linear kernel (row-major · column-major · row · batched-row) · RMSNorm · RoPE+KV · SwiGLU conversions · GQA attention (five kernels + the decode kernel) against the CPU reference; paths (all at once · one at a time · two pieces · split feeding · small chunks · conversation) — logits and cache; all rows of logits for three chat prompts, and the last row beyond position 1024, against the CPU reference's full forward pass; repeatability of the terminal runner; GPT-2 module source unchanged |
| Qwen2 decode megakernel ([`시험/메가15.py`](../시험/메가15.py)) | megakernel vs the stage-14 decode graph — logits and tokens at each of 64 steps for three chat prompts and contexts 1024 · 2100 · 8192, then the KV cache bytes; three turns of the runner's conversation (stopping inside the kernel at the end token) — tokens and cache; GPT-2 module source unchanged |
| k-quants (stage 26) | every k-quant kernel (243 cases: 11 tile shapes × epilogues × formats, row kernels, embeddings) on real weights of three GGUF files; full forward passes of Q4_K_M · Q5_K_M · Q6_K against the CPU reference |
| Qwen3 (stage 37 — [`시험/머리정규화37.py`](../시험/머리정규화37.py) · [`시험/참조7B.py`](../시험/참조7B.py) · [`시험/참조생성.py`](../시험/참조생성.py)) | the head-norm kernel (rows 1 · 4 · 64 · 2048 × the weights of every layer of 8B, random and extreme values) against the CPU reference — v untouched; every row's logits of three chat prompts + 16 generated tokens for 1.7B · 8B (Q8_0 · Q4_K_M) · two synthetic 14B-shape models against the CPU reference's full forward pass; five-path, Vulkan and shared32-emulation summaries equal to CUDA; greedy generation with the CPU reference (not a criterion — compared with llama.cpp) |
| Log-probabilities (stage 38 — [`시험/로그확률38.py`](../시험/로그확률38.py) · [`시험/로그확률서버38.py`](../시험/로그확률서버38.py)) | the geul version (many rows at once and one at a time) against the Python reference — target logprob, entropy and top 20 of synthetic and real logit rows; the reference computed from the logit rows the server recorded (generation steps and echo rows) against the response values, and the rows' FNV hashes; the same tokens scored with echo (cache on and off) against the values from generation; concurrent against one at a time; CUDA against Vulkan |
| Quick check (CI) | without a GPU: the CPU reference against exact rational arithmetic and against an independent per-value dequantizer; byte-identical regeneration of every module; register spills in SASS — see [reproduce.md](reproduce.md) |
