# File-pair reproducibility

*Korean version: [문서/파일쌍.md](../문서/파일쌍.md).*

A benchmark item of the GeulPU app (from 0.11.1). It runs GGUF files that quantize the same model in different ways in **exact mode**
and answers two questions:

1. **Is the execution reproducible?** Every step of 32 math problems produces a logit row; its FNV-1a 64 hash is compared with the
   hash the reference PC recorded. Equal hashes on another GPU mean that GPU computed the same bits, step by step.
2. **How much do the files differ?** For files compared with the reference file `default`: the first step where the greedy answers
   diverge, and the per-token log-probability difference when `default`'s answers are scored by the other file (echo scoring — the
   same arithmetic and summation order as `도구/debugger.py compare`).

The files come from Wei-Liang Liu and Cheng-Yen Tsai's paper on the verifiability gap in quantized GGUF supply chains
([arXiv:2610.19498](https://arxiv.org/abs/2610.19498), artifact repository
[gguf-verifiability-artifact](https://github.com/Rlonglong/gguf-verifiability-artifact)). Their paper makes the *file* verifiable —
whether a quantization re-derives bit for bit; this item makes the *execution* verifiable — whether the logits a file produces are the
same on every GPU.

## Files

| Label | File | Size | SHA-256 | Source |
|---|---|---|---|---|
| `default` | Qwen2.5-1.5B-Instruct-Q4_K_M-b10448-default-arm64.gguf | 986,048,512 | `dd707be38b3665750ee68abd253865c83886a34cec2ffa837d5aa433438381a5` | [Rlong30/qwen2.5-1.5b-q4km-fp-contraction-pair](https://huggingface.co/Rlong30/qwen2.5-1.5b-q4km-fp-contraction-pair) |
| `ffpoff` | Qwen2.5-1.5B-Instruct-Q4_K_M-b10448-ffpoff.gguf | 986,048,512 | `52e47664d0a0001bd58ffdd16beb6e6a0b3609bbfee7a439a462dc6285a23281` | same |
| `imatrix` | Qwen2.5-1.5B-Instruct-Q4_K_M.gguf | 986,048,768 | `1adf0b11065d8ad2e8123ea110d1ec956dab4ab038eab665614adba04b6c3370` | [bartowski/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF) |
| `7b-q4km` | Qwen2.5-7B-Instruct-Q4_K_M.gguf | 4,683,074,240 | `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423` | [bartowski/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF) |
| `7b-q8_0` | Qwen2.5-7B-Instruct-Q8_0.gguf | 8,098,525,888 | `9c6a6e61664446321d9c0dd7ee28a0d03914277609e21bc0e1fce4abe780ce1b` | same |

- `default` and `ffpoff` were quantized from the same bartowski f16 at llama.cpp release b10448 without an imatrix, by two builds of
  `llama-quantize`: arm64 with Apple clang and floating-point contraction on, and the same host with `ggml-quants.c` compiled with
  `-ffp-contract=off` (byte-identical to the x86_64 default build). Header and metadata are identical; 197 of 338 tensors differ
  (4.2 × 10⁻⁴ of all bytes); their wikitext-2 perplexities are 10.7437 and 10.7500 — two byte-different files that perplexity does not
  tell apart.
- `imatrix` is bartowski's own imatrix quantization — a different quantization, not a third build of the same one.
- `7b-q4km` and `7b-q8_0` are bartowski's files, which the paper re-derived byte for byte.

## Settings

Exact mode (two int8 digits per activation, 16-bit KV cache, no fast attention — the [contract](../계약/contract.md)), greedy decoding
(temperature 0), up to 512 new tokens, context 1,024. The 32 problems are those of the app's reproducibility check; each prompt is
Qwen2.5's chat template with its default system prompt, sent as token ids to `/v1/completions` so that the chat-template code cannot
change it between versions ([교차검증/쌍/requests.json](../교차검증/쌍/requests.json), made by
[도구/쌍기준.py](../도구/쌍기준.py)). The reference hashes are in [교차검증/쌍/expected.json](../교차검증/쌍/expected.json).

## Running it

1. Install GeulPU 0.11.1 or later and open **Benchmark → File-pair reproducibility**.
2. Download the files from the card (each one is checked with SHA-256). A file runs if the graphics memory is at least 1 GB larger
   than the file — the 1.5B files need about 2 GB, `7b-q8_0` about 9 GB.
3. Tick the files and press **Run**. The first run on a GPU builds its kernel modules and measures kernel variants, which takes a few
   minutes (more than ten on integrated graphics); later runs reuse them.
4. **Export zip** from the result. The folder holds `summary.json` (per file: same n / 32, the first differing step of any problem
   that differs; `pairs` — between files), `environment.json` (GeulPU version and commit, SHA-256 of the contract document and of
   its CPU reference, GPU, driver, OS), `<label>/responses/NN.json` (each response with its per-step hashes), `<label>/run.jsonl`
   (a run log in the debugger's format) and `pair-default-<B>.json` (the per-token differences).

AMD GPUs run it through Vulkan; on AMD's Windows driver (32 KB of shared memory) k-quant files need 0.11.1 or later.

## Checking a run without a GPU

The run logs can be re-computed on a CPU with the contract's reference implementation (NumPy only — macOS and Linux work):

```
python -I 도구/repro_check.py <result>/default/run.jsonl Qwen2.5-1.5B-Instruct-Q4_K_M-b10448-default-arm64.gguf --requests 0,1
```

It recomputes every step's logit row of the chosen requests and compares the FNV-1a 64 hashes and the greedy tokens with the record.
One request of the 1.5B file takes about ten minutes on a desktop CPU, so pick a few.

## Reference results

@@결과@@

## Citing

The paper: Wei-Liang Liu and Cheng-Yen Tsai, arXiv:2610.19498, DOI 10.48550/arXiv.2610.19498 (BibTeX in the artifact repository's
README). GeulPU: see [CITATION.cff](../CITATION.cff) (Zenodo concept DOI 10.5281/zenodo.23548056).
