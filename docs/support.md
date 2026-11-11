# What GeulPU runs — environments, architectures, formats and models

**[한국어](../지원범위.md)** · [README](../README.md) · [User guide](user-guide.md) · [Glossary](glossary.md)

This page lists, in tables, **which environments** (OS and GPU) GeulPU runs on and **which model architectures, weight formats and
model files** it runs there. It describes main as of 2026-11-10 — release 0.23.0 plus the 0.24.0 work in progress (the deepseek2
architecture: GLM-4.7-Flash and Kanana-2). The tables come from the engine's architecture registry ([구조/등록.gl](../구조/등록.gl) and
each architecture's `엔진.gl`), the app's model list ([앱/data/models.json](../앱/data/models.json)) and the check records
([문서/0.24.0.md](../문서/0.24.0.md) and others).

[Environments](#1-environments--os-and-gpu-backend) · [Architectures](#2-architectures--what-each-model-family-gets) ·
[Formats](#3-weight-formats--per-environment) · [Model list](#4-model-list--files-the-app-downloads) ·
[Memory](#5-by-graphics-memory-and-ram) · [Not yet](#6-not-yet-supported) · [Checked hardware](#7-checked-hardware)

✓ means supported, — means not yet.

## 1. Environments — OS and GPU backend

| OS | GPU backend | GPUs and conditions | Checked on |
|---|---|---|---|
| Windows x64 (10, 11) | CUDA | NVIDIA RTX 30 series and newer (Ampere, Ada, Blackwell). Only the graphics driver is needed — no CUDA toolkit | RTX 3080 Ti · RTX 4070 Ti · RTX 5070 |
| Windows x64 | Vulkan 1.3 — 48 KB or more of shared memory | AMD, Intel, NVIDIA RTX 20 and older — GPUs not on the CUDA path | Radeon RX 9060 XT (two PCs) · RTX 4070 Ti (Vulkan) |
| Windows x64 | Vulkan 1.3 — 32 KB of shared memory | GPUs with 32 KB of shared memory, such as AMD's Windows driver and Intel integrated GPUs (some formats and models only — sections 2 and 3) | Intel UHD Graphics 770 · the 32 KB limit emulated on an RTX 4070 Ti |
| Linux x86-64 (glibc 2.34 or newer) | CUDA | NVIDIA, driver only | H100 NVL (MIG 3g.47gb — remote server) · WSL2. Independent test (UPM GING, 0.21.0): RTX 6000 Ada · H100 NVL MIG · A100 MIG |
| Linux x86-64 | Vulkan | In the package, but not yet checked on a real Linux GPU (software Vulkan only) | — |
| macOS | — | Not supported | — |

On first start the app checks the GPU and picks CUDA for NVIDIA RTX 30 and newer, Vulkan otherwise (changeable in the settings). With the same
model and settings the output is bit-identical across GPUs and backends ([section 7](#7-checked-hardware)).

## 2. Architectures — what each model family gets

The architecture is the GGUF file's `general.architecture`. Fine-tuned models run when the architecture matches (the chat template in
the file is executed as it is).

| Architecture | Model families | Weight formats | CUDA | Vulkan | Vulkan 32 KB | Computation modes | 8-bit KV cache · fast attention | Speculative decoding | Prompt cache | Logit lens |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen2` | Qwen2.5 · Qwen2.5-Coder · DeepSeek-R1-Distill-Qwen · A.X 4.0 Light | Q8_0 · Q4_0 · k-quants · IQ4_XS · F16 · BF16 · F32 | ✓ | ✓ | ✓ ¹ | exact · fast | ✓ ² | ✓ | ✓ | ✓ |
| `qwen3` | Qwen3 | same | ✓ | ✓ | ✓ ¹ | exact · fast | ✓ ² | ✓ | ✓ | ✓ |
| `llama` | Llama 3.1 · 3.2 · Mistral 7B v0.3 · DeepSeek-R1-Distill-Llama · HyperCLOVAX-SEED · MiniCPM5 · Mi:dm 2.0 · Kanana 1.5 | same | ✓ | ✓ | ✓ ¹ | exact · fast | ✓ ² | ✓ | ✓ | ✓ |
| `qwen35` | Qwen3.5 · Qwen3.8 (full-attention and linear-attention layers mixed) | Q8_0 · k-quants · IQ4_XS · F16 · BF16 · F32 | ✓ | — | — | exact · fast | — | — | — | — |
| `qwen35moe` | Qwen3.5 · 3.6 MoE | same | ✓ | — | — | exact | — | — | — | — |
| `gemma4` | Gemma 4 E2B · E4B · 12B · 26B-A4B · 31B | all matrices Q8_0 or all k-quants ⁴ | ✓ | — | — | exact | — | — | — | ✓ |
| `gemma3` | Gemma 3 270M · 1B · 4B · 12B · 27B (0.24.0 — fine-tunes of those sizes too) | same ³ | ✓ | — | — | exact | — | — | — | — |
| `deepseek2` | GLM-4.7-Flash · Kanana-2-30B-A3B (0.24.0 — MLA attention + MoE) | k-quants (per-head matrices also Q8_0 · Q5_0 · k-quants) | ✓ | — | — | exact | — | — | — | — |

1. Vulkan with 32 KB of shared memory takes Q8_0, Q4_0 and k-quant files of models with head size 128 and RoPE over the whole head
   (not F16 · BF16 · F32 or IQ4_XS).
2. The 8-bit KV cache and fast attention need a k-quant or IQ4_XS file of a model with head size 128 and query width equal to the
   hidden width (`qwen3`'s fast attention is F2B, with an exponent per block).
3. Gemma 3 270M and 1B run from Q8_0 files only — their widths are not multiples of 256, so the files with k-quant names (Q4_K_M
   and so on) contain Q5_0 and Q8_0 matrices as well.
4. Gemma 4 E2B and E4B (per-layer inputs and KV sharing — 0.24.0) run from k-quant files only — the matrices of the per-layer
   inputs come along as F32 and BF16. 26B-A4B (MoE — 0.24.0) runs from k-quant files only too — the downs whose depth is not a multiple
   of 256 are stored by the file as Q8_0 or Q5_0 (Q5_0 becomes Q8_0 blocks at load — the same values). Q8_0, F16 and BF16 files of
   these three sizes are not supported yet.

MoE architectures (`qwen35moe`, `deepseek2`, the 26B-A4B of `gemma4`) keep expert weights that do not fit in graphics memory in RAM (pinned memory). **Tool
calls** depend on the chat template, not the architecture — the server reads Qwen's `<tool_call>` JSON, Llama 3's JSON, Qwen3.5 and
Qwen3-Coder XML, MiniCPM XML, Gemma 4's `<|tool_call>call:`, `<tool_call>` JSON written as text (Mi:dm 2.0), functionary's
`<function=name>` (Kanana 1.5) and GLM's `<arg_key>` form (GLM-4.5 · 4.7 — 0.24.0). Details in [문서/에이전트.md](../문서/에이전트.md).

## 3. Weight formats — per environment

| Format | CUDA | Vulkan (48 KB or more) | Vulkan 32 KB | Architectures |
|---|---|---|---|---|
| Q8_0 | ✓ | ✓ | ✓ | `qwen2` · `qwen3` · `llama` · `qwen35` · `qwen35moe` · `gemma4` · `gemma3` |
| Q4_0 (files whose tensors are all Q4_0) | ✓ | ✓ | ✓ | `qwen2` · `qwen3` · `llama` |
| k-quants — Q4_K_M · Q5_K_M · Q6_K (files mixing Q4_K · Q5_K · Q6_K) | ✓ | ✓ | ✓ ¹ | all |
| IQ4_XS | ✓ | — | — | `qwen2` · `qwen3` · `llama` · `qwen35` · `qwen35moe` |
| F16 · BF16 · F32 (unquantized files — converted at load to the precision contract's 16-bit block format) | ✓ | ✓ | — | `qwen2` · `qwen3` · `llama` · `qwen35` · `qwen35moe` |
| Q5_0 | ✓ | — | — | only `deepseek2`'s per-head matrices (Kanana-2's `attn_k_b`) |
| Q2_K · Q3_K · IQ1–IQ3 · IQ4_NL · MXFP4 · Q4_1 · Q5_1 | — | — | — | not yet (stage 24-4 of the 0.24.0 plan) |
| Split GGUF (`-00001-of-…`) | merge into one file with llama.cpp's `llama-gguf-split --merge`, then as above | | | |

What a file whose matrices use different formats may mix is in [문서/형식섞임.md](../문서/형식섞임.md). The app's **Find models** shows,
for each Hugging Face GGUF repository, which files run by these tables and whether they fit this PC's graphics memory.

## 4. Model list — files the app downloads

The files in the app's model list (the app checks their SHA-256 after downloading). **Checked** is the backend and date this file
was run on with GeulPU; under **Tool calls**, ✓ means the tool-call test passed in the app (with the server for the Gemma 4 31B, which does not fit on
this PC), — means the chat template has no tools, and
blank means not yet tested with that file. `crosscheck` is the reference model of the reproducibility check; `untested` means the
bits were checked only with a synthetic model of the same shape. File sizes are in units of 10⁹ bytes (Windows Explorer's
1024-based units show them a little smaller).

| Model | Architecture | Size | Format · file | CUDA | Vulkan | Vulkan 32 KB | Tool calls | Checked | License | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-7B | `qwen2` | 7B | Q8_0 · 8.1 GB | ✓ | ✓ | ✓ | | crosscheck (reproducibility reference) | MIT | |
| Qwen2.5-7B-Instruct | `qwen2` | 7B | Q8_0 · 8.1 GB | ✓ | ✓ | ✓ | | CUDA · 2026-10-15 | Apache-2.0 | |
| Qwen2.5-7B-Instruct | `qwen2` | 7B | Q4_K_M · 4.7 GB | ✓ | ✓ | ✓ | | CUDA · 2026-10-15 | Apache-2.0 | |
| Qwen2.5-3B-Instruct | `qwen2` | 3B | Q8_0 · 3.6 GB | ✓ | ✓ | ✓ | | CUDA · Vulkan · 2026-10-23 | Qwen Research License | research and evaluation only |
| Qwen2.5-1.5B-Instruct | `qwen2` | 1.5B | Q8_0 · 1.9 GB | ✓ | ✓ | ✓ | | CUDA · Vulkan · 2026-10-23 | Apache-2.0 | |
| DeepSeek-R1-Distill-Qwen-7B | `qwen2` | 7B | Q4_K_M · 4.7 GB | ✓ | ✓ | ✓ | | CUDA · 2026-10-15 | MIT | |
| Qwen2.5-Coder-7B-Instruct | `qwen2` | 7B | Q8_0 · 8.1 GB | ✓ | ✓ | ✓ | | CUDA · 2026-10-15 | Apache-2.0 | |
| Qwen2.5-Coder-14B-Instruct | `qwen2` | 14B | Q4_K_M · 9.0 GB | ✓ | ✓ | ✓ | | CUDA · Vulkan · 2026-10-23 | Apache-2.0 | |
| Qwen2.5-14B-Instruct | `qwen2` | 14B | Q4_K_M · 9.0 GB | ✓ | ✓ | ✓ | | CUDA · 2026-10-15 | Apache-2.0 | |
| DeepSeek-R1-Distill-Qwen-14B | `qwen2` | 14B | Q4_K_M · 9.0 GB | ✓ | ✓ | ✓ | | CUDA · 2026-10-15 | MIT | |
| DeepSeek-R1-Distill-Llama-8B | `llama` | 8B | Q4_K_M · 4.9 GB | ✓ | ✓ | ✓ | — | CUDA · Vulkan · 2026-10-23 | MIT · Llama 3.1 Community License | |
| Qwen3-8B | `qwen3` | 8B | Q8_0 · 8.7 GB | ✓ | ✓ | ✓ | | CUDA · Vulkan · 2026-10-15 | Apache-2.0 | |
| Qwen3-8B | `qwen3` | 8B | Q4_K_M · 5.0 GB | ✓ | ✓ | ✓ | | CUDA · Vulkan · 2026-10-15 | Apache-2.0 | |
| Qwen3-4B | `qwen3` | 4B | Q8_0 · 4.3 GB | ✓ | ✓ | ✓ | ✓ | CUDA · Vulkan · 2026-10-23 | Apache-2.0 | |
| Qwen3-4B | `qwen3` | 4B | Q4_K_M · 2.5 GB | ✓ | ✓ | ✓ | ✓ | CUDA · Vulkan · 2026-10-23 | Apache-2.0 | |
| Qwen3-1.7B | `qwen3` | 1.7B | Q8_0 · 1.8 GB | ✓ | ✓ | ✓ | | CUDA · Vulkan · 32 KB emulation · 2026-10-15 | Apache-2.0 | |
| Qwen3-0.6B | `qwen3` | 0.6B | Q8_0 · 0.6 GB | ✓ | ✓ | ✓ | ✓ | CUDA · Vulkan · 2026-10-23 | Apache-2.0 | |
| Qwen3-14B | `qwen3` | 14B | Q4_K_M · 9.0 GB | ✓ | ✓ | ✓ | | untested (bits checked with a synthetic model) | Apache-2.0 | |
| Qwen3.6-35B-A3B | `qwen35moe` | 35B | Q4_K_M · 20.4 GB | ✓ | — | — | | CUDA · 2026-10-24 | Apache-2.0 | MoE — about 10.5 GB of RAM on a 12 GB GPU |
| Qwen3.5-9B | `qwen35` | 9B | Q4_K_M · 5.7 GB | ✓ | — | — | | CUDA · 2026-10-25 | Apache-2.0 | |
| Qwen3.5-4B | `qwen35` | 4B | Q4_K_M · 2.7 GB | ✓ | — | — | | CUDA · 2026-10-25 | Apache-2.0 | |
| Qwen3.5-0.8B | `qwen35` | 0.8B | Q8_0 · 0.8 GB | ✓ | — | — | | CUDA · 2026-10-24 | Apache-2.0 | |
| Gemma-4-E2B-it | `gemma4` | 4.6B | Q4_K_M · 3.1 GB | ✓ | — | — | ✓ | CUDA · 2026-11-11 | Apache-2.0 | per-layer inputs (PLE) · 0.24.0 |
| Gemma-4-E4B-it | `gemma4` | 7.5B | Q4_K_M · 5.0 GB | ✓ | — | — | ✓ | CUDA · 2026-11-11 | Apache-2.0 | per-layer inputs (PLE) · 0.24.0 |
| Gemma-4-12B-it | `gemma4` | 12B | Q4_K_M · 7.1 GB | ✓ | — | — | ✓ | CUDA · 2026-10-25 | Apache-2.0 | |
| Gemma-4-26B-A4B-it | `gemma4` | 25.2B | Q4_K_M · 16.8 GB | ✓ | — | — | ✓ | CUDA · 2026-11-11 | Apache-2.0 | MoE — about 10 GB of RAM on a 12 GB GPU · 0.24.0 |
| Gemma-4-31B-it | `gemma4` | 31B | Q4_K_M · 18.3 GB | ✓ | — | — | ✓ | CUDA (H100 · Linux — with the server) · 2026-11-11 | Apache-2.0 | 24 GB of graphics memory or more · not run in the app's screens · 0.24.0 |
| Gemma-3-4B-it | `gemma3` | 4B | Q4_K_M · 2.5 GB | ✓ | — | — | — | CUDA · 2026-11-10 | Gemma Terms of Use | text only · 0.24.0 |
| Gemma-3-12B-it | `gemma3` | 12B | Q4_K_M · 7.3 GB | ✓ | — | — | — | CUDA · 2026-11-10 | Gemma Terms of Use | text only · 0.24.0 |
| Gemma-3-27B-it | `gemma3` | 27B | Q4_K_M · 16.5 GB | ✓ | — | — | — | CUDA (H100 · Linux — with the server) · 2026-11-11 | Gemma Terms of Use | text only · a 24 GB GPU is recommended · not run in the app's screens · 0.24.0 |
| Gemma-3-1B-it | `gemma3` | 1B | Q8_0 · 1.1 GB | ✓ | — | — | — | CUDA · 2026-11-10 | Gemma Terms of Use | Q8_0 files only · 0.24.0 |
| Gemma-3-270M-it | `gemma3` | 270M | Q8_0 · 0.3 GB | ✓ | — | — | — | CUDA · 2026-11-10 | Gemma Terms of Use | Q8_0 files only · 0.24.0 |
| GLM-4.7-Flash | `deepseek2` | 30B | Q4_K_M · 18.3 GB | ✓ | — | — | ✓ | CUDA · 2026-11-10 | MIT | MoE — about 10 GB of RAM on a 12 GB GPU · 0.24.0 |
| Llama-3.1-8B-Instruct | `llama` | 8B | Q4_K_M · 4.9 GB | ✓ | ✓ | ✓ | | CUDA · Vulkan · 2026-10-16 | Llama 3.1 Community License | |
| Llama-3.1-8B-Instruct | `llama` | 8B | Q8_0 · 8.5 GB | ✓ | ✓ | ✓ | | CUDA · Vulkan · 2026-10-16 | Llama 3.1 Community License | |
| Llama-3.2-3B-Instruct | `llama` | 3B | Q8_0 · 3.4 GB | ✓ | ✓ | ✓ | | CUDA · Vulkan · 32 KB emulation · 2026-10-16 | Llama 3.2 Community License | |
| Mistral-7B-Instruct-v0.3 | `llama` | 7B | Q4_K_M · 4.4 GB | ✓ | ✓ | ✓ | | CUDA · Vulkan · 2026-10-16 | Apache-2.0 | |
| Mistral-7B-Instruct-v0.3 | `llama` | 7B | Q8_0 · 7.7 GB | ✓ | ✓ | ✓ | | CUDA · Vulkan · 2026-10-16 | Apache-2.0 | |
| HyperCLOVAX-SEED-Text-Instruct-1.5B | `llama` | 1.5B | Q4_K_M · 1.1 GB | ✓ | ✓ | ✓ | — | CUDA · Vulkan · 2026-10-23 | HyperCLOVA X SEED Model License | |
| A.X-4.0-Light | `qwen2` | 7B | Q4_K_M · 4.4 GB | ✓ | ✓ | ✓ | ✓ | CUDA · Vulkan · 2026-11-09 | Apache-2.0 | context up to 16,384 |
| Midm-2.0-Base-Instruct | `llama` | 11.5B | Q4_K_M · 7.0 GB | ✓ | ✓ | ✓ | ✓ | CUDA · Vulkan · 2026-11-10 | MIT | |
| kanana-1.5-8b-instruct-2505 | `llama` | 8B | Q4_K_M · 4.9 GB | ✓ | ✓ | ✓ | ✓ | CUDA · Vulkan · 2026-11-10 | Apache-2.0 | |
| Kanana-2-30B-A3B-Thinking-2601 | `deepseek2` | 30B | Q4_K_M · 18.6 GB | ✓ | — | — | ✓ | CUDA · 2026-11-10 | Kanana License Agreement | MoE — about 11 GB of RAM on a 12 GB GPU · 0.24.0 |
| MiniCPM5-2B | `llama` | 2B | Q4_K_M · 1.6 GB | ✓ | ✓ | ✓ | ✓ | CUDA · Vulkan · 2026-10-23 | Apache-2.0 | |

All CUDA checks ran on an RTX 4070 Ti. Files outside the list run too when their architecture and format are in sections 2 and 3 —
add them with **Add a GGUF file** or **Find models** (the engine checks a file before loading it). License conditions (use policies,
commercial use) are in [the user guide's supported models](user-guide.md#supported-models) and in the app's model descriptions.

## 5. By graphics memory and RAM

Files GeulPU has run, by graphics memory. A dense model needs the file and the context (KV cache) in graphics memory; an MoE model keeps
the experts that do not fit in RAM (that much RAM must be free as well).

| Graphics memory | Dense models | MoE models (need extra RAM) |
|---|---|---|
| 4 GB | Qwen3-0.6B Q8_0 · Qwen3.5-0.8B Q8_0 · Gemma-3-270M Q8_0 · Gemma-3-1B Q8_0 · HyperCLOVAX-SEED 1.5B Q4_K_M · MiniCPM5-2B Q4_K_M | — |
| 6 GB | Qwen3-1.7B Q8_0 · Qwen2.5-1.5B Q8_0 · Qwen3-4B Q4_K_M · Qwen3.5-4B Q4_K_M · Gemma-3-4B Q4_K_M · Gemma-4-E2B Q4_K_M · Llama-3.2-3B Q8_0 · Qwen2.5-3B Q8_0 | — |
| 8 GB | 7B · 8B · 9B Q4_K_M (Qwen3-8B · Qwen3.5-9B · Qwen2.5-7B · R1-Distill-Qwen-7B · R1-Distill-Llama-8B · Llama-3.1-8B · Mistral-7B · A.X 4.0 Light · Kanana 1.5) · Gemma-4-E4B Q4_K_M · Qwen3-4B Q8_0 | — |
| 12 GB | 7B · 8B Q8_0 · Mi:dm 2.0 Q4_K_M · Gemma-4-12B Q4_K_M · Gemma-3-12B Q4_K_M · 14B Q4_K_M (with a shorter context) | Qwen3.6-35B-A3B (about 10.5 GB of RAM) · GLM-4.7-Flash (about 10 GB) · Kanana-2-30B-A3B (about 11 GB) · Gemma-4-26B-A4B (about 10 GB) |
| 16 GB and more | Qwen2.5-14B · Qwen2.5-Coder-14B · R1-Distill-Qwen-14B Q4_K_M | as above, with less RAM |
| 24 GB and more | Gemma-3-27B Q4_K_M (about 17 GB at a context of 4,096) · Gemma-4-31B Q4_K_M (about 20 GB) — measured on the remote H100 server (they do not fit on this PC) | — |

## 6. Not yet supported

| Architecture (GGUF) | Models | Why · plan |
|---|---|---|
| `glm4moe` | GLM-4.5 · 4.6 · 4.7 (355B) · GLM-4.5-Air (106B) | no such architecture yet — ordinary (GQA) attention + MoE. The only GLM GeulPU takes is GLM-4.7-Flash (`deepseek2`) |
| `glm-dsa` | GLM-5 · 5.1 · 5.2 · 5.3 (754B) | no such architecture yet — the same MLA + MoE as `deepseek2` plus DSA (sparse attention over 2,048 keys per query) and 64 heads. About 450 GB at 4 bits |
| `glm5-next` | GLM-5.3-Flash (321B) | no such architecture yet — linear-attention layers, DSA layers, mHC and image input. About 190 GB at 4 bits |
| other `deepseek2` models | DeepSeek-V3 · R1 · Kimi-K2 · DeepSeek-V2 (-Lite) | files with expert group selection, softmax routing or MTP layers are not taken yet |
| `qwen3moe` · `mixtral` · `phi3` and others | Qwen3-30B-A3B · Qwen3-235B-A22B · Mixtral · Phi | no such architecture yet |
| `raon1` | Raon-1 mini (Krafton) | stage 24-3 of the 0.24.0 plan |
| `gemma4` variants · `exaone4` | the Q8_0 · F16 · BF16 files of Gemma 4 E2B · E4B · 26B-A4B, EXAONE 4.0 · 4.5 | stage 24-5 of the 0.24.0 plan |
| some `gemma3` files | the files of the Gemma 3 270M and 1B with k-quant names | the Q4_K_M and similar files mix in Q5_0 and Q8_0 matrices — use the Q8_0 file |
| Image and audio input | vision and audio parts of multimodal models | text only |

Formats: Q2_K · Q3_K · IQ1–IQ3 · IQ4_NL · MXFP4 · Q4_1 · Q5_1 (stage 24-4 of the 0.24.0 plan). Features: Vulkan for `qwen35`, `qwen35moe`,
`gemma4`, `gemma3` and `deepseek2`; fast mode for `qwen35moe`, `gemma4`, `gemma3` and `deepseek2`; speculative decoding and the prompt cache for the newer
architectures. Speed: `deepseek2` decoding is still slower than llama.cpp's fastest configuration (about 0.6–0.85 times; long prompts are processed
1.5–1.75 times faster — [문서/MoE전문가캐시.md](../문서/MoE전문가캐시.md)). `gemma3` is slower than llama.cpp as well (decoding at about 0.65–0.9 times,
prompt processing at about 0.6–1.05 times — [구조/gemma3/기록.md](../구조/gemma3/기록.md)). So are Gemma 4 E2B · E4B (decoding at
about 0.8 times, prompt processing at about 0.7–1.03 times — [구조/gemma4/기록.md](../구조/gemma4/기록.md)). So is Gemma 4 26B-A4B
(decoding at about 0.7–0.8 times on an H100 NVL MIG slice; prompt processing at about 0.72 times for 2,048 tokens and 1.43 times
for 512 — same record).

## 7. Checked hardware

| GPU | Architecture | OS · backend | What was checked |
|---|---|---|---|
| RTX 4070 Ti 12 GB | Ada (sm_89) | Windows · CUDA · Vulkan | reference PC — every architecture, format and model file against the CPU reference, bit for bit; the 32 KB shared-memory limit emulated |
| RTX 3080 Ti · RTX 5070 | Ampere · Blackwell | Windows · CUDA | the reproducibility check (DeepSeek-R1-Distill-Qwen-7B Q8_0 — 32 math problems, 82,673 steps): outputs and logits bit-identical to the 4070 Ti |
| Radeon RX 9060 XT | RDNA4 | Windows · Vulkan (two PCs, two drivers) | the same check equal to the CUDA reference in all 82,673 steps (exact mode) |
| Intel UHD Graphics 770 | integrated GPU | Windows · Vulkan (32 KB shared memory) | five requests · 240 steps of Qwen2.5-1.5B Q4_K_M with logits equal to CUDA (0.12.0) |
| H100 NVL — MIG 3g.47gb | Hopper (sm_90) | Linux · CUDA (remote server) | 0.12.1's reproducibility test 24/24; 0.24.0's GLM-4.7-Flash, Kanana-2, Gemma 3 4B · 12B and Gemma 4 26B-A4B modules and test outputs byte-identical to the 4070 Ti (for the 26B-A4B, between the 4070 Ti's expert cache and all experts in VRAM here). The Gemma 3 27B and Gemma 4 31B, which do not fit on the 4070 Ti, are verified here (their modules are byte-identical; chat checked with the server) |
| RTX 6000 Ada · H100 NVL MIG · A100 MIG 7g.40gb | Ada · Hopper · Ampere | Linux · CUDA — independent test by UPM GING (0.21.0) | reproducibility kit 144/144, reproducibility check 32/32, all of their own suites passed ([repro-kit/independent](../repro-kit/independent/upm-ging-2026-11/README.md)) |

This page is updated in every release that changes the architectures, formats or model list. Changes are listed in
[CHANGELOG.md](../CHANGELOG.md).
