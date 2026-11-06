# llama.cpp replay kit — recompute llama.cpp's runs on your GPU

[한국어](README.ko.md)

GeulPU can recompute a run that an **unmodified llama-server** produced, with its own GPU kernels, and compare every generated token
and every step's top-20 log-probabilities **bit for bit** ([docs/llama-replay.md](../../docs/llama-replay.md)). The runs in this kit
were made by llama.cpp b12008 (the official Windows CUDA build) on an NVIDIA RTX 4070 Ti. The recomputation emulates that GPU's
arithmetic, so it should give the same bits on **any** GPU — including AMD and Intel GPUs through Vulkan. We have checked the RTX 4070 Ti
(CUDA and Vulkan) and an Intel UHD 770. This kit is for checking another GPU, in particular an **AMD** one.

| | |
|---|---|
| What it runs | 6 receipts: Qwen3.5-9B Q4_K_M (hybrid) and Qwen3-8B Q4_K_M (dense), three requests each (86 · 468/401 · 1,181/1,136 prompt tokens, up to 256 generated tokens) |
| What it needs | Windows or Linux with a GPU that has Vulkan 1.3; GeulPU 0.22.0; Python 3.8+ (standard library only); 11 GB of disk for the two models |
| Time | A few minutes per receipt on a recent discrete GPU (the first one also builds the replay kernels); much longer on an integrated GPU |
| What it touches | Nothing of yours: it starts its own GeulPU on another port with a temporary data folder and removes it at the end |

## Steps

1. **Install GeulPU 0.22.0** — `pip install geulpu==0.22.0`, or unpack the Windows zip from
   [the release page](https://github.com/wwoosshh/GeulPUv2/releases/tag/v0.22.0).
2. **Get the two model files** into one folder (or pass `--download` in step 3):

   | File | Source | Size (bytes) | SHA-256 |
   |---|---|---|---|
   | `Qwen3.5-9B-Q4_K_M.gguf` | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | 5,680,522,464 | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |
   | `Qwen3-8B-Q4_K_M.gguf` | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) | 5,027,783,488 | `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785` |

3. **Run the kit** from this folder:

   ```
   python run_replay_kit.py --models-dir <folder with the models>
   ```

   With the Windows zip instead of pip, add `--app <path>\GeulPU.exe`. With several GPUs, add `--device <part of the Vulkan device name>`
   (for example `--device Radeon`). On an NVIDIA GPU you can also run `--layer cuda`.
4. **Send back** the file `result-<gpu>-<time>.json` it writes (it holds the GPU name, driver and per-receipt results; no personal data).

## Reading the result

Each receipt prints `SAME` when every token and every log-probability matched the llama.cpp run bit for bit, with the same number of
compared values as on the reference PC (`expected.json`). `DIFFERENT` names the first step where the bits differ. A refusal (for
example a driver that cannot build a kernel) is reported with its reason. Exit code 0 means all six matched.

Two driver facts we know of are already handled: AMD's Windows driver allows 32 KB of shared memory per workgroup (the replay kernels
fit in it) and 2 GiB per allocation (the replay allocates each weight tensor separately). The kit is new on AMD — anything else it
finds is exactly what we want to learn.
