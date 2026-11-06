# llama.cpp replay check — verify a llama-server run bit for bit

*(From GeulPU 0.22.0. The full record — every stage with its criteria and results — is [문서/라마재현.md](../문서/라마재현.md), in Korean.)*

GeulPU can take a run that an **unmodified llama-server** produced and recompute it with its own GPU kernels, which follow llama.cpp's
CUDA arithmetic instruction by instruction. It then compares the generated tokens — and, when the client asked for them, the top-20
log-probabilities of every step — **bit for bit**. The recomputation runs on CUDA or on Vulkan: a run made by llama.cpp on an NVIDIA
RTX 4070 Ti has been recomputed bit-identically on the Intel integrated graphics of the same PC.

What the replay imitates is llama.cpp **as it runs on an RTX 4070 Ti (sm_89, 60 SMs)**. Runs made on other GPUs divide the work
differently and are not verified yet — see [What we need from other GPUs](#what-we-need-from-other-gpus).

## What is verified

| | Supported |
|---|---|
| llama.cpp builds | b11963 (the build Ollama v0.40.14 launches), b12008, b12138 — CUDA, Windows |
| Models | Qwen3.5 (0.8B · 4B · 9B — hybrid), Qwen3, Qwen2.5 (incl. Coder 14B), Llama 3.1 · 3.2, Mistral 7B, DeepSeek-R1-Distill-Qwen; GGUF Q4_K · Q5_K · Q6_K · Q8_0 |
| Requests | `/completion`, `/v1/chat/completions`, streaming, temperature 0, temperature > 0 with a seed (default sampler chain) |
| Server | the default server (4 slots, unified KV, prompt cache, RAM prompt cache), `-np N`, overlapping requests (continuous batching — only when the server wrote a batch log) |
| Verdict | same / different, the first diverging step, the number of differing values; the recomputed run opens in the debugger |

Not verified yet: runs made on other GPUs, other engines (vLLM, SGLang), answers cut by a stop word, temperature > 0 without a seed or
without recorded tokens, non-default sampler chains (penalties, mirostat, grammar, …).

## Three ways to get a record

GeulPU verifies **records**. From the least intrusive:

### 1. A conversation log — nothing touches the run

The client talks to llama-server as usual. Afterwards you give GeulPU what you already have: the conversation, the sampling values the
client sent, and the server's command line and executable. GeulPU recomputes the answers and checks that their text is byte-identical
(temperature 0 only — sampled tokens cannot be traced back from text).

Save the conversation as JSON:

```json
{"sampling": {"temperature": 0, "max_tokens": 64, "seed": 1},
 "request_fields": {"chat_template_kwargs": {"enable_thinking": false}},
 "messages": [
   {"role": "system", "content": "..."},
   {"role": "user", "content": "..."},
   {"role": "assistant", "content": "...the answer exactly as received..."},
   {"role": "user", "content": "..."},
   {"role": "assistant", "content": "...", "sampling": {"seed": 2}}
 ]}
```

- `sampling` holds the values the client sent with every request (`temperature`, `max_tokens`, `seed`, `top_k`, `top_p`, `min_p`, …);
  an assistant message may carry its own `sampling` to override them for that turn.
- `request_fields` holds other fields the client sent with every request (for example `chat_template_kwargs`); omit it if there were none.
- Each assistant message is one request: the messages before it are the request, its content is the answer. Keep the answer text exactly
  as the server returned it.
- One file per conversation. Conversations that share a prefix (a branch in a chat UI) are not supported yet.

Then, in GeulPU's **Benchmark → llama.cpp replay check** card: save the **server settings** (the exact command line that started
llama-server and the path of its executable — GeulPU reads the build with `--version`, which never contacts the running server) and
use **Import conversation log** with the file's path. One token-less receipt per assistant message is created. Verify them **in order**:
each verified turn stores its regenerated tokens, which the next turn's prompt-cache emulation needs.

### 2. The transparent recorder (default)

Point the client at GeulPU instead of llama-server: `http://127.0.0.1:44544/llama/completion` or `…/llama/v1/chat/completions`. GeulPU
forwards every request **byte for byte**, returns the response unchanged (streams too), never queries the server while it runs, and
writes a receipt with the request, the response as sent, the server's command line and build, and the model's path and size. A test
with a proxy between the recorder and llama-server showed that the server receives exactly the client's requests with byte-identical
bodies, and the responses equal those of direct requests.

### 3. The probing recorder (opt-in)

The recorder can add `n_probs` · `return_tokens` (`logprobs` · `top_logprobs` · `verbose` for chat) to each request and query
`/slots`, `/props` and `/tokenize` around it. The receipt then holds token ids and log-probabilities, and the check compares the top-20
log-probabilities of every step bit for bit. This mode **changes the run**: with `n_probs`, llama-server ends its sampler chain with a
distribution sampler even at temperature 0, so a step whose top two logits are exactly equal can pick a different token. Use it only when
you want the log-probability comparison and accept that.

## Verifying

Benchmark → **llama.cpp replay check**: choose a receipt, CUDA or Vulkan, and run. The job checks the model file (SHA-256 or size),
rebuilds the prompt tokens with a temporary CPU-only llama-server of the same build when the receipt has none, emulates the server's
slot and prompt-cache state from the earlier receipts of the same server, recomputes the run, and compares. The result names the first
diverging step and the number of differing values; the recomputed run is also written as a debugger run log, so every step's
probabilities and alternatives can be inspected.

Receipts that the check cannot verify are refused with a reason (an unsupported build or sampler setting, a run that overlapped
another request without a server batch log, a missing earlier turn, …).

## What we need from other GPUs

To extend the replay to another GPU generation (for example A100 or H100), we need a few llama-server runs made on that GPU, with:

1. the conversation log above (or receipts from the recorder), with temperature 0 and a seed;
2. the exact llama-server command line, its build (`llama-server --version`) and the operating system;
3. the model file's name and SHA-256 (any supported GGUF from the table; Llama-3.1-8B-Instruct Q4_K_M or Qwen3-8B Q8_0 is ideal);
4. if several requests may have overlapped, the server started with `LLAMA_BATCH_DEBUG=1` and `-v --log-file <file>`, and that file.

Send them with the GPU model and driver version. They become the ground truth against which the kernels for that generation are built.
