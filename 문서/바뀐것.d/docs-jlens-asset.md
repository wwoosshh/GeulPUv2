갈래: 더함

- **Qwen3-0.6B 의 J 파일(Jacobian lens)을 릴리스에 붙인다**: `Qwen3-0.6B-Q8_0.jlens.safetensors`(56,626,912 바이트, SHA-256
  `20f46104ff65671a8c03ee58c384fe2f01cc2a0b367d8e11955a212329ae4d38`). Qwen 공식 `Qwen3-0.6B-Q8_0.gguf` 옆에 이 이름 그대로 두고 모델을
  불러오면 연구 패널의 층 렌즈가 Jacobian lens 도 읽는다(같은 모델의 다른 양자화 파일이면 그 GGUF 이름에 맞춰 바꾼다). 맞춘 기록(파일의
  `__metadata__` 에도 있다): 원본 Qwen/Qwen3-0.6B(BF16, 리비전 c1899de289a04d12100db370d81485cdf75e47ca), jlens(anthropics/jacobian-lens
  581d398613e5602a5af361e1c34d3a92ea82ba8e)를 `도구/jlens_fit.py` 로, torch 2.14.1+cu130 · transformers 5.20.2, 프롬프트는 wikitext-2 test
  (wiki.test.raw, SHA-256 173c87a5…)의 앞 절반에서 문단을 이어 1,200 자 이상인 100 개(앞 128 토큰), 원천 층 0 ~ 26 → 목표 층 27, 앞 16
  자리 뺌, RTX 4070 Ti 에서 1,019 초. 견준 파일: Qwen 공식 Q8_0(SHA-256 9465e63a…) — 층 9 ~ 26 에서 원본(BF16)과 J-lens 1위 일치가
  93.8 % 이상이고 logit lens 의 일치보다 1.7 %p 넘게 낮지 않아, 미리 정한 기준으로 '옮겨 감'(문서/층렌즈.md 의 2c). 낮은 비트의 파일
  (Q4_K_M 등)에서는 층마다의 1위가 원본과 덜 맞는다.

---

- **The release carries a J file (Jacobian lens) for Qwen3-0.6B.** `Qwen3-0.6B-Q8_0.jlens.safetensors` (56,626,912 bytes, SHA-256
  `20f46104ff65671a8c03ee58c384fe2f01cc2a0b367d8e11955a212329ae4d38`). Put it, under this name, next to Qwen's official
  `Qwen3-0.6B-Q8_0.gguf` and load the model: the research panel's layer lens then reads the Jacobian lens as well (for another quantized
  file of the same model, rename it after that GGUF). Fit record (also in the file's `__metadata__`): original Qwen/Qwen3-0.6B (BF16,
  revision c1899de289a04d12100db370d81485cdf75e47ca), jlens (anthropics/jacobian-lens 581d398613e5602a5af361e1c34d3a92ea82ba8e) through
  `도구/jlens_fit.py`, torch 2.14.1+cu130 · transformers 5.20.2, 100 prompts from the first half of the wikitext-2 test split
  (wiki.test.raw, SHA-256 173c87a5…, paragraphs joined to at least 1,200 characters, first 128 tokens), source layers 0–26 → target layer
  27, first 16 positions skipped, 1,019 s on an RTX 4070 Ti. Checked file: Qwen's official Q8_0 (SHA-256 9465e63a…) — in layers 9–26 the
  J-lens top-1 agreement with the original (BF16) is at least 93.8 % and at most 1.7 points below the logit lens's, which carries over
  under the criterion set before measuring (2c in 문서/층렌즈.md). On low-bit files (Q4_K_M and the like) the per-layer top-1 agrees less
  with the original.
