갈래: 더함

- **Jacobian lens 를 GPU 없이**: `도구/repro_check.py --layer-lens 5,12 --jlens <모델>.jlens.safetensors` 가 같은 스텝의 행을 원본 모델에서
  맞춘 J_l 로 끝 층의 바탕에 옮겨 읽는다(Anthropic 의 Jacobian lens — 계약 5.4). `도구/jlens_fit.py` 가 원본(Hugging Face) 모델에서 jlens 로
  J 파일을 맞춰 쓴다. Qwen3-0.6B 에서 BF16 으로 맞춘 J 가 GGUF Q8_0 에 옮겨 감을 쟀다(층 9 ~ 26 에서 원본과 1위 일치 93.9 % 이상).

---

- **The Jacobian lens without a GPU.** `도구/repro_check.py --layer-lens 5,12 --jlens <model>.jlens.safetensors` moves the same steps'
  rows into the final layer's basis with J_l fitted on the original model and reads them (Anthropic's Jacobian lens — contract 5.4).
  `도구/jlens_fit.py` fits a J file with jlens on the original (Hugging Face) model. A J fitted on Qwen3-0.6B in BF16 was measured to
  carry over to the GGUF Q8_0 file (top-1 agreement with the original of at least 93.9 % in layers 9–26).
