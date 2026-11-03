갈래: 고침

- **원래 가중치의 형식을 config.json 이 아니라 저장된 텐서에서 읽음**: 'GGUF 만들기'의 변환 형식 기본값과 카드의 안내 · 작업 공간의
  `original_dtype` · 원래 모델과 견주기의 반올림 알림이 원래 모델의 형식을 config.json 의 `torch_dtype` · `dtype` 으로만 정해, 가중치는
  float32 인데 config 에 bfloat16 이 적힌 폴더(학습 루프가 float32 주 가중치로 저장하거나 DeepSpeed `zero_to_fp32.py` 의 가중치 옆에 학습
  때의 config 를 둔 경우)를 BF16 으로 변환하고 "원래 값 그대로"라고 했다. 이제 safetensors 의 머리(가중치는 읽지 않음)에서 2 차원 이상
  텐서의 형식 가운데 값이 가장 많은 것을 쓰고(llama.cpp 의 `--outtype auto` 처럼 config 가 아니라 텐서를 봄), 머리를 읽지 못할 때만
  config.json 을 본다. 둘이 다르면 카드와 견주기의 안내에 config.json 의 값도 적고, 작업 공간 목록(`GET /api/eval`)과 견주기 결과에
  `original_config_dtype` 을 남긴다. 이미 견준 결과는 그때 적은 형식 그대로다(다시 견주면 바뀐다).

---

- **The original weights' type now comes from the stored tensors, not config.json.** Make GGUF's default conversion type and
  its note, the workspace's `original_dtype` and the rounding notice of Compare with the original model took the original
  model's type from config.json's `torch_dtype` · `dtype` alone, so a folder whose weights are float32 but whose config says
  bfloat16 (a training loop that saves its float32 master weights, or DeepSpeed `zero_to_fp32.py` weights next to the config
  from training) was converted to BF16 and called "the original values exactly". The type now comes from the safetensors
  header (the weights are not read) — the type holding the most values among tensors of two or more dimensions (like llama.cpp's
  `--outtype auto`, the tensors rather than the config) — and config.json only when the header cannot be read. When the two
  differ, the notes on the card and the comparison also give config.json's value, and the workspace list (`GET /api/eval`) and
  the comparison result keep `original_config_dtype`. Comparisons already made keep the type they recorded (comparing again
  updates it).
