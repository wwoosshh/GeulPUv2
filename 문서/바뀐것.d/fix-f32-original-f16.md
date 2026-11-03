갈래: 고침

- **float32 로 저장한 원래 모델은 F16 으로 변환**: 전체 미세조정처럼 가중치를 float32 로 저장한 모델을 'GGUF 만들기'의 기본 변환 형식
  BF16 으로 바꾸면, BF16 으로 반올림한 것만으로 기준 모델이 원래 모델과 KL 3.6 × 10⁻⁴ 만큼 달라(F16 이면 2.0 × 10⁻⁶) Q8_0 · Q6_K 같은
  형식의 비교가 학습한 모델이 아니라 반올림한 모델을 기준으로 재어졌는데, 화면은 이를 짚지 않았다(#233). 이제 변환 형식의 기본은 원래
  가중치의 형식(config.json 의 torch_dtype · dtype)을 따른다 — bfloat16 은 BF16, float32 · float16 은 F16(llama.cpp 의
  `--outtype auto` 와 같은 고름). 'GGUF 만들기' 카드는 원래 가중치의 형식과 그 까닭을 보이고, API 에서 `outtype` 을 빼도 같게 고른다.
  '원래 모델과 견주기'는 float32 · float16 원래 모델을 BF16 기준과 견준 결과면 반올림만으로 KL 이 10⁻⁴ 대가 될 수 있다고 알린다
  (`original_dtype` 을 결과에 남긴다).

---

- **Original models saved in float32 convert to F16.** Converting a model whose weights are saved in float32 (common for full
  fine-tuning) with Make GGUF's default conversion type BF16 made the reference model differ from the original by KL 3.6 × 10⁻⁴
  from BF16 rounding alone (F16: 2.0 × 10⁻⁶), so Q8_0 · Q6_K comparisons were measured against the rounded model rather than
  the trained one, and the screen did not point it out (#233). The default conversion type now follows the original weights'
  type (torch_dtype · dtype in config.json): bfloat16 → BF16, float32 · float16 → F16 (the same choice as llama.cpp's
  `--outtype auto`). The Make GGUF card shows the original weights' type and why, and the API picks the same way when `outtype`
  is left out. Compare with the original model notes that a float32 · float16 original compared with a BF16 reference can reach
  a KL in the 10⁻⁴ range from rounding alone (the result keeps `original_dtype`).
