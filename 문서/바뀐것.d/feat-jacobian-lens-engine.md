갈래: 더함

- **연구 패널의 Jacobian lens**: 모델 파일 옆에 `<이름>.jlens.safetensors`(`도구/jlens_fit.py` 로 원본 모델에서 맞춘 J 파일)를 두고 모델을
  불러오면, 걸음 상세의 '층 렌즈'가 logit lens 와 Jacobian lens 를 함께 읽어 바꿔 볼 수 있다(계약 5.4). 앞 층에서도 그 자리가 무엇을 향하는지
  logit lens 보다 일찍 보인다. API 는 `/v1/completions` 의 echo 에 `geulpu.layer_lens: {"jacobian": true}` — 응답 `geulpu.jacobian_lens`. 재현
  파일에 셈한 Jacobian lens 와 J 파일의 SHA-256 이 들어가고, `repro_check --jlens` 가 GPU 없이 비트까지 견준다.

---

- **The Jacobian lens in the research panel.** Put `<name>.jlens.safetensors` (a J file fitted on the original model with
  `도구/jlens_fit.py`) next to the model file and load the model; the step detail's layer lens then reads both the logit lens and the
  Jacobian lens and switches between them (contract 5.4). What the position points at shows earlier in the layers than with the logit
  lens. API: `geulpu.layer_lens: {"jacobian": true}` on a `/v1/completions` echo — response `geulpu.jacobian_lens`. Reproduction files
  carry the computed Jacobian lens and the J file's SHA-256, and `repro_check --jlens` checks them bit for bit without a GPU.
