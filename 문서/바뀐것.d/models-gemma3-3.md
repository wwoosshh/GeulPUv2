갈래: 더함

- **새 모델 구조 gemma3 — Google 의 Gemma 3 270M · 1B · 4B · 12B 가 실행되고 앱의 모델 목록에 들었다**(0.24.0): Gemma 4 의 블록에서
  값(v)을 정규화하지 않고, 점수 스케일이 1/√헤드 크기이며, 전체 어텐션 층의 RoPE 에 선형 스케일링이 있는 구조를 계약에 더했다(계약
  4.14). 텍스트만 쓰고(그림 입력은 아직 쓰지 않는다), 아직 CUDA · 정밀 모드만 된다. 4B · 12B 는 k-양자화(Q4_K_M 등)와 Q8_0 파일,
  270M · 1B 는 Q8_0 파일만 된다(이 두 크기의 k-양자화 이름의 파일에는 다른 형식이 섞여 있다 — 앱의 모델 찾기가 Q8_0 파일만 받게
  한다). 27B 는 아직 안 된다. 네 모델 모두 다섯 실행 경로가 비트까지 같고, 채팅 템플릿 · 토크나이저가 jinja2 · llama.cpp 와 같다.
  270M · 1B · 4B 는 전체 파일의 모든 행 로짓이 CPU 참조와 같고, 4B · 12B 는 원격 서버의 H100 에서도 출력이 이 PC(RTX 4070 Ti)와
  비트까지 같았다. 정밀 모드의 손실은 fp64 기준으로 llama.cpp 의 0.001 배보다 작다(4B). 속도는 아직 llama.cpp 보다 느리다(생성 약
  0.65 ~ 0.9 배). Gemma 이용 약관(Gemma Terms of Use)을 따르는 모델이다. 기록: [구조/gemma3/기록.md](../../구조/gemma3/기록.md).

---

- **New model architecture gemma3 — Google's Gemma 3 270M, 1B, 4B and 12B run and are in the app's model list.** The contract
  gains the Gemma 3 form of the Gemma 4 block (section 4.14): values are not normalized, the score scale is 1/√(head size), and the
  full-attention layers use linear RoPE scaling. Text only (image input is not used yet); for now only CUDA in exact mode. The 4B
  and 12B take k-quant (Q4_K_M and others) and Q8_0 files; the 270M and 1B take Q8_0 files only (their files with k-quant names
  mix in other formats — the app's model search offers only the Q8_0 file). The 27B does not run yet. All four have five
  bit-identical execution paths, and chat template and tokenizer equal to jinja2 and llama.cpp. The 270M, 1B and 4B have all-row
  logits of the full file equal to the CPU reference, and on the remote H100 server the outputs of the 4B and 12B were
  bit-identical to this PC (RTX 4070 Ti). Exact mode loses less than 0.001 times what llama.cpp loses against an fp64 reference
  (4B). Speed is still behind llama.cpp (decoding at about 0.65–0.9 times). The models are under the Gemma Terms of Use. Record:
  [구조/gemma3/기록.md](../../구조/gemma3/기록.md).
