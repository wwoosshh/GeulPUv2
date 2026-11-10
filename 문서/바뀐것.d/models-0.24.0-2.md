갈래: 더함

- **새 모델 구조 deepseek2 — zai-org 의 GLM-4.7-Flash(30B)와 카카오의 Kanana-2-30B-A3B Thinking(30B)이 실행되고 앱의 모델 목록에
  들었다**(0.24.0 24-2): MLA 어텐션(키 · 값을 저차원으로 압축한 어텐션)과 시그모이드 라우팅 MoE 를 계약에 더했다(계약 4.13). 그래픽
  메모리에 들지 않는 전문가 가중치는 RAM(고정 메모리)에 둔다. 아직 CUDA · 정밀 모드만 된다. 두 모델 모두 다섯 실행 경로가 비트까지
  같고 모든 행 로짓이 CPU 참조와 같으며, 채팅 템플릿 · 토크나이저가 jinja2 · llama.cpp 와 같고, 채팅과 도구 호출이 된다. 원격 서버의
  H100 에서도 출력이 이 PC(RTX 4070 Ti)와 비트까지 같았다. 정밀 모드의 손실은 fp64 기준으로 llama.cpp 의 약 0.13 배(GLM)다. 속도는
  아직 llama.cpp 보다 느리다(생성 약 2.5 ~ 2.9 배).

---

- **New model architecture deepseek2 — zai-org's GLM-4.7-Flash (30B) and Kakao's Kanana-2-30B-A3B Thinking (30B) run and are in
  the app's model list.** MLA attention (attention over low-rank compressed keys and values) and sigmoid-routed MoE join the
  contract (section 4.13). Expert weights that do not fit in graphics memory stay in RAM (pinned memory). For now only CUDA in
  exact mode. Both models have five bit-identical execution paths and all-row logits equal to the CPU reference, chat template and
  tokenizer equal to jinja2 and llama.cpp, and working chat and tool calls. On the remote H100 server the outputs were also
  bit-identical to this PC (RTX 4070 Ti). Exact mode loses about 0.13 times what llama.cpp loses against an fp64 reference (GLM).
  Speed is still behind llama.cpp (decode about 2.5–2.9 times slower).
