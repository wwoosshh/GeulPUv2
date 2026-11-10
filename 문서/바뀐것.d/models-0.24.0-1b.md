갈래: 더함

- **앱의 모델 목록에 KT 의 Mi:dm 2.0 Base(11.5B · Q4_K_M)와 카카오의 Kanana 1.5 8B(Q4_K_M)**(0.24.0 24-1): 템플릿 해석기 · 홀수 어휘의
  정렬 · 마커가 텍스트인 도구 호출을 고친 뒤 모델 검증을 모두 통과했다 — 다섯 실행 경로가 비트까지 같고 모든 행 로짓이 CPU 참조와
  같으며, 품질은 llama.cpp 대비 KL 비 0.86(Mi:dm) · 0.71(Kanana), 채팅 템플릿 · 토크나이저가 jinja2 · llama.cpp 와 같고, CUDA · Vulkan
  에서 채팅이, CUDA 에서 도구 호출이 된다.

---

- **KT's Mi:dm 2.0 Base (11.5B, Q4_K_M) and Kakao's Kanana 1.5 8B (Q4_K_M) in the app's model list.** After the fixes to the
  template interpreter, odd-vocabulary alignment and plain-text tool-call markers, both pass every model check: five execution
  paths bit-identical and all-row logits equal to the CPU reference, quality against llama.cpp with KL ratio 0.86 (Mi:dm) and 0.71
  (Kanana), chat template and tokenizer equal to jinja2 and llama.cpp, chat on CUDA and Vulkan, and tool calls on CUDA.
