갈래: 더함

- **앱의 모델 목록에 SK텔레콤의 한국어 모델 A.X 4.0 Light(7B · Q4_K_M)**(0.24.0 24-1): 모델 검증을 모두 통과했다 — 다섯 실행 경로가 비트까지
  같고 모든 행 로짓이 CPU 참조와 같으며(긴 문맥은 모델의 최대 문맥 길이 16,384 까지), 품질은 llama.cpp 대비 KL 비 1.03 · 1위 일치 98.3 %,
  채팅 템플릿 · 토크나이저가 jinja2 · llama.cpp 와 같고, CUDA · Vulkan 에서 채팅이, CUDA 에서 도구 호출이 된다. 같은 단계에서 확인한 KT 의
  Mi:dm 2.0 · 카카오의 Kanana 1.5 는 아직 넣지 않았다(까닭은 [0.24.0.md](../0.24.0.md) 의 24-1 결과).

---

- **SK Telecom's Korean model A.X 4.0 Light (7B, Q4_K_M) in the app's model list.** It passed every model check: five execution paths
  bit-identical and all-row logits equal to the CPU reference (long context up to the model's maximum context length of 16,384), quality
  against llama.cpp with KL ratio 1.03 and 98.3 % top-1 agreement, chat template and tokenizer equal to jinja2 and llama.cpp, chat on
  CUDA and Vulkan, and tool calls on CUDA. KT's Mi:dm 2.0 and Kakao's Kanana 1.5, checked in the same stage, are not listed yet.
