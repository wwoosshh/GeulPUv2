갈래: 더함

- **llama.cpp 재현 검증 — llama.cpp b11963(Ollama v0.40.14 가 쓰는 빌드)**: 검증이 b12008 말고 b11963 으로 돈 llama-server 의
  영수증도 받습니다. 두 빌드는 받는 모델 · 기본 설정에서 셈이 같습니다 — 같은 요청의 응답이 토큰 · 로그 확률까지 같았습니다
  ([라마재현.md](../라마재현.md) 19단계).

---

- **llama.cpp replay check — llama.cpp b11963 (the build Ollama v0.40.14 uses).** The check now also accepts receipts from a
  llama-server built at b11963, besides b12008. The two builds compute the same for the supported models and default settings:
  the same requests gave identical tokens and log probabilities ([문서/라마재현.md](../라마재현.md), stage 19).
