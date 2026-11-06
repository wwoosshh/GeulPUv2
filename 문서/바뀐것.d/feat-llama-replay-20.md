갈래: 더함

- **llama.cpp 재현 검증 — Ollama 가 띄운 llama-server**: Ollama(v0.40.14)가 자기 빌드로 띄운 llama-server(빌드 번호 대신 커밋으로 알리는
  b11963)의 영수증도 받습니다. Ollama 의 llama-server 에 기록기를 붙여 남긴 영수증 9개가 비트까지 같았습니다 — 이 PC 에서 Ollama 는 CUDA
  판을 실행 때 드라이버가 짓는 PTX 로 돌렸는데, 그 값도 공식 빌드와 같았습니다([라마재현.md](../라마재현.md) 20단계).

---

- **llama.cpp replay check — the llama-server that Ollama launches.** Receipts from the llama-server that Ollama (v0.40.14)
  builds and launches are now accepted (it reports its build as a commit — b11963 — rather than a build number). Nine
  receipts recorded by attaching the recorder to Ollama's llama-server matched bit for bit; on this PC Ollama ran its CUDA
  kernels from PTX compiled by the driver at run time, and the values still matched the official build
  ([문서/라마재현.md](../라마재현.md), stage 20).
