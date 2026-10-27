갈래: 더함

- **리눅스 · 데이터센터 GPU 에서의 기록**: 0.12.1 을 Ubuntu 22.04 와 H100 NVL 의 MIG 인스턴스(3g.47gb)에서 소스로 빌드해 돌렸다 —
  글(geul) 컴파일러와 파이썬 3 만으로 빌드되고, 커널 모듈이 Windows 와 바이트까지 같고, 대화 24 개의 생성 토큰이 Windows 의 RTX 4070 Ti
  와 모두 같았다. 같은 HTTP 요청으로 llama.cpp b11900 · vLLM 0.32.0 과 비교한 속도와 다시 측정하는 스크립트는 문서/H100.md · 재기/H100/
  (리눅스 빌드는 이제 WSL2 와 네이티브 리눅스에서 시험한 것이다).

---

- **A record on Linux and a datacenter GPU.** 0.12.1 was built from source and run on Ubuntu 22.04 with an H100 NVL MIG instance
  (3g.47gb): it builds with the geul compiler and Python 3 alone, its kernel modules are byte-identical to Windows, and the generated
  tokens of 24 conversations were all identical to an RTX 4070 Ti on Windows. The speed against llama.cpp b11900 and vLLM 0.32.0 over
  the same HTTP requests, and scripts to measure again, are in docs/h100.md and 재기/H100/ (the Linux build is now tested under WSL2
  and on native Linux).
