갈래: 바꿈

- **요청 2 ~ 4 개를 함께 생성할 때 빨라질 수 있습니다.** 엔진이 커널을 고를 때 요청 2 개와 3 ~ 4 개를 따로 재서 고릅니다(전에는 요청 8 개로
  잰 고름을 2 ~ 8 개 모두에 썼습니다). H100(MIG 조각)에서 Qwen3.8-27B Q4_K_M 의 동시 2 요청이 정밀 51.8 → 62.9, 빠른 60.6 → 78.9 tok/s 가
  됐습니다. 요청 하나와 5 개 이상은 그대로이고, 출력 비트도 그대로입니다. 처음 한 번은 커널을 다시 재므로 서버가 뜨는 데 몇 초 더 걸립니다
  — 자세한 것은 [구조/qwen35/기록.md](../../구조/qwen35/기록.md) 의 '속도 S11'.

---

- **Faster generation with 2–4 concurrent requests.** The engine now measures and picks its kernels separately for 2 requests and for 3–4
  requests (before, the choice measured at 8 requests served 2–8). On an H100 (MIG slice), Qwen3.8-27B Q4_K_M with 2 concurrent requests went
  from 51.8 to 62.9 (exact) and from 60.6 to 78.9 (fast) tok/s. One request and 5 or more are unchanged, and so are the output bits. Kernels
  are measured once more on the first start, so that start takes a few seconds longer — details in section 'S11' of
  [구조/qwen35/기록.md](../../구조/qwen35/기록.md).
