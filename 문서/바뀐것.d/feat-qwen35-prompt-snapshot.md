갈래: 바꿈

- **Qwen3.5 계열이 프롬프트 캐시를 더 많이 이어 씀**: 추론 엔진이 프롬프트의 마지막 행 앞에서 GDN 상태의 스냅숏을 남긴다. 다음 요청의
  앞이 슬롯의 끝보다 먼저 갈려도(지난 답을 다르게 그린 다음 차례 · 같은 프롬프트를 다시 보냄) 그 위치부터 잇는다 — 전에는 처음부터 다시
  셈했다. 출력 비트는 그대로다. RTX 4070 Ti 에서 Qwen3.5-4B 의 3,000 토큰 프롬프트를 다시 보내면 첫 토큰이 351 → 9 ms, 지난 답 안에서 갈린
  다음 요청은 360 → 29 ms 다. 슬롯마다 GDN 상태 한 벌만큼(4B · 9B 약 53 MB, 27B 약 157 MB) GPU 메모리를 더 쓴다.

---

- **The Qwen3.5 family reuses more of the prompt cache.** The inference engine keeps a snapshot of the GDN state just before the
  last prompt row, so a request that diverges before the end of the slot (a next turn that renders the previous answer differently,
  or the same prompt sent again) resumes there instead of recomputing from token 0. Output bits are unchanged. On an RTX 4070 Ti,
  resending a 3,000-token Qwen3.5-4B prompt cuts the first token from 351 to 9 ms, and a next request that diverges inside the
  previous answer from 360 to 29 ms. Each slot uses one more GDN state of GPU memory (about 53 MB for 4B and 9B, 157 MB for 27B).
