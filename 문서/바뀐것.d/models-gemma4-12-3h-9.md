갈래: 바꿈

- **Gemma 4 가 같은 GPU 에서 훨씬 긴 문맥을 쓴다**: 슬라이딩 윈도 층의 KV 캐시를 창만큼의 고리 버퍼로 두고, 공통 KV 캐시에는 전체 어텐션
  층만 그 층의 폭으로 둔다(출력 비트는 그대로). 12B(RTX 4070 Ti 12 GB · 슬롯 하나)의 문맥이 7,360 에서 118,144 토큰으로 — 위치마다의 KV
  캐시가 약 442 KB 에서 18 KB 로 준다(윈도 층의 고리는 슬롯마다 약 0.57 GB 로 정해져 있다). 서버의 슬롯 캐시는 고리가 아직 가진 만큼(덩이
  512 면 513 토큰)까지 되돌려 이어 쓴다.

---

- **Gemma 4 fits a much longer context on the same GPU.** The sliding-window layers keep their KV cache in a ring buffer the size of
  the window, and the shared KV cache holds only the global-attention layers at their own width (output bits unchanged). On 12B
  (RTX 4070 Ti 12 GB, one slot) the context grows from 7,360 to 118,144 tokens — the KV cache per position drops from about 442 KB to
  18 KB (the window layers' ring is a fixed 0.57 GB per slot). The server's slot cache can still roll back and reuse as far as the
  ring holds (513 tokens with 512-row chunks).
