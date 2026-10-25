갈래: 바꿈

- **Qwen3.5 의 긴 문맥 생성이 빨라졌다**: 위치 16,384 부터 Qwen3.5 · Qwen3.6 MoE 의 생성 걸음 어텐션이 KV 헤드를 함께 쓰는 질의 헤드의
  블록을 이웃하게 띄우고 다음 조각을 L2 로 미리 읽는다 — 프롬프트 32,768 뒤의 생성이 4B 7.5 % · 9B 4.5 % 빠르다(빠른 모드 4B 78.0 → 83.8 ·
  9B 54.3 → 56.8 토큰/초). 그보다 짧은 문맥은 그대로이고, 출력은 비트까지 그대로다(같은 셈 · 같은 차례).

---

- **Faster long-context generation for Qwen3.5.** From position 16,384 the decode attention of Qwen3.5 and Qwen3.6 MoE launches the
  query heads that share a KV head as neighbouring blocks and prefetches the next chunk into L2 — generation after a 32,768-token prompt
  is 7.5 % faster on the 4B and 4.5 % on the 9B (fast mode: 4B 78.0 → 83.8, 9B 54.3 → 56.8 tokens/s). Shorter contexts are unchanged,
  and the output is bit-identical (same arithmetic, same order).
