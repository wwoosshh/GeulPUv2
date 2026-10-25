갈래: 바꿈

- **Qwen3.5 의 GDN 순환이 3 배 빨라졌다**: Qwen3.5 · Qwen3.6 MoE 의 선형 어텐션(GDN) 순환을 덩이(토큰 8)마다 공유 메모리로 미리 읽고, 스레드가 열
  둘을 맡아 셈한다 — 프롬프트 2,048 의 GDN 순환이 64 → 20 ms, 프롬프트 2,048 의 시간이 4B 13 ~ 16 % · 9B 7 ~ 10 % 짧다(빠른 모드 4B 261 → 219 ms ·
  9B 386 → 348 ms). 출력은 비트까지 그대로다(같은 계약 · 같은 차례).

---

- **3× faster GDN recurrence for Qwen3.5.** The linear-attention (GDN) recurrence of Qwen3.5 and Qwen3.6 MoE now stages each 8-token chunk
  in shared memory ahead of use and gives each thread two columns — the recurrence at a 2,048-token prompt drops from 64 to 20 ms, so such
  prompts are 13–16 % faster on the 4B and 7–10 % on the 9B (fast mode: 4B 261 → 219 ms, 9B 386 → 348 ms). The output is bit-identical
  (same contract, same order).
