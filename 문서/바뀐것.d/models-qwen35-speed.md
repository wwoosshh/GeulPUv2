갈래: 바꿈

- **Qwen3.5 의 생성이 빨라졌다**: Qwen3.5 · Qwen3.6 MoE 의 생성 걸음에서 헤드 256 어텐션을 키 조각(64)마다 나눠 GPU 전체로 셈한다 —
  층마다의 어텐션이 깊이 2,048 에서 16 배 짧고, 4B 의 생성이 32 ~ 53 %(문맥이 깊을수록 더) · 9B 가 15 ~ 35 % 빠르다. 출력은 비트까지
  그대로다(같은 계약 · 같은 차례).

---

- **Faster generation for Qwen3.5.** For Qwen3.5 and Qwen3.6 MoE, the head-256 attention of a generation step is now split into
  64-key tiles spread over the whole GPU — 16 times shorter per layer at depth 2,048, so generation is 32–53 % faster on the 4B (more at
  deeper context) and 15–35 % faster on the 9B. The output is bit-identical (same contract, same order).
