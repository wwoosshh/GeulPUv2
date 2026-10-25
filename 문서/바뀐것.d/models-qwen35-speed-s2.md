갈래: 바꿈

- **Qwen3.5 의 프롬프트가 빨라졌다**: Qwen3.5 · Qwen3.6 MoE 의 프롬프트(덩이) 어텐션을 int8 텐서 코어로 셈한다 — 프롬프트 2,048 의
  어텐션이 28 배 짧고, 프롬프트 2,048 의 시간이 4B 3.9 ~ 4.6 배 · 9B 2.8 ~ 3.4 배 짧다(빠른 모드 4B 1,207 → 263 ms · 9B 1,332 → 388 ms). 출력은 비트까지
  그대로다(같은 계약 · 같은 차례). 값 캐시의 바이트 배치가 바뀌었다(KV 해시가 다르다 — 값은 같다).

---

- **Faster prompts for Qwen3.5.** The prompt (chunk) attention of Qwen3.5 and Qwen3.6 MoE now runs on int8 tensor cores — 28 times
  shorter attention at a 2,048-token prompt, so such prompts are 3.9–4.6× faster on the 4B and 2.8–3.4× on the 9B (fast mode: 4B 1,207 →
  263 ms, 9B 1,332 → 388 ms). The output is bit-identical (same contract, same order); the byte layout of the value cache changed (KV
  hashes differ — the values do not).
