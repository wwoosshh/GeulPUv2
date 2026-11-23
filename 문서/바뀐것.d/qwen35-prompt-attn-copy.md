갈래: 바꿈

- **긴 프롬프트에서 Qwen3.5 계열의 프롬프트 처리가 빨라집니다.** 프롬프트 어텐션이 키 · 값을 판으로 옮길 때 주소를 덜 셈합니다(모든 카드).
  계산 능력 8.0 이상인 카드에서는 키 · 값을 사분씩 버퍼 둘에 번갈아 올려, 앞 사분을 셈하는 동안 다음 사분이 옵니다. 출력 비트는 그대로입니다
  — 자세한 것은 [구조/qwen35/기록.md](../../구조/qwen35/기록.md) 의 '속도 S9'.

---

- **Faster Qwen3.5-family prompt processing for long prompts.** Prompt attention spends fewer instructions on the addresses of its key/value copies
  (all cards). On cards with compute capability 8.0 and up it also loads keys and values a quarter at a time into two alternating buffers, so the
  next quarter arrives while the previous one is computed. The output bits are unchanged — details in section 'S9' of
  [구조/qwen35/기록.md](../../구조/qwen35/기록.md).
