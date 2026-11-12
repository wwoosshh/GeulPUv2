갈래: 바꿈

- **Qwen3.5 · 3.6 · 3.8 의 긴 프롬프트 처리가 빨라졌다**: 프롬프트 어텐션 커널이 같은 점수를 두 번 계산하지 않고, 반복 본문이 명령 캐시에
  들어가는 크기가 되었다. H100 NVL 의 MIG 인스턴스에서 30,077 토큰 프롬프트가 Qwen3.8-27B 는 55.0 → 37.1 초, Qwen3.6-35B-A3B 는 18.3 →
  10.8 초가 되었다. 출력은 비트까지 그대로다. 기록: [구조/qwen35/기록.md](../../구조/qwen35/기록.md).

---

- **Long prompts run faster on Qwen3.5 · 3.6 · 3.8.** The prompt attention kernel no longer computes each score twice, and its loop
  body now fits in the instruction cache. On an H100 NVL MIG instance a 30,077-token prompt takes 37.1 s instead of 55.0 s on
  Qwen3.8-27B and 10.8 s instead of 18.3 s on Qwen3.6-35B-A3B. Outputs are bit-identical. Record:
  [구조/qwen35/기록.md](../../구조/qwen35/기록.md).
