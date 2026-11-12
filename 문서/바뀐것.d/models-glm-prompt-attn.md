갈래: 바꿈

- **GLM-4.7-Flash · Kanana-2 의 긴 프롬프트 처리가 빨라졌다**: 프롬프트 어텐션 커널이 질의와 지수를 공유 메모리에 한 번만 읽고, 워프마다
  같은 소프트맥스를 되풀이하지 않는다. H100 NVL 의 MIG 인스턴스에서 30,077 토큰 프롬프트가 GLM-4.7-Flash 는 166.5 → 79.6 초,
  Kanana-2 30B-A3B 는 272.2 → 120.7 초가 되었다. 출력은 비트까지 그대로다. 기록:
  [구조/deepseek2/기록.md](../../구조/deepseek2/기록.md).

---

- **Long prompts run faster on GLM-4.7-Flash and Kanana-2.** The prompt attention kernel now reads the query and the exponent scales
  into shared memory once and no longer repeats the same softmax in every warp. On an H100 NVL MIG instance a 30,077-token prompt takes
  79.6 s instead of 166.5 s on GLM-4.7-Flash and 120.7 s instead of 272.2 s on Kanana-2 30B-A3B. Outputs are bit-identical. Record:
  [구조/deepseek2/기록.md](../../구조/deepseek2/기록.md).
