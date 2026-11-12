갈래: 바꿈

- **Gemma 4 · Gemma 3 의 긴 프롬프트 처리가 빨라졌다**: 프롬프트 어텐션 커널이 질의와 지수를 공유 메모리에 한 번만 읽고, 워프마다 같은
  소프트맥스를 되풀이하지 않으며, 슬라이딩 윈도 층에서는 행이 보는 그룹만 실행한다. H100 NVL 의 MIG 인스턴스에서 30,077 토큰 프롬프트가
  Gemma 4 31B 는 100.0 → 59.4 초, 26B-A4B 는 21.8 → 12.2 초, Gemma 3 27B 는 48.9 → 38.2 초가 되었다. 출력은 비트까지 그대로다. 기록:
  [구조/gemma4/기록.md](../../구조/gemma4/기록.md).

---

- **Long prompts run faster on Gemma 4 and Gemma 3.** The prompt attention kernel now reads the query and the exponent scales into
  shared memory once, no longer repeats the same softmax in every warp, and on sliding-window layers launches only the groups the rows
  can see. On an H100 NVL MIG instance a 30,077-token prompt takes 59.4 s instead of 100.0 s on Gemma 4 31B, 12.2 s instead of 21.8 s
  on 26B-A4B and 38.2 s instead of 48.9 s on Gemma 3 27B. Outputs are bit-identical. Record:
  [구조/gemma4/기록.md](../../구조/gemma4/기록.md).
