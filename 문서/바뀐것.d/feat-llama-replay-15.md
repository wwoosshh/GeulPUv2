갈래: 더함

- **llama.cpp 재현 검증 — SPM 토크나이저(Mistral)**: 검증이 걸음마다 토큰의 바이트를 지을 때 SPM 어휘(`▁` 는 빈칸, `<0xAB>` 는 그 바이트)도
  셈합니다. Mistral-7B-Instruct-v0.3 Q4_K_M 의 영수증 10개가 CUDA 에서 비트까지 같았습니다(한국어처럼 글자가 바이트 토큰 여럿에 걸친
  걸음 포함 — [라마재현.md](../라마재현.md) 15단계).

---

- **llama.cpp replay check — SPM tokenizer (Mistral).** When the check builds each step's token bytes it now also handles SPM
  vocabularies (`▁` is a space, `<0xAB>` is that byte). Ten Mistral-7B-Instruct-v0.3 Q4_K_M receipts matched bit for bit on
  CUDA, including steps where a character spans several byte tokens, as in Korean ([문서/라마재현.md](../라마재현.md),
  stage 15).
