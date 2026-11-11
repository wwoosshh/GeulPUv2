갈래: 고침

- **GLM-4.7-Flash · Kanana-2 를 긴 문맥으로 띄워도 첫 프롬프트에서 서버가 끝나지 않는다**: deepseek2 구조의 모델을 문맥 32,768 로 띄우면
  첫 프롬프트에서 "계획의 띄움이 너무 많다" 를 내고 서버가 끝났다. 프롬프트 어텐션은 문맥이 길수록 쿼리 행을 작은 블록으로 나눠 커널을
  더 많이 실행하는데, 엔진이 미리 기록해 두는 커널 실행 목록의 크기가 그 수를 세지 않았다. 이제 블록 수만큼 크게 잡는다. 출력 비트는
  그대로다.

---

- **GLM-4.7-Flash and Kanana-2 no longer stop on their first prompt with a long context.** A deepseek2-architecture model started
  with a 32,768-token context exited with "계획의 띄움이 너무 많다" on its first prompt. Prompt attention splits the query rows into
  smaller blocks as the context grows and so launches more kernels, but the size of the engine's prerecorded kernel launch list did
  not count them; it now grows with the number of blocks. Output bits are unchanged.
