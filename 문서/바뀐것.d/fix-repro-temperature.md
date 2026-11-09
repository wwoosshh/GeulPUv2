갈래: 고침

- **요청에 온도가 없던 실행의 재현 파일**: 재현 파일이 이런 실행을 온도 0 으로 적어, 엔진이 온도 0.8 로 샘플링한 답을 그리디 디코딩으로 다시
  생성하고 확인이 다르다고 했습니다. 이제 실제로 쓴 온도와 top_k · top_p · min_p · 시드를 적습니다 — 이전에 기록한 실행도 그 값으로
  ([사용성시험.md](../사용성시험.md) §5).

---

- **Reproduction files of runs whose request had no temperature.** The file wrote such a run as temperature 0, so the check
  regenerated with greedy decoding an answer the engine had sampled at temperature 0.8 and reported a difference. The file now writes
  the temperature, top_k, top_p, min_p and seed actually used — also for runs recorded earlier ([사용성시험.md](../사용성시험.md) §5).
