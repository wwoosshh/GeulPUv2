갈래: 고침

- **엔진이 받지 않는 모델로 평가해도 앱이 꺼지지 않음**: 엔진이 받지 않는 GGUF(지원 표 밖의 크기 · RoPE 늘이기가 든 파일 등)를
  기준 모델이나 버전으로 평가하면 앱 전체가 '잘못된 메모리 접근'으로 꺼지던 것을 고쳤다(#213). 이제 평가 요청이 그 까닭과 함께
  거절되고, '평가 실행'에서는 그런 버전을 고를 수 없으며 기준 모델을 받지 않으면 그 까닭이 보인다. 비교용 모델을 받지 않으면 배운 것 ·
  잊은 것만 세지 않고 평가는 이어간다.

---

- **Evaluating a model the engine does not support no longer crashes the app.** Evaluating a GGUF the engine does not
  support (a size outside the support table, a file with RoPE scaling, …) as the reference model or a version closed the
  whole app with an access violation (#213). The evaluation request is now refused with the reason, such versions cannot be
  picked in Run evaluation, and an unsupported reference model shows why. An unsupported baseline model only skips the
  learned / forgotten counts; the evaluation goes on.
