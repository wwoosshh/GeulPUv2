갈래: 고침

- **데이터 줄의 max_tokens**: 모델 평가에서 데이터 줄의 max_tokens 는 실행 설정보다 앞서는데, 결과 줄 · 재현 정보에는 실행 설정만 보였다(#201).
  이제 결과 줄이 '답 512토큰(데이터 줄의 값 600개)'처럼, 재현 정보가 데이터 줄의 값을 쓴 대화의 수를 함께 보인다. GeulPU --help 에 남아 있던
  이슈 번호도 뺐다.

---

- **max_tokens on data lines.** In Model evaluation, a data line's max_tokens takes precedence over the run setting, but the result row
  and the reproduction info showed only the run setting (#201). The result row now reads like 'answers 512 tokens (600 from the data
  lines)' and the reproduction info shows how many conversations used the data line's value. An issue number left in GeulPU --help is
  gone too.
