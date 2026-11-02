갈래: 더함

- **모델 평가의 바탕(API)**: 앱에 작업 공간(모델 하나의 GGUF 버전들과 기준 모델) · 평가 작업 · 결과 · 전후 비교를 더했다(`/api/eval` …).
  기준 모델의 그리디 답을 두 모델이 echo 로 점수 매겨 KL divergence(기준 모델의 상위 64 개 + 나머지 확률) · top-1 일치율 · Δp · PPL 비를
  내고, 저마다의 채팅 템플릿으로 낸 답이 달라진 대화 · 처음 갈라진 스텝 · 그 스텝에서 기준 모델의 마진을 낸다. 텍스트 파일은
  llama-perplexity 와 같은 위치(문맥 512 의 뒤쪽 절반)로 점수 매긴다. 내장 데이터 '기본 대화 60'(GeulPU 가 쓴 문항 — 여섯 분야). 화면은
  다음 단계에 붙는다.

---

- **Model evaluation backend (API).** The app gains workspaces (the GGUF versions of one model plus a reference model), evaluation jobs,
  results and before/after comparison (`/api/eval` …). Both models echo-score the reference model's greedy answers, giving KL divergence
  (the reference's top 64 tokens plus the remaining probability), top-1 agreement, Δp and the PPL ratio; each model also answers with its
  own chat template, giving the conversations whose answers changed, the first divergent step and the reference margin at that step.
  Text files are scored at the same positions as llama-perplexity (the second half of each 512-token chunk). Built-in data: 'basic 60'
  (prompts written for GeulPU, six categories). The screens come in the next step.
