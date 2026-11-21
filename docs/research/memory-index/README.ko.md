# 연구: 가중치에 둔 기억 색인

[English](README.md)

에이전트가 장기 기억의 색인을 어디에 두는가에 대한 연구다. 기억의 내용은 파일에 그대로 두고, 색인만 — 어떤 기억이 있고 각각이 어떤
상황에서 필요한지 — 모델 가중치에 붙이는 LoRA 어댑터에 학습시킨다.

**상태: 검증했고 채택하지 않는다(2026-11-21). 여기 있는 것은 GeulPU 의 기능이 아니다.** 첫 시험의 합성 데이터에서 본 우위는 공개
벤치마크의 기억에서 재현되지 않았고, 이 설계를 GeulPU 에 넣어 볼 계획은 없다. 결론과 수치는 [results.ko.md](results.ko.md)에 있다.

| 문서 | English | 한국어 |
|---|---|---|
| 그 뒤의 결과 — 공개 벤치마크, 실제 대화, 결론 (2026-11-22) | [results.md](results.md) | [results.ko.md](results.ko.md) |
| 제안 — 문제, 설계, 선행 연구, 보이지 않은 것 | [proposal.md](proposal.md) | [proposal.ko.md](proposal.ko.md) |
| 첫 시험의 근거 — 기억 200개, 4B 모델, 미리 정한 기준 | [evidence.md](evidence.md) | [evidence.ko.md](evidence.ko.md) |

Zenodo 에 날짜와 함께 보관했다: [10.5281/zenodo.24184984](https://doi.org/10.5281/zenodo.24184984) (초안 1, 2026-11-13). 모든 판:
[10.5281/zenodo.24184983](https://doi.org/10.5281/zenodo.24184983).

제안서와 근거 문서는 2026-11-13 에 쓴 그대로 두고, 맨 위에 그 뒤의 결과를 가리키는 안내만 넣었다.

이 연구는 GeulPU 밖의 PyTorch 에서 돌렸으므로, GeulPU 로 돌린 실험을 담는 [experiments/](../../../experiments/) 에는 넣지 않았다. 데이터와
스크립트는 공개하지 않았다.
