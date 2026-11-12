# 연구: 가중치에 둔 기억 색인

[English](README.md)

에이전트가 장기 기억의 색인을 어디에 두는가에 대한 연구다. 기억의 내용은 파일에 그대로 두고, 색인만 — 어떤 기억이 있고 각각이 어떤
상황에서 필요한지 — 모델 가중치에 붙이는 LoRA 어댑터에 학습시킨다.

**상태: 검증 중인 아이디어다. 여기 있는 것은 GeulPU 의 기능이 아니다.** 아이디어가 버티면 GeulPU 에 보통의 문맥 기억 옆의 실험적 선택
사항으로 넣어 볼 수 있다.

| 문서 | English | 한국어 |
|---|---|---|
| 제안 — 문제, 설계, 선행 연구, 보이지 않은 것 | [proposal.md](proposal.md) | [proposal.ko.md](proposal.ko.md) |
| 첫 시험의 근거 — 기억 200개, 4B 모델, 미리 정한 기준 | [evidence.md](evidence.md) | [evidence.ko.md](evidence.ko.md) |

Zenodo 에 날짜와 함께 보관했다: [10.5281/zenodo.24184984](https://doi.org/10.5281/zenodo.24184984) (초안 1, 2026-11-13). 모든 판:
[10.5281/zenodo.24184983](https://doi.org/10.5281/zenodo.24184983).

이 연구는 GeulPU 밖의 PyTorch 에서 돌렸으므로, GeulPU 로 돌린 실험을 담는 [experiments/](../../../experiments/) 에는 넣지 않았다. 자료 ·
스크립트 · 뒤의 결과(더 큰 모델에서의 비교가 진행 중이다)는 여기와 Zenodo 기록에 더한다.
