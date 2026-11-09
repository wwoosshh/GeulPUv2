# 계획 — Short title of the experiment

판정 기준과 이 계획은 어떤 실행보다 먼저 적는다 — 첫 실행 뒤에는 고칠 수 없다. 실험 폴더를 git 저장소에 두면 먼저 커밋해 이력으로도 남긴다.

## 연구 질문

질문 하나 — `experiment.json` 의 `question` 과 같다.

## 가설

무엇을 기대하는지와 그 까닭.

## 방법

모델(파일 · SHA-256 · 출처), GeulPU 버전과 계산 방식, 실행과 그 설정, 실행이 쓰는 GeulPU 기능(예: 텍스트 생성 · 분기 · 로짓 렌즈 — CLI 의
이름은 `run.complete` · `run.branch` · `run.lens`).

## 판정 기준

`experiment.json` 의 `criteria` 와 같은 목록 — 무엇이 성립해야 하는가, 되도록 수로.

## 하지 않는 것

이 실험이 보이려 하지 않는 것.
