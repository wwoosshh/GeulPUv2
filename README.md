# GeulPUv2 — 내 PC 의 그래픽카드로 돌리는 AI 채팅, 실사용 판

[GeulPU](https://github.com/wwoosshh/GeulPU)(연구 저장소 — 2026-10-16 연구 동결)의 엔진과 앱을 **실사용**(성능 · 안정성 · 편안함)
목적으로 옮긴 프로젝트다. 무엇을 옮기고 무엇을 두고 왔는지, 남은 결정은 [문서/이관범위.md](문서/이관범위.md)에, 판마다 바뀐 것은
[문서/바뀐것.md](문서/바뀐것.md)에 적는다. GPU 일(커널 · 엔진 · 서버 · 앱의 관리자)은 글(geul) 언어로 쓴다.

## 폴더

| 폴더 | 무엇 |
|---|---|
| [엔진/](엔진/) | 글(geul) 언어 엔진 — GGUF 읽기 · 가중치 · 토크나이저 셋 · 엔진 · 서버(CUDA `서버.gl` · Vulkan `서버V.gl`) · 장치 확인 · 자기 진단 · 터미널 실행기. `리눅스/` 는 리눅스 대상의 덧판 |
| [커널/](커널/) | 커널 생성기(`커널생성.gl` + `생성*.gl` — 모델 모양 · 형식 · 장치마다 커널 소스를 낸다) · 진단 커널 |
| [앱/](앱/) | `GeulPU.exe`(관리자 `관리자.gl`) · `web/` 화면 · `data/models.json` 모델 목록 · `꾸리기.ps1` 릴리스 꾸리기 |
| [계약/](계약/) | 정밀도 계약([계약.md](계약/계약.md) · [contract.md](계약/contract.md))과 CPU 참조(`정수계약.py` · `로그확률.py` · `gguf읽기.py`) — 모든 비트 시험의 기준 |
| [시험/](시험/) | `빠른점검.py`(커밋마다의 CI — GPU 없이 빌드 · 모듈 바이트 · SASS 넘침 · 가져오기 표), 계약 자체 시험, 7B CPU 참조, 뽑기 · 로그확률 시험, `모듈해시.txt` · `SASS넘침.txt` · `글저장소커밋.txt` |
| [교차검증/](교차검증/) | 앱의 재현성 검증 기준(`expected.json` · `requests.json`) — 꾸리기가 빌드 해시를 여기와 견준다 |
| [문서/](문서/) | 이관 범위 · 바뀐 것 |

## 빌드

글(geul) 컴파일러와 PTX · SPIR-V 뒷단은 글 저장소 `research/gpu` 가지의 고정 커밋([시험/글저장소커밋.txt](시험/글저장소커밋.txt))을
옆 폴더 `../geul-lang-v2` 에 둔다(GeulPU README 의 '소스에서 빌드하기' 2 와 같다).

```
앱\꾸리기.ps1 -GeulRoot ..\geul-lang-v2 [-GeulC ..\geul-lang-v2\build\self_컴파일러.exe] -NoZip
python -X utf8 시험\빠른점검.py --글 ..\geul-lang-v2
```

꾸린 것은 `빌드\앱\GeulPU-<판>\`(저장소 밖). 자료(모델 · 이 GPU 용 모듈 · 설정 · 결과)는 `%USERPROFILE%\GeulPU` — GeulPU 앱과
같은 폴더를 쓴다(`GEULPU_HOME` 으로 바꿈). 쓰는 법은 [앱/읽어보기.txt](앱/읽어보기.txt).

## 라이선스

MIT — [LICENSE](LICENSE).
