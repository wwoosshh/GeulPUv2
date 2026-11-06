# GeulPU 0.21.0 의 독립 검증 — 마드리드 공과대학교 GING 그룹 (2026년 11월)

마드리드 공과대학교(Universidad Politécnica de Madrid) DIT 의 GING 그룹(Next Generation Internet Group —
[ging.github.io](https://ging.github.io)) 연구자들이 2026년 11월 5–6일에 NVIDIA GPU 셋에서 GeulPU 0.21.0 을 시험했다. 이 키트와 함께,
AI 에이전트가 돌린 그들 자신의 시험 모음을 썼고, 결과를 여기에 실으라고 보내 왔다(2026년 11월 6일 메일). 이 폴더는 받은 것을
그대로 둔다:

[English](README.md)

- `geulpu-0.21.0-independent-test-summary-v7.pdf` — 그들의 한 장짜리 요약. 머리에 "independent test results; not measurements
  reported by the engine authors"(독립 시험 결과이며 엔진 저자가 보고한 측정이 아니다)라고 적혀 있다.
- `bundle/` — 그들의 재현 묶음: 시험 코드, 고정한 요청 모음, SHA-256 목록, CPU 검사용 합성 자료. [bundle/README.md](bundle/README.md)
  에 파일마다 하는 일과 돌리는 법이 있다. 이 사본에서 `cd bundle/code && sha256sum -c SHA256SUMS` 가 통과한다. 모델 파일 · 자격
  증명 · 실행 기록은 들어 있지 않다.

받은 파일은 `geulpu-0.21.0-independent-test-summary-v7.pdf`(7,450 바이트, SHA-256
`9e4832f596d85f20391b90a8d4949a69b6d4ebf9f4a7c3dda636949c72ed4e5e`)와 tar.gz 묶음(67,961 바이트, SHA-256
`dc8cf70bcaf09d1ef7c05c1fe38d5e8f1740d21cf6c1876ba812255d50a95341`)이고, 묶음은 그대로 풀어 두었다.

## 결과

리눅스의 GeulPU 0.21.0(커밋 `0efe2a58`, PyPI 휠 `e71335f5…`)을 RTX 6000 Ada(카드 통째, 48 GB — 공개 데모를 함께 띄운 채), H100 NVL
MIG 구획(3g.47gb, 47 GB), A100 MIG 구획(7g.40gb, 40 GB)에서 돌렸다. 비교는 모두 프롬프트 토큰 ID · 생성 토큰 ID · 걸음마다 로짓 해시로
한다. 모든 검사가 세 GPU 에서 다 통과했다:

| 검사 | 세 GPU 저마다 |
|---|---|
| 이 키트 — 우리 RTX 4070 Ti 기준과 요청 144 개(모델 3 × default · exact 설정 × 요청 12 × 하나씩 · 8 개 한꺼번에) | 144/144 |
| 앱의 재현성 검증(DeepSeek-R1-Distill-Qwen-7B Q8_0, 32 문제) | 32/32 |
| 그들의 핵심 모음, Qwen3-8B Q4_K_M: 첫 실행 기준과의 답 비교 · 요청 겹침 · 요청 16 개의 4 갈래 몰아 보내기 · 캐시 축출과 SSD 복원 | 36/36 · 15/15 · 통과 · 4/4 |
| 그들의 추가 모음, Qwen3.5-9B Q4_K_M 과 Qwen3-8B Q8_0: 첫 실행 기준 · 토큰화 일치 · 겹침 · 요청 17 개의 4 갈래 · SSD 복원(Q8_0) | 36/36 과 37/37 · 17/17 · 15/15 · 통과 · 통과 |
| 추측 디코딩 켬과 끔, Qwen3-8B Q8_0(갈래마다 37/37, 두 갈래 사이 60 줄) | 37/37 · 37/37 · 60/60 |
| 동시 대화 8 개(슬롯 0–7, 문맥 8,192), Qwen3-8B Q8_0 | 23/23 |
| Llama-3.1-8B-Instruct Q8_0(둘째 모델 계열): 모음 · 동시 대화 8 개 · 추측 디코딩 켬과 끔 | 37/37 · 23/23 · 37/37 + 60/60 |
| 세 GPU 에서 같은 출력(그들의 요청 — 셋을 가로지른 관찰 하나이지 통과 셋이 아니다) | Qwen 58/58 · Llama 23/23 |

키트의 요청 144 개는 윗 폴더의 설명대로다: 생각을 켠 수학, 논리, 코드, 도구 호출 둘, 한국어, 씨앗을 준 온도 0.7–0.8 의 스페인어
표집과 여러 턴 대화 표집, 6천 토큰 기록. 그래서 탐욕 답만이 아니라 씨앗 표집 · 도구 호출 · 생각에서도 같은 비트가 나온 것이다.
A100 과 RTX 6000 Ada 는 GeulPU 가 처음 돈 GPU 이고, Qwen3.5-9B 는 처음으로 다른 GPU 와 비교됐다.

## 보고서 읽을 때

보고서의 두 줄은 시험의 발견이 아니라 GeulPU 의 생김새다:

- "KV8 / fast attention: engaged only in core Q4; N/A (unsupported) on Q8, Qwen3.5, Llama". 앱은 8비트 KV 캐시를 CUDA 의 빠른 계산에서
  k-양자화 모델(Q4_K_M · Q5_K_M · Q6_K …)에만 쓰고, 빠른 어텐션은 그 캐시 위에서만 쓴다(`앱/일.gl`). 그래서 Q8_0 파일과 Qwen3.5 는
  기본 설정에서도 빠른 계산 · 16비트 KV 캐시로 돈다(키트 README 에도 있다). 그들의 모음은 둘을 요청하고 모델을 불러올 때마다
  서버의 실제 상태를 확인했다.
- "Serialized tool-call text can differ between identical runs; where it did, prompt IDs, generated token IDs and logit hashes still
  matched." 도구 호출의 id(`call_…`)는 요청이 들어온 시각과 서버 안의 번호로 만들어 호출마다 새 값이다 — 응답의 `created` 칸과
  같다. 함수 이름 · 인자 · 메시지 글은 생성 토큰에서 나온다. 그들의 하네스는 이 키트처럼 메시지 글 전체의 해시를 참고로만 남기고
  토큰과 로짓으로 판정한다.

## 범위 — 그들의 말로

- 그들 자신의 모음은 기본인 빠른 모드로 돌렸다(키트의 144 개는 default 와 exact 둘 다). exact 모드의 모음과 고장 주입은 뺐고,
  Gemma 4 는 돌리지 않았다.
- 세 GPU 가 모두 NVIDIA 다. Vulkan · AMD · Intel GPU 에 대해서는 아무 말도 하지 않는다.
- 시험한 모델 · 양자화 · GPU 에서의 경험적 관찰이지 보증이 아니다.
- RTX 6000 Ada 의 동시 대화 8 개 칸은 공개 데모가 카드를 같이 쓰는 채로 메모리 문턱의 여유 43 MiB 로 통과했다. H100 구획의 첫 Llama
  시도는 교란 모듈에서 멈췄다 — 긴 요청과 겹친 요청이 15 개 중 10 개였다(14 개 필요). 이것은 INCOMPLETE 기록으로 남겨 두었고, 요청
  간격 0.5 초로 다시 돌린 것이 15/15 통과했다.
- 보고서의 시간은 참고 값이고 속도 주장은 없다.

## 출처

시험은 GING 그룹이 자기 장비에서 설계하고 돌렸고, Pedro Reviriego 가 이 저장소에 실으라고 보냈다. 이 폴더의 파일은 그들의 것이며,
그들이 바란 대로 다른 사람이 시험을 재현할 수 있도록 여기에 싣는다.
