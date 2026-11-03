# GeulPU NPU 기준 측정 v1

Ryzen AI 7 350 / Radeon 860M / 32GB Windows PC에서 NPU 실행의 기준을 만든다.
이 디렉터리의 코드는 GeulPU 추론 엔진을 수정하지 않는다. 외부 런타임의 HTTP 실행을 기록하며,
GeulPU 정밀도 계약을 통과했다는 의미로 결과를 해석하지 않는다.

## 고정한 실행 환경

- FastFlowLM **v1.0.10**, Windows 일반 배포본(XRT). HRX 배포본이 아니다.
- Qwen3-0.6B-NPU2, 태그 `qwen3:0.6b`. GeulPU와 공통 모델 계열이고 작은 초기 검증 대상이다.
- 카탈로그 양자화 이름 Q4_1, 실제 파일 `model.q4nx`. GeulPU GGUF 파일과 바이트가 같은 모델이 아니다.
- NPU 드라이버 32.0.203.329, `flm validate --json`의 ready=true를 확인했다.
- NPU performance 모드, 문맥 용량 32768, 단일 요청, 출력 상한 128.
- 프로세스 밖 환경 변수·시스템 드라이버·설치 상태는 변경하지 않았다. 모델과 휴대용 런타임은 `빌드/npu/`에 있다.

공식 ZIP: https://github.com/ROCm/FastFlowLM/releases/download/v1.0.10/fastflowlm_1.0.10_windows_amd64.zip

## 다시 실행

Node.js 18 이상과 PowerShell, 위의 FLM 및 모델이 필요하다. 이 PC에는 내려받아 두었다.
재다운로드가 필요한 경우 ZIP을 `빌드/npu/flm-1.0.10/`에 풀고 다음처럼 모델을 준비한다.
모델 저장 환경 변수는 FLM이 그 밑에 `models/`를 한 번 더 붙이는 **기본 디렉터리**다.

```powershell
$env:FLM_MODEL_PATH = (Join-Path (Get-Location) '빌드/npu/models')
& ./빌드/npu/flm-1.0.10/flm.exe pull qwen3:0.6b
& ./재기/npu/run.ps1 -Repeat 5 -PowerMode performance
```

`run.ps1`은 포트 44560이 사용 중이면 중단한다. 자기 서버만 시작하고 종료하며, 요청 전 모델과
런타임/커널 파일 SHA-256, 드라이버, 전원 계획, 배터리 상태를 기록한다. 모델을 자동 다운로드하지 않는다.
결과 폴더는 `빌드/npu/results/<시각>-<전력모드>/`다. 이 폴더는 기존 gitignore로 제외된다.

이미 열린 다른 런타임의 로컬 OpenAI 호환 서버도 측정할 수 있다:

```powershell
node 재기/npu/bench.mjs --url http://127.0.0.1:44560 --model qwen3:0.6b --out 빌드/npu/results/manual --repeat 5 --request-options 재기/npu/flm-greedy.json
node 재기/npu/bench.mjs --summarize 빌드/npu/results/manual/requests.jsonl
node --test 재기/npu/bench.test.mjs
```

`flm-greedy.json`은 FLM 전용이다. 다른 서버에는 그 서버가 지원하는 옵션 파일을 따로 써야 한다.
FLM의 seed 지원을 가정하지 않는다. 요청에 남긴 seed=7은 요청 기록용이며 결정성을 보장하지 않는다.
Greedy 조건은 temperature=0, top_k=1, top_p=1, presence/frequency penalty=0,
repetition_penalty=1, think=false를 매번 보낸다.

## 측정 정의

| 지표 | 정의와 주의점 |
|---|---|
| 첫 출력 지연 | HTTP 요청 시작 → 첫 비어 있지 않은 content/reasoning_content 델타. 숨겨진 제어 토큰의 첫 생성 시각과 다를 수 있다. |
| 전체 지연 | HTTP 요청 시작 → SSE 종료. 전송·파싱 비용을 포함한다. |
| 서버 prefill TPS | FLM usage.prefill_speed_tps. 서버 내부 정의이며 클라이언트 첫 출력 지연으로 역산하지 않는다. |
| 서버 decode TPS | FLM usage.decoding_speed_tps. 각 응답의 실제 completion_tokens와 함께 저장한다. |
| 전체 출력 TPS | completion_tokens / 전체 응답 시간. 생성 전용 TPS와 구분한다. |
| 출력 반복성 | content + reasoning_content + finish_reason의 SHA-256 일치. 토큰 ID나 로짓 비트 일치를 뜻하지 않는다. |
| 캐시 | cached_tokens를 기록. 알 수 없으면 null. API 보고만으로 내부 상태 전체가 초기화됐다고 가정하지 않는다. |
| 메모리 | 별도 monitor.ps1의 프로세스 working set/private bytes. NPU SRAM 용량 또는 모델 가중치 크기와 다르다. |
| 전력 | 현재 미측정(null). 전체 시스템 전력 시계열 또는 외부 계측이 있어야 J/token을 계산한다. |

스트림 조각 개수는 토큰 개수가 아니다. usage가 없으면 토큰 속도는 null로 남긴다.
SSE 종료 표시나 finish_reason이 없거나 HTTP 오류가 나면 성공으로 세지 않는다.
원시 요청, 모든 응답 이벤트와 상대 시각을 `requests.jsonl`에 저장한다. 별도 서버의 생성 로그를
`geulpu-run-log/1`로 위장하지 않고 `geulpu-npu-benchmark/1`로 구분한다.

## 현재 시험 집합의 범위

한국어 기술 설명 자료를 8/64/256행으로 만든 합성 입력 세 종류다. 정확한 토큰 수는 서버 usage를 따른다.
각 조건 워밍업 1회(통계 제외) + 측정 5회. 반복마다 조건 순서를 순환한다.
프롬프트는 공개 가능한 합성 텍스트이며, 실제 개인 대화나 파일을 보내지 않는다.
출력은 최대 128토큰으로 잘리므로 답변의 완결성·한국어 품질 평가로 쓰지 않는다.

5회 표본의 p95는 사실상 최댓값이다. 운영 지연 꼬리를 추정하는 충분한 표본이 아니다.
콜드 프로세스 로딩, 여러 요청의 동시 처리, 배터리별 성능, 열 포화, 장문 품질, GPU 대비 속도는 이 실험에 포함되지 않는다.
초기 실행과 엄격 greedy 실행은 요청 옵션이 다르므로 한 표본으로 합치지 않는다.

## 확장할 평가 행렬

1. 같은 런타임으로 0.6B → 4B, 입력 약 0.5K/4K/12K, 출력 128/512, 문맥 용량 고정.
2. performance/balanced, AC/배터리를 각각 독립 실행. 첫 실행/워밍업 이후/반복 프롬프트를 분리.
3. 캐시 꺼짐/켜짐은 해당 버전의 실제 지원을 확인하고 캐시 수치와 출력을 함께 검사.
4. 동시 요청 1/2/4: 요청별 지연, 합산 TPS, 큐 대기, 오류율. 소켓 수를 배치 크기로 해석하지 않는다.
5. 전력: 같은 밝기·전원 조건에서 idle과 workload의 시스템 전력을 적분. 총 J/token과 idle 차감 J/token을 구분.
6. 품질: 고정 한국어·코딩·JSON·도구 호출 사례를 별도 판정. 같은 토크나이저/가중치가 아니면 순수 백엔드 비교라고 부르지 않는다.
7. 실행 패키지 비교(FLM Q4NX vs GeulPU GGUF)와 수치 계약 비교(같은 GGUF + CPU 참조)를 별도 실험으로 유지.

## GeulPU 적용 경계

### 1단계: 관리 계층에 외부 NPU 실행기를 추가

`브라우저 → GeulPU 관리자 /v1 → backend adapter → FLM → NPU`

| 현재 파일 | 확인한 동작 | 제안하는 변경 |
|---|---|---|
| 앱/일.gl | 모듈준비 후 server-cuda/server-vulkan을 선택하고 --ptx 등을 전달 | 외부 런타임 유형은 GPU 모듈생성 경로를 건너뛰고 전용 실행 인자·health 검사를 사용 |
| 앱/웹.gl | /v1 스트리밍을 상위 서버로 전달 | 기존 프록시를 활용하되 지원하지 않는 필드를 명시적으로 거절/변환; 사용자가 요청한 의미를 조용히 버리지 않음 |
| 앱/상태.gl, 앱/관리자.gl | cuda/vulkan 상태·설정 및 모델 지원 판정 | runtime, device, artifact format, capabilities를 구분해서 기록·표시 |
| 앱/디버거.gl | 기록 시 geulpu.logit_hash/prompt_tokens 자동 삽입 | 기능이 확인된 백엔드에만 삽입; 미지원은 null/unsupported로 기록 |
| 도구/debugger.py | GeulPU 토큰·로짓 해시를 이용한 재생/분기 | 텍스트 재생과 토큰 일치와 비트 일치 판정을 분리; 누락된 배열끼리 같다고 판정하지 않음 |
| 앱/문맥저장소.gl, 엔진/저장소.gl | GeulPU KV 형식의 캐시 관리 | FLM 캐시와 공유하지 않음. runtime/model/contract별 별도 네임스페이스 |
| 구조/등록.gl, 엔진/구조표.gl | qwen/llama 등 모델 구조의 함수 테이블 | NPU는 모델 구조가 아니므로 여기 새 모델처럼 등록하지 않음 |

`device=npu` 하나만 추가해서 기존 CUDA/Vulkan 함수를 호출하는 구현은 부족하다.
외부 엔진과 자체 엔진의 경계는 GPU 함수 테이블보다 위인 **요청 실행 계층**에 두는 편이 맞다.

제안하는 기능 표시는 true/false/unknown 세 상태로 한다. 최초 FLM 어댑터의 보장 범위는 채팅/SSE/usage/텍스트 기록이다.
seed 재현, raw token IDs, logit hashes, echo scoring, token branch, 층별 lens, KV export,
speculative invariance는 검증 전까지 제공하지 않는다. logprobs는 특정 런타임 버전에서 별도 시험 후 활성화한다.
정밀도 표기는 exact 대신 `external-flm-q4nx` 같은 별도 계약 식별자를 사용한다.

어댑터의 수용 시험: 동일 요청을 FLM 직접/프록시 경유로 보내 내용·usage·오류 의미·SSE 종료를 비교하고,
부가 지연을 측정한다. FLM 자체 반복 출력이 다르면 프록시 비교만으로 비트 동일성을 주장하지 않는다.
고정 응답 테스트 서버로 프록시의 전달 정확성을 먼저 검증한다.

### 2단계: GeulPU의 검증 원칙을 NPU에 적용

- 환경/가중치/토크나이저/커널 해시를 실행 헤더에 고정한다.
- 요청 순서·동시성·캐시·전력 모드 변화마다 텍스트/토큰/로짓을 관측 가능한 수준에서 비교한다.
- 속도 회귀와 정확성 회귀를 따로 판정한다. 성능 합격이 정확성 합격을 대신하지 않는다.
- 반복 출력이 같더라도 CPU 참조와 대조하지 않았다면 GeulPU exact 계약 통과라고 표시하지 않는다.

### 3단계: 실제 GeulPU NPU 커널 연구

IRON/MLIR-AIE로 Q8_0 선형층 하나를 구현해 계약/정수계약.py와 대조하는 작은 실험부터 시작한다.
가중치 정수·블록 스케일을 유지하고, GeulPU 두 자리 활성값의 부분합을 정확히 얻은 뒤,
스케일 적용·반올림·누산 순서까지 검증한다. 이어 RMSNorm/RoPE/softmax/attention을 확장한다.
처음부터 글→NPU 전체 컴파일러를 만들지 않고, 검증된 커널 패키지 + 작은 호스트 브리지로 시작할 수 있다.

합격 기준은 (a) 경계값·임의값·실제 가중치에서 비트 불일치 0, (b) 전송/대기/후처리를 포함한 시간,
(c) 전체 시스템 에너지다. CPU에서 부분합 후처리를 하는 실험은 가능하지만 전송량이 커져 느려질 수 있다.
블록별 결과를 매번 CPU로 내리는 구성은 정확성 실험용과 제품 실행용을 구분한다.
현재 FLM 속도는 다른 수치 방식의 구현 기준이며 전용 GeulPU NPU 커널의 예상 속도로 사용하지 않는다.

## 근거

- 실행한 고정 릴리스: https://github.com/ROCm/FastFlowLM/releases/tag/v1.0.10
- 요청 필드 의미(변경될 수 있는 최신 문서): https://fastflowlm.com/docs/instructions/server/request_parameters/
- 저수준 NPU 프로그래밍: https://github.com/Xilinx/mlir-aie
- GeulPU 계약: ../../계약/계약.md
- 실제 측정 결과: results/2026-10-27.md

웹의 최신 문서는 고정 바이너리보다 앞설 수 있다. CLI 옵션은 실제 v1.0.10 --help와 실행 결과를 우선했다.
