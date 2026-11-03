# GeulPU 계약을 보존하는 NPU 선형층 실험

이 코드는 GeulPU 앱에 연결된 제품 백엔드가 아니라 `시험/참조7B.py` 모델 순전파에 연결한 실험 백엔드다. Q8_0/Q4_0 선형층의 정수 내적은 AMD NPU에서 실행하며, 나머지 처리는 CPU에서 수행한다. 장치 오류 시 CPU 내적으로 대체하지 않고 실패한다.

## 계산 경계

1. `정수계약.자리로`가 활성값을 32칸 블록별 부호 있는 INT8 자리 두 개와 지수로 나눈다.
2. `exact_kernel.py`가 각 자리·블록의 INT8 × INT8 내적을 XDNA2에서 INT32로 계산한다. 최대 절댓값은 `32 × 128 × 128 = 524288`이므로 중간합도 INT32 범위 안이다. 정수 덧셈 순서가 바뀌어도 오버플로가 없으므로 결과가 같다.
3. `exact_linear.py`가 자리별 지수 복원, FP32 자리 합, 블록 오름차순 `정수계약.fma32`, 마지막 bias 덧셈을 수행한다. 서로 다른 블록을 먼저 합치거나 FP16/BF16으로 바꾸지 않는다.

입력·가중치·스케일·계약 모드가 같다는 조건이다. 테스트는 부동소수점 허용오차 대신 FP32의 uint32 표현을 비교한다. INT8 가중치와 32의 배수 K, 유한한 활성값·스케일을 지원한다. 다른 가중치 형식의 NPU 구현, 다른 장치/드라이버 간 비트 일치, 무작위 샘플링의 재현성은 이 실험의 검증 범위 밖이다.

## 실행

프로젝트 루트에서 준비된 개발 환경을 사용한다. NPU를 사용하는 시험은 순차 실행한다.

```powershell
./재기/npu/run-exact.ps1 -Mode unit
./재기/npu/run-exact.ps1 -Mode kernel
./재기/npu/run-exact.ps1 -Mode boundary
./재기/npu/run-exact.ps1 -Mode model -Tokens '9707,1879,13' -Repeats 2 -Output '빌드/npu/results/exact/hello-world.jsonl'
```

`unit`은 CPU 정수 oracle로 후처리만 검사한다. `kernel`과 `boundary`는 실제 NPU를 요구한다. `model`은 동일 GGUF의 CPU 참조 순전파 후, 모든 Q8/Q4 선형 호출을 NPU 경로로 바꿔 반복 실행한다. 각 호출에서 같은 입력에 대한 CPU 결과를 즉시 대조하고 차이가 있으면 중단한다. CPU 참조 계산은 비교용이며 모델에 전달하는 값은 NPU 부분합으로 계산한 값이다.

추적 파일은 각 선형층의 이름, 모양, 값 범위, 표본 값, SHA256, 다른 비트가 있는 원소 수, 최대 절대 차이, CPU/혼합 경로 시간을 담는다. 잔차는 블록별로 비교한다. 최종 로짓은 `.npy`로 저장한다. 실패 시 JSONL의 마지막 상태와 첫 불일치 좌표·비트 표현을 확인한다. 고정 토큰열 순전파 시험이며, 긴 대화·지속 KV 캐시·자기회귀 생성 속도 측정은 아니다.

## 도구와 모델

- AMD Ryzen AI 7 350 / Krackan XDNA2, 드라이버 32.0.203.329, 펌웨어 1.0.21.43.
- 장치 XRT 2.19.0, 개발용 XRT SDK 2.21.75. 이 조합에서 실제 실행은 성공했지만 다른 환경 호환성을 보장하지 않는다.
- Python 3.13.18, mlir_aie 1.3.4, llvm-aie 22.0.0.2026102001+4f5df13d, Visual Studio 2026.
- MLIR-AIE 체크아웃 `1631e1b04cb22dfe3fb99eb910d6b679863a948c`. 설치된 wheel의 Runtime API에 맞춰 작성했다. 최신 main 예제를 그대로 실행하면 API 차이가 있다.
- [공식 MLIR-AIE Windows 설치 안내](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWinNative.md)
- [공식 Qwen3-0.6B-Q8_0.gguf](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q8_0.gguf), 639446688 bytes.
- 모델 SHA256: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`.

도구·모델은 무시되는 `빌드/npu/dev`, `빌드/npu/models` 아래에 둔다. 시스템 드라이버는 변경하지 않았다. 이 모델은 앞선 FLM Q4NX 성능 시험과 정밀도 형식이 다르므로 해당 속도와 직접 비교할 수 없다.

## 최적화 방향

최신 실행 계획 경로는 [ARCHITECTURE.md](ARCHITECTURE.md)에 정리했다. `-Backend planned`는 가중치 패킹 캐시, 2048타일 작업 묶음, 벡터화된 결과 재배치를 사용한다. 이후 확장 시험에서 입력 버퍼 동기화 누락을 발견해 dot/tiled 모두 명시적 H2D 동기화를 추가했다. 이전 작은 입력 시험의 통과가 모든 버퍼 크기의 동기화 안전성을 증명한 것은 아니다. 최신 비교에는 이 수정이 적용된 기준 경로를 사용한다.

### 구현한 타일 경로

`tiled_kernel.py` / `tiled_linear.py`는 8개의 활성값 자리·행과 32개 출력 채널의 dot32를 한 타일에서 계산한다. 가중치는 각 타일에서 활성값 8행이 공유하며, 활성값은 출력 32채널이 공유한다. 128개 타일을 한 번의 NPU 호출로 처리한다. 부분 타일과 남는 행은 0으로 채우고 유효한 결과만 사용한다.

CPU 쪽에서는 NPU 입력 버퍼의 호스트 뷰를 재사용하여 불필요한 역방향 동기화를 줄이고, 최대 1024개 출력 채널 단위로 FP32 후처리를 수행한다. 블록 오름차순 FMA는 그대로다. 현재 다중 코어, 호출 간 가중치 캐시, 전송과 계산 중첩은 구현하지 않았다.

```powershell
./재기/npu/run-exact.ps1 -Mode tiled-test
./재기/npu/run-exact.ps1 -Mode model -Backend tiled -Tokens '9707,1879,13' -Repeats 2 -Output '빌드/npu/results/optimized/tiled.jsonl'
./재기/npu/run-exact.ps1 -Mode model -Backend dot -Tokens '9707,1879,13' -Repeats 2 -Output '빌드/npu/results/optimized/dot.jsonl'
```

`compare_optimization.py`로 두 JSONL 및 저장된 로짓을 비교한다. 패킹 시간, NPU 호출과 전송을 합한 시간, CPU 후처리 시간을 별도로 기록한다. NPU 호출 시간에는 Python/XRT 호출·복사·동기화가 포함되며 장치 연산만의 시간으로 해석하면 안 된다.

### 남은 작업

두 경로 모두 단일 코어이며 FP32 후처리는 CPU에 남아 있다. 타일 경로도 네이티브 행렬 MAC 명령을 사용하는 GEMM 구현은 아니므로 NPU의 행렬 연산 성능을 대표하지 않는다. 다음 최적화는 다중 코어 분배, 행렬 명령 활용, 전송과 계산 중첩이다. 각 32칸 블록의 부분합 및 FP32 후처리 순서를 유지하면서 이 시험을 통과해야 한다.
