갈래: 더함

- **계산 능력 7.x 의 NVIDIA 카드(Tesla V100 · T4, RTX 20, GTX 16)에서도 CUDA 로 실행합니다.** 이 카드들에는 GeulPU 의 커널이 쓰는 정수
  텐서곱 명령이 없어 지금까지 CUDA 경로를 쓰지 못했습니다. 이제 앱이 장치를 보고 CUDA 모듈을 대체 명령으로 만듭니다(PTX 백엔드의
  `--target sm_70` — 커널 소스는 그대로). 값은 비트까지 같고 프롬프트 처리는 느립니다. 빠른 모드, 8비트 KV 캐시, Qwen3.5 · Gemma 같은
  CUDA 전용 구조도 이 카드에서 돕니다. Linux 의 Tesla V100 에서만 확인했습니다 — 자세한 범위는
  [지원범위.md](../../지원범위.md)의 1 · 7 절에 있습니다.

---

- **NVIDIA cards with compute capability 7.x (Tesla V100 · T4, RTX 20, GTX 16) now run on CUDA.** They lack the integer tensor-product
  instruction GeulPU's kernels use, so until now they could not use the CUDA path. The app now looks at the device and builds its CUDA
  modules with replacement instructions (`--target sm_70` of the PTX backend — the kernel sources are unchanged). The values are the
  same bit for bit; prompt processing is slower. Fast mode, the 8-bit KV cache and the CUDA-only architectures (Qwen3.5, Gemma …) run on
  these cards too. Checked on a Tesla V100 under Linux only — see sections 1 and 7 of
  [docs/support.md](../../docs/support.md).
