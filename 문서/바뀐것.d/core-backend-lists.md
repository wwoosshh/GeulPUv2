갈래: 바꿈

- **모델을 처음 올릴 때 커널 모듈을 만드는 뒷단이 더 빨라지고 메모리를 덜 씁니다.** 모듈을 PTX 로 바꾸는 뒷단이 약 1.24 배,
  SPIR-V 로 바꾸는 뒷단이 약 1.15 배 빨라졌고(Windows, 모듈 다섯), 가장 큰 모듈에서 최대 메모리가 3.7 GB → 1.6 GB(PTX),
  2.8 GB → 1.6 GB(SPIR-V)로 줄었습니다. 만들어지는 커널은 바이트까지 그대로입니다.

---

- **The backends that build kernel modules when a model is first loaded are faster and use less memory.** Turning a module
  into PTX is about 1.24× faster and into SPIR-V about 1.15× (Windows, five modules), and the peak memory for the largest
  module drops from 3.7 GB to 1.6 GB (PTX) and from 2.8 GB to 1.6 GB (SPIR-V). The kernels are byte-identical.
