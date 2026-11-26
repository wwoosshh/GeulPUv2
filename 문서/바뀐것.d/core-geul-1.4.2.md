갈래: 바꿈

- **모델을 처음 올릴 때 커널 모듈을 만드는 뒷단이 더 빨라졌습니다.** 글(geul) 컴파일러를 1.4.2 로 올려, 모듈을 PTX 로 바꾸는 뒷단이
  약 1.18 배, SPIR-V 로 바꾸는 뒷단이 약 1.17 배 빨라졌습니다(Windows, 모듈 다섯). 만들어지는 커널은 바이트까지 그대로입니다.

---

- **The backends that build kernel modules when a model is first loaded are faster.** The geul compiler moves to 1.4.2, so turning a
  module into PTX is about 1.18× faster and into SPIR-V about 1.17× (Windows, five modules). The kernels are byte-identical.
