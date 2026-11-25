갈래: 바꿈

- **모델을 처음 올릴 때 커널 모듈을 만드는 뒷단이 빨라졌습니다.** 글(geul) 컴파일러를 1.4.1 로 올려, 모듈을 PTX 로 바꾸는 뒷단이
  약 1.5 배, SPIR-V 로 바꾸는 뒷단이 약 1.56 배 빨라졌습니다(Windows, 모듈 다섯). 만들어지는 커널은 그대로입니다 — PTX 는 명령까지
  같고 가상 레지스터 선언만 늘었으며, SPIR-V 는 바이트까지 같습니다.

---

- **The backends that build kernel modules when a model is first loaded are faster.** The geul compiler moves to 1.4.1, so turning a
  module into PTX is about 1.5× faster and into SPIR-V about 1.56× (Windows, five modules). The kernels are unchanged — the PTX has the
  same instructions and only declares more virtual registers, and the SPIR-V is byte-identical.
