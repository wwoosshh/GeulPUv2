갈래: 고침

- **Vulkan 의 곱해더하기는 한 번만 반올림**: SPIR-V 뒷단이 곱해더하기(GLSL Fma)에 NoContraction 을 단다. 없으면 드라이버가 곱과
  합을 따로 반올림해도 되어, NVIDIA 밖의 GPU 에서 Vulkan 의 출력 비트가 CUDA 와 달라질 수 있었다(Intel UHD 770 에서 본 일).
  NVIDIA 의 Vulkan 출력 비트는 그대로다. 모든 모듈의 SPIR-V 가 바뀐다.

---

- **Vulkan multiply-add rounds once.** The SPIR-V backend now decorates fused multiply-add (GLSL Fma) with NoContraction.
  Without it a driver may round the product and the sum separately, so Vulkan output bits could differ from CUDA on GPUs other
  than NVIDIA (seen on an Intel UHD 770). Output bits on NVIDIA's Vulkan are unchanged. Every module's SPIR-V changes.
