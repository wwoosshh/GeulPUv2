갈래: 더함

- **리눅스의 Vulkan(#135)**: 리눅스 꾸러미에도 Vulkan 판(server-vulkan · device-vulkan · diag-vulkan · spirv-backend 와 진단 커널)이
  든다 — NVIDIA 밖의 GPU(AMD · Intel …)를 리눅스에서 Vulkan 으로 쓰는 길이다. Vulkan 선언의 리눅스 판이 `libvulkan.so.1` 을 가져오고,
  빠른 점검이 윈도 판과 같은 함수인지, 리눅스 Vulkan 판이 `libc.so.6` · `libvulkan.so.1` 만 가져오는지 본다. Windows 실행 파일 ·
  모듈 · 출력 비트는 그대로다([리눅스벌칸.md](../리눅스벌칸.md)).

---

- **Vulkan on Linux (#135).** The Linux package now carries the Vulkan build too (server-vulkan, device-vulkan, diag-vulkan,
  spirv-backend and the diagnosis kernels) — the way to run GeulPU on GPUs other than NVIDIA (AMD, Intel …) on Linux. The Linux
  Vulkan declarations import `libvulkan.so.1`; the quick check verifies that they declare the same functions as the Windows ones
  and that the Linux Vulkan programs import only `libc.so.6` and `libvulkan.so.1`. Windows binaries, modules and output bits are
  unchanged.
