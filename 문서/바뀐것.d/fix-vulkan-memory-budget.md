갈래: 고침

- **내장 GPU 에서 큰 모델을 불러오다 'Vulkan 오류 -4'로 멈추던 것**: Vulkan 층이 남은 그래픽 메모리를 힙 크기 − 잡은 양으로 세어,
  노트북의 내장 GPU(Intel Arc 130V 등 — 힙이 Windows · 다른 앱과 나눠 쓰는 시스템 메모리)에서 실제보다 많이 잡고 장치를 잃었다. 이제
  드라이버가 알려 주는 예산(VK_EXT_memory_budget)에서 지금 쓰는 양을 뺀 만큼으로 문맥을 줄이고, 그래도 넘치는 할당은 내장 GPU 에서 '그래픽
  메모리가 모자란다(필요 · 남은 양)'로 미리 멈춘다. 장치를 잃으면 잡은 메모리 · 남은 양과 앞서 낸 커널을 적는다. 장치 확인이 내장 GPU 와
  지금 쓸 수 있는 양을 알려, 앱이 그 값을 그 GPU 의 메모리로 쓴다. 메모리가 넉넉할 때 답은 비트까지 그대로다.

---

- **Loading a large model on an integrated GPU no longer stops with "Vulkan 오류 -4".** The Vulkan layer counted free graphics memory
  as heap size minus its own allocations, so on integrated GPUs such as a laptop's Intel Arc 130V — whose heap is system memory shared
  with Windows and other apps — it allocated more than was really there and lost the device. It now sizes the context from the driver's
  budget (VK_EXT_memory_budget) minus current usage, and on integrated GPUs it stops an allocation that would still not fit with a
  clear "graphics memory is short" message (needed and free amounts). When the device is lost anyway, it logs the allocated and free
  memory and the last kernels it issued. The device check reports integrated GPUs and the memory available now, and the app uses that
  value as the GPU's memory. With enough memory, replies are bit-identical.
