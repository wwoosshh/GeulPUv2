갈래: 고침

- **NVIDIA 드라이버가 없는 PC 에서 앱을 켤 때 오류 창이 뜨지 않음**: NVIDIA 드라이버가 없는 Windows PC(Intel · AMD GPU)에서 앱을 켜면
  'nvcuda.dll이(가) 없어 코드 실행을 진행할 수 없습니다' 창(server-cuda.exe · device-cuda.exe)이 모델 수보다 두 번 더 뜨고, 확인을
  다 눌러야 앱이 떴다. 앱은 CUDA 를 먼저 시험하려고 이 두 프로그램을 띄우는데, nvcuda.dll 이 없으면 뜨지 못하고 Windows 가 그때마다
  창을 띄웠기 때문이다. 이제 nvcuda.dll 이 없으면 CUDA 프로그램을 띄우지 않고 바로 Vulkan 으로 가며, 앱이 띄운 프로그램이 뜨지 못해도
  창 없이 실패 코드만 돌려준다.

---

- **No error boxes at startup on PCs without the NVIDIA driver.** On Windows PCs without the NVIDIA driver (Intel or AMD
  GPUs), starting the app showed "The code execution cannot proceed because nvcuda.dll was not found" (server-cuda.exe ·
  device-cuda.exe) two times more than the number of models, and the app came up only after every box was closed. The app
  starts these two programs to try CUDA first; without nvcuda.dll they cannot start, and Windows showed a box each time. Now
  the app skips the CUDA programs when nvcuda.dll is missing and goes straight to Vulkan, and a program the app starts that
  cannot load returns an error code without a box.
