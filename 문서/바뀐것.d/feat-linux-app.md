갈래: 더함

- **리눅스용 앱(소스에서 빌드)**: 리눅스 x86-64 · NVIDIA GPU 에서 앱(화면 · 장치 확인 · 모델 · 추론 서버 · 벤치마크의 재현성 검증과 파일 쌍
  재현성)이 돈다. `bash 앱/꾸리기.sh <글 저장소>` 로 꾸리고, 원격 서버에서는 SSH 포트 포워딩으로 화면을 연다. 화면 없이 검사 하나를 돌리고
  끝내는 `GeulPU --검사 재현성|쌍 [--빠름]`(`--check r1|pair [--quick]`)도 더했다 — 결과 폴더와 zip 을 알리고, 모든 문제가 기준과 같으면 끝 코드
  0. 앱 안의 내려받기와 Vulkan 은 리눅스에서 아직 없다(모델은 따로 받아 자료 폴더의 모델 폴더에 둔다).

---

- **The app on Linux (built from source).** On Linux x86-64 with an NVIDIA GPU the app runs — the screen, device check, models, the
  inference server and the benchmark's reproducibility check and file-pair reproducibility. `bash 앱/꾸리기.sh <geul repo>` packages it;
  on a remote server open the screen through SSH port forwarding. New `GeulPU --check r1|pair [--quick]` (`--검사 재현성|쌍 [--빠름]`)
  runs one check without the screen, prints the result folder and zip and exits with 0 when every problem matches the reference.
  Downloads inside the app and Vulkan are not on Linux yet (put the model files in the data folder's model folder).
