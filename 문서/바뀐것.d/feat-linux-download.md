갈래: 더함

- **리눅스 앱의 내려받기**: 리눅스에서도 앱 안에서 모델을 받는다 — 화면의 다운로드 · Hugging Face 찾기의 확인 · `GeulPU --hf 저장소/파일.gguf`.
  윈도와 같은 길이다: `.part` 에 받고, 일시정지하면 그 자리에서 이어 받고(Range), 끊기면 다섯 번까지 다시 받고, 다 받으면 SHA-256 을
  확인한 뒤 모델로 더한다. HTTPS 는 시스템의 curl 이 한다(프록시는 `https_proxy` 같은 환경 변수). curl 이 없으면 설치하라고 알린다.

---

- **Downloads in the Linux app.** The Linux app now downloads models itself — the screen's download, the Hugging Face finder's check,
  and `GeulPU --hf <repo>/<file>.gguf`. It works as on Windows: the file goes to `.part`, a pause resumes where it stopped (Range), a
  dropped connection is retried up to five times, and the SHA-256 is verified before the model is added. HTTPS is done by the system's
  curl (proxies from environment variables such as `https_proxy`); without curl the app says to install it.
