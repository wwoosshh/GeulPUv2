갈래: 더함

- **실행 파일의 버전 정보와 코드 서명 정책**: Windows 의 `GeulPU.exe` 에 제품 이름 · 버전 · 설명 · 라이선스(VS_VERSIONINFO)를 넣는다 — 탐색기의
  속성 창에 보이고, 코드 서명의 조건이다(`앱/버전정보.py`, 꾸리기 때). README 에 '코드 서명 정책' 절(SignPath Foundation 의 무료 오픈소스
  서명 — 팀 역할 · 서명하는 파일 · 승인), 저장소에 개인정보 처리 방침(`PRIVACY.md` — 망으로 보내는 것과 저장하는 것 전부)이 생겼고, 릴리스
  노트가 그 정책을 가리킨다. `bin/` 의 엔진 실행 파일은 그대로다(교차검증의 빌드 해시).

---

- **Version information in the executable, and a code signing policy.** The Windows `GeulPU.exe` now carries a product name, version,
  description and license (VS_VERSIONINFO) — shown in Explorer's Properties, and a requirement for code signing (`앱/버전정보.py`, at
  packaging). The README gains a "Code signing policy" section (free open-source signing by SignPath Foundation — team roles, what is
  signed, approval), the repository gains a privacy policy (`PRIVACY.md` — everything the app sends and stores), and the release notes
  point to the policy. The engine binaries in `bin/` are unchanged (the cross-check compares their build hashes).
