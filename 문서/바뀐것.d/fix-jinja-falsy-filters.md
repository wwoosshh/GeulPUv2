갈래: 고침

- **채팅 템플릿 해석기가 거짓인 값에 쓴 select · reject · selectattr · rejectattr · map 을 빈 목록으로 돌려준다**(jinja2 와 같게): 전에는
  '반복할 수 없는 값' 오류로 멈췄다. 카카오 Kanana 1.5 의 템플릿은 도구가 없을 때 `tools` 를 none 으로 두고 `tools | selectattr(…)` 를
  불러, 모든 요청이 템플릿 오류로 끝났다. 전에 오류 없이 그려지던 템플릿의 결과는 그대로다.

---

- **The chat-template interpreter returns an empty list for select, reject, selectattr, rejectattr and map on a falsy value**, as
  jinja2 does; it used to stop with a "not iterable" error. Kakao's Kanana 1.5 template sets `tools` to none when no tools are given
  and then calls `tools | selectattr(…)`, so every request failed in the template. Templates that rendered before render the same.
