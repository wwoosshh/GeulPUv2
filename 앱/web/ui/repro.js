// 앱리디자인 E6-2(문서/화면구현.md §6) — 재현 › 검사의 두 카드(연구 패널의 '재현' 탭에서 옮겼다): 재현 파일 확인(geulpu-capsule/1 — 같은 모델
// 파일로 이 PC 에서 점수 매기기와 다시 생성을 해 비트까지 비교 — F3-4 부터 앱의 check.repro 가 한다)과 엔진 · 커널 해시(system.provenance — 기준 빌드와 비교). app.js 의 검사 탭이
// #repro-cards 를 그리면(geulpu:repro-cards) 그 안을 채운다.
import { T, el } from './dom.js';
import { call } from './ops.js';
import { appError } from './runs.js';

const LAYER = { cuda: 'CUDA', vulkan: 'Vulkan' };
const st = { busy: '', result: null, prov: null, provErr: '', fileName: '' };
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function stateNow() {
  const r = await call('system.state');
  if (!r.ok || !r.json) throw new Error(appError(r));
  return r.json;
}

/** 재현 파일의 모델(SHA-256)을 불러온다 — 이미 불러온 모델이면 그대로 */
async function ensureModel(m) {
  let s = await stateNow();
  if (s.server && s.server.model === m.id && s.server.state === 'ready') return s;
  st.busy = T`모델을 불러오는 중 — ${m.name || m.id}`;
  draw();
  const r = await call('model.load', { model: m.id });
  if (!r.ok) throw new Error(appError(r));
  for (let i = 0; i < 480; i++) {
    await sleep(500);
    s = await stateNow();
    if (s.server && s.server.model === m.id && s.server.state === 'ready') return s;
    if (s.job && s.job.state === 'failed') throw new Error(s.job.reason || T('모델을 불러오지 못했습니다'));
  }
  throw new Error(T('모델을 불러오지 못했습니다 — 시간이 다 됐습니다'));
}

/** 재현 파일 확인 — 재현 파일의 모델(SHA-256)을 불러온 뒤 앱의 check.repro 가 점수 매기기 · 다시 생성 · 지문을 비트까지 비교한다(F3-4 — CLI 의
   check repro 와 같은 판정). 화면은 그 답을 연구 패널 때와 같은 줄로 보인다 */
async function checkCapsule(file) {
  st.result = null;
  st.fileName = file.name;
  const raw = await file.text();
  let cap;
  try { cap = JSON.parse(raw); } catch (_) { throw new Error(T('JSON 파일이 아닙니다')); }
  if (!cap || cap.format !== 'geulpu-capsule/1' || !cap.request || !cap.result || !cap.model) throw new Error(T('GeulPU 재현 파일이 아닙니다'));
  // 같은 모델 파일이 설치되어 있으면 불러온다 — 없으면 앱이 404 와 그 까닭으로 답한다
  const s0 = await stateNow();
  const models = Array.isArray(s0.models) ? s0.models : [];
  const m = models.find(x => x.sha256 && x.sha256 === cap.model.sha256);
  if (m) await ensureModel(m);
  st.busy = T('재현 파일의 토큰을 이 PC에서 다시 계산하는 중');
  draw();
  const r = await call('check.repro', {}, { body: raw });
  if (!r.ok || !r.json) throw new Error(appError(r));
  const v = r.json;
  const md = v.made_on || {};
  const sv = v.here || {};
  const diff = new Set(Array.isArray(v.differences) ? v.differences : []);
  const yTxt = y => (y ? '×' + y : T('YaRN 없음'));
  const here = `${sv.device || '-'} · ${LAYER[sv.layer] || sv.layer || '-'}`;
  const where = T` (재현 파일에 적힌 만든 곳 ${md.device || '-'} · ${LAYER[md.layer] || md.layer || '-'} → 여기 ${here})`;
  st.result = {
    ok: !!v.ok,
    lines: v.ok ? [T`비트까지 같음 — ${v.steps}스텝의 로그확률 · 엔트로피 · 상위 ${v.top_k} · 로짓 해시, 다시 생성한 토큰, 지문` + where]
      : [v.score.same ? null : T`점수 매기기가 스텝 ${v.score.first_diff_step}에서 다름`, v.generate.same ? null : T('다시 생성한 토큰 · 로짓 해시가 다름'),
        v.check_seed ? T('점수는 같으니 시드를 확인하세요 — 다른 도구로 다시 쓴 재현 파일은 시드(64비트 정수)가 정밀도를 잃었을 수 있습니다') : null,
        v.fingerprint.same === false ? T('지문이 다름') : null,
        diff.has('precision') ? T('재현 파일과 계산 방식이 다릅니다 — 설정의 계산 방식을 재현 파일과 같게 하고 모델을 다시 불러오세요') : null,
        diff.has('kv8') ? T('재현 파일과 KV 캐시 설정이 다릅니다 — 설정의 KV 캐시 8비트를 재현 파일과 같게 하고 모델을 다시 불러오세요') : null,
        diff.has('fast_attention') ? T('재현 파일과 빠른 어텐션 설정이 다릅니다 — 설정의 빠른 어텐션을 재현 파일과 같게 하고 모델을 다시 불러오세요') : null,
        diff.has('yarn') ? T`재현 파일과 YaRN 배가 다릅니다(재현 파일 ${yTxt(md.yarn || 0)} · 지금 ${yTxt(sv.yarn || 0)}) — 설정의 문맥 길이를 재현 파일을 만들 때와 같게 하고 모델을 다시 불러오세요` : null,
      ].filter(Boolean),
  };
}

async function loadProv() {
  st.provErr = '';
  st.busy = T('해시를 계산하는 중');
  draw();
  try {
    const r = await call('system.provenance');
    if (!r.ok || !r.json) throw new Error(appError(r));
    st.prov = r.json;
  } catch (e) {
    st.provErr = e.message;
  }
  st.busy = '';
  draw();
}

function provView(p) {
  const rows = (p.binaries || []).map(b => el('tr', null,
    el('td', null, String(b.file).replace(/^bin[\\/]/, '')),
    el('td', null, el('code', null, b.sha256 ? b.sha256.slice(0, 12) + '…' : '-')),
    el('td', null, b.reference ? (b.same ? el('span', { class: 'tag ok' }, T('기준과 같음')) : el('span', { class: 'tag err' }, T('기준과 다름')))
      : el('span', { class: 'tag dim' }, T('기준 없음')))));
  return el('div', { class: 'col g8' },
    el('p', { class: 'muted m0' }, T('엔진 바이너리의 SHA-256을 재현성 검증의 기준 빌드와 비교합니다. 직접 빌드한 버전은 다를 수 있습니다.')),
    el('table', { class: 'xp-files' }, el('tbody', null, rows)),
    p.module ? el('p', { class: 'small m0' }, T('커널 모듈'), ' ', el('code', null, `${p.module.file} · ${LAYER[p.module.layer] || p.module.layer} · ${p.module.sha256 ? p.module.sha256.slice(0, 16) + '…' : '-'}`)) : null,
    el('p', { class: 'small muted m0' }, `GeulPU v${(p.app || {}).version || ''}${(p.app || {}).commit ? ' · ' + T('커밋') + ' ' + p.app.commit.slice(0, 7) : ''} · ${T('계약')} ${p.contract || ''}`));
}

function cards() {
  const input = el('input', {
    type: 'file', accept: '.json,application/json', class: 'input', 'aria-label': T('재현 파일 고르기'), disabled: st.busy ? true : null,
    onchange: ev => {
      const f = ev.target.files && ev.target.files[0];
      if (!f) return;
      st.busy = T('재현 파일을 읽는 중');
      draw();
      checkCapsule(f).catch(e => { st.result = { ok: false, lines: [e.message] }; }).finally(() => { st.busy = ''; draw(); });
    },
  });
  const res = st.result;
  return [
    el('div', { class: 'card rp-card' },
      el('h3', null, T('재현 파일 확인')),
      el('p', { class: 'muted' }, T('실행 화면에서 내보낸 재현 파일(geulpu-capsule/1)을 고르면, 같은 모델 파일(SHA-256)로 이 PC에서 점수 매기기와 다시 생성을 해 모든 값이 비트까지 같은지 봅니다. 그 모델을 불러옵니다.')),
      input,
      st.busy ? el('p', { class: 'muted', role: 'status' }, st.busy) : null,
      res ? el('div', { class: 'note-line' + (res.ok ? '' : ' err'), role: 'status' },
        el('div', { class: 'col g8' }, st.fileName ? el('b', null, st.fileName) : null, res.lines.map(t => el('span', null, t)))) : null),
    el('div', { class: 'card rp-card' },
      el('h3', null, T('엔진 · 커널 해시')),
      st.prov ? provView(st.prov) : el('div', null,
        el('button', { type: 'button', class: 'btn sm', disabled: st.busy ? true : null, onclick: loadProv }, T('엔진 · 커널 해시 보기'))),
      st.provErr ? el('p', { class: 'xp-err' }, st.provErr) : null),
  ];
}

function draw() {
  const host = document.getElementById('repro-cards');
  if (host) host.replaceChildren(...cards());
}

document.addEventListener('geulpu:repro-cards', draw);
