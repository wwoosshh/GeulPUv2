// 앱리디자인 E6-2(문서/화면구현.md §6) — 재현 › 검사의 두 카드(연구 패널의 '재현' 탭에서 옮겼다): 재현 파일 확인(geulpu-capsule/1 — 같은 모델
// 파일로 이 PC 에서 점수 매기기와 다시 생성을 해 비트까지 비교)과 엔진 · 커널 해시(system.provenance — 기준 빌드와 비교). app.js 의 검사 탭이
// #repro-cards 를 그리면(geulpu:repro-cards) 그 안을 채운다.
import { T, el } from './dom.js';
import { call } from './ops.js';
import { fpOf } from './research.js';
import { appError } from './runs.js';

const LAYER = { cuda: 'CUDA', vulkan: 'Vulkan' };
const st = { busy: '', result: null, prov: null, provErr: '', fileName: '' };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const eqA = (x, y) => Array.isArray(x) && Array.isArray(y) && x.length === y.length && x.every((v, i) => v === y[i]);

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

async function post(op, bodyText) {
  const r = await call(op, {}, { body: bodyText });
  if (!r.ok) throw new Error(appError(r));
  return r.json;
}

/** 연구 패널의 '재현 파일 불러와 확인'과 같은 판정 — 점수 매기기 · 다시 생성 · 지문, 계산 방식 · KV · 빠른 어텐션 · YaRN 의 까닭 */
async function checkCapsule(file) {
  st.result = null;
  st.fileName = file.name;
  const raw = await file.text();
  let cap;
  try { cap = JSON.parse(raw); } catch (_) { throw new Error(T('JSON 파일이 아닙니다')); }
  if (!cap || cap.format !== 'geulpu-capsule/1' || !cap.request || !cap.result || !cap.model) throw new Error(T('GeulPU 재현 파일이 아닙니다'));
  const rq = cap.request;
  const rs = cap.result;
  // 시드는 64비트 정수라 JSON.parse 가 정밀도를 잃는다 — 원문에서 읽는다. 못 읽으면 다른 시드로 다시 생성해 '다름'이 되므로 멈춘다
  const ri = raw.indexOf('"request"');
  const sm = ri >= 0 ? /"seed"\s*:\s*(\d+)\s*[,}]/.exec(raw.slice(ri)) : null;
  if (rq.temperature > 0 && !sm) throw new Error(T('재현 파일의 시드를 정수로 읽지 못했습니다 — 정밀도를 잃은 사본일 수 있습니다'));
  const s0 = await stateNow();
  const models = Array.isArray(s0.models) ? s0.models : [];
  const m = models.find(x => x.sha256 && x.sha256 === cap.model.sha256);
  if (!m) {
    const same = models.find(x => !x.sha256 && cap.model.size && x.size === cap.model.size);
    throw new Error(same ? T`크기가 같은 모델(${same.name || same.id})이 있지만 SHA-256을 모릅니다 — 모델 화면에서 무결성을 확인한 뒤 다시 불러오세요`
      : T`같은 모델 파일(SHA-256 ${String(cap.model.sha256 || '').slice(0, 12)}…)이 설치되어 있지 않습니다 — ${cap.model.name || ''} ${cap.model.quant || ''}`);
  }
  const s = await ensureModel(m);
  st.busy = T('재현 파일의 토큰을 이 PC에서 다시 계산하는 중');
  draw();
  const P = rq.prompt_tokens.length;
  const n = rs.tokens.length;
  const k = (rs.top_ids[0] || []).length || 1;
  const j1 = await post('run.score', JSON.stringify({
    prompt: rq.prompt_tokens.concat(rs.tokens), echo: true, max_tokens: 0, logprobs: Math.min(k, 20), temperature: 0,
    geulpu: { echo_from: P, logit_hash: true, top_logprobs: k },
  }));
  const c = j1.choices[0];
  const lp = c.logprobs || {};
  const g = j1.geulpu || {};
  const sc = {
    lp: (lp.token_logprobs || []).slice(P), ent: (g.prompt_entropy || []).slice(P), topIds: (g.prompt_top_ids || []).slice(P),
    topLp: (g.prompt_top_logprobs || []).slice(P), hashes: g.prompt_logit_hashes || [],
  };
  let d = 0;
  for (; d < n; d++) {
    if (sc.lp[d] !== rs.token_logprobs[d] || sc.hashes[d] !== rs.logit_hashes[d] || sc.ent[d] !== rs.entropy[d]
      || !eqA(sc.topIds[d] || [], rs.top_ids[d] || []) || !eqA(sc.topLp[d] || [], rs.top_logprobs[d] || [])) break;
  }
  const scoreOk = d === n && sc.lp.length === n;
  st.busy = T('재현 파일의 요청으로 다시 생성하는 중');
  draw();
  const body = { prompt: rq.prompt_tokens, max_tokens: rq.max_tokens, temperature: rq.temperature, cache_prompt: false, geulpu: { logit_hash: true } };
  if (rq.temperature > 0) Object.assign(body, { top_k: rq.top_k, top_p: rq.top_p, min_p: rq.min_p });
  if (rq.step_offset) body.geulpu.step_offset = rq.step_offset;
  let txt = JSON.stringify(body);
  if (rq.temperature > 0 && sm) txt = txt.slice(0, -1) + ',"seed":' + sm[1] + '}';
  const j2 = await post(rq.step_offset ? 'run.branch' : 'run.complete', txt);
  const genOk = eqA((j2.geulpu || {}).tokens || [], rs.tokens) && eqA((j2.geulpu || {}).logit_hashes || [], rs.logit_hashes);
  let fpOk = true;
  if (cap.fingerprint) {
    const v = await fpOf(cap.model.sha256, rq.prompt_tokens, rs.tokens, rs.logit_hashes);
    fpOk = v.fp === cap.fingerprint.token && v.fpLogit === cap.fingerprint.logit;
  }
  const md = cap.made_on || {};
  const sv = s.server || {};
  const precCap = md.precision || 'exact';
  const precHere = sv.precision || 'exact';
  const yTxt = y => (y ? '×' + y : T('YaRN 없음'));
  const here = `${(s.device && s.device.name) || '-'} · ${LAYER[sv.layer] || sv.layer || '-'}`;
  const where = T` (재현 파일에 적힌 만든 곳 ${md.device || '-'} · ${LAYER[md.layer] || md.layer || '-'} → 여기 ${here})`;
  const ok = scoreOk && genOk && fpOk;
  st.result = {
    ok,
    lines: ok ? [T`비트까지 같음 — ${n}스텝의 로그확률 · 엔트로피 · 상위 ${k} · 로짓 해시, 다시 생성한 토큰, 지문` + where]
      : [scoreOk ? null : T`점수 매기기가 스텝 ${d + 1}에서 다름`, genOk ? null : T('다시 생성한 토큰 · 로짓 해시가 다름'),
        !genOk && scoreOk && rq.temperature > 0 ? T('점수는 같으니 시드를 확인하세요 — 다른 도구로 다시 쓴 재현 파일은 시드(64비트 정수)가 정밀도를 잃었을 수 있습니다') : null,
        fpOk ? null : T('지문이 다름'),
        precCap !== precHere ? T('재현 파일과 계산 방식이 다릅니다 — 설정의 계산 방식을 재현 파일과 같게 하고 모델을 다시 불러오세요') : null,
        !!md.kv8 !== !!sv.kv8 ? T('재현 파일과 KV 캐시 설정이 다릅니다 — 설정의 KV 캐시 8비트를 재현 파일과 같게 하고 모델을 다시 불러오세요') : null,
        !!md.fast_attention !== !!sv.fast_attention ? T('재현 파일과 빠른 어텐션 설정이 다릅니다 — 설정의 빠른 어텐션을 재현 파일과 같게 하고 모델을 다시 불러오세요') : null,
        (md.yarn || 0) !== (sv.yarn || 0) ? T`재현 파일과 YaRN 배가 다릅니다(재현 파일 ${yTxt(md.yarn || 0)} · 지금 ${yTxt(sv.yarn || 0)}) — 설정의 문맥 길이를 재현 파일을 만들 때와 같게 하고 모델을 다시 불러오세요` : null,
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
