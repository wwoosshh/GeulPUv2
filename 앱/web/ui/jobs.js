// 앱리디자인 E5(문서/화면구현.md §5 · 화면설계.md §3) — 작업 화면. 앱의 작업(모델 불러오기 · 확인 · 속도 측정 · 검사 · 평가 …)은 한 번에 하나다:
// 지금 작업(상태의 job)의 단계 · 진행 · 걸린 시간 · 작업 로그(job.log) · 중지(job.cancel), 그리고 이 브라우저가 본 지난 작업(최근 30 개 — 앱은 지금
// 작업 하나만 알려 준다). 상태는 app.js 가 1 초마다 읽는 것을 geulpu:state 로 받는다. 실행 중인 작업이 있으면 사이드바의 '작업'에 점.
import { T, TM, el, fmtMs } from './dom.js';
import { call } from './ops.js';

const KEY = 'geulpu.jobs';
const MAX = 30;
const LOG_MAX = 200000;          // 화면에 두는 작업 로그의 글자 수(끝 쪽)
const page = document.getElementById('page-jobs');
const view = document.getElementById('view-jobs');
const st = { job: null, hist: loadHist(), log: { id: 0, from: 0, text: '' }, logBusy: false, cancel: null, sig: '' };

function loadHist() {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) || '[]');
    return Array.isArray(v) ? v.filter(x => x && typeof x === 'object') : [];
  } catch (_) {
    return [];
  }
}
function saveHist() {
  try { localStorage.setItem(KEY, JSON.stringify(st.hist.slice(0, MAX))); } catch (_) { /* 기억하지 못해도 이번에는 */ }
}

/** 상태의 작업을 지난 작업에 — 앱을 다시 켜면 번호가 처음부터라 번호 · 종류 · 제목을 함께 견준다. 바뀌었으면 참 */
function remember(j) {
  if (!j || !j.id) return false;
  const key = x => `${x.id}:${x.kind}:${x.title}`;
  const row = { id: j.id, kind: j.kind || '', title: j.title || '', state: j.state || '', seconds: j.seconds, result: j.result || '', summary: j.summary || '', reason: j.reason || '' };
  const i = st.hist.findIndex(x => key(x) === key(row));
  if (i >= 0) {
    const old = st.hist[i];
    const same = ['state', 'seconds', 'result', 'summary', 'reason'].every(k => old[k] === row[k]);
    if (same) return false;
    st.hist[i] = Object.assign(old, row);
  } else {
    row.at = Date.now();
    st.hist.unshift(row);
    st.hist.length = Math.min(st.hist.length, MAX);
  }
  saveHist();
  return true;
}

const shown = () => view.classList.contains('active');
const secs = s => (typeof s === 'number' ? fmtMs(Math.round(s * 1000)) : '–');
const firstLine = s => String(s || '').split('\n')[0];
function stateTag(state) {
  const [cls, label] = {
    running: ['acc', T('실행 중')], done: ['ok', T('끝', '상태')], failed: ['err', T('실패')], cancelled: ['dim', T('중지됨')],
  }[state] || ['dim', state || T('알 수 없음')];
  return el('span', { class: 'tag ' + cls }, label);
}

async function loadLog() {
  const j = st.job;
  if (!j || st.logBusy || !shown()) return;
  if (st.log.id !== j.id) st.log = { id: j.id, from: 0, text: '' };
  st.logBusy = true;
  try {
    const r = await call('job.log', { from: st.log.from });
    if (r.ok && r.json && r.json.id === j.id) {
      if (r.json.text) st.log.text = (st.log.text + r.json.text).slice(-LOG_MAX);
      st.log.from = r.json.size;
      const pre = page.querySelector('#jobs-log');
      if (pre) {
        const near = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
        pre.textContent = TM(st.log.text) || T('아직 로그가 없습니다');
        if (near) pre.scrollTop = pre.scrollHeight;
      }
    }
  } catch (_) { /* 다음 상태에서 다시 */ }
  st.logBusy = false;
}

async function cancel() {
  st.cancel = { busy: true };
  render(true);
  let r = null;
  try { r = await call('job.cancel', {}); } catch (_) { r = null; }
  st.cancel = r && r.ok ? null : { error: (r && r.json && r.json.error && TM(r.json.error.message)) || T('앱에 닿지 못했습니다') };
  render(true);
}

function current() {
  const j = st.job;
  if (!j || !j.id) return el('div', { class: 'empty-box' }, el('div', { class: 'big' }, T('지금 작업이 없습니다')));
  const running = j.state === 'running';
  const pct = j.state === 'done' ? 100 : Math.max(0, Math.min(100, (j.progress || 0) / 10));
  return el('div', { class: 'card job jobs-now' },
    el('div', { class: 'head' },
      el('div', { class: 't' }, el('div', { class: 'ink h3' }, TM(j.title || j.kind || '')), el('div', { class: 'small muted' }, running ? TM(j.step || '') : '')),
      stateTag(j.state),
      el('span', { class: 'small muted' }, secs(j.seconds)),
      running ? el('button', { type: 'button', class: 'btn sm', disabled: st.cancel && st.cancel.busy ? true : null, onclick: cancel }, T('중지')) : null),
    el('div', { class: 'bar' + (j.state === 'done' ? ' done' : j.state === 'failed' ? ' fail' : '') }, el('i', { style: `width:${pct}%` })),
    j.state === 'done' && j.summary ? el('p', { class: 'jobs-sum' }, TM(firstLine(j.summary))) : null,
    j.reason && j.state !== 'done' ? el('p', { class: 'jobs-sum bad' }, TM(j.reason)) : null,
    st.cancel && st.cancel.error ? el('p', { class: 'jobs-sum bad' }, st.cancel.error) : null,
    el('div', { class: 'small muted' }, T`작업 ${j.id} · ${j.kind || ''}`),
    el('pre', { class: 'log jobs-log', id: 'jobs-log' }, (st.log.id === j.id && TM(st.log.text)) || T('아직 로그가 없습니다')));
}

function history() {
  const rows = st.hist.filter(x => !st.job || x.id !== st.job.id || x.kind !== st.job.kind || x.title !== st.job.title);
  if (!rows.length) return el('p', { class: 'muted' }, T('이 브라우저가 본 지난 작업이 없습니다.'));
  return el('ol', { class: 'jobs-list' }, rows.map(x => el('li', { class: 'jobs-row' },
    el('span', { class: 'id' }, String(x.id)),
    el('span', { class: 't' }, TM(x.title || x.kind),
      x.state === 'done' && x.summary ? el('span', { class: 'muted' }, ' — ' + TM(firstLine(x.summary))) : null,
      x.reason && x.state !== 'done' ? el('span', { class: 'bad' }, ' — ' + TM(x.reason)) : null),
    stateTag(x.state),
    el('span', { class: 'small muted' }, secs(x.seconds)))));
}

function render(force) {
  if (!shown()) return;
  const sig = JSON.stringify([st.job, st.hist.length, st.hist[0] && st.hist[0].state, st.cancel]);
  if (!force && sig === st.sig) return;
  st.sig = sig;
  const pre = page.querySelector('#jobs-log');
  const keep = pre ? pre.scrollTop : null;
  page.replaceChildren(el('div', { class: 'jobs' },
    el('h1', null, T('작업')),
    el('p', { class: 'lead muted' }, T('오래 걸리는 일(모델 불러오기 · 확인 · 속도 측정 · 검사 · 평가 …)은 앱에서 한 번에 하나씩 실행됩니다. 대기열은 없습니다 — 작업이 실행 중이면 새 작업을 받지 않으니, 그 작업이 끝난 뒤에 다시 시작하세요.')),
    el('h2', null, T('지금 작업')),
    current(),
    el('h2', null, T('지난 작업'), ' ', el('span', { class: 'muted' }, T('이 브라우저가 본 것 — 최근 30 개'))),
    history()));
  const pre2 = page.querySelector('#jobs-log');
  if (pre2 && keep !== null) pre2.scrollTop = keep;
}

document.addEventListener('geulpu:state', ev => {
  const s = ev.detail || {};
  const j = s.job && s.job.id ? s.job : null;
  const changed = remember(j);
  st.job = j;
  const rail = document.querySelector('.rail-btn[data-group="jobs"]');
  if (rail) rail.classList.toggle('busy', !!(j && j.state === 'running'));
  if (shown()) {
    render(changed);
    if (j && (j.state === 'running' || st.log.id !== j.id || st.log.from < (j.log_size || 0))) loadLog();
  }
});
document.addEventListener('geulpu:view', ev => {
  if (ev.detail && ev.detail.view === 'jobs') {
    st.sig = '';
    render(true);
    loadLog();
  }
});
