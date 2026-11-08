// 앱리디자인 E2(문서/화면구현.md §2 · 화면설계.md §3) — 실험 화면: 실험 목록(첫 화면) · 새 실험 · 실험 하나(헤더 · 실행 정보 줄 · 탭 계획 · 결과 ·
// 기록). 앱은 기능 목록 클라이언트(ops.js)로만 부르고, 실험 화면에서 부른 기능은 X-GeulPU-Experiment 를 붙여 그 실험의 명령 기록에 남긴다.
// 다른 곳(CLI · AI 세션)에서 이 실험으로 부른 호출이 끝나면 명령 창이 넘겨 준 이벤트(geulpu:event)로 다시 읽는다.
import { toCommand } from './command.js';
import { EN, T, TM, baseName, clock, copyButton, el, fmtMs, sourceLabel } from './dom.js';
import { call, registry } from './ops.js';

const page = document.getElementById('page-exp');
const view = document.getElementById('view-exp');
const ROOT_KEY = 'geulpu.expRoot';
const TABS = [['plan', T('계획')], ['results', T('결과', '탭')], ['records', T('기록')]];

const st = {
  root: loadRoot(),     // 실험 폴더 위치('' 이면 앱의 기본 — 데이터 폴더의 experiments)
  list: null,           // experiment.list 의 답
  listError: null,
  cur: null,            // 연 실험의 경로
  exp: null,            // experiment.show 의 답
  expError: null,
  tab: 'plan',
  docs: {},             // 파일 → {text} · {missing} · {error}
  commands: null,       // 명령 기록의 줄(객체)
  check: null,          // 점검 작업
};
let reg = null;
registry().then(r => { reg = r; if (st.cur) render(); }).catch(() => { /* 명령 기록은 기능 id 로 보인다 */ });

const sleep = ms => new Promise(r => setTimeout(r, ms));
const samePath = (a, b) => String(a || '').replace(/[\\/]+$/, '').replace(/\//g, '\\').toLowerCase()
  === String(b || '').replace(/[\\/]+$/, '').replace(/\//g, '\\').toLowerCase();
const shown = () => view.classList.contains('active');

function loadRoot() {
  try { return localStorage.getItem(ROOT_KEY) || ''; } catch (_) { return ''; }
}
function setRoot(p) {
  st.root = p || '';
  try {
    if (st.root) localStorage.setItem(ROOT_KEY, st.root);
    else localStorage.removeItem(ROOT_KEY);
  } catch (_) { /* 기억하지 못해도 이번에는 그 위치로 */ }
  loadList();
}
const rootParam = () => (st.root ? { root: st.root } : {});
const appError = r => TM((r && r.json && r.json.error && r.json.error.message) || (r ? 'HTTP ' + r.status : T('앱에 닿지 못했습니다')));

function statusLabel(s) {
  return { planned: T('계획', '상태'), running: T('진행 중'), done: T('끝', '상태') }[s] || s || T('알 수 없음');
}
function resultTag(r) {
  if (r === 'pass') return el('span', { class: 'tag ok' }, T('통과'));
  if (r === 'fail') return el('span', { class: 'tag err' }, T('실패'));
  return el('span', { class: 'tag dim' }, T('아직'));
}
function when(t) {
  if (typeof t !== 'number') return String(t || '');
  const d = new Date(t * 1000);
  return [d.getFullYear(), String(d.getMonth() + 1).padStart(2, '0'), String(d.getDate()).padStart(2, '0')].join('-') + ' ' + clock(t);
}

// ---- 읽기 ----

async function loadList() {
  st.listError = null;
  try {
    const r = await call('experiment.list', rootParam());
    if (r.ok && r.json) st.list = r.json;
    else st.listError = appError(r);
  } catch (_) {
    st.listError = T('앱에 닿지 못했습니다');
  }
  if (!st.cur) render();
}

async function loadExp() {
  const path = st.cur;
  try {
    const r = await call('experiment.show', { path });
    if (path !== st.cur) return;
    if (r.ok && r.json) { st.exp = r.json; st.expError = null; } else st.expError = appError(r);
  } catch (_) {
    st.expError = T('앱에 닿지 못했습니다');
  }
  render();
}

async function loadDoc(name) {
  const path = st.cur;
  st.docs[name] = { loading: true };
  let d;
  try {
    const r = await call('experiment.read', { path, file: name });
    if (r.ok && r.json) d = { text: r.json.text };
    else if (r.status === 404) d = { missing: true };
    else d = { error: appError(r) };
  } catch (_) {
    d = { error: T('앱에 닿지 못했습니다') };
  }
  if (path !== st.cur) return;
  st.docs[name] = d;
  render();
}

async function loadCommands() {
  const path = st.cur;
  st.commands = 'loading';
  let rows = [];
  try {
    const r = await call('experiment.read', { path, file: 'records/commands.jsonl' });
    if (r.ok && r.json) {
      for (const ln of String(r.json.text).split('\n')) {
        if (!ln.trim()) continue;
        try { rows.push(JSON.parse(ln)); } catch (_) { rows.push({ broken: ln }); }
      }
    } else if (r.status !== 404) rows = { error: appError(r) };
  } catch (_) {
    rows = { error: T('앱에 닿지 못했습니다') };
  }
  if (path !== st.cur) return;
  st.commands = rows;
  render();
}

function openExp(path) {
  st.cur = path;
  st.exp = null;
  st.expError = null;
  st.tab = 'plan';
  st.docs = {};
  st.commands = null;
  st.check = null;
  render();
  loadExp();
  view.scrollTop = 0;
}
function backToList() {
  st.cur = null;
  render();
  loadList();
}
function setTab(k) {
  st.tab = k;
  render();
}

// ---- 실험 안의 호출 — 점검(experiment.check, 작업). X-GeulPU-Experiment 로 그 실험의 명령 기록에 남는다 ----

async function runCheck() {
  const path = st.cur;
  const exp = st.exp.path;
  st.check = { running: true, job: null };
  render();
  let r;
  try {
    r = await call('experiment.check', { path: exp }, { experiment: exp });
  } catch (_) {
    r = null;
  }
  if (!r || !r.ok || !r.json || typeof r.json.job !== 'number') {
    if (path === st.cur) { st.check = { error: appError(r) }; render(); }
    return;
  }
  const job = r.json.job;
  if (path === st.cur) { st.check.job = job; render(); }
  for (;;) {
    await sleep(500);
    let s;
    try { s = await call('system.state'); } catch (_) { continue; }
    const j = s && s.json && s.json.job;
    if (!j) continue;
    if (j.id === job && j.state !== 'running') {
      if (path === st.cur) { st.check = { done: true, job, state: j.state, summary: j.summary, reason: j.reason, result: j.result }; render(); }
      return;
    }
    if (j.id > job) {
      if (path === st.cur) { st.check = { error: T('결과를 읽기 전에 다음 작업이 시작되었습니다') }; render(); }
      return;
    }
  }
}

// ---- 새 실험 ----

function newExperiment() {
  const name = el('input', { type: 'text', id: 'xp-name', autocomplete: 'off', spellcheck: 'false', placeholder: 'margin-branch', 'aria-describedby': 'xp-name-hint' });
  const title = el('input', { type: 'text', id: 'xp-title', autocomplete: 'off' });
  const question = el('textarea', { id: 'xp-q', rows: '3' });
  const err = el('p', { class: 'xp-err', role: 'alert' });
  const form = el('div', { class: 'xp-form' },
    el('h2', null, T('새 실험')),
    el('label', { for: 'xp-name' }, T('이름')), name,
    el('p', { class: 'muted', id: 'xp-name-hint' }, T('영어 소문자 · 숫자 · - 만 씁니다. 폴더 이름은 오늘 날짜와 이 이름입니다.')),
    el('label', { for: 'xp-title' }, T('제목')), title,
    el('label', { for: 'xp-q' }, T('연구 질문')), question,
    err);
  const ok = el('button', {
    class: 'btn primary', type: 'button',
    onclick: async () => {
      const n = name.value.trim();
      if (!/^[a-z0-9][a-z0-9-]{0,63}$/.test(n)) {
        err.textContent = T('이름은 영어 소문자 · 숫자로 시작하고 영어 소문자 · 숫자 · - 만 64 자까지 씁니다');
        name.focus();
        return;
      }
      ok.disabled = true;
      let r;
      try {
        r = await call('experiment.create', { name: n, title: title.value.trim(), question: question.value.trim(), ...rootParam() });
      } catch (_) {
        r = null;
      }
      ok.disabled = false;
      if (!r || !r.ok || !r.json || !r.json.path) {
        err.textContent = appError(r);
        return;
      }
      globalThis.closeModal();
      openExp(r.json.path);
    },
  }, T('만들기'));
  const cancel = el('button', { class: 'btn ghost', type: 'button', onclick: () => globalThis.closeModal() }, T('취소'));
  globalThis.modal(form, [cancel, ok], false, '#xp-name');
}

function changeRoot() {
  if (typeof globalThis.browse !== 'function') return;
  globalThis.browse((st.list && st.list.root) || st.root || '', p => setRoot(p),
    { title: T('실험 폴더 위치 선택'), lead: T('실험 폴더들을 담을 폴더를 선택하세요. 이 브라우저에 기억합니다.') });
}

// ---- 그리기 ----

function render() {
  if (st.cur) renderExp();
  else renderList();
}

function expItem(x) {
  const c = x.criteria || { total: 0, pass: 0, fail: 0 };
  return el('button', { class: 'xp-item', role: 'listitem', type: 'button', onclick: () => openExp(x.path) },
    el('span', { class: 't' }, x.title || x.id),
    el('span', { class: 'tag dim' }, statusLabel(x.status)),
    el('span', { class: 'meta' },
      el('span', { class: 'id' }, x.id),
      x.models && x.models.length ? el('span', null, x.models.join(' · ')) : null,
      el('span', null, T`실행 ${x.runs || 0}`),
      el('span', null, T`판정 ${c.pass} / ${c.total} 통과`),
      c.fail ? el('span', { class: 'bad' }, T`실패 ${c.fail}`) : null,
      el('span', null, T`명령 ${x.commands || 0}`)));
}

function renderList() {
  const kids = [
    el('h1', null, T('실험')),
    el('p', { class: 'lead' }, T('실험마다 연구 질문 · 판정 기준 · 실행 · 기록이 한 폴더에 남습니다.')),
    el('div', { class: 'xp-root' },
      el('span', null, T('실험 폴더 위치')),
      el('code', null, (st.list && st.list.root) || st.root || '…'),
      el('button', { class: 'btn sm', type: 'button', onclick: changeRoot }, T('위치 바꾸기')),
      st.root ? el('button', { class: 'btn sm ghost', type: 'button', onclick: () => setRoot('') }, T('기본 위치로')) : null),
    el('div', { class: 'xp-actions' }, el('button', { class: 'btn primary', type: 'button', onclick: newExperiment }, T('새 실험'))),
  ];
  if (st.listError) {
    kids.push(el('div', { class: 'empty-box' }, el('div', { class: 'big' }, T('실험 목록을 읽지 못했습니다')), el('div', null, st.listError)));
  } else if (!st.list) {
    kids.push(el('p', { class: 'muted' }, T('읽는 중…')));
  } else {
    if (!st.list.experiments.length) {
      kids.push(el('div', { class: 'empty-box' }, el('div', { class: 'big' }, T('아직 실험이 없습니다')),
        el('div', null, T('새 실험을 만들거나, CLI 에서 GeulPU experiment create --name <이름> 으로 만듭니다.'))));
    }
    kids.push(el('div', { class: 'xp-list', role: 'list', 'aria-label': T('실험 목록') },
      st.list.experiments.map(expItem),
      el('button', { class: 'xp-item', role: 'listitem', type: 'button', onclick: () => globalThis.show && globalThis.show('chat') },
        el('span', { class: 't' }, T('대화')),
        el('span', { class: 'tag dim' }, T('채팅')),
        el('span', { class: 'meta' }, el('span', null, T('채팅 화면으로 — 대화는 지금 이 브라우저에 저장됩니다'))))));
  }
  page.replaceChildren(el('div', { class: 'xp' }, kids));
}

function runInfo(e) {
  const parts = [];
  for (const m of e.models || []) {
    const name = typeof m === 'string' ? m : (m && (m.name || m.file)) || '';
    const sha = m && typeof m === 'object' && m.sha256 ? String(m.sha256).slice(0, 8) : '';
    parts.push(el('span', null, T('모델', '하나'), ' ', el('b', null, name), sha ? el('code', null, ' ' + sha) : null));
  }
  const g = e.geulpu || {};
  if (g.mode) parts.push(el('span', null, T('계산 방식'), ' ', el('b', null, g.mode === 'exact' ? T('정밀') : g.mode === 'fast' ? T('빠름') : g.mode)));
  if (g.version) parts.push(el('span', null, 'GeulPU ', el('b', null, g.version)));
  return parts.length ? el('div', { class: 'xp-prov', 'aria-label': T('실행 정보') }, parts) : null;
}

function criteriaList(e) {
  const cs = Array.isArray(e.criteria) ? e.criteria : [];
  if (!cs.length) return el('p', { class: 'muted' }, T('판정 기준이 없습니다'));
  return el('ul', { class: 'xp-criteria' }, cs.map(c => el('li', null, el('code', null, c.id || ''), el('span', null, c.text || ''), resultTag(c.result))));
}

function docBlock(name) {
  const d = st.docs[name];
  if (d === undefined) {
    loadDoc(name);
    return el('p', { class: 'muted' }, T('읽는 중…'));
  }
  if (d.loading) return el('p', { class: 'muted' }, T('읽는 중…'));
  if (d.missing) return el('p', { class: 'muted' }, T`아직 없습니다: ${name}`);
  if (d.error) return el('p', { class: 'muted' }, d.error);
  const box = el('div', { class: 'xp-doc' });
  if (typeof globalThis.md === 'function') box.innerHTML = globalThis.md(d.text);
  else box.append(el('pre', { class: 'xp-pre' }, d.text));
  return box;
}

function commandRow(c) {
  if (c.broken !== undefined) return el('li', { class: 'c-line fail' }, el('div', { class: 'msg' }, T('읽지 못한 줄'), ' ', el('code', null, c.broken.slice(0, 200))));
  const [label, kind] = sourceLabel(c.source, c.session);
  const params = c.params === undefined ? null : c.params;
  const short = reg ? toCommand(reg, c.op, params, 80) : c.op;
  const ok = c.status >= 200 && c.status < 400;
  return el('li', { class: 'c-line ' + (ok ? 'done' : 'fail') },
    el('div', { class: 'who' }, el('span', { class: 'src ' + kind }, label), el('span', { class: 'op' }, c.op), el('span', { class: 'grow' }),
      el('span', { class: 'at' }, when(c.time))),
    el('div', { class: 'cmd' }, short),
    el('div', { class: 'res' }, el('span', { class: 'state' }, `${c.status} · ${fmtMs(c.ms || 0)}`),
      reg ? copyButton(() => toCommand(reg, c.op, params, 0)) : null));
}

function checkBox() {
  const c = st.check;
  if (c.error) return el('p', { class: 'xp-check' }, el('span', { class: 'tag err' }, T('점검하지 못함')), c.error);
  if (c.running) return el('p', { class: 'xp-check' }, el('span', { class: 'tag acc' }, T('점검 중')), c.job ? T`작업 ${c.job}` : '');
  const good = c.state === 'done';
  return el('div', { class: 'xp-check-done' },
    el('p', { class: 'xp-check' }, el('span', { class: 'tag ' + (good ? 'ok' : 'err') }, good ? T('통과') : T('실패')), TM(c.summary || c.reason || '')),
    c.result !== undefined && c.result !== null ? el('pre', { class: 'xp-pre' }, typeof c.result === 'string' ? c.result : JSON.stringify(c.result, null, 2)) : null);
}

function planTab(e) {
  return [
    el('h2', null, T('판정 기준'), ' ', el('span', { class: 'muted' }, T('(실행 전에 적음)'))),
    criteriaList(e),
    el('h2', null, 'plan.md'),
    docBlock('plan.md'),
  ];
}

function resultsTab(e) {
  return [el('h2', null, T('판정')), criteriaList(e), el('h2', null, 'results.md'), docBlock('results.md')];
}

function recordsTab(e) {
  const out = [
    el('div', { class: 'xp-actions' },
      el('button', { class: 'btn', type: 'button', disabled: st.check && st.check.running ? true : null, onclick: runCheck }, T('점검')),
      el('span', { class: 'muted' }, T('형식과 파일 해시를 점검 도구로 확인합니다(experiment check).'))),
  ];
  if (st.check) out.push(checkBox());
  out.push(el('h2', null, T('명령 기록'), ' ', el('span', { class: 'muted' }, 'records/commands.jsonl')));
  if (st.commands === null) {
    loadCommands();
    out.push(el('p', { class: 'muted' }, T('읽는 중…')));
  } else if (st.commands === 'loading') {
    out.push(el('p', { class: 'muted' }, T('읽는 중…')));
  } else if (st.commands.error) {
    out.push(el('p', { class: 'muted' }, st.commands.error));
  } else if (!st.commands.length) {
    out.push(el('p', { class: 'muted' }, T`아직 없습니다 — 이 실험 화면에서 기능을 부르면 남고, CLI 에서는 명령에 이 옵션을 붙입니다: --experiment ${e.id}`));
  } else {
    out.push(el('ol', { class: 'xp-cmds' }, st.commands.map(commandRow)));
  }
  const files = Array.isArray(e.files) ? e.files : [];
  out.push(el('h2', null, T('파일'), ' ', el('span', { class: 'muted' }, T('experiment.json 의 files'))));
  out.push(files.length
    ? el('table', { class: 'xp-files' }, el('tbody', null, files.map(f => el('tr', null,
      el('td', null, el('code', null, f.path || '')), el('td', null, typeof f.bytes === 'number' ? f.bytes.toLocaleString(EN() ? 'en-US' : 'ko-KR') + ' B' : ''),
      el('td', null, el('code', null, f.sha256 ? String(f.sha256).slice(0, 8) : ''))))))
    : el('p', { class: 'muted' }, T('목록에 적은 파일이 없습니다')));
  const rep = e.reproduce && Array.isArray(e.reproduce.commands) ? e.reproduce.commands : [];
  if (rep.length) out.push(el('h2', null, T('재현 명령')), el('pre', { class: 'xp-pre' }, rep.join('\n')));
  return out;
}

function renderExp() {
  const crumbs = el('nav', { class: 'xp-crumbs', 'aria-label': T('경로') },
    el('button', { type: 'button', onclick: backToList }, T('실험')),
    el('span', { class: 'sep', 'aria-hidden': 'true' }, '›'),
    el('span', { class: 'here' }, st.exp ? st.exp.experiment.id || baseName(st.cur) : baseName(st.cur)));
  if (st.expError) {
    page.replaceChildren(el('div', { class: 'xp' }, crumbs, el('div', { class: 'empty-box' }, el('div', { class: 'big' }, T('실험을 읽지 못했습니다')), el('div', null, st.expError))));
    return;
  }
  if (!st.exp) {
    page.replaceChildren(el('div', { class: 'xp' }, crumbs, el('p', { class: 'muted' }, T('읽는 중…'))));
    return;
  }
  const e = st.exp.experiment || {};
  const kids = [
    crumbs,
    el('h1', null, e.title || e.id),
    e.question ? el('p', { class: 'lead' }, e.question) : null,
    el('div', { class: 'xp-meta' },
      el('span', { class: 'tag dim' }, statusLabel(e.status)),
      e.created ? el('span', null, T`만든 날 ${e.created}`) : null,
      Array.isArray(e.authors) && e.authors.length ? el('span', null, e.authors.join(', ')) : null,
      el('span', { class: 'id', title: st.exp.path }, baseName(st.exp.path))),
    runInfo(e),
    el('div', { class: 'xp-tabs', role: 'tablist', 'aria-label': T('실험 탭') },
      TABS.map(([k, label]) => el('button', {
        type: 'button', role: 'tab', class: 'xp-tab' + (st.tab === k ? ' on' : ''), 'aria-selected': String(st.tab === k),
        onclick: () => setTab(k),
      }, label))),
    el('section', { class: 'xp-panel', role: 'tabpanel' }, st.tab === 'plan' ? planTab(e) : st.tab === 'results' ? resultsTab(e) : recordsTab(e)),
  ];
  page.replaceChildren(el('div', { class: 'xp' }, kids));
}

// ---- 화면 전환 · 이벤트 ----

document.addEventListener('geulpu:view', ev => {
  if (ev.detail && ev.detail.view === 'exp' && (!ev.detail.same || !page.childElementCount)) {
    if (st.cur) loadExp();
    else loadList();
  }
});

document.addEventListener('geulpu:open-experiment', ev => {
  const path = ev.detail && ev.detail.path;
  if (!path) return;
  if (typeof globalThis.show === 'function') globalThis.show('exp');
  openExp(path);
});

// 이 실험에 걸린 호출이 끝나면(어느 출처든) 다시 읽는다 — start 의 experiment 를 호출 번호로 기억해 둔다
const callExperiment = new Map();
let refreshTimer = 0;
let refreshFor = null;
function scheduleRefresh(exp) {
  refreshFor = exp || refreshFor;
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => {
    const exp2 = refreshFor;
    refreshFor = null;
    if (!shown()) return;
    if (st.cur) {
      if (!exp2 || samePath(exp2, st.cur) || samePath(exp2, st.exp && st.exp.path)) {
        st.commands = null;
        loadExp();
      }
    } else loadList();
  }, 300);
}
document.addEventListener('geulpu:event', ev => {
  const e = ev.detail;
  if (!e) return;
  if (e.type === 'start') {
    if (e.experiment) callExperiment.set(e.call, e.experiment);
    if (callExperiment.size > 1000) callExperiment.clear();
    return;
  }
  if (e.type !== 'end') return;
  const exp = callExperiment.get(e.call);
  callExperiment.delete(e.call);
  if (exp || (e.op && e.op.startsWith('experiment.'))) scheduleRefresh(exp);
});

// 앱이 이미 이 화면을 열었으면(모듈이 늦게 읽혔을 때) 바로 읽는다
if (shown()) loadList();
