// 앱리디자인 E2(문서/화면구현.md §2 · 화면설계.md §3) — 실험 화면: 실험 목록(첫 화면) · 새 실험 · 실험 하나(헤더 · 실행 정보 줄 · 탭 계획 · 실행 ·
// 비교 · 결과 · 기록). 앱은 기능 목록 클라이언트(ops.js)로만 부르고, 실험 화면에서 부른 기능은 X-GeulPU-Experiment 를 붙여 그 실험의 명령 기록에
// 남긴다. 다른 곳(CLI · AI 세션)에서 이 실험으로 부른 호출이 끝나면 명령 창이 넘겨 준 이벤트(geulpu:event)로 다시 읽는다.
// E3(문서/화면구현.md §3) — 실행 · 비교 탭과 실행 화면(ui/runs.js), 계획 · 결과 탭의 편집(experiment.update).
// E4(문서/화면구현.md §4) — 스텝 · 층 화면(ui/steps.js — 경로: 실험 › id › 실행 › 스텝 k).
// E5(문서/화면구현.md §5) — 기본 실험 "대화"(채팅의 답이 실행으로 남는 실험 — 찾거나 만든다), 기록 탭의 '디버거의 실행 로그'.
import { toCommand } from './command.js';
import { EN, T, TM, baseName, clock, copyButton, el, fmtMs, samePath, sourceLabel } from './dom.js';
import { call, registry } from './ops.js';
import { awaitRun, compareTab, forgetErrors, runTitle, runView, runsTab, selectStep, setCompare } from './runs.js';
import { stepTitle, stepView } from './steps.js';

const page = document.getElementById('page-exp');
const view = document.getElementById('view-exp');
const ROOT_KEY = 'geulpu.expRoot';
const TABS = [['plan', T('계획')], ['runs', T('실행', '탭')], ['compare', T('비교', '탭')], ['results', T('결과', '탭')], ['records', T('기록')]];

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
  run: null,            // E3 — 연 실행의 id(실행 화면)
  step: null,           // E4 — 연 스텝(실행 안의 위치 — 스텝 · 층 화면), 실행 화면이면 null
  edit: null,           // E3 — 편집 중인 탭('plan' · 'results')과 그 값
  saving: false,
  editError: null,
};
let reg = null;
registry().then(r => { reg = r; if (st.cur) render(); }).catch(() => { /* 명령 기록은 기능 id 로 보인다 */ });

const sleep = ms => new Promise(r => setTimeout(r, ms));
const shown = () => view.classList.contains('active');

// E3 — 명령 창이 입력 줄에 이 실험을 붙이게(화면설계.md §5): 실험을 열거나 닫을 때, 다른 화면으로 갈 때 알린다
function announce() {
  const path = shown() && st.cur && st.exp ? st.exp.path : null;
  document.dispatchEvent(new CustomEvent('geulpu:context', { detail: { experiment: path, id: path ? (st.exp.experiment || {}).id || baseName(path) : null } }));
}

/** runs.js 에 넘기는 연 실험 */
function ctx() {
  return {
    path: st.exp.path,
    exp: st.exp,
    rerender: render,
    reload: () => loadExp(),
    openRun: id => openRun(id),
    openStep: (id, k) => openStep(id, k),
    openCompare: (a, b) => {
      setCompare(ctx(), a, b);
      st.run = null;
      st.step = null;
      st.tab = 'compare';
      render();
      view.scrollTop = 0;
    },
  };
}

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
    if (r.ok && r.json) {
      st.exp = r.json;
      st.expError = null;
      forgetErrors(r.json.path);
    } else st.expError = appError(r);
  } catch (_) {
    st.expError = T('앱에 닿지 못했습니다');
  }
  announce();
  if (!st.edit) render();          // 편집하는 동안은 그리지 않는다(입력이 지워지지 않게) — 저장하면 그린다
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
  st.run = null;
  st.step = null;
  st.edit = null;
  st.editError = null;
  render();
  loadExp();
  view.scrollTop = 0;
}
function backToList() {
  st.cur = null;
  st.run = null;
  st.step = null;
  st.edit = null;
  announce();
  render();
  loadList();
}
function setTab(k) {
  st.tab = k;
  st.run = null;
  st.step = null;
  render();
}
function openRun(id) {
  st.run = id;
  st.step = null;
  st.edit = null;
  render();
  view.scrollTop = 0;
}
function closeRun() {
  st.run = null;
  st.step = null;
  st.tab = 'runs';
  render();
}
// E4 — 스텝 · 층 화면. 화면을 바꾸면 그 화면에 초점(← → 로 바로 옮기게)
function openStep(id, k) {
  const moved = st.run !== id || st.step === null;
  st.run = id;
  st.step = k;
  st.edit = null;
  render();
  const r = page.querySelector('.st-view');
  if (r) r.focus({ preventScroll: true });
  if (moved) view.scrollTop = 0;
}
function closeStep() {
  if (st.exp && st.run !== null && st.step !== null) selectStep(ctx(), st.run, st.step);
  st.step = null;
  render();
  view.scrollTop = 0;
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
  // F3-1 — 접근성 이름을 라벨 글로(placeholder 'margin-branch' 가 이름으로 잡혔다)
  const name = el('input', { type: 'text', id: 'xp-name', autocomplete: 'off', spellcheck: 'false', placeholder: 'margin-branch', 'aria-label': T('이름'), 'aria-describedby': 'xp-name-hint' });
  const title = el('input', { type: 'text', id: 'xp-title', autocomplete: 'off' });
  const question = el('textarea', { id: 'xp-q', rows: '3' });
  // F3-2 — 저자(쉼표로 여럿) — experiment.json 의 authors. 비우면 템플릿의 글이 남고 점검이 경고한다
  const authors = el('input', { type: 'text', id: 'xp-authors', autocomplete: 'off', 'aria-describedby': 'xp-authors-hint' });
  const err = el('p', { class: 'xp-err', role: 'alert' });
  const form = el('div', { class: 'xp-form' },
    el('h2', null, T('새 실험')),
    el('label', { for: 'xp-name' }, T('이름')), name,
    el('p', { class: 'muted', id: 'xp-name-hint' }, T('영어 소문자 · 숫자 · - 만 씁니다. 폴더 이름은 오늘 날짜와 이 이름입니다.')),
    el('label', { for: 'xp-title' }, T('제목')), title,
    el('label', { for: 'xp-q' }, T('연구 질문')), question,
    el('label', { for: 'xp-authors' }, T('저자')), authors,
    el('p', { class: 'muted', id: 'xp-authors-hint' }, T('여럿이면 쉼표로 나눕니다.')),
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
      // F3-2 — 템플릿은 화면 언어로(lang), 저자는 쉼표로 나눠 배열로
      const who = authors.value.split(',').map(s => s.trim()).filter(Boolean);
      const body = { name: n, title: title.value.trim(), question: question.value.trim(), lang: EN() ? 'en' : 'ko', ...rootParam() };
      if (who.length) body.authors = who;
      try {
        r = await call('experiment.create', body);
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

// ---- E5 — 기본 실험 "대화"(문서/화면구현.md §5) ----
// 채팅의 답이 실행으로 남는 실험: 지금 실험 폴더 위치의 `<날짜>-chat`(제목 "대화" · "Chat"). 처음 보낼 때 찾거나 만들고 경로를 이 브라우저에
// 기억한다. app.js 의 채팅이 globalThis.geulpuChatExperiment 로 묻고, 그 답의 실행 id 는 geulpuAwaitRun 으로(end 이벤트) 받는다
const CHAT_KEY = 'geulpu.chatExp';
const CHAT_TITLES = ['대화', 'Chat', 'Conversations'];   // 만들 때의 화면 언어로(T('대화')) — 어느 언어로 만들었어도 찾는다
function isChatExp(x) { return !!x && /-chat$/.test(String(x.id || '')) && CHAT_TITLES.includes(x.title); }
let chatOk = null;            // 이번에 확인한 경로
let chatFind = null;          // 찾거나 만드는 중이면 그 약속(한 번에 하나)
function chatSaved() {
  try {
    const v = JSON.parse(localStorage.getItem(CHAT_KEY) || 'null');
    return v && v.root === (st.root || '') && v.path ? v.path : null;
  } catch (_) {
    return null;
  }
}
function chatSave(path) {
  try { localStorage.setItem(CHAT_KEY, JSON.stringify({ root: st.root || '', path })); } catch (_) { /* 다음에 다시 찾는다 */ }
}
function chatExperiment() {
  const saved = chatSaved();
  if (saved && chatOk === saved) return Promise.resolve(saved);
  if (chatFind) return chatFind;
  chatFind = (async () => {
    if (saved) {
      const r = await call('experiment.show', { path: saved });
      if (r.ok && r.json && isChatExp(r.json.experiment)) return (chatOk = saved);
    }
    const lr = await call('experiment.list', rootParam());
    const hit = lr.ok && lr.json && Array.isArray(lr.json.experiments) ? lr.json.experiments.find(isChatExp) : null;
    if (hit) {
      chatSave(hit.path);
      return (chatOk = hit.path);
    }
    const cr = await call('experiment.create', Object.assign({ name: 'chat', title: T('대화'), question: T('가벼운 대화 — 채팅의 답마다 실행으로 남습니다') }, rootParam()));
    if (cr.ok && cr.json && cr.json.path) {
      chatSave(cr.json.path);
      if (!st.cur) loadList();
      return (chatOk = cr.json.path);
    }
    return null;
  })().catch(() => null).finally(() => { chatFind = null; });
  return chatFind;
}
globalThis.geulpuChatExperiment = chatExperiment;
globalThis.geulpuAwaitRun = (exp, body) => awaitRun(null, exp, body);
// 채팅 화면 위의 경로 '실험 › 대화'(ui/nav.js) — 그 실험의 화면으로(아직 없으면 실험 목록)
document.addEventListener('geulpu:open-chat-experiment', () => {
  if (typeof globalThis.show === 'function') globalThis.show('exp');
  const p = chatSaved();
  if (p) openExp(p);
  else if (st.cur) backToList();
});

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
    // E5 — 기본 실험 "대화"는 맨 아래 한 줄(누르면 채팅 화면 — 그 실험의 화면은 채팅 화면 위의 경로에서)
    const chat = st.list.experiments.find(isChatExp) || null;
    kids.push(el('div', { class: 'xp-list', role: 'list', 'aria-label': T('실험 목록') },
      st.list.experiments.filter(x => x !== chat).map(expItem),
      el('button', { class: 'xp-item', role: 'listitem', type: 'button', onclick: () => globalThis.show && globalThis.show('chat') },
        el('span', { class: 't' }, T('대화')),
        el('span', { class: 'tag dim' }, T('채팅')),
        el('span', { class: 'meta' }, chat
          ? [el('span', { class: 'id' }, chat.id), el('span', null, T`실행 ${chat.runs || 0}`), el('span', null, T('채팅의 답마다 실행으로 남습니다'))]
          : el('span', null, T('채팅 화면으로 — 처음 보내면 이 실험이 만들어지고 답마다 실행으로 남습니다'))))));
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
  return el('ul', { class: 'xp-criteria' }, cs.map(c => el('li', null, el('code', null, c.id || ''),
    el('span', null, c.text || '', c.note ? el('span', { class: 'note' }, c.note) : null), resultTag(c.result))));
}

// ---- E3 — 계획 · 결과의 편집(experiment.update — 이 실험의 명령 기록에 남는다) ----

const runCount = e => (Array.isArray(e.runs) ? e.runs.length : 0);

async function startEdit(tab) {
  const e = st.exp.experiment || {};
  const doc = tab === 'plan' ? 'plan.md' : 'results.md';
  if (!st.docs[doc] || st.docs[doc].loading) await loadDoc(doc);
  const d = st.docs[doc] || {};
  const cs = Array.isArray(e.criteria) ? e.criteria : [];
  st.editError = null;
  st.edit = {
    tab,
    title: e.title || '',
    question: e.question || '',
    doc: d.text || '',
    status: e.status || 'planned',
    criteria: cs.map(c => ({ id: String(c.id || ''), text: String(c.text || ''), result: c.result || 'pending', note: c.note || '' })),
  };
  if (tab === 'plan' && !st.edit.criteria.length) st.edit.criteria.push({ id: 'C1', text: '', result: 'pending', note: '' });
  render();
}

function cancelEdit() {
  st.edit = null;
  st.editError = null;
  render();
}

async function saveEdit() {
  const ed = st.edit;
  let body;
  if (ed.tab === 'plan') {
    const crit = ed.criteria.map(c => ({ id: c.id.trim(), text: c.text.trim(), result: c.result, note: c.note })).filter(c => c.id || c.text);
    const ids = crit.map(c => c.id);
    if (!ed.title.trim() || !ed.question.trim()) st.editError = T('제목과 연구 질문을 적으세요');
    else if (!crit.length || crit.some(c => !c.id || !c.text)) st.editError = T('판정 기준마다 id 와 내용을 적으세요(하나 이상)');
    else if (new Set(ids).size !== ids.length) st.editError = T('판정 기준의 id 가 겹칩니다');
    else st.editError = null;
    if (st.editError) { render(); return; }
    body = { path: st.exp.path, title: ed.title.trim(), question: ed.question.trim(), plan: ed.doc, criteria: crit.map(c => (c.note ? c : { id: c.id, text: c.text, result: c.result })) };
  } else {
    body = {
      path: st.exp.path, results: ed.doc, status: ed.status,
      criteria: ed.criteria.map(c => (c.note.trim() ? { id: c.id, text: c.text, result: c.result, note: c.note.trim() } : { id: c.id, text: c.text, result: c.result })),
    };
  }
  st.saving = true;
  render();
  let r;
  try {
    r = await call('experiment.update', body, { experiment: st.exp.path });
  } catch (_) {
    r = null;
  }
  st.saving = false;
  if (!r || !r.ok || !r.json) {
    st.editError = appError(r);
    render();
    return;
  }
  st.exp = r.json;
  st.edit = null;
  st.editError = null;
  st.docs = {};
  render();
}

function editButtons() {
  return el('div', { class: 'xp-actions' },
    el('button', { class: 'btn primary', type: 'button', disabled: st.saving ? true : null, onclick: saveEdit }, st.saving ? T('저장하는 중…') : T('저장')),
    el('button', { class: 'btn ghost', type: 'button', disabled: st.saving ? true : null, onclick: cancelEdit }, T('취소')),
    st.editError ? el('span', { class: 'xp-err', role: 'alert' }, st.editError) : null);
}

function planEdit() {
  const ed = st.edit;
  const rows = ed.criteria.map((c, i) => el('div', { class: 'xe-crit' },
    el('input', { type: 'text', class: 'xe-id', value: c.id, 'aria-label': T`판정 기준 ${i + 1} 의 id`, oninput: ev => { c.id = ev.target.value; } }),
    el('input', { type: 'text', class: 'xe-text', value: c.text, 'aria-label': T`판정 기준 ${i + 1} 의 내용`, placeholder: T('답이 예가 되려면 무엇이 맞아야 하는가'), oninput: ev => { c.text = ev.target.value; } }),
    el('button', { type: 'button', class: 'btn sm ghost', 'aria-label': T`판정 기준 ${i + 1} 빼기`, onclick: () => { ed.criteria.splice(i, 1); render(); } }, T('빼기'))));
  const addCrit = () => {
    let n = ed.criteria.length + 1;
    while (ed.criteria.some(c => c.id === 'C' + n)) n++;
    ed.criteria.push({ id: 'C' + n, text: '', result: 'pending', note: '' });
    render();
    const last = page.querySelector('.xe-crit:last-of-type .xe-text');
    if (last) last.focus();
  };
  const plan = el('textarea', { id: 'xe-plan', rows: '14', spellcheck: 'false', oninput: ev => { ed.doc = ev.target.value; } });
  plan.value = ed.doc;
  const question = el('textarea', { id: 'xe-q', rows: '2', oninput: ev => { ed.question = ev.target.value; } });
  question.value = ed.question;
  return [
    el('p', { class: 'muted' }, T('계획과 판정 기준은 첫 실행 전에 적습니다. 첫 실행 뒤에는 바꾸지 않습니다 — 판정과 결과는 결과 탭에서 적습니다.')),
    el('div', { class: 'xe-form' },
      el('label', { for: 'xe-title' }, T('제목')),
      el('input', { type: 'text', id: 'xe-title', value: ed.title, oninput: ev => { ed.title = ev.target.value; } }),
      el('label', { for: 'xe-q' }, T('연구 질문')), question,
      el('div', { class: 'xe-label' }, T('판정 기준')),
      el('div', { class: 'xe-crits' }, rows),
      el('div', null, el('button', { type: 'button', class: 'btn sm', onclick: addCrit }, T('판정 기준 더하기'))),
      el('label', { for: 'xe-plan' }, 'plan.md'), plan),
    editButtons(),
  ];
}

function resultsEdit() {
  const ed = st.edit;
  const resLabel = { pending: T('아직'), pass: T('통과'), fail: T('실패') };
  const rows = ed.criteria.map((c, i) => el('div', { class: 'xe-verdict' },
    el('code', null, c.id), el('span', { class: 'xe-ctext' }, c.text),
    el('select', { class: 'input', 'aria-label': T`${c.id} 의 판정`, onchange: ev => { c.result = ev.target.value; } },
      ['pending', 'pass', 'fail'].map(v => el('option', { value: v, selected: c.result === v ? true : null }, resLabel[v]))),
    el('input', { type: 'text', class: 'xe-note', value: c.note, placeholder: T('메모(근거 · 수치)'), 'aria-label': T`${c.id} 의 메모`, oninput: ev => { c.note = ev.target.value; } })));
  const doc = el('textarea', { id: 'xe-results', rows: '12', spellcheck: 'false', oninput: ev => { ed.doc = ev.target.value; } });
  doc.value = ed.doc;
  return [
    el('div', { class: 'xe-form' },
      el('div', { class: 'xe-label' }, T('판정')),
      el('div', { class: 'xe-verdicts' }, rows.length ? rows : el('p', { class: 'muted' }, T('판정 기준이 없습니다'))),
      el('label', { for: 'xe-status' }, T('상태')),
      el('select', { id: 'xe-status', class: 'input', onchange: ev => { ed.status = ev.target.value; } },
        ['planned', 'running', 'done'].map(v => el('option', { value: v, selected: ed.status === v ? true : null }, statusLabel(v)))),
      el('label', { for: 'xe-results' }, 'results.md'), doc),
    editButtons(),
  ];
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
  if (st.edit && st.edit.tab === 'plan') return planEdit();
  const locked = runCount(e) > 0;
  return [
    el('div', { class: 'xp-actions' },
      locked ? el('span', { class: 'muted' }, T('첫 실행 뒤에는 계획과 판정 기준을 바꾸지 않습니다(실행 전에 적고 고정).'))
        : el('button', { class: 'btn', type: 'button', onclick: () => startEdit('plan') }, T('편집'))),
    el('h2', null, T('판정 기준'), ' ', el('span', { class: 'muted' }, T('(실행 전에 적음)'))),
    criteriaList(e),
    el('h2', null, 'plan.md'),
    docBlock('plan.md'),
  ];
}

function resultsTab(e) {
  if (st.edit && st.edit.tab === 'results') return resultsEdit();
  return [
    el('div', { class: 'xp-actions' }, el('button', { class: 'btn', type: 'button', onclick: () => startEdit('results') }, T('편집')),
      // F3-1 — 판정과 결과를 적은 이 탭에서도 점검(기록 탭과 같은 기능 · 같은 결과)
      el('button', { class: 'btn', type: 'button', disabled: st.check && st.check.running ? true : null, onclick: runCheck }, T('점검')),
      el('span', { class: 'muted' }, T('판정 · 메모 · results.md · 상태를 적습니다.'))),
    st.check ? checkBox() : null,
    el('h2', null, T('판정')), criteriaList(e), el('h2', null, 'results.md'), docBlock('results.md'),
  ];
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
  // E5 — 디버거(실험 › 디버거): 실험 밖에서 기록한 서버의 실행 로그
  out.push(el('h2', null, T('디버거의 실행 로그')),
    el('div', { class: 'xp-actions' },
      el('button', { class: 'btn', type: 'button', onclick: () => globalThis.show && globalThis.show('debug') }, T('디버거 열기')),
      el('span', { class: 'muted' }, T('실험 밖에서 기록한 서버의 실행 로그(디버거 v1 — 기록 켜기 · 가져오기 · 비교)는 디버거 화면에서 봅니다.'))));
  return out;
}

function renderExp() {
  const expId = st.exp ? st.exp.experiment.id || baseName(st.cur) : baseName(st.cur);
  const crumbs = el('nav', { class: 'xp-crumbs', 'aria-label': T('경로') },
    el('button', { type: 'button', onclick: backToList }, T('실험')),
    el('span', { class: 'sep', 'aria-hidden': 'true' }, '›'),
    st.run ? el('button', { type: 'button', class: 'id', onclick: closeRun }, expId) : el('span', { class: 'here' }, expId),
    st.run ? el('span', { class: 'sep', 'aria-hidden': 'true' }, '›') : null,
    st.run && st.exp && st.step !== null ? el('button', { type: 'button', class: 'id', onclick: closeStep }, runTitle(ctx(), st.run)) : null,
    st.run && st.exp && st.step !== null ? el('span', { class: 'sep', 'aria-hidden': 'true' }, '›') : null,
    st.run && st.exp ? el('span', { class: 'here' }, st.step !== null ? stepTitle(ctx(), st.run, st.step) : runTitle(ctx(), st.run)) : null);
  if (st.run && st.exp && !st.expError) {
    page.replaceChildren(el('div', { class: 'xp' }, crumbs, st.step !== null ? stepView(ctx(), st.run, st.step) : runView(ctx(), st.run)));
    return;
  }
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
    isChatExp(e) ? el('div', { class: 'xp-actions' },
      el('button', { class: 'btn primary', type: 'button', onclick: () => globalThis.show && globalThis.show('chat') }, T('채팅 화면')),
      el('span', { class: 'muted' }, T('기본 실험 — 채팅의 답마다 실행으로 남습니다. 실행 탭에서 답마다 토큰별 출력 · 분기 · 스텝 · 층 화면을 봅니다.'))) : null,
    el('div', { class: 'xp-tabs', role: 'tablist', 'aria-label': T('실험 탭') },
      TABS.map(([k, label]) => el('button', {
        type: 'button', role: 'tab', class: 'xp-tab' + (st.tab === k ? ' on' : ''), 'aria-selected': String(st.tab === k),
        onclick: () => setTab(k),
      }, label))),
    el('section', { class: 'xp-panel', role: 'tabpanel' }, tabPanel(e)),
  ];
  page.replaceChildren(el('div', { class: 'xp' }, kids));
}

function tabPanel(e) {
  if (st.tab === 'plan') return planTab(e);
  if (st.tab === 'runs') return runsTab(ctx());
  if (st.tab === 'compare') return compareTab(ctx());
  if (st.tab === 'results') return resultsTab(e);
  return recordsTab(e);
}

// ---- 화면 전환 · 이벤트 ----

document.addEventListener('geulpu:view', ev => {
  if (ev.detail && ev.detail.view === 'exp' && (!ev.detail.same || !page.childElementCount)) {
    if (st.cur) loadExp();
    else loadList();
  }
  announce();
});

// F3-1 — 사이드바의 '실험'과 채팅 · 디버거 화면의 경로 '실험'은 늘 실험 목록으로(마지막에 연 실험 · 실행이 아니라)
document.addEventListener('geulpu:exp-list', () => {
  if (st.cur || st.run) backToList();
});

document.addEventListener('geulpu:open-experiment', ev => {
  const path = ev.detail && ev.detail.path;
  if (!path) return;
  if (typeof globalThis.show === 'function') globalThis.show('exp');
  openExp(path);
});

// E3 — 명령 창의 '실행 r3' 링크 · AI 실행 자동 열기(auto: 참). 편집하는 동안의 자동 열기는 건너뛴다(입력을 잃지 않게)
document.addEventListener('geulpu:open-run', ev => {
  const d = ev.detail || {};
  if (!d.experiment || !d.run) return;
  if (d.auto && (st.edit || /^st-/.test((document.activeElement && document.activeElement.id) || ''))) return;   // E4 — 스텝 화면의 입력 중에도
  if (typeof globalThis.show === 'function') globalThis.show('exp');
  if (st.cur && st.exp && samePath(st.exp.path, d.experiment)) {
    st.run = d.run;
    st.step = null;
    st.edit = null;
    render();
    loadExp();
  } else {
    openExp(d.experiment);
    st.run = d.run;
  }
  view.scrollTop = 0;
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
