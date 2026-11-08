// 앱리디자인 E1(문서/화면구현.md §1 · 화면설계.md §5) — 명령 창. 모든 화면의 오른쪽에서 기능 호출을 한 줄씩 보이고(어느 출처든 — GUI · 사용자
// CLI · AI 세션 · 파이썬), 입력 줄에 친 GeulPU 명령을 CLI 와 같은 요청으로 보낸다. 이벤트는 롱 폴링(GET /api/events)으로 받고, 보이는 명령은
// 이벤트의 요청 본문(params)을 CLI 형식으로 다시 쓴 것이다. 앱은 기능 목록 클라이언트(ops.js)로만 부른다.
import { callable, commandName, parse, toCommand } from './command.js';
import { CHEVRON, EN, T, TM, baseName, clock, copyButton, el, fmtBytes, fmtMs, icon, sourceLabel as labelOf } from './dom.js';
import { call, registry, send } from './ops.js';

const KEY = 'geulpu.console';
const MIN_W = 280;
const MAX_W = 640;
const MAX_LINES = 400;
const SHORT = 72;              // 줄에 보이는 값의 길이(전체는 줄을 눌러서)
const OUT_MAX = 20000;         // 응답을 펼쳐 보일 때의 글자 수

const app = document.getElementById('app');
const box = document.getElementById('console');
const handle = document.getElementById('c-handle');

// ---- 접기 · 폭(이 브라우저에 기억) ----
function loadState() {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) || '{}');
    return { w: Math.min(MAX_W, Math.max(MIN_W, Number(v.w) || 380)), folded: !!v.folded };
  } catch (_) {
    return { w: 380, folded: false };
  }
}
const ui = loadState();
function saveState() {
  try { localStorage.setItem(KEY, JSON.stringify(ui)); } catch (_) { /* 저장하지 못해도 화면은 그대로 */ }
}
function applyLayout() {
  app.style.setProperty('--console-w', ui.w + 'px');
  app.classList.toggle('c-folded', ui.folded);
  foldBtn.setAttribute('aria-label', ui.folded ? T('명령 창 펼치기') : T('명령 창 접기'));
  foldBtn.setAttribute('aria-expanded', String(!ui.folded));
  handle.setAttribute('aria-valuenow', String(ui.w));
  if (!ui.folded) { unseen = 0; countEl.textContent = ''; }
}
function setFolded(f) {
  ui.folded = f;
  applyLayout();
  saveState();
  if (!f) stick(true);
}

// ---- 명령 창의 요소 ----
const countEl = el('span', { class: 'c-count', 'aria-live': 'polite' });
const foldBtn = el('button', { class: 'icon-btn c-fold', type: 'button', title: T('접기 · 펼치기 ( ] )'), onclick: () => setFolded(!ui.folded) }, icon(CHEVRON));
const list = el('ol', { class: 'c-lines', role: 'log', 'aria-label': T('기능 호출') });
const input = el('input', {
  id: 'c-in', type: 'text', autocomplete: 'off', spellcheck: 'false', 'aria-label': T('GeulPU 명령'),
  placeholder: T('GeulPU 명령 (예: run complete --prompt …)'),
});
const form = el('form', { class: 'c-input' }, el('span', { 'aria-hidden': 'true' }, '›'), input);
box.replaceChildren(
  el('header', { class: 'c-head' }, el('span', { class: 'c-title' }, T('명령 창')), countEl, foldBtn),
  list,
  form,
);
let unseen = 0;
applyLayout();

// 줄 목록의 맨 아래를 보고 있으면 새 줄을 따라간다
let atBottom = true;
list.addEventListener('scroll', () => { atBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 40; });
function stick(force) {
  if (force || atBottom) list.scrollTop = list.scrollHeight;
}

// ---- 크기 조절 핸들 ----
function setWidth(w) {
  ui.w = Math.min(MAX_W, Math.max(MIN_W, Math.round(w)));
  applyLayout();
}
handle.addEventListener('pointerdown', e => {
  if (ui.folded || e.button !== 0) return;
  e.preventDefault();
  handle.setPointerCapture(e.pointerId);
  const right = app.getBoundingClientRect().right;
  const move = ev => setWidth(right - ev.clientX - handle.offsetWidth / 2);
  const up = () => {
    handle.removeEventListener('pointermove', move);
    handle.removeEventListener('pointerup', up);
    handle.removeEventListener('pointercancel', up);
    app.classList.remove('c-resizing');
    saveState();
  };
  app.classList.add('c-resizing');
  handle.addEventListener('pointermove', move);
  handle.addEventListener('pointerup', up);
  handle.addEventListener('pointercancel', up);
});
handle.addEventListener('keydown', e => {
  const step = { ArrowLeft: 16, ArrowRight: -16, Home: MAX_W, End: -MAX_W }[e.key];
  if (step === undefined || ui.folded) return;
  e.preventDefault();
  setWidth(ui.w + step);
  saveState();
});

// ---- 줄 ----
const lines = new Map();       // 호출 번호 → 줄 기록
const pending = [];            // 입력 줄로 보낸 요청 — 그 호출의 이벤트가 오면 같은 줄에 잇는다

// 출처 — 입력 줄에 친 명령은 사용자가 친 CLI 명령이다(화면설계.md §5 · 시안)
const sourceLabel = rec => (rec.typed ? labelOf('cli', '') : labelOf(rec.source, rec.session));

function commandText(rec, short) {
  if (rec.typed) return rec.text;
  if (!reg) return rec.op;
  const o = reg.byId.get(rec.op);
  if (!o) return rec.op;
  if (rec.params_bytes !== undefined) return 'GeulPU ' + commandName(o) + '  ' + T`(본문 ${fmtBytes(rec.params_bytes)} — 명령 창에는 담지 않음)`;
  return toCommand(reg, rec.op, rec.params === undefined ? null : rec.params, short ? SHORT : 0);
}

function stateText(rec) {
  if (rec.state === 'run') return T('실행 중');
  if (rec.state === 'sent') return T('보내는 중');
  if (rec.closed) return T`${rec.status} · ${fmtMs(rec.ms)} · 응답 전에 화면 쪽이 끊음`;
  if (rec.status !== undefined && rec.ms !== undefined) return `${rec.status} · ${fmtMs(rec.ms)}`;
  if (rec.status !== undefined) return String(rec.status);
  return '';
}

function render(rec) {
  const li = rec.li || (rec.li = el('li', { class: 'c-line' }));
  li.className = 'c-line ' + (rec.state || '') + (rec.typed ? ' typed' : '');
  if (rec.state === 'note' && !rec.typed) {
    li.replaceChildren(el('div', { class: 'msg' }, rec.message));
    return li;
  }
  const [label, kind] = sourceLabel(rec);
  const cmd = el('div', { class: 'cmd', title: rec.open || rec.local ? null : T('누르면 명령 전체') }, commandText(rec, !rec.open));
  if (!rec.local) {
    cmd.addEventListener('click', () => { rec.open = !rec.open; render(rec); });
  }
  const res = [];
  const st = stateText(rec);
  if (st) res.push(el('span', { class: 'state' }, st));
  // E2 — 실험이 있는 줄은 그 실험 화면으로(ui/experiments.js 가 연다)
  if (rec.experiment) {
    const path = rec.experiment;
    res.push(el('button', {
      type: 'button', class: 'linkish exp', title: path,
      onclick: () => document.dispatchEvent(new CustomEvent('geulpu:open-experiment', { detail: { path } })),
    }, T`실험 ${baseName(path)}`));
  }
  if (rec.op && reg && reg.byId.has(rec.op) && !rec.local) res.push(copyButton(() => (rec.typed ? rec.text : commandText(rec, false))));
  if (rec.result !== undefined) {
    res.push(el('button', { type: 'button', class: 'linkish', 'aria-expanded': String(!!rec.showOut), onclick: () => { rec.showOut = !rec.showOut; render(rec); } },
      rec.showOut ? T('응답 접기') : T('응답 보기')));
  }
  const kids = [
    el('div', { class: 'who' }, el('span', { class: 'src ' + kind }, label), rec.op ? el('span', { class: 'op' }, rec.op) : null,
      el('span', { class: 'grow' }), rec.time ? el('span', { class: 'at' }, clock(rec.time)) : null),
    cmd,
  ];
  if (rec.message) kids.push(el('div', { class: 'msg' + (rec.state === 'fail' ? ' bad' : '') }, rec.message));
  if (rec.next) kids.push(el('div', { class: 'next' }, rec.next));
  if (rec.summary) kids.push(el('div', { class: 'msg' }, rec.summary));
  if (rec.body) kids.push(rec.body);
  if (res.length) kids.push(el('div', { class: 'res' }, res));
  if (rec.result !== undefined && rec.showOut) {
    const text = typeof rec.result === 'string' ? rec.result : JSON.stringify(rec.result, null, 2);
    kids.push(el('pre', { class: 'out' }, text.length > OUT_MAX ? text.slice(0, OUT_MAX) + '\n…' + T`(${text.length - OUT_MAX}자 더 있음)` : text));
  }
  li.replaceChildren(...kids);
  return li;
}

function add(rec) {
  list.append(render(rec));
  while (list.children.length > MAX_LINES) {
    const first = list.firstElementChild;
    for (const [k, r] of lines) if (r.li === first) lines.delete(k);
    first.remove();
  }
  if (ui.folded) {
    unseen++;
    countEl.textContent = String(unseen);
  }
  stick(false);
}

function note(text) {
  add({ local: true, state: 'note', text: '', message: text });
}

// ---- 이벤트(롱 폴링) ----
let reg = null;
let after = 0;

function onEvent(e) {
  // E2 — 다른 화면 모듈도 이벤트를 받는다(실험 화면의 기록 탭 등)
  document.dispatchEvent(new CustomEvent('geulpu:event', { detail: e }));
  if (e.type === 'start') {
    let rec = null;
    if (e.source === 'cli' && !e.session) {
      const i = pending.findIndex(p => p.op === e.op && Date.now() - p.t < 30000);
      if (i >= 0) rec = pending.splice(i, 1)[0].rec;
    }
    if (rec) {
      rec.call = e.call;
      if (rec.state === 'sent') rec.state = 'run';
      if (e.experiment) rec.experiment = e.experiment;
      lines.set(e.call, rec);
      render(rec);
      return;
    }
    rec = { call: e.call, op: e.op, source: e.source, session: e.session, time: e.time, state: 'run' };
    if (e.params !== undefined) rec.params = e.params;
    if (e.params_bytes !== undefined) rec.params_bytes = e.params_bytes;
    if (e.experiment) rec.experiment = e.experiment;
    lines.set(e.call, rec);
    add(rec);
  } else if (e.type === 'end') {
    let rec = lines.get(e.call);
    if (!rec) {
      rec = { call: e.call, op: e.op, source: e.source, session: e.session, time: e.time };
      lines.set(e.call, rec);
      add(rec);
    }
    rec.status = e.status;
    rec.ms = e.ms;
    rec.closed = !!e.client_closed;
    if (rec.state === 'run' || !rec.typed || rec.state === 'sent') rec.state = e.status >= 200 && e.status < 400 && !e.client_closed ? 'done' : 'fail';
    render(rec);
    stick(false);
  }
}

const sleep = ms => new Promise(r => setTimeout(r, ms));
async function poll() {
  let first = true;
  for (;;) {
    try {
      if (!reg) reg = await registry();
      const r = await call('system.events', { after, wait: first ? 0 : 25000 });
      if (!r.ok || !r.json || !Array.isArray(r.json.events)) {
        await sleep(2000);
        continue;
      }
      const { events, last, oldest } = r.json;
      if (last < after) {
        note(T('앱이 다시 켜졌습니다 — 이벤트를 처음부터 받습니다'));
        after = 0;
        continue;
      }
      if (after > 0 && oldest > after + 1) note(T`이벤트 ${oldest - after - 1}개를 놓쳤습니다(앱은 최근 1,024개만 둡니다)`);
      for (const e of events) onEvent(e);
      after = Math.max(after, last);
      first = false;
    } catch (_) {
      await sleep(2000);
    }
  }
}

// ---- 입력 줄 ----
const history = [];
let hpos = 0;

function helpBody(o) {
  if (!o) {
    return el('div', { class: 'help' },
      el('p', null, T('명령은 영어 이름(run branch) · 한국어 이름(실행 분기) · 기능 id(run.branch) 가운데 하나입니다.')),
      el('p', null, 'GeulPU <command> [--<param> <value> …] [--json <body>] [--experiment <folder|id>] [--wait] [--dry-run]'),
      el('p', null, T('ops 는 모든 명령을, help <명령> 은 명령 하나의 인자를 보입니다.')));
  }
  const tr = (o.params || []).map(p => el('tr', null,
    el('td', null, el('code', null, '--' + p.name)), el('td', null, p.type || 'string'), el('td', null, p.required ? T('필수') : ''),
    el('td', null, (p.description && (EN() ? p.description.en : p.description.ko)) || '')));
  return el('div', { class: 'help' },
    el('p', null, (o.title && (EN() ? o.title.en : o.title.ko)) || o.id),
    el('p', { class: 'muted' }, `${o.http ? o.http.method + ' ' + o.http.path : ''} · ${commandName(o)}${o.cli && o.cli.ko ? ' · ' + o.cli.ko : ''}`),
    tr.length ? el('table', null, el('tbody', null, tr)) : el('p', { class: 'muted' }, T('인자 없음')));
}

function opsBody() {
  const rows = reg.operations.filter(callable).map(o => el('li', null, el('code', null, commandName(o)), ' ',
    el('span', { class: 'muted' }, (o.cli && o.cli.ko ? o.cli.ko + ' · ' : '') + ((o.title && (EN() ? o.title.en : o.title.ko)) || ''))));
  return el('ul', { class: 'help ops' }, rows);
}

async function waitJob(rec, job) {
  rec.summary = T`작업 ${job}: 끝나기를 기다리는 중`;
  render(rec);
  for (;;) {
    await sleep(500);
    let r;
    try { r = await call('system.state'); } catch (_) { continue; }
    const j = r.json && r.json.job;
    if (!j) continue;
    if (j.id === job && j.state !== 'running') {
      rec.summary = T`작업 ${job}: ${j.state}${j.summary ? ' — ' + TM(j.summary) : ''}`;
      rec.result = { job: j };
      if (j.state !== 'done') rec.state = 'fail';
      render(rec);
      return;
    }
    if (j.id > job) {
      rec.summary = T`작업 ${job}: 결과를 읽기 전에 다음 작업(${j.id})이 시작되었습니다`;
      rec.state = 'fail';
      render(rec);
      return;
    }
  }
}

async function runLine(text) {
  if (!reg) {
    try { reg = await registry(); } catch (_) {
      note(T('기능 목록을 읽지 못했습니다 — 앱이 켜져 있는지 확인하세요'));
      return;
    }
  }
  const r = parse(reg, text);
  if (r.kind === 'empty') return;
  const rec = { typed: true, local: false, text, op: r.op ? (typeof r.op === 'string' ? r.op : r.op.id) : null, time: Date.now() / 1000 };
  if (r.kind === 'error') {
    rec.state = 'fail';
    rec.local = true;
    rec.message = EN() ? r.message : r.message_ko;
    rec.next = EN() ? r.next : r.next_ko;
    add(rec);
    return;
  }
  if (r.kind === 'help' || r.kind === 'ops' || r.kind === 'guide') {
    rec.local = true;
    rec.state = 'note';
    rec.body = r.kind === 'help' ? helpBody(r.op) : r.kind === 'ops' ? opsBody() : el('p', { class: 'help' }, T('에이전트용 안내 한 쪽은 터미널에서 GeulPU guide 로 봅니다(docs/cli.md).'));
    add(rec);
    return;
  }
  const o = r.options;
  const unused = ['port', 'session', 'timeout', 'out'].find(k => o[k] !== null);
  if (unused) {
    rec.state = 'fail';
    rec.local = true;
    rec.message = T`명령 창에서는 쓰지 않는 옵션입니다: --${unused}`;
    add(rec);
    return;
  }
  if (o.dryRun) {
    rec.local = true;
    rec.state = 'note';
    rec.result = { op: r.op, method: r.method, path: r.path, body: r.body === null ? null : JSON.parse(r.body) };
    rec.showOut = true;
    add(rec);
    return;
  }
  let experiment = '';
  if (o.experiment !== null) {
    const ex = await call('experiment.show', { path: o.experiment }).catch(() => null);
    if (!ex || !ex.ok || !ex.json || !ex.json.path) {
      rec.state = 'fail';
      rec.local = true;
      rec.message = T`실험 폴더가 아닙니다: ${o.experiment}`;
      add(rec);
      return;
    }
    experiment = ex.json.path;
    rec.experiment = experiment;
  }
  rec.state = 'sent';
  add(rec);
  // 이 호출의 이벤트(출처 cli · 세션 없음 · 같은 기능)가 오면 이 줄에 잇는다 — 이벤트를 남기는 기능(POST)만
  if (r.method === 'POST') pending.push({ op: r.op, rec, t: Date.now() });
  let ans;
  try {
    ans = await send(r, { source: 'cli', experiment });
  } catch (e) {
    const i = pending.findIndex(p => p.rec === rec);
    if (i >= 0) pending.splice(i, 1);
    rec.state = 'fail';
    rec.message = T('앱에 닿지 못했습니다');
    render(rec);
    return;
  }
  rec.result = ans.json !== null ? ans.json : ans.text;
  // 상태는 답으로 — POST 는 이벤트의 end 가 걸린 시간을 더 적는다(이벤트가 답보다 늦게 와도 같은 줄)
  if (rec.status === undefined) rec.status = ans.status;
  if (rec.state === 'sent' || rec.state === 'run' || r.method !== 'POST') rec.state = ans.ok ? 'done' : 'fail';
  if (!ans.ok && ans.json && ans.json.error) rec.message = TM(ans.json.error.message || '');
  if (ans.ok && ans.json && typeof ans.json.job === 'number') {
    if (o.wait) {
      render(rec);
      waitJob(rec, ans.json.job);
      return;
    }
    rec.summary = T`작업 ${ans.json.job}: 시작함(끝날 때까지 기다리려면 명령에 --wait)`;
  }
  render(rec);
  const live = document.getElementById('sr-live');
  if (live) live.textContent = ans.ok ? T`명령을 마쳤습니다: ${commandName(reg.byId.get(r.op))}` : T`명령이 실패했습니다: ${ans.status}`;
}

form.addEventListener('submit', e => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  if (history[history.length - 1] !== text) history.push(text);
  hpos = history.length;
  input.value = '';
  stick(true);
  runLine(text);
});
input.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    e.stopPropagation();
    input.blur();
  } else if (e.key === 'ArrowUp' && hpos > 0) {
    e.preventDefault();
    input.value = history[--hpos];
  } else if (e.key === 'ArrowDown' && hpos < history.length) {
    e.preventDefault();
    hpos++;
    input.value = hpos < history.length ? history[hpos] : '';
  }
});

// 단축키 — 다른 입력 칸 · 창이 열려 있지 않을 때: / 입력 줄로, ] 접기 · 펼치기
document.addEventListener('keydown', e => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
  const t = e.target;
  if (t && (t.closest('input, textarea, select, [contenteditable=""], [contenteditable="true"]'))) return;
  const modal = document.getElementById('modal');
  if (modal && !modal.hidden) return;
  if (e.key === '/') {
    e.preventDefault();
    if (ui.folded) setFolded(false);
    input.focus();
  } else if (e.key === ']') {
    e.preventDefault();
    setFolded(!ui.folded);
  }
});

poll();
