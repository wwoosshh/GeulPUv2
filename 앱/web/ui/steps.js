// 앱리디자인 E4(문서/화면구현.md §4 · 화면설계.md §3 의 수준 5 · §9) — 스텝 · 층 화면: 실행 하나의 스텝 하나. 상위 후보(echo 스코어링으로 N 개),
// 로짓 렌즈 표(서버가 J 파일을 열었으면 Jacobian lens 도), 주어진 텍스트의 점수. 계산은 모두 X-GeulPU-Experiment 를 붙여 부르므로 앱이 그 실험의
// 실행으로 남기고(실험.gl), 화면은 그 실행의 기록 파일에서 읽는다 — 다시 열면 이미 기록된 실행을 보이고 다시 계산하지 않는다. 점수 · 로짓 렌즈
// 실행의 실행 화면(그 표)과 실행 목록의 설명도 여기서 만든다.
import { EN, T, el } from './dom.js';
import { call } from './ops.js';
import { deltaNow, matchLine } from './research.js';
import {
  appError, busyLine, candText, dataById, dataOf, doBranch, entryOf, fetchPieces, fx, isBusy, isIds, measure, pct, pieceText, quoteTok,
  recordOf, runInfo, runsOf, selectStep, stepVals, summaryOfRun,
} from './runs.js';

export const MEASURE_OPS = new Set(['run.score', 'run.lens']);
const LENS_K = [3, 5, 10, 20];          // 로짓 렌즈의 층마다 상위 후보 수(서버는 0 ~ 20)
const TOP_MAX = 128;                    // geulpu.top_logprobs 의 한도
const SCORE_TOP = 20;                   // 점수 — 위치마다 상위 후보 수(순위를 알 수 있는 범위)
const COLS = 3;                         // 로짓 렌즈 표에 열로 보이는 상위 후보 수(나머지는 줄을 펼쳐서)
const CTX_BEFORE = 24;                  // 헤더의 텍스트 — 스텝 앞의 토큰 수
const CTX_AFTER = 8;                    //                 스텝 뒤의 토큰 수

const st = {
  topN: 20,
  lensK: 5,
  view: 'logit',                        // 로짓 렌즈 표 — 'logit' · 'jacobian'
  open: new Set(),                      // 펼친 층 '<실험 경로>|<실행 id>|<렌즈>|<층>'
  text: new Map(),                      // '<실험 경로>|<실행 id>|<스텝>' → 점수 입력
  noCtx: new Map(),                     // 같은 키 → 문맥 없이 점수(E6-2 — 연구 패널의 '맥락 없이')
  inMode: new Map(),                    // 같은 키 → 입력의 종류 'text' · 'ids'(앱리디자인 E6 — 토큰 id 는 run.detokenize 로 글을 보인다)
  idText: new Map(),                    // 같은 키 → 토큰 id 입력의 글
  scoreError: null,                     // {key, text}
  server: null,                         // system.state 의 답(불러온 모델 · J 파일)
  serverTry: 0,
  serverLoading: false,
  ctx: null,                            // 마지막으로 그린 화면의 ctx(모델이 바뀌면 다시 그린다)
};

// ---- 불러온 모델(system.state) ----

function serverState(ctx) {
  st.ctx = ctx;
  if (!st.serverLoading && Date.now() - st.serverTry > 10000) {
    st.serverLoading = true;
    st.serverTry = Date.now();
    call('system.state', {})
      .then(r => { if (r.ok && r.json) st.server = r.json; }, () => { /* 앱에 닿지 못하면 지난 값 */ })
      .then(() => {
        st.serverLoading = false;
        const sv = st.server && st.server.server;
        if (sv && sv.state === 'starting') setTimeout(refreshServer, 2000);      // 모델을 불러오는 중이면 곧 다시 읽는다
        if (st.ctx && document.querySelector('.st-view')) st.ctx.rerender();
      });
  }
  return st.server;
}
function refreshServer() {
  st.serverTry = 0;
  if (st.ctx && document.querySelector('.st-view')) st.ctx.rerender();
}
// 모델을 불러오거나 내리거나 설정을 바꾸면(작업이 끝나면 상태가 바뀐다) 다시 읽는다
document.addEventListener('geulpu:event', ev => {
  const e = ev.detail;
  if (e && e.type === 'end' && typeof e.op === 'string' && /^(model|settings)\./.test(e.op)) setTimeout(refreshServer, 1000);
});

/** 불러온 모델 — {ready, id, name, sha, lensOk, jsha, jfile, jreason} 또는 null(상태를 아직 모르거나 모델이 없음) */
function loaded(stt) {
  const sv = stt && stt.server;
  if (!sv || !sv.model) return null;
  const m = (Array.isArray(stt.models) ? stt.models : []).find(x => x && x.id === sv.model) || null;
  return {
    ready: sv.state === 'ready', id: sv.model, name: (m && (m.name || m.id)) || sv.model, sha: (m && m.sha256) || '',
    lensOk: !!(m && m.shape && m.shape.layer_lens_ok), jsha: sv.jlens_sha256 || '', jfile: sv.jlens_file || '', jreason: sv.jlens_reason || '',
  };
}

// ---- 점수 · 렌즈 실행 ----

const validLens = L => !!(L && typeof L === 'object' && Number.isInteger(L.layers) && L.layers > 0 && Array.isArray(L.top_ids) &&
  L.top_ids.length === L.layers && L.top_ids.every(Array.isArray) && Array.isArray(L.top_logprobs) && L.top_logprobs.length === L.layers &&
  Array.isArray(L.ranks) && Array.isArray(L.token_logprobs) && Array.isArray(L.entropy) && Array.isArray(L.logit_hashes));

/** 점수 · 렌즈 실행의 기록 → {entry, rec, q(프롬프트 토큰), at(echo_from), tok(위치 at 의 토큰), g(확장 필드), res, isLens, lens, jac, d} — 토큰
   번호의 프롬프트와 echo_from 이 없으면 null */
function measureOf(entry, c) {
  const req = c.rec.request && typeof c.rec.request === 'object' ? c.rec.request : {};
  const gq = req.geulpu && typeof req.geulpu === 'object' ? req.geulpu : {};
  const q = req.prompt;
  if (!isIds(q) || !Number.isInteger(gq.echo_from) || gq.echo_from < 1 || gq.echo_from >= q.length) return null;
  const res = c.rec.response && typeof c.rec.response === 'object' ? c.rec.response : null;
  const g = (res && res.geulpu) || {};
  return {
    entry, rec: c.rec, q, at: gq.echo_from, tok: q[gq.echo_from], g, res, isLens: entry.operation === 'run.lens',
    lens: validLens(g.layer_lens) ? g.layer_lens : null, jac: validLens(g.jacobian_lens) ? g.jacobian_lens : null,
    d: dataOf(entry, c.rec, c.text),
  };
}

/** q 의 앞 P + k 토큰이 실행 d 의 프롬프트 토큰 + 답의 스텝 0 ~ k − 1 인가 */
function samePrefix(q, d, k) {
  const P = d.ptoks.length;
  if (q.length < P + k) return false;
  for (let i = 0; i < P; i++) if (q[i] !== d.ptoks[i]) return false;
  for (let i = 0; i < k; i++) if (q[P + i] !== d.ids[i]) return false;
  return true;
}

/** 두 기록을 같은 모델로 계산했는가(SHA-256 — 모르면 같다고 본다) */
function sameModel(a, b) {
  const x = (a.server || {}).sha256;
  const y = (b.server || {}).sha256;
  return !x || !y || x === y;
}

/** 점수 · 렌즈 실행의 원래 실행과 그 스텝 — {d(원래 실행의 값), k(그 실행 안의 위치)}. 원래 실행이 없거나 아직 읽는 중이면 null */
function baseOf(ctx, entry, m) {
  if (!entry.base) return null;
  const bd = dataById(ctx, entry.base);
  if (!bd || !bd.ptoks || !bd.n) return null;
  const k = m.at - bd.ptoks.length;
  if (k < 0 || k >= bd.n || !samePrefix(m.q, bd, k)) return null;
  return { d: bd, k };
}

/** 실행 d 의 스텝 k 를 계산한 점수 · 렌즈 실행(같은 모델이 앞, 그 안에서 최근 것이 앞) — 프롬프트의 앞 P + k 토큰이 같고 echo_from 이 P + k.
   같은 앞부분의 행은 그 뒤 토큰과 상관없이 같은 분포를 내므로 어느 점수 실행이든 이 스텝의 상위 후보를 준다 */
function measuresAt(ctx, d, k) {
  const at = d.ptoks.length + k;
  const out = { lens: [], other: [], scores: [], cover: [], loading: false };
  const runs = runsOf(ctx);
  for (let i = runs.length - 1; i >= 0; i--) {
    const e = runs[i];
    if (!MEASURE_OPS.has(e.operation) || e.id === d.entry.id) continue;
    const c = recordOf(ctx, e);
    if (c.loading) {
      out.loading = true;
      continue;
    }
    if (!c.rec) continue;
    const m = measureOf(e, c);
    if (!m) continue;
    if (m.at !== at) {
      // E6-2 — 이 스텝의 행(at)을 덮는 점수 실행(예: '모든 스텝의 상위 후보') — 상위 후보로만 쓴다
      if (!m.isLens && m.at < at && at < m.q.length && samePrefix(m.q, d, k)) {
        m.sameModel = sameModel(m.rec, d.rec);
        out.cover.push(m);
      }
      continue;
    }
    if (!samePrefix(m.q, d, k)) continue;
    m.sameModel = sameModel(m.rec, d.rec);
    if (m.isLens) (m.tok === d.ids[k] && m.lens ? out.lens : out.other).push(m);
    else out.scores.push(m);
  }
  const order = (x, y) => (x.sameModel === y.sameModel ? 0 : x.sameModel ? -1 : 1);
  out.lens.sort(order);
  out.scores.sort(order);
  return out;
}

/** 점수 실행의 첫 위치(스텝)의 상위 후보 — {ids, lps} 또는 null. echo 의 prompt_top_ids · prompt_top_logprobs 는 프롬프트의 위치마다(echo_from
   앞은 null), prompt_logit_hashes 는 echo_from 부터 */
function firstTop(m, pos = m.at) {
  const ids = Array.isArray(m.g.prompt_top_ids) ? m.g.prompt_top_ids[pos] : null;
  const lps = Array.isArray(m.g.prompt_top_logprobs) ? m.g.prompt_top_logprobs[pos] : null;
  if (!isIds(ids) || !ids.length || !Array.isArray(lps) || lps.length !== ids.length) return null;
  return { ids, lps };
}

/** 이 스텝의 상위 후보 — 점수 실행 가운데 상위 후보가 가장 많은 것(같은 모델 먼저, 수가 같으면 상위 후보 읽기 · 최근 것) */
function bestTop(ms, d, k) {
  let best = null;
  const read = m => m.q.length === m.at + 1 && m.tok === d.ids[k];
  const at = d.ptoks.length + k;
  for (const m of ms.scores.concat(ms.cover || [])) {
    const t = firstTop(m, at);
    if (!t) continue;
    const better = !best || (m.sameModel && !best.m.sameModel) || (m.sameModel === best.m.sameModel &&
      (t.ids.length > best.ids.length || (t.ids.length === best.ids.length && read(m) && !read(best.m))));
    if (better) best = { m, pos: at, ...t };
  }
  return best;
}

/** 점수 실행의 위치 pos 에서 로그확률 lp 인 후보의 글(OpenAI 형식의 top_logprobs — 같은 글의 토큰은 합쳐져 있다). 없으면 null */
function dictText(m, pos, lp) {
  const lp0 = m.res && Array.isArray(m.res.choices) && m.res.choices[0] && m.res.choices[0].logprobs;
  const o = lp0 && Array.isArray(lp0.top_logprobs) ? lp0.top_logprobs[pos] : null;
  if (!o || typeof o !== 'object') return null;
  for (const [t, v] of Object.entries(o)) if (v === lp) return t;
  return null;
}

/** 점수 · 렌즈 실행의 위치 pos 토큰의 글(한 글자의 바이트가 나뉜 토큰은 첫 토큰에 모은 글 — 나머지는 '') */
function posText(m, pos) {
  const t = m.d.ptexts ? m.d.ptexts[pos] : undefined;
  if (typeof t === 'string') return t;
  if (t === null) return '';
  return pieceText(m.q[pos]);
}

/** 점수 실행이 매긴 텍스트(위치 at 부터) */
const scoredText = m => m.q.slice(m.at).map((_, j) => posText(m, m.at + j) || '').join('');

/** 점수 실행의 로그확률 합(위치 at 부터) — 하나라도 없으면 null */
function scoreSum(m) {
  const lp = m.res && Array.isArray(m.res.choices) && m.res.choices[0] && m.res.choices[0].logprobs;
  const tl = lp && Array.isArray(lp.token_logprobs) ? lp.token_logprobs : null;
  if (!tl) return null;
  let s = 0;
  for (let pos = m.at; pos < m.q.length; pos++) {
    if (typeof tl[pos] !== 'number') return null;
    s += tl[pos];
  }
  return s;
}

/** 상위 후보 읽기(점수 실행 가운데 프롬프트가 원래 실행의 앞부분 + 샘플링된 토큰인 것)인가 */
const isTopRead = (m, b) => !m.isLens && b && m.q.length === m.at + 1 && m.tok === b.d.ids[b.k];

/** 실행 목록 · 실행 화면의 설명 — 'r1 의 스텝 3 · 28 층' · 'r1 의 스텝 3 · 상위 후보 20 개' · 'r1 의 스텝 3 에 " The" · 로그확률 −2.310'.
   기록을 아직 읽지 못했거나 표로 보일 수 없는 실행이면 null(E3 의 설명) */
export function measureLabel(ctx, entry) {
  const c = recordOf(ctx, entry);
  if (!c.rec) return null;
  const m = measureOf(entry, c);
  if (!m) return null;
  const b = baseOf(ctx, entry, m);
  const parts = [];
  if (b) {
    const stepNo = b.d.off + b.k;
    const own = m.tok === b.d.ids[b.k] && (m.isLens || m.q.length === m.at + 1);
    const what = m.isLens ? posText(m, m.at) : scoredText(m);
    parts.push(own ? T`${b.d.entry.id} 의 스텝 ${stepNo}` : T`${b.d.entry.id} 의 스텝 ${stepNo} 에 ${quoteTok(what === null ? '#' + m.tok : what)}`);
  } else {
    parts.push(T`토큰 ${m.q.length} 개`);
  }
  if (m.isLens) {
    if (m.lens) parts.push(T`${m.lens.layers} 층`);
    if (m.jac) parts.push('Jacobian lens');
  } else if (isTopRead(m, b)) {
    const t = firstTop(m);
    if (t) parts.push(T`상위 후보 ${t.ids.length} 개`);
  } else {
    const s = scoreSum(m);
    if (s !== null) parts.push(T`로그확률 ${fx(s, 3)}`);
  }
  return parts.join(' · ');
}

// ---- 표 ----

const fmtInt = n => (Number.isInteger(n) ? n.toLocaleString(EN() ? 'en-US' : 'ko-KR') : '–');

/** 이어진 정수들을 묶어 쓴다 — [21, 22, 23, 24, 26] → '21 ~ 24 · 26' */
function ranges(a) {
  const out = [];
  for (let i = 0; i < a.length;) {
    let j = i;
    while (j + 1 < a.length && a[j + 1] === a[j] + 1) j++;
    out.push(j > i ? `${a[i]} ~ ${a[j]}` : String(a[i]));
    i = j + 1;
  }
  return out.join(' · ');
}

/** 후보 목록 — rows: [{id, lp, text}]. sampledId 는 표시, onBranch(id) 가 있으면 샘플링되지 않은 후보마다 분기 단추 */
function candList(rows, sampledId, onBranch) {
  return el('div', { class: 'st-cands', role: 'list' }, rows.map((r, j) => {
    const sampled = r.id === sampledId;
    const name = r.text === null ? '#' + r.id : quoteTok(r.text);
    return el('div', { class: 'st-cand' + (sampled ? ' sampled' : ''), role: 'listitem' },
      el('span', { class: 'r' }, String(j + 1)),
      el('span', { class: 'w' }, name, sampled ? el('span', { class: 'tag dim' }, T('샘플링됨')) : null),
      el('span', { class: 'lp' }, fx(r.lp, 3)),
      el('span', { class: 'pb', 'aria-hidden': 'true' }, el('span', { style: `width:${Math.max(1, Math.exp(r.lp) * 100).toFixed(1)}%` })),
      el('span', { class: 'p' }, pct(Math.exp(r.lp))),
      onBranch && !sampled ? el('button', {
        type: 'button', class: 'btn sm', disabled: isBusy() ? true : null, 'aria-label': T`이 토큰으로 분기: ${name}`,
        onclick: () => onBranch(r.id),
      }, T('분기', '단추')) : el('span', null));
  }));
}

/** 로짓 렌즈 표 — m: 렌즈 실행, who: 순위를 보이는 토큰의 이름(샘플링된 토큰 · 토큰), base: {id, hash, stepNo} 또는 null(원래 실행의 그 스텝) */
function lensTable(ctx, m, who, base) {
  const out = [];
  const view = m.jac && st.view === 'jacobian' ? 'jacobian' : 'logit';
  const L = view === 'jacobian' ? m.jac : m.lens;
  if (!L) return [el('p', { class: 'muted' }, T('이 실행의 응답에 로짓 렌즈가 없습니다.'))];
  if (m.jac) {
    out.push(el('div', { class: 'seg-ctl st-seg', role: 'radiogroup', 'aria-label': T('렌즈') },
      [['logit', T('로짓 렌즈')], ['jacobian', 'Jacobian lens']].map(([v, label]) => el('button', {
        type: 'button', role: 'radio', class: view === v ? 'on' : null, 'aria-checked': String(view === v),
        onclick: () => { st.view = v; ctx.rerender(); },
      }, label))));
  }
  const n = L.layers;
  const ids = [m.tok];
  for (let l = 0; l < n; l++) ids.push(...L.top_ids[l]);
  const missing = [...new Set(ids)].filter(id => pieceText(id) === null);
  if (missing.length) fetchPieces(missing, ctx);
  const tokText = posText(m, m.at);
  const show = id => {
    const t = id === m.tok && tokText ? tokText : pieceText(id);
    return t === null ? '#' + id : quoteTok(t);
  };
  const last = n - 1;
  // 렌즈의 요약 — 앱이 계산한 값(experiment.summary 의 lens · jacobian_lens, F3-3). 받기 전에는 끝까지 1 위인 층을 칠하지 않는다
  const q = summaryOfRun(ctx, m.entry, deltaNow());
  const ls = q.val ? (view === 'jacobian' ? q.val.jacobian_lens : q.val.lens) : null;
  const first = ls ? ls.last_top1_from_layer : n;
  const K = Math.max(...L.top_ids.map(a => a.length));
  const cols = Math.min(COLS, K);
  const key = ctx.path + '|' + m.entry.id + '|' + view;
  const rows = [];
  for (let l = 0; l < n; l++) {
    const k2 = key + '|' + l;
    const open = st.open.has(k2);
    const toggle = () => {
      if (st.open.has(k2)) st.open.delete(k2);
      else st.open.add(k2);
      ctx.rerender();
    };
    rows.push(el('tr', { class: l >= first ? 'fin' : null },
      el('td', { class: 'n' }, el('button', {
        type: 'button', class: 'st-l', 'aria-expanded': String(open), 'aria-label': T`층 ${l} 의 상위 후보 ${L.top_ids[l].length} 개`, onclick: toggle,
      }, String(l))),
      Array.from({ length: cols }, (_, j) => {
        const id = L.top_ids[l][j];
        return el('td', { class: j === 0 ? 'top1' : null }, id === undefined ? '–' : [show(id), ' ', el('span', { class: 'lp' }, fx(L.top_logprobs[l][j], 2))]);
      }),
      el('td', { class: 'num' }, fmtInt(L.ranks[l])),
      el('td', { class: 'num lp' }, fx(L.token_logprobs[l], 3)),
      el('td', { class: 'num lp' }, fx(L.entropy[l], 3))));
    if (open) {
      rows.push(el('tr', { class: 'st-more' + (l >= first ? ' fin' : '') }, el('td', { colspan: String(cols + 4) },
        el('div', { class: 'st-chips' }, L.top_ids[l].map((id, j) => el('span', { class: 'st-chip' + (id === m.tok ? ' tok' : '') },
          el('span', { class: 'r' }, String(j + 1)), show(id), el('span', { class: 'lp' }, fx(L.top_logprobs[l][j], 3))))))));
    }
  }
  out.push(el('div', { class: 'st-lens-wrap' }, el('table', { class: 'st-lens' },
    el('thead', null, el('tr', null,
      el('th', { scope: 'col' }, T('층')),
      Array.from({ length: cols }, (_, j) => el('th', { scope: 'col' }, T`${j + 1} 위`)),
      el('th', { scope: 'col', class: 'num' }, T`${show(m.tok)} 의 순위`),
      el('th', { scope: 'col', class: 'num' }, T('로그확률')),
      el('th', { scope: 'col', class: 'num' }, T('엔트로피')))),
    el('tbody', null, rows))));
  if (ls) {
    const ones = ls.token_top1_layers;
    out.push(el('p', { class: 'st-note' },
      T`마지막 층의 1 위 ${show(ls.last_top1)} 는 층 ${ls.last_top1_from_layer} 부터 끝까지 1 위입니다.`, ' ',
      ones.length ? T`${who} ${show(m.tok)} 는 층 ${ranges(ones)} 에서 1 위입니다.` : T`${who} ${show(m.tok)} 는 어느 층에서도 1 위가 아닙니다.`));
  } else out.push(el('p', { class: 'st-note muted' }, q.error || T('읽는 중…')));
  if (view === 'jacobian') {
    out.push(el('p', { class: 'st-note muted' }, T`Jacobian lens — 층마다 J 파일(${String(L.jlens_sha256 || '').slice(0, 8)})의 J 로 마지막 층의 공간에 옮긴 뒤 읽습니다(정밀도 계약 5.4, J 를 쓴 층 ${fmtInt(L.source_layers)} 개). 마지막 층은 J 없이 읽어 로짓 렌즈와 같습니다.`));
  }
  if (base && base.hash) {
    const same = L.logit_hashes[last] === base.hash;
    out.push(el('p', { class: 'xp-check' }, el('span', { class: same ? 'tag ok' : 'tag err' }, same ? T('같음') : T('다름')),
      same ? T`마지막 층의 로짓 해시가 ${base.id} 의 스텝 ${base.stepNo} 의 해시와 같습니다.`
        : T`마지막 층의 로짓 해시가 ${base.id} 의 스텝 ${base.stepNo} 의 해시와 다릅니다 — 다른 모델이나 계산 방식으로 계산한 실행입니다.`));
  }
  return out;
}

/** 점수 표 — 위치 at 부터 토큰마다 로그확률 · 확률 · 순위(상위 후보 안이면)와 합. stepBase 는 첫 위치의 스텝 번호(원래 실행을 모르면 null) */
function scoreTable(ctx, m, stepBase) {
  const lp = m.res && Array.isArray(m.res.choices) && m.res.choices[0] && m.res.choices[0].logprobs;
  const tl = lp && Array.isArray(lp.token_logprobs) ? lp.token_logprobs : null;
  if (!tl) return el('p', { class: 'muted' }, T('이 실행의 응답에 토큰마다의 로그확률이 없습니다.'));
  const tops = Array.isArray(m.g.prompt_top_ids) ? m.g.prompt_top_ids : [];
  const topLps = Array.isArray(m.g.prompt_top_logprobs) ? m.g.prompt_top_logprobs : [];
  const missing = [];
  const rows = [];
  let nLp = 0;
  let minP = null;
  let minAt = null;
  for (let pos = m.at; pos < m.q.length; pos++) {
    const j = pos - m.at;
    const v = tl[pos];
    const list = Array.isArray(tops[pos]) ? tops[pos] : null;
    const r = list ? list.indexOf(m.q[pos]) : -1;
    const t = posText(m, pos);
    if (t === null) missing.push(m.q[pos]);
    if (typeof v === 'number') {
      nLp++;
      if (minP === null || Math.exp(v) < minP) { minP = Math.exp(v); minAt = j; }
    }
    // 대안 — 이 위치의 상위 후보 가운데 매긴 토큰이 아닌 것 셋(E6-2 — 연구 패널의 점수 탭)
    const alts = [];
    if (list) {
      for (let q = 0; q < list.length && alts.length < 3; q++) {
        if (list[q] === m.q[pos]) continue;
        const at2 = dictText(m, pos, (topLps[pos] || [])[q]) ?? pieceText(list[q]);
        if (at2 === null) missing.push(list[q]);
        alts.push(`${at2 === null ? '#' + list[q] : quoteTok(at2)} ${fx((topLps[pos] || [])[q], 2)}`);
      }
    }
    rows.push(el('tr', null,
      el('td', { class: 'n' }, stepBase === null ? '–' : String(stepBase + j)),
      el('td', { class: 'w' }, t === null ? '#' + m.q[pos] : quoteTok(t)),
      el('td', { class: 'num lp' }, fx(v, 3)),
      el('td', { class: 'num' }, typeof v === 'number' ? pct(Math.exp(v)) : '–'),
      el('td', { class: 'num' }, r >= 0 ? String(r + 1) : list ? T`${list.length} 위 밖` : '–'),
      el('td', { class: 'alt' }, alts.join(' · '))));
  }
  if (missing.length) fetchPieces(missing, ctx);
  const s = scoreSum(m);
  return el('div', { class: 'st-lens-wrap' }, el('table', { class: 'st-score' },
    el('thead', null, el('tr', null, el('th', { scope: 'col' }, T('스텝')), el('th', { scope: 'col' }, T('토큰')),
      el('th', { scope: 'col', class: 'num' }, T('로그확률')), el('th', { scope: 'col', class: 'num' }, T('확률')), el('th', { scope: 'col', class: 'num' }, T('순위')),
      el('th', { scope: 'col' }, T('대안')))),
    el('tbody', null, rows),
    el('tfoot', null, el('tr', null, el('td', { colspan: '2' }, T('합')), el('td', { class: 'num lp' }, fx(s, 3)),
      el('td', { class: 'num' }, s === null ? '–' : pct(Math.exp(s))), el('td', null, ''), el('td', null, '')),
    s !== null && nLp ? el('tr', null, el('td', { colspan: '6', class: 'muted' },
      T`평균 로그확률 ${fx(s / nLp, 4)} · 퍼플렉서티 ${fx(Math.exp(-s / nLp), 3)} · 가장 낮은 확률 ${pct(minP)}`,
      minAt === null ? '' : T` (스텝 ${stepBase === null ? minAt : stepBase + minAt})`)) : null)));
}

function rawDetails(rec) {
  return [
    el('details', { class: 'run-raw' }, el('summary', null, T('요청', '하나')), el('pre', { class: 'xp-pre' }, JSON.stringify(rec.request, null, 2))),
    el('details', { class: 'run-raw' }, el('summary', null, T('응답(JSON)')), el('pre', { class: 'xp-pre' }, JSON.stringify(rec.response, null, 2))),
  ];
}

const runLink = (ctx, id) => el('button', { type: 'button', class: 'linkish', onclick: () => ctx.openRun(id) }, `${T('실행', '경로')} ${id}`);

/** 점수 · 로짓 렌즈 실행의 실행 화면 — 그 표와 원래 실행의 스텝 화면으로 가는 링크(요청 · 응답 JSON 은 접어서) */
export function measureView(ctx, entry, c) {
  const m = measureOf(entry, c);
  if (!m) {
    return [el('p', { class: 'muted' }, T('이 실행의 요청에는 토큰 번호의 프롬프트와 geulpu.echo_from 이 없어 표로 보이지 않습니다.')), ...rawDetails(c.rec)];
  }
  const b = baseOf(ctx, entry, m);
  const out = [];
  if (b) {
    out.push(el('p', { class: 'st-back' }, el('button', {
      type: 'button', class: 'btn', onclick: () => ctx.openStep(b.d.entry.id, b.k),
    }, T`${b.d.entry.id} 의 스텝 ${b.d.off + b.k} 화면`)));
  }
  if (m.isLens) {
    const own = b && m.tok === b.d.ids[b.k];
    out.push(el('h2', null, T('로짓 렌즈')),
      el('div', { class: 'st-measure' }, lensTable(ctx, m, own ? T('샘플링된 토큰') : T('토큰', '렌즈'),
        b && b.d.hashes ? { id: b.d.entry.id, hash: b.d.hashes[b.k], stepNo: b.d.off + b.k } : null)));
  } else {
    out.push(el('h2', null, T('점수')), el('div', { class: 'st-measure' }, scoreTable(ctx, m, b ? b.d.off + b.k : null)));
    const t = firstTop(m);
    if (t) {
      const rows = t.ids.map((id, j) => ({ id, lp: t.lps[j], text: dictText(m, m.at, t.lps[j]) ?? pieceText(id) }));
      const missing = rows.filter(r => r.text === null).map(r => r.id);
      if (missing.length) fetchPieces(missing, ctx);
      out.push(el('h2', null, T`첫 위치의 상위 후보 ${t.ids.length} 개`), candList(rows, b ? b.d.ids[b.k] : null, null));
    }
  }
  out.push(...rawDetails(c.rec));
  return out;
}

// ---- 계산(실험의 실행으로) ----

/** 실행 d 의 스텝 k 앞까지의 토큰(프롬프트 토큰 + 답의 스텝 0 ~ k − 1) */
const before = (d, k) => d.ptoks.concat(d.ids.slice(0, k));

function doTop(ctx, d, k) {
  const N = Math.max(1, Math.min(TOP_MAX, Math.round(Number(st.topN)) || 20));
  st.topN = N;
  // 5-1 — 요청은 앱이 짓는다(experiment.score — 그 스텝까지 echo, echo_from = 프롬프트 길이 + 스텝)
  const body = { run: d.entry.id, step: d.off + k, top_logprobs: N };
  measure(ctx, 'experiment.score', JSON.stringify(body), T`${d.entry.id} 의 스텝 ${d.off + k} 에서 상위 후보 ${N} 개를 계산하는 중`, d.entry.id);
}

function doLens(ctx, d, k, jac) {
  const K = LENS_K.includes(st.lensK) ? st.lensK : 5;
  // 5-1 — 요청은 앱이 짓는다(experiment.lens)
  const body = jac ? { run: d.entry.id, step: d.off + k, top_k: K, jacobian: true } : { run: d.entry.id, step: d.off + k, top_k: K };
  measure(ctx, 'experiment.lens', JSON.stringify(body), T`${d.entry.id} 의 스텝 ${d.off + k} 에서 로짓 렌즈를 계산하는 중`, d.entry.id);
}

async function doScore(ctx, d, k, key) {
  const text = st.text.get(key) || '';
  const byIds = st.inMode.get(key) === 'ids';
  st.scoreError = null;
  if (!text.trim()) {
    st.scoreError = { key, text: byIds ? T('점수를 계산할 토큰 id 를 적으세요') : T('점수를 계산할 텍스트를 적으세요') };
    ctx.rerender();
    return;
  }
  let toks = null;
  if (byIds) {
    // 토큰 id 그대로 — 텍스트로 적을 수 없는 특수 토큰도. 그 글은 run.detokenize 로(명령 창에 그 명령이 남는다)
    const parts = text.trim().split(/[\s,]+/);
    if (!parts.every(x => /^\d{1,9}$/.test(x))) {
      st.scoreError = { key, text: T('토큰 id 는 0 이상의 정수를 쉼표나 공백으로 나눠 적습니다') };
      ctx.rerender();
      return;
    }
    toks = parts.map(Number);
    try {
      const r = await call('run.detokenize', { tokens: toks });
      if (r.ok && r.json && typeof r.json.prompt === 'string') st.idText.set(key, r.json.prompt);
      else st.scoreError = { key, text: appError(r) };
    } catch (_) {
      st.scoreError = { key, text: T('앱에 닿지 못했습니다') };
    }
  } else {
    try {
      const r = await call('run.tokenize', { prompt: text, add_special_tokens: false });
      if (r.ok && r.json && isIds(r.json.tokens)) toks = r.json.tokens;
      else st.scoreError = { key, text: appError(r) };
    } catch (_) {
      st.scoreError = { key, text: T('앱에 닿지 못했습니다') };
    }
  }
  if (toks && !toks.length) st.scoreError = { key, text: T('텍스트가 토큰이 되지 않았습니다') };
  if (st.scoreError) {
    ctx.rerender();
    return;
  }
  if (st.noCtx.get(key)) {
    // E6-2 — 문맥 없이: 시작 토큰(BOS)을 붙이는 모델은 그 뒤에, 아니면 첫 토큰을 문맥으로(그 토큰은 점수가 없다)
    let bos = null;
    try {
      const r = await call('run.tokenizer_info');
      if (r.ok && r.json && r.json.add_bos_token && Number.isInteger(r.json.bos_token_id)) bos = r.json.bos_token_id;
    } catch (_) { /* 앱에 닿지 못하면 시작 토큰 없이 */ }
    const q = (bos === null ? [] : [bos]).concat(toks);
    if (q.length < 2) {
      st.scoreError = { key, text: T('이 모델은 시작 토큰이 없어 첫 토큰에는 점수가 없습니다 — 토큰이 둘 이상이어야 합니다.') };
      ctx.rerender();
      return;
    }
    const nb = { prompt: q, echo: true, max_tokens: 0, logprobs: 1, geulpu: { echo_from: 1, top_logprobs: SCORE_TOP, logit_hash: true } };
    const id = await measure(ctx, 'run.score', JSON.stringify(nb), T('문맥 없이 점수를 계산하는 중'), d.entry.id);
    if (id) ctx.openRun(id);
    return;
  }
  const body = {
    prompt: before(d, k).concat(toks), echo: true, max_tokens: 0, logprobs: 1,
    geulpu: { echo_from: d.ptoks.length + k, top_logprobs: SCORE_TOP, logit_hash: true },
  };
  measure(ctx, 'run.score', JSON.stringify(body), T`${d.entry.id} 의 스텝 ${d.off + k} 에서 점수를 계산하는 중`, d.entry.id);
}

// ---- 스텝 · 층 화면 ----

const tokAt = (d, i) => (d.texts ? (d.texts[i] === null ? '' : d.texts[i]) : `#${d.ids[i]} `);

/** 프롬프트의 끝(헤더의 텍스트 — 스텝이 답의 앞쪽일 때) */
function promptTail(d) {
  let s = '';
  if (typeof d.req.prompt === 'string') s = d.req.prompt;
  else if (Array.isArray(d.req.messages) && d.req.messages.length) {
    const m = d.req.messages[d.req.messages.length - 1];
    s = m && typeof m.content === 'string' ? m.content : '';
  } else if (d.ptexts) s = d.ptexts.join('');
  return s.length > 80 ? s.slice(-80) : s;
}

function contextLine(d, k) {
  const from = Math.max(0, k - CTX_BEFORE);
  const to = Math.min(d.n, k + 1 + CTX_AFTER);
  let pre = '';
  for (let i = from; i < k; i++) pre += tokAt(d, i);
  let post = '';
  for (let i = k + 1; i < to; i++) post += tokAt(d, i);
  const head = from === 0 ? promptTail(d) : '';
  const tok = d.texts && d.texts[k] !== null ? d.texts[k] : d.raw ? d.raw[k] : '#' + d.ids[k];
  return el('p', { class: 'st-ctx', 'aria-label': T('스텝 앞뒤의 텍스트') },
    from > 0 || head ? el('span', { class: 'run-p' }, '…' + head) : null,
    pre, el('mark', { class: 'st-tok' }, tok || ' '), post, to < d.n ? el('span', { class: 'run-p' }, '…') : null);
}

function section(title, ...kids) {
  return el('section', { class: 'st-sec', 'aria-label': title }, el('h2', null, title), kids);
}

function candSection(ctx, d, k, ms, can) {
  const at = d.ptoks ? d.ptoks.length + k : null;
  const top = ms ? bestTop(ms, d, k) : null;
  const kids = [];
  let rows = [];
  if (top) {
    rows = top.ids.map((id, j) => ({ id, lp: top.lps[j], text: id === d.ids[k] && d.raw ? d.raw[k] : dictText(top.m, at, top.lps[j]) ?? pieceText(id) }));
    const echoHash = Array.isArray(top.m.g.prompt_logit_hashes) ? top.m.g.prompt_logit_hashes[top.pos - top.m.at] : null;
    const runHash = d.hashes ? d.hashes[k] : null;
    kids.push(el('p', { class: 'st-meta' }, T`상위 후보 ${top.ids.length} 개`, ' · ', runLink(ctx, top.m.entry.id),
      top.m.sameModel ? null : el('span', { class: 'tag err' }, T('다른 모델')),
      echoHash && runHash ? el('span', { class: echoHash === runHash ? 'tag ok' : 'tag err' },
        echoHash === runHash ? T('로짓 해시가 실행과 같음') : T('로짓 해시가 실행과 다름')) : null));
  } else if (d.topIds && d.topLp) {
    const tids = d.topIds[k] || [];
    rows = tids.map((id, j) => ({ id, lp: d.topLp[k][j], text: candText(d, k, j) }));
    kids.push(el('p', { class: 'st-meta' }, T`실행의 기록 — 상위 ${tids.length} 개(샘플링 설정 전 — 모델의 분포)`));
  } else {
    kids.push(el('p', { class: 'muted' }, T('이 실행에는 스텝마다의 후보가 없습니다 — logprobs 와 geulpu.top_logprobs 를 넣어 실행하면 보입니다.')));
  }
  const missing = rows.filter(r => r.text === null).map(r => r.id);
  if (missing.length) fetchPieces(missing, ctx);
  if (rows.length) kids.push(candList(rows, d.ids[k], d.ptoks ? id => doBranch(ctx, d, k, id) : null));
  const nIn = el('input', {
    id: 'st-top-n', class: 'input', type: 'number', min: '1', max: String(TOP_MAX), step: '1', value: String(st.topN),
    oninput: ev => { st.topN = ev.target.value; },
  });
  kids.push(el('div', { class: 'xp-actions st-form' },
    el('label', { class: 'st-inline', for: 'st-top-n' }, T('후보 수')), nIn,
    el('button', { type: 'button', class: 'btn', disabled: can ? null : true, onclick: () => doTop(ctx, d, k) }, T('상위 후보 계산')),
    el('span', { class: 'muted' }, T('echo 스코어링으로 이 스텝의 상위 후보를 1 ~ 128 개 계산해 이 실험의 실행으로 남깁니다.'))));
  return section(T('상위 후보'), kids);
}

function lensSection(ctx, d, k, ms, can, srv) {
  const kids = [];
  const m = ms ? ms.lens[0] || null : null;
  const lensOk = !srv || srv.lensOk;
  const jac = !!(srv && srv.jsha);
  const sel = el('select', {
    id: 'st-lens-k', class: 'input', onchange: ev => { st.lensK = Number(ev.target.value); },
  }, LENS_K.map(v => el('option', { value: String(v), selected: v === st.lensK ? true : null }, String(v))));
  kids.push(el('div', { class: 'xp-actions st-form' },
    el('label', { class: 'st-inline', for: 'st-lens-k' }, T('층마다 상위 후보')), sel,
    el('button', { type: 'button', class: 'btn', disabled: can && lensOk ? null : true, onclick: () => doLens(ctx, d, k, jac) },
      jac ? T('로짓 렌즈 · Jacobian lens 계산') : T('로짓 렌즈 계산'))));
  if (srv && !srv.lensOk) kids.push(el('p', { class: 'muted' }, T('불러온 모델의 구조는 로짓 렌즈를 받지 않습니다.')));
  else if (srv && !jac) {
    kids.push(el('p', { class: 'muted st-note' }, srv.jreason ? T`모델 옆의 J 파일을 받지 않았습니다: ${srv.jreason}`
      : T('Jacobian lens: 모델 파일 옆에 <이름>.jlens.safetensors(도구/jlens_fit.py로 원본 모델에서 맞춘 J 파일)를 두고 모델을 다시 불러오면 함께 읽습니다.')));
  }
  if (m) {
    kids.push(el('p', { class: 'st-meta' }, T`층 ${m.lens.layers} 개 · 층마다 상위 후보 ${Math.max(...m.lens.top_ids.map(a => a.length))} 개`, ' · ',
      runLink(ctx, m.entry.id), m.sameModel ? null : el('span', { class: 'tag err' }, T('다른 모델'))));
    kids.push(el('div', { class: 'st-measure' }, lensTable(ctx, m, T('샘플링된 토큰'),
      d.hashes ? { id: d.entry.id, hash: d.hashes[k], stepNo: d.off + k } : null)));
  } else {
    kids.push(el('p', { class: 'muted' }, T('층마다 이 스텝의 계산을 끝의 출력으로 읽어, 그 층에서 멈췄다면 어떤 토큰을 냈을지 보입니다(정밀도 계약 5.3). 계산하면 이 실험의 실행으로 남습니다.')));
  }
  if (ms && ms.other.length) {
    kids.push(el('p', { class: 'st-meta' }, T('다른 토큰의 로짓 렌즈: '),
      ms.other.map((o, i) => [i ? ' · ' : '', runLink(ctx, o.entry.id), ' ', quoteTok(posText(o, o.at) ?? '#' + o.tok)])));
  }
  return section(T('로짓 렌즈'), kids);
}

function scoreSection(ctx, d, k, ms, can) {
  const key = ctx.path + '|' + d.entry.id + '|' + k;
  const kids = [];
  const mode = st.inMode.get(key) || 'text';
  const modeSel = el('select', {
    id: 'st-score-mode', class: 'input st-inmode', 'aria-label': T('점수 입력의 종류'),
    onchange: ev => { st.inMode.set(key, ev.target.value); st.idText.delete(key); st.scoreError = null; ctx.rerender(); },
  }, el('option', { value: 'text', selected: mode === 'text' ? true : null }, T('텍스트')),
  el('option', { value: 'ids', selected: mode === 'ids' ? true : null }, T('토큰 id')));
  const input = el('input', {
    id: 'st-score-in', class: 'input', type: 'text', value: st.text.get(key) || '', autocomplete: 'off',
    placeholder: mode === 'ids' ? T('예: 151645, 198') : T('예: " Paris"'), 'aria-label': mode === 'ids' ? T('토큰 id') : T('텍스트'),
    oninput: ev => { st.text.set(key, ev.target.value); if (mode === 'ids') st.idText.delete(key); },
    onkeydown: ev => { if (ev.key === 'Enter' && can) { ev.preventDefault(); doScore(ctx, d, k, key); } },
  });
  const noCtx = el('input', { type: 'checkbox', id: 'st-score-noctx', checked: st.noCtx.get(key) ? true : null, onchange: ev => { st.noCtx.set(key, ev.target.checked); } });
  kids.push(el('div', { class: 'xp-actions st-form' }, modeSel, input,
    el('button', { type: 'button', class: 'btn', disabled: can ? null : true, onclick: () => doScore(ctx, d, k, key) }, T('점수 계산')),
    el('label', { class: 'run-check', for: 'st-score-noctx' }, noCtx, ' ', T('문맥 없이'))));
  if (mode === 'ids' && st.idText.has(key)) kids.push(el('p', { class: 'st-meta' }, T('토큰 id 의 글: '), el('b', null, quoteTok(st.idText.get(key)))));
  kids.push(el('p', { class: 'muted st-note' }, mode === 'ids'
    ? T('토큰 id 를 그대로 이 스텝부터 이어 붙이고, 토큰마다의 로그확률을 계산합니다(echo 스코어링) — 텍스트로 적을 수 없는 특수 토큰(예: <|im_end|>)도 됩니다. 글은 run.detokenize 로 보입니다.')
    : T('텍스트를 토큰으로 나눠(특수 토큰 없이) 이 스텝부터 이어 붙이고, 토큰마다의 로그확률을 계산합니다(echo 스코어링). 앞의 공백도 텍스트입니다.')));
  if (st.scoreError && st.scoreError.key === key) kids.push(el('p', { class: 'xp-check' }, el('span', { class: 'tag err' }, T('실패')), st.scoreError.text));
  const list = ms ? ms.scores.filter(x => !(x.q.length === x.at + 1 && x.tok === d.ids[k])) : [];
  for (const x of list.slice(0, 5)) {
    const s = scoreSum(x);
    kids.push(el('div', { class: 'st-scored' },
      el('p', { class: 'st-meta' }, runLink(ctx, x.entry.id), ' ', el('b', null, quoteTok(scoredText(x))),
        s === null ? null : T` · 로그확률 ${fx(s, 3)} · 확률 ${pct(Math.exp(s))}`, x.sameModel ? null : el('span', { class: 'tag err' }, T('다른 모델'))),
      scoreTable(ctx, x, d.off + k)));
  }
  if (list.length > 5) kids.push(el('p', { class: 'muted' }, T`이 스텝의 점수 실행 ${list.length} 개 가운데 최근 5 개 — 나머지는 실행 탭에서.`));
  return section(T('점수'), kids);
}

/** 스텝 · 층 화면(경로: 실험 › id › 실행 › 스텝 k) — k 는 실행 안의 위치(0 부터), 보이는 스텝 번호는 실행 화면과 같다(분기는 원래 실행의 번호를 잇는다) */
export function stepView(ctx, id, k0) {
  const entry = entryOf(ctx, id);
  if (!entry) return [el('div', { class: 'empty-box' }, el('div', { class: 'big' }, T`이 실험에 없는 실행입니다: ${id}`))];
  const c = recordOf(ctx, entry);
  if (c.loading) return [el('p', { class: 'muted' }, T('읽는 중…'))];
  if (c.error) return [el('div', { class: 'empty-box' }, el('div', { class: 'big' }, T('기록 파일을 읽지 못했습니다')), el('div', null, c.error))];
  const d = dataOf(entry, c.rec, c.text);
  if (!d.n) return [el('div', { class: 'empty-box' }, el('div', { class: 'big' }, T('이 실행에는 생성한 스텝이 없습니다')))];
  const k = Math.max(0, Math.min(d.n - 1, k0));
  const v = stepVals(d, k);
  const srv = loaded(serverState(ctx));
  const ms = d.ptoks ? measuresAt(ctx, d, k) : null;
  const tok = d.texts && d.texts[k] !== null ? d.texts[k] : d.raw ? d.raw[k] : '#' + d.ids[k];
  const go = k2 => {
    selectStep(ctx, id, k2);
    ctx.openStep(id, k2);
  };
  // 계산할 수 있는가 — 프롬프트 토큰이 기록된 실행, 진행 중인 실행 없음, 모델이 준비됨(상태를 아직 모르면 막지 않는다 — 앱이 까닭을 답한다)
  const notes = [];
  if (!d.ptoks) notes.push(T('이 실행은 프롬프트 토큰을 기록하지 않아 이 스텝을 다시 계산할 수 없습니다(geulpu.prompt_tokens).'));
  if (st.server && (!srv || !srv.ready)) notes.push(T('불러온 모델이 없습니다 — 모델 화면에서 모델을 불러오면 계산할 수 있습니다.'));
  const can = !!d.ptoks && !isBusy() && !(st.server && (!srv || !srv.ready));
  const root = el('div', { class: 'st-view', tabindex: '-1', 'aria-label': T`스텝 ${d.off + k}` },
    el('h1', null, T`스텝 ${d.off + k}`, ' ', el('span', { class: 'run-kind' }, quoteTok(tok))),
    el('p', { class: 'lead muted' }, T('샘플링된 토큰과 이 스텝의 분포, 층마다의 예측. 이 화면에서 계산한 것은 이 실험의 실행으로 남습니다.')),
    runInfo(ctx, d),
    matchLine(ctx, d),
    ...busyLine(ctx, id),
    el('div', { class: 'st-top' },
      contextLine(d, k),
      el('dl', { class: 'run-kv st-kv' },
        el('dt', null, T('마진')), el('dd', null, v.margin === null ? '–' : fx(v.margin, 4) + ' nat'),
        el('dt', null, T('샘플링된 토큰의 확률')), el('dd', null, pct(v.prob) + (v.rank ? T` · ${v.rank} 위` : '')),
        el('dt', null, T('엔트로피')), el('dd', null, v.ent === null ? '–' : fx(v.ent, 3) + ' nat'),
        el('dt', null, T('로짓 해시')), el('dd', null, v.hash ? el('code', { class: 'hash', title: v.hash }, String(v.hash).slice(0, 8)) : '–')),
      el('div', { class: 'st-nav' },
        el('button', { type: 'button', class: 'btn sm', disabled: k <= 0 ? true : null, onclick: () => go(k - 1) }, '← ', T`스텝 ${d.off + k - 1}`),
        el('button', { type: 'button', class: 'btn sm', disabled: k >= d.n - 1 ? true : null, onclick: () => go(k + 1) }, T`스텝 ${d.off + k + 1}`, ' →'),
        el('span', { class: 'muted' }, T('← → 이전 · 다음 스텝')))),
    notes.map(t => el('p', { class: 'xp-check run-warn' }, el('span', { class: 'tag err' }, '!'), t)),
    ms && ms.loading ? el('p', { class: 'muted' }, T('이 실험의 점수 · 로짓 렌즈 실행을 읽는 중…')) : null,
    candSection(ctx, d, k, ms, can),
    lensSection(ctx, d, k, ms, can, srv),
    scoreSection(ctx, d, k, ms, can));
  root.addEventListener('keydown', ev => {
    if (ev.altKey || ev.ctrlKey || ev.metaKey || ev.target.closest('input, textarea, select')) return;
    if (ev.key === 'ArrowLeft' && k > 0) {
      ev.preventDefault();
      go(k - 1);
    } else if (ev.key === 'ArrowRight' && k < d.n - 1) {
      ev.preventDefault();
      go(k + 1);
    }
  });
  // 다시 그려도(기록을 읽었을 때 · 다른 출처의 호출이 끝났을 때) 화면 · 입력 필드의 초점과 커서를 그대로 둔다
  const a = document.activeElement;
  if (a && a.classList && a.classList.contains('st-view')) queueMicrotask(() => root.focus({ preventScroll: true }));
  else if (a && a.id && /^st-/.test(a.id)) {
    const keep = { id: a.id, s: a.selectionStart, e: a.selectionEnd };
    queueMicrotask(() => {
      const x = document.getElementById(keep.id);
      if (!x) return;
      x.focus({ preventScroll: true });
      try { if (keep.s !== null && keep.s !== undefined) x.setSelectionRange(keep.s, keep.e); } catch (_) { /* number 입력 필드에는 커서가 없다 */ }
    });
  }
  return [root];
}

/** 경로에 쓰는 스텝 화면의 이름 */
export function stepTitle(ctx, id, k) {
  const d = dataById(ctx, id);
  return T`스텝 ${(d ? d.off : 0) + k}`;
}
