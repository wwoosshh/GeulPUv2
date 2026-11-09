// 앱리디자인 E6-2(문서/화면구현.md §6) — 실행 화면 · 비교 탭의 연구 도구. 연구 패널(채팅 옆)에서 옮겼다: 모델 · 계산 방식 맞추기, 요약 수치 · δ,
// 실행 정보에 더한 값(끝난 까닭 · 토큰/초 · 추측 디코딩 · 지문), 프롬프트 캐시의 재사용, 모든 스텝의 상위 후보, 캐시 없이 재실행, 다른 모델로 점수,
// 내보내기(토큰 표 geulpu-tokens/1 · 재현 파일 geulpu-capsule/1 · 파이썬), 비교의 처음 다른 스텝 · 수치 · 다른 프롬프트로 점수.
// 계산은 모두 기능 목록의 기능(run.score · model.load · settings.set)으로 하고, 점수는 이 실험의 실행으로 남는다(measure).
// F1(문서/사용성시험.md §1) — 여기서 부른 모델 불러오기 · 계산 방식 바꾸기도 이 실험의 명령 기록에 남긴다(실험 헤더 — CLI 의 --experiment 와 같다).
import { T, el } from './dom.js';
import { call } from './ops.js';
import {
  appError, dataOf, fetchPieces, fx, isIds, measure, pct, pieceText, quoteTok, rawRequest, recordOf, runsOf, stepVals, summaryOfRun,
} from './runs.js';

const DELTA_KEY = 'geulpu.lab.delta';                 // 연구 패널과 같은 키(이 브라우저에 기억한 δ 를 그대로)
const TOP_DEFAULT = 20;
const TOP_MAX = 128;
const CAPSULE_K = 10;                                  // 재현 파일을 위해 새로 계산할 때의 상위 후보 수(연구 패널과 같다)
const rs = {
  delta: readDelta(),
  topK: TOP_DEFAULT,
  state: null,                                         // app.js 가 1 초마다 읽는 상태(geulpu:state)
  ctx: null,                                           // 마지막으로 그린 실험 화면(다시 그리기)
  fp: new Map(),                                       // '<실험>|<실행>' → {fp, fpLogit} · null(계산 중)
  busy: null,                                          // 모델 불러오기 · 계산 방식 바꾸기 중의 글
  err: null,
  otherModel: '',                                      // 다른 모델로 점수 — 고른 모델 id
  sum: new Map(),                                      // '<실험>|<실행>' → 마지막으로 받은 요약(δ 를 바꾸는 동안 보인다)
};

function readDelta() {
  try {
    const v = JSON.parse(localStorage.getItem(DELTA_KEY));
    return typeof v === 'number' && v > 0 && v <= 5 ? v : 0.05;
  } catch (_) {
    return 0.05;
  }
}
function saveDelta(v) {
  rs.delta = v;
  try { localStorage.setItem(DELTA_KEY, JSON.stringify(v)); } catch (_) { /* 기억하지 못해도 이번 화면에는 쓴다 */ }
}
export const deltaNow = () => rs.delta;
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---- 앱 상태 — 불러온 모델 · 계산 방식이 바뀌면 다시 그린다 ----

const stateSig = s => {
  const sv = (s && s.server) || {};
  return JSON.stringify([sv.model, sv.state, sv.precision, sv.kv8, sv.fast_attention, sv.yarn,
    ((s && s.models) || []).map(m => [m.id, m.sha256 || ''])]);
};
document.addEventListener('geulpu:state', ev => {
  const before = stateSig(rs.state);
  rs.state = ev.detail;
  if (stateSig(rs.state) !== before && rs.ctx && document.querySelector('.rs-match, .run-split, .st-view')) rs.ctx.rerender();
});
export const appState = () => rs.state;

const models = () => (rs.state && Array.isArray(rs.state.models) ? rs.state.models : []);
const modelBySha = sha => models().find(m => m && m.sha256 && m.sha256 === sha) || null;
const modeText = s => [s.precision === 'fast' ? T('빠름') : T('정밀'), s.kv8 ? T('KV 8비트') : null, s.fast_attention ? T('빠른 어텐션') : null]
  .filter(Boolean).join(' · ');
const yarnText = y => (y ? '×' + y : T('YaRN 없음'));

// ---- 모델 · 계산 방식 맞추기(실행 화면 · 스텝 · 층 화면) ----

/** 불러온 모델(SHA-256)이나 계산 방식이 실행과 다르면 그 줄과 맞추는 단추. 같으면 null */
export function matchLine(ctx, d) {
  rs.ctx = ctx;
  const want = (d && d.rec && d.rec.server) || {};
  const stt = rs.state;
  if (!stt || !stt.server || !want.sha256) return null;
  const sv = stt.server;
  const cur = models().find(m => m && m.id === sv.model) || null;
  const lines = [];
  const btns = [];
  if (rs.busy) lines.push(rs.busy);
  else if (!(sv.state === 'ready' && cur && cur.sha256 === want.sha256)) {
    const target = modelBySha(want.sha256);
    lines.push(sv.state === 'ready' && cur ? T`불러온 모델(${cur.name || cur.id})이 이 실행의 모델과 다릅니다 — 여기서 계산하면 불러온 모델의 값입니다.`
      : T('불러온 모델이 없습니다.'));
    if (target) btns.push(el('button', { type: 'button', class: 'btn sm', onclick: () => loadModel(ctx, target.id) }, T('이 실행의 모델 불러오기')));
    else lines.push(T`같은 모델 파일(SHA-256 ${String(want.sha256).slice(0, 12)}…)이 설치되어 있지 않습니다 — 모델 › 파일에서 그 파일을 더하거나 무결성을 확인하세요.`);
  } else {
    const off = (want.precision && want.precision !== sv.precision) || !!want.kv8 !== !!sv.kv8 || !!want.fast_attention !== !!sv.fast_attention;
    if (off) {
      lines.push(T`계산 방식이 이 실행과 다릅니다 — 실행 ${modeText(want)} · 지금 ${modeText(sv)}. 여기서 계산하면 지금 방식의 값입니다.`);
      if (typeof globalThis.geulpuSetMode === 'function') {
        btns.push(el('button', { type: 'button', class: 'btn sm', onclick: () => matchMode(ctx, want) }, T('이 실행의 계산 방식으로')));
      }
    }
    if ((want.yarn || 0) !== (sv.yarn || 0)) {
      lines.push(T`문맥 길이(YaRN 배)가 이 실행과 다릅니다 — 실행 ${yarnText(want.yarn)} · 지금 ${yarnText(sv.yarn)}. 설정 › 앱의 문맥 길이를 그 실행과 같게 하고 모델을 다시 불러오세요.`);
    }
  }
  if (rs.err) lines.push(rs.err);
  if (!lines.length) return null;
  return el('div', { class: 'note-line rs-match', role: 'status' },
    el('div', { class: 'col g8' }, lines.map(t => el('span', null, t)), btns.length ? el('div', { class: 'row g8' }, btns) : null));
}

/** 이 화면의 실험 폴더 — 모델 · 계산 방식을 바꾼 호출도 그 실험의 명령 기록에 남도록 */
const expPath = ctx => (ctx && ctx.exp && ctx.exp.path) || '';

/** 모델 하나를 불러오고(작업) 준비될 때까지 기다린다. exp 가 있으면 그 실험의 명령 기록에 남긴다. 성공하면 참 */
async function waitLoaded(id, exp) {
  const r = await call('model.load', { model: id }, { experiment: exp || '' });
  if (!r.ok) throw new Error(appError(r));
  for (let i = 0; i < 480; i++) {
    await sleep(500);
    const s = await call('system.state');
    const sv = s.ok && s.json ? s.json.server || {} : {};
    if (s.ok && s.json) rs.state = s.json;
    if (sv.model === id && sv.state === 'ready') return true;
    if (s.ok && s.json && s.json.job && s.json.job.state === 'failed') throw new Error(s.json.job.reason || T('모델을 불러오지 못했습니다'));
  }
  throw new Error(T('모델을 불러오지 못했습니다 — 시간이 다 됐습니다'));
}

async function loadModel(ctx, id) {
  rs.busy = T('모델을 불러오는 중…');
  rs.err = null;
  ctx.rerender();
  try { await waitLoaded(id, expPath(ctx)); } catch (e) { rs.err = e.message; }
  rs.busy = null;
  ctx.rerender();
}

async function matchMode(ctx, want) {
  rs.busy = T('계산 방식을 바꾸는 중…');
  rs.err = null;
  ctx.rerender();
  try {
    const ok = await globalThis.geulpuSetMode(want.precision || 'exact', { kv8: !!want.kv8, f2: !!want.fast_attention, experiment: expPath(ctx) });
    if (!ok) rs.err = T('계산 방식을 바꾸지 못했습니다 — 답이 생성되는 중이거나 작업이 실행 중이면 끝난 뒤에 다시 누르세요.');
  } catch (e) {
    rs.err = e.message;
  }
  rs.busy = null;
  ctx.rerender();
}

// ---- 요약 수치 · δ ----

/** 토큰별 출력 위의 요약 수치와 δ(마진 < δ 인 스텝 사이를 옮겨 다닌다) — 값은 앱이 계산한다(experiment.summary, F3-3). δ 를 바꿔 새 값을
   받는 동안은 그 실행의 앞 값을 보인다 */
export function summaryView(ctx, d, sel, onSel) {
  rs.ctx = ctx;
  const q = summaryOfRun(ctx, d.entry, rs.delta);
  const last = ctx.path + '|' + d.entry.id;
  if (q.val) rs.sum.set(last, q.val);
  const s = q.error ? null : q.val || rs.sum.get(last) || null;
  if (!s) return el('div', { class: 'rs-sum' }, el('p', { class: 'muted' }, q.error || T('읽는 중…')));
  const lows = s.low_margin_steps.map(x => x - s.first_step);
  const prev = [...lows].reverse().find(k => k < sel);
  const next = lows.find(k => k > sel);
  const dIn = el('input', {
    id: 'rs-delta', class: 'input rs-delta-in', type: 'number', min: '0.001', max: '5', step: '0.01', value: String(rs.delta),
    onchange: ev => {
      const v = parseFloat(ev.target.value);
      if (v > 0 && v <= 5) {
        saveDelta(v);
        ctx.rerender();
      }
    },
  });
  const mm = s.min_margin;
  return el('div', { class: 'rs-sum' },
    el('dl', { class: 'run-kv rs-kv' },
      el('dt', null, T('평균 로그확률')), el('dd', null, fx(s.mean_logprob, 4)),
      el('dt', null, T('퍼플렉서티')), el('dd', null, fx(s.perplexity, 3)),
      el('dt', null, T('평균 엔트로피')), el('dd', null, s.mean_entropy === null ? '–' : fx(s.mean_entropy, 3) + ' nat'),
      el('dt', null, T('가장 작은 마진')), el('dd', null, mm ? T`${fx(mm.margin, 4)} nat · 스텝 ${mm.step}` : '–'),
      el('dt', null, T('마진 < δ 인 스텝')), el('dd', null, `${s.low_margin_steps.length} / ${s.margin_steps}`),
      el('dt', null, T('1위가 아닌 스텝')), el('dd', null, `${s.not_first_steps.length} / ${s.ranked_steps}`)),
    el('div', { class: 'xp-actions rs-delta' },
      el('label', { class: 'st-inline', for: 'rs-delta' }, T('마진 기준 δ')), dIn,
      el('button', { type: 'button', class: 'btn sm', disabled: prev === undefined ? true : null, onclick: () => onSel(prev, true) }, T('이전 — 마진 < δ')),
      el('button', { type: 'button', class: 'btn sm', disabled: next === undefined ? true : null, onclick: () => onSel(next, true) }, T('다음 — 마진 < δ'))));
}

// ---- 실행 정보에 더하는 값 · 지문 ----

async function sha256Hex(s) {
  const b = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s));
  return [...new Uint8Array(b)].map(x => x.toString(16).padStart(2, '0')).join('');
}
/** 지문(geulpu-fp/1) — 연구 패널 · 도구/repro_check.py 와 같은 계산: 토큰 지문 = SHA-256(모델 · 프롬프트 · 답 토큰), 로짓 지문 = SHA-256(토큰 지문 · 로짓 해시) */
export async function fpOf(sha, prompt, tokens, hashes) {
  const fp = await sha256Hex(`geulpu-fp/1\nmodel ${sha}\nprompt ${prompt.join(',')}\ntokens ${tokens.join(',')}\n`);
  return { fp, fpLogit: await sha256Hex(`${fp}\n${hashes.join(',')}\n`) };
}
function fingerprint(ctx, d) {
  const sha = (d.rec.server || {}).sha256;
  if (!sha || !d.ptoks || !Array.isArray(d.hashes) || d.hashes.length !== d.n) return null;
  const key = ctx.path + '|' + d.entry.id;
  if (!rs.fp.has(key)) {
    rs.fp.set(key, null);
    fpOf(sha, d.ptoks, d.ids, d.hashes).then(v => { rs.fp.set(key, v); if (rs.ctx) rs.ctx.rerender(); }, () => { rs.fp.delete(key); });
  }
  return rs.fp.get(key);
}

/** 실행 정보 줄에 더하는 것 — 끝난 까닭 · 토큰/초 · 추측 디코딩 · 백엔드 · KV 8비트 · 빠른 어텐션 · 지문 */
export function infoExtras(ctx, d) {
  const g = d.g || {};
  const s = d.rec.server || {};
  const out = [];
  const fin = d.ch && d.ch.finish_reason;
  if (fin) out.push(el('span', null, T('끝난 까닭'), ' ', el('b', null, String(fin))));
  if (typeof g.generate_us === 'number' && g.generate_us > 0 && d.n > 1) out.push(el('span', null, el('b', null, fx((d.n - 1) / (g.generate_us / 1e6), 1)), ' ', T('토큰/초')));
  if (typeof g.draft_proposed === 'number' && g.draft_proposed > 0) out.push(el('span', null, T('추측 디코딩'), ' ', el('b', null, `${g.draft_accepted || 0}/${g.draft_proposed}`)));
  if (s.layer) out.push(el('span', null, T('백엔드'), ' ', el('b', null, s.layer === 'cuda' ? 'CUDA' : s.layer === 'vulkan' ? 'Vulkan' : String(s.layer))));
  if (s.kv8 || s.fast_attention) out.push(el('span', null, [s.kv8 ? T('KV 8비트') : null, s.fast_attention ? T('빠른 어텐션') : null].filter(Boolean).join(' · ')));
  const f = d.n ? fingerprint(ctx, d) : null;
  if (f) {
    out.push(el('span', { title: T`토큰 지문 ${f.fp} · 로짓 지문 ${f.fpLogit} (geulpu-fp/1)` }, T('지문'), ' ',
      el('code', null, f.fp.slice(0, 8)), ' · ', el('code', null, f.fpLogit.slice(0, 8))));
  }
  return out;
}

// ---- 모든 스텝의 상위 후보(답 전체의 echo 점수 실행) ----

const sameIds = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

/** 이 실행의 답 전체를 점수 매긴 실행들(프롬프트 = 프롬프트 토큰 + 답 토큰, echo_from = 프롬프트 길이) — 상위 후보가 많은 것부터 */
export function fullScores(ctx, d) {
  if (!d.ptoks || !d.n) return [];
  const want = d.ptoks.concat(d.ids);
  const sha = (d.rec.server || {}).sha256;
  const out = [];
  for (const e of runsOf(ctx)) {
    if (e.operation !== 'run.score' || e.id === d.entry.id) continue;
    const c = recordOf(ctx, e);
    if (!c.rec) continue;
    const req = c.rec.request && typeof c.rec.request === 'object' ? c.rec.request : {};
    const gq = req.geulpu && typeof req.geulpu === 'object' ? req.geulpu : {};
    if (!isIds(req.prompt) || gq.echo_from !== d.ptoks.length || !sameIds(req.prompt, want)) continue;
    const res = c.rec.response && typeof c.rec.response === 'object' ? c.rec.response : null;
    const g = (res && res.geulpu) || {};
    const lp = res && Array.isArray(res.choices) && res.choices[0] && res.choices[0].logprobs;
    const P = d.ptoks.length;
    const tops = Array.isArray(g.prompt_top_ids) ? g.prompt_top_ids[P] : null;
    if (!Array.isArray(tops) || !lp || !Array.isArray(lp.token_logprobs)) continue;
    out.push({
      entry: e, rec: c.rec, g, K: tops.length, sameModel: !sha || ((c.rec.server || {}).sha256 === sha),
      lp: lp.token_logprobs.slice(P, P + d.n), ent: Array.isArray(g.prompt_entropy) ? g.prompt_entropy.slice(P, P + d.n) : [],
      topIds: g.prompt_top_ids.slice(P, P + d.n), topLp: Array.isArray(g.prompt_top_logprobs) ? g.prompt_top_logprobs.slice(P, P + d.n) : [],
      hashes: Array.isArray(g.prompt_logit_hashes) ? g.prompt_logit_hashes.slice(0, d.n) : [],
    });
  }
  return out.sort((a, b) => (a.sameModel === b.sameModel ? b.K - a.K : a.sameModel ? -1 : 1));
}

/** 스텝 k 의 상위 후보 — 답 전체의 점수 실행 가운데 같은 모델 · 후보가 가장 많은 것. 없으면 null */
export function coverTop(ctx, d, k) {
  const f = fullScores(ctx, d).find(x => x.sameModel);
  if (!f || !Array.isArray(f.topIds[k]) || !Array.isArray(f.topLp[k])) return null;
  return { ids: f.topIds[k], lps: f.topLp[k], run: f.entry.id, K: f.K };
}

function scoreAllBody(d, K) {
  return JSON.stringify({
    prompt: d.ptoks.concat(d.ids), echo: true, max_tokens: 0, logprobs: Math.min(K, 20),
    geulpu: { echo_from: d.ptoks.length, top_logprobs: K, logit_hash: true },
  });
}
function scoreAll(ctx, d, K) {
  // 5-1 — 답 전체의 점수는 앱이 짓는다(experiment.score — 재현 파일에 드는 것). 다른 모델로 점수는 그대로 run.score
  return measure(ctx, 'experiment.score', JSON.stringify({ run: d.entry.id, top_logprobs: K }), T`${d.entry.id} 의 모든 스텝에서 상위 후보 ${K} 개를 계산하는 중`, d.entry.id);
}

// ---- 캐시 없이 재실행 ----

async function rerunNoCache(ctx, d, text) {
  const raw = rawRequest(text);
  if (!raw) return;
  const body = raw.trim() === '{}' ? '{"cache_prompt":false}' : raw.replace(/^\s*\{/, '{"cache_prompt":false,');
  const id = await measure(ctx, d.rec.op, body, T`${d.entry.id} 을 캐시 없이 재실행하는 중`, d.entry.id);
  if (id) ctx.openCompare(d.entry.id, id);
}

// ---- 다른 모델로 점수 ----

/** 모델의 양자화 이름 — 앱의 모델 목록(app.js 의 quantOf)과 같은 차례: 모양의 형식 → 파일 형식(general.file_type) → 행렬의 형식 */
function quantLabel(m) {
  const s = (m && m.shape) || {};
  if (!(s.experts > 0) && s.quant && !['K', 'KI', 'W16'].includes(s.quant)) return s.quant;
  return s.file_type || s.weight_type || s.quant || '';
}

/** 이 실행의 모델과 어휘 · 채팅 템플릿이 같은, 설치된 다른 모델(연구 패널과 같은 기준) */
function otherModels(d) {
  const base = modelBySha((d.rec.server || {}).sha256);
  if (!base || !base.shape || !base.shape.vocab) return [];
  return models().filter(m => m && m.id !== base.id && m.shape && m.shape.ok && m.shape.vocab === base.shape.vocab
    && m.template === base.template && !m.device_reason);
}

/** 같은 토큰을 다른 모델로 점수 매긴 실행(그 모델의 SHA-256 — 이 실행과 다른 것) — 최근 것부터 */
function otherScores(ctx, d) {
  if (!d.ptoks || !d.n) return [];
  const want = d.ptoks.concat(d.ids);
  const sha = (d.rec.server || {}).sha256;
  const out = [];
  for (const e of runsOf(ctx)) {
    if (e.operation !== 'run.score') continue;
    const c = recordOf(ctx, e);
    if (!c.rec) continue;
    const req = c.rec.request && typeof c.rec.request === 'object' ? c.rec.request : {};
    const gq = req.geulpu && typeof req.geulpu === 'object' ? req.geulpu : {};
    const osha = (c.rec.server || {}).sha256;
    if (!osha || osha === sha || !isIds(req.prompt) || gq.echo_from !== d.ptoks.length || !sameIds(req.prompt, want)) continue;
    const lp = c.rec.response && Array.isArray(c.rec.response.choices) && c.rec.response.choices[0] && c.rec.response.choices[0].logprobs;
    if (!lp || !Array.isArray(lp.token_logprobs)) continue;
    const m = modelBySha(osha);
    out.push({ entry: e, sha: osha, name: m ? (m.name || m.id) + (quantLabel(m) ? ' · ' + quantLabel(m) : '') : String(osha).slice(0, 12), lp: lp.token_logprobs.slice(d.ptoks.length, d.ptoks.length + d.n) });
  }
  return out.reverse();
}

async function scoreWithModel(ctx, d, id) {
  rs.busy = T('다른 모델을 불러오는 중…');
  rs.err = null;
  ctx.rerender();
  try {
    await waitLoaded(id, expPath(ctx));
  } catch (e) {
    rs.err = e.message;
    rs.busy = null;
    ctx.rerender();
    return;
  }
  rs.busy = null;
  await measure(ctx, 'run.score', scoreAllBody(d, 2), T`${d.entry.id} 의 토큰을 다른 모델로 점수 매기는 중`, d.entry.id);
}

// ---- 토큰마다의 Δ(다른 모델 · 다른 프롬프트) ----

/** own[k] · other[k] 의 Δ(other − own) — 둘 다 있는 스텝만 */
function deltas(own, other) {
  const out = [];
  for (let k = 0; k < Math.min(own.length, other.length); k++) {
    if (typeof own[k] === 'number' && typeof other[k] === 'number') out.push({ k, own: own[k], other: other[k], delta: other[k] - own[k] });
  }
  return out;
}

function deltaView(ctx, d, rows, title, file, otherLabel) {
  if (!rows.length) return null;
  let sum = 0;
  for (const r of rows) sum += r.delta;
  const big = [...rows].sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 5);
  const maxAbs = Math.max(1e-9, ...rows.map(r => Math.abs(r.delta)));
  const textOf = k => (d.texts ? d.texts[k] ?? '' : '#' + d.ids[k]);
  // 끝 토큰(EOS) — 응답에 글이 없어 그 토큰의 글(<|im_end|> 등)로. 가장 큰 차이의 이름은 한 글자의 이어지는 바이트도 토큰 번호로 보인다
  const isEos = k => !!d.texts && d.texts[k] === '' && k === d.n - 1;
  const shown = k => (isEos(k) ? pieceText(d.ids[k]) ?? '' : textOf(k));
  const nameOf = k => (isEos(k) ? pieceText(d.ids[k]) ?? '#' + d.ids[k] : d.texts && d.texts[k] === null ? '#' + d.ids[k] : textOf(k));
  if (rows.some(r => isEos(r.k) && pieceText(d.ids[r.k]) === null)) fetchPieces([d.ids[d.n - 1]], rs.ctx);
  const strip = el('div', { class: 'run-strip rs-dstrip', role: 'img', 'aria-label': title },
    rows.map(r => el('span', {
      class: 'run-tok rs-d ' + (r.delta < 0 ? 'neg' : 'pos') + (isEos(r.k) ? ' eos' : ''), title: T`스텝 ${d.off + r.k}: Δ ${fx(r.delta, 4)}`,
      style: `--a:${(Math.abs(r.delta) / maxAbs).toFixed(3)}`,
    }, shown(r.k))));
  const csv = () => ['step,id,text,logprob,' + otherLabel + ',delta']
    .concat(rows.map(r => [r.k + 1, d.ids[r.k], JSON.stringify(textOf(r.k)), r.own, r.other, r.delta].join(','))).join('\n') + '\n';
  return el('div', { class: 'rs-delta-box' },
    el('h3', null, title),
    el('p', { class: 'st-meta' }, T`Δ 의 합 ${fx(sum, 4)} · 평균 ${fx(sum / rows.length, 5)} · 스텝 ${rows.length} 개`,
      ' · ', T('가장 큰 차이: '), big.map((r, i) => [i ? ', ' : '', T`스텝 ${d.off + r.k} ${quoteTok(nameOf(r.k))} ${fx(r.delta, 3)}`])),
    strip,
    el('div', { class: 'xp-actions' }, el('button', { type: 'button', class: 'btn sm', onclick: () => download(file, csv(), 'text/csv') }, 'CSV')));
}

// ---- 내보내기 ----

function download(name, text, type) {
  const a = el('a', { href: URL.createObjectURL(new Blob([text], { type })), download: name });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}
const stamp = () => new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '');
const expId = ctx => (ctx.exp && (ctx.exp.id || (ctx.exp.experiment && ctx.exp.experiment.id))) || String((ctx.exp && ctx.exp.path) || 'run').split(/[\\/]/).pop();
const baseName = (ctx, d) => `geulpu-${String(expId(ctx)).replace(/[^\w.-]+/g, '_')}-${d.entry.id}`;

/** 토큰 표의 줄 — 답 전체의 점수 실행이 있으면 그 값(끝 토큰 스텝까지 · 상위 후보가 더 많다), 없으면 실행의 기록 */
function tokenRows(ctx, d) {
  const f = fullScores(ctx, d).find(x => x.sameModel) || null;
  const rows = [];
  for (let k = 0; k < d.n; k++) {
    const v = stepVals(d, k);
    const ids = f ? f.topIds[k] || [] : (d.topIds && d.topIds[k]) || [];
    const lps = f ? f.topLp[k] || [] : (d.topLp && d.topLp[k]) || [];
    const lp = f && typeof f.lp[k] === 'number' ? f.lp[k] : v.lp;
    const ent = f && typeof f.ent[k] === 'number' ? f.ent[k] : v.ent;
    const r = ids.indexOf(d.ids[k]);
    rows.push({
      step: k + 1, id: d.ids[k], text: d.raw ? d.raw[k] : d.texts ? d.texts[k] : null, logprob: lp, prob: lp === null ? null : Math.exp(lp),
      rank: r >= 0 ? r + 1 : null, entropy: ent, margin: lps.length > 1 ? lps[0] - lps[1] : null, logit_hash: d.hashes ? d.hashes[k] || null : null,
      top: ids.map((id, q) => ({ id, logprob: lps[q] })),
    });
  }
  return { rows, from: f ? f.entry.id : null, K: f ? f.K : (d.topIds && d.topIds[0] ? d.topIds[0].length : 0) };
}
const CSV_HEAD = 'step,id,text,logprob,prob,rank,entropy,margin,logit_hash';
const csvOf = rows => [CSV_HEAD].concat(rows.map(r => [r.step, r.id, JSON.stringify(r.text ?? ''), r.logprob ?? '', r.prob ?? '', r.rank ?? '',
  r.entropy ?? '', r.margin ?? '', r.logit_hash || ''].join(','))).join('\n') + '\n';

function runMeta(ctx, d) {
  const s = d.rec.server || {};
  const m = modelBySha(s.sha256);
  const f = rs.fp.get(ctx.path + '|' + d.entry.id);
  return {
    experiment: expId(ctx), run: d.entry.id, operation: d.rec.op,
    model: { name: m ? m.name || m.id : d.rec.model || '', sha256: s.sha256 || '' },
    params: { temperature: d.req.temperature ?? null, max_tokens: d.req.max_tokens ?? null, seed: d.seedText || null, step_offset: d.off || 0 },
    prompt_tokens: d.ptoks, fingerprint: f ? { token: f.fp, logit: f.fpLogit } : null,
    made_on: { layer: s.layer || '', precision: s.precision || 'exact', kv8: !!s.kv8, fast_attention: !!s.fast_attention, yarn: s.yarn || 0 },
  };
}

function exportTokens(ctx, d, kind) {
  const t = tokenRows(ctx, d);
  const name = `${baseName(ctx, d)}-tokens-${stamp()}`;
  if (kind === 'csv') return download(name + '.csv', csvOf(t.rows), 'text/csv');
  const head = {
    format: 'geulpu-tokens/1', contract: '5.2', what: 'answer', logprob: 'natural log, model distribution (temperature 1), contract 5.2',
    top_from: t.from ? `run ${t.from} (top ${t.K})` : `run ${d.entry.id} (top ${t.K})`,
  };
  download(name + '.json', JSON.stringify(Object.assign(head, runMeta(ctx, d), { steps: t.rows })) + '\n', 'application/json');
}

/** 재현 파일(geulpu-capsule/1 — 도구/repro_check.py 가 읽는다) — 앱의 experiment.repro 가 만든다(F3-4 — CLI 의 experiment repro 와 같은 파일).
   답 전체 점수 실행이 없으면 먼저 점수를 매긴다(상위 후보 10) */
async function exportCapsule(ctx, d) {
  if (!fullScores(ctx, d).some(x => x.sameModel)) {
    const id = await scoreAll(ctx, d, CAPSULE_K);
    if (!id) return;
  }
  let r = null;
  for (let i = 0; i < 20; i++) {                      // 점수 실행이 실험 기록에 들어갈 때까지(그 전에는 409)
    r = await call('experiment.repro', { path: ctx.path, run: d.entry.id });
    if (r.ok || r.status !== 409) break;
    await sleep(250);
  }
  if (!r || !r.ok || !r.json) {
    rs.err = appError(r);
    ctx.rerender();
    return;
  }
  const fp = String((r.json.fingerprint && r.json.fingerprint.token) || '');
  download(`geulpu-repro-${fp.slice(0, 12)}.json`, r.text, 'application/json');
}

function pythonCode(ctx, d) {
  const port = (rs.state && rs.state.app && rs.state.app.port) || 44544;
  const pre = `from openai import OpenAI   # pip install openai — ${T('앱에서 모델을 불러온 상태에서')}\nc = OpenAI(base_url="http://127.0.0.1:${port}/v1", api_key="geulpu")\n`;
  const score = d.ptoks ? pre
    + `prompt = ${JSON.stringify(d.ptoks)}\nanswer = ${JSON.stringify(d.ids)}\n`
    + `r = c.completions.create(model="geulpu", prompt=prompt + answer, echo=True, max_tokens=0, logprobs=10, temperature=0,\n`
    + `                         extra_body={"geulpu": {"echo_from": len(prompt), "logit_hash": True}})\n`
    + `lp = r.choices[0].logprobs.token_logprobs[len(prompt):]   # ${T('이 실행의 값과 비트까지 같다')}\n` : null;
  const gen = pre
    + `r = c.chat.completions.create(model="geulpu", messages=[{"role": "user", "content": "${T('하늘은 왜 파란가?')}"}], max_tokens=128,\n`
    + `                              temperature=0, logprobs=True, top_logprobs=5, extra_body={"geulpu": {"logit_hash": True}})\n`
    + `for t in r.choices[0].logprobs.content: print(t.token, t.logprob)\n`;
  const code = (title, py) => el('div', { class: 'rs-code' },
    el('div', { class: 'row g8' }, el('span', { class: 'grow' }, title),
      el('button', { type: 'button', class: 'btn sm ghost', onclick: () => navigator.clipboard.writeText(py).catch(() => { }) }, T('복사'))),
    el('pre', { class: 'xp-pre' }, py));
  return el('details', { class: 'run-raw' }, el('summary', null, T('파이썬 코드')),
    el('p', { class: 'muted' }, T('이 화면의 값은 모두 OpenAI 호환 API로도 받습니다 — 같은 모델 파일 · 같은 계산 방식이면 같은 비트입니다.')),
    score ? code(T('이 실행을 점수 매기기(echo)'), score) : null, code(T('생성하며 로그확률 받기'), gen));
}

// ---- 프롬프트 캐시(기록의 geulpu.store — 연구 패널과 같은 줄) ----

const tokShow = s => (s === null || s === undefined ? '' : s === '' ? '∅' : String(s).replace(/\n/g, '↵').replace(/\t/g, '→'));
function storeLines(d) {
  const st = d.g && d.g.store;
  if (!st || typeof st !== 'object') return null;
  const n = d.ptoks ? d.ptoks.length : 0;
  const reuse = (st.slot_cached_tokens || 0) + (st.slot_copy_tokens || 0) + (st.restored_tokens || 0);
  const lines = [T`다시 쓴 프롬프트 ${reuse}/${n || reuse} 토큰 — 이 슬롯 ${st.slot_cached_tokens || 0} · 다른 슬롯 ${st.slot_copy_tokens || 0} · 프롬프트 캐시 ${st.restored_tokens || 0}`];
  const at = st.reuse_end || 0;
  if (st.break === 'diverged') lines.push(T`캐시가 끊긴 위치: 토큰 ${at} — 앞 내용과 달라짐` + ` [${tokShow(st.break_prev_text)} → ${tokShow(st.break_next_text)}]`);
  else if (st.break === 'store_end') lines.push(T`캐시가 끊긴 위치: 토큰 ${at} — 프롬프트 캐시에 있던 데까지`);
  else if (st.break === 'new') lines.push(T('같은 앞부분이 없어 처음부터 계산'));
  else if (st.break === 'extended') lines.push(T('앞 내용을 끝까지 다시 씀'));
  else if (st.break === 'full') lines.push(T('프롬프트를 끝까지 다시 씀'));
  else if (st.break === 'off') lines.push(T('캐시 없이 계산'));
  return lines.map(x => el('div', { class: 'small muted' }, x));
}

// ---- 실행 화면의 도구 줄(토큰별 출력 아래) ----

export function toolsView(ctx, d, text, busy) {
  rs.ctx = ctx;
  const out = [];
  const can = !busy && !!d.ptoks;
  // 계산
  const full = fullScores(ctx, d).find(x => x.sameModel);
  const kIn = el('input', {
    id: 'rs-topk', class: 'input rs-k', type: 'number', min: '2', max: String(TOP_MAX), step: '1', value: String(rs.topK),
    oninput: ev => { rs.topK = ev.target.value; },
  });
  const others = otherModels(d);
  if (rs.otherModel && !others.some(m => m.id === rs.otherModel)) rs.otherModel = '';
  const pickOther = el('select', { class: 'input', 'aria-label': T('점수 매길 다른 모델'), onchange: ev => { rs.otherModel = ev.target.value; } },
    el('option', { value: '' }, T('다른 모델 고르기')),
    others.map(m => el('option', { value: m.id, selected: m.id === rs.otherModel ? true : null }, `${m.name || m.id}${quantLabel(m) ? ' · ' + quantLabel(m) : ''}`)));
  out.push(el('section', { class: 'rs-sec', 'aria-label': T('계산') }, el('h2', null, T('계산')),
    el('div', { class: 'xp-actions st-form' },
      el('label', { class: 'st-inline', for: 'rs-topk' }, T('모든 스텝의 상위 후보')), kIn,
      el('button', {
        type: 'button', class: 'btn', disabled: can ? null : true,
        onclick: () => { const K = Math.max(2, Math.min(TOP_MAX, Math.round(Number(rs.topK)) || TOP_DEFAULT)); rs.topK = K; scoreAll(ctx, d, K); },
      }, T('계산', '단추')),
      el('span', { class: 'muted' }, full ? T`실행 ${full.entry.id} 의 상위 ${full.K} 개 — 스텝 상세와 스텝 · 층 화면이 그 후보를 보입니다.`
        : T('답 전체를 echo 로 점수 매겨 스텝마다의 상위 후보를 이 실험의 실행으로 남깁니다.'))),
    el('div', { class: 'xp-actions' },
      el('button', { type: 'button', class: 'btn', disabled: busy || !rawRequest(text) ? true : null, onclick: () => rerunNoCache(ctx, d, text) }, T('캐시 없이 재실행')),
      el('span', { class: 'muted' }, T('같은 요청을 프롬프트 캐시 없이(cache_prompt: false) 다시 보내 토큰 · 로짓 해시를 비교합니다.'))),
    others.length ? el('div', { class: 'xp-actions' }, pickOther,
      el('button', {
        type: 'button', class: 'btn', disabled: can ? null : true,
        onclick: () => { if (rs.otherModel) scoreWithModel(ctx, d, rs.otherModel); },
      }, T('불러와 점수 매기기')),
      el('span', { class: 'muted' }, T('같은 토큰을 어휘 · 채팅 템플릿이 같은 다른 모델(예: 다른 양자화)로 점수 매깁니다 — 그 모델을 불러옵니다.')))
      : el('p', { class: 'muted' }, T('어휘와 채팅 템플릿이 같은 다른 모델이 설치되어 있지 않습니다.'))));
  // 다른 모델의 Δ
  const own = full ? full.lp : d.lps || [];
  for (const o of otherScores(ctx, d).slice(0, 2)) {
    const v = deltaView(ctx, d, deltas(own, o.lp), T`다른 모델로 점수 — ${o.name} (실행 ${o.entry.id}) · Δ = 그 모델 − 이 실행`,
      `${baseName(ctx, d)}-vs-model-${stamp()}.csv`, 'other_logprob');
    if (v) out.push(v);
  }
  // 프롬프트 캐시
  const sl = storeLines(d);
  if (sl) out.push(el('section', { class: 'rs-sec', 'aria-label': T('프롬프트 캐시') }, el('h2', null, T('프롬프트 캐시')), sl));
  // 내보내기
  out.push(el('section', { class: 'rs-sec', 'aria-label': T('내보내기') }, el('h2', null, T('내보내기')),
    el('div', { class: 'xp-actions' },
      el('span', null, T('토큰 표')),
      el('button', { type: 'button', class: 'btn sm', 'aria-label': T('토큰 표 JSON'), onclick: () => exportTokens(ctx, d, 'json') }, 'JSON'),
      el('button', { type: 'button', class: 'btn sm', 'aria-label': T('토큰 표 CSV'), onclick: () => exportTokens(ctx, d, 'csv') }, 'CSV'),
      // F3-1 — 재현 파일 단추는 이름에 '재현 파일'을(토큰 표의 'JSON' 과 가려지게)
      el('button', {
        type: 'button', class: 'btn sm', disabled: busy || !d.ptoks || !(d.rec.server || {}).sha256 ? true : null,
        title: T('요청 · 결과 · 만든 곳 — 재현 › 검사의 재현 파일 확인 · 도구/repro_check.py 가 읽는 파일'), onclick: () => exportCapsule(ctx, d),
      }, T('재현 파일(JSON)'))),
    el('p', { class: 'muted' }, full ? T`토큰 표 · 재현 파일의 스텝마다의 값은 점수 실행 ${full.entry.id} 에서(끝 토큰의 스텝까지).`
      : T('재현 파일은 끝 토큰의 스텝까지 값이 있어야 해서, 없으면 먼저 답 전체를 점수 매깁니다(상위 후보 10 — 그 점수 실행은 이 실험에 남습니다).')),
    // F3-5 — 다시 확인 세션: 받은 파일을 확인하는 곳을 내보내기 옆에서도
    el('p', { class: 'muted' }, T('받은 재현 파일을 확인하는 곳: 재현 › 검사 › 재현 파일 확인.')),
    pythonCode(ctx, d)));
  return out;
}

// ---- 비교 탭 ----

/** 위치 pos(프롬프트 + 답의 순서)의 토큰 글 — 답이면 스텝의 글, 프롬프트면 기록의 글. 없으면 /detokenize 로 읽어 둔 것(아직 없으면 null) */
function posText(d, pos, id) {
  const P = d.ptoks.length;
  if (pos >= P) return d.texts ? d.texts[pos - P] : pieceText(id);
  return d.ptexts && typeof d.ptexts[pos] === 'string' ? d.ptexts[pos] : pieceText(id);
}

/** 앱의 비교 값의 위치 하나(id · prob · rank) → 화면의 {id, text, prob, rank}. 없으면 null */
const posOf = (d, pos, x) => (x ? { id: x.id, text: posText(d, pos, x.id), prob: x.prob, rank: x.rank } : null);

/** 두 실행의 처음 다른 스텝 · 수치 표 · 다른 프롬프트로 점수(Δ). v 는 앱의 비교 값(experiment.compare, F3-3 — 받기 전에는 null) */
export function compareExtras(ctx, da, db, v) {
  rs.ctx = ctx;
  const out = [];
  // 처음 다른 스텝의 두 토큰 — 분기 실행에서는 그 토큰이 프롬프트(개입한 토큰)에 있으므로 앱이 echo 기록에서 확률 · 순위를 읽는다
  if (v && !v.same) {
    const f = v.first;
    // 확률 · 순위는 기록에 있을 때만(프롬프트 위치는 echo 기록일 때만 있다)
    const side = x => {
      const kids = [el('b', null, x.text === null ? '#' + x.id : quoteTok(x.text))];
      if (x.prob !== null || x.rank) kids.push(' ', T`확률 ${pct(x.prob)}`, x.rank ? T` · ${x.rank} 위` : '');
      return el('span', null, ...kids);
    };
    const line = (label, a, b, k) => {
      const miss = [a, b].filter(x => x.text === null).map(x => x.id);
      if (miss.length) fetchPieces(miss, ctx);
      return el('div', { class: 'xp-actions rs-div' },
        el('span', null, label, el('code', null, da.entry.id), ' ', side(a), ' · ', el('code', null, db.entry.id), ' ', side(b)),
        k >= 0 && k < da.n ? el('button', { type: 'button', class: 'btn sm', onclick: () => ctx.openStep(da.entry.id, k) }, T`${da.entry.id} 의 이 스텝 보기`) : null);
    };
    const ia = posOf(da, f.position, f.a);
    const ib = posOf(db, f.position, f.b);
    if (ia && ib) out.push(line(!f.in_prompt ? T`처음 다른 스텝 ${f.step}: ` : T`프롬프트가 처음 다른 위치 ${f.position}: `, ia, ib, f.position - da.ptoks.length));
    // 프롬프트가 다르면 답끼리도 — 답이 처음 다른 스텝과 그 스텝의 두 토큰(각 실행의 기록에서 확률 · 순위, F2)
    const fa = v.first_answer_step;
    if (fa) {
      if (fa.step === null) out.push(el('p', { class: 'muted' }, T`답의 토큰이 스텝 ${fa.from} ~ ${fa.to} 에서 모두 같습니다.`));
      else {
        const xa = posOf(da, da.ptoks.length + fa.step - da.off, fa.a);
        const xb = posOf(db, db.ptoks.length + fa.step - db.off, fa.b);
        if (xa && xb) out.push(line(T`답이 처음 다른 스텝 ${fa.step}: `, xa, xb, fa.step - da.off));
      }
    }
  }
  // 수치 표 — 앱의 요약(experiment.summary)
  const qa = summaryOfRun(ctx, da.entry, rs.delta);
  const qb = summaryOfRun(ctx, db.entry, rs.delta);
  out.push(el('h2', null, T('두 실행의 수치')));
  if (!qa.val || !qb.val) out.push(el('p', { class: 'muted' }, qa.error || qb.error || T('읽는 중…')));
  else {
    const row = (label, f) => el('tr', null, el('td', null, label), el('td', null, f(qa.val)), el('td', null, f(qb.val)));
    out.push(el('table', { class: 'xp-files run-diff rs-cmp' }, el('tbody', null,
      el('tr', null, el('th', null, ''), el('th', null, da.entry.id), el('th', null, db.entry.id)),
      row(T('토큰'), s => String(s.steps)),
      row(T('평균 로그확률'), s => fx(s.mean_logprob, 4)),
      row(T('퍼플렉서티'), s => fx(s.perplexity, 3)),
      row(T('마진 < δ 인 스텝'), s => `${s.low_margin_steps.length} / ${s.margin_steps}`),
      row(T('1위가 아닌 스텝'), s => `${s.not_first_steps.length} / ${s.ranked_steps}`))));
  }
  // 프롬프트가 다르면 — A 의 답을 B 의 프롬프트로
  if (da.ptoks && db.ptoks && !sameIds(da.ptoks, db.ptoks) && da.n) {
    const want = db.ptoks.concat(da.ids);
    let found = null;
    for (const e of runsOf(ctx)) {
      if (e.operation !== 'run.score') continue;
      const cr = recordOf(ctx, e);
      if (!cr.rec) continue;
      const req = cr.rec.request && typeof cr.rec.request === 'object' ? cr.rec.request : {};
      const gq = req.geulpu && typeof req.geulpu === 'object' ? req.geulpu : {};
      if (!isIds(req.prompt) || gq.echo_from !== db.ptoks.length || !sameIds(req.prompt, want)) continue;
      const lp = cr.rec.response && Array.isArray(cr.rec.response.choices) && cr.rec.response.choices[0] && cr.rec.response.choices[0].logprobs;
      if (lp && Array.isArray(lp.token_logprobs)) found = { entry: e, lp: lp.token_logprobs.slice(db.ptoks.length, db.ptoks.length + da.n) };
    }
    out.push(el('h2', null, T`${da.entry.id} 의 답을 ${db.entry.id} 의 프롬프트로`));
    if (found) {
      const own = (fullScores(ctx, da).find(x => x.sameModel) || {}).lp || da.lps || [];
      const v = deltaView(ctx, da, deltas(own, found.lp), T`점수 실행 ${found.entry.id} · Δ = ${db.entry.id} 의 프롬프트 − ${da.entry.id}`,
        `geulpu-${da.entry.id}-under-${db.entry.id}-${stamp()}.csv`, 'other_logprob');
      if (v) out.push(v);
    } else {
      const body = JSON.stringify({
        prompt: want, echo: true, max_tokens: 0, logprobs: 1, geulpu: { echo_from: db.ptoks.length, logit_hash: true },
      });
      out.push(el('div', { class: 'xp-actions' },
        el('button', {
          type: 'button', class: 'btn', onclick: () => measure(ctx, 'run.score', body, T`${da.entry.id} 의 답을 ${db.entry.id} 의 프롬프트로 점수 매기는 중`, da.entry.id),
        }, T('점수 매기기')),
        el('span', { class: 'muted' }, T('두 프롬프트에서 같은 답 토큰의 로그확률을 비교합니다 — 점수는 이 실험의 실행으로 남습니다.'))));
    }
  }
  return out;
}
