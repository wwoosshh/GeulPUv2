// 앱리디자인 E1(문서/화면구현.md §1) — 기능 목록 클라이언트. 기능 목록(GET /api/operations)을 읽고, 기능 id 와 인자로 요청을 만들어 앱에
// 보낸다. 새 화면 코드는 앱을 이것으로만 부른다(옛 app.js · eval.js 의 호출은 단계 E5 · E6 에서 옮긴다).
import { index, requestFor } from './command.js';

let registryPromise = null;

/** 기능 목록(한 번 읽어 둔다 — 실패하면 다음에 다시 읽는다). */
export function registry() {
  if (!registryPromise) {
    registryPromise = fetch('/api/operations', { headers: { 'X-GeulPU-Source': 'gui' } })
      .then(r => {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(index)
      .catch(e => {
        registryPromise = null;
        throw e;
      });
  }
  return registryPromise;
}

/**
 * 만든 요청({method, path, body})을 보낸다. source 는 출처(gui · cli), session 은 세션 이름, experiment 는 실험 폴더(명령 기록을 남길 때).
 * 답은 {status, ok, json, text} — 앱의 오류 답도 그대로 돌려준다(던지지 않는다). 앱에 닿지 못하면 던진다.
 */
export async function send(req, { source = 'gui', session = '', experiment = '', signal } = {}) {
  const headers = { 'X-GeulPU-Source': source };
  if (session) headers['X-GeulPU-Session'] = session;
  if (experiment) headers['X-GeulPU-Experiment'] = experiment;
  const opt = { method: req.method, headers, signal };
  if (req.method === 'POST') {
    headers['Content-Type'] = 'application/json';
    opt.body = req.body == null ? '{}' : req.body;
  }
  const r = await fetch(req.path, opt);
  const text = await r.text();
  let json = null;
  try { json = text ? JSON.parse(text) : null; } catch (_) { json = null; }
  return { status: r.status, ok: r.ok, json, text };
}

/** 기능 하나를 부른다 — POST 는 params 를 본문으로, GET 은 쿼리 문자열로. */
export async function call(opId, params = {}, opt = {}) {
  const reg = await registry();
  return send(requestFor(reg, opId, params), opt);
}
