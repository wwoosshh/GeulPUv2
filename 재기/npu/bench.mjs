import { createHash } from 'node:crypto';
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

export const hash = s => createHash('sha256').update(s).digest('hex');
export function parseEvent(frame) {
  const data = frame.split('\n').filter(s => s.startsWith('data:')).map(s => s.slice(5).trimStart()).join('\n');
  return !data ? null : data === '[DONE]' ? { done: true } : JSON.parse(data);
}
export async function measure(url, request, timeoutMs = 180000) {
  const start = performance.now();
  const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request), signal: AbortSignal.timeout(timeoutMs) });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${(await response.text()).slice(0, 1000)}`);
  if (!response.headers.get('content-type')?.includes('text/event-stream')) throw new Error('Expected SSE response');
  let pending = '', text = '', reasoning = '', first = null, last = null, done = false, usage = null, finish = null;
  const events = [], decoder = new TextDecoder();
  const consume = frame => {
    const e = parseEvent(frame);
    if (!e) return;
    if (e.done) { done = true; return; }
    if (e.error) throw new Error(JSON.stringify(e.error));
    const ms = performance.now() - start;
    events.push({ ms, event: e });
    if (e.usage) usage = e.usage;
    const c = e.choices?.[0], d = c?.delta;
    if (c?.finish_reason) finish = c.finish_reason;
    if (d?.content || d?.reasoning_content) {
      first ??= ms; last = ms;
      text += d.content ?? ''; reasoning += d.reasoning_content ?? '';
    }
  };
  for await (const bytes of response.body) {
    pending += decoder.decode(bytes, { stream: true });
    // Normalize only complete CRLF pairs; a trailing CR can span network chunks.
    pending = pending.replace(/\r\n/g, '\n');
    let end;
    while ((end = pending.indexOf('\n\n')) >= 0) { consume(pending.slice(0, end)); pending = pending.slice(end + 2); }
  }
  pending += decoder.decode();
  if (pending.trim()) consume(pending);
  if (!done || !finish) throw new Error('Incomplete SSE stream (missing DONE or finish_reason)');
  const total = performance.now() - start;
  const count = usage?.completion_tokens;
  return { first_output_ms: first, last_output_ms: last, total_ms: total, usage, finish_reason: finish,
    output_tokens_per_total_second: Number.isFinite(count) && count > 0 ? count * 1000 / total : null,
    text, reasoning, output_sha256: hash(JSON.stringify({ text, reasoning, finish })), events };
}
export function stats(values) {
  const v = values.filter(Number.isFinite).sort((a,b) => a-b);
  if (!v.length) return null;
  const mid = Math.floor(v.length / 2);
  return { n: v.length, median: v.length % 2 ? v[mid] : (v[mid-1]+v[mid])/2,
    p95: v[Math.ceil(v.length*.95)-1], min: v[0], max: v.at(-1) };
}
async function main() {
  const args = process.argv.slice(2);
  const get = (key, fallback) => { const i = args.indexOf(key); return i < 0 ? fallback : args[i+1]; };
  if (args.includes('--summarize')) {
    const logPath = get('--summarize');
    const [header, ...rows] = readFileSync(logPath,'utf8').trim().split('\n').map(JSON.parse);
    console.log(JSON.stringify(summarize(header, rows),null,2)); return;
  }
  if (args.includes('--help')) { console.log('node bench.mjs --url http://127.0.0.1:44560 --model qwen3:0.6b --out DIR [--repeat 5] [--metadata FILE]'); return; }
  const base = get('--url', 'http://127.0.0.1:44560');
  const u = new URL(base);
  if (!['127.0.0.1','localhost','[::1]'].includes(u.hostname)) throw new Error('Local endpoints only');
  const model = get('--model', 'qwen3:0.6b'), repeats = Number(get('--repeat', '5'));
  if (!Number.isInteger(repeats) || repeats < 1) throw new Error('repeat must be a positive integer');
  const out = resolve(get('--out', `빌드/npu/results/${Date.now()}`));
  mkdirSync(out, { recursive: true });
  const log = resolve(out, 'requests.jsonl');
  // Refuse to append a different experiment to an existing run.
  writeFileSync(log, '', { flag: 'wx' });
  const metadataPath = get('--metadata', null);
  const optionsPath = get('--request-options', null);
  const requestOptions = optionsPath ? JSON.parse(readFileSync(optionsPath,'utf8').replace(/^\uFEFF/,'')) : {};
  const cases = [8, 64, 256].map((n,i) => ({ id: ['short','medium','long'][i], prompt:
    '다음 자료를 읽고 기술적 의미와 주의점을 한국어로 자세히 설명하세요. /no_think\n' +
    Array.from({length:n}, (_,j) => `항목 ${j+1}: 정수 행렬곱의 부분합을 계산한 뒤 블록 척도를 적용한다. GPU와 NPU의 실행 시간과 정확성을 비교한다.`).join('\n') }));
  const header = { format: 'geulpu-npu-benchmark/1', at: new Date().toISOString(), model, base, repeats,
    node: process.version, platform: process.platform, request_options:requestOptions,
    metadata: metadataPath ? JSON.parse(readFileSync(metadataPath,'utf8').replace(/^\uFEFF/,'')) : {},
    scope: 'HTTP end-to-end; first visible/reasoning output is not exact first model token; no NPU-only power measurement',
    cache_policy: 'uncontrolled; repeated prompts may reuse cache; never interpret as uncached prefill throughput',
    cases: cases.map(c=>({id:c.id,prompt_sha256:hash(c.prompt),characters:c.prompt.length})) };
  appendFileSync(log, JSON.stringify(header)+'\n');
  const rows = [];
  for (let round = -1; round < repeats; round++) {
    // Rotate case order to reduce a fixed order/thermal confound.
    for (let index = 0; index < cases.length; index++) {
      const c = cases[(index + Math.max(0,round)) % cases.length];
      const request = { temperature:0, seed:7, ...requestOptions, model, messages: [{role:'user',content:c.prompt}],
        max_tokens:128, stream:true, stream_options:{include_usage:true} };
      const row = { type: 'request', case:c.id, round, warmup:round<0, at:new Date().toISOString(), request };
      try { row.result = await measure(new URL('/v1/chat/completions',base), request); }
      catch(e) { row.error = String(e); }
      appendFileSync(log,JSON.stringify(row)+'\n'); rows.push(row);
      console.log(JSON.stringify({case:c.id,round,ms:row.result?.total_ms,usage:row.result?.usage,error:row.error}));
    }
  }
  writeFileSync(resolve(out,'summary.json'),JSON.stringify(summarize(header,rows),null,2)+'\n');
  if (rows.some(r=>r.error)) process.exitCode=1;
}
export function summarize(header, rows) {
  return { ...header, cases: header.cases.map(c => {
    const all = rows.filter(r=>!r.warmup && r.case===c.id), ok=all.filter(r=>r.result);
    return { id:c.id, attempted:all.length, success:ok.length, errors:all.filter(r=>r.error).map(r=>r.error),
      first_output_ms:stats(ok.map(r=>r.result.first_output_ms)), total_ms:stats(ok.map(r=>r.result.total_ms)),
      output_tokens_per_total_second:stats(ok.map(r=>r.result.output_tokens_per_total_second)),
      native_prefill_tps:stats(ok.map(r=>r.result.usage?.prefill_speed_tps)),
      native_decode_tps:stats(ok.map(r=>r.result.usage?.decoding_speed_tps)),
      native_prefill_seconds:stats(ok.map(r=>r.result.usage?.prefill_duration_ttft)),
      native_decode_seconds:stats(ok.map(r=>r.result.usage?.decoding_duration)),
      cached_tokens:ok.map(r=>r.result.usage?.prompt_tokens_details?.cached_tokens ?? null),
      prompt_tokens:[...new Set(ok.map(r=>r.result.usage?.prompt_tokens ?? null))],
      completion_tokens:ok.map(r=>r.result.usage?.completion_tokens ?? null),
      distinct_outputs:new Set(ok.map(r=>r.result.output_sha256)).size,
      repeat_text_equal:ok.length > 1 ? new Set(ok.map(r=>r.result.output_sha256)).size === 1 : null,
      token_id_equality:null, logit_bit_equality:null, energy_joules:null };
  }) };
}
if (process.argv[1] && import.meta.url===pathToFileURL(resolve(process.argv[1])).href) main().catch(e=>{console.error(e);process.exitCode=1;});
