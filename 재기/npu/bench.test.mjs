import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { measure, stats, summarize } from './bench.mjs';

test('SSE network chunks are not token counts; UTF-8, reasoning, usage and finish survive split bytes', async () => {
  const events=[{choices:[{delta:{role:'assistant'}}]}, {choices:[{delta:{reasoning_content:'생각'}}]},
    {choices:[{delta:{content:'한글 답변'}}]}, {choices:[{delta:{},finish_reason:'length'}]},
    {choices:[],usage:{prompt_tokens:100,completion_tokens:9}}];
  const server=createServer((req,res)=>{res.writeHead(200,{'content-type':'text/event-stream'});
    const bytes=Buffer.from(events.map(e=>'data: '+JSON.stringify(e)+'\r\n\r\n').join('')+'data: [DONE]\r\n\r\n');
    for(let i=0;i<bytes.length;i+=7) res.write(bytes.subarray(i,i+7)); res.end(); });
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  try { const r=await measure(`http://127.0.0.1:${server.address().port}`,{});
    assert.equal(r.text,'한글 답변'); assert.equal(r.reasoning,'생각'); assert.equal(r.usage.completion_tokens,9);
    assert.equal(r.events.length,5); assert.equal(r.finish_reason,'length'); assert.ok(r.first_output_ms>=0);
  } finally { await new Promise(r=>server.close(r)); }
});
test('truncated streams fail instead of becoming successful measurements',async()=>{
  const server=createServer((req,res)=>{res.writeHead(200,{'content-type':'text/event-stream'});res.end('data: {"choices":[]}\n\n');});
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  try {await assert.rejects(measure(`http://127.0.0.1:${server.address().port}`,{}),/Incomplete/);}
  finally {await new Promise(r=>server.close(r));}
});
test('missing metrics are not zero',()=>{assert.equal(stats([null,undefined]),null);assert.equal(stats([1,4,2,3]).median,2.5);});
test('summary excludes warmup and never equates text equality with logit equality',()=>{
  const result={first_output_ms:10,total_ms:20,output_sha256:'same',usage:{completion_tokens:2,decoding_speed_tps:100}};
  const s=summarize({cases:[{id:'x'}]},[{case:'x',warmup:true,result:{...result,total_ms:999}},
    {case:'x',result},{case:'x',result},{case:'x',error:'failed'}]).cases[0];
  assert.equal(s.attempted,3); assert.equal(s.success,2); assert.equal(s.total_ms.median,20);
  assert.equal(s.native_decode_tps.median,100); assert.equal(s.repeat_text_equal,true);
  assert.equal(s.logit_bit_equality,null); assert.equal(s.energy_joules,null);
});
