"""Separate Q4 FastFlowLM NPU reference; not an exact-contract comparison."""
import ctypes
import hashlib
import json
import os
import socket
import subprocess
import time
import traceback
import urllib.request
from native_post import HERE,ROOT
from system_metrics import Resources,kernel

def main():
    report=dict(complete=False,scope='FastFlowLM Qwen3 0.6B Q4_1; distinct weights/quantization and no Geul bit contract',runs=[])
    port=44567
    server=None
    handle=None
    log=None
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        with socket.socket() as s:s.bind(('127.0.0.1',port))
        env=os.environ.copy()
        env['FLM_MODEL_PATH']=str(ROOT/'빌드/npu/models')
        exe=ROOT/'빌드/npu/flm-1.0.10/flm.exe'
        log=(HERE/'results/native-npu-server.log').open('w',encoding='utf8')
        server=subprocess.Popen([str(exe),'serve','qwen3:0.6b','--host','127.0.0.1','--port',str(port),'--ctx-len','1024','--pmode','performance','--cors','0'],env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        base=f'http://127.0.0.1:{port}'
        for _ in range(90):
            if server.poll() is not None:raise RuntimeError('FLM server exited; see log')
            try:
                with urllib.request.urlopen(base+'/v1/models',timeout=2) as response:response.read()
                break
            except Exception:time.sleep(1)
        else:raise RuntimeError('FLM readiness timeout')
        handle=kernel.OpenProcess(0x410,False,server.pid)
        if not handle:raise ctypes.WinError(ctypes.get_last_error())
        def request(prompt,max_tokens):
            body=dict(model='qwen3:0.6b',messages=[dict(role='user',content=prompt)],max_tokens=max_tokens,stream=True,
                stream_options=dict(include_usage=True),temperature=0,top_k=1,top_p=1,think=False)
            req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            first=None
            text=''
            usage=None
            finish=None
            done=False
            start=time.perf_counter()
            with urllib.request.urlopen(req,timeout=180) as response:
                for line in response:
                    line=line.decode('utf8').strip()
                    if not line.startswith('data:'):continue
                    payload=line[5:].strip()
                    if payload=='[DONE]':done=True;break
                    event=json.loads(payload)
                    if event.get('usage'):usage=event['usage']
                    for choice in event.get('choices',[]):
                        delta=choice.get('delta',{})
                        value=(delta.get('content') or '')+(delta.get('reasoning_content') or '')
                        if value:
                            if first is None:first=time.perf_counter()-start
                            text+=value
                        if choice.get('finish_reason'):finish=choice['finish_reason']
            assert done and finish and first is not None
            return dict(first_visible_seconds=first,total_seconds=time.perf_counter()-start,usage=usage,finish=finish,
                output_hash=hashlib.sha256(text.encode()).hexdigest(),output=text)
        request('Say hello. /no_think',8)
        prompts=[f'Explain why exact arithmetic matters in matrix multiplication. Example number {i}. Provide a detailed explanation. /no_think' for i in (1,2)]
        for prompt in prompts:
            with Resources(handle) as resources:result=request(prompt,64)
            result['resources']=resources.result
            result['prompt']=prompt
            report['runs'].append(result)
            print('native NPU request',round(result['total_seconds'],3),'seconds',result['usage'],flush=True)
        report['complete']=True
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        if handle:kernel.CloseHandle(handle)
        if server is not None and server.poll() is None:
            server.terminate()
            server.wait(timeout=20)
        if log:log.close()
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        (HERE/'results/native-npu-system.json').write_text(json.dumps(report,indent=2),encoding='utf8',newline='\n')

if __name__=='__main__':main()
