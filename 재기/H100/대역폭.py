"""GPU 메모리 대역폭(복사: 읽기 + 쓰기)과 장치 정보 — torch 로(vllm.sh 의 가상환경). python 재기/H100/대역폭.py results.jsonl"""
import json
import sys
import time

import torch

x = torch.empty(2 * 1024 ** 3, dtype=torch.uint8, device="cuda")
y = torch.empty_like(x)
for _ in range(5):
    y.copy_(x)
torch.cuda.synchronize()
n = 30
t0 = time.perf_counter()
for _ in range(n):
    y.copy_(x)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / n
p = torch.cuda.get_device_properties(0)
rec = {"tag": "bandwidth", "device": torch.cuda.get_device_name(0), "sm": p.multi_processor_count, "mem_GB": p.total_memory / 1e9,
       "copy_bytes": 2 * x.numel(), "copy_GBps": 2 * x.numel() / dt / 1e9, "torch": torch.__version__, "cuda": torch.version.cuda,
       "at": time.strftime("%Y-%m-%d %H:%M:%S")}
print(json.dumps(rec), flush=True)
with open(sys.argv[1], "a", encoding="utf-8") as w:
    w.write(json.dumps(rec) + "\n")
