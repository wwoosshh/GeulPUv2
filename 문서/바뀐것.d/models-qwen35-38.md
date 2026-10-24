갈래: 고침

- **MTP 층이 든 Qwen3.5 · 3.8 GGUF 의 층 수**: 파일 끝의 MTP(다음 토큰 예측) 층(`nextn_predict_layers`)을 본 계산의 층으로 읽으려다 "텐서가 없다"로
  멈추던 것을 고쳤다 — MTP 층은 llama.cpp 처럼 건너뛴다. 층마다의 종류 표(`attention.recurrent_layers`)가 있으면 그것과 맞는지도 본다.

---

- **Layer count of Qwen3.5 · 3.8 GGUFs with an MTP layer.** The engine no longer stops with "tensor not found" on files that carry a
  multi-token-prediction layer at the end (`nextn_predict_layers`); like llama.cpp it skips that layer. When the file lists the layer
  kinds (`attention.recurrent_layers`), the engine checks they match.
