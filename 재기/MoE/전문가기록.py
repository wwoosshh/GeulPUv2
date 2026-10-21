#!/usr/bin/env python3
"""MoE 재기 — llama.cpp(b11727) C API 로 토큰열을 넣으며 층마다 고른 전문가(ffn_moe_topk)를 기록한다(ctypes).

  python 전문가기록.py <gguf> <일감.json> <출력.npz> [--ub 2048] [--ctx 131072]
  llama.cpp(b11727)의 DLL 폴더는 환경 변수 GEULPU_LLAMA(기본 E:/geulpu-work/moe/llama-b11727 — 이 PC 의 것).

일감.json: {"spans": [{"tokens": [...], "kind": "prompt"|"output"|"reprocess", "req": i, "reset": bool}, ...]}
  reset 이 참이면 그 조각 앞에서 KV 를 비운다(처음부터 다시). 조각은 차례대로 이어 넣는다.
출력.npz: ids [토큰, 층, 고름] (uint16), kind [토큰] (0 prompt · 1 output · 2 reprocess), req [토큰], pos [토큰]

구조체 자리(b11727 의 llama.h · ggml.h 에서 셈 — 기본값으로 확인한다):
  llama_model_params: tensor_buft_overrides 8, n_gpu_layers 16
  llama_context_params: n_ctx 0 · n_batch 4 · n_ubatch 8 · n_seq_max 12, flash_attn_type 52, cb_eval 88, cb_eval_user_data 96,
                        type_k 104 · type_v 108
  ggml_tensor: type 0, ne 16, nb 48, name 256
전문가 가중치는 CPU(호스트)에, 나머지는 GPU 에(llama.cpp 의 -cmoe 와 같은 덮어쓰기). 고른 전문가는 셈의 값이 아니라 라우터의 고름이라
엔진의 셈 차이(일괄 · 생성)로 바뀌는 일은 드물다(동점 근처뿐).
"""
import ctypes
import json
import os
import re
import struct
import sys
import time

import numpy as np

DLL = os.environ.get("GEULPU_LLAMA", r"E:\geulpu-work\moe\llama-b11727")
c_p = ctypes.c_void_p
로그 = []


def 열기():
    os.add_dll_directory(DLL)
    os.environ["PATH"] = DLL + os.pathsep + os.environ.get("PATH", "")
    L = ctypes.CDLL(os.path.join(DLL, "llama.dll"))
    G = ctypes.CDLL(os.path.join(DLL, "ggml.dll"))
    B = ctypes.CDLL(os.path.join(DLL, "ggml-base.dll"))
    콜백 = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, c_p)(
        lambda lv, s, u: 로그.append(s.decode("utf-8", "replace") if s else ""))
    L.llama_log_set(콜백, None)
    G.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
    G.ggml_backend_load_all_from_path(DLL.encode("utf-8"))
    L.llama_backend_init()
    return L, G, B, 콜백


def 찾기(이름, *dlls):
    for d in dlls:
        try:
            return getattr(d, 이름)
        except AttributeError:
            pass
    raise AttributeError(이름)


def main():
    gguf, 일감파일, 출력 = sys.argv[1:4]
    ub = int(sys.argv[sys.argv.index("--ub") + 1]) if "--ub" in sys.argv else 2048
    n_ctx = int(sys.argv[sys.argv.index("--ctx") + 1]) if "--ctx" in sys.argv else 131072
    일감 = json.load(open(일감파일, encoding="utf-8"))
    L, G, B, 로그콜백 = 열기()

    # 전문가를 CPU 에 두는 덮어쓰기
    dev_by_type = 찾기("ggml_backend_dev_by_type", G, B)
    dev_by_type.restype, dev_by_type.argtypes = c_p, [ctypes.c_int]
    dev_buft = 찾기("ggml_backend_dev_buffer_type", G, B)
    dev_buft.restype, dev_buft.argtypes = c_p, [c_p]
    cpu_buft = dev_buft(dev_by_type(0))                     # GGML_BACKEND_DEVICE_TYPE_CPU
    assert cpu_buft, "CPU 버퍼 형식을 못 찾음"
    무늬 = ctypes.create_string_buffer(rb"\.ffn_(up|down|gate_up|gate)_(ch|)exps")
    덮어쓰기 = (ctypes.c_void_p * 4)(ctypes.cast(무늬, c_p).value, cpu_buft, None, None)

    for n in ("llama_model_default_params", "llama_context_default_params"):
        getattr(L, n).restype = c_p
        getattr(L, n).argtypes = [c_p]
    L.llama_model_load_from_file.restype = c_p
    L.llama_model_load_from_file.argtypes = [ctypes.c_char_p, c_p]
    L.llama_init_from_model.restype = c_p
    L.llama_init_from_model.argtypes = [c_p, c_p]
    L.llama_batch_init.restype = c_p
    L.llama_batch_init.argtypes = [c_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    L.llama_decode.restype = ctypes.c_int32
    L.llama_decode.argtypes = [c_p, c_p]
    L.llama_get_memory.restype = c_p
    L.llama_get_memory.argtypes = [c_p]
    L.llama_memory_clear.argtypes = [c_p, ctypes.c_bool]
    L.llama_free.argtypes = [c_p]
    L.llama_model_free.argtypes = [c_p]

    mp = ctypes.create_string_buffer(256)
    L.llama_model_default_params(mp)
    print("모델 구조체 n_gpu_layers 기본값", struct.unpack_from("<ii", mp.raw, 16))
    struct.pack_into("<i", mp, 16, 99)
    struct.pack_into("<Q", mp, 8, ctypes.addressof(덮어쓰기))
    t0 = time.time()
    model = L.llama_model_load_from_file(gguf.encode("utf-8"), mp)
    assert model, "모델을 못 읽음\n" + "".join(로그[-30:])
    print(f"모델 {time.time() - t0:.1f} s")

    cp = ctypes.create_string_buffer(512)
    L.llama_context_default_params(cp)
    기본 = struct.unpack_from("<IIII", cp.raw, 0) + struct.unpack_from("<ii", cp.raw, 104)
    assert 기본 == (512, 2048, 512, 1, 1, 1), f"문맥 구조체 기본값이 예상과 다르다: {기본}"
    assert struct.unpack_from("<QQ", cp.raw, 88) == (0, 0), "cb_eval 자리가 비어 있지 않다"
    struct.pack_into("<IIII", cp, 0, n_ctx, ub, ub, 1)
    struct.pack_into("<i", cp, 52, 1)                                  # flash attention 켬

    ggml_get_name = 찾기("ggml_get_name", B, G)
    ggml_get_name.restype, ggml_get_name.argtypes = ctypes.c_char_p, [c_p]
    tensor_get = 찾기("ggml_backend_tensor_get", B, G)
    tensor_get.restype, tensor_get.argtypes = None, [c_p, c_p, ctypes.c_size_t, ctypes.c_size_t]

    모음 = {}                     # 층 → [배열(토큰, 고름)] (ubatch 차례)
    잡음 = re.compile(rb"ffn_moe_topk-(\d+)$")

    def 보기(t, ask, user):
        이름 = ggml_get_name(t)
        m = 잡음.match(이름)
        if ask:
            return m is not None
        if m is None:
            return True
        typ = struct.unpack_from("<i", ctypes.string_at(t, 4))[0]
        ne0, ne1 = struct.unpack_from("<qq", ctypes.string_at(t + 16, 16))
        nb0, nb1 = struct.unpack_from("<QQ", ctypes.string_at(t + 48, 16))
        assert typ == 26 and nb0 == 4, f"topk 형식 {typ} nb0 {nb0}"          # GGML_TYPE_I32
        n = (ne1 - 1) * nb1 + ne0 * 4
        buf = ctypes.create_string_buffer(n)
        tensor_get(t, buf, 0, n)
        a = np.lib.stride_tricks.as_strided(np.frombuffer(buf.raw, np.int32), shape=(ne1, ne0), strides=(nb1, 4)).copy()
        모음.setdefault(int(m.group(1)), []).append(a)
        return True

    cb = ctypes.CFUNCTYPE(ctypes.c_bool, c_p, ctypes.c_bool, c_p)(보기)
    struct.pack_into("<Q", cp, 88, ctypes.cast(cb, c_p).value)
    ctx = L.llama_init_from_model(model, cp)
    assert ctx, "문맥을 못 만듦\n" + "".join(로그[-30:])
    mem = L.llama_get_memory(ctx)

    batch = ctypes.create_string_buffer(56)
    L.llama_batch_init(batch, ub, 0, 1)
    p_tok, p_pos, p_nseq, p_seq, p_log = (struct.unpack_from("<Q", batch.raw, o)[0] for o in (8, 24, 32, 40, 48))
    seqp = ctypes.cast(p_seq, ctypes.POINTER(ctypes.c_void_p))
    for i in range(ub):
        ctypes.cast(seqp[i], ctypes.POINTER(ctypes.c_int32))[0] = 0
    하나들 = np.ones(ub, np.int32)          # 이름에 묶어 둔다 — 임시 배열의 주소만 넘기면 복사 전에 해제될 수 있다(라마.py 의 첫 판 버그)
    ctypes.memmove(p_nseq, 하나들.ctypes.data, 4 * ub)

    ids_all, kind_all, req_all, pos_all = [], [], [], []
    종류 = {"prompt": 0, "output": 1, "reprocess": 2}
    pos = 0
    t0 = time.time()
    n_tok = 0
    for s in 일감["spans"]:
        if s.get("reset"):
            L.llama_memory_clear(mem, True)
            pos = 0
        toks = np.asarray(s["tokens"], np.int32)
        for a in range(0, len(toks), ub):
            chunk = np.ascontiguousarray(toks[a:a + ub])
            n = len(chunk)
            ps = np.arange(pos, pos + n, dtype=np.int32)
            # 모든 행에 로짓 — llama.cpp 는 마지막 층의 FFN 을 로짓을 낼 행에만 셈하므로(inp_out_ids) 그래야 마지막 층의 고르기도 모든 토큰에 있다
            lg = np.ones(n, np.int8)
            ctypes.memmove(p_tok, chunk.ctypes.data, 4 * n)
            ctypes.memmove(p_pos, ps.ctypes.data, 4 * n)
            ctypes.memmove(p_log, lg.ctypes.data, n)
            struct.pack_into("<i", batch, 0, n)
            모음.clear()
            r = L.llama_decode(ctx, batch)
            assert r == 0, f"llama_decode 실패 {r}\n" + "".join(로그[-25:])
            층들 = sorted(모음)
            per = np.stack([np.concatenate(모음[il], 0) for il in 층들], 1)        # [n, 층, 고름]
            assert per.shape[0] == n, (per.shape, n)
            ids_all.append(per.astype(np.uint16))
            kind_all.append(np.full(n, 종류[s["kind"]], np.uint8))
            req_all.append(np.full(n, s.get("req", -1), np.int32))
            pos_all.append(ps)
            pos += n
            n_tok += n
        print(f"조각 req {s.get('req')} {s['kind']} {len(toks)} 토큰 · 위치 {pos} · 누적 {n_tok} · {time.time() - t0:.0f} s",
              flush=True)
    np.savez_compressed(출력, ids=np.concatenate(ids_all), kind=np.concatenate(kind_all), req=np.concatenate(req_all),
                        pos=np.concatenate(pos_all), layers=np.array(층들))
    print(f"끝 {n_tok} 토큰 {time.time() - t0:.0f} s → {출력}")
    L.llama_free(ctx)
    L.llama_model_free(model)


if __name__ == "__main__":
    main()
