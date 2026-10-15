#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MS MARCO passage（BEIR 版，8,841,823 段）→ BGE-small-en-v1.5 向量（384 维，L2 归一化）→ fvecs + 精确 top-100 + qrels。

用 $PY_ENC（torch + sentence-transformers）运行：
  prepare.py encode --shard 0 --nshards 2 --device cuda:0   # 语料分片编码（两张卡各跑一片）
  prepare.py queries                                       # dev 查询（6,980 条，有 qrels）编码，加 BGE 检索指令前缀
  prepare.py finalize                                      # 合并分片 → msmarco_base.fvecs；GPU 精确 top-100；qrels；精确检索的参考指标
数据来源：https://huggingface.co/datasets/BeIR/msmarco （corpus/queries parquet）与 BeIR/msmarco-qrels（dev.tsv）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import DATA_MSMARCO, write_fvecs, write_ivecs  # noqa: E402

RAW = os.path.join(DATA_MSMARCO, "raw")
MODEL = "BAAI/bge-small-en-v1.5"
MODEL_DIR = os.path.join(DATA_MSMARCO, "bge-small-en-v1.5")   # 由 get_model.sh 用 curl 下载到本地（离线加载）
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
CHUNK = 200_000


def load_model(device):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.backends.cuda.matmul.allow_tf32 = False
    m = SentenceTransformer(MODEL_DIR if os.path.isdir(MODEL_DIR) else MODEL, device=device)
    m.half()
    m.max_seq_length = 256
    return m


def corpus_texts():
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(RAW, "corpus.parquet"), columns=["_id", "title", "text"])
    ids = t.column("_id").to_pylist()
    titles = t.column("title").to_pylist()
    texts = t.column("text").to_pylist()
    docs = [(ti + " " + tx).strip() if ti else tx for ti, tx in zip(titles, texts)]
    return ids, docs


def cmd_encode(a):
    ids, docs = corpus_texts()
    n = len(docs)
    lo, hi = n * a.shard // a.nshards, n * (a.shard + 1) // a.nshards
    model = load_model(a.device)
    os.makedirs(os.path.join(DATA_MSMARCO, "shards"), exist_ok=True)
    t0 = time.time()
    for s in range(lo, hi, CHUNK):
        e = min(hi, s + CHUNK)
        out = os.path.join(DATA_MSMARCO, "shards", f"emb_{s:08d}_{e:08d}.npy")
        if os.path.exists(out):
            continue
        emb = model.encode(docs[s:e], batch_size=512, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        np.save(out + ".tmp.npy", emb.astype(np.float32))
        os.replace(out + ".tmp.npy", out)
        print(f"shard {a.shard}: {e - lo}/{hi - lo} docs, {(e - lo) / (time.time() - t0):.0f} docs/s", flush=True)
    if a.shard == 0:
        json.dump(ids, open(os.path.join(DATA_MSMARCO, "corpus_ids.json"), "w"))


def cmd_queries(a):
    import pyarrow.parquet as pq
    q = pq.read_table(os.path.join(RAW, "queries.parquet"), columns=["_id", "text"])
    qtext = dict(zip(q.column("_id").to_pylist(), q.column("text").to_pylist()))
    qrels = {}
    for line in open(os.path.join(RAW, "dev.tsv")):
        if line.startswith("query-id"):
            continue
        qid, did, score = line.rstrip("\n").split("\t")
        if int(score) > 0:
            qrels.setdefault(qid, []).append(did)
    qids = list(qrels.keys())   # dev.tsv 中出现的顺序
    model = load_model(a.device)
    emb = model.encode([QUERY_INSTRUCTION + qtext[x] for x in qids], batch_size=512, normalize_embeddings=True, convert_to_numpy=True)
    np.save(os.path.join(DATA_MSMARCO, "query_emb.npy"), emb.astype(np.float32))
    json.dump(dict(qids=qids, qrels=qrels), open(os.path.join(DATA_MSMARCO, "qrels_raw.json"), "w"))
    print("queries", len(qids), emb.shape, flush=True)


def cmd_finalize(a):
    import glob
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    files = sorted(glob.glob(os.path.join(DATA_MSMARCO, "shards", "emb_*.npy")))
    ids = json.load(open(os.path.join(DATA_MSMARCO, "corpus_ids.json")))
    n = len(ids)
    base = np.concatenate([np.load(f) for f in files], axis=0)
    assert base.shape[0] == n, (base.shape, n)
    d = base.shape[1]
    t0 = time.time()
    write_fvecs(os.path.join(DATA_MSMARCO, "msmarco_base.fvecs"), base)
    print("wrote base", base.shape, f"{time.time() - t0:.0f}s", flush=True)
    Q = np.load(os.path.join(DATA_MSMARCO, "query_emb.npy"))
    write_fvecs(os.path.join(DATA_MSMARCO, "msmarco_query.fvecs"), Q)
    # 精确 top-100（内积 = 余弦；两边都已 L2 归一化，所以与 L2 排序一致）
    dev = torch.device(a.device)
    qt = torch.from_numpy(Q).to(dev)
    best_s = torch.full((Q.shape[0], 100), -1e9, device=dev)
    best_i = torch.zeros((Q.shape[0], 100), dtype=torch.int64, device=dev)
    step = 200_000   # 6,980 条查询 × 20 万段落的得分矩阵约 5.6 GB（100 万会超过 24 GB 显存）
    for s in range(0, n, step):
        bt = torch.from_numpy(base[s:s + step]).to(dev)
        sc = qt @ bt.T
        vs, ix = torch.topk(sc, 100, dim=1)
        cat_s = torch.cat([best_s, vs], 1)
        cat_i = torch.cat([best_i, ix + s], 1)
        top = torch.topk(cat_s, 100, dim=1)
        best_s, best_i = top.values, torch.gather(cat_i, 1, top.indices)
    gt = best_i.cpu().numpy().astype(np.int32)
    write_ivecs(os.path.join(DATA_MSMARCO, "msmarco_gt.ivecs"), gt)
    raw = json.load(open(os.path.join(DATA_MSMARCO, "qrels_raw.json")))
    pos = {x: i for i, x in enumerate(ids)}
    rel = [[pos[dd] for dd in raw["qrels"][q] if dd in pos] for q in raw["qids"]]
    json.dump(dict(qids=raw["qids"], rel=rel), open(os.path.join(DATA_MSMARCO, "msmarco_qrels.json"), "w"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ir_metrics import ir_metrics
    ref = ir_metrics(gt, rel)
    c = int(round(4 * np.sqrt(n)))
    meta = dict(n=n, d=d, c=c, nq=int(Q.shape[0]), model=MODEL, query_instruction=QUERY_INSTRUCTION, max_seq_length=256,
                fp16=True, exact_search=ref, created=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(meta, open(os.path.join(DATA_MSMARCO, "msmarco_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "queries", "finalize"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dict(encode=cmd_encode, queries=cmd_queries, finalize=cmd_finalize)[a.cmd](a)
