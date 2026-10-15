# -*- coding: utf-8 -*-
"""实验的公共模块：数据集登记、fvecs/ivecs 读写、结果目录约定。

路径全部来自环境变量（见 env.sh），本地与服务器共用同一份代码：
  SCDKM_DATA   7 个 1M 级数据集所在目录
  DATA10M      Deep-10M 所在目录
  DATA_MSMARCO MS MARCO（BGE-small 编码后）所在目录
  RES          本次实验的结果根目录（每个数据集一个子目录）
"""
import hashlib
import json
import os

import numpy as np

DATA = os.environ.get("SCDKM_DATA", "/path/to/data")
DATA10M = os.environ.get("DATA10M", "/path/to/workspace/data10m")
DATA_MSMARCO = os.environ.get("DATA_MSMARCO", "/path/to/workspace/data_msmarco")
RES = os.environ.get("RES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

# name -> (base, query, gt, c)；c = round(4*sqrt(n))，与 FAISS / SuperKMeans 的经验规则一致
DATASETS = {
    "sift":     (f"{DATA}/sift/sift_base.fvecs", f"{DATA}/sift/sift_query.fvecs", f"{DATA}/sift/sift_groundtruth.ivecs", 4000),
    "glove200": (f"{DATA}/glove200_base.fvecs", f"{DATA}/glove200_query.fvecs", f"{DATA}/glove200_gt.ivecs", 4352),
    "yahoo":    (f"{DATA}/yahoo_base.fvecs", f"{DATA}/yahoo_query.fvecs", f"{DATA}/yahoo_gt.ivecs", 3292),
    "gist":     (f"{DATA}/gist_base.fvecs", f"{DATA}/gist_query.fvecs", f"{DATA}/gist_gt.ivecs", 4000),
    "mxbai":    (f"{DATA}/mxbai_base.fvecs", f"{DATA}/mxbai_query.fvecs", f"{DATA}/mxbai_gt.ivecs", 3509),
    "dbpedia":  (f"{DATA}/dbpedia/dbpedia_base.fvecs", f"{DATA}/dbpedia/dbpedia_query.fvecs", f"{DATA}/dbpedia/dbpedia_gt.ivecs", 1917),
    "wiki":     (f"{DATA}/wiki_base.fvecs", f"{DATA}/wiki_query.fvecs", f"{DATA}/wiki_gt.ivecs", 2041),
    "deep10m":  (f"{DATA10M}/deep10m_base.fvecs", f"{DATA10M}/deep10m_query.fvecs", f"{DATA10M}/deep10m_gt.ivecs", 12649),
    "msmarco":  (f"{DATA_MSMARCO}/msmarco_base.fvecs", f"{DATA_MSMARCO}/msmarco_query.fvecs", f"{DATA_MSMARCO}/msmarco_gt.ivecs", 11894),
}
_meta = os.path.join(DATA_MSMARCO, "msmarco_meta.json")
if os.path.exists(_meta):   # c = round(4√n)，以编码后实际的 n 为准
    _b, _q, _g, _ = DATASETS["msmarco"]
    DATASETS["msmarco"] = (_b, _q, _g, int(json.load(open(_meta))["c"]))
STATIC_DS = ["sift", "glove200", "yahoo", "gist", "mxbai", "dbpedia", "wiki", "deep10m"]
SEEDS = [42, 1, 2, 3, 4]
NQ = 2000


def c_of(ds):
    return int(os.environ.get("SCDKM_C", DATASETS[ds][3]))


def res_dir(ds):
    """结果子目录；SCDKM_C 覆盖簇数时（c 扫描）自动分到 <ds>_c<c>。"""
    c = os.environ.get("SCDKM_C")
    d = os.path.join(RES, ds if not c else f"{ds}_c{c}")
    os.makedirs(d, exist_ok=True)
    return d


def read_fvecs(path, count=None):
    x = np.fromfile(path, dtype=np.float32, count=-1 if count is None else count * (1 + _dim(path)))
    d = int(x[:1].view(np.int32)[0])
    x = x.reshape(-1, d + 1)
    return np.ascontiguousarray(x[:, 1:])


def _dim(path):
    return int(np.fromfile(path, dtype=np.int32, count=1)[0])


def read_ivecs(path, count=None):
    d = _dim(path)
    x = np.fromfile(path, dtype=np.int32, count=-1 if count is None else count * (1 + d))
    return np.ascontiguousarray(x.reshape(-1, d + 1)[:, 1:])


def write_fvecs(path, a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    n, d = a.shape
    out = np.empty((n, d + 1), np.float32)
    out[:, 0] = np.array([d], np.int32).view(np.float32)[0]
    out[:, 1:] = a
    out.tofile(path)


def write_ivecs(path, a):
    a = np.ascontiguousarray(a, dtype=np.int32)
    n, d = a.shape
    out = np.empty((n, d + 1), np.int32)
    out[:, 0] = d
    out[:, 1:] = a
    out.tofile(path)


def load_centroids(path):
    return np.load(path) if path.endswith(".npy") else read_fvecs(path)


def load_queries(ds, nq=NQ):
    _b, q, g, _c = DATASETS[ds]
    Q = read_fvecs(q, nq)
    G = read_ivecs(g, nq)
    n = min(Q.shape[0], G.shape[0])
    return np.ascontiguousarray(Q[:n]), np.ascontiguousarray(G[:n, :10])


def cpp_train_s(log_path):
    """C++ bench 日志里的 train_ms（只含 Train()，不含读数据）。"""
    for line in open(log_path, errors="replace"):
        if line.startswith("train_ms,"):
            return float(line.split(",")[1]) / 1000.0
    raise ValueError(f"no train_ms in {log_path}")


def labels_digest(labels):
    return hashlib.md5(np.ascontiguousarray(labels, dtype=np.int32).tobytes()).hexdigest()


def dump_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)
