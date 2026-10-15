# -*- coding: utf-8 -*-
"""MS MARCO 检索指标（二值相关性）：MRR@10、nDCG@10、Recall@100。返回均值与逐查询数组。"""
import numpy as np


def per_query(I, rel):
    """I: nq×k 的检索结果（按得分降序，-1 表示空位）；rel: 每条查询的相关文档下标列表。"""
    nq = I.shape[0]
    rr10 = np.zeros(nq)
    ndcg10 = np.zeros(nq)
    r100 = np.zeros(nq)
    disc = 1.0 / np.log2(np.arange(2, 12))
    for qi in range(nq):
        R = set(rel[qi])
        if not R:
            continue
        top = I[qi]
        hit10 = [1.0 if x in R else 0.0 for x in top[:10]]
        for r, h in enumerate(hit10):
            if h:
                rr10[qi] = 1.0 / (r + 1)
                break
        dcg = float(np.dot(hit10, disc[:len(hit10)]))
        idcg = float(disc[:min(len(R), 10)].sum())
        ndcg10[qi] = dcg / idcg
        r100[qi] = len(R.intersection(top[:100].tolist())) / len(R)
    return dict(mrr10=rr10, ndcg10=ndcg10, recall100=r100)


def ir_metrics(I, rel):
    pq = per_query(I, rel)
    return {k: float(v.mean()) for k, v in pq.items()}
