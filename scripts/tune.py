"""Tune retrieval hyper-parameters and abstention thresholds on the DEV set only.

Writes app/data/retrieval_config.json. The held-out test queries are never read here;
scripts/evaluate.py reads the written config and reports on the test set.
"""
import itertools
import json
from pathlib import Path

from evalkit import (
    BM25F, HybridRetriever, LSARetriever, ROOT, build_systems, choose_tau,
    load_corpus, load_queries, mrr_at, rows_for,
)

passages = load_corpus()
dev = load_queries("queries_dev.json")
ids = [p.id for p in passages]


class _Rank:
    """Wrap a score function so rows_for() can score it with a tau-free confidence."""

    def __init__(self, fn):
        self.fn = fn

    def rank_and_conf(self, q):
        order, conf = self.fn(q)
        return order, conf


def dev_mrr(system) -> float:
    return mrr_at(rows_for(system, passages, dev))


import numpy as np  # noqa: E402

# 1) BM25F: defaults first so ties keep the defaults.
best = None
grid = itertools.product([False, True], [1.2, 0.9, 1.5, 2.0], [0.75, 0.5, 0.9], [1.0, 0.0, 2.0, 3.0], [0.5, 0.0, 1.0])
for bigrams, k1, b, w_sec, w_title in grid:
    weights = {"text": 1.0, "section": w_sec, "title": w_title}
    weights = {f: w for f, w in weights.items() if w}
    m = BM25F(passages, weights=weights, k1=k1, b=b, stem=True, bigrams=bigrams)
    sys_ = _Rank(lambda q, m=m: (np.argsort(-m.scores(q), kind="stable").tolist(), float(m.coverages(q).max())))
    score = dev_mrr(sys_)
    if best is None or score > best[0] + 1e-9:
        best = (score, {"weights": weights, "k1": k1, "b": b, "stem": True, "bigrams": bigrams})
bm25f_params = best[1]
print("BM25F dev MRR@5 =", round(best[0], 4), bm25f_params)

# 2) LSA components.
best = None
for comp in [24, 8, 16, 32, 48]:
    m = LSARetriever(passages, components=comp)
    sys_ = _Rank(lambda q, m=m: (np.argsort(-m.scores(q), kind="stable").tolist(), float(m.scores(q).max())))
    score = dev_mrr(sys_)
    if best is None or score > best[0] + 1e-9:
        best = (score, comp)
lsa_comp = best[1]
print("LSA dev MRR@5 =", round(best[0], 4), "components =", lsa_comp)

# 3) RRF k for the hybrid.
best = None
for rk in [20, 5, 60]:
    hy = HybridRetriever(passages, bm25=bm25f_params, lsa_components=lsa_comp, rrf_k=rk)
    sys_ = _Rank(lambda q, hy=hy: (hy.rank(q), 0.0))
    score = dev_mrr(sys_)
    if best is None or score > best[0] + 1e-9:
        best = (score, rk)
rrf_k = best[1]
print("Hybrid dev MRR@5 =", round(best[0], 4), "rrf_k =", rrf_k)

cfg = {
    "tuned_on": "evaluation/queries_dev.json",
    "bm25f": {"params": bm25f_params},
    "lsa": {"params": {"components": lsa_comp}},
    "hybrid": {"bm25": bm25f_params, "lsa_components": lsa_comp, "rrf_k": rrf_k},
}

# 4) Abstention thresholds, one per system, from dev end-to-end accuracy.
systems = build_systems(passages, cfg)
taus = {}
for name, system in systems.items():
    taus[name] = choose_tau(rows_for(system, passages, dev))
    print(f"tau[{name}] = {taus[name]:.4f}")
cfg["tau"] = taus
cfg["hybrid"]["tau"] = taus["hybrid"]

out = ROOT / "app" / "data" / "retrieval_config.json"
out.write_text(json.dumps(cfg, indent=2) + "\n")
print("wrote", out.relative_to(ROOT))
