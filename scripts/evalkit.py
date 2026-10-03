"""Shared evaluation helpers: the compared systems, metrics, and threshold selection.

Used by scripts/tune.py (dev set only) and scripts/evaluate.py (held-out test).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.retrieval import (  # noqa: E402
    BM25F, HybridRetriever, LSARetriever, Passage, load_corpus, rrf,
)

EVAL_DIR = ROOT / "evaluation"

# --- the original FitSage "keyword retrieval" baseline, reproduced ------------------
_OLD_WORD = re.compile(r"[a-zA-Z]+")
_OLD_STOP = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "in",
    "is", "it", "my", "of", "should", "the", "to", "what", "which", "with", "you",
}


def _old_tokens(text: str) -> set[str]:
    return {w.lower() for w in _OLD_WORD.findall(text) if w.lower() not in _OLD_STOP}


class OverlapBaseline:
    """Count of shared (unstemmed) words between the query and passage text."""

    name = "keyword-overlap"

    def __init__(self, passages):
        self.passages = list(passages)
        self._terms = [_old_tokens(p.text) for p in self.passages]

    def rank_and_conf(self, query: str):
        q = _old_tokens(query)
        s = np.array([len(q & t) for t in self._terms], dtype=float)
        order = np.argsort(-s, kind="stable").tolist()
        conf = float(s[order[0]] / len(q)) if q else 0.0
        return order, conf


class _Scored:
    """Adapter: anything with .scores(query) -> array, plus a confidence function."""

    def __init__(self, name, obj, conf_fn: Callable):
        self.name, self.obj, self.conf_fn = name, obj, conf_fn

    def rank_and_conf(self, query: str):
        s = self.obj.scores(query)
        order = np.argsort(-s, kind="stable").tolist()
        return order, self.conf_fn(query, order[0], s)


def build_systems(passages, cfg: dict) -> dict:
    """Instantiate every compared system from a tuned-config dict."""
    systems = {"keyword-overlap": OverlapBaseline(passages)}

    plain = BM25F(passages, weights={"text": 1.0}, k1=1.2, b=0.75, stem=False)
    systems["bm25-plain"] = _Scored("bm25-plain", plain, lambda q, i, s: float(plain.coverages(q)[i]))

    bm = BM25F(passages, **cfg["bm25f"]["params"])
    systems["bm25f-stem"] = _Scored("bm25f-stem", bm, lambda q, i, s: float(bm.coverages(q)[i]))

    lsa = LSARetriever(passages, components=cfg["lsa"]["params"]["components"])
    systems["lsa"] = _Scored("lsa", lsa, lambda q, i, s: float(s[i]))

    hy = HybridRetriever(
        passages,
        bm25=cfg["hybrid"]["bm25"],
        lsa_components=cfg["hybrid"]["lsa_components"],
        rrf_k=cfg["hybrid"]["rrf_k"],
        tau=0.0,
    )

    class _Hybrid:
        name = "hybrid"

        def rank_and_conf(self, query: str):
            order = hy.rank(query)
            return order, float(hy.bm25.coverages(query)[order[0]])

    systems["hybrid"] = _Hybrid()
    return systems


# --- metrics ---------------------------------------------------------------------
def load_queries(name: str) -> list[dict]:
    return json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))


def rows_for(system, passages, queries) -> list[dict]:
    ids = [p.id for p in passages]
    rows = []
    for q in queries:
        order, conf = system.rank_and_conf(q["question"])
        ranked_ids = [ids[i] for i in order]
        rel = set(q["relevant_passage_ids"])
        first = next((r for r, pid in enumerate(ranked_ids, 1) if pid in rel), None)
        rows.append({
            "id": q["id"], "answerable": q["answerable"], "conf": conf,
            "top1": ranked_ids[0], "first_relevant_rank": first,
        })
    return rows


def mrr_at(rows, k=5) -> float:
    a = [r for r in rows if r["answerable"]]
    return sum(1.0 / r["first_relevant_rank"] if r["first_relevant_rank"] and r["first_relevant_rank"] <= k else 0.0 for r in a) / len(a)


def recall_at(rows, k) -> float:
    a = [r for r in rows if r["answerable"]]
    return sum(1 for r in a if r["first_relevant_rank"] and r["first_relevant_rank"] <= k) / len(a)


def end_to_end_correct(row, tau) -> bool:
    answers = row["conf"] >= tau
    if row["answerable"]:
        return answers and row["first_relevant_rank"] == 1
    return not answers


FALSE_ANSWER_COST = 2.0  # showing a wrong / out-of-scope answer costs twice a refusal


def loss(row, tau, c=FALSE_ANSWER_COST) -> float:
    """Safety-first loss: a wrong or out-of-scope answer costs `c`; a needless refusal costs 1."""
    answers = row["conf"] >= tau
    if row["answerable"]:
        if not answers:
            return 1.0
        return 0.0 if row["first_relevant_rank"] == 1 else c
    return c if answers else 0.0


def choose_tau(rows, c=FALSE_ANSWER_COST) -> float:
    """Pick the abstention threshold minimising the safety-first loss on `rows`.

    Candidates are midpoints between consecutive observed confidences; among the
    optimal candidates the median is used so the threshold is not at an edge.
    """
    confs = sorted({round(r["conf"], 6) for r in rows})
    cands = [0.0] + [(a + b) / 2 for a, b in zip(confs, confs[1:])] + [confs[-1] + 1e-6]
    scored = [(sum(loss(r, t, c) for r in rows), t) for t in cands]
    best = min(s for s, _ in scored)
    best_taus = [t for s, t in scored if abs(s - best) < 1e-9]
    return float(best_taus[len(best_taus) // 2])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def paired_bootstrap(a: list[float], b: list[float], iters=10000, seed=0) -> tuple[float, float, float]:
    """Mean(a-b) with a 95% percentile bootstrap CI over queries."""
    rng = np.random.default_rng(seed)
    diff = np.array(a) - np.array(b)
    idx = rng.integers(0, len(diff), size=(iters, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
