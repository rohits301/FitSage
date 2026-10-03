"""Retrieval over the curated passage corpus.

Three components, each independently testable and each reported separately in
``evaluation/results.json``:

* ``BM25F``          - lexical ranking over passage text, section heading and source title
* ``LSARetriever``   - latent-semantic ranking (TF-IDF -> truncated SVD -> cosine)
* ``HybridRetriever``- reciprocal-rank fusion of the two, plus a coverage-based
                       confidence used to abstain on out-of-corpus questions

No neural embedding model is used: the only dense representation is LSA.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from app.services.text import tokenize

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class Passage:
    id: str
    organization: str
    title: str
    section: str
    url: str
    text: str
    retrieved: str
    kind: str = "prose"  # prose | table | list (table/list passages are flattened as the page renders them)
    verified: str = ""  # date the text was matched against the rendered source page


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float
    confidence: float  # IDF-weighted share of query terms found in this passage (0-1)


def load_corpus(path: Optional[Path] = None) -> list[Passage]:
    raw = json.loads((path or DATA_DIR / "corpus.json").read_text(encoding="utf-8"))
    return [Passage(**row) for row in raw]


def load_config(path: Optional[Path] = None) -> dict:
    return json.loads((path or DATA_DIR / "retrieval_config.json").read_text(encoding="utf-8"))


class BM25F:
    """BM25F over (text, section, title) with per-field weights.

    With ``weights={"text": 1.0}`` and ``stem=False`` this reduces to ordinary
    BM25 over passage text, which is the lexical baseline in the evaluation.
    """

    def __init__(
        self,
        passages: Sequence[Passage],
        *,
        weights: Optional[dict[str, float]] = None,
        k1: float = 1.2,
        b: float = 0.75,
        stem: bool = True,
        bigrams: bool = False,
    ) -> None:
        self.passages = list(passages)
        self.weights = weights or {"text": 1.0, "section": 1.0, "title": 0.5}
        self.k1, self.b, self.stem, self.bigrams = k1, b, stem, bigrams
        self._tf: list[dict[str, Counter]] = []
        self._len: dict[str, list[int]] = {f: [] for f in self.weights}
        df: Counter = Counter()
        for p in self.passages:
            per_field: dict[str, Counter] = {}
            seen: set[str] = set()
            for field in self.weights:
                toks = tokenize(getattr(p, field), stem=stem, bigrams=bigrams)
                per_field[field] = Counter(toks)
                self._len[field].append(len(toks))
                seen.update(toks)
            self._tf.append(per_field)
            df.update(seen)
        n = len(self.passages)
        self._idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._oov_idf = math.log(1.0 + (n + 0.5) / 0.5)  # an unseen term is maximally rare
        self._avg = {f: (sum(v) / n) or 1.0 for f, v in self._len.items()}
        self._terms = [set().union(*[set(c) for c in fields.values()]) for fields in self._tf]

    def _score(self, q_terms: Sequence[str], i: int) -> float:
        score = 0.0
        for t in set(q_terms):
            idf = self._idf.get(t)
            if idf is None:
                continue
            tf = 0.0
            for field, w in self.weights.items():
                f = self._tf[i][field].get(t, 0)
                if f and w:
                    norm = 1.0 - self.b + self.b * self._len[field][i] / self._avg[field]
                    tf += w * f / norm
            if tf:
                score += idf * tf / (self.k1 + tf)
        return score

    def coverage(self, q_terms: Sequence[str], i: int) -> float:
        """IDF-weighted fraction of distinct query terms that appear in passage i."""
        terms = set(q_terms)
        if not terms:
            return 0.0
        total = sum(self._idf.get(t, self._oov_idf) for t in terms)
        hit = sum(self._idf[t] for t in terms if t in self._terms[i])
        return hit / total if total else 0.0

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query, stem=self.stem, bigrams=self.bigrams)
        return np.array([self._score(q, i) for i in range(len(self.passages))])

    def coverages(self, query: str) -> np.ndarray:
        q = tokenize(query, stem=self.stem, bigrams=self.bigrams)
        return np.array([self.coverage(q, i) for i in range(len(self.passages))])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        s, c = self.scores(query), self.coverages(query)
        order = np.argsort(-s, kind="stable")[:k]
        return [Hit(self.passages[i], float(s[i]), float(c[i])) for i in order if s[i] > 0]


class LSARetriever:
    """TF-IDF -> truncated SVD (latent semantic analysis) -> cosine similarity."""

    def __init__(self, passages: Sequence[Passage], *, components: int = 24) -> None:
        self.passages = list(passages)
        docs = [tokenize(f"{p.title} {p.section} {p.text}") for p in self.passages]
        vocab = sorted({t for d in docs for t in d})
        self._index = {t: j for j, t in enumerate(vocab)}
        n = len(docs)
        df = np.zeros(len(vocab))
        tf = np.zeros((n, len(vocab)))
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                tf[i, self._index[t]] = 1.0 + math.log(f)  # sublinear tf
                df[self._index[t]] += 1
        self._idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        x = tf * self._idf
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        k = max(1, min(components, len(s)))
        self._v = vt[:k].T  # terms x k
        d = u[:, :k] * s[:k]
        self._docs = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)

    def scores(self, query: str) -> np.ndarray:
        q = np.zeros(len(self._index))
        for t, f in Counter(tokenize(query)).items():
            j = self._index.get(t)
            if j is not None:
                q[j] = (1.0 + math.log(f)) * self._idf[j]
        qk = q @ self._v
        norm = np.linalg.norm(qk)
        if norm < 1e-12:
            return np.zeros(len(self.passages))
        return self._docs @ (qk / norm)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        return [Hit(self.passages[i], float(s[i]), float(s[i])) for i in order if s[i] > 0]


def rrf(rankings: Sequence[Sequence[int]], k: int = 20) -> list[int]:
    """Reciprocal-rank fusion over several best-first lists of passage indices."""
    fused: Counter = Counter()
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] += 1.0 / (k + rank + 1)
    return [idx for idx, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]


class HybridRetriever:
    """BM25F + LSA fused with RRF. Abstains when lexical coverage is below ``tau``."""

    def __init__(
        self,
        passages: Sequence[Passage],
        *,
        bm25: Optional[dict] = None,
        lsa_components: int = 24,
        rrf_k: int = 20,
        tau: float = 0.0,
    ) -> None:
        self.passages = list(passages)
        self.bm25 = BM25F(self.passages, **(bm25 or {}))
        self.lsa = LSARetriever(self.passages, components=lsa_components)
        self.rrf_k, self.tau = rrf_k, tau

    @classmethod
    def from_config(cls, passages: Sequence[Passage], config: Optional[dict] = None) -> "HybridRetriever":
        cfg = (config or load_config())["hybrid"]
        return cls(
            passages,
            bm25=cfg["bm25"],
            lsa_components=cfg["lsa_components"],
            rrf_k=cfg["rrf_k"],
            tau=cfg["tau"],
        )

    def rank(self, query: str) -> list[int]:
        """All passages, best first, ignoring the abstention threshold."""
        lex = np.argsort(-self.bm25.scores(query), kind="stable").tolist()
        sem = np.argsort(-self.lsa.scores(query), kind="stable").tolist()
        return rrf([lex, sem], k=self.rrf_k)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        cov = self.bm25.coverages(query)
        lex = self.bm25.scores(query)
        order = self.rank(query)
        if not order or cov[order[0]] < self.tau:
            return []
        return [Hit(self.passages[i], float(lex[i]), float(cov[i])) for i in order[:k]]
