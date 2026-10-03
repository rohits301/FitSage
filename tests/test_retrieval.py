import json
from pathlib import Path

from app.services.retrieval import BM25F, HybridRetriever, LSARetriever, load_config, load_corpus, rrf
from app.services.text import tokenize

ROOT = Path(__file__).resolve().parents[1]
PASSAGES = load_corpus()


def test_corpus_integrity():
    ids = [p.id for p in PASSAGES]
    assert len(ids) == len(set(ids))
    for p in PASSAGES:
        assert p.url.startswith("https://") and p.text.strip() and p.verified
        assert p.kind in {"prose", "table", "list"}
    assert {p.organization for p in PASSAGES} == {"NIH Office of Dietary Supplements", "NIH MedlinePlus", "PubMed"}


def test_tokenizer_stems_and_drops_stop_words():
    assert tokenize("What are the supplements for runners?") == ["supplement", "runner"]
    assert "vitamin_d" in tokenize("vitamin D", bigrams=True)


def test_bm25f_finds_the_obvious_passage():
    top = BM25F(PASSAGES, **load_config()["bm25f"]["params"]).search("which foods contain magnesium", k=1)[0]
    assert top.passage.id == "mg-3"


def test_lsa_scores_have_one_value_per_passage():
    assert LSARetriever(PASSAGES, components=16).scores("zinc").shape == (len(PASSAGES),)


def test_rrf_prefers_items_ranked_well_by_both_lists():
    assert rrf([[1, 2, 3], [3, 1, 2]])[0] == 1


def test_hybrid_abstains_on_out_of_corpus_question():
    hy = HybridRetriever.from_config(PASSAGES)
    assert hy.search("Which running shoes should I buy?") == []


def test_frozen_test_set_regression_floor():
    """Guards against accidental retrieval regressions; floors are well below the measured values."""
    hy = HybridRetriever.from_config(PASSAGES)
    rows = json.loads((ROOT / "evaluation" / "queries_test.json").read_text())
    ans = [r for r in rows if r["answerable"]]
    top1 = sum(1 for r in ans if PASSAGES[hy.rank(r["question"])[0]].id in r["relevant_passage_ids"])
    assert top1 / len(ans) >= 0.85
