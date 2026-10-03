"""Evaluate every system on the held-out test queries (and, for reference, on dev).

    python scripts/tune.py        # dev only; writes app/data/retrieval_config.json
    python scripts/evaluate.py    # reads that config; writes evaluation/results.json

The test set is never used for tuning. Thresholds and hyper-parameters come from
the config file produced by tune.py.
"""
import hashlib
import json

from evalkit import (
    EVAL_DIR, FALSE_ANSWER_COST, ROOT, build_systems, end_to_end_correct, load_corpus, load_queries,
    loss, mrr_at, paired_bootstrap, recall_at, rows_for, wilson,
)

ORDER = ["keyword-overlap", "bm25-plain", "bm25f-stem", "lsa", "hybrid"]

passages = load_corpus()
cfg = json.loads((ROOT / "app" / "data" / "retrieval_config.json").read_text())
systems = build_systems(passages, cfg)
by_id = {p.id: p for p in passages}


def summarize(rows, tau):
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    top1 = sum(1 for r in ans if r["first_relevant_rank"] == 1)
    abst = sum(1 for r in una if r["conf"] < tau)
    false_abst = sum(1 for r in ans if r["conf"] < tau)
    e2e = sum(end_to_end_correct(r, tau) for r in rows)
    shown = [r for r in rows if r["conf"] >= tau]
    bad = sum(1 for r in shown if not r["answerable"] or r["first_relevant_rank"] != 1)
    lo, hi = wilson(top1, len(ans))
    return {
        "recall@1": round(top1 / len(ans), 4), "recall@1_ci95": [round(lo, 3), round(hi, 3)],
        "recall@3": round(recall_at(rows, 3), 4),
        "mrr@5": round(mrr_at(rows, 5), 4),
        "unanswerable_correctly_abstained": f"{abst}/{len(una)}",
        "answerable_wrongly_abstained": f"{false_abst}/{len(ans)}",
        "answers_shown": f"{len(shown)}/{len(rows)}",
        "wrong_or_out_of_scope_answers_shown": f"{bad}/{len(shown)}",
        "safety_loss_per_query": round(sum(loss(r, tau) for r in rows) / len(rows), 4),
        "end_to_end_accuracy": round(e2e / len(rows), 4),
        "end_to_end_ci95": [round(x, 3) for x in wilson(e2e, len(rows))],
    }


report = {
    "metrics": {
        "recall@1": "answerable queries whose top-ranked passage is labelled relevant (no abstention)",
        "mrr@5": "mean reciprocal rank of the first relevant passage within the top 5",
        "safety_loss_per_query": f"answerable: 0 if top-1 relevant, {FALSE_ANSWER_COST} if a wrong passage is shown, 1 if the system needlessly abstains; unanswerable: {FALSE_ANSWER_COST} if answered, 0 if abstained",
        "end_to_end_accuracy": "answerable: top-1 relevant AND the system answers; unanswerable: the system abstains",
    },
    "note": f"Queries were written by the project author(s), not by independent users, against a {len(passages)}-passage corpus. "
            "Treat results as a regression benchmark for this corpus, not a general accuracy claim.",
    "test_set_sha256": hashlib.sha256((EVAL_DIR / "queries_test.json").read_bytes()).hexdigest(),
    "corpus_passages": len(passages),
    "corpus_sources": len({p.url for p in passages}),
    "config": cfg,
}

all_rows = {}
for split, fname in [("dev", "queries_dev.json"), ("test", "queries_test.json")]:
    queries = load_queries(fname)
    report[split] = {"n_queries": len(queries), "n_answerable": sum(q["answerable"] for q in queries), "systems": {}}
    all_rows[split] = {}
    for name in ORDER:
        rows = rows_for(systems[name], passages, queries)
        all_rows[split][name] = rows
        report[split]["systems"][name] = {"tau": round(cfg["tau"][name], 4), **summarize(rows, cfg["tau"][name])}

# Paired bootstrap on the test split: hybrid minus each baseline.
test_rows = all_rows["test"]
report["test"]["hybrid_minus_baseline"] = {}
for base in ["keyword-overlap", "bm25-plain", "bm25f-stem"]:
    a1 = [1.0 if r["first_relevant_rank"] == 1 else 0.0 for r in test_rows["hybrid"] if r["answerable"]]
    b1 = [1.0 if r["first_relevant_rank"] == 1 else 0.0 for r in test_rows[base] if r["answerable"]]
    d, lo, hi = paired_bootstrap(a1, b1)
    a2 = [1.0 if end_to_end_correct(r, cfg["tau"]["hybrid"]) else 0.0 for r in test_rows["hybrid"]]
    b2 = [1.0 if end_to_end_correct(r, cfg["tau"][base]) else 0.0 for r in test_rows[base]]
    d2, lo2, hi2 = paired_bootstrap(a2, b2)
    report["test"]["hybrid_minus_baseline"][base] = {
        "recall@1_diff": round(d, 4), "recall@1_diff_ci95": [round(lo, 3), round(hi, 3)],
        "end_to_end_diff": round(d2, 4), "end_to_end_diff_ci95": [round(lo2, 3), round(hi2, 3)],
    }

# Failure analysis for the shipped system (hybrid) on test.
queries = {q["id"]: q for q in load_queries("queries_test.json")}
tau = cfg["tau"]["hybrid"]
fails = []
for r in test_rows["hybrid"]:
    if not end_to_end_correct(r, tau):
        q = queries[r["id"]]
        fails.append({
            "id": r["id"], "question": q["question"], "expected": q["relevant_passage_ids"],
            "top1": r["top1"], "confidence": round(r["conf"], 3), "answered": r["conf"] >= tau,
        })
report["test"]["hybrid_failures"] = fails

(EVAL_DIR / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

for split in ("dev", "test"):
    s = report[split]
    print(f"\n== {split}: {s['n_queries']} queries ({s['n_answerable']} answerable) ==")
    print(f"{'system':16} {'R@1':>6} {'R@3':>6} {'MRR@5':>6} {'abstain-ok':>11} {'false-abst':>11} {'e2e':>6}")
    for name in ORDER:
        m = s["systems"][name]
        print(f"{name:16} {m['recall@1']:6.3f} {m['recall@3']:6.3f} {m['mrr@5']:6.3f} "
              f"{m['unanswerable_correctly_abstained']:>11} {m['answerable_wrongly_abstained']:>11} {m['end_to_end_accuracy']:6.3f}")
print("\nhybrid minus baseline (test, paired bootstrap):")
for k, v in report["test"]["hybrid_minus_baseline"].items():
    print(f"  vs {k:16} R@1 {v['recall@1_diff']:+.3f} {v['recall@1_diff_ci95']}   e2e {v['end_to_end_diff']:+.3f} {v['end_to_end_diff_ci95']}")
print(f"\nhybrid failures on test: {len(fails)}")
for f in fails:
    print(" -", f["id"], f["question"], "| expected", f["expected"], "| got", f["top1"], "| conf", f["confidence"], "| answered" if f["answered"] else "| abstained")
