"""Run a transparent retrieval evaluation for FitSage.

This measures top-1 source retrieval accuracy. Add a held-out set and a
human-reviewed answer-faithfulness rubric before making a resume-quality claim.
"""
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.demo_sources import SOURCES
from app.services.retrieval import DemoRetriever, tokens


def keyword_retrieve(question: str) -> Optional[str]:
    query = tokens(question)
    scored = []
    for source in SOURCES:
        score = len(query & tokens(source["text"]))
        scored.append((score, source["id"]))
    return max(scored, default=(0, None))[1]


def main() -> None:
    cases = json.loads((ROOT / "evaluation" / "queries.json").read_text())
    retriever = DemoRetriever()
    rows = []
    for case in cases:
        expected = set(case["expected_source_ids"])
        rag_result = retriever.retrieve(case["question"], limit=1)
        rag_id = rag_result[0].id if rag_result else None
        keyword_id = keyword_retrieve(case["question"])
        rows.append({
            "id": case["id"], "question": case["question"], "expected": sorted(expected),
            "rag_result": rag_id, "keyword_result": keyword_id,
            "rag_correct": rag_id in expected, "keyword_correct": keyword_id in expected,
        })
    total = len(rows)
    report = {
        "metric": "top-1 source retrieval accuracy",
        "sample_size": total,
        "rag_accuracy": sum(row["rag_correct"] for row in rows) / total,
        "keyword_accuracy": sum(row["keyword_correct"] for row in rows) / total,
        "rows": rows,
    }
    output = ROOT / "evaluation" / "results.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output.relative_to(ROOT)}")
    print(f"RAG retrieval accuracy: {report['rag_accuracy']:.1%}")
    print(f"Keyword retrieval accuracy: {report['keyword_accuracy']:.1%}")


if __name__ == "__main__":
    main()
