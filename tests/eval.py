"""
eval.py — measure the system against ground truth (brief §11, Day 7).

"Build a test set, don't trust vibes." We score three things over the
eval_queries.json pairs:

  1. Retrieval precision @ k  — was the expected page in the top-k results?
     (The single most important metric: if the right page isn't retrieved,
      no amount of LLM cleverness can produce a correct grounded answer.)
  2. Modality routing          — when the answer lives in an image, did an
     image element appear in the top-k?
  3. Answer-contains           — does the produced answer include the expected
     substring? (In free/extractive mode this measures whether the key fact is
     present in the retrieved evidence; with an LLM key it measures the written
     answer. Case-insensitive.)

Run:
    python tests/eval.py            # k=5, no rerank
    python tests/eval.py --k 5 --rerank
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from retrieval.retriever import get_retriever
from synthesis.answerer import ask

EVAL_PATH = Path(__file__).parent / "eval_queries.json"


def run_eval(k: int = 5, rerank: bool = False) -> None:
    cases = json.loads(EVAL_PATH.read_text())
    retriever = get_retriever()

    hits_at_k = 0          # a page containing the answer is in top-k
    answer_ok = 0          # final answer contains the expected value

    print(f"\nEvaluating {len(cases)} queries (k={k}, rerank={rerank}, "
          f"providers: emb={config.EMBEDDING_PROVIDER}/synth={config.SYNTHESIS_PROVIDER})\n")
    print(f"{'pg?':>4} {'ans?':>5}  query")
    print("-" * 72)

    for c in cases:
        doc_id = Path(c["doc"]).stem  # scope each query to its source document
        results = retriever.search(c["query"], k=k, rerank=rerank, doc_id=doc_id)
        pages = [int(r["metadata"]["page"]) for r in results]

        # 1. retrieval precision: any page that contains the answer is in top-k
        page_hit = any(p in pages for p in c["expected_pages"])
        hits_at_k += page_hit

        # 2. answer correctness (case-insensitive substring)
        res = ask(c["query"], k=k, rerank=rerank, doc_id=doc_id)
        haystack = res.answer.lower()
        ans_hit = all(s.lower() in haystack for s in c.get("expected_answer_contains", []))
        answer_ok += ans_hit

        def mark(b):
            return " ✓ " if b else " ✗ "
        print(f"{mark(page_hit)} {mark(ans_hit)}  {c['query'][:58]}")

    n = len(cases)
    print("-" * 72)
    print(f"\nRetrieval precision @ {k} : {hits_at_k}/{n}  = {hits_at_k / n:.0%}")
    print(f"Answer accuracy           : {answer_ok}/{n}  = {answer_ok / n:.0%}")
    print("\nBrief targets: >80% retrieval precision, >70% answer correctness.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval + answers against ground truth.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()
    run_eval(k=args.k, rerank=args.rerank)


if __name__ == "__main__":
    main()
