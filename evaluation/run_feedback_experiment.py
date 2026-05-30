"""
Experiment 2 — Relevance feedback gain (Week 3).

Compares hybrid retrieval against one round of Rocchio feedback on a sample of
SCIDOCS queries, using two feedback strategies:

* Explicit feedback — judged-relevant docs among the retrieved hits (from qrels)
  are marked relevant, the rest non-relevant. Simulates a user who marks the
  results they actually see. This is the headline result for the proposal's
  "measurable query-expansion gains" objective.
* Pseudo feedback — the top-3 hybrid hits are blindly assumed relevant, with no
  user input. Included as a contrast: it degrades on a low-precision, citation-
  based corpus like SCIDOCS.

Results are written to evaluation/results/feedback_report.json.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from retriever import ScholARRetriever

SCIDOCS_FOLDER = "evaluation/scidocs/scidocs"
SCIDOCS_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SAMPLE_SIZE = 20
SEED = 42
TOP_K = 10
CANDIDATE_K = 200
RRF_K = 10
PSEUDO_RELEVANT = 3


def corpus_to_documents(corpus: dict) -> list:
    return [
        {
            "id": doc_id,
            "title": doc.get("title", ""),
            "text": doc.get("text", ""),
            "abstract": doc.get("text", ""),
        }
        for doc_id, doc in corpus.items()
    ]


def results_to_beir(ranked: list) -> dict[str, float]:
    return {hit["id"]: hit["score"] for hit in ranked}


def load_scidocs():
    save_dir = os.path.dirname(SCIDOCS_FOLDER)
    if not os.path.exists(SCIDOCS_FOLDER):
        print("Downloading SCIDOCS...")
        os.makedirs(save_dir, exist_ok=True)
        util.download_and_unzip(SCIDOCS_URL, save_dir)
    return GenericDataLoader(data_folder=SCIDOCS_FOLDER).load(split="test")


def evaluate_run(evaluator: EvaluateRetrieval, qrels: dict, results: dict) -> dict:
    ndcg, map_, _, precision = evaluator.evaluate(qrels, results, [5, 10])
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, [10], metric="mrr")
    return {
        "nDCG@10": ndcg["NDCG@10"],
        "MAP@10": map_["MAP@10"],
        "P@5": precision["P@5"],
        "MRR@10": mrr["MRR@10"],
    }


def run_feedback_experiment():
    corpus, queries, qrels = load_scidocs()
    print(f"SCIDOCS — corpus: {len(corpus)} | queries: {len(queries)}")

    documents = corpus_to_documents(corpus)
    retriever = ScholARRetriever()
    retriever.build_from_corpus(documents)

    query_ids = list(queries.keys())
    random.seed(SEED)
    sampled = random.sample(query_ids, min(SAMPLE_SIZE, len(query_ids)))
    print(f"Evaluating {len(sampled)} queries (seed={SEED})")

    before_results = {}
    explicit_results = {}
    pseudo_results = {}
    explicit_coverage = 0

    for qid in sampled:
        query_text = queries[qid]
        hybrid_hits = retriever.search_hybrid(
            query_text,
            top_k=TOP_K,
            rrf_k=RRF_K,
            candidate_k=CANDIDATE_K,
        )
        before_results[qid] = results_to_beir(hybrid_hits)

        query_vector = retriever.encode_query(query_text)

        # Explicit feedback: judged-relevant retrieved hits marked relevant,
        # the remaining retrieved hits marked non-relevant.
        judged = qrels.get(qid, {})
        rel_idx = [
            i for i, hit in enumerate(hybrid_hits) if judged.get(hit["id"], 0) > 0
        ]
        nonrel_idx = [
            i for i, hit in enumerate(hybrid_hits) if judged.get(hit["id"], 0) == 0
        ]
        if rel_idx:
            explicit_coverage += 1
            explicit_hits, _ = retriever.apply_rocchio_feedback(
                query_vector,
                hybrid_hits,
                relevant_indices=rel_idx,
                nonrelevant_indices=nonrel_idx,
                top_k=TOP_K,
            )
            explicit_results[qid] = results_to_beir(explicit_hits)
        else:
            # No judged-relevant doc retrieved — nothing for the user to mark.
            explicit_results[qid] = before_results[qid]

        # Pseudo feedback: top-3 hybrid hits blindly assumed relevant.
        pseudo_hits, _ = retriever.apply_pseudo_rocchio(
            query_vector,
            hybrid_hits,
            top_k_relevant=PSEUDO_RELEVANT,
            top_k=TOP_K,
        )
        pseudo_results[qid] = results_to_beir(pseudo_hits)

    sampled_qrels = {qid: qrels[qid] for qid in sampled if qid in qrels}
    evaluator = EvaluateRetrieval()

    before = evaluate_run(evaluator, sampled_qrels, before_results)
    explicit = evaluate_run(evaluator, sampled_qrels, explicit_results)
    pseudo = evaluate_run(evaluator, sampled_qrels, pseudo_results)

    print(f"\n{'='*72}")
    print("  Experiment 2 — Rocchio Feedback Gain (SCIDOCS)")
    print(f"{'='*72}")
    print(f"  Queries sampled        : {len(sampled)}")
    print(f"  Explicit feedback used : {explicit_coverage}/{len(sampled)} queries "
          "(had >=1 judged-relevant hit in top-10)")
    print(f"  Pseudo-relevant        : top {PSEUDO_RELEVANT} hybrid hits")
    print(
        f"\n  {'Metric':<12} {'Before':>10} {'Explicit':>10} {'(Δ)':>9}"
        f" {'Pseudo':>10} {'(Δ)':>9}"
    )
    print(f"  {'-'*64}")

    for metric in ("nDCG@10", "MAP@10", "P@5", "MRR@10"):
        b = before[metric]
        e = explicit[metric]
        p = pseudo[metric]
        de, dp = e - b, p - b
        se = "+" if de >= 0 else ""
        sp = "+" if dp >= 0 else ""
        print(
            f"  {metric:<12} {b:>10.4f} {e:>10.4f} {se}{de:>8.4f}"
            f" {p:>10.4f} {sp}{dp:>8.4f}"
        )
    print(f"{'='*72}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {
        "dataset": "scidocs",
        "queries_sampled": len(sampled),
        "seed": SEED,
        "explicit_coverage": explicit_coverage,
        "pseudo_relevant": PSEUDO_RELEVANT,
        "rrf_k": RRF_K,
        "before": before,
        "explicit": explicit,
        "pseudo": pseudo,
    }
    json_path = os.path.join(RESULTS_DIR, "feedback_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    run_feedback_experiment()
