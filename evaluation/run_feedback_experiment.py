"""
Experiment 2 — Relevance feedback gain (Week 3).

Compares hybrid retrieval before vs. after one round of pseudo-Rocchio
feedback on a sample of SCIDOCS queries.
"""

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
    after_results = {}

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
        refined_hits, _ = retriever.apply_pseudo_rocchio(
            query_vector,
            hybrid_hits,
            top_k_relevant=PSEUDO_RELEVANT,
            top_k=TOP_K,
        )
        after_results[qid] = results_to_beir(refined_hits)

    sampled_qrels = {qid: qrels[qid] for qid in sampled if qid in qrels}
    evaluator = EvaluateRetrieval()

    ndcg_b, map_b, _, prec_b = evaluator.evaluate(sampled_qrels, before_results, [5, 10])
    mrr_b = EvaluateRetrieval.evaluate_custom(
        sampled_qrels, before_results, [10], metric="mrr"
    )

    ndcg_a, map_a, _, prec_a = evaluator.evaluate(sampled_qrels, after_results, [5, 10])
    mrr_a = EvaluateRetrieval.evaluate_custom(
        sampled_qrels, after_results, [10], metric="mrr"
    )

    print(f"\n{'='*60}")
    print("  Experiment 2 — Rocchio Feedback Gain (SCIDOCS)")
    print(f"{'='*60}")
    print(f"  Queries sampled : {len(sampled)}")
    print(f"  Pseudo-relevant : top {PSEUDO_RELEVANT} hybrid hits")
    print(f"  {'Metric':<20} {'Before':>12} {'After':>12} {'Delta':>10}")
    print(f"  {'-'*56}")

    rows = [
        ("nDCG@10", ndcg_b["NDCG@10"], ndcg_a["NDCG@10"]),
        ("MAP@10", map_b["MAP@10"], map_a["MAP@10"]),
        ("P@5", prec_b["P@5"], prec_a["P@5"]),
        ("MRR@10", mrr_b["MRR@10"], mrr_a["MRR@10"]),
    ]
    for name, before, after in rows:
        delta = after - before
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<20} {before:>12.4f} {after:>12.4f} {sign}{delta:>9.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_feedback_experiment()
