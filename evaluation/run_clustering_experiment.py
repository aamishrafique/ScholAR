"""
Experiment 3 — Clustering quality via silhouette score (Week 3).

For each sampled SCIDOCS query, retrieves the top-20 hybrid results,
clusters them with K-Means on TF-IDF features, and reports mean
silhouette scores for k ∈ {3, 4, 5, 6}.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from clustering import ResultClusterer
from retriever import ScholARRetriever

SCIDOCS_FOLDER = "evaluation/scidocs/scidocs"
SCIDOCS_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip"
)
SAMPLE_SIZE = 20
SEED = 42
CLUSTER_POOL = 20
RRF_K = 10
CANDIDATE_K = 200
K_RANGE = [3, 4, 5, 6]


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


def load_scidocs():
    save_dir = os.path.dirname(SCIDOCS_FOLDER)
    if not os.path.exists(SCIDOCS_FOLDER):
        print("Downloading SCIDOCS...")
        os.makedirs(save_dir, exist_ok=True)
        util.download_and_unzip(SCIDOCS_URL, save_dir)
    return GenericDataLoader(data_folder=SCIDOCS_FOLDER).load(split="test")


def run_clustering_experiment():
    corpus, queries, _ = load_scidocs()
    documents = corpus_to_documents(corpus)
    retriever = ScholARRetriever()
    retriever.build_from_corpus(documents)

    clusterer = ResultClusterer(k_range=K_RANGE)

    query_ids = list(queries.keys())
    random.seed(SEED)
    sampled = random.sample(query_ids, min(SAMPLE_SIZE, len(query_ids)))
    print(f"Clustering top-{CLUSTER_POOL} hybrid results for {len(sampled)} queries")

    per_k_scores: dict[int, list[float]] = {k: [] for k in K_RANGE}
    best_k_counts: dict[int, int] = {k: 0 for k in K_RANGE}

    for qid in sampled:
        hits = retriever.search_hybrid(
            queries[qid],
            top_k=CLUSTER_POOL,
            rrf_k=RRF_K,
            candidate_k=CANDIDATE_K,
        )
        output = clusterer.cluster_results(hits)
        for k, score in output["silhouette_scores"].items():
            per_k_scores[k].append(score)
        best_k = output["n_clusters"]
        if best_k in best_k_counts:
            best_k_counts[best_k] += 1

    print(f"\n{'='*60}")
    print("  Experiment 3 — Clustering Quality (SCIDOCS)")
    print(f"{'='*60}")
    print(f"  Queries sampled : {len(sampled)}")
    print(f"  Results per query: {CLUSTER_POOL}")
    print(f"  k candidates    : {K_RANGE}")
    print(f"\n  {'k':<6} {'Mean Silhouette':>18} {'Std':>10}")
    print(f"  {'-'*36}")

    best_mean_k = None
    best_mean = -1.0
    for k in K_RANGE:
        scores = per_k_scores[k]
        if not scores:
            continue
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance**0.5
        marker = ""
        if mean > best_mean:
            best_mean = mean
            best_mean_k = k
            marker = " <-- best mean"
        print(f"  {k:<6} {mean:>18.4f} {std:>10.4f}{marker}")

    print(f"\n  Silhouette-selected k (per query):")
    for k in K_RANGE:
        print(f"    k={k}: {best_k_counts[k]} queries")
    print(f"\n  Recommended k (highest mean silhouette): {best_mean_k}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_clustering_experiment()
