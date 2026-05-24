import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from retriever import ScholARRetriever

DATASET_CONFIGS = {
    "scidocs": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
        "folder": "evaluation/scidocs/scidocs",
        "split": "test",
    },
    "trec-covid": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
        "folder": "evaluation/trec-covid/trec-covid",
        "split": "test",
    },
}
CANDIDATE_K = 200
TOP_K = 10


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


def run_bm25(retriever: ScholARRetriever, queries: dict, top_k: int) -> dict:
    return {
        query_id: {
            hit["id"]: hit["score"]
            for hit in retriever.search_bm25(query_text, top_k=top_k)
        }
        for query_id, query_text in queries.items()
    }


def run_faiss(retriever: ScholARRetriever, queries: dict, top_k: int) -> dict:
    return {
        query_id: {
            hit["id"]: hit["score"]
            for hit in retriever.search_faiss(
                retriever.encode_query(query_text), top_k=top_k
            )
        }
        for query_id, query_text in queries.items()
    }


def run_hybrid(
    retriever: ScholARRetriever,
    queries: dict,
    top_k: int,
    rrf_k: int = 60,
    bm25_weight: float = 0.3,
    faiss_weight: float = 0.7,
) -> dict:
    return {
        query_id: {
            hit["id"]: hit["score"]
            for hit in retriever.search_hybrid(
                query_text,
                top_k=top_k,
                rrf_k=rrf_k,
                candidate_k=CANDIDATE_K,
                bm25_weight=bm25_weight,
                faiss_weight=faiss_weight,
            )
        }
        for query_id, query_text in queries.items()
    }


def print_results_table(
    label: str, ndcg: dict, map_score: dict, precision: dict, mrr: dict
):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  {'Metric':<25} {'Score':>10}")
    print(f"  {'-'*35}")
    print(f"  {'nDCG@10':<25} {ndcg.get('NDCG@10', 0):.4f}")
    print(f"  {'MAP@10':<25} {map_score.get('MAP@10', 0):.4f}")
    print(f"  {'Precision@5':<25} {precision.get('P@5', 0):.4f}")
    print(f"  {'MRR@10':<25} {mrr.get('MRR@10', 0):.4f}")
    print(f"{'='*55}")


def tune_rrf_k(
    retriever: ScholARRetriever,
    queries: dict,
    qrels: dict,
    evaluator: EvaluateRetrieval,
) -> int:
    """
    Sweeps rrf_k values with fixed weights (0.3 / 0.7) to find
    the best k for the given benchmark.
    Returns the best k value to use in the final evaluation.
    """
    print(f"\n{'='*55}")
    print("  RRF k Tuning  (bm25_weight=0.3, faiss_weight=0.7)")
    print(f"{'='*55}")
    print(f"  {'k':<10} {'nDCG@10':>10} {'MAP@10':>10} {'MRR@10':>10}")
    print(f"  {'-'*42}")

    best_k = 60
    best_ndcg = 0.0

    for k in [10, 20, 30, 60, 100]:
        results = run_hybrid(
            retriever, queries, TOP_K, rrf_k=k, bm25_weight=0.3, faiss_weight=0.7
        )
        ndcg, map_, _, _ = evaluator.evaluate(qrels, results, [10])
        mrr = EvaluateRetrieval.evaluate_custom(qrels, results, [10], metric="mrr")

        is_best = ndcg["NDCG@10"] > best_ndcg
        marker = " <-- best" if is_best else ""

        if is_best:
            best_ndcg = ndcg["NDCG@10"]
            best_k = k

        print(
            f"  {k:<10} {ndcg['NDCG@10']:>10.4f} {map_['MAP@10']:>10.4f}"
            f" {mrr['MRR@10']:>10.4f}{marker}"
        )

    print(f"\n  Best k = {best_k}  (nDCG@10 = {best_ndcg:.4f})")
    print(f"{'='*55}")
    return best_k


def evaluate_dataset(dataset_name: str, config: dict):
    print(f"\n{'#'*55}")
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"{'#'*55}")

    save_dir = os.path.dirname(config["folder"])
    if not os.path.exists(config["folder"]):
        print(f"Downloading {dataset_name}...")
        os.makedirs(save_dir, exist_ok=True)
        util.download_and_unzip(config["url"], save_dir)
    else:
        print(f"{dataset_name} already downloaded.")

    corpus, queries, qrels = GenericDataLoader(data_folder=config["folder"]).load(
        split=config["split"]
    )
    print(f"Corpus: {len(corpus)} | Queries: {len(queries)} | Qrels: {len(qrels)}")

    documents = corpus_to_documents(corpus)
    retriever = ScholARRetriever()
    retriever.build_from_corpus(documents)

    evaluator = EvaluateRetrieval()
    best_k = tune_rrf_k(retriever, queries, qrels, evaluator)

    print("\nRunning BM25...")
    bm25_results = run_bm25(retriever, queries, TOP_K)
    ndcg, map_, _, precision = evaluator.evaluate(qrels, bm25_results, [5, 10])
    mrr = EvaluateRetrieval.evaluate_custom(qrels, bm25_results, [10], metric="mrr")
    print_results_table("BM25 (Lexical)", ndcg, map_, precision, mrr)

    print("\nRunning FAISS + SBERT...")
    faiss_results = run_faiss(retriever, queries, TOP_K)
    ndcg, map_, _, precision = evaluator.evaluate(qrels, faiss_results, [5, 10])
    mrr = EvaluateRetrieval.evaluate_custom(qrels, faiss_results, [10], metric="mrr")
    print_results_table("FAISS + SBERT (Semantic)", ndcg, map_, precision, mrr)

    print(f"\nRunning Hybrid (RRF, k={best_k}, weights=0.3/0.7)...")
    hybrid_results = run_hybrid(
        retriever, queries, TOP_K, rrf_k=best_k, bm25_weight=0.3, faiss_weight=0.7
    )
    ndcg, map_, _, precision = evaluator.evaluate(qrels, hybrid_results, [5, 10])
    mrr = EvaluateRetrieval.evaluate_custom(qrels, hybrid_results, [10], metric="mrr")
    print_results_table(
        f"Hybrid — BM25 + SBERT (RRF, k={best_k})", ndcg, map_, precision, mrr
    )


def main():
    for name, config in DATASET_CONFIGS.items():
        evaluate_dataset(name, config)


if __name__ == "__main__":
    main()
