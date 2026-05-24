"""
Experiment 4 — Latency benchmarks (Week 4).

Measures average query response time for each retrieval mode on the
full arXiv CS-paper index (BM25 + FAISS). Results are written to
evaluation/results/latency_report.json and latency_report.txt.
"""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clustering import ResultClusterer
from retriever import ScholARRetriever

PROCESSED_PATH = "data/processed/cs_papers.pkl"
BM25_INDEX_PATH = "indexes/bm25/bm25_index.pkl"
BM25_IDS_PATH = "indexes/bm25/paper_ids.pkl"
FAISS_INDEX_PATH = "indexes/faiss/faiss_index.bin"
FAISS_IDS_PATH = "indexes/faiss/paper_ids.pkl"
QUERIES_FILE = os.path.join(os.path.dirname(__file__), "queries_latency.txt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

TOP_K = 10
CLUSTER_POOL = 20
CANDIDATE_K = 200
RRF_K = 60
WARMUP_QUERIES = 3


def load_queries() -> list[str]:
    with open(QUERIES_FILE, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def summarize_ms(times: list[float]) -> dict:
    if not times:
        return {"mean_ms": 0, "median_ms": 0, "p95_ms": 0, "min_ms": 0, "max_ms": 0}
    sorted_t = sorted(times)
    p95_idx = max(0, int(len(sorted_t) * 0.95) - 1)
    return {
        "mean_ms": round(statistics.mean(times) * 1000, 2),
        "median_ms": round(statistics.median(times) * 1000, 2),
        "p95_ms": round(sorted_t[p95_idx] * 1000, 2),
        "min_ms": round(min(times) * 1000, 2),
        "max_ms": round(max(times) * 1000, 2),
        "n": len(times),
    }


def bench_bm25(retriever: ScholARRetriever, query: str) -> None:
    retriever.search_bm25(query, top_k=TOP_K)


def bench_faiss(retriever: ScholARRetriever, query: str) -> None:
    qv = retriever.encode_query(query)
    retriever.search_faiss(qv, top_k=TOP_K)


def bench_hybrid(retriever: ScholARRetriever, query: str) -> None:
    retriever.search_hybrid(
        query, top_k=TOP_K, rrf_k=RRF_K, candidate_k=CANDIDATE_K
    )


def bench_hybrid_cluster(retriever: ScholARRetriever, query: str) -> None:
    hits = retriever.search_hybrid(
        query, top_k=CLUSTER_POOL, rrf_k=RRF_K, candidate_k=CANDIDATE_K
    )
    ResultClusterer().cluster_results(hits)


def bench_rocchio(retriever: ScholARRetriever, query: str) -> None:
    hybrid = retriever.search_hybrid(
        query, top_k=TOP_K, rrf_k=RRF_K, candidate_k=CANDIDATE_K
    )
    qv = retriever.encode_query(query)
    retriever.apply_pseudo_rocchio(qv, hybrid, top_k_relevant=3, top_k=TOP_K)


def run_mode(name: str, fn, retriever: ScholARRetriever, queries: list[str]) -> dict:
    times = []
    for query in queries:
        start = time.perf_counter()
        fn(retriever, query)
        times.append(time.perf_counter() - start)
    stats = summarize_ms(times)
    stats["mode"] = name
    return stats


def print_table(rows: list[dict], corpus_size: int, device: str):
    print(f"\n{'='*72}")
    print("  Experiment 4 — Latency (arXiv CS corpus)")
    print(f"{'='*72}")
    print(f"  Corpus size     : {corpus_size:,} documents")
    print(f"  Device (SBERT)  : {device}")
    print(f"  Queries timed   : {rows[0]['n'] if rows else 0}")
    print(f"  Warmup queries  : {WARMUP_QUERIES} (excluded)")
    print(f"\n  {'Mode':<28} {'Mean':>10} {'Median':>10} {'P95':>10}")
    print(f"  {'-'*60}")
    for row in rows:
        print(
            f"  {row['mode']:<28} {row['mean_ms']:>8.1f}ms"
            f" {row['median_ms']:>8.1f}ms {row['p95_ms']:>8.1f}ms"
        )
    print(f"{'='*72}\n")


def main():
    if not os.path.exists(PROCESSED_PATH):
        print(
            "Indexes not found. Run: python src/preprocess.py && "
            "python src/build_bm25.py && python src/build_faiss.py"
        )
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    retriever = ScholARRetriever()
    retriever.load_arxiv_indexes(
        PROCESSED_PATH,
        BM25_INDEX_PATH,
        BM25_IDS_PATH,
        FAISS_INDEX_PATH,
        FAISS_IDS_PATH,
    )

    queries = load_queries()
    warmup = queries[:WARMUP_QUERIES]
    timed = queries[WARMUP_QUERIES:]

    print(f"Warming up ({WARMUP_QUERIES} queries)...")
    for q in warmup:
        bench_hybrid(retriever, q)

    modes = [
        ("BM25", bench_bm25),
        ("FAISS + SBERT", bench_faiss),
        ("Hybrid (RRF)", bench_hybrid),
        ("Hybrid + K-Means cluster", bench_hybrid_cluster),
        ("Hybrid + Rocchio (pseudo)", bench_rocchio),
    ]

    rows = [run_mode(name, fn, retriever, timed) for name, fn in modes]

    report = {
        "corpus_documents": len(retriever.papers),
        "device": retriever.device,
        "queries_timed": len(timed),
        "warmup_queries": WARMUP_QUERIES,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "modes": rows,
    }

    json_path = os.path.join(RESULTS_DIR, "latency_report.json")
    txt_path = os.path.join(RESULTS_DIR, "latency_report.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_table(rows, len(retriever.papers), retriever.device)
    print(f"Saved: {json_path}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("ScholAR — Experiment 4 Latency Report\n")
        f.write(f"Corpus: {report['corpus_documents']:,} documents\n")
        f.write(f"Device: {report['device']}\n\n")
        for row in rows:
            f.write(
                f"{row['mode']}: mean={row['mean_ms']}ms "
                f"median={row['median_ms']}ms p95={row['p95_ms']}ms\n"
            )
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
