import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from retriever import ScholARRetriever
from rocchio import RocchioFeedback

PROCESSED_PATH = "data/processed/cs_papers.pkl"
BM25_INDEX_PATH = "indexes/bm25/bm25_index.pkl"
BM25_IDS_PATH = "indexes/bm25/paper_ids.pkl"
FAISS_INDEX_PATH = "indexes/faiss/faiss_index.bin"
FAISS_IDS_PATH = "indexes/faiss/paper_ids.pkl"
TOP_K = 5


def print_results(label: str, results: list):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    for rank, result in enumerate(results, start=1):
        paper = result["paper"]
        title = paper.get("title", "N/A")[:70]
        url = paper.get("url", "")
        score = result["score"]
        print(f"  {rank}. [{score:.4f}] {title}")
        print(f"       {url}")
    print()


def demo_retrieval(retriever: ScholARRetriever, query: str):
    print(f"\n{'#'*65}")
    print(f'  Query: "{query}"')
    print(f"{'#'*65}")

    bm25_results = retriever.search_bm25(query, top_k=TOP_K)
    query_vector = retriever.encode_query(query)
    faiss_results = retriever.search_faiss(query_vector, top_k=TOP_K)
    hybrid_results = retriever.search_hybrid(query, top_k=TOP_K)

    print_results("BM25 — Lexical", bm25_results)
    print_results("FAISS + SBERT — Semantic", faiss_results)
    print_results("Hybrid — RRF Fusion", hybrid_results)

    return hybrid_results, query_vector


def demo_pseudo_feedback(
    retriever: ScholARRetriever,
    initial_results: list,
    original_query_vector: np.ndarray,
    top_k_relevant: int = 3,
):
    rocchio = RocchioFeedback(alpha=1.0, beta=0.75, gamma=0.25)

    updated_query_vector = rocchio.pseudo_feedback(
        query_vector=original_query_vector,
        retriever=retriever,
        initial_results=initial_results,
        top_k_relevant=top_k_relevant,
    )

    refined_results = retriever.search_faiss(updated_query_vector, top_k=TOP_K)
    print_results(
        f"After Pseudo Rocchio Feedback (top {top_k_relevant} treated as relevant)",
        refined_results,
    )
    return updated_query_vector


def demo_explicit_feedback(
    retriever: ScholARRetriever,
    initial_results: list,
    original_query_vector: np.ndarray,
    relevant_indices: list,
    nonrelevant_indices: list,
):
    """
    Simulates explicit user feedback.

    relevant_indices    : positions in initial_results marked relevant (0-indexed)
    nonrelevant_indices : positions in initial_results marked not relevant (0-indexed)
    """
    rocchio = RocchioFeedback(alpha=1.0, beta=0.75, gamma=0.25)

    relevant_vectors = []
    for idx in relevant_indices:
        doc_id = initial_results[idx]["id"]
        vec = retriever.get_embedding_by_id(doc_id)
        if vec is not None:
            relevant_vectors.append(vec)

    nonrelevant_vectors = []
    for idx in nonrelevant_indices:
        doc_id = initial_results[idx]["id"]
        vec = retriever.get_embedding_by_id(doc_id)
        if vec is not None:
            nonrelevant_vectors.append(vec)

    updated_query_vector = rocchio.update_query(
        query_vector=original_query_vector,
        relevant_vectors=relevant_vectors,
        nonrelevant_vectors=nonrelevant_vectors,
    )

    refined_results = retriever.search_faiss(updated_query_vector, top_k=TOP_K)
    print_results(
        f"After Explicit Rocchio Feedback "
        f"(relevant: {relevant_indices}, non-relevant: {nonrelevant_indices})",
        refined_results,
    )


def main():
    retriever = ScholARRetriever()
    retriever.load_arxiv_indexes(
        PROCESSED_PATH, BM25_INDEX_PATH, BM25_IDS_PATH, FAISS_INDEX_PATH, FAISS_IDS_PATH
    )

    query = "graph neural networks for node classification"
    initial_results, query_vector = demo_retrieval(retriever, query)

    print("\n--- Relevance Feedback Demo ---")

    demo_pseudo_feedback(retriever, initial_results, query_vector, top_k_relevant=3)

    demo_explicit_feedback(
        retriever,
        initial_results,
        query_vector,
        relevant_indices=[0, 1],
        nonrelevant_indices=[4],
    )


if __name__ == "__main__":
    main()
