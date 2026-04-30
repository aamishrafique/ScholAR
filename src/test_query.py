import pickle
import faiss
from sentence_transformers import SentenceTransformer
from preprocess import PROCESSED_PATH, tokenize_and_stem


BM25_INDEX_PATH = "indexes/bm25/bm25_index.pkl"
BM25_IDS_PATH = "indexes/bm25/paper_ids.pkl"
FAISS_INDEX_PATH = "indexes/faiss/faiss_index.bin"
FAISS_IDS_PATH = "indexes/faiss/paper_ids.pkl"


def test_bm25(query: str, top_k=5):
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_IDS_PATH, "rb") as f:
        paper_ids = pickle.load(f)
    with open(PROCESSED_PATH, "rb") as f:
        papers = pickle.load(f)

    paper_lookup = {p["id"]: p for p in papers}
    tokenized_query = tokenize_and_stem(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = scores.argsort()[::-1][:top_k]

    print(f"\n=== BM25 results for: '{query}' ===")
    for rank, idx in enumerate(top_indices, 1):
        paper = paper_lookup[paper_ids[idx]]
        print(f"{rank}. [{round(scores[idx], 3)}] {paper['title']}")
        print(f"   {paper['url']}\n")


def test_faiss(query: str, top_k=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_IDS_PATH, "rb") as f:
        paper_ids = pickle.load(f)
    with open(PROCESSED_PATH, "rb") as f:
        papers = pickle.load(f)

    paper_lookup = {p["id"]: p for p in papers}
    query_vector = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vector, top_k)

    print(f"\n=== FAISS (SBERT) results for: '{query}' ===")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        paper = paper_lookup[paper_ids[idx]]
        print(f"{rank}. [{round(float(score), 3)}] {paper['title']}")
        print(f"   {paper['url']}\n")


if __name__ == "__main__":
    query = "transformer models for natural language processing"
    test_bm25(query)
    test_faiss(query)
