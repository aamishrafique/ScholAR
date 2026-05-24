import quiet_imports  # noqa: F401 — before transformers / sentence-transformers

import pickle
import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from preprocess import tokenize_and_stem

MODEL_NAME = "all-MiniLM-L6-v2"


class ScholARRetriever:

    def __init__(self, model_name=MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.bm25 = None
        self.faiss_index = None
        self.papers = []
        self.bm25_ids = []
        self.faiss_ids = []
        self.faiss_id_to_idx = {}
        self.paper_lookup = {}

    def load_arxiv_indexes(
        self,
        processed_path,
        bm25_index_path,
        bm25_ids_path,
        faiss_index_path,
        faiss_ids_path,
    ):
        print("Loading arXiv papers...")
        with open(processed_path, "rb") as f:
            self.papers = pickle.load(f)
        self.paper_lookup = {p["id"]: p for p in self.papers}

        with open(bm25_index_path, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(bm25_ids_path, "rb") as f:
            self.bm25_ids = pickle.load(f)

        self.faiss_index = faiss.read_index(faiss_index_path)
        with open(faiss_ids_path, "rb") as f:
            self.faiss_ids = pickle.load(f)
        self.faiss_id_to_idx = {
            doc_id: idx for idx, doc_id in enumerate(self.faiss_ids)
        }

        print(f"Loaded {len(self.papers)} papers.")

    def build_from_corpus(self, documents: list, batch_size=512):
        self.papers = documents
        self.paper_lookup = {d["id"]: d for d in documents}

        print(f"Building BM25 index over {len(documents)} documents...")
        tokenized = [
            tokenize_and_stem(
                d.get("title", "")
                + " "
                + d.get("title", "")
                + " "
                + d.get("text", d.get("abstract", ""))
            )
            for d in tqdm(documents, desc="Tokenizing")
        ]
        self.bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
        self.bm25_ids = [d["id"] for d in documents]

        print("Encoding documents with Sentence-BERT...")
        texts = [
            d.get("title", "") + " " + d.get("text", d.get("abstract", ""))
            for d in documents
        ]
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i : i + batch_size]
            emb = self.model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            all_embeddings.append(emb)

        embeddings = np.vstack(all_embeddings).astype("float32")
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        self.faiss_ids = [d["id"] for d in documents]
        self.faiss_id_to_idx = {
            doc_id: idx for idx, doc_id in enumerate(self.faiss_ids)
        }

        print(f"Indexes built. {self.faiss_index.ntotal} vectors stored.")

    def encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        return vec.astype("float32")

    def get_embedding_by_id(self, doc_id: str) -> np.ndarray:
        idx = self.faiss_id_to_idx.get(doc_id)
        if idx is None:
            return None
        return self.faiss_index.reconstruct(idx).reshape(1, -1)

    def search_bm25(self, query: str, top_k: int = 10) -> list:
        tokens = tokenize_and_stem(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            {
                "id": self.bm25_ids[i],
                "paper": self.paper_lookup.get(self.bm25_ids[i], {}),
                "score": float(scores[i]),
            }
            for i in top_indices
        ]

    def search_faiss(self, query_vector: np.ndarray, top_k: int = 10) -> list:
        scores, indices = self.faiss_index.search(query_vector, top_k)
        return [
            {
                "id": self.faiss_ids[idx],
                "paper": self.paper_lookup.get(self.faiss_ids[idx], {}),
                "score": float(score),
                "faiss_idx": int(idx),
            }
            for idx, score in zip(indices[0], scores[0])
        ]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        rrf_k: int = 60,
        candidate_k: int = 200,
        bm25_weight: float = 0.3,
        faiss_weight: float = 0.7,
    ) -> list:
        """
        Weighted Reciprocal Rank Fusion over BM25 + FAISS candidate lists.

        bm25_weight / faiss_weight control each list's contribution.
        Default (0.3 / 0.7) favours semantic retrieval, which outperforms
        lexical retrieval on the SCIDOCS benchmark.
        """
        bm25_results = self.search_bm25(query, top_k=candidate_k)
        query_vector = self.encode_query(query)
        faiss_results = self.search_faiss(query_vector, top_k=candidate_k)

        rrf_scores = {}

        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + bm25_weight / (
                rrf_k + rank
            )

        for rank, result in enumerate(faiss_results, start=1):
            doc_id = result["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + faiss_weight / (
                rrf_k + rank
            )

        top_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

        return [
            {
                "id": doc_id,
                "paper": self.paper_lookup.get(doc_id, {}),
                "score": rrf_scores[doc_id],
            }
            for doc_id in top_ids
        ]

    def apply_rocchio_feedback(
        self,
        query_vector: np.ndarray,
        results: list,
        relevant_indices: list[int],
        nonrelevant_indices: list[int] | None = None,
        top_k: int = 10,
    ) -> tuple[list, np.ndarray]:
        """
        Update the dense query vector with explicit relevance judgments
        and re-query FAISS.
        """
        from rocchio import RocchioFeedback

        rocchio = RocchioFeedback()
        relevant_vectors = []
        for idx in relevant_indices:
            vec = self.get_embedding_by_id(results[idx]["id"])
            if vec is not None:
                relevant_vectors.append(vec)

        nonrelevant_vectors = []
        for idx in nonrelevant_indices or []:
            vec = self.get_embedding_by_id(results[idx]["id"])
            if vec is not None:
                nonrelevant_vectors.append(vec)

        updated = rocchio.update_query(
            query_vector, relevant_vectors, nonrelevant_vectors or None
        )
        return self.search_faiss(updated, top_k=top_k), updated

    def apply_pseudo_rocchio(
        self,
        query_vector: np.ndarray,
        results: list,
        top_k_relevant: int = 3,
        top_k: int = 10,
    ) -> tuple[list, np.ndarray]:
        """Pseudo-relevance feedback: top results treated as relevant."""
        from rocchio import RocchioFeedback

        rocchio = RocchioFeedback()
        updated = rocchio.pseudo_feedback(
            query_vector, self, results, top_k_relevant=top_k_relevant
        )
        return self.search_faiss(updated, top_k=top_k), updated
