import numpy as np


class RocchioFeedback:
    """
    Rocchio relevance feedback over dense query vectors.

    Formula:
        q' = alpha * q
           + beta  * mean(relevant_vectors)
           - gamma * mean(non_relevant_vectors)

    The updated vector is L2-normalized before being used
    to re-query the FAISS index.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.25):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def update_query(
        self,
        query_vector: np.ndarray,
        relevant_vectors: list,
        nonrelevant_vectors: list = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        query_vector        : shape (1, D) — original query embedding
        relevant_vectors    : list of np.ndarray shape (1, D)
        nonrelevant_vectors : list of np.ndarray shape (1, D), optional

        Returns
        -------
        updated query vector, shape (1, D), L2-normalized
        """
        updated = self.alpha * query_vector

        if relevant_vectors:
            relevant_matrix = np.vstack(relevant_vectors)
            updated = updated + self.beta * relevant_matrix.mean(axis=0, keepdims=True)

        if nonrelevant_vectors:
            nonrelevant_matrix = np.vstack(nonrelevant_vectors)
            updated = updated - self.gamma * nonrelevant_matrix.mean(
                axis=0, keepdims=True
            )

        updated = updated / (np.linalg.norm(updated) + 1e-10)
        return updated.astype("float32")

    def pseudo_feedback(
        self,
        query_vector: np.ndarray,
        retriever,
        initial_results: list,
        top_k_relevant: int = 3,
    ) -> np.ndarray:
        """
        Automatic relevance feedback — treats the top-k results as
        relevant without requiring explicit user input.

        Parameters
        ----------
        query_vector    : original query embedding
        retriever       : ScholARRetriever instance (for get_embedding_by_id)
        initial_results : ranked list from any search method
        top_k_relevant  : number of top results to treat as relevant

        Returns
        -------
        updated query vector, shape (1, D), L2-normalized
        """
        relevant_ids = [r["id"] for r in initial_results[:top_k_relevant]]
        relevant_vectors = []

        for doc_id in relevant_ids:
            vec = retriever.get_embedding_by_id(doc_id)
            if vec is not None:
                relevant_vectors.append(vec)

        if not relevant_vectors:
            print("Warning: no embeddings found for pseudo-feedback documents.")
            return query_vector

        return self.update_query(query_vector, relevant_vectors)
