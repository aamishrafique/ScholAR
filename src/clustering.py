"""
K-Means clustering of retrieval results using TF-IDF features.

Used for faceted browsing: top results are grouped into topic clusters
with auto-generated labels (top TF-IDF terms per cluster).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def result_text(result: dict) -> str:
    """Concatenate title + abstract/text for a retrieval hit."""
    paper = result.get("paper", result)
    title = paper.get("title", "")
    body = paper.get("abstract", paper.get("text", ""))
    return f"{title} {body}".strip()


class ResultClusterer:
    """Cluster ranked results with K-Means on TF-IDF vectors."""

    DEFAULT_K_RANGE = [3, 4, 5, 6]

    def __init__(self, k_range: list[int] | None = None, random_state: int = 42):
        self.k_range = k_range or self.DEFAULT_K_RANGE
        self.random_state = random_state

    def cluster_results(
        self,
        results: list,
        n_clusters: int | None = None,
        top_terms: int = 3,
    ) -> dict:
        """
        Cluster *results* and return assignments plus human-readable labels.

        If *n_clusters* is None, pick k from *k_range* via silhouette score.
        """
        texts = [result_text(r) for r in results]
        n_docs = len(texts)

        if n_docs == 0:
            return self._empty_output()

        if n_docs == 1:
            return {
                "n_clusters": 1,
                "assignments": np.array([0]),
                "cluster_labels": {0: texts[0][:40] or "single"},
                "silhouette_scores": {},
                "silhouette": 0.0,
            }

        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        matrix = vectorizer.fit_transform(texts)

        if n_clusters is None:
            n_clusters, silhouette_scores = self._select_k(matrix)
        else:
            n_clusters = min(n_clusters, n_docs)
            silhouette_scores = self._score_all_k(matrix)

        kmeans = KMeans(
            n_clusters=n_clusters, random_state=self.random_state, n_init=10
        )
        assignments = kmeans.fit_predict(matrix)

        cluster_labels = self._label_clusters(
            vectorizer, kmeans, top_terms
        )

        sil = 0.0
        if len(set(assignments)) > 1 and n_docs > n_clusters:
            sil = float(silhouette_score(matrix, assignments))

        return {
            "n_clusters": n_clusters,
            "assignments": assignments,
            "cluster_labels": cluster_labels,
            "silhouette_scores": silhouette_scores,
            "silhouette": sil,
        }

    def _select_k(self, matrix) -> tuple[int, dict[int, float]]:
        scores = self._score_all_k(matrix)
        if not scores:
            return 1, scores
        best_k = max(scores, key=scores.get)
        return best_k, scores

    def _score_all_k(self, matrix) -> dict[int, float]:
        n_samples = matrix.shape[0]
        scores = {}
        for k in self.k_range:
            if k >= n_samples or k < 2:
                continue
            labels = KMeans(
                n_clusters=k, random_state=self.random_state, n_init=10
            ).fit_predict(matrix)
            if len(set(labels)) < 2:
                continue
            scores[k] = float(silhouette_score(matrix, labels))
        return scores

    def _label_clusters(
        self,
        vectorizer: TfidfVectorizer,
        kmeans: KMeans,
        top_terms: int,
    ) -> dict[int, str]:
        feature_names = vectorizer.get_feature_names_out()
        labels = {}
        for cluster_id in range(kmeans.n_clusters):
            center = kmeans.cluster_centers_[cluster_id]
            top_indices = center.argsort()[::-1][:top_terms]
            terms = [feature_names[i] for i in top_indices if center[i] > 0]
            labels[cluster_id] = (
                ", ".join(terms) if terms else f"Cluster {cluster_id + 1}"
            )
        return labels

    def _empty_output(self) -> dict:
        return {
            "n_clusters": 0,
            "assignments": np.array([]),
            "cluster_labels": {},
            "silhouette_scores": {},
            "silhouette": 0.0,
        }


def group_results_by_cluster(results: list, cluster_output: dict) -> dict[int, list]:
    """Group retrieval hits by cluster id (preserving rank order within each)."""
    groups: dict[int, list] = {}
    assignments = cluster_output.get("assignments", [])
    for idx, result in enumerate(results):
        if idx < len(assignments):
            cid = int(assignments[idx])
            groups.setdefault(cid, []).append(result)
    return groups
