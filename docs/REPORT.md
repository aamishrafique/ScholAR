# ScholAR — Final Technical Report

**Team:** ScholAR (Aamish Rafique, Peter Spedale)  
**Course:** CSC 575 — Information Retrieval  
**Project type:** Type A + Type B (Hybrid retrieval system)

---

## 1. Introduction

Scientific literature is growing faster than researchers can manually survey it. Keyword search
misses paraphrases and related concepts; pure semantic search can surface loosely related work.
**ScholAR** addresses this by combining **lexical** (BM25) and **semantic** (Sentence-BERT +
FAISS) retrieval over the arXiv CS corpus, fusing ranked lists with **Reciprocal Rank Fusion
(RRF)**. Users can refine results via **Rocchio relevance feedback** and browse **K-Means
topic clusters** in a Streamlit interface.

**Goals:**

1. Build an end-to-end retrieval pipeline (preprocess → index → query → rank).
2. Demonstrate measurable gains from hybrid fusion and relevance feedback.
3. Support faceted browsing through unsupervised clustering of result sets.
4. Evaluate rigorously on BEIR benchmarks (SCIDOCS, TREC-COVID).

---

## 2. Related Work

**BM25** (Robertson & Zaragoza, 2009) remains the standard lexical ranker for ad-hoc retrieval,
balancing term frequency with document-length normalization.

**Dense retrieval** with bi-encoders (Reimers & Gurevych, 2019; Sentence-BERT) maps queries and
documents into a shared embedding space, capturing synonymy and paraphrase beyond exact term overlap.

**BEIR** (Thakur et al., 2021) provides heterogeneous zero-shot evaluation suites; we use
**SCIDOCS** (citation-based scientific relevance) and **TREC-COVID** (biomedical ad-hoc search).

**Reciprocal Rank Fusion** (Cormack et al., 2009) merges multiple ranked lists without score
calibration, using only rank positions—well suited to combining incompatible BM25 and cosine scores.

**Rocchio feedback** (classic relevance feedback) shifts the query vector toward relevant and away
from non-relevant document embeddings; we apply it in embedding space over FAISS.

**K-Means on TF-IDF** clusters result snippets for browsing; silhouette score guides the choice of
cluster count *k*.

---

## 3. Methodology

### 3.1 Corpus and preprocessing

- **Source:** arXiv metadata snapshot (Kaggle), filtered to `cs.*` categories.
- **Fields:** title, abstract, authors, categories, DOI, arXiv URL.
- **BM25 text:** doubled title + abstract; lowercased, punctuation stripped, stop words removed,
  Porter-stemmed.
- **Dense text:** single title + abstract (no stemming) for `all-MiniLM-L6-v2`.

### 3.2 Indexing

| Component | Implementation | Parameters |
|-----------|----------------|------------|
| Lexical | `rank_bm25.BM25Okapi` | k1=1.5, b=0.75 |
| Dense | Sentence-BERT + FAISS `IndexFlatIP` | dim=384, L2-normalized |
| Persistence | Pickle (BM25), binary (FAISS) | — |

### 3.3 Retrieval

1. **BM25:** stemmed query → scores over full corpus → top-*k*.
2. **FAISS:** encode query → inner-product search → top-*k*.
3. **Hybrid:** retrieve top-200 from each list; fuse with weighted RRF:

   `score(d) = w_bm25/(k + rank_bm25) + w_faiss/(k + rank_faiss)`

   Defaults: `w_bm25=0.3`, `w_faiss=0.7`; *k* tuned per dataset.

### 3.4 Relevance feedback

`q' = α·q + β·mean(V_relevant) − γ·mean(V_nonrelevant)`, then L2-normalize and re-query FAISS.

Defaults: α=1.0, β=0.75, γ=0.25. Pseudo-feedback uses top-3 hybrid hits as relevant.

### 3.5 Clustering

Top-20 hybrid results → TF-IDF (max 5000 features) → K-Means with *k* ∈ {3,4,5,6} selected by
silhouette score → cluster label = top-3 TF-IDF terms at centroid.

---

## 4. Experiments and Results

### Experiment 1 — Retrieval comparison (BEIR)

| Method | SCIDOCS nDCG@10 | SCIDOCS MAP@10 | TREC-COVID nDCG@10 | TREC-COVID MAP@10 |
|--------|-----------------|----------------|--------------------|-------------------|
| BM25 | 0.1477 | 0.0858 | 0.4404 | 0.0095 |
| FAISS + SBERT | 0.2164 | 0.1294 | 0.4723 | 0.0105 |
| Hybrid (RRF) | **0.2164** (k=10) | 0.1293 | **0.6025** (k=100) | 0.0146 |

**Findings:**

- On **SCIDOCS**, semantic retrieval dominates; hybrid matches SBERT (BM25 adds little).
- On **TREC-COVID**, both signals are strong; hybrid improves nDCG@10 by **+27.6%** over the
  best single method.
- Optimal RRF smoothing *k* is dataset-dependent (10 vs. 100).

### Experiment 2 — Relevance feedback gain

Twenty SCIDOCS queries (seed=42). Baseline is hybrid RRF @10. We compare one round of Rocchio
feedback under two strategies: **explicit** (judged-relevant retrieved hits marked relevant, the
rest non-relevant — simulating a user marking the results they see) and **pseudo** (top-3 hybrid
hits blindly assumed relevant). The updated query re-queries FAISS @10.

| Metric | Before (hybrid) | Explicit Δ | Pseudo Δ |
|--------|-----------------|-----------|----------|
| nDCG@10 | 0.3139 | **0.4451 (+0.1312)** | 0.2773 (−0.0366) |
| MAP@10 | 0.2155 | **0.3488 (+0.1333)** | 0.1878 (−0.0277) |
| P@5 | 0.2400 | **0.3300 (+0.0900)** | 0.2200 (−0.0200) |
| MRR@10 | 0.4240 | **0.7500 (+0.3260)** | 0.3424 (−0.0817) |

**Findings:**

- **Explicit feedback yields large, consistent gains** — +41.8% nDCG@10 and a near-doubling of
  MRR@10 (0.42 → 0.75) — satisfying the objective of measurable query-expansion improvement.
  Explicit feedback was applicable to 15 of 20 queries (those with ≥1 judged-relevant hit in the
  top-10); the other 5 fall back to the baseline.
- **Pseudo feedback degrades every metric.** With P@5 ≈ 0.24, the top-3 "assumed relevant" hits are
  mostly off-topic on this citation-based corpus, so the Rocchio centroid drifts away from the
  relevant region.
- Together these show the mechanism is sound, but its benefit hinges on the *quality* of the
  relevance signal: real user judgments help substantially, whereas blind pseudo-feedback hurts on
  low-precision retrieval.

> Methodology note: explicit feedback marks relevant documents drawn from the retrieved list, so the
> gain reflects a simulated user re-ranking toward documents they confirmed relevant (a realistic
> upper bound for one feedback round, not a held-out generalization estimate).

Reproduce with `python evaluation/run_feedback_experiment.py` (writes
`evaluation/results/feedback_report.json`).

### Experiment 3 — Clustering quality

Twenty SCIDOCS queries (seed=42): mean silhouette scores for *k* ∈ {3,4,5,6} on the top-20
hybrid results per query.

| k | Mean silhouette | Std |
|---|-----------------|-----|
| 3 | 0.0233 | 0.0078 |
| 4 | 0.0228 | 0.0102 |
| 5 | 0.0252 | 0.0081 |
| 6 | **0.0256** | 0.0076 |

Per-query silhouette-selected *k*: k=3 (2 queries), k=4 (3), k=5 (7), k=6 (8). The
highest mean silhouette is at **k=6**.

Run `python evaluation/run_clustering_experiment.py` to reproduce. Absolute silhouette
values are low because each result set is only 20 short title+abstract snippets over sparse
TF-IDF, so topic groups are weakly separated — adequate for coarse faceted browsing rather
than hard partitioning. The optimizer favours larger *k* (5–6), splitting the small result
set into finer sub-topics.

### Experiment 4 — Latency

Twenty timed queries (after 3 warmup) on the full arXiv CS index.

Run `python evaluation/run_latency_experiment.py`. Results are saved to
`evaluation/results/latency_report.json`.

**Expected ordering (GPU):** BM25 ≪ FAISS ≈ Hybrid < Hybrid+cluster < Hybrid+Rocchio.

BM25 scans the full inverted index in Python; FAISS search is fast once the query is encoded.
Hybrid runs both paths plus fusion. Clustering and Rocchio add post-retrieval cost.

---

## 5. Discussion

**Strengths**

- Hybrid fusion yields large gains when lexical and semantic rankers are both informative.
- Modular pipeline: shared `ScholARRetriever`, BEIR eval harness, Streamlit UI.
- Rocchio and clustering extend retrieval into interactive exploration.

**Limitations**

- `IndexFlatIP` is exact but memory-bound; ~900k × 384-dim vectors (our full CS corpus) is
  feasible, but it does not scale to tens of millions without approximate indexing.
- BM25 `get_scores` over the full corpus is O(N) per query—dominates latency at scale.
- Pseudo-Rocchio can drift if top results are off-topic.
- Cluster labels are TF-IDF terms, not abstractive summaries.

**Future work**

- Approximate FAISS (IVF, HNSW) and batched BM25/WAND for sub-second latency at scale.
- Cross-encoder reranking on RRF candidates.
- Learned fusion weights instead of fixed 0.3/0.7.

---

## 6. Conclusion

ScholAR demonstrates that **hybrid lexical–semantic retrieval** with RRF meaningfully outperforms
single-method baselines on suitable benchmarks, while **Rocchio feedback** and **K-Means clustering**
support interactive refinement and browsing. The system fulfills the CSC 575 proposal objectives:
full pipeline, BEIR evaluation, relevance feedback, clustering UI, and reproducible experiment scripts.

---

## 7. References

1. Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* FnTIR.
2. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT.* EMNLP. https://arxiv.org/abs/1908.10084
3. Thakur, N., et al. (2021). *BEIR.* NeurIPS. https://arxiv.org/abs/2104.08663
4. Cormack, G., et al. (2009). *Reciprocal Rank Fusion.* SIGIR.
5. Rocchio, J. (1971). *Relevance feedback in information retrieval.* In Salton (Ed.), *The SMART Retrieval System.*

---

## Appendix — Reproducibility

```bash
python src/preprocess.py
python src/build_bm25.py
python src/build_faiss.py
python evaluation/run_all_experiments.py
streamlit run src/app.py
```

See `README.md` and `SUBMISSION.md` for full setup and packaging instructions.
