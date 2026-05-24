# ScholAR

**CSC 575 — Team ScholAR:** Aamish Rafique, Peter Spedale

## Overview

ScholAR is a two-stage hybrid retrieval system for scientific literature sourced from arXiv.
The system combines lexical and semantic search to retrieve and rank papers based on a user
query. Retrieval is followed by relevance feedback, which allows users to iteratively refine
results using the Rocchio algorithm.

The system is evaluated on two BEIR benchmarks — SCIDOCS and TREC-COVID — across three
retrieval configurations: BM25-only, FAISS+SBERT-only, and hybrid fusion via Reciprocal
Rank Fusion (RRF).

---

## Architecture

### Retrieval Components

| Component     | Method          | Matching Strategy                        | Speed    |
| ------------- | --------------- | ---------------------------------------- | -------- |
| BM25 Index    | Lexical Search  | Keyword frequency + document length norm | Fast     |
| FAISS + SBERT | Semantic Search | Dense vector cosine similarity           | Moderate |
| Hybrid RRF    | Fusion          | Weighted rank-based score combination    | Moderate |

### BM25 (Lexical)

BM25 operates on a traditional inverted index. Queries and documents are tokenized, stemmed,
and stripped of stop words. The scoring function rewards term frequency while penalizing
document length to prevent longer documents from dominating results.

### FAISS + Sentence-BERT (Semantic)

Each document is encoded into a 384-dimensional dense vector using the `all-MiniLM-L6-v2`
Sentence-BERT model. Vectors are L2-normalized and stored in a FAISS `IndexFlatIP` index,
enabling cosine similarity search via inner product. This method captures semantic
relationships between terms, surfacing relevant results even when query keywords are absent
from the document.

### Hybrid Retrieval — Weighted RRF

Weighted Reciprocal Rank Fusion merges the BM25 and FAISS ranked lists into a single
unified ranking. Each document receives a combined score based on its rank position in each
list, weighted by the contribution of each method:

```
score(d) = (w_bm25 / (k + rank_bm25(d))) + (w_faiss / (k + rank_faiss(d)))
```

Default weights: `w_bm25 = 0.3`, `w_faiss = 0.7`. The `k` parameter is tuned per dataset.
This asymmetric weighting reflects that semantic retrieval consistently outperforms lexical
retrieval on scientific literature benchmarks.

### Relevance Feedback — Rocchio

After an initial retrieval, users can mark results as relevant or non-relevant. The Rocchio
algorithm updates the dense query vector accordingly:

```
q' = α·q + β·mean(relevant vectors) − γ·mean(non-relevant vectors)
```

Default parameters: `α = 1.0`, `β = 0.75`, `γ = 0.25`. The updated vector is
L2-normalized and used to re-query the FAISS index. Pseudo-relevance feedback is also
supported, which automatically treats the top-k results as relevant without explicit
user input.

### Result Clustering — K-Means on TF-IDF

After retrieval, the top-20 results are clustered into topic groups using K-Means on
TF-IDF vectors (title + abstract). The number of clusters `k ∈ {3, 4, 5, 6}` is chosen
by silhouette score. Each cluster receives an auto-generated label from its top-3
TF-IDF terms, enabling faceted browsing in the Streamlit UI.

---

## Setup Instructions

### Requirements

- Python 3.10
- CUDA 12.8 (for GPU acceleration)
- Anaconda or virtualenv

### Installation

Create and activate a virtual environment:

```bash
conda create -n scholar python=3.10
conda activate scholar
```

Install PyTorch with CUDA 12.8 support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```
beir
faiss-cpu
kaggle
nltk
numpy
rank_bm25
scikit-learn
sentence-transformers
streamlit
tqdm
```

Download required NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## Project Structure

```
scholar/
├── data/
│   ├── raw/                        # Raw arXiv JSON snapshot
│   └── processed/                  # Filtered and serialized CS papers
├── indexes/
│   ├── bm25/
│   │   ├── bm25_index.pkl          # Serialized BM25 index
│   │   └── paper_ids.pkl           # Ordered list of paper IDs
│   └── faiss/
│       ├── faiss_index.bin         # FAISS binary index
│       └── paper_ids.pkl           # Ordered list of paper IDs
├── evaluation/
│   ├── scidocs/                    # BEIR SCIDOCS benchmark files
│   ├── trec-covid/                 # BEIR TREC-COVID benchmark files
│   ├── run_beir.py                 # Experiment 1 — retrieval comparison
│   ├── run_feedback_experiment.py  # Experiment 2 — Rocchio feedback gain
│   ├── run_clustering_experiment.py # Experiment 3 — silhouette / k tuning
│   ├── run_latency_experiment.py   # Experiment 4 — query latency
│   ├── run_all_experiments.py      # Run experiments 1–4 in sequence
│   └── queries_latency.txt         # Query set for latency benchmarks
├── docs/
│   ├── REPORT.md                   # Final technical report (Week 4)
│   ├── PRESENTATION_OUTLINE.md     # Demo/presentation video script
│   └── LITERATURE_REVIEW_OUTLINE.md
├── scripts/
│   └── package_submission.py       # Zip code + docs for submission
├── SUBMISSION.md                   # Submission checklist
├── src/
│   ├── preprocess.py               # Filtering, cleaning, tokenization
│   ├── build_bm25.py               # BM25 index construction
│   ├── build_faiss.py              # FAISS index construction
│   ├── retriever.py                # Core retriever — BM25, FAISS, Hybrid
│   ├── rocchio.py                  # Rocchio relevance feedback
│   ├── clustering.py               # K-Means clustering + cluster labels
│   ├── app.py                      # Streamlit web UI
│   ├── demo_retrieval.py           # CLI demo — retrieval + Rocchio
│   └── test_query.py               # Sanity check query for both indexes
└── requirements.txt
```

---

## Data Preparation

### Dataset

The primary corpus is the [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv),
containing approximately 2.3 million papers in JSON format (~4 GB).

Download via Kaggle API:

```bash
kaggle datasets download -d Cornell-University/arxiv -p data/raw/ --unzip
```

Place the resulting file at:

```
data/raw/arxiv-metadata-oai-snapshot.json
```

### Filtering CS Papers

The dataset is filtered to Computer Science papers (categories prefixed with `cs.`) and
capped at 500,000 documents for computational feasibility.

```bash
python src/preprocess.py
```

Fields extracted per paper:

| Field        | Description                   |
| ------------ | ----------------------------- |
| `id`         | arXiv paper ID                |
| `title`      | Paper title                   |
| `abstract`   | Paper abstract                |
| `authors`    | Author list                   |
| `categories` | arXiv subject categories      |
| `doi`        | DOI identifier (if available) |
| `url`        | Direct arXiv link             |

Output saved to `data/processed/cs_papers.pkl`.

### Preprocessing Pipeline

Applied to all documents before indexing:

1. Lowercase normalization
2. Punctuation removal
3. Stop word removal (NLTK English stop words)
4. Porter stemming
5. Title field doubled to increase its weight in BM25 scoring

---

## BM25 Implementation

BM25 is built using the `rank_bm25` library with the following parameters:

| Parameter | Value | Description                          |
| --------- | ----- | ------------------------------------ |
| `k1`      | 1.5   | Term frequency saturation            |
| `b`       | 0.75  | Document length normalization factor |

Build the index:

```bash
python src/build_bm25.py
```

Output:

- `indexes/bm25/bm25_index.pkl` — Serialized BM25 index
- `indexes/bm25/paper_ids.pkl` — Corresponding paper ID list

---

## FAISS + SBERT Implementation

### Model

| Property      | Value                       |
| ------------- | --------------------------- |
| Model         | `all-MiniLM-L6-v2`          |
| Embedding dim | 384                         |
| Normalization | L2 (cosine via dot product) |
| Batch size    | 512                         |
| Device        | CUDA (GPU)                  |

### Index Type

`faiss.IndexFlatIP` — Exact inner product search over L2-normalized vectors,
equivalent to cosine similarity.

Build the index:

```bash
python src/build_faiss.py
```

Expected output:

```
Using device: cuda
Loading papers...
Loading model: all-MiniLM-L6-v2
Encoding documents in batches...
Encoding: 100%|████████████████| 1781/1781 [36:00<00:00,  1.21s/it]
FAISS index saved. 500000 vectors stored.
```

Output:

- `indexes/faiss/faiss_index.bin` — FAISS binary index
- `indexes/faiss/paper_ids.pkl` — Corresponding paper ID list

---

## How to Run

### Step 1 — Preprocess and Filter Papers

```bash
python src/preprocess.py
```

### Step 2 — Build BM25 Index

```bash
python src/build_bm25.py
```

### Step 3 — Build FAISS Index

```bash
python src/build_faiss.py
```

### Step 4 — Run Retrieval Demo (Hybrid + Rocchio)

```bash
python src/demo_retrieval.py
```

### Step 5 — Run Benchmark Evaluation

```bash
python evaluation/run_beir.py
```

SCIDOCS and TREC-COVID are downloaded automatically on first run.

### Step 6 — Launch Streamlit UI (Week 3)

```bash
streamlit run src/app.py
```

The UI provides a search box, ranked results, cluster tabs (TF-IDF labels), and
**Relevant** / **Not relevant** buttons that trigger Rocchio feedback.

### Step 7 — Week 3 Experiments

```bash
# Experiment 2: feedback gain (20 sampled SCIDOCS queries)
python evaluation/run_feedback_experiment.py

# Experiment 3: clustering quality (silhouette for k = 3, 4, 5, 6)
python evaluation/run_clustering_experiment.py
```

### Step 8 — Week 4: Latency, report & submission

```bash
# Experiment 4: average query latency on full arXiv CS index
python evaluation/run_latency_experiment.py

# Or run all four experiments sequentially
python evaluation/run_all_experiments.py

# Package code + docs for upload (excludes data/indexes)
python scripts/package_submission.py
```

**Deliverables**

| Deliverable | Location |
|-------------|----------|
| Technical report | [`docs/REPORT.md`](docs/REPORT.md) |
| Presentation outline | [`docs/PRESENTATION_OUTLINE.md`](docs/PRESENTATION_OUTLINE.md) |
| Literature review outline | [`docs/LITERATURE_REVIEW_OUTLINE.md`](docs/LITERATURE_REVIEW_OUTLINE.md) |
| Submission checklist | [`SUBMISSION.md`](SUBMISSION.md) |
| Latency results (generated) | `evaluation/results/latency_report.json` |

Record presentation and literature-review videos using the outlines above (not automated).

---

## Example Queries & Outputs

**Query:** `graph neural networks for node classification`

### BM25 Results

```
=================================================================
  BM25 — Lexical
=================================================================
  1. [22.7182] Graph Decipher: A transparent dual-attention graph neural network to
       https://arxiv.org/abs/2201.01381

  2. [22.3169] Label-Consistency based Graph Neural Networks for Semi-supervised Node
       https://arxiv.org/abs/2007.13435

  3. [22.1178] On Calibration of Graph Neural Networks for Node Classification
       https://arxiv.org/abs/2206.01570

  4. [22.0682] Distributional Signals for Node Classification in Graph Neural Network
       https://arxiv.org/abs/2304.03507

  5. [21.9361] Graph Convolutional Network For Semi-supervised Node Classification Wi
       https://arxiv.org/abs/2404.12724
```

### FAISS + SBERT Results

```
=================================================================
  FAISS + SBERT — Semantic
=================================================================
  1. [0.7308] Customized Graph Neural Networks
       https://arxiv.org/abs/2005.12386

  2. [0.7259] Graph Neural Networks for Small Graph and Giant Network Representation
       https://arxiv.org/abs/1908.00187

  3. [0.7130] Infinite Width Graph Neural Networks for Node Regression/Classification
       https://arxiv.org/abs/2310.08176

  4. [0.7125] Incorporating Heterophily into Graph Neural Networks for Graph Class...
       https://arxiv.org/abs/2203.07678

  5. [0.7090] Graph Neural Networks: Taxonomy, Advances and Trends
       https://arxiv.org/abs/2012.08752
```

### Hybrid — RRF Results

```
=================================================================
  Hybrid — RRF Fusion
=================================================================
  1. [0.0280] Graph Convolutional Network For Semi-supervised Node Classification Wi
       https://arxiv.org/abs/2404.12724

  2. [0.0278] On Calibration of Graph Neural Networks for Node Classification
       https://arxiv.org/abs/2206.01570

  3. [0.0275] Incorporating Heterophily into Graph Neural Networks for Graph Class...
       https://arxiv.org/abs/2203.07678

  4. [0.0267] Revisiting Neighborhood Aggregation in Graph Neural Networks for Node
       https://arxiv.org/abs/2407.15284

  5. [0.0266] Infinite Width Graph Neural Networks for Node Regression/Classification
       https://arxiv.org/abs/2310.08176
```

### After Pseudo Rocchio Feedback (top 3 as relevant)

```
=================================================================
  After Pseudo Rocchio Feedback (top 3 treated as relevant)
=================================================================
  1. [0.8317] Incorporating Heterophily into Graph Neural Networks for Graph Class...
       https://arxiv.org/abs/2203.07678

  2. [0.8113] Customized Graph Neural Networks
       https://arxiv.org/abs/2005.12386

  3. [0.8052] Graph Convolutional Network For Semi-supervised Node Classification Wi
       https://arxiv.org/abs/2404.12724

  4. [0.7960] Graph Neural Networks for Small Graph and Giant Network Representation
       https://arxiv.org/abs/1908.00187

  5. [0.7914] On Calibration of Graph Neural Networks for Node Classification
       https://arxiv.org/abs/2206.01570
```

---

## Observations

### Lexical vs. Semantic Retrieval

| Behavior          | BM25                                                                    | FAISS + SBERT                                                               |
| ----------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Matching strategy | Exact keyword overlap                                                   | Semantic vector similarity                                                  |
| Handles synonyms  | No — "NLP" and "Natural Language Processing" treated as different terms | Yes — understands "NLP" ↔ "Natural Language Processing"                     |
| Result 1          | Contains exact query terms in title                                     | Semantically equivalent title with different phrasing                       |
| Unexpected result | None — all titles match keywords directly                               | DistilCamemBERT (French BERT model) — related by architecture, not keywords |
| Score range       | 15–22 (raw frequency-based)                                             | 0.70–0.75 (cosine similarity, bounded 0–1)                                  |
| Strength          | Precise, fast, reliable for known terms                                 | Broad coverage, handles paraphrasing and synonyms                           |
| Weakness          | Misses semantically related documents                                   | May surface loosely related results                                         |

### Hybrid Fusion Behaviour

Hybrid RRF promotes documents that rank well in both lists. Results that appear exclusively
in one list are demoted relative to documents that achieve moderate ranks in both. This
produces a more balanced result set that inherits the strengths of each individual method.

### Relevance Feedback Behaviour

After pseudo Rocchio feedback using the top 3 hybrid results as relevant, FAISS cosine
similarity scores increase from the 0.73 range to the 0.83 range. The updated query
vector shifts toward the semantic neighbourhood of the confirmed relevant documents,
surfacing more tightly related papers in subsequent retrieval.

---

## Evaluation

Both indexes are built over each benchmark corpus independently. Retrieval is run across
all queries and scored against human relevance judgments using BEIR's evaluation harness.
The `k` parameter in RRF is tuned per dataset by sweeping `k ∈ {10, 20, 30, 60, 100}`
with fixed weights `(w_bm25=0.3, w_faiss=0.7)`.

### SCIDOCS (25,657 documents — 1,000 queries)

A citation-based benchmark where relevance reflects semantic co-citation patterns rather
than lexical overlap. BM25 is significantly weaker than SBERT on this task.

| Method              | nDCG@10 | MAP@10 | P@5    | MRR@10 |
| ------------------- | ------- | ------ | ------ | ------ |
| BM25                | 0.1477  | 0.0858 | 0.1062 | 0.2649 |
| FAISS + SBERT       | 0.2164  | 0.1294 | 0.1572 | 0.3594 |
| Hybrid — RRF (k=10) | 0.2164  | 0.1293 | 0.1576 | 0.3595 |

On SCIDOCS, the hybrid matches the best individual method. When BM25 underperforms by a
large margin, fusion recovers to the stronger baseline but cannot exceed it.

### TREC-COVID (171,332 documents — 50 queries)

A biomedical retrieval benchmark where both lexical and semantic cues are informative.
Both methods are competitive, enabling the hybrid to achieve a clear synergistic gain.

| Method               | nDCG@10    | MAP@10     | P@5        | MRR@10     |
| -------------------- | ---------- | ---------- | ---------- | ---------- |
| BM25                 | 0.4404     | 0.0095     | 0.4960     | 0.7304     |
| FAISS + SBERT        | 0.4723     | 0.0105     | 0.5480     | 0.7244     |
| Hybrid — RRF (k=100) | **0.6025** | **0.0146** | **0.6720** | **0.7594** |

The hybrid outperforms BM25 by +36.8% and SBERT by +27.6% on nDCG@10.

> **Note:** MAP scores on TREC-COVID are low across all methods due to the small query set
> (50 queries) and sparse relevance judgments over a large corpus. nDCG@10 and P@5 are the
> primary metrics for this benchmark.

### Cross-Dataset Summary

| Dataset    | BM25 nDCG@10 | SBERT nDCG@10 | Hybrid nDCG@10 | Hybrid gain vs. best individual |
| ---------- | ------------ | ------------- | -------------- | ------------------------------- |
| SCIDOCS    | 0.1477       | 0.2164        | 0.2164         | 0.0%                            |
| TREC-COVID | 0.4404       | 0.4723        | **0.6025**     | **+27.6%**                      |

Hybrid retrieval gains are corpus-dependent. When both retrieval methods are competitive,
fusion produces a meaningful improvement. When one method significantly underperforms,
fusion converges to the stronger baseline. Optimal `k` also varies: SCIDOCS favours
`k=10` (low smoothing amplifies the dominant SBERT signal) while TREC-COVID favours
`k=100` (high smoothing blends two competitive signals more evenly).

---

## Current Status

| Task                                       | Status   |
| ------------------------------------------ | -------- |
| arXiv dataset downloaded                   | Complete |
| CS paper filtering                         | Complete |
| Text preprocessing pipeline                | Complete |
| BM25 index construction                    | Complete |
| FAISS + SBERT index construction           | Complete |
| Core retriever class (BM25, FAISS, Hybrid) | Complete |
| Weighted RRF fusion                        | Complete |
| RRF k hyperparameter tuning                | Complete |
| Rocchio relevance feedback (pseudo)        | Complete |
| Rocchio relevance feedback (explicit)      | Complete |
| K-Means clustering + silhouette tuning     | Complete |
| Streamlit UI (search, clusters, feedback)  | Complete |
| Experiment 2 — feedback gain evaluation    | Complete |
| Experiment 3 — clustering quality eval     | Complete |
| SCIDOCS benchmark evaluation               | Complete |
| TREC-COVID benchmark evaluation            | Complete |
| Experiment 4 — latency benchmarks          | Complete |
| Final technical report                     | Complete |
| Presentation & literature review outlines  | Complete |
| Submission packaging script                | Complete |
