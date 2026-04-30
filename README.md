# ScholAR

## Overview

ScholAR is a two-stage hybrid retrieval system for scientific literature sourced from arXiv.
The system combines lexical and semantic search to retrieve and rank papers based on a user query.

This document covers the Week 1 implementation, which includes data preparation, text
preprocessing, BM25 index construction, and FAISS dense index construction using Sentence-BERT
embeddings.

---

## Architecture

The system uses two independent retrieval methods that operate in parallel:

| Component     | Method          | Matching Strategy                        | Speed    |
| ------------- | --------------- | ---------------------------------------- | -------- |
| BM25 Index    | Lexical Search  | Keyword frequency + document length norm | Fast     |
| FAISS + SBERT | Semantic Search | Dense vector cosine similarity           | Moderate |

### BM25 (Lexical)

BM25 operates on a traditional inverted index. Queries and documents are tokenized, stemmed,
and stripped of stop words. The scoring function rewards term frequency while penalizing
document length to prevent longer documents from dominating results.

### FAISS + Sentence-BERT (Semantic)

Each document is encoded into a 384-dimensional dense vector using the
`all-MiniLM-L6-v2` Sentence-BERT model. Vectors are L2-normalized and stored in a FAISS
`IndexFlatIP` index, enabling cosine similarity search via inner product. This method
captures semantic relationships between terms, surfacing relevant results even when
query keywords are absent from the document.

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
sentence-transformers
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
│   └── scidocs/                    # BEIR SCIDOCS benchmark files
├── src/
│   ├── preprocess.py               # Filtering, cleaning, tokenization
│   ├── build_bm25.py               # BM25 index construction
│   ├── build_faiss.py              # FAISS index construction
│   └── test_query.py               # Query testing for both indexes
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

### Step 4 — Run a Test Query

```bash
python src/test_query.py
```

---

## Example Queries & Outputs

**Query:** `transformer models for natural language processing`

### BM25 Results

```
=== BM25 results for: 'transformer models for natural language processing' ===
1. [16.266] Enhanced Transformer Architecture for Natural Language Processing
   https://arxiv.org/abs/2310.10930

2. [15.665] Partial Tensorized Transformers for Natural Language Processing
   https://arxiv.org/abs/2310.20077

3. [15.286] Language Invariant Properties in Natural Language Processing
   https://arxiv.org/abs/2109.13037

4. [15.211] NL-Augmenter: A Framework for Task-Sensitive Natural Language Augmentation
   https://arxiv.org/abs/2112.02721

5. [15.203] On the validity of pre-trained transformers for natural language processing
            in the software engineering domain
   https://arxiv.org/abs/2109.04738
```

### FAISS + SBERT Results

```
=== FAISS (SBERT) results for: 'transformer models for natural language processing' ===
1. [0.746] Introduction to Transformers: an NLP Perspective
   https://arxiv.org/abs/2311.17633

2. [0.729] A Review of Bangla Natural Language Processing Tasks and the Utility of
            Transformer Models
   https://arxiv.org/abs/2107.03844

3. [0.717] The Unreasonable Effectiveness of Transformer Language Models in
            Grammatical Error Correction
   https://arxiv.org/abs/1906.01733

4. [0.710] DistilCamemBERT: a distillation of the French model CamemBERT
   https://arxiv.org/abs/2205.11111

5. [0.703] N-Grammer: Augmenting Transformers with latent n-grams
   https://arxiv.org/abs/2207.06366
```

---

## Observations

| Behavior          | BM25                                                                    | FAISS + SBERT                                                               |
| ----------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Matching strategy | Exact keyword overlap                                                   | Semantic vector similarity                                                  |
| Handles synonyms  | No — "NLP" and "Natural Language Processing" treated as different terms | Yes — understands "NLP" ↔ "Natural Language Processing"                     |
| Result 1          | Contains exact query terms in title                                     | Semantically equivalent title with different phrasing                       |
| Unexpected result | None — all titles match keywords directly                               | DistilCamemBERT (French BERT model) — related by architecture, not keywords |
| Score range       | 15–16 (raw frequency-based)                                             | 0.70–0.75 (cosine similarity, bounded 0–1)                                  |
| Strength          | Precise, fast, reliable for known terms                                 | Broad coverage, handles paraphrasing and synonyms                           |
| Weakness          | Misses semantically related documents                                   | May surface loosely related results                                         |

The divergence in results confirms that each method surfaces a distinct subset of relevant
documents. This motivates the hybrid fusion approach planned for Week 2.

---

## Current Status

| Task                              | Status   |
| --------------------------------- | -------- |
| arXiv dataset downloaded          | Complete |
| CS paper filtering                | Complete |
| Text preprocessing pipeline       | Complete |
| BM25 index construction           | Complete |
| FAISS + SBERT index construction  | Complete |
| BEIR SCIDOCS benchmark setup      | Complete |
| Sanity check query (both indexes) | Complete |
