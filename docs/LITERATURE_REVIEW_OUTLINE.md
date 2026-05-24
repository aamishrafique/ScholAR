# ScholAR — Literature Review Video Outline (~5–7 min)

Separate video focused on **prior work** referenced in the proposal and report.

---

## 1. Introduction (30 s)

- Information retrieval for scientific literature
- Why classical + neural methods are both still relevant

## 2. Lexical retrieval — BM25 (1.5 min)

**Robertson & Zaragoza (2009)**

- Probabilistic relevance framework
- Term frequency saturation (k1) and length normalization (b)
- Why BM25 remains a strong baseline for keyword-heavy queries
- **ScholAR use:** `rank_bm25`, doubled title weighting

## 3. Dense retrieval — Sentence-BERT (1.5 min)

**Reimers & Gurevych (2019)**

- Siamese BERT networks for sentence embeddings
- `all-MiniLM-L6-v2`: 384-dim, fast, good quality
- Cosine similarity via normalized inner product
- **ScholAR use:** bi-encoder + FAISS `IndexFlatIP`

## 4. Evaluation methodology — BEIR (1 min)

**Thakur et al. (2021)**

- Zero-shot heterogeneous benchmark
- SCIDOCS: citation-based scientific task
- TREC-COVID: biomedical search (COVID-19)
- Standard metrics: nDCG, MAP, MRR, P@k
- **ScholAR use:** `beir` harness, Experiments 1–3

## 5. Fusion — Reciprocal Rank Fusion (1 min)

**Cormack et al. (2009)**

- Merge ranked lists using reciprocal ranks
- Robust when scores are incomparable (BM25 vs. cosine)
- **ScholAR use:** weighted RRF, k tuned per dataset

## 6. Relevance feedback — Rocchio (45 s)

- Classic vector space feedback formula
- **ScholAR use:** dense vectors, explicit + pseudo modes

## 7. Closing (30 s)

- How these pieces compose ScholAR
- Pointer to technical report (`docs/REPORT.md`)

---

## Suggested on-screen citations

Display paper title + authors + year when discussing each work; link arXiv URLs from the proposal references section.
