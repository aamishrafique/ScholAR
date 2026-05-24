# ScholAR — Presentation Video Outline (~8–10 min)

Use this outline when recording your Week 4 demo/presentation video.

---

## 1. Title & team (30 s)

- Project name: **ScholAR**
- Team members, CSC 575
- One-line pitch: hybrid search over arXiv CS papers

## 2. Problem & motivation (1 min)

- Volume of CS literature on arXiv
- Keyword search vs. semantic search trade-off
- Goal: combine both + let users refine and browse

## 3. System architecture (2 min)

Show diagram or README architecture table:

- Preprocess → BM25 index + FAISS/SBERT index
- Query → BM25 top-200 + FAISS top-200 → RRF fusion
- Optional: Rocchio on dense vectors; K-Means on top-20 for clusters

**Live demo (recommended):** `streamlit run src/app.py`

- Run a query (e.g. “graph neural networks for node classification”)
- Show ranked list, cluster tabs, mark relevant → Apply Rocchio feedback

## 4. Key design choices (1.5 min)

- Why BM25 + SBERT (complementary signals)
- Why RRF (no score normalization needed)
- Title doubling for BM25 only; raw text for SBERT
- Rocchio in embedding space

## 5. Evaluation results (2.5 min)

**Experiment 1 — BEIR**

| | SCIDOCS | TREC-COVID |
|---|---------|-------------|
| Best single | SBERT 0.216 nDCG@10 | SBERT 0.472 |
| Hybrid | 0.216 (tie) | **0.603** (+27.6%) |

**Experiment 2–3:** Briefly mention feedback + clustering scripts

**Experiment 4:** Show latency table from `evaluation/results/latency_report.txt`
— BM25 vs. hybrid vs. hybrid+extras

## 6. Limitations & future work (1 min)

- Full-corpus BM25 scoring is slow
- Flat FAISS index memory
- Pseudo-feedback assumptions

## 7. Conclusion (30 s)

- Objectives met: pipeline, hybrid eval, Rocchio, clustering UI
- Hybrid shines when both lexical and semantic cues matter (TREC-COVID)
- Thank you / questions

---

## Recording tips

- Record terminal + Streamlit side-by-side or switch scenes
- Pre-load indexes before recording to avoid long waits
- Keep console clean (`quiet_imports` already suppresses HF spam)
