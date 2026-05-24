# ScholAR — Submission Checklist

**Team:** Aamish Rafique, Peter Spedale  
**Course:** CSC 575

---

## Included in repository

| Item | Location |
|------|----------|
| Source code | `src/` |
| Evaluation scripts | `evaluation/` |
| Technical report | `docs/REPORT.md` |
| README (setup & results) | `README.md` |
| Project proposal | `CSC 575 Project Proposal.pdf` |
| Presentation outline | `docs/PRESENTATION_OUTLINE.md` |
| Literature review outline | `docs/LITERATURE_REVIEW_OUTLINE.md` |
| Dependencies | `requirements.txt` |

## Not included (build locally)

| Item | How to obtain |
|------|----------------|
| arXiv raw data | Kaggle: Cornell-University/arxiv |
| Processed papers | `python src/preprocess.py` |
| BM25 / FAISS indexes | `python src/build_bm25.py`, `src/build_faiss.py` |
| BEIR datasets | Auto-download via `evaluation/run_beir.py` |

## Before submitting

- [ ] All experiments run without errors (see below)
- [ ] `evaluation/results/latency_report.json` generated (Exp 4)
- [ ] Streamlit UI tested (`streamlit run src/app.py`)
- [ ] Presentation video recorded (use `docs/PRESENTATION_OUTLINE.md`)
- [ ] Literature review video recorded (use `docs/LITERATURE_REVIEW_OUTLINE.md`)
- [ ] Zip/upload excludes `data/`, `indexes/`, large zips (use packaging script)

## Run all experiments

```bash
conda activate scholar
pip install -r requirements.txt
# + PyTorch with CUDA per README

python src/preprocess.py
python src/build_bm25.py
python src/build_faiss.py

python evaluation/run_all_experiments.py
```

Or individually:

```bash
python evaluation/run_beir.py                  # Exp 1
python evaluation/run_feedback_experiment.py # Exp 2
python evaluation/run_clustering_experiment.py # Exp 3
python evaluation/run_latency_experiment.py  # Exp 4
```

## Package for upload

```bash
python scripts/package_submission.py
```

Creates `dist/scholar-submission.zip` with code and docs only (no data/indexes).

## Grader quick start

1. Install dependencies (README § Setup).
2. Download arXiv snapshot to `data/raw/arxiv-metadata-oai-snapshot.json`.
3. Build indexes (three commands above).
4. `streamlit run src/app.py` OR `python src/demo_retrieval.py`.
5. Read `docs/REPORT.md` for full methodology and results.
