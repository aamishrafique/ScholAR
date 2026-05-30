"""
ScholAR Streamlit UI — Demo (small corpus).

Identical pipeline to src/app.py (BM25 + Sentence-BERT + Hybrid RRF + Rocchio
feedback + K-Means clustering), but instead of loading the full ~900k-document
prebuilt indexes, it samples a small subset of the processed corpus and builds
the BM25 + FAISS indexes in memory at startup. This loads in seconds and every
query is near-instant — ideal for a live demo.

Run from project root:  streamlit run src/app_demo.py

Optionally set the corpus size:  set SCHOLAR_DEMO_SIZE=3000  (PowerShell: $env:SCHOLAR_DEMO_SIZE=3000)
"""

import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

import quiet_imports  # noqa: F401 — before transformers / sentence-transformers

import streamlit as st

from clustering import ResultClusterer, group_results_by_cluster
from retriever import ScholARRetriever

PROCESSED_PATH = "data/processed/cs_papers.pkl"

DISPLAY_K = 10
CLUSTER_K = 20
DEFAULT_DEMO_SIZE = int(os.environ.get("SCHOLAR_DEMO_SIZE", "2000"))
RANDOM_SEED = 42


def corpus_ready() -> bool:
    return os.path.exists(PROCESSED_PATH)


@st.cache_resource(show_spinner="Loading corpus…")
def load_papers() -> list:
    with open(PROCESSED_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Building demo indexes (small corpus)…")
def load_retriever(corpus_size: int, seed: int) -> ScholARRetriever:
    papers = load_papers()

    if corpus_size >= len(papers):
        subset = papers
    else:
        rng = random.Random(seed)
        subset = rng.sample(papers, corpus_size)

    retriever = ScholARRetriever()
    retriever.build_from_corpus(subset)
    return retriever


def paper_year(paper: dict) -> str:
    pid = paper.get("id", "")
    if not pid or "." not in pid:
        return "—"
    prefix = pid.split(".")[0]
    if len(prefix) >= 2 and prefix[:2].isdigit():
        yy = int(prefix[:2])
        return str(2000 + yy if yy < 90 else 1900 + yy)
    return "—"


def run_search(retriever: ScholARRetriever, query: str, mode: str) -> list:
    if mode == "BM25":
        return retriever.search_bm25(query, top_k=DISPLAY_K)
    if mode == "Semantic (FAISS)":
        qv = retriever.encode_query(query)
        return retriever.search_faiss(qv, top_k=DISPLAY_K)
    return retriever.search_hybrid(query, top_k=DISPLAY_K, rrf_k=60)


def render_paper_card(result: dict, rank: int, key_prefix: str):
    paper = result.get("paper", {})
    title = paper.get("title", "Untitled")
    authors = paper.get("authors", "Unknown authors")
    abstract = paper.get("abstract", "")[:400]
    url = paper.get("url", f"https://arxiv.org/abs/{result.get('id', '')}")
    year = paper_year(paper)
    score = result.get("score", 0.0)

    st.markdown(f"**{rank}. {title}**")
    st.caption(f"{authors} · {year} · score: {score:.4f}")
    if abstract:
        st.write(abstract + ("…" if len(paper.get("abstract", "")) > 400 else ""))
    st.markdown(f"[arXiv]({url})")

    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("Relevant", key=f"{key_prefix}_rel"):
        st.session_state.relevant.add(rank - 1)
        st.session_state.nonrelevant.discard(rank - 1)
    if c2.button("Not relevant", key=f"{key_prefix}_nrel"):
        st.session_state.nonrelevant.add(rank - 1)
        st.session_state.relevant.discard(rank - 1)

    if (rank - 1) in st.session_state.relevant:
        st.success("Marked relevant")
    elif (rank - 1) in st.session_state.nonrelevant:
        st.error("Marked not relevant")


def init_session():
    defaults = {
        "results": [],
        "query": "",
        "query_vector": None,
        "cluster_output": None,
        "relevant": set(),
        "nonrelevant": set(),
        "feedback_applied": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def main():
    st.set_page_config(page_title="ScholAR (Demo)", page_icon="📚", layout="wide")
    init_session()

    st.title("ScholAR · Demo")
    st.caption(
        "Hybrid scientific literature search · BM25 + Sentence-BERT + Rocchio feedback "
        "· small in-memory corpus for fast demos"
    )

    if not corpus_ready():
        st.error(
            f"Processed corpus not found at `{PROCESSED_PATH}`.\n\n"
            "Build it first: `python src/preprocess.py`"
        )
        return

    total_papers = len(load_papers())

    st.sidebar.markdown("### Demo corpus")
    corpus_size = st.sidebar.slider(
        "Number of papers",
        min_value=200,
        max_value=min(20000, total_papers),
        value=min(DEFAULT_DEMO_SIZE, total_papers),
        step=200,
        help="Smaller = faster. Indexes are rebuilt in memory when this changes.",
    )
    st.sidebar.caption(f"Sampled from {total_papers:,} processed papers (seed {RANDOM_SEED}).")

    retriever = load_retriever(corpus_size, RANDOM_SEED)

    col_q, col_mode = st.columns([3, 1])
    with col_q:
        query = st.text_input("Search query", placeholder="e.g. graph neural networks")
    with col_mode:
        mode = st.selectbox("Retrieval mode", ["Hybrid", "BM25", "Semantic (FAISS)"])

    if st.button("Search", type="primary") and query.strip():
        st.session_state.query = query.strip()
        st.session_state.results = run_search(retriever, st.session_state.query, mode)
        st.session_state.query_vector = retriever.encode_query(st.session_state.query)
        st.session_state.relevant = set()
        st.session_state.nonrelevant = set()
        st.session_state.feedback_applied = False

        cluster_hits = st.session_state.results[:CLUSTER_K]
        clusterer = ResultClusterer()
        st.session_state.cluster_output = clusterer.cluster_results(cluster_hits)

    if not st.session_state.results:
        st.info("Enter a query and click **Search** to retrieve papers.")
        return

    cluster_out = st.session_state.cluster_output or {}
    sil_scores = cluster_out.get("silhouette_scores", {})
    if sil_scores:
        best_k = max(sil_scores, key=sil_scores.get)
        st.sidebar.markdown("### Clustering")
        st.sidebar.write(f"Selected **k = {cluster_out.get('n_clusters', best_k)}**")
        st.sidebar.write(f"Silhouette: **{cluster_out.get('silhouette', 0):.3f}**")
        for k, s in sorted(sil_scores.items()):
            st.sidebar.write(f"k={k}: {s:.3f}")

    st.sidebar.markdown("### Relevance feedback")
    st.sidebar.write(f"Relevant: {len(st.session_state.relevant)}")
    st.sidebar.write(f"Not relevant: {len(st.session_state.nonrelevant)}")

    if st.sidebar.button("Apply Rocchio feedback"):
        rel = sorted(st.session_state.relevant)
        nrel = sorted(st.session_state.nonrelevant)
        if not rel and not nrel:
            st.sidebar.warning("Mark at least one result as relevant or not relevant.")
        else:
            refined, updated_qv = retriever.apply_rocchio_feedback(
                st.session_state.query_vector,
                st.session_state.results,
                relevant_indices=rel,
                nonrelevant_indices=nrel,
                top_k=DISPLAY_K,
            )
            st.session_state.results = refined
            st.session_state.query_vector = updated_qv
            st.session_state.feedback_applied = True
            clusterer = ResultClusterer()
            st.session_state.cluster_output = clusterer.cluster_results(
                st.session_state.results[:CLUSTER_K]
            )
            st.sidebar.success("Query updated — results re-ranked.")

    if st.session_state.feedback_applied:
        st.info("Showing results after Rocchio relevance feedback.")

    tab_ranked, tab_clusters = st.tabs(["Ranked list", "Cluster tabs"])

    with tab_ranked:
        for rank, hit in enumerate(st.session_state.results, start=1):
            with st.container(border=True):
                render_paper_card(hit, rank, f"flat_{rank}")

    with tab_clusters:
        cluster_output = st.session_state.cluster_output or {}
        groups = group_results_by_cluster(
            st.session_state.results[:CLUSTER_K], cluster_output
        )
        labels = cluster_output.get("cluster_labels", {})
        if not groups:
            st.write("Not enough results to cluster.")
        else:
            cluster_tabs = st.tabs(
                [
                    labels.get(cid, f"Cluster {cid + 1}")
                    for cid in sorted(groups.keys())
                ]
            )
            for tab, cid in zip(cluster_tabs, sorted(groups.keys())):
                with tab:
                    for rank, hit in enumerate(groups[cid], start=1):
                        with st.container(border=True):
                            render_paper_card(hit, rank, f"cl_{cid}_{rank}")


if __name__ == "__main__":
    main()
