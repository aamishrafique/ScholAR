import os
import pickle
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from preprocess import PROCESSED_PATH, get_document_text, tokenize_and_stem

BM25_INDEX_PATH = "indexes/bm25/bm25_index.pkl"
BM25_IDS_PATH = "indexes/bm25/paper_ids.pkl"


def build_bm25_index():
    print("Loading papers...")
    with open(PROCESSED_PATH, "rb") as f:
        papers = pickle.load(f)

    print("Tokenizing documents...")
    tokenized_corpus = []
    paper_ids = []

    for paper in tqdm(papers, desc="Tokenizing"):
        tokens = tokenize_and_stem(get_document_text(paper))
        tokenized_corpus.append(tokens)
        paper_ids.append(paper["id"])

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

    os.makedirs("indexes/bm25", exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(BM25_IDS_PATH, "wb") as f:
        pickle.dump(paper_ids, f)

    print(f"BM25 index saved. {len(papers)} documents indexed.")


if __name__ == "__main__":
    build_bm25_index()
