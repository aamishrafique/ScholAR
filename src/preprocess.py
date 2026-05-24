import json
import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm

RAW_PATH = "data/raw/arxiv-metadata-oai-snapshot.json"
PROCESSED_PATH = "data/processed/cs_papers.pkl"
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def is_cs_paper(categories: str) -> bool:
    return any(cat.startswith("cs.") for cat in categories.split())


def load_and_filter_cs_papers(max_papers=500_000):
    papers = []

    with open(RAW_PATH, "r") as f:
        for line in tqdm(f, desc="Filtering CS papers"):
            try:
                paper = json.loads(line)
                if is_cs_paper(paper.get("categories", "")):
                    papers.append(
                        {
                            "id": paper["id"],
                            "title": paper.get("title", "").replace("\n", " ").strip(),
                            "abstract": paper.get("abstract", "")
                            .replace("\n", " ")
                            .strip(),
                            "authors": paper.get("authors", ""),
                            "categories": paper.get("categories", ""),
                            "doi": paper.get("doi", ""),
                            "url": f"https://arxiv.org/abs/{paper['id']}",
                        }
                    )
                # if len(papers) >= max_papers:
                #     break
            except Exception:
                continue

    os.makedirs("data/processed", exist_ok=True)
    with open(PROCESSED_PATH, "wb") as f:
        pickle.dump(papers, f)

    print(f"Saved {len(papers)} CS papers to {PROCESSED_PATH}")
    return papers


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def tokenize_and_stem(text: str) -> list[str]:
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens


def get_document_text(paper: dict) -> str:
    return paper["title"] + " " + paper["title"] + " " + paper["abstract"]
    # Title is doubled to give it more weight in BM25


if __name__ == "__main__":
    load_and_filter_cs_papers()
