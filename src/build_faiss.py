import quiet_imports  # noqa: F401

import os
import pickle
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from preprocess import PROCESSED_PATH, get_document_text

FAISS_INDEX_PATH = "indexes/faiss/faiss_index.bin"
FAISS_IDS_PATH = "indexes/faiss/paper_ids.pkl"
EMBEDDING_DIM = 384
BATCH_SIZE = 512
MODEL_NAME = "all-MiniLM-L6-v2"


def build_faiss_index():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading papers...")
    with open(PROCESSED_PATH, "rb") as f:
        papers = pickle.load(f)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    texts = [get_document_text(p) for p in papers]
    paper_ids = [p["id"] for p in papers]

    print("Encoding documents in batches...")
    all_embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):
        batch = texts[i : i + BATCH_SIZE]
        embeddings = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        )
        all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings).astype("float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(all_embeddings)

    os.makedirs("indexes/faiss", exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_IDS_PATH, "wb") as f:
        pickle.dump(paper_ids, f)

    print(f"FAISS index saved. {index.ntotal} vectors stored.")


if __name__ == "__main__":
    build_faiss_index()
