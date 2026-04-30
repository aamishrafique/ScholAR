import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader


SCIDOCS_PATH = "evaluation/scidocs"


def download_scidocs():
    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip"
    )
    os.makedirs(SCIDOCS_PATH, exist_ok=True)
    data_path = util.download_and_unzip(url, SCIDOCS_PATH)
    print(f"SCIDOCS downloaded to: {data_path}")


def load_scidocs():
    corpus, queries, qrels = GenericDataLoader(
        data_folder=os.path.join(SCIDOCS_PATH, "scidocs")
    ).load(split="test")

    print(f"Corpus size:  {len(corpus)} papers")
    print(f"Queries:      {len(queries)}")
    print(f"Qrels:        {len(qrels)}")
    return corpus, queries, qrels


if __name__ == "__main__":
    download_scidocs()
    load_scidocs()
