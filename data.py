"""Load and sample the 20 Newsgroups dataset, and read the test queries.

Also holds the file paths shared by every other script, so they all agree on
where the embedded data lives.
"""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
VECTORS_PATH = DATA_DIR / "vectors.npy"
METADATA_PATH = DATA_DIR / "metadata.json"
QUERY_VECTORS_PATH = DATA_DIR / "query_vectors.npy"
QUERIES_PATH = ROOT / "queries.md"

N_DOCS = 6000
MIN_CHARS = 50  # drop near-empty posts; they embed to meaningless vectors
SEED = 42  # fixed seed so every run samples the same documents


def load_documents(n_docs=N_DOCS, seed=SEED):
    """Return a list of {"id", "text", "category"} dicts sampled from 20 Newsgroups."""
    # remove= strips email headers, signatures and quoted replies, so the
    # embedding reflects what the post says rather than who sent it.
    dataset = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )

    candidates = [
        (text.strip(), dataset.target_names[label])
        for text, label in zip(dataset.data, dataset.target)
        if len(text.strip()) >= MIN_CHARS
    ]

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=n_docs, replace=False)

    # The position in this list becomes the point id in Qdrant and the row
    # index in vectors.npy, so ids line up everywhere.
    return [
        {"id": i, "text": candidates[j][0], "category": candidates[j][1]}
        for i, j in enumerate(chosen)
    ]


def load_queries(path=QUERIES_PATH):
    """Read the numbered test queries (lines like `1. some query`) from queries.md."""
    queries = []
    for line in path.read_text().splitlines():
        match = re.match(r"^\d+\.\s+(.+)$", line.strip())
        if match:
            queries.append(match.group(1))
    return queries


def load_embeddings():
    """Load what embed.py saved: document vectors, metadata, query vectors."""
    vectors = np.load(VECTORS_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    query_vectors = np.load(QUERY_VECTORS_PATH)
    return vectors, metadata, query_vectors


if __name__ == "__main__":
    docs = load_documents()
    categories = {d["category"] for d in docs}
    print(f"Sampled {len(docs)} documents across {len(categories)} categories")
    print(f"Example ({docs[0]['category']}): {docs[0]['text'][:200]!r}")
