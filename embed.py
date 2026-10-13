"""Embed the sampled documents and the test queries, and save them to data/.

Run once: `python embed.py`. Every later script loads from data/ instead of
re-embedding.
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from data import (
    DATA_DIR,
    METADATA_PATH,
    QUERY_VECTORS_PATH,
    VECTORS_PATH,
    load_documents,
    load_queries,
)

MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dimensional, fast on CPU


def load_raw_model():
    """Load MiniLM *without* its final Normalize layer.

    The stock model is Transformer -> Pooling -> Normalize, so every vector it
    returns has length 1. On unit vectors, cosine, dot product and Euclidean
    distance always rank results identically (dot = cos, and
    |a-b|^2 = 2 - 2cos), which would make the metric comparison meaningless.

    Keeping only Transformer -> Pooling gives the raw mean-pooled vectors,
    whose lengths vary per text. Cosine still ignores length (Qdrant
    normalizes vectors in a Cosine collection), but dot product and Euclidean
    now see it, so the three metrics can disagree.
    """
    full_model = SentenceTransformer(MODEL_NAME)
    return SentenceTransformer(modules=[full_model[0], full_model[1]])


def main():
    DATA_DIR.mkdir(exist_ok=True)
    model = load_raw_model()

    docs = load_documents()
    print(f"Embedding {len(docs)} documents with {MODEL_NAME} ...")
    vectors = model.encode(
        [d["text"] for d in docs], batch_size=64, show_progress_bar=True
    ).astype(np.float32)

    queries = load_queries()
    query_vectors = model.encode(queries).astype(np.float32)

    np.save(VECTORS_PATH, vectors)
    np.save(QUERY_VECTORS_PATH, query_vectors)
    METADATA_PATH.write_text(json.dumps(docs))

    norms = np.linalg.norm(vectors, axis=1)
    print(f"Saved vectors {vectors.shape} and {len(queries)} query vectors to {DATA_DIR}/")
    print(f"Vector lengths: min {norms.min():.2f}, mean {norms.mean():.2f}, max {norms.max():.2f}")


if __name__ == "__main__":
    main()
