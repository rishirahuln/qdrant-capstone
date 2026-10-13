"""Create and fill the Qdrant collections used by compare.py.

Run after embed.py: `python qdrant_setup.py`. Re-running recreates every
collection from scratch.

Collections (all hold identical vectors and payload):
  news_cosine       Cosine distance, default HNSW   (Part 2 + Part 3 baseline)
  news_euclid       Euclidean distance, default HNSW (Part 2)
  news_dot          Dot product, default HNSW        (Part 2)
  news_undertuned   Cosine distance, weak HNSW (m=4, ef_construct=8) (Part 3)
"""

import time

from qdrant_client import QdrantClient, models

from data import load_embeddings

QDRANT_URL = "http://localhost:6333"

# name -> (distance metric, HNSW graph settings)
# m: links per node in the graph. ef_construct: candidate list size while
# building it. Qdrant's defaults are m=16, ef_construct=100.
COLLECTIONS = {
    "news_cosine": (models.Distance.COSINE, {"m": 16, "ef_construct": 100}),
    "news_euclid": (models.Distance.EUCLID, {"m": 16, "ef_construct": 100}),
    "news_dot": (models.Distance.DOT, {"m": 16, "ef_construct": 100}),
    "news_undertuned": (models.Distance.COSINE, {"m": 4, "ef_construct": 8}),
}

# Thresholds are in KB. Our whole dataset is only ~9 MB, so Qdrant's defaults
# would skip building the HNSW graph (indexing_threshold) and brute-force every
# query (full_scan_threshold). Lowering both makes HNSW actually get used.
INDEXING_THRESHOLD_KB = 100
FULL_SCAN_THRESHOLD_KB = 10


def get_client():
    return QdrantClient(url=QDRANT_URL, timeout=60)


def create_collection(client, name, distance, hnsw):
    if client.collection_exists(name):
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=384, distance=distance),
        hnsw_config=models.HnswConfigDiff(
            m=hnsw["m"],
            ef_construct=hnsw["ef_construct"],
            full_scan_threshold=FULL_SCAN_THRESHOLD_KB,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=INDEXING_THRESHOLD_KB
        ),
    )


def upload(client, name, vectors, metadata):
    # Point id = row index, so results from every method can be compared by id.
    client.upload_collection(
        collection_name=name,
        vectors=vectors,
        payload=[{"category": d["category"], "text": d["text"]} for d in metadata],
        ids=[d["id"] for d in metadata],
        wait=True,
    )


def wait_until_indexed(client, name, timeout_s=300):
    """Block until Qdrant has finished building the HNSW graph for every point."""
    start = time.time()
    while time.time() - start < timeout_s:
        info = client.get_collection(name)
        if (
            info.status == models.CollectionStatus.GREEN
            and info.indexed_vectors_count == info.points_count
        ):
            return info
        time.sleep(1)
    raise TimeoutError(f"{name} was not fully indexed after {timeout_s}s")


def main():
    vectors, metadata, _ = load_embeddings()
    client = get_client()

    for name, (distance, hnsw) in COLLECTIONS.items():
        create_collection(client, name, distance, hnsw)
        upload(client, name, vectors, metadata)
        info = wait_until_indexed(client, name)
        print(
            f"{name:<17} {distance.value:<7} m={hnsw['m']:<3} "
            f"ef_construct={hnsw['ef_construct']:<4} "
            f"points={info.points_count} indexed={info.indexed_vectors_count}"
        )


if __name__ == "__main__":
    main()
