"""A minimal IVF (inverted file) index built with k-means.

Build: cluster all vectors into n_clusters groups. Keep each cluster's centroid
and the list of vector ids assigned to it (its "inverted list").

Search: compare the query against the centroids only, pick the nprobe closest
clusters, then score the query exactly against just the vectors in those
clusters. Small nprobe = fewer vectors scored = faster, but the true nearest
neighbour may sit in a cluster we never opened.

Similarity is cosine, to match the news_cosine collection we compare against.
Cosine on normalized vectors is just a dot product, so every vector is
normalized once at build time.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

N_CLUSTERS = 40  # roughly sqrt(6000) / 2, i.e. ~150 vectors per cluster


@dataclass
class IVFIndex:
    centroids: np.ndarray  # (n_clusters, dim), unit length
    lists: list  # lists[c] = array of vector ids in cluster c
    vectors: np.ndarray  # (n_vectors, dim), unit length; row i = point id i


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def build_ivf_index(vectors, n_clusters=N_CLUSTERS, seed=42):
    unit_vectors = normalize(vectors)

    # k-means uses Euclidean distance; on unit vectors that ranks points the
    # same way cosine does, so the clusters group vectors by direction.
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = kmeans.fit_predict(unit_vectors)

    lists = [np.where(labels == c)[0] for c in range(n_clusters)]
    centroids = normalize(kmeans.cluster_centers_)
    return IVFIndex(centroids=centroids, lists=lists, vectors=unit_vectors)


def search_ivf(index, query_vector, nprobe, k):
    """Return (ids, scores, n_scanned) for the top-k cosine matches."""
    query = normalize(query_vector)

    # Step 1: coarse search, which clusters is the query closest to?
    centroid_scores = index.centroids @ query
    probe = np.argsort(-centroid_scores)[:nprobe]

    # Step 2: exact search, but only over vectors in those clusters.
    candidate_ids = np.concatenate([index.lists[c] for c in probe])
    scores = index.vectors[candidate_ids] @ query

    top = np.argsort(-scores)[:k]
    return candidate_ids[top], scores[top], len(candidate_ids)


if __name__ == "__main__":
    from data import load_embeddings

    vectors, metadata, query_vectors = load_embeddings()
    index = build_ivf_index(vectors)
    sizes = [len(ids) for ids in index.lists]
    print(f"{len(sizes)} clusters, sizes min {min(sizes)} / mean {np.mean(sizes):.0f} / max {max(sizes)}")

    ids, scores, scanned = search_ivf(index, query_vectors[0], nprobe=1, k=5)
    print(f"Query 1, nprobe=1, scanned {scanned} of {len(vectors)} vectors:")
    for i, s in zip(ids, scores):
        print(f"  {s:.3f}  {metadata[i]['category']}")
