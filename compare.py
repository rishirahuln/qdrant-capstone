"""Run the test queries through every search method and record the results.

  python compare.py            all 5 queries, writes results/*.json + summary
  python compare.py --query 2  live demo: one query, prints every method's top-5

Methods:
  Part 2  Cosine / Euclid / Dot collections, exact search (metric is the only change)
  Part 3  news_cosine exact (ground truth), HNSW default + weak graph at several ef
  Part 4  our own IVF index at several nprobe

"Overlap" = how many of a method's top-5 ids are also in the exact cosine
top-5 for the same query (5/5 = identical result set).
"""

import argparse
import csv
import json
import statistics
import time

import numpy as np
from qdrant_client import models

from data import RESULTS_DIR, load_embeddings, load_queries
from ivf import N_CLUSTERS, build_ivf_index, search_ivf
from qdrant_setup import COLLECTIONS, get_client

TOP_K = 5
EF_VALUES = [None, 8, 16, 64, 128]  # None = Qdrant's default (ef = ef_construct)
NPROBE_VALUES = [1, 2, 4, 8]
REPEATS = 20  # latency = median of this many runs, after one warm-up run

METRIC_COLLECTIONS = {"Cosine": "news_cosine", "Euclid": "news_euclid", "Dot": "news_dot"}
HNSW_COLLECTIONS = {"default": "news_cosine", "undertuned": "news_undertuned"}


# --- search functions ---
# Each takes a query vector and returns ([(id, score), ...], search_ms), where
# search_ms is the time spent actually searching, excluding network overhead.

def qdrant_search(client, collection, exact=False, ef=None):
    params = models.SearchParams(exact=exact, hnsw_ef=ef)

    def search(query_vector):
        # The low-level REST call (instead of client.query_points) because its
        # response includes "time": Qdrant's own server-side search time.
        # A client-side timer would mostly measure HTTP into the Docker VM.
        response = client.http.search_api.query_points(
            collection_name=collection,
            query_request=models.QueryRequest(
                query=query_vector.tolist(), limit=TOP_K, params=params
            ),
        )
        results = [(p.id, p.score) for p in response.result.points]
        return results, response.time * 1000

    return search


def ivf_search(index, nprobe):
    def search(query_vector):
        start = time.perf_counter()
        ids, scores, _ = search_ivf(index, query_vector, nprobe, TOP_K)
        ms = (time.perf_counter() - start) * 1000
        return list(zip(ids.tolist(), scores.tolist())), ms

    return search


def timed(search, query_vector):
    """Warm up once, then return (results, median search ms, median round-trip ms).

    For Qdrant, round trip includes HTTP; for IVF the two are the same thing
    because it runs inside this Python process.
    """
    results, _ = search(query_vector)
    search_times, round_trips = [], []
    for _ in range(REPEATS):
        start = time.perf_counter()
        _, search_ms = search(query_vector)
        round_trips.append((time.perf_counter() - start) * 1000)
        search_times.append(search_ms)
    return results, statistics.median(search_times), statistics.median(round_trips)


def overlap(results, truth):
    return len({i for i, _ in results} & {i for i, _ in truth})


# --- the list of methods compared in Parts 3-5 ---

def build_index_methods(client, ivf_index):
    """Return [(group, config dict, search fn)] for exact, HNSW and IVF."""
    methods = [("exact", {"method": "Qdrant exact (brute force)"},
                qdrant_search(client, "news_cosine", exact=True))]

    for config_name, collection in HNSW_COLLECTIONS.items():
        hnsw = COLLECTIONS[collection][1]
        for ef in EF_VALUES:
            config = {
                "method": f"HNSW {config_name} (m={hnsw['m']}, ef_construct={hnsw['ef_construct']})",
                "collection": collection,
                "m": hnsw["m"],
                "ef_construct": hnsw["ef_construct"],
                "ef": ef if ef is not None else f"default ({hnsw['ef_construct']})",
            }
            methods.append(("hnsw", config, qdrant_search(client, collection, ef=ef)))

    for nprobe in NPROBE_VALUES:
        methods.append(("ivf", {"method": "IVF (ours)", "n_clusters": N_CLUSTERS, "nprobe": nprobe},
                        ivf_search(ivf_index, nprobe)))

    # Probing every cluster scans every vector: an in-process exact baseline,
    # so IVF latency can be compared without HTTP overhead in the way.
    methods.append(("ivf", {"method": "IVF all clusters = in-process exact",
                            "n_clusters": N_CLUSTERS, "nprobe": N_CLUSTERS},
                    ivf_search(ivf_index, N_CLUSTERS)))
    return methods


def label(config):
    if "ef" in config:
        return f"{config['method']}, ef={config['ef']}"
    if "nprobe" in config:
        return f"{config['method']}, nprobe={config['nprobe']}"
    return config["method"]


def describe(point_id, score, metadata):
    doc = metadata[point_id]
    snippet = " ".join(doc["text"].split())[:70]
    return f"id={point_id:<5} score={score:8.4f}  [{doc['category']}] {snippet}"


# --- demo mode: one query, everything printed as it runs ---

def demo(query_number):
    vectors, metadata, query_vectors = load_embeddings()
    queries = load_queries()
    q = query_vectors[query_number - 1]
    client = get_client()
    print(f'Query {query_number}: "{queries[query_number - 1]}"\n')

    print("=== Part 2: distance metrics (exact search, identical vectors) ===")
    for metric, collection in METRIC_COLLECTIONS.items():
        results, ms, _ = timed(qdrant_search(client, collection, exact=True), q)
        note = "  (Euclid score is a distance: lower = closer)" if metric == "Euclid" else ""
        print(f"\n{metric} [{collection}] search {ms:.3f} ms{note}")
        for point_id, score in results:
            print("   " + describe(point_id, score, metadata))

    print("\n=== Parts 3-4: indexing strategies vs exact cosine search ===")
    print("Building IVF index ...")
    methods = build_index_methods(client, build_ivf_index(vectors))
    truth = None
    for group, config, search in methods:
        results, ms, round_trip = timed(search, q)
        truth = truth or results  # the first method is the exact baseline
        print(f"\n{label(config)}: overlap {overlap(results, truth)}/{TOP_K}, "
              f"search {ms:.3f} ms (round trip {round_trip:.3f} ms)")
        for point_id, score in results:
            print("   " + describe(point_id, score, metadata))


# --- full mode: every query, every method, saved to results/ ---

def run_distance_metrics(client, vectors, metadata, queries, query_vectors):
    output = []
    for text, q in zip(queries, query_vectors):
        entry = {"query": text, "results": {}}
        for metric, collection in METRIC_COLLECTIONS.items():
            results, ms, _ = timed(qdrant_search(client, collection, exact=True), q)
            entry["results"][metric] = {
                "search_ms": round(ms, 3),
                "top5": [
                    {
                        "rank": rank,
                        "id": point_id,
                        "score": round(score, 4),
                        "vector_norm": round(float(np.linalg.norm(vectors[point_id])), 3),
                        "category": metadata[point_id]["category"],
                        "text": " ".join(metadata[point_id]["text"].split())[:120],
                    }
                    for rank, (point_id, score) in enumerate(results, start=1)
                ],
            }
        output.append(entry)
    return output


def run_index_methods(client, vectors, queries, query_vectors):
    ivf_index = build_ivf_index(vectors)
    methods = build_index_methods(client, ivf_index)

    # Ground truth: exact cosine top-5 for every query.
    exact = qdrant_search(client, "news_cosine", exact=True)
    truths = [exact(q)[0] for q in query_vectors]

    rows = []
    for group, config, search in methods:
        per_query = []
        for text, q, truth in zip(queries, query_vectors, truths):
            results, ms, round_trip = timed(search, q)
            record = {
                "query": text,
                "overlap": overlap(results, truth),
                "search_ms": round(ms, 4),
                "round_trip_ms": round(round_trip, 4),
                "ids": [i for i, _ in results],
            }
            if group == "ivf":
                _, _, scanned = search_ivf(ivf_index, q, config["nprobe"], TOP_K)
                record["vectors_scanned"] = scanned
            per_query.append(record)

        row = {
            "group": group,
            **config,
            "avg_overlap": statistics.mean(r["overlap"] for r in per_query),
            "avg_search_ms": round(statistics.mean(r["search_ms"] for r in per_query), 4),
            "avg_round_trip_ms": round(statistics.mean(r["round_trip_ms"] for r in per_query), 4),
            "per_query": per_query,
        }
        if group == "ivf":
            row["avg_vectors_scanned"] = statistics.mean(r["vectors_scanned"] for r in per_query)
        rows.append(row)
        print(f"  {label(config):<62} overlap {row['avg_overlap']:.1f}/{TOP_K}  "
              f"search {row['avg_search_ms']:.3f} ms  round trip {row['avg_round_trip_ms']:.3f} ms")
    return rows


def write_json(name, data):
    path = RESULTS_DIR / name
    path.write_text(json.dumps(data, indent=2))
    print(f"wrote {path.relative_to(RESULTS_DIR.parent)}")


def run_all():
    vectors, metadata, query_vectors = load_embeddings()
    queries = load_queries()
    client = get_client()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Part 2: distance metrics ...")
    write_json("distance_metrics.json",
               run_distance_metrics(client, vectors, metadata, queries, query_vectors))

    print("Parts 3-4: exact vs HNSW vs IVF (averaged over all queries) ...")
    rows = run_index_methods(client, vectors, queries, query_vectors)
    exact_row = [r for r in rows if r["group"] == "exact"]
    write_json("hnsw.json", exact_row + [r for r in rows if r["group"] == "hnsw"])
    write_json("ivf.json", exact_row + [r for r in rows if r["group"] == "ivf"])

    summary_path = RESULTS_DIR / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "avg_overlap_with_exact", "avg_search_ms", "avg_round_trip_ms"])
        for r in rows:
            writer.writerow([label(r), r["avg_overlap"], r["avg_search_ms"], r["avg_round_trip_ms"]])
    print(f"wrote {summary_path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", type=int, choices=range(1, 6),
                        help="demo mode: run only this query (1-5) and print everything")
    args = parser.parse_args()
    if args.query:
        demo(args.query)
    else:
        run_all()
