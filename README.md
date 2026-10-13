# Qdrant capstone: comparing search algorithms

This project embeds 6,000 posts from 20 Newsgroups, loads them into Qdrant, and compares:

- three **distance metrics**: Cosine, Euclidean and Dot product
- **exact search** against **HNSW**, with a default graph and an under-tuned graph across several `ef` values
- a hand-built **IVF index** (k-means and `nprobe` search)

The findings are in [comparison.md](comparison.md).

## Requirements

- Python 3.10+ (developed on 3.12)
- Docker. On macOS without Docker Desktop, Colima works:
  ```bash
  brew install colima docker docker-compose
  colima start
  ```

## Setup

```bash
# 1. Start Qdrant (Part 0). Data persists in ./qdrant_storage
docker compose up -d
#    Dashboard: http://localhost:6333/dashboard

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer not to use Compose, this single command starts Qdrant the same way:

```bash
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant:v1.19.1
```

**Dashboard check:** with the container running, `http://localhost:6333/dashboard` loads the Qdrant web UI. Before setup it shows no collections. After step 4 below it lists `news_cosine`, `news_euclid`, `news_dot` and `news_undertuned`.

## Run

```bash
python embed.py          # Part 1: sample + embed the dataset to data/ (~40 s on CPU)
python qdrant_setup.py   # Parts 2-3: create and index the 4 Qdrant collections
python ivf.py            # Part 4: optional sanity check of the IVF index
python compare.py        # Parts 2-5: run every method, write results/
```

**Live demo:** run one query through every method (three metrics, both HNSW configs at each `ef`, and IVF at each `nprobe`), printing results as it goes:

```bash
python compare.py --query 1
```

## Files

| File | Purpose |
|---|---|
| `data.py` | Loads and deduplicates 20 Newsgroups, samples 6,000 posts (seed 42), reads `queries.md`, holds shared paths |
| `embed.py` | Embeds documents and queries with `all-MiniLM-L6-v2` without its Normalize layer (see the docstring) |
| `qdrant_setup.py` | Creates and fills the Cosine, Euclid, Dot and under-tuned HNSW collections, and waits for indexing |
| `ivf.py` | `build_ivf_index` (k-means, 40 clusters) and `search_ivf(index, query_vector, nprobe, k)` |
| `compare.py` | Runs all methods and records overlap with exact search and latency |
| `queries.md` | The 5 test queries |
| `comparison.md` | Tables and written findings |
| `results/distance_metrics.json` | Top-5 ids, scores, vector lengths and categories per metric per query |
| `results/hnsw.json` | Exact and each HNSW config × `ef`: overlap and latency, per query and averaged |
| `results/ivf.json` | Each `nprobe`: overlap, latency and vectors scanned |
| `results/summary.csv` | One row per method (the combined table) |
