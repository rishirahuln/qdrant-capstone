# Comparison and findings

**Setup:** 6,000 deduplicated 20 Newsgroups posts were embedded with `all-MiniLM-L6-v2` (384 dimensions). The 5 queries are listed in [queries.md](queries.md), and the raw numbers are in [results/](results/).

**Important detail:** the vectors are the model's *raw* mean-pooled output. The stock model ends in a Normalize layer, which makes every vector length 1, and on unit vectors cosine, dot product and Euclidean distance always rank results identically. Dropping that layer (see `embed.py`) leaves real length differences: document vectors range from 1.17 to 6.07 in length (median 2.12), and the query vectors range from 5.9 to 7.6.

**Definitions used in the tables:**

- **Overlap** is the number of a method's top-5 ids that also appear in exact cosine search's top-5, averaged over the 5 queries.
- **Search ms** for Qdrant is Qdrant's own server-side time (the `time` field of the REST response). For IVF it is the time inside Python. Both are the median of 20 runs.
- **Round trip** adds HTTP into the Docker VM (about 1.7 ms). That overhead hides most of the differences between Qdrant methods.

---

## Part 2: Distance metrics

All three collections hold identical vectors and use exact search, so **the metric is the only thing that changes**. Full top-5 lists are in `results/distance_metrics.json`.

### Case 1: dot product promotes long vectors (query 2, "which motorcycle helmet should I buy")

| Rank | Cosine | Dot product |
|---|---|---|
| 1 | #25 rec.motorcycles (cos 0.560, length 1.96) | **#2902 sci.electronics** (dot 11.17, length **5.37**) |
| 2 | #161 rec.motorcycles (0.558, 2.09) | #5017 rec.motorcycles (10.78, 4.03) |
| 3 | #3316 rec.motorcycles (0.536, 1.83) | #89 misc.forsale (9.72, 4.71) |

Dot product is `|q| × |d| × cos θ`. Post #2902 has a cosine of only about 0.27 with the query, half that of #25, but its vector is 2.7× longer, so its dot product is higher (11.17 vs about 8.4). Cosine returns five motorcycle posts. Dot product returns long posts from electronics and for-sale groups: it rewards magnitude, not just direction. The same happens on query 4 (space shuttle), where dot product's #1 result is a Windows post.

### Case 2: Euclidean changes the top result (query 3, "is there any evidence that god exists")

| Rank | Cosine | Euclidean (distance, lower = closer) |
|---|---|---|
| 1 | #2873 alt.atheism (cos 0.558, length 1.71) | **#5533 soc.religion.christian** (5.505, length 2.84) |
| 2 | #1176 alt.atheism (0.515) | #2873 alt.atheism (5.525) |
| 5 | #5533 soc.religion.christian (0.487) | #3640 alt.atheism |

Euclidean distance satisfies `|q − d|² = |q|² + |d|² − 2|q||d| cos θ`. The `|q|²` term is the same for every document, so ranking depends on `|d|² − 2|q||d| cos θ`. That expression is smallest when `|d| ≈ |q| cos θ ≈ 6.3 × 0.49 ≈ 3.1`. #5533 (length 2.84) sits near that sweet spot. #2873 (length 1.71) points in a better direction but is "too short", so it ends up slightly farther away in straight-line distance (5.525 vs 5.505). Cosine measures angle only, Euclidean measures position (angle and length), and dot product rewards length outright.

On queries 4 and 5 Euclidean returns almost the same list as cosine, because most document lengths (around 2) are close enough to the sweet spot. Of the three metrics, dot product disagrees with cosine the most.

---

## Part 3: HNSW vs exact search

Exact search uses `news_cosine` with `exact=true`. Both HNSW collections use identical vectors. With no `ef` passed, Qdrant searches with `ef = ef_construct`.

| Configuration | ef | Overlap | Search ms |
|---|---|---|---|
| Exact (brute force) | n/a | 5.0 | 0.430 |
| HNSW default (m=16, ef_construct=100) | 8 | 4.6 | 0.172 |
| | 16 | 5.0 | 0.203 |
| | 64 | 5.0 | 0.291 |
| | 100 (default) | 5.0 | 0.334 |
| | 128 | 5.0 | 0.382 |
| HNSW under-tuned (m=4, ef_construct=8) | 8 (default) | 3.2 | 0.150 |
| | 16 | 3.4 | 0.160 |
| | 64 | 3.6 | 0.206 |
| | 128 | 4.0 | 0.244 |

Raising `ef` always costs time. It only buys accuracy until the result already matches exact search: the default graph is perfect from `ef=16`. On the under-tuned graph even `ef=128` stays at 4.0, because a large search budget can't make up for a poorly connected graph. On query 3 it stays at 3/5 no matter the `ef`. HNSW graph building is not fully deterministic, so rebuilding the collections can shift these numbers slightly.

---

## Part 4: Our IVF index

`ivf.py` uses 40 k-means clusters with about 150 vectors each. Here, "vectors scanned" counts only the vectors that get exact scoring, not the 40 centroid comparisons.

| nprobe | Overlap | Search ms | Vectors scanned (of 6,000) |
|---|---|---|---|
| 1 | 4.4 | 0.024 | 141 (2.4%) |
| 2 | 4.6 | 0.051 | 316 |
| 4 | 4.8 | 0.111 | 574 |
| 8 | 4.8 | 0.261 | 1,142 (19%) |
| 40 (all clusters = exact) | 5.0 | 1.606 | 6,000 |

---

## Part 5: Combined comparison

Overlap with exact top-5 and search latency, averaged across the 5 queries:

| Method | Work knob | Overlap /5 | Search ms | Round trip ms |
|---|---|---|---|---|
| Qdrant exact (brute force) | n/a | **5.0** | 0.430 | 2.79 |
| HNSW default, m=16 / ef_construct=100 | ef=8 | 4.6 | 0.172 | 1.88 |
| | ef=16 | **5.0** | 0.203 | 1.94 |
| | ef=64 | **5.0** | 0.291 | 2.06 |
| | ef=100 (default) | **5.0** | 0.334 | 2.39 |
| | ef=128 | **5.0** | 0.382 | 2.21 |
| HNSW under-tuned, m=4 / ef_construct=8 | ef=8 (default) | 3.2 | 0.150 | 1.89 |
| | ef=16 | 3.4 | 0.160 | 1.86 |
| | ef=64 | 3.6 | 0.206 | 1.99 |
| | ef=128 | 4.0 | 0.244 | 2.00 |
| IVF (ours), 40 clusters | nprobe=1 | 4.4 | 0.024 | in-process |
| | nprobe=2 | 4.6 | 0.051 | in-process |
| | nprobe=4 | 4.8 | 0.111 | in-process |
| | nprobe=8 | 4.8 | 0.261 | in-process |
| | nprobe=40 (exact) | **5.0** | 1.606 | in-process |

### More work, closer to exact

Every approximate method got closer to exact search as it was given more work. HNSW with the default graph went from 4.6/5 at `ef=8` to 5/5 from `ef=16` onward. The under-tuned graph climbed from 3.2 to 4.0. IVF went from 4.4 at `nprobe=1` to 4.8 at `nprobe=8`, and to 5.0 when every cluster is probed. Latency rose with the knob every time: HNSW from 0.17 ms to 0.38 ms, IVF from 0.024 ms to 0.26 ms.

This is the speed/accuracy trade-off from the indexing unit. Both indexes avoid comparing the query with every vector by guessing where the neighbours are.

- **HNSW** walks a graph and keeps the best `ef` candidates as it goes. A bigger `ef` explores more paths, so it is less likely to settle on a local best that isn't the true nearest.
- **IVF** opens only the `nprobe` clusters whose centroids look closest. A bigger `nprobe` opens more clusters, so a true neighbour sitting just across a cluster boundary is less likely to be skipped.

In both cases the extra work is extra vectors scored, which is why time grows with accuracy. Pushed to the limit (`nprobe` = all clusters), the approximate method *becomes* exact search and pays its full cost.

The knob only helps if the structure underneath is sound. The under-tuned graph never reached 5/5 even at `ef=128`, because its `m=4` / `ef_construct=8` links leave some neighbours hard to reach.

On only 6,000 vectors, brute force is cheap (0.43 ms), so HNSW's speed-up here is modest (about 2×). Brute force grows linearly with collection size while HNSW grows roughly logarithmically, so the gap widens greatly at millions of vectors. Our in-process exact baseline (1.6 ms) is slower than Qdrant's exact search because numpy copies the vectors before scoring, while Qdrant uses SIMD-optimised Rust running in parallel across segments.

### Metric choice and index choice are separate decisions

In Part 2 the dataset, the vectors and the search method (exact) were all identical, yet "is there any evidence that god exists" returned a different #1 under Euclidean (#5533) than under cosine (#2873). "Which motorcycle helmet should I buy" returned an electronics post first under dot product. The **metric** decides what "closest" means, so it decides the correct answer. The **index** (exact, HNSW, IVF) only decides how quickly and how reliably we find that answer. A perfect index on the wrong metric still returns the wrong documents.

---

## Review notes

**Why does IVF disagree with exact search at low nprobe, and how does raising nprobe help?** IVF only scores vectors inside the probed clusters. For query 4 (space shuttle), two of the true top-5 sit in the 2nd and 3rd closest clusters, so `nprobe=1` finds 3/5 and `nprobe=4` finds 5/5. For query 5 (encryption), true result #3666 sits in the cluster ranked **33rd of 40**: it is a boundary post whose cluster's centroid points away from the query. Only `nprobe ≥ 33` finds it, which is why `nprobe=8` stays at 4/5. Raising `nprobe` widens the set of candidates. Whether IVF finds a given neighbour depends on how close its cluster's centroid is to the query, not on how close the neighbour itself is.

**Why can cosine and dot product return a different top result from identical vectors?** Cosine divides out both lengths and compares angle only. Dot product equals `|q||d| cos θ`, so a long vector with a mediocre angle can outscore a short vector with a good angle (case 1 above). On normalized vectors the two would always agree.

**What single change would make the under-tuned HNSW behave like the default?** Raise `ef_construct` (8 → 100). I tested each change alone (5 queries, `ef` default/16/128):

| Collection | Overlap |
|---|---|
| m=4, ef_construct=8 (under-tuned) | 3.2 – 4.0 |
| m=16, ef_construct=8 (only m raised) | 3.4 – 4.4 |
| **m=4, ef_construct=100** (only ef_construct raised) | **5.0 at every ef** |

`ef_construct` is the candidate list size used *while building* the graph. With only 8 candidates, each new node links to whichever neighbours a shallow search happened to find, often poor ones. A larger value finds each node's true nearest neighbours, so the graph's links are good and search can follow them to the right region. Raising `m` alone gives each node more links, but they are still chosen by the same shallow search. At this dataset size, 4 well-chosen links per node turned out to be enough.

**Qdrant only ships HNSW. When would a real system want IVF?**

- **Very large datasets with limited memory.** HNSW stores `m` to `2m` links for every vector and works best with everything in RAM. IVF stores only the centroids plus one cluster id per vector, and pairs naturally with compression (IVF-PQ in FAISS), so billions of vectors can fit in memory or be scanned from disk cluster by cluster.
- **Fast or frequent rebuilds and bulk loads.** k-means plus assignment is cheap compared with inserting every vector into a graph, and new vectors just get appended to their nearest cluster's list.
- **Predictable, tunable cost.** One IVF query scores about `nprobe × N / k` vectors, a fixed and easy-to-budget amount. That also suits batched GPU scans, which is why GPU vector search is often IVF-based.
- **Accepting slightly lower recall for lower cost.** In our run, `nprobe=1` scored only 2.4% of the vectors and still found 4.4/5.

The cost is what Part 4 showed: recall suffers for boundary points, and keeping recall high needs a larger `nprobe` or re-clustering as the data drifts. Qdrant chooses HNSW because, at RAM-resident scale, it gives higher recall per unit of work and handles live inserts and filtered search well.
