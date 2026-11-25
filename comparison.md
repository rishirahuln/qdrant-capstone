# Comparison and findings

## What I set up

I took 6,000 posts from the 20 Newsgroups dataset and removed duplicate posts. Then I turned each post into a vector with the `all-MiniLM-L6-v2` model. Each vector has 384 numbers. My 5 test queries are in [queries.md](queries.md), and all the raw numbers are in [results/](results/).

**One thing I had to change first.** The model normally rescales every vector to a length of 1. I found out that when every vector has length 1, cosine, dot product and Euclidean distance always give the same ranking, so there would be nothing to compare in Part 2. In `embed.py` I removed that last rescaling step. Now the vectors have different lengths: documents range from 1.17 to 6.07 (typical length 2.12), and my queries range from 5.9 to 7.6.

**Words I use in the tables:**

- **Overlap:** how many of a method's top 5 results are also in the exact search top 5. 5/5 means the same results. I average it over my 5 queries.
- **Search ms:** how long the search itself took, in milliseconds. For Qdrant, I use the time Qdrant reports for its own work. For my IVF index, I timed it in Python. Each number is the middle value (median) of 20 runs.
- **Round trip ms:** the search time plus the time to send the request to Docker and get the answer back. Sending the request adds about 1.7 ms, which is bigger than most of the differences I'm trying to measure. That's why I mainly compare search ms.

---

## Part 2: Distance metrics

I made three collections with exactly the same vectors. The only difference is the distance metric. I used exact search in all three, so the metric is the only thing that changes. The full top 5 lists are in `results/distance_metrics.json`.

### Example 1: dot product prefers long vectors (query 2, "which motorcycle helmet should I buy")

| Rank | Cosine | Dot product |
|---|---|---|
| 1 | #25 rec.motorcycles (cos 0.560, length 1.96) | **#2902 sci.electronics** (dot 11.17, length **5.37**) |
| 2 | #161 rec.motorcycles (0.558, 2.09) | #5017 rec.motorcycles (10.78, 4.03) |
| 3 | #3316 rec.motorcycles (0.536, 1.83) | #89 misc.forsale (9.72, 4.71) |

With cosine, all of my top results are motorcycle posts. With dot product, the #1 result is an electronics post.

This happens because dot product = (query length) × (document length) × (cosine). Post #2902 is not very similar to the query: its cosine is only about 0.27, half of #25's. But its vector is 2.7 times longer, and that pushes its dot product up to 11.17, while #25 only gets about 8.4. So dot product rewards long vectors, not only vectors that point the same way as the query. I saw the same thing on query 4 (space shuttle), where dot product's #1 result was a Windows post.

### Example 2: Euclidean changes the #1 result (query 3, "is there any evidence that god exists")

| Rank | Cosine | Euclidean (distance, lower = closer) |
|---|---|---|
| 1 | #2873 alt.atheism (cos 0.558, length 1.71) | **#5533 soc.religion.christian** (5.505, length 2.84) |
| 2 | #1176 alt.atheism (0.515) | #2873 alt.atheism (5.525) |
| 5 | #5533 soc.religion.christian (0.487) | #3640 alt.atheism |

Cosine only looks at the angle between two vectors. Euclidean measures the straight-line distance between the two points, so both the angle and the length matter.

Working through the formula `|q − d|² = |q|² + |d|² − 2|q||d| cos θ`, the closest document is one whose length is about (query length) × (cosine). For this query that's about 6.3 × 0.49 ≈ 3.1. Post #5533 has length 2.84, close to that "best length". Post #2873 points in a slightly better direction, but its length is only 1.71, which is too short. So in straight-line distance, #5533 ends up a little closer (5.505 vs 5.525).

On queries 4 and 5, Euclidean gave almost the same list as cosine, because most documents have a length near 2, which is close enough. Of the three metrics, dot product was the most different from cosine.

**Summary:** cosine looks at angle only. Euclidean looks at angle and length together. Dot product gives a bonus to long vectors.

---

## Part 3: HNSW vs exact search

Exact search compares the query with every single vector, so I used it as the "correct answer". HNSW is faster because it walks through a graph and skips most of the vectors.

I tested two HNSW collections with the same vectors:

- **default:** `m=16`, `ef_construct=100`, Qdrant's normal settings
- **under-tuned:** `m=4`, `ef_construct=8`, deliberately weak settings

`ef` is how many candidates HNSW keeps while it searches. If I don't set `ef`, Qdrant uses the `ef_construct` value.

| Configuration | ef | Overlap | Search ms |
|---|---|---|---|
| Exact (brute force) | n/a | 5.0 | 0.464 |
| HNSW default (m=16, ef_construct=100) | 8 | 4.6 | 0.219 |
| | 16 | 5.0 | 0.258 |
| | 64 | 5.0 | 0.375 |
| | 100 (default) | 5.0 | 0.413 |
| | 128 | 5.0 | 0.470 |
| HNSW under-tuned (m=4, ef_construct=8) | 8 (default) | 1.8 | 0.168 |
| | 16 | 2.6 | 0.190 |
| | 64 | 3.6 | 0.258 |
| | 128 | 3.8 | 0.273 |

**What I noticed:**

- A higher `ef` always made the search slower.
- A higher `ef` only made the default graph more accurate up to a point. It was already perfect at `ef=16`, so going higher just cost time. At `ef=128` it was even a little slower than exact search (0.470 vs 0.464 ms), because with only 6,000 vectors, checking all of them is cheap.
- On average, the under-tuned graph never got to 5/5, even at `ef=128`. Queries 1 and 2 did reach 5/5 there, but the others didn't:
  - **Query 4 (space shuttle):** 0/5 at `ef=8`, then stuck at 1/5 from `ef=16` all the way to `ef=128`. Most results were sci.space posts, the right topic, but not the 5 closest posts.
  - **Query 5 (email encryption):** 0/5 at both `ef=8` and `ef=16`. It answered with posts about cars, Windows and religion, and only recovered to 3/5 at `ef=64` and 4/5 at `ef=128`.

  Searching longer doesn't help much when the graph itself is badly connected.
- HNSW builds its graph a little differently each time. Before this final run I had built the same weak settings once more, and that graph scored 2.8–4.4 instead of 1.8–3.8. The default graph gave the same accuracy both times, so a weak graph is also a less predictable one.

---

## Part 4: My IVF index

My IVF index in `ivf.py` splits the 6,000 vectors into 40 groups (clusters) using k-means, about 150 vectors per group. To search, it compares the query with the 40 group centers, opens the `nprobe` closest groups, and checks every vector inside them.

| nprobe | Overlap | Search ms | Vectors checked (of 6,000) |
|---|---|---|---|
| 1 | 4.4 | 0.030 | 141 (2.4%) |
| 2 | 4.6 | 0.069 | 316 |
| 4 | 4.8 | 0.132 | 574 |
| 8 | 4.8 | 0.278 | 1,142 (19%) |
| 40 (all groups = exact search) | 5.0 | 1.699 | 6,000 |

The "vectors checked" column doesn't count the 40 comparisons with group centers.

---

## Part 5: Everything in one table

Averaged across my 5 queries:

| Method | Setting | Overlap /5 | Search ms | Round trip ms |
|---|---|---|---|---|
| Qdrant exact (brute force) | n/a | **5.0** | 0.464 | 3.02 |
| HNSW default, m=16 / ef_construct=100 | ef=8 | 4.6 | 0.219 | 2.35 |
| | ef=16 | **5.0** | 0.258 | 2.47 |
| | ef=64 | **5.0** | 0.375 | 2.68 |
| | ef=100 (default) | **5.0** | 0.413 | 2.59 |
| | ef=128 | **5.0** | 0.470 | 2.63 |
| HNSW under-tuned, m=4 / ef_construct=8 | ef=8 (default) | 1.8 | 0.168 | 2.19 |
| | ef=16 | 2.6 | 0.190 | 2.45 |
| | ef=64 | 3.6 | 0.258 | 2.73 |
| | ef=128 | 3.8 | 0.273 | 2.49 |
| My IVF, 40 clusters | nprobe=1 | 4.4 | 0.030 | runs in Python |
| | nprobe=2 | 4.6 | 0.069 | runs in Python |
| | nprobe=4 | 4.8 | 0.132 | runs in Python |
| | nprobe=8 | 4.8 | 0.278 | runs in Python |
| | nprobe=40 (exact) | **5.0** | 1.699 | runs in Python |

### More work means closer to exact search

Every fast method got closer to exact search when I gave it more work to do. The default HNSW went from 4.6/5 at `ef=8` to 5/5 at `ef=16` and above. The under-tuned HNSW went from 1.8 to 3.8. My IVF went from 4.4 at `nprobe=1` to 4.8 at `nprobe=8`, and reached 5.0 when it opened every group. Each time, the search also got slower: HNSW from 0.22 ms to 0.47 ms, and IVF from 0.030 ms to 0.28 ms.

This matches the speed vs accuracy trade-off from the indexing unit. Both indexes are fast because they don't check every vector. They guess where the closest vectors probably are.

- **HNSW** keeps a list of the best `ef` candidates as it walks the graph. A bigger `ef` means it looks around more, so it's less likely to stop at a "pretty good" result and miss the best one.
- **IVF** only opens the `nprobe` groups whose centers are closest. A bigger `nprobe` opens more groups, so it's less likely to miss a good result sitting just across a group's edge.

Either way, more work means checking more vectors, and that's why accuracy and time go up together. If IVF opens every group, it turns into exact search and costs the same.

Extra search work can't fix a bad index, though. My under-tuned graph never reached 5/5.

With only 6,000 vectors, exact search is already fast (0.46 ms), so HNSW is only about 1.8 times faster here (at `ef=16`). Exact search gets slower in proportion to the number of vectors, but HNSW slows down much more gently, so the gap would be much bigger with millions of vectors.

### Choosing a metric and choosing an index are separate decisions

In Part 2, I kept the dataset, the vectors and the search method (exact) the same and only changed the metric. The rankings still changed. For "is there any evidence that god exists", the #1 result was #2873 with cosine but #5533 with Euclidean. For "which motorcycle helmet should I buy", dot product put an electronics post first.

So the **metric** decides what "closest" means, which decides what the right answer is. The **index** (exact, HNSW or IVF) only decides how fast and how reliably I find that answer. Even a perfect index will return the wrong posts if I pick the wrong metric.

---

## Extra experiments

### A query that sits between two topics
I tried a 6th query, "computer software NASA uses to control spacecraft", which mixes space and computers. (It's commented out in `queries.md`, so the main results above use only my 5 queries.) Its 5 correct results were spread over 4 different IVF clusters, so `nprobe=1` found only **1 of 5**, and even `nprobe=8` found 4. My original queries 1–3 were all 5/5 at `nprobe=1`, because each one's answers sit in a single cluster. Queries that mix topics are where IVF struggles.

### Number of IVF clusters
I changed `N_CLUSTERS` in `ivf.py` from 40 to 10 and to 100. These runs used 6 queries (including the one above), so there are 30 correct results in total, and 0.17 in the average is a single result.

| N_CLUSTERS | Posts per cluster | nprobe=1 overlap | nprobe=1 ms | nprobe=8 overlap | nprobe=8 ms | nprobe=8 checks about |
|---|---|---|---|---|---|---|
| 10 | ~600 | 3.7 | 0.114 | 5.0 | 1.216 | 80% of the data |
| 40 | ~150 | 3.8 | 0.025 | 4.7 | 0.276 | 20% |
| 100 | ~60 | 3.3 | 0.018 | 4.5 | 0.091 | 8% |

- **10 clusters:** `nprobe=1` checked 4 times as many posts as with 40 clusters but wasn't more accurate. With 20 newsgroups squeezed into 10 clusters, each centroid is a blurry average of several topics, so the closest centroid isn't always the right cluster.
- **100 clusters:** fast, but topics get split into several small clusters, so `nprobe=1` lost 3 more results than with 40.
- **40 clusters** (about √N, and about 2 per newsgroup) was the best balance.
- The same `nprobe` isn't the same amount of work: `nprobe=8` opened 80% of the data with 10 clusters but only 8% with 100. The fair way to compare is accuracy for the same number of vectors checked, or the same search time.

---

## Review notes

**Why does my IVF index sometimes disagree with exact search at a low nprobe, and how does a higher nprobe help?**

IVF only checks the vectors inside the groups it opens. If a correct result lives in a group that isn't opened, IVF can't find it.

- For query 4 (space shuttle), two of the correct top 5 were in the 2nd and 3rd closest groups. So `nprobe=1` found 3/5 and `nprobe=4` found 5/5.
- For query 5 (encryption), the correct result #3666 is in the group ranked **33rd of 40**. The post itself is close to the query, but the center of its group is far from it. Only `nprobe` of 33 or more would find it, which is why `nprobe=8` still got 4/5.

A higher `nprobe` opens more groups, so fewer correct results get missed. What matters is how close a result's *group center* is to the query, not just how close the result itself is.

**Why can cosine and dot product give a different #1 result from the same vectors?**

Cosine ignores length and only compares direction. Dot product multiplies by both lengths, so a long vector pointing in a "so-so" direction can beat a short vector pointing in a good direction (see Example 1). If all vectors had length 1, the two would always agree.

**What one change would make the under-tuned HNSW work like the default?**

Raise `ef_construct` from 8 to 100. To check, I rebuilt only the under-tuned collection, changing one setting at a time, and ran my 5 queries. I compare rows at the same `ef`, because with no `ef` given, Qdrant searches with `ef = ef_construct`:

| Collection | ef=8 | ef=16 | ef=64 | ef=128 |
|---|---|---|---|---|
| m=4, ef_construct=8 (under-tuned) | 1.8 | 2.6 | 3.6 | 3.8 |
| m=16, ef_construct=8 (only `m` raised) | 1.4 | 2.4 | 3.4 | 4.0 |
| **m=4, ef_construct=100** (only `ef_construct` raised) | **4.2** | **5.0** | **5.0** | **5.0** |
| m=16, ef_construct=100 (default) | 4.6 | 5.0 | 5.0 | 5.0 |

Raising only `ef_construct` made the graph as good as the default. Raising only `m` gave each point 4 times as many links but didn't help: it stayed about as bad as the under-tuned graph, and searches got a bit slower (0.306 ms instead of 0.245 ms at `ef=128`), because every hop has more neighbors to check.

`ef_construct` is how many candidates Qdrant looks at *while building* the graph. With only 8, each new point gets linked to whatever neighbors a quick search found, and those are often not the real closest ones. With 100, each point gets linked to its real closest neighbors, so later searches can follow good links to the right place. Raising `m` only adds more links picked from the same small set of poor candidates. For a dataset this size, 4 good links per point were enough.

The remaining miss at `ef=8` (4.2) isn't a graph problem: the default graph also misses one there. `ef=8` is just a very small search budget for picking 5 results.

**Qdrant only offers HNSW. When would a real system want IVF instead?**

- **Huge datasets and not much memory.** HNSW stores many links for every vector and works best when everything fits in RAM. IVF only needs the group centers and which group each vector belongs to. It also combines well with compressing the vectors, so it can handle billions of vectors, or read one group at a time from disk.
- **Building or rebuilding the index often.** Running k-means and sorting vectors into groups is quicker than inserting every vector into a graph. New vectors can just be added to their closest group.
- **Predictable cost.** Each IVF search checks about `nprobe × (number of vectors) / (number of groups)` vectors, so it's easy to know how much work a query will take. This also works well on GPUs, which are good at checking lots of vectors at once.
- **When slightly lower accuracy is OK.** In my test, `nprobe=1` checked only 2.4% of the vectors and still found 4.4 of the 5 correct results.

The downside is what I saw in Part 4: results near a group's edge get missed. Keeping accuracy high means a bigger `nprobe`, or redoing the groups as the data changes. For data that fits in memory, HNSW gets higher accuracy for the same amount of work and handles adding new points well, which is probably why Qdrant uses it.
