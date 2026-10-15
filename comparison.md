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

**What I noticed:**

- A higher `ef` always made the search slower.
- A higher `ef` only made the default graph more accurate up to a point. It was already perfect at `ef=16`, so going higher just cost time.
- The under-tuned graph never got to 5/5, even at `ef=128`. On query 3 it stayed at 3/5 no matter what `ef` I used. Searching longer doesn't help when the graph itself is badly connected.
- HNSW builds its graph a little differently each time, so rebuilding the collections may change these numbers slightly.

---

## Part 4: My IVF index

My IVF index in `ivf.py` splits the 6,000 vectors into 40 groups (clusters) using k-means, about 150 vectors per group. To search, it compares the query with the 40 group centers, opens the `nprobe` closest groups, and checks every vector inside them.

| nprobe | Overlap | Search ms | Vectors checked (of 6,000) |
|---|---|---|---|
| 1 | 4.4 | 0.024 | 141 (2.4%) |
| 2 | 4.6 | 0.051 | 316 |
| 4 | 4.8 | 0.111 | 574 |
| 8 | 4.8 | 0.261 | 1,142 (19%) |
| 40 (all groups = exact search) | 5.0 | 1.606 | 6,000 |

The "vectors checked" column doesn't count the 40 comparisons with group centers.

---

## Part 5: Everything in one table

Averaged across my 5 queries:

| Method | Setting | Overlap /5 | Search ms | Round trip ms |
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
| My IVF, 40 clusters | nprobe=1 | 4.4 | 0.024 | runs in Python |
| | nprobe=2 | 4.6 | 0.051 | runs in Python |
| | nprobe=4 | 4.8 | 0.111 | runs in Python |
| | nprobe=8 | 4.8 | 0.261 | runs in Python |
| | nprobe=40 (exact) | **5.0** | 1.606 | runs in Python |

### More work means closer to exact search

Every fast method got closer to exact search when I gave it more work to do. The default HNSW went from 4.6/5 at `ef=8` to 5/5 at `ef=16` and above. The under-tuned HNSW went from 3.2 to 4.0. My IVF went from 4.4 at `nprobe=1` to 4.8 at `nprobe=8`, and reached 5.0 when it opened every group. Each time, the search also got slower: HNSW from 0.17 ms to 0.38 ms, and IVF from 0.024 ms to 0.26 ms.

This matches the speed vs accuracy trade-off from the indexing unit. Both indexes are fast because they don't check every vector. They guess where the closest vectors probably are.

- **HNSW** keeps a list of the best `ef` candidates as it walks the graph. A bigger `ef` means it looks around more, so it's less likely to stop at a "pretty good" result and miss the best one.
- **IVF** only opens the `nprobe` groups whose centers are closest. A bigger `nprobe` opens more groups, so it's less likely to miss a good result sitting just across a group's edge.

Either way, more work means checking more vectors, and that's why accuracy and time go up together. If IVF opens every group, it turns into exact search and costs the same.

Extra search work can't fix a bad index, though. My under-tuned graph never reached 5/5.

With only 6,000 vectors, exact search is already fast (0.43 ms), so HNSW is only about 2 times faster here. Exact search gets slower in proportion to the number of vectors, but HNSW slows down much more gently, so the gap would be much bigger with millions of vectors.

### Choosing a metric and choosing an index are separate decisions

In Part 2, I kept the dataset, the vectors and the search method (exact) the same and only changed the metric. The rankings still changed. For "is there any evidence that god exists", the #1 result was #2873 with cosine but #5533 with Euclidean. For "which motorcycle helmet should I buy", dot product put an electronics post first.

So the **metric** decides what "closest" means, which decides what the right answer is. The **index** (exact, HNSW or IVF) only decides how fast and how reliably I find that answer. Even a perfect index will return the wrong posts if I pick the wrong metric.

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

Raise `ef_construct` from 8 to 100. To check, I changed one setting at a time and ran my 5 queries at `ef` = default, 16 and 128:

| Collection | Overlap |
|---|---|
| m=4, ef_construct=8 (under-tuned) | 3.2 – 4.0 |
| m=16, ef_construct=8 (only m raised) | 3.4 – 4.4 |
| **m=4, ef_construct=100** (only ef_construct raised) | **5.0 at every ef** |

`ef_construct` is how many candidates Qdrant looks at *while building* the graph. With only 8, each new point gets linked to whatever neighbors a quick search found, and those are often not the real closest ones. With 100, each point gets linked to its real closest neighbors, so later searches can follow good links to the right place. Raising `m` gives each point more links, but those links are still picked by the same quick search, so it didn't help as much. For a dataset this size, 4 good links per point were enough.

**Qdrant only offers HNSW. When would a real system want IVF instead?**

- **Huge datasets and not much memory.** HNSW stores many links for every vector and works best when everything fits in RAM. IVF only needs the group centers and which group each vector belongs to. It also combines well with compressing the vectors, so it can handle billions of vectors, or read one group at a time from disk.
- **Building or rebuilding the index often.** Running k-means and sorting vectors into groups is quicker than inserting every vector into a graph. New vectors can just be added to their closest group.
- **Predictable cost.** Each IVF search checks about `nprobe × (number of vectors) / (number of groups)` vectors, so it's easy to know how much work a query will take. This also works well on GPUs, which are good at checking lots of vectors at once.
- **When slightly lower accuracy is OK.** In my test, `nprobe=1` checked only 2.4% of the vectors and still found 4.4 of the 5 correct results.

The downside is what I saw in Part 4: results near a group's edge get missed. Keeping accuracy high means a bigger `nprobe`, or redoing the groups as the data changes. For data that fits in memory, HNSW gets higher accuracy for the same amount of work and handles adding new points well, which is probably why Qdrant uses it.
