# Search & Relevance — Interview Cheat Sheet

**Goal:** Retrieve documents for a query, then **score** them by how well they match.
**Common pipelines:**

* **Lexical** (bag‑of‑words): Inverted index → BM25 / TF‑IDF.
* **Semantic** (vectors): Encoder → Similarity (cosine / L2).
* **Set overlap:** Token sets → Jaccard.

---

## Quick Comparison (at a glance)

| Metric      | What it measures                             | Input needed                          | Range              | Bigger = better?   | Scale sensitivity                           | Typical use                         |
| ----------- | -------------------------------------------- | ------------------------------------- | ------------------ | ------------------ | ------------------------------------------- | ----------------------------------- |
| **BM25**    | Probabilistic term matching with length norm | Term counts, doc length, corpus stats | Unbounded          | Yes                | Robust to length (b) and TF saturation (k₁) | Web search, classic IR ranking      |
| **TF‑IDF**  | Term importance (local TF × global rarity)   | Term counts, DF                       | Unbounded          | Yes                | No length norm by default                   | Baseline ranking, features          |
| **Jaccard** | Set overlap of tokens                        | Token sets                            | 0–1                | Yes                | Ignores term frequency                      | Dedup, near‑exact match, short text |
| **Cosine**  | Angle between vectors                        | Dense/sparse vectors                  | –1…1 (usually 0–1) | Yes                | Scale‑invariant                             | Embedding search, semantic          |
| **L2**      | Straight‑line distance                       | Dense vectors                         | ≥0                 | **Lower = better** | Sensitive to magnitude                      | Embedding search (ANN)              |

---

## 1) BM25 (Best Match 25)

**Formula (per query term *t*):**

$$
\text{BM25}(q,d)=\sum_{t\in q} IDF(t)\cdot \frac{TF(t,d)\cdot (k_1+1)}{TF(t,d)+k_1\cdot\big(1-b+b\cdot \frac{|d|}{avgdl}\big)}
$$

$$
IDF(t)=\ln\Big(\frac{N-n_t+0.5}{n_t+0.5}+1\Big)
$$

**Knobs:** $k_1\approx[1.2,2.0]$ controls TF saturation, $b\approx0.75$ controls length normalization.

**Mini‑example:**

* $N=1000$, $n_t=100$ ⇒ $IDF\approx \ln(9.96)\approx 2.30$
* $TF=3$, $k_1=1.5$, $b=0.75$, $|d|=100$, $avgdl=120$
  Denominator factor $=k_1(1-b+b|d|/avgdl)=1.5(0.25+0.75\cdot0.833)\approx1.3125$
  Term score $=2.30 \times \frac{3\cdot2.5}{3+1.3125}\approx 2.30\times1.739\approx **4.0**$

**Pros:** Great real‑world ranking; handles long docs.
**Cons:** Purely lexical; synonyms missed.
**Use when:** Primary ranking over an inverted index.

---

## 2) TF‑IDF

**Formula:**

$$
TF\_IDF(t,d)=TF(t,d)\times \log\frac{N}{DF(t)}\quad(\text{base }10\text{ or }e,\text{ be consistent})
$$

**Mini‑example:** $TF=3$, $N=100$, $DF=10$ ⇒ $IDF=\log(10)=1$ ⇒ score $=3$.

**Pros:** Simple, fast baseline; captures rarity.
**Cons:** No length norm; linear TF growth; lexical only.
**Use when:** Baseline ranking or features for classical models.

---

## 3) Jaccard Similarity (token sets)

**Formula:**

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

**Mini‑example:** Query `{apple, phone}`, Doc `{apple, launch, phone}` ⇒ $J=2/3\approx0.67$.

**Pros:** Very simple; good for dedup/near‑exact matching.
**Cons:** Ignores term frequency and order; brittle to morphology/stopwords.
**Use when:** Short strings, titles, dedup, candidate filtering.

---

## 4) Cosine Similarity (vectors)

**Formula:**

$$
\cos(\theta)=\frac{A\cdot B}{\|A\|\ \|B\|}
$$

**Mini‑example:** $A=[1,2],\ B=[2,3]$
Dot $=8$; $\|A\|=\sqrt{5}$, $\|B\|=\sqrt{13}$ ⇒ cosine $\approx 8/(2.236\cdot3.606)\approx **0.99**$.

**Pros:** Scale‑invariant; works great with embeddings.
**Cons:** Requires good vectorization; not an inverted‑index score.
**Use when:** Semantic retrieval with dense vectors.

---

## 5) L2 (Euclidean) Distance

**Formula:**

$$
L2(A,B)=\sqrt{\sum_i (A_i-B_i)^2}
$$

**Mini‑example:** $[1,2]$ vs $[2,3]$ ⇒ $\sqrt{(−1)^2+(−1)^2}=\sqrt{2}\approx **1.41**$ (lower is better).

**Pros:** Natural geometric distance; used by many ANN libs.
**Cons:** Sensitive to vector magnitudes; needs normalization.
**Use when:** ANN over normalized embeddings or specific vector spaces.

---

## When to Use What (fast rules)

* **BM25 vs TF‑IDF:** Prefer **BM25** for ranking search results; TF‑IDF is fine as a baseline or feature.
* **Cosine vs L2:** If vectors are **length‑normalized**, cosine ≈ monotonic with dot product; L2 is fine but be mindful of scale.
* **Jaccard:** Great for **exact/near‑exact** matching, short text, and duplicate detection; not for nuanced ranking.

---

## Preprocessing Tips (matter in interviews)

* **Tokenization & normalization:** lowercase, remove punctuation; consider stemming/lemmatization.
* **Stopwords:** Remove for Jaccard/TF‑IDF/BM25 to reduce noise (task‑dependent).
* **Length normalization:** Built into BM25; consider for TF‑IDF (e.g., cosine over TF‑IDF vectors).
* **Embeddings:** Normalize vectors for cosine; consider whitening or mean pooling for sentence embeddings.

---

## Complexity & Indexing

* **BM25/TF‑IDF:** Inverted index; score is **O(#postings for query terms)**.
* **Jaccard:** **O(|A|+|B|)** with hashed sets.
* **Cosine/L2 (dense):** **O(d)** per pair; use **ANN** (HNSW/IVF‑PQ) for scale.

---

## Tiny Pseudocode Snippets

**BM25 scoring (per query):**

```python
score = 0
for t in query_terms:
    tf = tf_in_doc[t]
    idf = log((N - df[t] + 0.5)/(df[t] + 0.5) + 1)
    norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc)/avgdl))
    score += idf * norm
```

**Cosine similarity:**

```python
cos = dot(a, b) / (norm(a) * norm(b))
```

---

## Common Interview Traps (and fixes)

* **“Why BM25 over TF‑IDF?”** → TF saturation + length norm → better real‑world ranking.
* **“Cosine or L2?”** → If vectors normalized, **cosine** is robust; L2 sensitive to magnitude.
* **“Jaccard seems low—why?”** → It ignores frequency; use BM25/TF‑IDF or embeddings for richer signals.
* **“Mix lexical & semantic?”** → Yes: **hybrid search** (BM25 score + α·cosine).
* **“IDF base?”** → Any base; be consistent (often natural log).

---

### One‑liners to remember

* **BM25**: TF saturates; length‑aware; state‑of‑the‑art lexical ranker.
* **TF‑IDF**: Simple rarity‑weighted counts; great baseline.
* **Jaccard**: Overlap of sets; frequency‑blind.
* **Cosine**: Angle between vectors; scale‑invariant.
* **L2**: Straight‑line distance; lower is better; scale‑sensitive.
