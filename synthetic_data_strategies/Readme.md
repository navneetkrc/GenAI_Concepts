---

---

# 📊 Part 1: Overview Dashboard (Executive Summary)

**Section:** Paper Summary Card
- **Paper Title:** *On Synthetic Data Strategies for Domain-Specific Generative Retrieval*
- **Authors:** AWS AI + CMU
- **Goal:**  
  👉 Investigate how to use **synthetic data** to train **domain-specific generative retrieval models** effectively, overcoming annotation challenges.

---

**Section:** Two-Stage Training Pipeline

| Stage | Goal | Methods |
|:-----|:----|:--------|
| **Stage 1**: Supervised Fine-Tuning (SFT) | Memorization and Generalization | - Train on *context2ID* (chunks → ID) + *query2ID* (synthetic queries → ID)<br>- Use multi-granularity (chunk-level, sentence-level)<br>- Add domain-specific constraints |
| **Stage 2**: Preference Learning (PL) | Better Ranking (Relevance) | - Use *Regularized Preference Optimization* (RPO)<br>- Hard negative mining based on retrieval results |

---

**Section:** Key Contributions (Highlight Cards)
- ✅ **Multi-Granular Synthetic Queries** improve retrieval.
- ✅ **Domain Constraints** in queries improve realism and relevance.
- ✅ **Context Memorization** via Context2ID improves performance.
- ✅ **Hard Negative Sampling** is critical for effective preference learning.

---

**Section:** Visual: High-level Workflow

```
Corpus ➔ Context Extraction ➔ Synthetic Query Generation
      ↘ Context2ID      ↘ Query2ID
          ➔ Stage 1: Supervised Fine-tuning (Memorization + Retrieval)
          ➔ Stage 2: Preference Learning (Ranking Optimization)
```

---

---

# 📈 Part 2: Detailed Metrics Dashboard (Performance Insights)

---

## Section: Synthetic Query Granularity Impact

**Goal:** Show how *sentence-level* queries improve retrieval compared to *only chunk-level*.

| Model Variant | HIT@4 | HIT@10 | MAP@10 | MRR@10 |
|:------------|:-----|:------|:------|:-----|
| Chunk-level queries only | 43.64 | 66.65 | 13.98 | 31.14 |
| Chunk + Sentence-level queries | **61.64** | **81.69** | **22.13** | **47.20** |

**📈 Chart Suggestion:**  
- Bar chart comparing the two setups.
- Y-axis: % metric values (HIT@4, HIT@10, etc.).
- Two groups of bars (Chunk-only vs Chunk+Sentence).

---

## Section: Constraints-Based Queries Impact

**Goal:** Show how **adding domain-specific constraints** boosts performance.

| Dataset | Metric | w/o Constraints | w/ Constraints |
|:--------|:------|:---------------|:-------------|
| MultiHop-RAG | HIT@4 | 61.64 | **69.98** |
| MultiHop-RAG | HIT@10 | 81.69 | **88.34** |
| AllSides | HIT@1 | 10.19 | **14.20** |
| AGNews | HIT@1 | 59.91 | **62.19** |

**📈 Chart Suggestion:**  
- Side-by-side bar charts per dataset.
- Bars: HIT@4, HIT@10, HIT@1, etc.
- Two colors: (Without Constraints vs With Constraints).

---

## Section: Importance of Context2ID Data

**Goal:** Show why memorizing document content matters!

| Dataset | Metric | w/o Context2ID | w/ Context2ID |
|:--------|:------|:--------------|:------------|
| MultiHop-RAG | HIT@4 | 41.33 | **69.98** |
| MultiHop-RAG | HIT@10 | 69.31 | **88.34** |
| Natural Questions | HIT@1 | 69.72 | **70.71** |

**📈 Chart Suggestion:**  
- Line chart or grouped bar chart
- X-axis: Dataset + Metric
- Y-axis: Score
- Two lines/bars: With vs Without Context2ID.

---

## Section: Preference Learning — Negative Sampling

**Goal:** Show that **Top-k negatives** are better than **random negatives** for preference learning.

| Method | HIT@4 | HIT@10 | MAP@10 | MRR@10 |
|:-------|:-----|:------|:------|:-----|
| SFT only (no preference learning) | 69.98 | 88.34 | 24.85 | 52.29 |
| Random Negative Sampling | 58.94 | 82.88 | 20.88 | 43.53 |
| Top-5 Hard Negatives | **71.53** | **89.62** | **26.36** | **55.40** |
| Top-10 Hard Negatives | 71.88 | 89.80 | 26.23 | 54.94 |

**📈 Chart Suggestion:**  
- Line chart showing each method's performance across HIT@4, HIT@10, etc.
- Clear color for Top-5, Top-10, Random, and Baseline.

---

## Section: Generative Retrieval vs Off-the-Shelf Baselines

**Goal:** Prove that synthetic data-based generative retrieval is competitive.

| Retriever | Performance Metric (e.g., HIT@4 for MultiHop-RAG) |
|:---------|:-------------------------------------------------|
| BM25 | Low |
| bge-large | Medium |
| Contriever-msmarco | High |
| E5-mistral-7b-instruct | Higher |
| GTE-Qwen2-7b-instruct | Higher |
| **Generative Retrieval (this paper)** | **Top Performance** ✅ |

**📈 Chart Suggestion:**  
- Simple horizontal bar chart.
- Y-axis: Model names
- X-axis: Score (higher is better).

---

# ✨ Dashboard Look

✅ Clean card layouts for each insight  
✅ Charts grouped by training stage: (SFT ➔ Preference ➔ Comparison)  
✅ Colors for *baseline* vs *improved methods*  
✅ Short annotations under each chart (1-line "takeaway" per graph)

---

---
# 📈 Part 3: Behind the Scenes Dashboard (Data Strategy and Training Flow)

---

## Section: Two-Stage Training Pipeline

**Visual: Simple Flowchart**

```
Corpus
  ↓
(Extract chunks + metadata)
  ↓
(Synthetic Query Generation by LLM)
  ↓
➔ Stage 1: Supervised Fine-Tuning
   - Context2ID (content → ID)
   - Query2ID (synthetic query → ID)
  ↓
➔ Stage 2: Preference Learning
   - Generate new synthetic queries (harder)
   - Mine Hard Negatives (Top-K errors)
   - Optimize with RPO
```

✅ Clean arrows, pastel background colors  
✅ Different box color for Stage 1 vs Stage 2

---

## Section: Synthetic Query Generation Process

**Mini Workflow Diagram:**

```
Corpus Chunk/Sentence
   ↓
Use LLM (Mixtral 8x7B)
   ↓
Three types of synthetic queries:
    - Chunk-Level Queries  (global facts)
    - Sentence-Level Queries (local details)
    - Constraints-Based Queries (domain-specific info like author, bias, topic)
```

**Note:**  
- "Constraints" like *political bias*, *author name*, *location* injected in prompts!
- Constraints improve model's ability to handle **real-world specific** queries.

---

## Section: Context2ID vs Query2ID

| Data Type | Description | Goal |
|:---------|:------------|:----|
| **Context2ID** | Chunk of text → Document ID | Force model to **memorize** document content |
| **Query2ID** | Synthetic query → Document ID | Teach model to **retrieve** relevant content from queries |

✅ Small, clean two-column table.

---

## Section: Preference Learning with Hard Negatives

**Visual: Step-by-Step Flow**

1. 🔍 Use model (after SFT) to retrieve Top-K documents for new hard queries.
2. 🔴 Select hard negatives (top retrieved docs that are wrong).
3. ✅ Positive = original ground-truth document.
4. 🏋️ Train model to prefer positive over hard negatives using **Regularized Preference Optimization (RPO)**.

✅ Add small "danger" ⚡️ sign near random negative sampling (because it's worse).

---

## Section: LLM Choices for Synthetic Data

| Task | LLM Used |
|:----|:--------|
| Synthetic Query Generation | **Mixtral 8x7B** |
| Keyword Identifier Generation | **Claude 3 Sonnet** |

✅ Add LLM logos/icons lightly (optional)

---

# ✨ Dashboard Look

✅ Flowy diagrams instead of just text  
✅ Icons for "LLM", "Corpus", "Training" steps  
✅ Very minimal text under each visual  
✅ Clear grouping: (Data Generation ➔ Stage 1 ➔ Stage 2)

---


---

# 📋 Part 4: Final Takeaways Dashboard (Key Learnings + Future Directions)

---

## Section: Key Takeaways (Highlight Cards)

✅ **1. Multi-Granularity Helps:**  
Generating synthetic queries at both *chunk* and *sentence* level captures more detailed signals ➔ **+18% HIT@4 boost**.

✅ **2. Domain Constraints Matter:**  
Instructing LLMs to add metadata like *author* or *bias* into synthetic queries makes models better handle **domain-specific searches**.

✅ **3. Memorization is Critical:**  
Training on raw context (Context2ID) helps models **memorize documents**, boosting both **retrieval** and **ranking**.

✅ **4. Hard Negatives are Essential:**  
Using *hard negative mining* (top retrieval errors) instead of random negatives **improves preference learning results** significantly.

✅ **5. Synthetic-Only Training Works:**  
Even without real user queries, synthetic data alone can train generative retrievers to beat **BM25**, **dense retrievers**, and **other strong baselines**.

---

## Section: Visual - High-Level Insights

```
Multi-Granular Queries + Context Memorization ➔ Stronger Initial Retrieval
+
Hard Negative Preference Learning ➔ Stronger Ranking
=
Competitive Domain-Specific Generative Retrieval
```
✅ Keep this flow very clean with just 3 steps connected by + and = arrows.

---

## Section: Future Opportunities (Optional Cards)

🚀 **Expand to Incremental Learning:**  
Update generative retrieval models when new documents appear without full retraining.

🚀 **More Complex Queries:**  
Generate multi-hop or multi-evidence synthetic queries for harder real-world scenarios.

🚀 **Domain Adaptation for Dense Retrieval:**  
Use similar synthetic strategies to adapt dense retrievers (e.g., DPR, Contriever) across domains.

🚀 **Automatic Metadata Extraction:**  
Enhance constraint-based queries by auto-detecting useful metadata in real-world corpora.

---

# ✨ Overall Dashboard Look:

✅ Summary cards (5 takeaways)  
✅ Simple future ideas cards (4 tiles)  
✅ A tiny visual that connects the whole training logic in one frame

---

---

# ✅ Dashboard Flow Complete!

**Full Dashboard Parts:**
- 📊 Part 1: Overview Dashboard
- 📈 Part 2: Detailed Metrics Dashboard
- 🔍 Part 3: Behind-the-Scenes Dashboard
- 📋 Part 4: Final Takeaways Dashboard

---

