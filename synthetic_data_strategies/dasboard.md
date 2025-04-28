# 🔍 On Synthetic Data Strategies for Domain-Specific Generative Retrieval

_A Github-Optimized Dashboard for Paper Summary, Results, and Learnings.

---

# 📑 Table of Contents
- [📊 1. Overview Dashboard (Executive Summary)](#-1-overview-dashboard-executive-summary)
- [📈 2. Detailed Metrics Dashboard (Performance Insights)](#-2-detailed-metrics-dashboard-performance-insights)
  - [🧪 Synthetic Query Granularity Impact](#-synthetic-query-granularity-impact)
  - [🧪 Constraints-Based Queries Impact](#-constraints-based-queries-impact)
  - [🧪 Context2ID Importance (Memorization)](#-context2id-importance-memorization)
  - [🧪 Preference Learning: Hard Negatives vs Random](#-preference-learning-hard-negatives-vs-random)
  - [📊 Off-the-Shelf Comparisons](#-off-the-shelf-comparisons)
- [🛠️ 3. Behind the Scenes Dashboard (Data Generation + Training)](#️-3-behind-the-scenes-dashboard-data-generation--training)
  - [📚 Two-Stage Training Pipeline (ASCII Diagram)](#-two-stage-training-pipeline-ascii-diagram)
  - [✍️ Synthetic Query Types](#️-synthetic-query-types)
  - [🧠 Context2ID vs Query2ID](#-context2id-vs-query2id)
  - [🔥 Hard Negative Mining in Preference Learning](#-hard-negative-mining-in-preference-learning)
  - [🧠 LLMs Used for Synthetic Data](#-llms-used-for-synthetic-data)
- [📋 4. Final Takeaways Dashboard](#-4-final-takeaways-dashboard)
  - [✅ Top 5 Key Learnings](#-top-5-key-learnings)
  - [🚀 Future Work Ideas](#-future-work-ideas)

---

# 📊 1. Overview Dashboard (Executive Summary)

...


---

## ✍️ Synthetic Query Types
- **Chunk-Level Queries**: Capture document-wide facts
- **Sentence-Level Queries**: Capture fine-grained details
- **Constraints-Based Queries**: Inject metadata/domain info

---

## 🧠 Context2ID vs Query2ID

| Data Type | What it Trains | Purpose |
|:---------|:---------------|:--------|
| Context2ID | Memorizing document content | Enables the model to "know" the corpus |
| Query2ID | Query understanding & retrieval | Trains matching user queries to correct documents |

---

## 🔥 Hard Negative Mining in Preference Learning
- Hard negatives = **Top-ranked wrong documents**.
- Preference Optimization Loss (RPO) **favors positives** over negatives.
- Avoid random negatives ➔ hurts model quality!

---

## 🧠 LLMs Used for Synthetic Data
| Task | LLM |
|:----|:---|
| Synthetic Query Generation | Mixtral 8x7B |
| Semantic Identifier Generation | Claude 3 Sonnet |

---

# 📋 4. Final Takeaways Dashboard

## ✅ Top 5 Key Learnings
- **Sentence-level queries** and **multi-granularity** matter.
- **Domain-specific constraints** enhance retrieval.
- **Context memorization** boosts performance.
- **Hard negatives** are crucial for ranking learning.
- **Synthetic data training** alone can reach SOTA levels!

---

## 🚀 Future Work Ideas
- **Incremental updates** to models without full retraining.
- **More complex queries**: multi-hop or multi-evidence.
- **Apply strategies to dense retrieval** domain adaptation.
- **Automatic metadata extraction** to create better constraint-based queries.

---

# ✨ End of Dashboard

> Made for better readability and professional presentation on GitHub.  
> Customize it with your repo theme/colors if needed! 🎨🚀
