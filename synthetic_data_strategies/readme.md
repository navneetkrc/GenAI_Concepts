# 🔍 On Synthetic Data Strategies for Domain-Specific Generative Retrieval

_A professional GitHub-friendly Dashboard Presentation_

---

# 📑 Table of Contents
> Jump directly to any section 🚀

- [📊 Overview Dashboard (Executive Summary)](#-overview-dashboard-executive-summary)
- [📈 Detailed Metrics Dashboard (Performance Insights)](#-detailed-metrics-dashboard-performance-insights)
  - [🧪 Synthetic Query Granularity Impact](#-synthetic-query-granularity-impact)
  - [🧪 Constraints-Based Queries Impact](#-constraints-based-queries-impact)
  - [🧪 Context2ID Importance (Memorization)](#-context2id-importance-memorization)
  - [🧪 Preference Learning: Hard Negatives vs Random](#-preference-learning-hard-negatives-vs-random)
  - [📊 Off-the-Shelf Comparisons](#-off-the-shelf-comparisons)
- [🛠️ Behind the Scenes Dashboard (Data Generation + Training)](#️-behind-the-scenes-dashboard-data-generation--training)
  - [📚 Two-Stage Training Pipeline](#-two-stage-training-pipeline)
  - [✍️ Synthetic Query Types](#️-synthetic-query-types)
  - [🧠 Context2ID vs Query2ID](#-context2id-vs-query2id)
  - [🔥 Hard Negative Mining](#-hard-negative-mining)
  - [🧠 LLMs Used for Synthetic Data](#-llms-used-for-synthetic-data)
- [📋 Final Takeaways Dashboard](#-final-takeaways-dashboard)
  - [✅ Top 5 Key Learnings](#-top-5-key-learnings)
  - [🚀 Future Work Ideas](#-future-work-ideas)

---
# 📊 Overview Dashboard (Executive Summary)

### 📄 Paper Info
- **Title:** On Synthetic Data Strategies for Domain-Specific Generative Retrieval
- **Authors:** AWS AI + CMU
- **Goal:** Overcome annotation challenges for domain-specific retrieval by using synthetic data.

---

### 🏗️ Two-Stage Training Pipeline

| Stage | Goal | Key Techniques |
|:------|:-----|:---------------|
| **Stage 1:** Fine-tuning | Memorization & Generalization | Context2ID + Query2ID (Multi-Granular + Constraints) |
| **Stage 2:** Preference Learning | Better Ranking | Hard Negative Mining + RPO |

---

### ✅ Key Contributions
- **Multi-Granularity** ➔ Detailed understanding from both chunks and sentences.
- **Domain Constraints** ➔ Improve domain relevance.
- **Context Memorization** ➔ Helps build better internal corpus knowledge.
- **Hard Negatives** ➔ Critical for ranking optimization.
- **Pure Synthetic Training** ➔ Competitive with SOTA retrieval models.

---
[🔝 Back to Top](#-table-of-contents)

---

# 📈 Detailed Metrics Dashboard (Performance Insights)

## 🧪 Synthetic Query Granularity Impact

| Setting | HIT@4 | HIT@10 | MAP@10 | MRR@10 |
|:--------|:------|:-------|:-------|:------|
| Chunk-Only Queries | 43.64 | 66.65 | 13.98 | 31.14 |
| Chunk + Sentence Queries | **61.64** | **81.69** | **22.13** | **47.20** |

> 📢 **Sentence-level queries add fine-grained strength!**

---

## 🧪 Constraints-Based Queries Impact

| Dataset | Metric | w/o Constraints | w/ Constraints |
|:--------|:-------|:---------------|:-------------|
| MultiHop-RAG | HIT@4 | 61.64 | **69.98** |
| AllSides | HIT@1 | 10.19 | **14.20** |
| AGNews | HIT@1 | 59.91 | **62.19** |

> 📢 **Domain-specific synthetic queries help a lot!**

---

## 🧪 Context2ID Importance (Memorization)

| Dataset | Metric | w/o Context2ID | w/ Context2ID |
|:--------|:-------|:--------------|:------------|
| MultiHop-RAG | HIT@4 | 41.33 | **69.98** |
| Natural Questions | HIT@1 | 69.72 | **70.71** |

> 📢 **Content memorization boosts results significantly!**

---

## 🧪 Preference Learning: Hard Negatives vs Random

| Method | HIT@4 | HIT@10 | MAP@10 | MRR@10 |
|:-------|:------|:-------|:-------|:------|
| Random Negatives | 58.94 | 82.88 | 20.88 | 43.53 |
| Top-5 Hard Negatives | **71.53** | **89.62** | **26.36** | **55.40** |
| Top-10 Hard Negatives | 71.88 | 89.80 | 26.23 | 54.94 |

> 📢 **Hard negatives > Random negatives!**

---

## 📊 Off-the-Shelf Comparisons

| Retriever | Relative Score |
|:----------|:--------------|
| BM25 | 🔵 |
| bge-large-en | 🟡 |
| Contriever-msmarco | 🟡 |
| E5-mistral-7b | 🟢 |
| GTE-Qwen2-7b | 🟢 |
| **Generative Retrieval (Synthetic)** | 🏆 |

---
[🔝 Back to Top](#-table-of-contents)

---

# 🛠️ Behind the Scenes Dashboard (Data Generation + Training)

## 📚 Two-Stage Training Pipeline


---

## ✍️ Synthetic Query Types
- **Chunk-level**: Document-wide concepts
- **Sentence-level**: Fine details
- **Constraints-based**: Metadata-driven customization

---

## 🧠 Context2ID vs Query2ID

| Data Type | Focus | Purpose |
|:----------|:------|:--------|
| Context2ID | Raw Content | Memorization |
| Query2ID | Synthetic Query | Query-to-Doc Matching |

---

## 🔥 Hard Negative Mining
- Retrieve top wrong docs after Stage 1
- Optimize model via **RPO loss** to prefer positives
- Random negatives = ❌ bad training signal

---

## 🧠 LLMs Used
| Purpose | LLM |
|:--------|:----|
| Synthetic Queries | Mixtral 8x7B |
| Document Identifiers | Claude 3 Sonnet |

---
[🔝 Back to Top](#-table-of-contents)

---

# 📋 Final Takeaways Dashboard

## ✅ Top 5 Key Learnings
- Sentence-level synthetic queries boost performance.
- Domain constraints make synthetic data more effective.
- Memorizing corpus content (Context2ID) is critical.
- Hard negatives drive better preference learning.
- Pure synthetic data can compete with SOTA models.

---

## 🚀 Future Work Ideas
- Incremental learning for new documents.
- Generate multi-hop or multi-evidence queries.
- Adapt synthetic strategies to dense retrieval.
- Automate metadata-based query generation.

---
# 🎯 End of Dashboard
