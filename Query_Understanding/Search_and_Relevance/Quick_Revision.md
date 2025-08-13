## **1. Core Search & IR Fundamentals**

They’ll expect depth, not just theory. Be ready to **explain + apply**.

* **Lexical Retrieval:** BM25, TF-IDF — formulas, when to use, limitations.
* **Vector Retrieval:** Dense embeddings (bi-encoder, cross-encoder), cosine vs L2, hybrid search scoring.
* **Query Understanding:** Normalization, tokenization, lemmatization, stopword removal, synonyms, query expansion.
* **Relevance Metrics:** NDCG, MRR, Recall\@K, Precision\@K — be ready to compute examples quickly.
* **Hybrid Search:** How to blend BM25 and vector similarity (scoring normalization).

**Prep Tip:** Be ready to solve a small “given query, doc, score” example to illustrate BM25 vs cosine difference.

---

## **2. Search Ranking & Recommendations**

In the context of experience related to recommendation/search:

* Discuss **ranking pipelines**: candidate generation → re-ranking → post-filtering.
* Show awareness of **learning-to-rank (LTR)** models (pointwise/pairwise/listwise).
* Explain **cold start handling** and **metadata enrichment** for sparse queries.
* Talk about **reducing NRQs** — exactly what you did with metadata enrichment & query generation.

**Prep Tip:** Prepare one **before–after impact story** for each major project (metrics improved, approach, trade-offs).

---

## **3. AI for Search**

They may test your **LLM-in-search** expertise:

* RAG pipeline structure, hallucination mitigation, chunking strategies.
* Prompt engineering for query rewriting, intent classification, summarization.
* Cross-language IR and multilingual embeddings.
* Multimodal retrieval (text+image+metadata).

**Prep Tip:** Be ready to explain **your Bing Chat-style Narrative Search** in 2 minutes — architecture + results.

---

## **4. System Design for Search**

They might do a **high-level system design**:

* “Design a search system for an e-commerce site” — architecture, indexing, ANN search, caching.
* Scalability: sharding, index refresh frequency, ANN recall trade-offs.
* Logging + feedback loop for continuous improvement.

**Prep Tip:** Have a **whiteboard-friendly diagram** in mind: ingestion → preprocessing → index → retrieval → ranking → feedback.

---

## **5. Coding / Problem Solving**

Even for senior roles, they might check **algorithmic thinking**:

* Small coding exercises around string matching, set similarity, vector scoring, or ranking computation.
* Efficient data structures for search (trie, inverted index, ANN graphs).

**Prep Tip:** Practice coding BM25 scoring or cosine similarity in 5-10 lines on paper.

---

## **6. Behavioral & Leadership**

Given your **Chief Engineer** title, expect questions on:

* Leading ML projects end-to-end.
* Stakeholder communication & business alignment.
* Mentoring juniors and setting tech direction.
* Handling production issues in search pipelines.

**Prep Tip:** Use **STAR** (Situation, Task, Action, Result) format with quantifiable outcomes.

---

### **Tonight’s Rapid Prep Checklist**

* Review **BM25, TF-IDF, cosine, L2, Jaccard** with numeric examples.
* Revise **ranking metrics** and be ready to calculate them.
* Rehearse **2-minute project stories** for related search work.
* Recall **system design patterns** for scalable search.
* Prepare **hybrid search score normalization** example.
* Brush up on **cross-encoder vs bi-encoder trade-offs**.

---
