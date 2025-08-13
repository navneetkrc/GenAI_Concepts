# Search & AI Interview Battle Card

*Compact enough for quick review right before the interview but deep enough to spark confident answers.*

---

## 1. Core Retrieval Formulas

### BM25
**Formula:** 
```
score(q,d) = Σ[t∈q] IDF(t) × [TF(t,d)(k+1)] / [TF(t,d) + k(1 - b + b × |d|/avgdl)]
```
**Key Notes:** Best for lexical ranking; handles TF saturation, doc length normalization.

### TF-IDF
**Formula:** `TF(t,d) × log(N/DF(t))`  
**Key Notes:** Baseline lexical weighting; no length normalization by default.

### Jaccard Similarity
**Formula:** `|A ∩ B| / |A ∪ B|`  
**Key Notes:** Overlap ratio of token sets; ignores term frequency.

### Cosine Similarity
**Formula:** `A·B / (|A||B|)`  
**Key Notes:** Scale-invariant similarity for vectors (text embeddings).

### L2 Distance
**Formula:** `√Σ(Ai - Bi)²`  
**Key Notes:** Lower is better; magnitude-sensitive.

---

## 2. Ranking Pipeline

**Flow:** Candidate Generation → Re-ranking → Post-Filtering

### Components:
- **Candidate Generation:** BM25, ANN search (Faiss/HNSW) to fetch top N
- **Re-ranking:** Cross-encoders / LTR models (pointwise/pairwise/listwise)
- **Post-Filtering:** Business rules (availability, location, price)

### Hybrid Search Scoring:
```
score_final = α × BM25_norm + (1 - α) × Cosine_norm
```

---

## 3. Relevance Metrics

| Metric | Formula | Example | Notes |
|--------|---------|---------|-------|
| **Precision@K** | Relevant / K | 3/5 = 0.6 | Quality of top K results |
| **Recall@K** | Relevant / Total Relevant | 3/4 = 0.75 | Coverage of relevant items |
| **MRR** | (1/|Q|) × Σ(1/rank_first) | First relevant at rank 2 → 0.5 | Mean Reciprocal Rank |
| **nDCG@K** | DCG / IDCG | - | Rewards top-ranked relevant docs |

---

## 4. System Design Essentials

### Architecture Flow:
1. **Ingestion:** Product metadata, logs, user behavior
2. **Processing:** Tokenization, lemmatization, embedding generation
3. **Indexing:**
   - **Lexical:** Elasticsearch / OpenSearch
   - **Vector:** Faiss / Milvus / Pinecone
4. **Retrieval:** Hybrid BM25 + ANN
5. **Re-ranking:** ML model (cross-encoder, GBDT)
6. **Feedback Loop:** Click logs → hard negatives → retraining

### Scaling Considerations:
- **Elasticsearch:** Sharding + replication
- **ANN Trade-offs:** Recall vs latency (efSearch, nlist parameters)
- **Fresh Data:** Incremental index updates

---

## 5. LLM in Search

### RAG Pipeline:
```
Query → Retriever → Context → LLM → Answer
```

### Key Areas:
- **Hallucination Mitigation:** Source citing, retrieval filtering, answer verification
- **Query Rewriting:** Paraphrasing, intent classification, multilingual expansion
- **Example Use Case:** Bing Chat-style narrative search over Samsung documentation

---

## 6. Project Soundbites

### Ecom Search – NRQ Reduction (-30%)
> **Impact:** Enriched metadata + generated synthetic queries → improved recall in low-coverage queries; used hybrid search to rank.

### Job/Candidate Search – Job-Candidate Match F1 ↑ 0.53→0.87
> **Impact:** Redesigned matching with better features & embedding similarity; fine-tuned model on domain-specific data.

### Ecom Search – Multilingual IR (35+ languages)
> **Impact:** Cross-language embeddings + per-language baselines; improved coverage for non-English markets.

---

## 7. Rapid-Fire Q&A Prep

**Q: Why BM25 over TF-IDF?**  
A: BM25 controls TF saturation & normalizes for document length → better real-world ranking.

**Q: Cosine or L2 for embeddings?**  
A: Cosine for normalized vectors (scale-invariant), L2 when magnitude matters.

**Q: How to reduce NRQs (No Result Queries)?**  
A: Metadata enrichment, query expansion, spelling correction, cross-sell retrieval.

**Q: How to evaluate a search model?**  
A: **Offline:** nDCG, MRR; **Online:** CTR, add-to-cart rate, NRQ drop.

**Q: What's the difference between pointwise, pairwise, and listwise ranking?**  
A: **Pointwise:** Score individual docs; **Pairwise:** Compare doc pairs; **Listwise:** Optimize entire ranking.

**Q: How do you handle embedding drift in production?**  
A: Monitor embedding distances, retrain periodically, A/B test new embeddings against baseline.

**Q: What's a typical hybrid search α value?**  
A: Usually 0.3-0.7 for BM25 weight; tune based on query type and domain.

---

## 8. Technical Deep-Dive Concepts

### Vector Search Optimization:
- **HNSW Parameters:** M (connections), efConstruction (build quality)
- **IVF Parameters:** nlist (clusters), nprobe (search clusters)
- **Product Quantization:** Compress vectors while maintaining recall

### Advanced Features:
- **Personalization:** User embeddings, collaborative filtering signals
- **Temporal Relevance:** Decay functions for time-sensitive content
- **Multi-modal Search:** Text + image embeddings, cross-modal retrieval

---
