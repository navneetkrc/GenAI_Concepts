# Summary of Q&A

### Q: What are architecture patterns for information retrieval & semantic search?

*   **1. Traditional Keyword-Based IR:**
    *   **Core Idea:** Match query keywords to document keywords.
    *   **Components:** Ingestor -> Parser/Tokenizer -> **Inverted Index** -> Query Parser -> Retrieval Engine -> **Scorer (BM25/TF-IDF)**.
    *   **Pros:** Fast, mature, good for specific keywords.
    *   **Cons:** Vocabulary mismatch, no semantic understanding.
*   **2. Semantic Search (Embedding-Based):**
    *   **Core Idea:** Represent query/documents as vectors; find nearest neighbors.
    *   **Components:** Embedding Model (e.g., SBERT, CLIP) -> **Vector Database** (using **ANN** like HNSW, IVF) -> Indexing Pipeline (offline) -> Query Pipeline (online). **Two-Tower Model** is common.
    *   **Pros:** Understands semantics/context, cross-lingual/multimodal capable.
    *   **Cons:** Computationally expensive, ANN is approximate, less interpretable.
*   **3. Hybrid Search:**
    *   **Core Idea:** Combine keyword and semantic strengths.
    *   **Patterns:** Query Time Fusion (e.g., **RRF**, weighted scores), Re-ranking (retrieve broad, re-rank with semantic/Cross-Encoder), Query Augmentation.
    *   **Pros:** Often best overall relevance, robust to query types.
    *   **Cons:** More complex, requires tuning fusion.
*   **4. LLM-Augmented Search / RAG:**
    *   **Core Idea:** Use retriever (any type) to find context chunks, feed to LLM for answer synthesis.
    *   **Components:** Query Analyzer -> **Retriever** -> Context Constructor -> **Generator (LLM)** -> Validator.
    *   **Other LLM Roles:** HyDE, LLM-based re-rankers.
    *   **Pros:** Direct answers, handles complex queries, reduces hallucination (if grounded).
    *   **Cons:** High latency/cost, depends heavily on retriever quality, prompt engineering needed.

### Q: Why it’s important to have very good search?

*   **User Experience/Satisfaction:** Saves time, enables task completion, builds trust, facilitates discovery.
*   **Business Value/Revenue:** Drives e-commerce conversions, content engagement, marketplace connections, SaaS adoption, lead generation.
*   **Operational Efficiency:** Boosts employee productivity, improves decision-making, fosters collaboration, aids compliance.
*   **Competitive Advantage:** Key differentiator, improves user retention.
*   **Foundation for Advanced AI:** Crucial for reliable **RAG**, recommendations, chatbots. Poor search = poor AI outcomes.

### Q: How can you achieve efficient and accurate search results in large-scale datasets?

*   **Efficient Indexing/Retrieval:**
    *   Keyword: Sharded/replicated **Inverted Indexes** with compression.
    *   Semantic: Distributed **Vector Databases** using **ANN** (HNSW, IVF, PQ, ScaNN), hardware acceleration (GPU/TPU).
*   **Accurate Ranking/Relevance:**
    *   **Hybrid Search:** Combine keyword (BM25) + semantic, fuse results (**RRF**, weighted scores).
    *   **Multi-stage Ranking (LTR):** Retrieve broad -> Fast ranker -> Powerful **Re-ranker (Cross-Encoders, LTR models)** on top-k.
    *   **Feature Engineering:** Use diverse signals (text, doc quality, user interaction, personalization).
    *   **Semantic Quality:** High-quality, fine-tuned embeddings; query understanding (expansion, HyDE).
*   **System Architecture/Optimization:**
    *   Distributed systems, extensive **caching**, async processing, efficient data pipelines.
*   **Data Quality/Iteration:**
    *   Clean data, optimized **chunking**, continuous evaluation (offline: **NDCG**, MAP; online: A/B tests, user metrics), monitoring.

### Q: Improving inaccurate retrieval in an existing RAG system?

*   **Phase 1: Diagnosis:**
    *   Collect failure examples, inspect retrieved chunks (relevance, completeness).
    *   Understand current architecture (keyword/semantic/hybrid, models, index, chunking).
*   **Phase 2: Targeted Improvements:**
    *   **Chunking Strategy:** Adjust size, overlap, use content-aware methods, add metadata.
    *   **Embedding Model:** Try better pre-trained models, **fine-tune** on domain/task data.
    *   **Hybrid Search:** Implement/tune combination of keyword + semantic, use **RRF** for fusion.
    *   **Query Processing:** Add expansion, rewriting, spelling correction, **HyDE**.
    *   **Re-ranking:** Implement a re-ranker (**Cross-Encoders**, LTR models) after initial retrieval.
    *   **Parameter Tuning:** Optimize BM25 (k1, b), ANN params (`ef_search`, `nprobe`).
*   **Phase 3: Evaluation & Iteration:**
    *   Establish retrieval metrics (**Hit Rate, MRR, NDCG**).
    *   Test changes iteratively (offline evaluation, A/B testing).
    *   Create feedback loop from RAG output quality to retriever improvements.

### Q: Explain the keyword-based retrieval method

*   **Core Idea:** Match literal query keywords to keywords in documents.
*   **Process:**
    1.  **Preprocessing (Indexing):** Text extraction, Tokenization, Normalization (lowercase, stop words, stem/lemma).
    2.  **Inverted Index:** Build map: `term -> [ (doc_id, term_freq, position), ...]`.
    3.  **Query Processing:** Apply same preprocessing to query.
    4.  **Retrieval:** Look up query terms in index, combine posting lists (AND/OR).
    5.  **Scoring/Ranking:** Calculate relevance using **TF-IDF** or (better) **BM25** (considers term saturation, doc length).
*   **Strengths:** Fast, mature, precise for specific terms.
*   **Weaknesses:** Vocabulary mismatch, no semantic understanding, needs normalization.

### Q: How to fine-tune re-ranking models?

*   **Goal:** Re-order candidates from initial retrieval for higher precision.
*   **Model Types:** **Cross-Encoders** (BERT-like, input=`(query, doc)`, high accuracy, slow), **LTR Models** (XGBoost/NNs on features, faster).
*   **Training Data:** Need `(query, candidate_doc, relevance_label)`. Labels from humans (gold) or implicit feedback (clicks - noisy). Generate instances using first-stage retriever's output.
*   **Fine-tuning Approach:**
    *   **Pointwise:** Predict absolute relevance (regression/classification). Loss: MSE/Cross-Entropy. Simple, but doesn't directly optimize rank.
    *   **Pairwise:** Predict relative order of two docs. Loss: Hinge/RankNet. Better for ranking than pointwise.
    *   **Listwise:** Optimize whole list ranking metric (e.g., NDCG). Loss: LambdaRank/LambdaMART, ListNet. Theoretically best, most complex.
*   **Training Loop:** Load pre-trained model (if cross-encoder), format input, use appropriate optimizer (AdamW) / LR scheduler, batch correctly for chosen approach.
*   **Evaluation:** Use held-out test set, evaluate with **NDCG@k**, MAP@k, MRR.
*   **Deployment:** Deploy as service, monitor online via A/B tests. Iterate with fresh data.

### Q: Explain most common metric used in information retrieval and when it fails?

*   **Most Common/Important:** **NDCG@k** (Normalized Discounted Cumulative Gain).
*   **Other Key Metrics:**
    *   **Precision@k (P@k):** % relevant in top `k`. *Fails:* Ignores recall, rank within `k`, graded relevance.
    *   **Recall@k (R@k):** % of *all* relevant found in top `k`. *Fails:* Needs total relevant count (often unknown), ignores precision, rank within `k`.
    *   **F1 Score@k:** Harmonic mean of P@k, R@k. *Fails:* Inherits P/R limits, assumes equal P/R importance.
    *   **MAP (Mean Average Precision):** Rank-aware avg precision for binary relevance. *Fails:* Binary relevance only, less intuitive score.
    *   **MRR (Mean Reciprocal Rank):** Rank of *first* relevant item. *Fails:* Ignores all subsequent relevant items.
    *   **NDCG@k:** Handles graded relevance, position-aware (discounted), normalized. *Fails:* Requires costly graded judgments, specific gain/discount choices arbitrary, doesn't inherently measure diversity.
*   **General Limitations:** Offline metrics are static, don't capture full UX (latency, UI), subject to judgment bias, averaging hides specific failures. **Need online A/B testing.**

### Q: Which evaluation metric for a Quora-like Q&A system (most pertinent answers quickly)?

*   **Chosen Metric:** **NDCG@k** (with small `k`, e.g., 3 or 5).
*   **Reasoning:**
    *   Handles **Graded Relevance:** Answers vary in quality ("most pertinent").
    *   **Positionally Sensitive:** Discount factor rewards finding best answers quickly (higher ranks).
    *   **Top-K Focus:** Aligns with users seeing first few answers.
    *   **Normalization:** Allows comparison across questions.
*   **Why not others?** P@k/MAP ignore graded relevance; R@k not primary goal; MRR only considers the *first* relevant hit, not necessarily the *best* one if multiple relevant exist.

### Q: Which metric to evaluate a recommendation system?

*   **Depends entirely on the system's goal.** Often use a combination.
*   **Offline Metrics:**
    *   *Accuracy/Relevance:* **P@k, R@k, MAP@k, NDCG@k, MRR** (choose based on goal - see comparison). **NDCG@k** often preferred if graded relevance available/inferable.
    *   *Beyond Accuracy:* **Catalog Coverage** (long-tail exposure), **Diversity** (item similarity within list), **Novelty/Serendipity** (unexpected relevant items).
*   **Online Metrics (A/B Testing - Ground Truth):**
    *   *User Engagement:* **CTR, CVR**, Session Duration/Depth, Watch Time.
    *   *Business:* Revenue, Subscription/Retention Rate.
    *   *User Satisfaction:* Surveys, Ratings.
*   **How to Choose:** Define goal (sales, engagement, discovery?) -> Consider data (explicit ratings?) -> Use multiple metrics (offline + online) -> A/B test.

### Q: Compare different information retrieval metrics and which one to use when?

*   **P@k:** Use when only top `k` seen, density matters (image grid).
*   **R@k:** Use when finding *all* relevant is critical (legal, medical).
*   **F1@k:** Use for simple P/R balance.
*   **MAP:** Use for rank-aware binary relevance evaluation (standard doc retrieval).
*   **MRR:** Use for finding *first* correct answer quickly (Q&A, known-item).
*   **NDCG@k:** Use for graded relevance, optimizing top-k order (web search, recs, Q&A - often best if feasible).
*   **Guidance:** Start with NDCG/MAP. Use MRR for single-answer tasks. Use P@k/R@k/F1 when simpler views needed or constraints exist. Monitor multiple metrics; **validate online**.

### Q: How does hybrid search works?

*   **Goal:** Combine keyword precision + semantic understanding.
*   **Process:**
    1.  **Parallel Execution:** Query sent to Keyword System (BM25/Inverted Index) AND Semantic System (Embeddings/Vector Index).
    2.  **Result Fusion:** Merge the two ranked lists.
        *   **RRF (Reciprocal Rank Fusion):** Preferred method. Score based on `1 / (k + rank)` in each list. Ignores original scores.
        *   **Score-Based:** Normalize scores (min-max, z-score), combine with weights (`alpha`). Requires careful tuning.
    3.  **(Optional) Re-ranking:** Apply powerful model (Cross-Encoder) to top results of fused list.
*   **Advantages:** Improved relevance, robustness to query types.
*   **Challenges:** Complexity, fusion tuning, potential latency increase.

### Q: How to merge and homogenize rankings from multiple methods?

*   **Challenge:** Scores from different systems (BM25, cosine sim, popularity) are incompatible.
*   **Methods:**
    1.  **Rank-Based Fusion (RRF):** Ignores scores, uses only rank (`1 / (k + rank)`). Simple, robust, often best baseline.
    2.  **Score-Based Fusion:** Requires **normalization** (min-max, z-score) then combination (weighted sum). Sensitive to normalization, requires weight tuning.
    3.  **Learning-to-Rank (LTR) Re-ranking:** Treat source scores/ranks as features for an ML model trained on relevance labels. Most complex, potentially most accurate, needs labeled data.
*   **Guidance:** Start with RRF -> Try Score-Based if confident in normalization/tuning -> Use LTR for highest accuracy if resources/data permit. Evaluate rigorously (offline + online).

### Q: How to handle multi-hop/multifaceted queries?

*   **Challenge:** Queries require reasoning across docs/data or have multiple constraints.
*   **Techniques:**
    1.  **Query Decomposition:** Break complex query into simpler sub-queries (using NLU/LLM), execute sequentially/parallelly.
    2.  **Iterative Retrieval:** Perform initial search, extract entities/info, use info for next search (agentic frameworks, LLM as reasoner).
    3.  **Knowledge Graph (KG) Integration:** Link query entities to KG, traverse graph relationships (SPARQL/Cypher). Good for structured relationships.
    4.  **Structured Data/Faceted Search:** Extract structured constraints (metadata filters) and combine with text/semantic search.
    5.  **Advanced RAG:** Retrieve context specifically for each hop/facet, feed comprehensive context + original query to LLM for synthesis.
*   **Approach:** Often hybrid; analyze query type -> route to appropriate strategy (KG, decomposition, faceted search, iterative retrieval) -> potentially synthesize with RAG.

### Q: What are different techniques to be used to improved retrieval?

*   **1. Input Data/Indexing:** Better cleaning, optimized **chunking**, metadata enrichment, tuning index params (BM25 k1/b).
*   **2. Query Understanding:** **Query expansion** (synonyms, embeddings, LLM rewrite), spelling correction, intent classification, decomposition, **HyDE**.
*   **3. Core Algorithms:** Use BM25/better ANN, use state-of-the-art embedding models, **fine-tune embeddings** (domain/task specific), implement **Hybrid Search** (Keyword + Semantic).
*   **4. Post-Retrieval:** **Re-ranking** (**Cross-Encoders**, LTR models), diversification (MMR), personalization, filtering/business rules.
*   **5. User Feedback:** Leverage implicit (clicks, dwell time) and explicit (ratings) feedback for training LTR / fine-tuning embeddings.
*   **Overall:** Requires iterative development, offline (**NDCG**, MAP) and online (A/B tests) evaluation.
