## Study Materials: Multimodal Search, RAG, LLMs

### Initial System Instructions:
Focus on topics such as
*   Multimodal Search
*   Search and Recommendations
*   LLM Fine-tuning
*   AI Research

The user requested detailed answers covering:
*   **Technical Concepts:** Transformer architecture, attention, fine-tuning (instruction tuning, PEFT), decoding.
*   **Practical Applications:** Deploying models, handling class imbalance, evaluating LLMs.
*   **Advanced Topics:** Hallucination control, prompt engineering, multimodal systems.
*   **Research Questions:** Optimizing LLMs for proprietary data, future applications.
*   **General:** Clear explanations, examples, practical insights, tips for behavioral interviews, and research presentation.

---

### Q&A Summary:

**Q1: What is chunking, and why do we chunk our data?**

*   **What:** Breaking large text/data into smaller, manageable segments (**chunks**).
*   **Why:**
    *   Overcome **LLM context window** limitations.
    *   Improve **retrieval relevance** for RAG/Search by focusing search on smaller, relevant segments.
    *   Enhance **embedding quality** by providing focused context.
    *   Improve **computational efficiency** (embedding, indexing, retrieval).
    *   Facilitate **granular operations** (summarization, analysis on parts of text).

**Q2: What factors influence chunk size?**

*   **LLM Context Window Limit:** Hard constraint; need space for query, prompt, output.
*   **Embedding Model Performance:** Models have optimal input lengths; too short/long degrades quality.
*   **Nature of the Data:** Dense vs. narrative; information locality (sentence vs. paragraph).
*   **Task Requirements:** Granularity needed for QA vs. summarization.
*   **Retrieval Precision/Recall Trade-off:** Smaller = higher precision, maybe lower recall context; Larger = more context, maybe lower precision.
*   **Computational Resources & Latency:** Cost of embedding, storing, retrieving many small vs. fewer large chunks.
*   **Chunk Overlap:** Interacts with size; mitigates context loss at boundaries.
*   **Process:** Start with baseline, consider data, **experiment & evaluate**.

**Q3: What are the different types of chunking methods?**

*   **Fixed-Size:** Simple, predictable size, ignores semantics (can cut mid-sentence). Often uses overlap.
*   **Character/Delimiter-Based (Recursive):** Common (e.g., LangChain); uses hierarchy of separators (`\n\n`, `\n`, `.`, ` `). More structure-aware than fixed-size.
*   **Document Structure-Aware:** Uses format structure (Markdown headers `#`, HTML tags `<div>`). Preserves logical organization. Needs parser.
*   **Linguistic Unit (Sentence Splitting):** Uses NLP (NLTK, spaCy) for sentence boundaries. Semantically meaningful but variable length.
*   **Code Chunking:** Language-specific, uses syntax (functions, classes).
*   **Semantic Chunking:** Advanced; uses embedding similarity to find topic shifts/breaks. Aims for coherent chunks, computationally expensive.

**Q4: How to find the ideal chunk size?**

*   **Empirical Process:** Not one magic number; requires experimentation.
*   **Steps:**
    1.  **Define Goal & Metrics:** What to optimize (Retrieval relevance - NDCG, MAP; RAG quality - faithfulness; Latency; Cost).
    2.  **Understand Constraints:** LLM context window, embedding model limits, budget.
    3.  **Characterize Data:** Density, structure, length.
    4.  **Establish Baseline & Hypothesis:** Choose starting sizes (e.g., 256, 512, 1024 tokens), method, overlap.
    5.  **Set Up Evaluation:** Golden dataset (queries/prompts), ground truth (optional), automated metrics, human evaluation plan.
    6.  **Run Experiments Systematically:** Isolate chunk size variable.
    7.  **Analyze Results:** Quantitative (metrics) and *qualitative* (review chunks/answers). Identify trade-offs.
    8.  **Iterate & Refine:** Narrow down range, test variations, potentially change chunking *method*.

**Q5: What is the best method to digitize and chunk complex documents like annual reports?**

*   **Phase 1: Digitization (High-Fidelity Extraction):**
    *   Use **advanced Document AI tools** (AWS Textract, Azure AI DI, Google Doc AI) for layout, tables, OCR.
    *   Prioritize **layout preservation** (reading order).
    *   Specialized **table extraction** (output as Markdown/HTML).
    *   Handle figures: Extract **captions/context**, use **multimodal models** for descriptions (optional), or ignore.
    *   Clean (headers/footers), structure output.
*   **Phase 2: Chunking (Hybrid, Structure-Aware):**
    *   Primary: Chunk by **document sections/subsections** (using headers).
    *   Treat **tables as distinct units** (Markdown representation + caption/context).
    *   Secondary: If sections/text blocks are too large, use **recursive character/sentence splitting** *within* them.
    *   Handle **figure captions** separately.
    *   Crucial: Attach **rich metadata** (source, page, section, content type) to *every* chunk.
*   **Recommendation:** DocAI Service -> Parse Structure -> Hierarchical Chunking (Sections > Tables > Internal Text Splitting) -> Rich Metadata -> Evaluate.

**Q6: How to handle tables during chunking?**

*   **Accurate Extraction First:** Use Document AI tools.
*   **LLM-Friendly Representation:** **Markdown** often best; HTML, JSON, text possible.
*   **Chunking Strategies:**
    *   A) **Atomic Chunk:** Entire table + caption/context (if fits size limit). Preserves integrity.
    *   B) **Splitting Large Tables:** Row-based split, **repeat headers** in each chunk. Handles size limits, complex.
    *   C) **Table Summarization:** Use LLM to summarize, chunk summary. Loses detail.
    *   D) **Hybrid:** Separate chunks for table and surrounding text, link via metadata.
*   **Metadata:** Essential (type, caption, ID, page, section).
*   **Recommendation:** Extract accurately -> Markdown -> Try Atomic -> Fallback to Splitting (repeat headers) -> Metadata -> Evaluate.

**Q7: How do you handle very large table for better retrieval?**

*   **Goal:** Retrieve specific relevant *parts*.
*   **Strategies:**
    1.  **Row-Level Chunking:** Chunk each row (or small group) + **prepend headers/caption** to each. High granularity, many chunks.
    2.  **Logical Section Splitting:** Split based on visual/logical breaks within table (if any).
    3.  **Summary + Row Chunking (Hybrid):** Index both LLM-generated summary and granular rows.
    4.  **Parent Document Retriever:** Index rows, store full table separately, retrieve rows then optionally fetch full table context.
    5.  **Metadata Filtering:** Add rich metadata (row index, key values) and use filters during search (e.g., search vector *AND* `key_column_value` = "Q2 2023"). Very effective.
    6.  **Embed Structure Differently:** Separate embeddings for data vs. summary/structure.
*   **Recommendation:** Accurate Extraction -> Markdown -> Row-level chunks w/ headers -> **Rich Metadata + Filtering** -> Consider Parent Retriever -> Evaluate.

**Q8: How can I ensure table headers are not lost during chunking?**

*   **Prerequisite:** Accurate header detection during extraction (DocAI tools).
*   **Representation:** Use structure-aware format (Markdown).
*   **Strategies:**
    1.  **Atomic Chunking:** If table fits, headers are naturally included.
    2.  **Replicate Headers (Most Robust for Splits):** When splitting large tables (by row, section), explicitly **prepend the header row(s)** to *each* resulting chunk.
    3.  **Row-Level Chunking:** Prepend headers to *every* single-row chunk.
*   **Context:** Include table caption/title in chunks.
*   **Metadata:** Store header info (optional, but replication in content is key for LLM).
*   **Key:** **Replicate headers** whenever splitting table data rows.

**Q9: What strategies can prevent table rows from being split across chunks?**

*   **1. Atomic Table Chunking:** If table fits, no internal rows are split.
*   **2. Row-Aware Chunking Logic:** When splitting must occur: Iterate through data rows, **check size *before* adding a complete row**. If next row exceeds limit, finalize current chunk, start new chunk (with replicated headers), add the row to the new chunk. Ensures splits happen *between* rows.
*   **3. Chunk by Logical Row Groups:** Similar to #2, but operate on groups of rows.
*   **4. Use Explicit Row Delimiters:** Less reliable; configure splitter to prioritize `\n` (or row delimiter) but long rows might still get split by character limit.
*   **5. Avoid Naive Text Splitting:** Don't use basic character/token splitters directly on table text.
*   **Recommendation:** Atomic if possible, otherwise **Row-Aware Logic**.

**Q10: How to handle list item during chunking?**

*   **Goal:** Maintain item integrity and list context.
*   **Strategies:**
    1.  **Structure-Aware Parsing (Best):** Use Markdown/HTML parsers. Chunk whole list, individual items, or groups of items. Handles nesting.
    2.  **Recursive Splitting w/ List Separators:** Simpler; add separators like `\n* `, `\n1. ` to hierarchy. Brittle to formatting variations.
    3.  **Treat Each Item as Separate Doc:** Pre-process to isolate items, then chunk individually. Loses surrounding list context easily.
    4.  **Secondary Splitting Within Long Items:** If a single list item > chunk size, isolate it, then apply sentence/recursive splitting *internally*.
*   **Considerations:** Intro sentence context, nesting, metadata (type, level, ID), whitespace.
*   **Recommendation:** Parser if possible (1), else Recursive w/ list separators (2). Use Secondary Splitting (4) for oversized items.

**Q11: How do you build production grade document processing and indexing pipeline?**

*   **Principles:** Scalability, Reliability (queues, retries, DLQ), Monitoring (logs, metrics, traces, alerts), Maintainability (modularity, IaC, CI/CD), Efficiency/Cost, Security, Testability, Versioning.
*   **Stages:**
    *   **Ingestion:** Sources, triggers, validation, deduplication, **queuing (SQS, Kafka)**.
    *   **Extraction/Parsing:** Worker pool, robust **DocAI tools/libs**, error handling (DLQ).
    *   **Preprocessing:** Normalization, cleaning, metadata enrichment, PII handling.
    *   **Chunking:** Configurable strategy, table/list logic, **metadata propagation**.
    *   **Embedding:** **Model serving endpoint**, batching, retries, versioning.
    *   **Indexing:** Choose **Vector DB** (Pinecone, Weaviate, Milvus, etc.), batch upserts, metadata storage, configure index params.
*   **Supporting Infra:** **Orchestration** (Step Functions, Airflow), **IaC** (Terraform, CDK), **CI/CD**, Config Mgmt, **Monitoring Stack**, Evaluation loop.

**Q12: How can I handle list items that exceed the maximum chunk size?**

*   **1. Isolate Item:** Identify the full text of the single oversized list item.
*   **2. Secondary Internal Chunking:** Apply sentence splitting or recursive splitting *only to the text within* this item.
*   **3. Maintain Context via Metadata:** Tag sub-chunks with `original_list_item_id`, `list_item_part_index` (e.g., 1/N), `is_continuation` flag.
*   **4. Include List Marker:** Add original marker (`*`, `1.`) to the *first* sub-chunk's text.
*   **5. Consider Overlap:** Use overlap for internal splits if needed.
*   **Key:** Isolate -> Split Internally -> Link with Metadata.

**Q13: How to handle graphs & charts in RAG?**

*   **Challenge:** Visual data ignored by text-only RAG.
*   **Strategies:**
    1.  **Extract Captions/Surrounding Text:** Baseline. Chunk text descriptions. LLM doesn't see chart.
    2.  **Generate Textual Descriptions:** Use multimodal model (GPT-4V, LLaVA) offline to create summaries/descriptions of charts. Chunk the *description*.
    3.  **Extract Underlying Data:** Use advanced DocAI if chart data is extractable (e.g., from source tables). Represent as Markdown table, chunk the table. Often difficult/impossible.
    4.  **Multimodal RAG (Advanced):** Embed images (CLIP). Index image & text embeddings. Retrieve relevant images/text. Feed *text AND images* to a **multimodal LLM** (GPT-4V, Gemini). Most complex, highest potential.
*   **Recommendations:** Start simple (1), add (2) or (3) if needed. Use (4) for critical visual tasks if complexity/cost justified. Hybrid approaches common. **Metadata linking** text/images is vital.

---
