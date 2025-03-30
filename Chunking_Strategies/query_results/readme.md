
# Chunking Showdown: Visualizing Naive vs. Late vs. Contextual Strategies for RAG

Retrieval-Augmented Generation (RAG) systems rely heavily on finding the *right* information to feed into a Large Language Model (LLM). But how do we ensure our retrieval system understands the context of the documents it searches? Simple "naive" chunking often breaks vital connections, leading to poor retrieval.

This post dives into a practical comparison of three chunking strategies designed to preserve context:

1.  **Naive Chunking:** The baseline - split first, embed chunks independently.
2.  **Late Chunking (Token Pooling):** Embed the whole document to get context-aware token embeddings, *then* pool these embeddings for each chunk.
3.  **Contextual Chunking:** Use an LLM to generate a summary or contextual information for each chunk *before* embedding.

We'll walk through the process of setting up the comparison, generating embeddings, running queries, and visualizing the results to understand the trade-offs.

## The Setup: Data, Models, and Tools

To make this comparison meaningful, we used real-world data and powerful embedding models:

*   **Data:** Text extracted from two technical support documents (PDFs for Samsung S24 and S25) using the `pymupdf4llm` library. This provides realistic, dense text.
*   **Embedding Model:** [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), chosen for its strong performance and large context window (8192 tokens), crucial for effective Late Chunking. We utilized the `sentence-transformers` library for easy interaction.
*   **Context Generation (for Contextual Chunking):** A summarization pipeline based on `facebook/bart-large-cnn` was used via the `transformers` library to generate context summaries.
*   **Core Libraries:** Python, `pandas`, `numpy`, `scikit-learn` (for PCA/t-SNE), `umap-learn`, `plotly`, `matplotlib`, `seaborn`.

## The Chunking Strategies Implemented

Here's a quick breakdown of how each strategy was implemented in our experiment:

### 1. Naive Chunking

*   **Process:** Each document (S24, S25) was split into fixed-size character chunks (e.g., 256 chars). Each chunk was then embedded *independently* using the Nomic model.
*   **Pros:** Simple, fast.
*   **Cons:** High risk of losing context that spans across chunk boundaries. Pronouns or references might become meaningless.

### 2. Late Chunking (Token Pooling Method)

*   **Process:**
    1.  The *entire* text of a document (S24 or S25) was fed into the Nomic model's tokenizer, respecting its long context window (e.g., 8192 tokens).
    2.  The model generated *token-level embeddings*, where each token's vector representation is influenced by the surrounding text within the entire document context.
    3.  Character boundaries for the desired chunks (e.g., 256 chars) were defined.
    4.  For each character chunk, we identified the corresponding token embeddings (using offset mapping).
    5.  The token embeddings belonging to a chunk were *averaged* (mean pooling) to create the final, single embedding vector for that chunk.
*   **Pros:** Captures document-wide context within each chunk's embedding. Theoretically better for queries requiring broader understanding. More robust than the simplified "single doc embedding" approach.
*   **Cons:** More computationally intensive than Naive (requires token embeddings for the whole doc). Relies on the embedding model's ability to produce meaningful token-level representations and handle long contexts well.

### 3. Contextual Chunking

*   **Process:**
    1.  Each document was split into initial character chunks.
    2.  For *each* chunk, a separate call was made to the BART summarization model, asking it to provide context relevant to that specific chunk based on the *entire document*.
    3.  The generated context summary was prepended or appended to the original chunk text.
    4.  This *combined* text (summary + chunk) was then embedded using the Nomic model.
*   **Pros:** Explicitly injects relevant context using an LLM. Can potentially "fill in gaps" or clarify ambiguities within a chunk.
*   **Cons:** Computationally expensive (requires multiple LLM calls for context generation). The quality heavily depends on the LLM used for summarization and the prompt engineering. Can introduce noise or hallucinations if the summary LLM isn't accurate.

## The Experiment: Querying and Comparing

1.  **Chunk Generation:** Both the S24 and S25 documents were processed using all three strategies, generating sets of chunks with associated embeddings, strategy labels, and source document labels.
2.  **Querying:** A list of ~20 representative queries (related to common phone issues, features, and manual instructions) was defined.
3.  **Similarity Calculation:** For each query, we calculated the cosine similarity between the query embedding and *all* chunk embeddings generated previously.
4.  **Top Results:** We identified the top 20 most similar chunks for *each combination* of source document (S24/S25) and chunking strategy (Naive/Late/Contextual).

## Visualizing the Differences

To understand the impact of these strategies, we generated two types of plots for each query:

### 1. Spatial Visualization (PCA / UMAP / t-SNE)

We used dimensionality reduction (PCA in this example, though UMAP/t-SNE were also options) to project the high-dimensional embeddings of the top 20 retrieved chunks (plus the query and full document reference points) into 2D space. To handle the density, we used faceting (subplots), creating one plot per strategy.

*   **Layout:** 1 row, 3 columns (Naive, Late, Contextual).
*   **Color:** Represents the source document (e.g., Blue=S24, Red=S25).
*   **Shape:** Represents the chunking strategy (e.g., Circle=Naive, Square=Late, Triangle=Contextual).
*   **Special Points:** Query (Star), Full Document Embeddings (Diamonds).

```markdown
_**Interpretation Goals:**_
*   _How scattered are the points for each strategy?_
*   _Do S24 and S25 points cluster separately?_
*   _How close are the retrieved chunks (shapes) to the query (star) within each subplot?_
*   _Where do the Late chunking squares cluster relative to the Naive circles and the Document Diamonds?_

[Placeholder for an example Spatial Plot Image - e.g., PCA_q1_how-to-fix-overheating.png]
*Caption: Example PCA plot for the query 'How to fix overheating?', showing top results. Note the separate cluster for Late chunking (squares) compared to the wider spread of Naive (circles).*
```

### 2. Similarity Distribution (Violin Plots)

These plots directly show the distribution of the cosine similarity scores for the top 20 results retrieved by each strategy, split by source document.

*   **Y-axis:** Chunking Strategy (Naive, Late, Contextual).
*   **X-axis:** Cosine Similarity Score (higher is better).
*   **Violin Shape:** Shows the density distribution of scores. Wider parts mean more chunks had that similarity score.
*   **Hue/Split:** Color and split within each violin show the distribution for S24 vs. S25 separately.
*   **Inner Sticks:** Represent individual data points (the actual similarity scores).

```markdown
_**Interpretation Goals:**_
*   _Which strategy consistently achieves higher similarity scores (violins shifted further right)?_
*   _Which strategy has a tighter distribution of high scores (less spread out)?_
*   _Are there significant differences in similarity distributions between S24 and S25 for a given strategy?_

[Placeholder for an example Violin Plot Image - e.g., violin_q1_how-to-fix-overheating.png]
*Caption: Example Violin Plot for the query 'How to fix overheating?'. We might observe that Contextual and Late chunking (split violins) show generally higher median similarities compared to Naive.*
```

## Key Takeaways & Observations

By running multiple queries and generating these visualizations (saved to a report file), several patterns emerge:

*   **Naive:** As expected, often retrieves relevant chunks but also pulls in less relevant ones, reflected in wider similarity distributions and more scattered spatial plots. Its performance is highly dependent on the query hitting keywords within a specific chunk.
*   **Late (Token Pooling):** Tends to create more cohesive clusters in the spatial plots, often centered closer to the overall document embedding than Naive chunks. This suggests it successfully incorporates document context. Its similarity scores might be more consistent, especially for queries requiring understanding beyond a single chunk. Its effectiveness hinges on the quality of the token embeddings from the long-context model.
*   **Contextual:** Performance varies significantly based on the quality of the LLM-generated summary. When effective, it can yield high similarity scores and shift chunks semantically (visible in spatial plots), potentially pulling them closer to the query if the summary adds crucial context. However, it adds computational overhead and complexity.

**Trade-offs:**

*   **Performance:** Late and Contextual generally aim to outperform Naive, especially for complex queries. The "best" depends on the data, query type, and specific models used.
*   **Complexity & Cost:** Naive is simplest. Late (Token Pooling) requires capable embedding models and more compute than Naive. Contextual adds LLM inference costs and prompt engineering effort.
*   **Interpretability:** Spatial plots help visualize semantic relationships. Violin plots give clear quantitative comparisons of retrieval scores.

## Conclusion

Choosing the right chunking strategy is crucial for effective RAG. While Naive chunking is simple, it often fails to capture necessary context. Both **Late Chunking (via token pooling)** and **Contextual Chunking** offer compelling alternatives by incorporating broader document understanding.

Our visualization approach, using faceted spatial plots and similarity distribution plots generated across multiple queries, provides a powerful way to compare these methods directly. The results highlight that Late Chunking effectively leverages long-context embedding models to maintain coherence, while Contextual Chunking relies on LLM capabilities to enrich individual chunks. The best choice depends on your specific requirements regarding retrieval accuracy, computational budget, and system complexity.


