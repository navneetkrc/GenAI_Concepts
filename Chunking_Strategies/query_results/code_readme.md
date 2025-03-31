## Beyond Naive Splitting: Comparing Chunking Strategies for RAG with Nomic Embeddings

Retrieval-Augmented Generation (RAG) has revolutionized how we interact with large language models (LLMs), allowing them to access and reason over external knowledge. A critical, yet often underestimated, component of RAG is **chunking**: splitting large documents into smaller, manageable pieces for embedding and retrieval.

Simply splitting text every N characters (Naive Chunking) is easy but often suboptimal. It can break sentences mid-thought, separate related concepts, or fail to provide enough context for the embedding model.

This post dives into a practical comparison of three chunking strategies:

1.  **Naive Chunking:** The simple baseline.
2.  **Late Chunking (Token Pooling):** Embeds the whole document first, then intelligently pools token embeddings for each chunk.
3.  **Contextual Chunking:** Enriches each chunk with a summary of the entire document before embedding.

We'll use the powerful `nomic-ai/nomic-embed-text-v1.5` model, known for its large context window and strong performance, and apply these strategies to real-world Samsung Galaxy S24 and S25 user manuals. We'll evaluate their effectiveness using multiple user queries and visualize the results to understand the trade-offs.

**(Code analyzed is based on the user-provided notebook "Chunking Strategies Compared Version 4")**

### The Core Problem: Why Chunking Matters

Embedding models transform text into dense numerical vectors (embeddings). Similar concepts should have similar vectors. In RAG, when a user asks a query, we embed the query and search our vector database for the *most similar* text chunks from our documents. These relevant chunks are then fed to an LLM along with the original query to generate a grounded answer.

The quality of the chunks directly impacts retrieval quality:
*   **Too small/context-poor:** Embeddings might be ambiguous or miss crucial information.
*   **Too large:** May exceed model context limits or dilute the core topic with irrelevant details.
*   **Poorly split:** Can separate related sentences, hindering semantic understanding.

### Chunking Strategies Explored

Let's break down the logic, implementation, and pros/cons of each strategy used in the code.

#### 1. Naive Chunking

*   **Concept:** The simplest approach. Split the document into fixed-size character chunks with potential overlap (though overlap wasn't explicitly used in the provided function).
*   **Flowchart:**

    ```mermaid
    graph LR
        A[Start: Document Text, Chunk Size] --> B{Iterate through document};
        B -- Get next chunk --> C[Chunk_i (Size S)];
        C --> D{Embed Chunk_i};
        D --> E[Store (Chunk_i, Embedding_i)];
        B -- More text? --> C;
        B -- No more text --> F[End: List of (Chunk, Embedding)];
    ```
*   **Implementation:**

    ```python
    # Cell 4: Define Naive Chunking function
    def naive_chunking(document, chunk_size=512, source_label=""):
        """
        Split the document into chunks first, then embed each chunk individually.
        Includes a source label and debug print.
        """
        chunks = [document[i:i + chunk_size] for i in range(0, len(document), chunk_size)]
        results = []
        print(f"  Naive Chunking ({source_label}): Processing {len(chunks)} potential chunks...")
        # ... (loop through chunks)
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                try:
                    # Embed each chunk independently
                    embedding = embedding_model.encode(chunk, convert_to_numpy=True)
                    results.append({
                        'text': chunk,
                        'embedding': embedding,
                        'strategy': 'Naive',
                        'source': source_label
                    })
                except Exception as e:
                    # ... error handling ...
        # ... (return results)
        return results
    ```
*   **Pros:** Simple, fast to implement.
*   **Cons:** Ignores sentence boundaries, potentially poor semantic coherence within chunks, lacks broader document context.

#### 2. Late Chunking (Token Pooling)

*   **Concept:** Leverage models like Nomic that provide *token-level* embeddings. Embed a large portion (or ideally all) of the document at once to get embeddings for each token, capturing context. Then, define logical text chunks (e.g., every 512 chars) and *average* the embeddings of the tokens that fall within each chunk's character boundaries.
*   **Flowchart:**

    ```mermaid
    graph LR
        A[Start: Document Text, Chunk Size, Max Length] --> B{Tokenize & Embed Full Document (up to Max Length)};
        B --> C[Get Token Embeddings & Offset Mapping];
        C --> D{Define Text Chunk Boundaries (e.g., every N chars)};
        D --> E{For each Text Chunk};
        E --> F[Find Tokens within Chunk Boundaries (using Offsets)];
        F --> G{Average Token Embeddings for the Chunk};
        G --> H[Store (Chunk Text, Averaged Embedding)];
        E -- More Chunks --> F;
        E -- No More Chunks --> I[End: List of (Chunk, Pooled Embedding)];
        B -- Error/Fallback --> J[Embed Truncated Doc & Assign to Chunks];
        J --> I;

    ```
*   **Implementation:** (Simplified view of Cell 5)

    ```python
    # Cell 5: Define Late Chunking function
    def late_chunking(document, chunk_size=512, target_max_length=8192, source_label=""):
        """
        Implements Late Chunking by embedding the full document to get token embeddings,
        then pooling token embeddings corresponding to each chunk. Includes a source label.
        """
        try:
            # 1. Tokenize & Get Token Embeddings for the document (up to target_max_length)
            # Uses embedding_model.tokenizer and embedding_model.encode(..., output_value='token_embeddings', ...)
            # Gets token_embeddings and offset_mapping
            # ... (tokenization and full doc embedding logic) ...
            token_embeddings = ...
            offset_mapping = ...

            # 2. Define chunk boundaries based on character count
            chunks_text_boundaries = []
            for i in range(0, len(document), chunk_size):
                # ... define start_char, end_char, chunk_text ...
                chunks_text_boundaries.append({'text': chunk_text, 'start_char': start_char, 'end_char': end_char})

            # 3. Map boundaries to tokens and pool embeddings
            results = []
            token_embeddings_np = token_embeddings.cpu().numpy()
            for chunk_info in chunks_text_boundaries:
                start_char = chunk_info['start_char']
                end_char = chunk_info['end_char']
                token_indices = []
                # Find tokens whose character spans overlap with the chunk's span
                for idx, (token_start, token_end) in enumerate(offset_mapping):
                     # ... (logic to find relevant token indices) ...
                     if token_start < end_char and token_end > start_char:
                         token_indices.append(idx)

                if token_indices:
                    # Average the embeddings of the identified tokens
                    chunk_token_embeddings = token_embeddings_np[token_indices]
                    chunk_embedding = np.mean(chunk_token_embeddings, axis=0)
                    results.append({
                        'text': chunk_info['text'],
                        'embedding': chunk_embedding,
                        'strategy': 'Late',
                        'source': source_label
                    })
            # ... (fallback logic if token pooling fails) ...
        except Exception as e:
            # Fallback: Embed a truncated part and assign same embedding to all chunks
            # ... (fallback logic) ...
        return results

    ```
*   **Pros:** Embeddings benefit from wider context during initial calculation, potentially better semantic representation.
*   **Cons:** More complex, requires models providing token embeddings, computationally more expensive upfront, still uses fixed character boundaries for final chunks (though boundaries could be smarter, e.g., sentence-based). The fallback mechanism reduces robustness if the primary method fails often.

#### 3. Contextual Chunking

*   **Concept:** Generate a summary of the entire document. Then, for each naive chunk, *append* the summary to it before generating the embedding. The idea is that the summary provides global context to the embedding model for each specific chunk.
*   **Flowchart:**

    ```mermaid
    graph LR
        A[Start: Document Text, Chunk Size] --> B{Generate Summary of Document};
        B --> C{Split Document into Naive Chunks};
        C --> D{For each Chunk_i};
        D --> E[Create Enriched Text = Chunk_i + Summary];
        E --> F{Embed Enriched Text};
        F --> G[Store (Chunk_i Text, Embedding)];
        D -- More Chunks --> E;
        D -- No More Chunks --> H[End: List of (Chunk, Contextual Embedding)];
    ```
*   **Implementation:**

    ```python
    # Cell 6: Define Contextual Chunking function
    def contextual_chunking(document, chunk_size=512, source_label=""):
        """
        Generate context-aware chunks by summarizing the document and appending
        the summary to each chunk *before* embedding. Includes a source label.
        """
        # 1. Generate Summary (using Hugging Face summarizer pipeline)
        summary = summarizer(document[:max_summary_input_length], ...)[0]['summary_text']
        # ... (handle potential summary errors) ...

        # 2. Split into naive chunks
        chunks = [document[i:i + chunk_size] for i in range(0, len(document), chunk_size)]
        results = []
        # 3. Enrich and Embed
        for chunk in chunks:
           if chunk.strip():
                # Append summary to chunk text before embedding
                enriched_chunk_for_embedding = chunk + f" [CONTEXT_SUMMARY_{source_label}] " + summary
                embedding = embedding_model.encode(enriched_chunk_for_embedding, convert_to_numpy=True)
                results.append({
                    'text': chunk, # Store the ORIGINAL chunk text
                    'embedding': embedding,
                    'strategy': 'Contextual',
                    'source': source_label
                })
        # ... (return results)
        return results
    ```
*   **Pros:** Infuses global context into each chunk's embedding, potentially improving relevance for queries requiring broader understanding. Relatively simple modification of Naive Chunking.
*   **Cons:** Requires an additional summarization step (latency, cost), summary quality impacts effectiveness, increases the text length being embedded per chunk.

### The Experiment Setup

*   **Libraries:** `sentence-transformers`, `transformers`, `torch`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `pymupdf4llm`, `kaleido`, `markdown-pdf`.
*   **Hardware:** GPU (CUDA) is recommended and utilized (`DEVICE = "cuda"`). `CUDA_LAUNCH_BLOCKING=1` is set for easier debugging of GPU errors.
*   **Embedding Model:** `nomic-ai/nomic-embed-text-v1.5` (via `sentence-transformers`). Chosen for its 8192 token context window and support for token embeddings (`output_value='token_embeddings'`). `trust_remote_code=True` is required.
*   **Summarizer:** `facebook/bart-large-cnn` (via `transformers` pipeline) for Contextual Chunking.
*   **Data:** Two PDF user manuals (Samsung S24, S25) downloaded and text extracted using `pymupdf4llm.to_markdown`. Basic text cleaning is applied.
*   **Queries:** A diverse list (`query_list` in Cell 10B) covering troubleshooting, features, and setup questions.
*   **Chunk Size:** `chunk_size = 512` characters used for all strategies.

### Code Execution Flow

1.  **Install & Import (Cells 1, 2):** Standard library setup. `CUDA_LAUNCH_BLOCKING` requires a kernel restart.
2.  **Initialize Models (Cell 3):** Load the Nomic embedding model and the BART summarizer onto the appropriate device. Cell 3.1 performs a basic encoding test to ensure the model loads and runs.
3.  **Define Chunking Functions (Cells 4, 5, 6):** Implement the logic described above for each strategy.
4.  **Load Data (Cell 7):** Download PDFs, extract text using `pymupdf4llm`, and perform basic cleaning.
5.  **Run Chunking (Cell 9):** Iterate through the loaded documents (S24, S25). For each document, apply Naive, Late, and Contextual chunking. Store all results (chunk text, embedding, strategy, source document label) in a single Pandas DataFrame `df_results`.
6.  **Define Analysis Helpers (Cell 10A):** Functions are created to:
    *   `calculate_top_results`: Compute cosine similarity between a query embedding and chunk embeddings, returning the top N results per source/strategy group.
    *   `prepare_visualization_data`: Perform dimensionality reduction (PCA, UMAP, or t-SNE) on the embeddings of top results plus reference points (query, full doc embeddings).
    *   `create_spatial_plot`: Generate the 3-subplot Plotly scatter plot.
    *   `create_similarity_violin_plot`: Generate the Seaborn violin plot using Matplotlib.
7.  **Process Queries (Cells 10B, 10C):**
    *   Define the `query_list` and embed the full S24 and S25 documents once as reference points (`sX_doc_ref_embedding`).
    *   Loop through each query in `query_list`:
        *   Embed the query.
        *   Call `calculate_top_results` using the query embedding and the main `df_results` DataFrame.
        *   Generate and *save* a violin plot (`violin_qX_...png`).
        *   Prepare data and generate/save a spatial plot (`pca_qX_...png`, or umap/tsne).
        *   Append results (top chunks DataFrame, plots) to a Markdown file (`query_analysis_report.md`).
8.  **Aggregate & Save (Cells 10C cont., 10D, 10E):** The loop saves results per query to the Markdown file. Optionally, convert the Markdown report to PDF and aggregate all top results across queries into a final DataFrame. A helper cell zips and downloads the generated images.
9.  **(Legacy Visualization Cells 11-20):** The original notebook included cells (11-20) to perform dimensionality reduction (PCA, UMAP, t-SNE) on *all* chunks together and plot them, along with cells specifically visualizing the top N results for a *single* `sample_query`. The loop in Cell 10C replaces/enhances this by performing analysis *per query* and saving results systematically. The violin plot (Cell 20) is functionally similar to `create_similarity_violin_plot`. The subplot approach in the later cells (e.g., modified Cell 12/13) is integrated into the `create_spatial_plot` function.

### Visualizing the Results

The code generates two key plot types per query, saved to the `query_results/images` folder and embedded in the Markdown report:

1.  **Similarity Violin Plot:** (`violin_qX_...png`)
    *   **What it shows:** The distribution of cosine similarity scores between the query and the *top N* retrieved chunks for each strategy/source combination.
    *   **How to read it:**
        *   **Y-axis:** Chunking strategy.
        *   **X-axis:** Cosine similarity (higher is better).
        *   **Violin shape:** Wider sections indicate more chunks with that similarity score.
        *   **Internal lines/sticks:** Often show median, quartiles, or individual points.
        *   **Hue/Split:** Colors differentiate source documents (S24 vs. S25). Split violins allow direct comparison within a strategy.
    *   **Interpretation:** Look for strategies that consistently yield higher similarity scores (violins shifted right). Compare distributions between S24 and S25 for the same strategy. Strategies with tight, high-similarity distributions are generally preferred.

    `[Image: Example Violin Plot showing similarity scores distribution for Naive, Late, Contextual strategies, split by S24/S25 source documents for a specific query.]`

2.  **Spatial Visualization Plot (PCA/UMAP/t-SNE):** (`pca_qX_...png`, etc.)
    *   **What it shows:** A 2D representation of the embedding space containing the *top N* retrieved chunks for the query, plus the query itself and reference embeddings for the full S24/S25 documents. It uses dimensionality reduction to make high-dimensional embeddings plottable. The code uses subplots to show each strategy side-by-side.
    *   **How to read it:**
        *   **Subplots:** Faceted by chunking strategy (Naive, Late, Contextual).
        *   **Points:** Each point is a chunk embedding (or query/doc ref).
        *   **Color:** Blue = S24 chunk/ref, Red = S25 chunk/ref, Black = Query.
        *   **Shape:** Circle = Naive, Square = Late, Triangle = Contextual, Star = Query, Diamond = Doc Ref.
        *   **Proximity:** Points closer together are more semantically similar in the high-dimensional space (as interpreted by the reduction technique).
    *   **Interpretation:**
        *   **Query Proximity:** Which points (strategy/source) cluster closest to the black star (Query)? This indicates relevance.
        *   **Strategy Clusters:** Do Late/Contextual chunks form tighter, more distinct clusters than Naive chunks within each subplot?
        *   **Source Separation:** How well are blue (S24) and red (S25) points separated? Does this vary by strategy?
        *   **Reference Points:** Where do the strategy clusters lie relative to their corresponding document reference diamond (e.g., blue squares near blue diamond)?

    `[Image: Example 3-Subplot Spatial Visualization (e.g., UMAP). Each subplot shows top chunks for Naive, Late, or Contextual strategy. Points colored by source (Blue=S24, Red=S25), shaped by strategy. Black star is Query, Blue/Red diamonds are S24/S25 doc references.]`

### Conclusion & Key Takeaways

This code provides a robust framework for comparing chunking strategies empirically. By analyzing real documents and multiple queries, we can observe:

*   **No Silver Bullet:** The best strategy often depends on the specific query, the document content, and the embedding model used.
*   **Context Matters:** Late Chunking and Contextual Chunking aim to incorporate broader document context, which *can* lead to better semantic representations and retrieval, especially for queries requiring more than just keyword matching. This should be visible in the spatial plots (tighter clusters closer to relevant references) and potentially higher similarity scores in violin plots.
*   **Trade-offs:** Naive is simple but often crude. Late Chunking is powerful but complex and relies on specific model capabilities. Contextual Chunking adds overhead (summarization) and its effectiveness depends on summary quality.
*   **Visualization is Key:** Plots provide intuitive insights into how different strategies represent information in the embedding space and how relevant they are to specific queries, going beyond simple similarity scores.

This experiment demonstrates the importance of thoughtfully choosing a chunking strategy for RAG systems. By adapting this code, you can test different models, chunk sizes, boundary methods (e.g., sentence splitting), and datasets to find the optimal approach for your specific needs.
