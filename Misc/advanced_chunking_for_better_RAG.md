## What is Chunking
**Chunking** refers to breaking text into smaller units ("chunks"), which are indexed for retrieval and generation tasks in LLM-based RAG systems. Proper chunking impacts both efficiency and the accuracy of responses.[1]

## Impact of Chunking
Chunking affects:
- **Retrieval Quality**: Well-chosen chunks improve retrieval relevance.
- **Database Cost**: More, smaller chunks increase storage cost; larger chunks may reduce granularity but save resources.
- **Query Latency**: Fewer chunks speed up retrieval.
- **LLM Latency/Cost**: Larger chunks may mean more context and slower, pricier LLM calls.
- **Hallucinations**: Overly large chunks can confuse the LLM, risking inaccurate outputs.[1]

## Factors Influencing Chunking
Considerations include:
- **Text Structure**: Different content types (sentences, tables, code, transcripts) need distinct chunking approaches.
- **Embedding Model**: Its context window and embedding length constraints determine practical chunk sizes.
- **LLM Context**: LLMs only process limited context; chunk size sets how much fits.
- **Question Type**: Simple fact retrieval versus complex queries may require different chunking strategies.[1]

## Types of Chunking

### Text Splitter Methods
- **Character Splitter**: Splits by characters or regex separators; merge logic helps fit chunk size and overlap.
- **Recursive Character Splitter**: Recursively splits by increasingly general separators (e.g., paragraph, sentence, word, character) for flexible chunking.
- **Sentence Splitter (e.g., Spacy)**: Splits into sentence groups, with a stride and overlap to preserve context.
- **Semantic Splitter**: Clusters related sentences using sentence similarity models, keeping coherent context.[1]

### LLM-Based Chunking
- **Propositions**: Splits text into atomic, self-contained factoids ("propositions") using LLMs, ensuring minimal and context-rich units for retrieval.[1]

### Multi-Vector Indexing
- Embeds smaller chunks, document summaries, or hypothetical answer questions to enrich retrieval and recall.

## Document-Specific Splitting
Tools like **Unstructured** can split PDFs and complex formats, selecting adaptive or specialized parsing strategies for mixed media documents.[1]

## Evaluating Chunking
Metrics:
- **Chunk Attribution**: Measures if a chunk actually influenced the LLM’s answer.
- **Chunk Utilization**: Measures how much of a retrieved chunk was relevant to the answer; low scores point to overly large or imprecise chunks.

## Conclusion
Effective chunking is key to high-performance RAG systems, balancing retrieval quality, cost, and latency, measured via chunk attribution/utilization metrics.[1]

[1](https://galileo.ai/blog/mastering-rag-advanced-chunking-techniques-for-llm-applications)
