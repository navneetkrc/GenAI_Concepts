## Summary of Query Understanding and Query Rewriting

**Query Understanding**

- Query understanding refers to the process of interpreting and extracting the user's intent from their search input, especially in contexts like e-commerce where queries are often short, ambiguous, or contain misspellings and abbreviations.
- Effective query understanding involves:
  - Normalizing text (e.g., lowercasing, removing punctuation and emojis)
  - Tokenizing queries into meaningful units
  - Identifying and correcting misspellings, including those involving brand names or phonetic errors
  - Expanding abbreviations and acronyms (e.g., "lv" to "louis vuitton")
  - Leveraging user behavior and session data to infer intent and context[2][5][6].
- Advanced systems use large language models (LLMs) and retrieval-augmented frameworks to dynamically adapt to new vocabulary, brand names, and evolving user language, improving both accuracy and robustness in interpreting queries[5][6].

**Query Rewriting**

- Query rewriting (QR) is the process of transforming a user's original query into one or more alternative queries that are semantically similar but phrased differently, to improve search relevance and coverage[1][3].
- The goal is to bridge the lexical gap between user queries and the way products or content are described in the catalog, thereby increasing the likelihood of retrieving relevant results[1][3].
- **Methods:**
  - *Discriminative approaches*: Use predefined mappings, synonym extraction, or search logs to suggest alternative queries. These are efficient but struggle with long-tail queries and lack flexibility for complex or ambiguous user intents[1].
  - *Generative approaches*: Employ LLMs or neural models to generate contextually rich and diverse rewrites. These can dynamically produce novel and relevant reformulations but may incur higher computational costs and latency[1][3].
  - *Hybrid pipelines*: Combine offline knowledge distillation (to create efficient, lightweight models) with online reinforcement learning (to adapt and refine rewrites in real time using user feedback or simulated interactions)[1][3].
- **Evaluation Metrics:**
  - *Relevance*: Alignment between the original intent and the rewritten query's results
  - *Diversity*: The breadth of distinct products or results retrieved
  - *User engagement*: Simulated or real feedback such as clicks, add-to-cart, and purchases[1].
- **Recent Advances:**
  - Reinforcement learning-based query rewriting (RLQR) and hybrid models have shown significant improvements in product coverage, relevance, and adaptability by leveraging both offline training and real-time user feedback[1][3].
  - Retrieval-augmented generation (RAG) and contextual fine-tuning further enhance spelling correction and query rewriting, especially for dynamic vocabularies and new brand names[5].

---

## Key Points Table

| Aspect               | Query Understanding                                  | Query Rewriting                                   |
|----------------------|-----------------------------------------------------|---------------------------------------------------|
| **Goal**             | Interpret user intent and context                    | Generate alternative, semantically similar queries |
| **Techniques**       | Normalization, tokenization, spelling correction, LLMs, retrieval augmentation | Discriminative models, generative LLMs, hybrid RL pipelines |
| **Challenges**       | Misspellings, abbreviations, evolving vocabulary, ambiguous intent | Long-tail queries, computational cost, semantic drift |
| **Recent Advances**  | Retrieval-augmented LLMs, context-aware correction  | RL-based rewriting, knowledge distillation, RAG    |
| **Evaluation**       | Precision, recall, F1 (for correction), user engagement | Relevance, diversity, user engagement, product coverage |

---

**In summary:**  
Query understanding is about accurately interpreting user intent, handling linguistic variations, and preparing the query for effective retrieval. Query rewriting transforms this understanding into optimized queries that maximize the chances of retrieving relevant results, using a blend of rule-based, generative, and reinforcement learning techniques-often evaluated by their impact on relevance, diversity, and user engagement[1][2][3][5][6].

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_e05483f9-371b-4e79-b8b2-ff30e92b3c87/9a496957-60f8-41e1-91d0-650bdc1f99ff/RL-Based-Query-Rewriting.pdf
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_e05483f9-371b-4e79-b8b2-ff30e92b3c87/55992e2e-42dc-465e-8fec-4d2a9fde6d45/Enhancing-Search-Using-Large-Language-Models-_-by-Whatnot-Engineering-_-Whatnot-Engineering-_-Medium.pdf
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_e05483f9-371b-4e79-b8b2-ff30e92b3c87/7ebdce90-2993-4e03-ba46-709c6a76425c/Enhancing-E-commerce-Search_-Reinforcement-Learnin.pdf
[4] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_e05483f9-371b-4e79-b8b2-ff30e92b3c87/33e5cb5d-f01c-45dd-96ee-924d0509a51c/How-Instacart-Uses-Machine-Learning-Driven-Autocomplete-to-Help-People-Fill-Their-Carts-_-by-Esther-Vasiete-_-tech-at-instacart.pdf
[5] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_e05483f9-371b-4e79-b8b2-ff30e92b3c87/66531d33-682e-4247-854f-c9629b9c2140/Retrieval-Augmented-Spelling-Correction-for-E-Commerce-Applications-Amazon-2024.pdf
[6] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_e05483f9-371b-4e79-b8b2-ff30e92b3c87/7484db3d-a2d0-44c3-b395-8fa62fc204fa/spelling-correction-using-phonetics-in-ecommerce-search-Amazon-2022.pdf
[7] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_e05483f9-371b-4e79-b8b2-ff30e92b3c87/ac9371da-0871-4362-a7af-12e0e9bb2f69/Target-AutoComplete_-Real-Time-Item-Recommendations-at-Target.pdf
[8] https://en.wikipedia.org/wiki/Query_understanding
[9] https://aclanthology.org/2023.acl-industry.59.pdf
[10] https://arxiv.org/html/2306.16004v1
[11] https://www.algolia.com/blog/product/query-understanding-101
[12] https://queryunderstanding.com/spelling-correction-471f71b19880
[13] https://github.com/SeekStorm/SeekStorm/issues/4
[14] https://spotintelligence.com/2024/04/03/query-understanding/
[15] https://queryunderstanding.com/query-rewriting-an-overview-d7916eb94b83
[16] https://www.amazon.science/publications/advancing-query-rewriting-in-e-commerce-via-shopping-intent-learning
[17] https://hackmd.io/@deep2233/H186a_qLC
[18] https://opensourceconnections.com/blog/2024/02/20/what-do-you-actually-mean-query-understanding-with-llms/
[19] https://arxiv.org/html/2403.09060v1
[20] https://sdm-dsre.github.io/pdf/query_rewrite.pdf
[21] https://www.sciencedirect.com/topics/computer-science/query-rewriting
[22] https://docs.vespa.ai/en/query-rewriting.html
[23] https://arxiv.org/html/2501.18056v1
[24] https://queryunderstanding.com/query-understanding-a-manifesto-367dc0be6745
[25] https://blog.langchain.dev/query-transformations/
[26] https://queryunderstanding.com
[27] https://www.linkedin.com/pulse/ai-query-understanding-daniel-tunkelang
[28] https://queryunderstanding.com/introduction-c98740502103
[29] https://arxiv.org/pdf/2403.09060.pdf

---
Answer from Perplexity: pplx.ai/share
