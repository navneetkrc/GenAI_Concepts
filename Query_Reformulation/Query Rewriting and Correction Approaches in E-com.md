<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Query Rewriting and Correction Approaches in E-commerce Search Systems

E-commerce search engines face unique challenges in understanding user intent, especially when queries are ambiguous, contain typos, or use non-standard terminology. This comprehensive analysis examines the most effective query rewriting and correction approaches that improve search relevance and user experience in e-commerce platforms. The latest research reveals significant advancements in neural methods, contextual understanding, and hybrid approaches that balance efficiency with improved search accuracy.

## Introduction to Query Rewriting in E-commerce

Query understanding plays a foundational role in the search process, with accurate interpretation of search queries being the critical first step toward delivering high-quality results on e-commerce platforms. Query rewriting, a technique that transforms problematic or ambiguous queries into well-formed ones that better match user intent, has become essential in modern e-commerce search systems. This technique addresses the fundamental challenge that occurs when users submit queries that don't exactly match the vocabulary used in product descriptions, creating a lexical gap that can lead to poor search results[^1_4].

The importance of query rewriting is underscored by consumer behavior data showing that 76% of online shoppers abandon retail websites after an unsuccessful search experience[^1_2]. This abandonment rate highlights how critical effective query understanding is to e-commerce success. Query rewriting serves multiple purposes: it corrects spelling errors, expands abbreviated terms, resolves ambiguities, and bridges the vocabulary mismatch between user terminology and product catalog descriptions. While head queries (frequently searched terms) typically have abundant historical data to aid interpretation, tail queries (less common searches) pose significant challenges to accurate understanding and often benefit most from rewriting approaches[^1_1].

E-commerce query rewriting differs from general web search rewriting due to the highly specialized nature of retail terminology, constantly changing product catalogs, and the direct commercial impact of search quality. Modern approaches to query rewriting have evolved from simple rule-based systems to sophisticated machine learning models that can capture nuanced relationships between queries and products while adapting to changing user behaviors and emerging trends in the retail landscape.

## Neural and Deep Learning Approaches

Advanced neural approaches have revolutionized query rewriting in e-commerce by significantly improving the ability to understand user intent and generate more relevant query alternatives. A prominent example is the Query Understanding EnhancEd mechaNism (QUEEN), which leverages deep learning to address large-scale query rewriting challenges in e-commerce search engines. QUEEN notably incorporates query annotations, which are byproducts of query processing pipelines in most e-commerce search engines, to model ambiguous product search queries. In empirical studies based on 38.5 million anonymous product search queries, QUEEN demonstrated a remarkable 6% relative improvement in sentence-level recall compared to other state-of-the-art baselines[^1_7].

Another significant advancement employs Seq2Seq (sequence-to-sequence) models extended with multiple auxiliary tasks specifically designed for e-commerce applications. These models transform tail queries (less common searches) into queries with similar linguistic characteristics as head queries (frequent searches) while preserving the original shopping intent. What makes this approach particularly valuable is its innovative training data construction process that relies solely on widely available search logs to generate query pairs and additional shopping intent information. This information enables the incorporation of auxiliary prediction tasks focused on product name and category prediction, helping the model better capture shopping intent. Additionally, these models introduce a query matching loss based on a novel co-attention scheme that improves source query representations, allowing the entire model to be built and trained end-to-end with standard components and training protocols[^1_1].

The effectiveness of neural approaches extends beyond simple rewriting. For instance, some implementations target specific challenges like spelling correction for new brand names, which often use unconventional spelling to create memorable impacts while ensuring brand uniqueness. Traditional spelling correction systems might wrongly "correct" intentionally unique spellings like "playgro" to "playground" or "biotanicals" to "botanicals." Addressing this challenge requires specialized neural approaches that can distinguish between genuine misspellings and novel brand names with unconventional spelling patterns[^1_3].

These neural approaches share a common advantage: they can learn complex patterns from large datasets of user queries without requiring explicit programming of linguistic rules. This ability to generalize from examples rather than following predefined rules makes neural methods particularly suited to the dynamic nature of e-commerce search, where new products, brands, and search patterns continuously emerge.

## Context and Intent-Based Query Rewriting

Context-aware query rewriting represents a significant advancement in e-commerce search technology by incorporating users' search history to better understand their shopping intent. Traditional query rewriting models often consider only the instant search query, which is frequently a short string offering limited information about true shopping intent. However, innovative context-aware approaches recognize that users typically enter multiple searches before making a purchase, creating a contextual history that provides valuable insights into their actual intent[^1_6].

One notable implementation builds an end-to-end context-aware query rewriting model that constructs a session graph using historical search queries and their contained words. This model employs a sophisticated graph attention mechanism to model cross-query relationships and compute contextual information for the entire session. By combining this contextual information with the instant search query through an aggregation network, the system generates more accurate session representations that are then decoded to produce improved rewritten queries. Empirical testing demonstrates the superiority of this contextual approach compared to state-of-the-art methods that lack historical context awareness[^1_6].

Shopping intent learning represents another powerful dimension of context-aware query rewriting. By incorporating auxiliary prediction tasks focused on product names and categories into the rewriting model, systems can capture nuanced shopping intent information that goes beyond the literal meaning of query terms. This approach has shown significant advantages over vanilla Seq2Seq models in experiments measuring rewriting quality, with practical applications demonstrated in sponsored search scenarios[^1_1].

The contextual approach addresses a common e-commerce search pattern where users repeatedly refine their queries to find exactly what they want. For example, a user might start with a general search like "dodge banners," move to "mopar poster," then type "dodger posters" (containing a typo), before finally purchasing through a search for "dodge posters." Context-aware systems can recognize this pattern and directly rewrite "dodger posters" to "dodge posters" based on the session history, saving the user time and potential frustration[^1_6].

These context-based approaches represent a significant improvement over traditional methods by acknowledging that queries don't exist in isolation but are part of a user's ongoing search journey, with each query providing additional context that helps clarify true shopping intent.

## Synonym and Dictionary-Based Approaches

Synonym and dictionary-based query rewriting approaches remain fundamental in e-commerce search systems due to their efficiency and interpretability. These methods typically follow a two-phase process: an offline phase for generating and filtering token-level synonyms, and an online phase where user queries are rewritten using these synonym dictionaries. The approach is particularly valuable for addressing vocabulary mismatches between user queries and product descriptions in e-commerce contexts[^1_5].

In the offline synonym generation phase, systems employ multiple techniques to create candidate synonyms for query tokens. These include leveraging user behavioral data (such as queries that lead to the same product purchases), inventory and taxonomy information, and open-source knowledge bases. A critical innovation in this process is the use of machine learning classifiers trained on human-judged binary relevance labels to filter the candidate synonyms, ensuring they're truly useful for query expansions without compromising result precision. This classifier-based filtering creates a scalable method to evaluate the effectiveness of synonyms generated from diverse sources[^1_5].

The online phase involves a sophisticated query processing workflow. First, raw user queries undergo segmentation into non-overlapping n-gram segments to maximize coverage in the rewrite dictionary. A common segmentation heuristic involves taking the longest matching segment from left among the query tokens, then splitting the remaining tokens recursively in the same manner. Next, the system looks up synonyms for the n-gram segments in the synonym dictionary and combines them into boolean expressions. For example, the query "ps 4 games" might be segmented as "ps 4" and "games," then rewritten as "((ps 4 OR playstation 4) AND (games OR game))" based on the synonym dictionary[^1_5].

This approach offers distinct advantages for e-commerce platforms. It allows for transparent and controllable query expansions while being computationally efficient during runtime, making it suitable for high-volume e-commerce sites. Furthermore, the dictionary-based approach can be continuously improved through feedback loops that incorporate new user behavior data and relevance judgments to refine synonym quality[^1_5].

Modern implementations often enhance traditional dictionary-based approaches with neural network components or contextual elements, creating hybrid systems that leverage the strengths of multiple methodologies while mitigating their individual weaknesses.

## Advanced Techniques: RAG and RL-Based Approaches

The latest frontier in e-commerce query rewriting incorporates cutting-edge techniques such as Retrieval Augmented Generation (RAG) and Reinforcement Learning (RL), delivering substantial performance improvements that address unique industry challenges. These approaches represent significant innovations that push beyond traditional methods to handle increasingly complex search scenarios in dynamic retail environments.

Retrieval Augmented Generation (RAG) has emerged as a powerful approach for spelling correction in e-commerce applications, particularly when dealing with unconventional brand names. This method works by retrieving product names from a catalog and incorporating them into the context used by a large language model (LLM) that has been fine-tuned for contextual spelling correction. The RAG framework shows consistent performance improvements with only minor latency increases compared to stand-alone LLM approaches. Quantitative evaluations and qualitative error analyses demonstrate RAG's value in addressing the challenge of distinguishing between genuine misspellings and novel brand names with unconventional spelling, such as "hygeeni" versus "hygiene"[^1_3].

Another significant advancement combines offline knowledge distillation with online reinforcement learning to create a powerful hybrid pipeline for query rewriting. This approach addresses fundamental limitations in current methods – discriminative models often struggle with natural language understanding and offer limited rewriting flexibility, while generative LLMs face high inference latency and costs in online settings despite producing high-quality rewrites. The hybrid approach creates a lightweight yet efficient student model through offline knowledge distillation from larger LLMs, then refines query rewriting dynamically using real-time feedback through reinforcement learning. A key innovation in this method is using LLMs as simulated human feedback, enabling scalable reward signals and cost-effective evaluation without manual annotations[^1_4].

Experimental results on datasets like the Amazon ESCI demonstrate significant improvements in query relevance, diversity, and adaptability with the hybrid RL approach. This method contributes to advancing LLM capabilities for domain-specific applications while offering a robust solution for dynamic and complex e-commerce search environments. The ability to balance efficiency and effectiveness makes these advanced techniques particularly valuable for large-scale e-commerce platforms handling millions of diverse queries daily[^1_4].

These cutting-edge approaches represent the direction in which e-commerce query rewriting is evolving, leveraging the most advanced AI techniques while adapting them to the specific challenges of online retail environments.

## Implementation Considerations for E-commerce Search Systems

Implementing effective query rewriting systems in e-commerce environments requires careful consideration of several practical factors that balance search quality with performance constraints. One critical consideration is latency, as search response times directly impact user experience and conversion rates. While sophisticated neural models and LLMs may generate high-quality rewrites, they often introduce unacceptable latency in production environments, which explains why many implementations use hybrid approaches or knowledge distillation to create more efficient models[^1_4].

Scoped suggestions represent an important implementation technique for improving search experiences. This approach helps users narrow down their searches on topics they're most interested in by suggesting queries within specific categories or brands. For example, if a user searches for "flat screen" on an electronics retail store, scoped suggestions might offer category-specific queries like "flat screen TVs" or "flat screen monitors," helping users quickly refine their searches to the most relevant category. Implementing scoped search suggestions using modern search UI components and backend systems is relatively straightforward and delivers immediate value in helping users see refined result sets quickly[^1_2].

Another important implementation consideration is the handling of new product introductions and emerging terminology. E-commerce catalogs continuously evolve with new products, brands, and features being introduced regularly. Query rewriting systems must adapt to these changes without requiring constant manual intervention. Some implementations address this by implementing feedback loops that incorporate search performance metrics and user engagement signals to continuously refine and update rewriting rules and models[^1_3][^1_5].

The integration of query rewriting within the broader search architecture also requires careful planning. Query rewriting typically sits early in the search pipeline, influencing downstream components like ranking and filtering. Some implementations use a multi-stage approach where different rewriting techniques are applied sequentially or in parallel, with their results combined through fusion techniques or A/B testing frameworks that can dynamically select the most effective rewrite for a given context[^1_4][^1_5].

Performance monitoring and evaluation frameworks are essential for maintaining and improving query rewriting systems over time. Metrics like search success rate, click-through rate on search results, conversion rates, and before-after comparisons of search quality for rewritten queries provide valuable feedback for continuous improvement. Some implementations include automatic A/B testing infrastructures that can evaluate new rewriting models or rules against current production systems using a small percentage of live traffic[^1_1][^1_7].

## Conclusion

Query rewriting and correction technologies have become indispensable components of modern e-commerce search systems, addressing the critical challenge of bridging the gap between user queries and relevant product listings. The approaches examined in this report demonstrate the remarkable evolution of these technologies, from basic dictionary-based methods to sophisticated neural networks that incorporate contextual understanding and shopping intent prediction.

The most effective implementations in contemporary e-commerce platforms typically employ hybrid approaches that combine multiple techniques to balance performance constraints with search quality. Neural methods provide powerful semantic understanding capabilities but may require complementary techniques like knowledge distillation or simplified runtime models to meet latency requirements. Context-aware approaches significantly enhance accuracy by considering users' search history but require thoughtful integration with existing search infrastructure. Traditional dictionary-based methods continue to offer value through their transparency and efficiency, particularly when enhanced with machine learning for synonym quality assessment[^1_1][^1_4][^1_5][^1_6].

Emerging techniques like Retrieval Augmented Generation (RAG) and reinforcement learning show tremendous promise for further advancing query rewriting capabilities, particularly for handling specialized challenges like unconventional brand names and adapting to rapidly changing product catalogs[^1_3][^1_4]. The consistent theme across successful implementations is their ability to understand true shopping intent beyond the literal meaning of query terms.

For e-commerce platforms looking to implement or improve query rewriting systems, the research suggests prioritizing approaches that balance semantic understanding with computational efficiency, incorporate user context where possible, and establish robust feedback loops for continuous improvement. As e-commerce search continues to evolve, query rewriting will remain a critical technology for connecting shoppers with the products they seek, directly impacting business outcomes through improved search success rates and enhanced user experience[^1_2][^1_7].

<div style="text-align: center">⁂</div>

[^1_1]: https://www.amazon.science/publications/advancing-query-rewriting-in-e-commerce-via-shopping-intent-learning

[^1_2]: https://www.elastic.co/blog/how-to-build-scoped-search-suggestions-and-search-query-corrections

[^1_3]: https://arxiv.org/html/2410.11655v1

[^1_4]: https://arxiv.org/abs/2501.18056

[^1_5]: https://ceur-ws.org/Vol-2410/paper20.pdf

[^1_6]: https://aclanthology.org/2023.acl-industry.59.pdf

[^1_7]: https://www.amazon.science/publications/queen-neural-query-rewriting-in-e-commerce


---

# Datasets for Query Rewriting in Search Systems

Query rewriting represents a crucial component of modern search systems, enabling more accurate interpretation of user queries and ultimately delivering improved search results. Effective query rewriting depends heavily on high-quality datasets for training and evaluation. This analysis examines several significant datasets specifically designed for query rewriting across different domains, highlighting their characteristics, construction methodologies, and applications for researchers and practitioners working on search optimization.

## E-commerce Query Rewriting Datasets

E-commerce platforms present unique challenges for query understanding, requiring specialized datasets that capture the distinctive patterns and intentions behind shopping-related searches. Several notable datasets have emerged to address these specific needs in the retail search domain.

### Taobao Query Rewriting Dataset

One significant dataset in the e-commerce domain comes from the Taobao search system. As outlined in recent research, this dataset focuses specifically on long-tail query rewriting challenges in e-commerce search[^2_2]. The dataset construction process sources rewrites from Taobao's previous-generation rewriting policy, creating training pairs where each user query is matched with a corresponding rewrite. This approach yields a substantial dataset where search log data forms the foundation for training query rewriting models[^2_2].

What makes this dataset particularly valuable is its focus on aligning with actual e-commerce search objectives. The dataset construction methodology follows a pattern where they select the top-ranked rewrite from the list generated by their previous policy as the gold standard candidate. This ensures that the dataset reflects realistic user queries and practical rewriting patterns. The formal construction can be represented as D = {(xi, yi)}Ni=1 where xi represents original queries sampled from the Taobao search distribution, and yi represents rewrites generated by their previous policy[^2_2].

This dataset also incorporates additional dimensions beyond simple query pairs, incorporating quality classification, query correction, and chain-of-thought components that make it particularly suited for fine-tuning large language models for query rewriting tasks[^2_2]. The multi-faceted nature of this dataset allows it to address the semantic gap problem often associated with long-tail queries in e-commerce settings.

### Shopping Intent Learning Dataset

Another approach to dataset construction for e-commerce query rewriting focuses on capturing shopping intent. As described in Amazon's research, this methodology relies on widely available search logs to generate source-target query pairs augmented with additional shopping intent information[^2_1]. This dataset is specifically designed to transform tail queries (less common searches) into queries with characteristics similar to head queries (frequent searches) while preserving the original shopping intent.

The dataset incorporates additional information beyond simple query pairs, providing auxiliary data on product names and categories that helps capture the full shopping intent. This supplementary information enables the training of models that can better understand the underlying purpose of e-commerce queries, making it particularly valuable for applications in sponsored search and product recommendations[^2_1].

## Conversational Query Rewriting Datasets

Conversational systems require specialized datasets that capture the contextual nature of multi-turn dialogues, where query understanding must account for previous interactions.

### Contextual Query Rewrite (CQR) Dataset

The Contextual Query Rewrite (CQR) Dataset, developed for spoken dialogue systems, represents a significant resource for conversational query rewriting research. Created by Alexa and available on GitHub since March 2019, this dataset contains crowd-sourced rewrites of dialogues from the Stanford Dialogue Corpus[^2_4]. The primary motivation behind this dataset is to facilitate research in dialogue state tracking using natural language as an interface.

The CQR dataset addresses a fundamental challenge in dialogue assistant systems: understanding context across multiple turns and communicating that context across different domain-specific agents. By providing natural language rewrites, this dataset explores using language itself as an API for cross-agent communication, eliminating the need to learn diverse schema mappings[^2_4]. This approach leverages the inherent syntactic and semantic regularities of language as a mechanism for tracking dialogue state.

The dataset is presented in JSON format, with each record representing a dialogue interaction. Its public availability makes it a valuable resource for researchers working on contextual understanding in conversational systems, particularly for applications involving multiple specialized dialogue agents that need to share context[^2_4].

### QueryR and ContextCarryoverR Datasets

Another significant pair of datasets for conversational query rewriting are the QueryR and ContextCarryoverR datasets. These datasets address two distinct but related challenges in conversational systems: friction reduction (recovering from speech recognition or system errors) and contextual carryover (maintaining context across conversation turns)[^2_3].

The QueryR dataset is weakly annotated by a defect detection model to identify patterns where a user's first utterance was problematic but the second was successful. This dataset includes approximately one million non-defective queries that don't require rewriting, providing necessary negative examples for model training[^2_3].

Complementing this, the ContextCarryoverR dataset contains human annotations specifically for contextual carryover queries. It includes one million queries that require context carryover and one million that don't, creating a balanced dataset for training models to determine when context should be maintained across conversation turns[^2_3].

These datasets are constructed from real-world conversational interactions, with training and validation data collected over a one-month period (split 9:1) and testing data from a subsequent one-week period. Importantly, there is no overlap between QueryR and CarryoverR, ensuring that models trained on both datasets learn to distinguish between different types of query rewriting needs[^2_3].

## General Search Query Datasets

Beyond specialized domains like e-commerce and conversational systems, some datasets focus on general search query rewriting applications.

### AOL4PS Dataset

The AOL4PS dataset represents a large-scale resource constructed specifically for personalized search evaluation[^2_5]. Collected and processed from AOL query logs, this dataset addresses a significant gap in publicly available personalized search datasets. While not exclusively focused on query rewriting, the dataset provides valuable information for query suggestion and document ranking tasks that often interface with query rewriting systems.

A key challenge addressed by this dataset is the lack of candidate documents in the original AOL query logs, which only contain records of user-clicked documents. The researchers behind AOL4PS generated candidate document lists for each query and annotated documents that users found satisfactory[^2_5]. To overcome computational bottlenecks in processing the large-scale logs, they implemented an improved BM25 algorithm that made the construction of this comprehensive dataset feasible.

The dataset underwent careful preprocessing to remove incorrect and redundant data, addressing challenges such as incomplete log records and data duplication caused by server merging[^2_5]. This cleaning process ensures that the resulting dataset provides a reliable foundation for search personalization research, including query suggestion components that often involve query rewriting techniques.

## Implementation Considerations for Dataset Selection

When selecting datasets for query rewriting research or application development, several factors should be considered to ensure alignment with specific use cases and requirements.

The domain specificity of the dataset is a critical consideration. E-commerce query rewriting presents different challenges than conversational or general web search rewriting, making datasets like the Taobao collection or shopping intent learning dataset more appropriate for retail applications. Conversely, the CQR dataset or QueryR/ContextCarryoverR combinations are better suited for developing conversational systems.

Dataset size and quality also vary significantly. The Taobao and AOL4PS datasets offer large-scale collections with millions of examples, while specialized conversational datasets may be smaller but offer higher-quality annotations. Researchers should consider whether their application benefits more from scale or annotation quality when selecting training data.

The nature of annotations differs across datasets as well. Some provide simple query pairs, while others include rich contextual information, intent classifications, or quality judgments. The training objectives should align with the annotation style to ensure the resulting models address the specific aspects of query rewriting most relevant to the target application.

## Conclusion

A variety of specialized datasets exist to support query rewriting research and development across different domains. E-commerce-focused collections like the Taobao dataset and shopping intent learning dataset capture the unique challenges of retail search. Conversational datasets including CQR, QueryR, and ContextCarryoverR address the complexities of maintaining context and recovering from errors in multi-turn dialogues. General search datasets like AOL4PS provide broader coverage for personalized search applications.

The availability of these diverse datasets enables researchers and practitioners to develop more effective query rewriting systems tailored to specific domains and use cases. By selecting appropriate training data and evaluation benchmarks, search system developers can significantly improve query understanding capabilities, ultimately delivering more relevant results and enhanced user experiences across e-commerce platforms, conversational assistants, and general search engines.

When implementing query rewriting systems, selecting the right combination of datasets for training and evaluation remains a critical decision that should be guided by the specific requirements and constraints of the target application domain. The continued development of specialized, high-quality datasets will further advance the state of query rewriting technology across all search domains.

<div style="text-align: center">⁂</div>

[^2_1]: https://www.amazon.science/publications/advancing-query-rewriting-in-e-commerce-via-shopping-intent-learning

[^2_2]: https://arxiv.org/pdf/2311.03758.pdf

[^2_3]: https://aclanthology.org/2023.acl-industry.58.pdf

[^2_4]: https://github.com/alexa/alexa-dataset-contextual-query-rewrite

[^2_5]: https://direct.mit.edu/dint/article-abstract/3/4/548/106757


---

## Main Differences Between Query Rewriting in E-commerce and Other Text Generation Tasks

Query rewriting in e-commerce is a highly specialized task that differs from general text generation in several important ways. These distinctions arise from the unique goals, constraints, and operational realities of e-commerce search systems.

### 1. **Purpose and Evaluation Criteria**

- **E-commerce Query Rewriting:**
The primary goal is to bridge the *lexical gap* between user queries and product catalog descriptions to maximize the relevance and diversity of retrieved products. Evaluation is not just about linguistic quality but about *search performance metrics* such as relevance, diversity, product coverage, click/add-to-cart/purchase rates, and alignment with user intent. The effectiveness of a rewrite is measured by how well it improves the retrieval of relevant products and boosts user engagement and conversions[^3_1][^3_3].
- **General Text Generation (e.g., summarization, translation, creative writing):**
The focus is on producing fluent, coherent, and contextually appropriate text. Evaluation relies on metrics like BLEU, ROUGE, or human judgments of readability and fidelity to source meaning, rather than downstream business or retrieval outcomes.


### 2. **Domain Constraints and Knowledge Requirements**

- **E-commerce:**
    - Must handle *rapidly evolving vocabularies* (e.g., new brands, product names, trends) and *ambiguous or misspelled queries* (including phonetic and unconventional spellings)[^3_5][^3_6].
    - Requires *domain-specific knowledge* of products, categories, and user shopping behaviors.
    - Needs to distinguish between genuine misspellings and valid, often novel, brand or product names-a challenge not common in general text tasks[^3_5][^3_6].
    - Often leverages *retrieval-augmented* or *catalog-aware* models to incorporate up-to-date product information directly into the rewriting process[^3_5].
- **Other Text Generation Tasks:**
    - Typically operate with more stable vocabularies and do not need to integrate real-time or evolving external knowledge bases as directly.
    - Domain adaptation is possible but not as tightly coupled to a dynamic inventory or catalog.


### 3. **Operational Constraints: Latency and Scalability**

- **E-commerce:**
    - Query rewriting must be *extremely fast* (low-latency) to avoid degrading the user experience during live search[^3_1][^3_2].
    - Solutions are often hybrid: heavy models (e.g., LLMs) are used offline for generating rewrites for popular queries, while lightweight models or caches serve rewrites in real time[^3_1][^3_2].
    - Scalability is critical due to the high volume and diversity of queries.
- **Other Text Generation Tasks:**
    - May allow for higher latency, especially in batch or non-interactive settings (e.g., document summarization, content creation).
    - Real-time constraints are less stringent unless deployed in interactive dialogue systems.


### 4. **Reward Signals and Learning Paradigms**

- **E-commerce:**
    - Increasingly uses *reinforcement learning* (RL) with *implicit user feedback* (e.g., clicks, purchases) or *simulated feedback* from LLMs to optimize rewrites for business outcomes[^3_1][^3_3].
    - Offline supervised learning is complemented with online RL to adapt to changing catalogs and user preferences[^3_1][^3_3].
    - Custom reward models assess not just linguistic similarity but also *business KPIs* like product coverage and conversion rates.
- **Other Text Generation Tasks:**
    - Predominantly use supervised learning with human-annotated data.
    - RL is less common and, when used, is typically guided by linguistic or task-specific metrics rather than business outcomes.


### 5. **Nature of Input and Output**

- **E-commerce:**
    - Inputs are often *short, ambiguous, or noisy* queries.
    - Outputs must be *semantically equivalent* or intent-preserving rewrites, often constrained to be concise and catalog-aligned[^3_1][^3_5].
    - Rewrites are not meant to be creative or verbose but to maximize the match between user intent and product offerings.
- **Other Text Generation Tasks:**
    - Inputs can be longer and more structured (e.g., paragraphs, documents).
    - Outputs may be creative, paraphrased, or expanded, with fewer constraints on length or structure.


### 6. **Adaptation to Business and User Context**

- **E-commerce:**
    - Must rapidly adapt to new products, brands, and user trends.
    - Often incorporates *user session context* or *personalization* to improve rewrite relevance[^3_1][^3_2].
    - Needs to handle both head (frequent) and tail (rare) queries, with special attention to long-tail coverage[^3_1][^3_3].
- **Other Text Generation Tasks:**
    - May adapt to user preferences or context but typically with less direct business impact or need for real-time adaptation.

---

### Summary Table

| Aspect | E-commerce Query Rewriting | Other Text Generation Tasks |
| :-- | :-- | :-- |
| **Goal** | Improve search relevance, diversity, conversions | Fluency, coherence, fidelity, creativity |
| **Evaluation** | Product coverage, click/add-to-cart rates, user intent | BLEU, ROUGE, human ratings |
| **Domain Knowledge** | Dynamic, catalog/product-aware, brand-specific | General, less dynamic |
| **Latency** | Low-latency, real-time critical | Often less critical |
| **Learning Paradigm** | RL with business/user feedback, hybrid offline/online | Supervised, some RL with linguistic rewards |
| **Input/Output** | Short, ambiguous queries / concise, intent-preserving | Longer text, more flexibility |
| **Adaptation** | Catalog updates, trends, personalization | Less frequent, less business-driven |


---

### References

- [RL-Based Query Rewriting with Distilled LLM for online E-Commerce Systems][^3_1]
- [Enhancing Search Using Large Language Models – Whatnot Engineering][^3_2]
- [Enhancing E-commerce Search: Reinforcement Learning for Improved Language Model Finetuning][^3_3]
- [Retrieval-Augmented Spelling Correction for E-Commerce Applications][^3_5]
- [Spelling Correction using Phonetics in E-commerce Search][^3_6]

These differences underscore why query rewriting in e-commerce is a unique, high-impact, and operationally demanding subfield of text generation, requiring specialized models, evaluation metrics, and deployment strategies.

<div style="text-align: center">⁂</div>

[^3_1]: RL-Based-Query-Rewriting.pdf

[^3_2]: Enhancing-Search-Using-Large-Language-Models-_-by-Whatnot-Engineering-_-Whatnot-Engineering-_-Me.pdf

[^3_3]: Enhancing-E-commerce-Search_-Reinforcement-Learnin.pdf

[^3_4]: How-Instacart-Uses-Machine-Learning-Driven-Autocomplete-to-Help-People-Fill-Their-Carts-_-by-Est.pdf

[^3_5]: Retrieval-Augmented-Spelling-Correction-for-E-Commerce-Applications-Amazon-2024.pdf

[^3_6]: spelling-correction-using-phonetics-in-ecommerce-search-Amazon-2022.pdf

[^3_7]: Target-AutoComplete_-Real-Time-Item-Recommendations-at-Target.pdf

[^3_8]: https://arxiv.org/html/2501.18056v1

[^3_9]: https://arxiv.org/html/2403.09060v1

[^3_10]: https://sigir-ecom.github.io/ecom22Papers/paper_5298.pdf

[^3_11]: https://arxiv.org/pdf/2311.03758.pdf

[^3_12]: https://queryunderstanding.com/spelling-correction-471f71b19880

[^3_13]: https://aclanthology.org/2025.coling-industry.49.pdf

[^3_14]: https://aclanthology.org/2023.acl-industry.59.pdf

[^3_15]: https://aclanthology.org/2025.naacl-industry.65.pdf

[^3_16]: https://sdm-dsre.github.io/pdf/query_rewrite.pdf

[^3_17]: https://arxiv.org/pdf/2103.00800.pdf

[^3_18]: https://dl.acm.org/doi/10.1145/2187980.2187989


---

# What are the advantages of combining offline knowledge distillation with online reinforcement learning for query rewriting

Combining **offline knowledge distillation** with **online reinforcement learning (RL)** for query rewriting in e-commerce search delivers a set of complementary advantages that address both efficiency and adaptability-two critical requirements for production search systems.

## Key Advantages

### 1. **Efficiency and Low Latency for Online Serving**

- **Offline knowledge distillation** allows a large, powerful teacher model (such as an LLM) to transfer its semantic and linguistic capabilities to a much smaller, efficient student model. This student model (e.g., MiniELM) can be served in real time, significantly reducing inference latency and computational costs compared to using large LLMs directly[^4_2].
- This approach ensures that the model retains strong language understanding and query rewriting ability while being practical for high-traffic, production environments where response time is critical[^4_2].


### 2. **Strong Initial Performance and Robustness**

- The offline phase, which includes supervised fine-tuning and knowledge distillation, provides a robust warm-start for the student model. The model inherits the teacher’s ability to generate high-quality, semantically accurate rewrites, making it immediately effective upon deployment[^4_2].
- This mitigates the risk of poor performance during the initial stages of online adaptation and helps avoid issues with long-tail queries or rare search patterns[^4_2].


### 3. **Continuous Adaptation to Evolving Data**

- **Online reinforcement learning** enables the model to adapt dynamically to changes in user behavior, product catalogs, and emerging trends. As new products are added and user preferences shift, the model can update its rewriting strategies based on real-time feedback and reward signals[^4_1][^4_2].
- This continuous learning loop prevents the model from becoming stale, a common issue with purely offline approaches, and ensures ongoing relevance and diversity in query rewrites[^4_2].


### 4. **Alignment with Business Objectives and User Preferences**

- RL enables the use of custom, multi-faceted reward signals that reflect business goals-such as relevance, diversity, product coverage, and user engagement (clicks, add-to-cart, purchases)[^4_2]. These signals can be derived from real or simulated user feedback, ensuring that the model optimizes for outcomes that matter most in e-commerce[^4_2].
- By integrating these objectives into the online learning loop, the system can directly maximize metrics that drive business value and user satisfaction[^4_2].


### 5. **Scalability and Reduced Annotation Costs**

- The hybrid approach leverages simulated human feedback (using LLMs as judges) to generate scalable, cost-effective reward signals for RL, reducing the need for manual annotation and enabling rapid iteration[^4_2].
- This makes it feasible to maintain and improve the model at scale, even as the catalog and user base grow.


### 6. **Empirical Performance Gains**

- Experiments on large-scale e-commerce datasets (e.g., Amazon ESCI) show that this hybrid pipeline achieves significant improvements over both purely supervised and purely RL-based baselines. Gains are observed in query relevance, diversity, product coverage, and simulated user engagement metrics[^4_2].
- The approach outperforms traditional supervised models and previous RL-based methods, demonstrating its practical effectiveness for real-world e-commerce search[^4_2].

---

## Summary Table

| Aspect | Offline Knowledge Distillation | Online Reinforcement Learning | Combined Hybrid Pipeline (Advantage) |
| :-- | :-- | :-- | :-- |
| **Latency** | Low (student model) | Low (student model) | Real-time serving possible |
| **Initial Performance** | Strong (inherits teacher’s ability) | Depends on initialization | Robust from deployment |
| **Adaptability** | Limited (static after training) | High (adapts to new data) | Both robust and adaptive |
| **Business Alignment** | Indirect (via training data) | Direct (via reward signals) | Optimizes for business/user objectives |
| **Annotation Cost** | Moderate (for distillation dataset) | Low (simulated feedback) | Scalable, less manual labeling |
| **Empirical Results** | Good | Good | Best-in-class for e-commerce QR |


---

## Conclusion

**Combining offline knowledge distillation with online reinforcement learning for query rewriting unites the strengths of both worlds:** it delivers a lightweight, high-performing model ready for production, while ensuring ongoing adaptation to user needs and catalog changes. This hybrid approach improves query relevance, diversity, and business outcomes, all while maintaining the efficiency and scalability required for large-scale e-commerce platforms[^4_1][^4_2].

> "The MiniELM model balances performance and efficiency through offline knowledge distillation and online reinforcement learning. Experiments show improvements in query relevance, diversity, and user engagement. By using simulated interactions via LLMs, MiniELM adapts to evolving user preferences and catalogs, eliminating resource-intensive annotations and providing a scalable, cost-effective solution for dynamic e-commerce environments."[^4_2]

<div style="text-align: center">⁂</div>

[^4_1]: Enhancing-E-commerce-Search_-Reinforcement-Learnin.pdf

[^4_2]: RL-Based-Query-Rewriting.pdf

[^4_3]: Retrieval-Augmented-Spelling-Correction-for-E-Commerce-Applications-Amazon-2024.pdf

[^4_4]: spelling-correction-using-phonetics-in-ecommerce-search-Amazon-2022.pdf

[^4_5]: How-Instacart-Uses-Machine-Learning-Driven-Autocomplete-to-Help-People-Fill-Their-Carts-_-by-Est.pdf

[^4_6]: Enhancing-Search-Using-Large-Language-Models-_-by-Whatnot-Engineering-_-Whatnot-Engineering-_-Me.pdf

[^4_7]: Target-AutoComplete_-Real-Time-Item-Recommendations-at-Target.pdf


---

## How Offline Knowledge Distillation Improves the Accuracy of Query Rewriting

Offline knowledge distillation is a process where a large, high-performing "teacher" model (such as a large language model, LLM) transfers its knowledge to a smaller, more efficient "student" model. In the context of query rewriting for e-commerce search, this approach has been shown to significantly improve the accuracy and effectiveness of query rewriting systems. Here’s how:

### 1. **Transferring Advanced Language Understanding**

Large teacher models are trained on vast amounts of data and possess superior language understanding and semantic reasoning capabilities. Through knowledge distillation, these capabilities are transferred to the student model, allowing it to generate high-quality, semantically accurate rewrites that better capture user intent-even when the student model is much smaller and more efficient for real-time deployment[^5_1].

### 2. **Aligning with Task-Specific Objectives**

During the offline phase, both the teacher and student models are fine-tuned on a custom query-to-query (Q2Q) dataset derived from real e-commerce search logs, such as the Amazon ESCI dataset. This dataset is curated to ensure that query pairs are semantically equivalent and relevant to the e-commerce context. The distillation process further ensures that the student model not only mimics the teacher’s outputs but also aligns closely with the specific objectives of query rewriting-such as improving relevance, diversity, and user engagement in product retrieval[^5_1].

### 3. **Mitigating Limitations of Vanilla LLMs**

Vanilla LLMs, when used directly for query rewriting, tend to generate overly verbose or long-tail rewrites that may not be optimal for downstream search tasks. Offline supervised fine-tuning and knowledge distillation help constrain the student model to produce more concise, relevant, and actionable rewrites, directly addressing the needs of e-commerce search pipelines[^5_1].

### 4. **Improving Performance on Long-Tail and Ambiguous Queries**

Discriminative models and traditional methods often struggle with long-tail queries or those lacking sufficient historical data. By leveraging the teacher’s broad generalization capabilities, the student model learns to handle a wider variety of query formulations, including rare or ambiguous queries, thus improving overall coverage and accuracy[^5_1].

### 5. **Reducing Computational Overhead Without Sacrificing Quality**

Knowledge distillation enables the deployment of a lightweight student model that retains much of the teacher’s accuracy but is suitable for low-latency, high-throughput environments typical of e-commerce search. This ensures that accuracy improvements gained from advanced LLMs are not lost due to resource constraints[^5_1].

### 6. **Empirical Evidence of Accuracy Gains**

Experimental results on large-scale e-commerce datasets, such as Amazon ESCI, show that student models trained via offline knowledge distillation consistently outperform both vanilla student models and those trained solely with supervised fine-tuning. Metrics such as Exact Match, ROUGE-L, and Cross Entropy demonstrate significant improvements, and qualitative analyses confirm that rewrites are more relevant, diverse, and aligned with user intent[^5_1].

---

**In summary:**
Offline knowledge distillation improves the accuracy of query rewriting by transferring the nuanced language understanding and task-specific expertise of large teacher models into efficient student models, ensuring high-quality, relevant, and diverse rewrites that are well-suited for real-world e-commerce search environments[^5_1].

---

[^5_1] RL-Based-Query-Rewriting.pdf

<div style="text-align: center">⁂</div>

[^5_1]: RL-Based-Query-Rewriting.pdf

[^5_2]: Enhancing-E-commerce-Search_-Reinforcement-Learnin.pdf

[^5_3]: Retrieval-Augmented-Spelling-Correction-for-E-Commerce-Applications-Amazon-2024.pdf

[^5_4]: Enhancing-Search-Using-Large-Language-Models-_-by-Whatnot-Engineering-_-Whatnot-Engineering-_-Me.pdf

[^5_5]: How-Instacart-Uses-Machine-Learning-Driven-Autocomplete-to-Help-People-Fill-Their-Carts-_-by-Est.pdf

[^5_6]: spelling-correction-using-phonetics-in-ecommerce-search-Amazon-2022.pdf

[^5_7]: Target-AutoComplete_-Real-Time-Item-Recommendations-at-Target.pdf


---

## Challenges of Integrating Offline Knowledge Distillation with Online Reinforcement Learning

Combining **offline knowledge distillation** with **online reinforcement learning (RL)** for query rewriting in e-commerce search brings together the strengths of both paradigms-efficient, high-quality language understanding and dynamic, real-time adaptation. However, this hybrid approach also introduces several notable challenges across technical, operational, and data dimensions.

### 1. **Distribution Shift and Semantic Drift**

- **Challenge:**
Offline knowledge distillation creates a student model that closely mimics the teacher’s outputs on a fixed training distribution. However, once deployed, the online RL phase exposes the model to evolving user behaviors, new products, and changing query patterns. This can lead to a *distribution shift*, where the live data diverges from the offline training data, potentially causing *semantic drift* in the model’s outputs[^6_2].
- **Impact:**
The student model may generate rewrites that are less relevant or misaligned with current user intent and catalog updates if not continuously and carefully adapted.


### 2. **Reward Signal Quality and Stability**

- **Challenge:**
Online RL relies on reward signals to guide learning. In practice, these signals are often derived from simulated feedback (e.g., LLMs acting as judges) or proxy metrics (relevance, diversity, engagement). If the reward model is fixed or not updated to reflect the latest user preferences and catalog changes, it may provide inaccurate or off-distribution assessments as the policy evolves[^6_2].
- **Impact:**
Poor or stale reward signals can mislead the RL process, resulting in suboptimal or even degraded query rewrites over time[^6_1][^6_2].


### 3. **Balancing Efficiency and Adaptability**

- **Challenge:**
Offline distillation is designed for efficiency-producing a lightweight student model for low-latency inference. Online RL, however, requires frequent updates and may increase computational overhead, especially if reward models or judge LLMs are complex[^6_2].
- **Impact:**
Maintaining real-time performance while continuously updating the model is technically demanding, particularly at e-commerce scale.


### 4. **Complexity of System Integration**

- **Challenge:**
Integrating two distinct training regimes-offline distillation and online RL-adds architectural complexity. The system must support seamless transitions between offline and online phases, manage model versioning, and ensure robust rollback mechanisms in case of performance regressions[^6_2].
- **Impact:**
Increased engineering and operational overhead, with higher potential for integration bugs or deployment issues.


### 5. **Lack of Ground Truth and Evaluation Complexity**

- **Challenge:**
There is no single “correct” rewrite for a given query; evaluation must rely on indirect metrics (e.g., product coverage, simulated user feedback, engagement rates)[^6_2]. Online RL further complicates evaluation as the model’s behavior and the environment are both changing.
- **Impact:**
Measuring true improvement and diagnosing failures becomes more difficult, requiring sophisticated monitoring and A/B testing frameworks.


### 6. **Simulated vs. Real User Feedback**

- **Challenge:**
Many hybrid pipelines use LLMs as simulated human feedback to generate scalable reward signals. While scalable, this approach may not fully capture real user preferences and can introduce biases or artifacts specific to the judge model[^6_2].
- **Impact:**
Over-reliance on simulated feedback risks optimizing for artificial metrics rather than actual user satisfaction and business outcomes.


### 7. **Resource Requirements and Scalability**

- **Challenge:**
RL methods, especially those involving LLM-based judges or complex reward models, can be computationally expensive. As the model and catalog scale, so do the resource demands for both training and serving[^6_2].
- **Impact:**
High operational costs and potential bottlenecks in large-scale, real-time environments.

---

## Summary Table

| Challenge | Description | Impact |
| :-- | :-- | :-- |
| Distribution Shift/Semantic Drift | Live data diverges from offline training, causing model misalignment | Reduced relevance, need for continuous adaptation |
| Reward Signal Quality | Fixed or outdated reward models can misguide RL | Suboptimal or degraded rewrites |
| Efficiency vs. Adaptability | RL updates may increase computational load, affecting latency | Hard to balance speed and dynamic learning |
| System Integration Complexity | Managing offline and online phases, versioning, rollbacks | Higher engineering/operational overhead |
| Evaluation Difficulty | No ground truth; metrics are indirect and environment is non-stationary | Hard to measure/diagnose improvements |
| Simulated vs. Real Feedback | LLM-based judges may not reflect real user preferences | Risk of optimizing for artificial, not real, metrics |
| Scalability/Resource Use | RL and LLM-based feedback are resource-intensive | High costs, possible bottlenecks |


---

## References

- [RL-based Query Rewriting with Distilled LLM for online E-Commerce Systems][^6_2]
- [Enhancing E-commerce Search: Reinforcement Learning for Improved Language Model Finetuning][^6_1]

---

**In summary:**
While integrating offline knowledge distillation with online reinforcement learning offers a powerful and adaptive solution for query rewriting, it introduces substantial challenges in distributional robustness, reward signal fidelity, system complexity, evaluation, and scalability. Addressing these challenges requires careful system design, ongoing monitoring, and a balanced approach to both efficiency and adaptability[^6_1][^6_2].

<div style="text-align: center">⁂</div>

[^6_1]: Enhancing-E-commerce-Search_-Reinforcement-Learnin.pdf

[^6_2]: RL-Based-Query-Rewriting.pdf

[^6_3]: Retrieval-Augmented-Spelling-Correction-for-E-Commerce-Applications-Amazon-2024.pdf

[^6_4]: Enhancing-Search-Using-Large-Language-Models-_-by-Whatnot-Engineering-_-Whatnot-Engineering-_-Me.pdf

[^6_5]: How-Instacart-Uses-Machine-Learning-Driven-Autocomplete-to-Help-People-Fill-Their-Carts-_-by-Est.pdf

[^6_6]: spelling-correction-using-phonetics-in-ecommerce-search-Amazon-2022.pdf

[^6_7]: Target-AutoComplete_-Real-Time-Item-Recommendations-at-Target.pdf

[^6_8]: https://arxiv.org/html/2501.18056v1

[^6_9]: https://openreview.net/pdf/479828f926eaf3b40e44161a00c0a1ffd8f4b7fd.pdf

[^6_10]: https://www.scribd.com/document/839425613/2501-18056v1

[^6_11]: https://openreview.net/forum?id=_xxbJ7oSJXX

[^6_12]: https://proceedings.neurips.cc/paper_files/paper/2022/file/01d78b294d80491fecddea897cf03642-Paper-Conference.pdf

[^6_13]: https://huggingface.co/papers?q=iterative+queries

[^6_14]: https://www.computer.org/csdl/journal/sc/5555/01/10654534/1ZMv8kfiTUQ


---

## How Online Reinforcement Learning Adapts to User Feedback in Query Rewriting

Online reinforcement learning (RL) enables query rewriting models in e-commerce search to **continuously adapt and improve** based on user feedback and evolving platform dynamics. Here’s how this adaptation process works, drawing on recent research and real-world deployments:

---

### 1. **Feedback Collection and Reward Signal Construction**

- **User Feedback as Reward:**
The RL system collects signals from user interactions-such as clicks, add-to-cart actions, and purchases-on the products retrieved by rewritten queries. These actions serve as **implicit feedback** indicating the quality and relevance of the rewrites[^7_2].
- **Simulated Feedback:**
When real-time human feedback is scarce or costly, models can use **simulated feedback** from large language models (LLMs) acting as judges. These LLMs are prompted with user profiles, original queries, and product lists to predict likely user actions, providing scalable and cost-effective reward signals[^7_2].

---

### 2. **Reward Model and Multi-Objective Optimization**

- **Composite Reward Functions:**
The RL framework combines multiple objectives into its reward signal, such as:
    - **Relevance:** How well the rewritten query retrieves products aligned with the original user intent.
    - **Diversity:** Whether the rewrite expands the range of relevant products retrieved.
    - **Engagement:** Predicted or observed rates of clicks, add-to-cart, and purchases[^7_2].
- **Dynamic Adaptation:**
The reward model is continually updated to reflect the latest user preferences, catalog changes, and business objectives, ensuring the system remains aligned with real-world needs[^7_1][^7_2].

---

### 3. **Policy Update and Model Refinement**

- **Policy Optimization:**
The query rewriting model (policy) is updated in real time using RL algorithms such as Direct Policy Optimization (DPO) or Proximal Policy Optimization (PPO). At each step:
    - The model generates candidate rewrites for a sampled user query.
    - The reward model evaluates these rewrites based on the composite reward signal.
    - The model parameters are adjusted to increase the likelihood of generating rewrites that maximize expected rewards[^7_2].
- **Preference-Based Learning:**
The RL system may use preference pairs (chosen vs. rejected rewrites) to directly optimize for user-preferred outputs, further aligning the model with user expectations[^7_2].

---

### 4. **Continuous, Real-World Adaptation**

- **Handling Evolving Data:**
As user behavior, product catalogs, and search trends change, the RL-based system adapts its rewriting strategies in near real time, preventing staleness and semantic drift that can affect static, offline-trained models[^7_1][^7_2].
- **Improvement Over Time:**
Experimental results show that as the RL process continues, metrics such as relevance, diversity, and simulated user engagement **consistently improve**, reflecting the model’s ability to learn from ongoing feedback and deliver better query rewrites[^7_2].

---

### 5. **Scalable and Efficient Learning**

- **No Need for Manual Annotation:**
By leveraging simulated or implicit feedback, RL-based systems eliminate the need for resource-intensive manual labeling, making continuous adaptation feasible at scale[^7_2].
- **Lightweight Updates:**
Efficient RL algorithms ensure that the model can be updated frequently without incurring significant computational overhead, supporting large-scale, real-time deployment[^7_2].

---

## Summary Table

| Step | Mechanism | Example Metrics/Signals |
| :-- | :-- | :-- |
| Feedback Collection | Implicit (user actions) or simulated (LLMs as judges) | Clicks, add-to-cart, purchases |
| Reward Signal Construction | Multi-objective, composite rewards | Relevance, diversity, engagement |
| Policy Update | RL algorithms (DPO, PPO) using reward signals | Preference-based learning |
| Continuous Adaptation | Frequent updates, dynamic reward models | Improved relevance/diversity over time |
| Scalability | No manual annotation, efficient updates | Real-time, large-scale deployment |


---

## Key Takeaway

**Online reinforcement learning adapts query rewriting models to user feedback by continuously optimizing for relevance, diversity, and engagement using real or simulated user interactions as reward signals. This enables the system to stay aligned with evolving user needs and catalog changes, delivering more effective and personalized search experiences in e-commerce.**[^7_1][^7_2]

<div style="text-align: center">⁂</div>

[^7_1]: Enhancing-E-commerce-Search_-Reinforcement-Learnin.pdf

[^7_2]: RL-Based-Query-Rewriting.pdf

[^7_3]: Retrieval-Augmented-Spelling-Correction-for-E-Commerce-Applications-Amazon-2024.pdf

[^7_4]: Enhancing-Search-Using-Large-Language-Models-_-by-Whatnot-Engineering-_-Whatnot-Engineering-_-Me.pdf

[^7_5]: How-Instacart-Uses-Machine-Learning-Driven-Autocomplete-to-Help-People-Fill-Their-Carts-_-by-Est.pdf

[^7_6]: spelling-correction-using-phonetics-in-ecommerce-search-Amazon-2022.pdf

[^7_7]: Target-AutoComplete_-Real-Time-Item-Recommendations-at-Target.pdf


---

# How does the combination of offline and online methods enhance the diversity of query reformulations

The combination of **offline** and **online methods**-specifically, offline knowledge distillation and online reinforcement learning (RL)-significantly enhances the **diversity of query reformulations** in e-commerce search. This hybrid approach leverages the strengths of both paradigms to deliver more varied, relevant, and user-aligned query rewrites. Here’s how the combination works to boost diversity:

---

## 1. **Offline Knowledge Distillation: Establishing a Strong, Diverse Foundation**

- **Knowledge distillation** transfers the broad language understanding and semantic flexibility of a large teacher model (such as an LLM) into a smaller, efficient student model[^8_2].
- During this offline phase, the student model is trained on a carefully curated set of query-to-query pairs, often filtered for semantic equivalence and coverage of diverse product intents[^8_2].
- This process ensures the student model inherits the teacher’s ability to generate a wide range of semantically rich and contextually appropriate reformulations, covering both common and rare (long-tail) queries[^8_2].

**Result:**
The offline phase provides a robust base model capable of producing a variety of reformulations, but these are still limited by the static data and distribution seen during training.

---

## 2. **Online Reinforcement Learning: Dynamic, User-Driven Diversity Enhancement**

- In the **online phase**, the model is continuously fine-tuned using RL with reward signals that explicitly measure both **relevance** and **diversity** of the reformulated queries[^8_2].
- The RL reward functions often include:
    - **Diversity score**: Measures the proportion of distinct products retrieved by the reformulated query compared to the original, encouraging the model to generate rewrites that surface different sets of relevant products[^8_2].
    - **User engagement metrics**: Incorporate simulated or real user interactions (clicks, add-to-cart, purchases), further incentivizing the model to explore reformulations that appeal to a broader set of user intents[^8_2].
- By leveraging real-time feedback (or simulated feedback from LLMs), the model adapts to changing user behavior, catalog updates, and emerging trends, continually learning to generate novel and effective reformulations[^8_2].

**Result:**
The online RL phase pushes the model to explore new reformulation strategies, directly optimizing for variety and coverage in the retrieved results, beyond what was seen in offline training.

---

## 3. **Synergistic Effect: Greater and Sustained Diversity**

- **Offline distillation** ensures the model starts with a high baseline of linguistic and semantic diversity, avoiding the narrowness of purely rule-based or discriminative approaches.
- **Online RL** dynamically expands this diversity by rewarding reformulations that retrieve new, relevant products, adapt to user feedback, and avoid redundancy[^8_2].
- Experimental results show that such hybrid pipelines (e.g., MiniELM) outperform both standalone supervised and RL-based methods in **product coverage** and **diversity metrics**. For example, RL-based query reformulation approaches have demonstrated up to a **28.6% increase in product coverage** compared to standard generative models, with hybrid models achieving even higher gains[^8_1][^8_2].

---

## 4. **Practical Example**

- **Offline:** The model learns to rewrite “red necklace” as “necklaces in red,” “affordable red necklaces for women,” etc.-covering a range of phrasings and intents[^8_2].
- **Online:** By observing which rewrites lead to new products being clicked or purchased, the model further refines its outputs to include, for example, “ruby pendant,” “crimson beaded necklace,” or “garnet jewelry for women”-expanding both lexical and semantic diversity in real time[^8_2].

---

## 5. **Conclusion**

**Combining offline and online methods ensures that query reformulation models are not only efficient and accurate but also continually evolving to provide a richer, more diverse set of reformulations.** This leads to broader product discovery, improved user satisfaction, and increased business value for e-commerce platforms[^8_1][^8_2].

---

> “The MiniELM model balances performance and efficiency through offline knowledge distillation and online reinforcement learning. Experiments show improvements in query relevance, diversity, and user engagement. By using simulated interactions via LLMs, MiniELM adapts to evolving user preferences and catalogs, eliminating resource-intensive annotations and providing a scalable, cost-effective solution for dynamic e-commerce environments.”[^8_2]

---

**References:**

[^8_1]: Enhancing E-commerce Search: Reinforcement Learning for Improved Language Model Finetuning

[^8_2]: RL-Based Query Rewriting with Distilled LLM for online E-Commerce Systems

<div style="text-align: center">⁂</div>

[^8_1]: Enhancing-E-commerce-Search_-Reinforcement-Learnin.pdf

[^8_2]: RL-Based-Query-Rewriting.pdf

[^8_3]: spelling-correction-using-phonetics-in-ecommerce-search-Amazon-2022.pdf

[^8_4]: Retrieval-Augmented-Spelling-Correction-for-E-Commerce-Applications-Amazon-2024.pdf

[^8_5]: How-Instacart-Uses-Machine-Learning-Driven-Autocomplete-to-Help-People-Fill-Their-Carts-_-by-Est.pdf

[^8_6]: Enhancing-Search-Using-Large-Language-Models-_-by-Whatnot-Engineering-_-Whatnot-Engineering-_-Me.pdf

[^8_7]: Target-AutoComplete_-Real-Time-Item-Recommendations-at-Target.pdf


---

## Specific Techniques Used in Offline Knowledge Distillation for Query Rewriting

Offline knowledge distillation for query rewriting in e-commerce search involves a series of carefully designed steps to transfer the knowledge and capabilities of a large, high-performing teacher model (typically a large language model, LLM) to a smaller, efficient student model. The goal is to create a lightweight model that retains the teacher's semantic understanding and rewriting ability but is suitable for real-time deployment. The key techniques are as follows:

---

### 1. **Supervised Fine-Tuning (SFT) on a Custom Query-to-Query (Q2Q) Dataset**

- **Dataset Construction:**
A custom Q2Q dataset is built from historical e-commerce search logs, such as the Amazon ESCI dataset. This involves:
    - Creating a bipartite graph mapping queries to products based on relevance labels.
    - Identifying query pairs that are mapped to at least *k* similar products (labeled as "Exact" or "Substitute"), thus ensuring semantic equivalence.
    - Filtering candidate pairs using a strong LLM (e.g., Llama 3.3, 70B) to verify semantic equivalence, removing the need for manual annotation[^9_1].
- **Fine-Tuning:**
Both the teacher (large LLM) and student (smaller, efficient model) are fine-tuned on this curated Q2Q dataset, aligning them with the specific task of generating accurate, intent-preserving query rewrites for e-commerce[^9_1].

---

### 2. **Knowledge Distillation (KD) from Teacher to Student**

- **Distillation Objective:**
After supervised fine-tuning, an explicit knowledge distillation step is performed to transfer the nuanced language and rewriting skills of the teacher model to the student model.
- **Reverse Kullback-Leibler (KL) Divergence Loss:**
The distillation process uses a *reverse KL divergence* loss, which is particularly effective for generation tasks with large vocabularies:

$$
D_{KL}(P_T(x) \parallel P_S(x))
$$
    - $P_T(x)$ is the probability distribution of the teacher, and $P_S(x)$ is that of the student.
    - This loss function minimizes the student’s tendency to overestimate low-probability regions of the teacher’s distribution, focusing the student on the teacher’s high-relevance predictions (major modes).
    - This approach is better suited for generative tasks than standard classification losses, as it helps the student model capture the most important aspects of the teacher’s output distribution[^9_1].
- **Implementation:**
The student model is trained to match the output distributions of the teacher for each input query, ensuring it inherits the teacher’s performance while maintaining computational efficiency[^9_1].

---

### 3. **Filtering and Verification with LLMs**

- **Semantic Verification:**
Before finalizing the Q2Q dataset for training, a strong LLM is used to filter and verify that query pairs are truly semantically equivalent. This step automates what would otherwise be a manual annotation process, ensuring high-quality training data[^9_1].

---

### 4. **Evaluation and Metrics**

- **Offline Metrics:**
During and after training, the distilled student model is evaluated using metrics such as:
    - **Exact Match:** Whether the generated rewrite matches the reference exactly.
    - **ROUGE-L:** Measures overlap between generated and reference rewrites.
    - **Cross Entropy Loss:** Measures the model’s predictive performance[^9_1].
- **Qualitative Analysis:**
The rewrites are also qualitatively analyzed to ensure they are concise, relevant, and suitable for downstream e-commerce search tasks[^9_1].

---

### 5. **Practical Model Choices**

- **Model Architectures:**
Typical teacher-student pairs include:
    - Teacher: Large models like GPT2-large or Llama-3 8B.
    - Student: Smaller models like GPT2-base or Llama-3 1B, optimized for low-latency inference[^9_1].
- **Hyperparameters:**
Training hyperparameters are typically aligned with best practices for language model fine-tuning and distillation, as referenced in recent literature[^9_1].

---

## Summary Table

| Step | Technique/Tool | Purpose/Outcome |
| :-- | :-- | :-- |
| Dataset Construction | Bipartite graph, LLM filtering | High-quality, semantically equivalent Q2Q pairs |
| Supervised Fine-Tuning | SFT on Q2Q dataset | Aligns teacher and student to query rewriting task |
| Knowledge Distillation | Reverse KL divergence loss | Transfers teacher’s knowledge to student efficiently |
| Semantic Verification | LLM-based filtering | Ensures data quality, removes manual annotation need |
| Evaluation | Exact Match, ROUGE-L, Cross Entropy | Measures accuracy and suitability for e-commerce search |
| Model Choices | Large teacher, small student (e.g., Llama, GPT2) | Balances performance and efficiency |


---

## References

- The above techniques and implementation details are extensively documented in [RL-Based Query Rewriting with Distilled LLM for online E-Commerce Systems][^9_1].

---

**In summary:**
Offline knowledge distillation for query rewriting combines supervised fine-tuning on a carefully constructed Q2Q dataset, semantic verification using LLMs, and a distillation step with reverse KL divergence loss to produce a lightweight, efficient student model that closely matches the performance of a large teacher model, all while being practical for real-time e-commerce search deployment[^9_1].

---

[^9_1] RL-Based-Query-Rewriting.pdf

<div style="text-align: center">⁂</div>

[^9_1]: RL-Based-Query-Rewriting.pdf

[^9_2]: Enhancing-E-commerce-Search_-Reinforcement-Learnin.pdf

[^9_3]: Enhancing-Search-Using-Large-Language-Models-_-by-Whatnot-Engineering-_-Whatnot-Engineering-_-Me.pdf

[^9_4]: How-Instacart-Uses-Machine-Learning-Driven-Autocomplete-to-Help-People-Fill-Their-Carts-_-by-Est.pdf

[^9_5]: Retrieval-Augmented-Spelling-Correction-for-E-Commerce-Applications-Amazon-2024.pdf

[^9_6]: spelling-correction-using-phonetics-in-ecommerce-search-Amazon-2022.pdf

[^9_7]: Target-AutoComplete_-Real-Time-Item-Recommendations-at-Target.pdf

