RAG Techniques for Delhi and London Queries
Here are examples applying advanced RAG techniques to queries about Delhi and London, based on the source material:
1. Query Decomposition [1]
•
Concept: Breaking down complex queries into simpler sub-queries for more precise results [1].
•
Example Query: "Compare the air quality and public transportation systems of Delhi and London, and suggest which city is better for eco-conscious travelers."
•
Sub-Queries:
◦
"What is the current Air Quality Index (AQI) for Delhi?" [1]
◦
"What is the current Air Quality Index (AQI) for London?" [1]
◦
"Describe the public transportation system in Delhi." [1]
◦
"Describe the public transportation system in London." [1]
◦
"What initiatives are Delhi and London taking to promote eco-friendly travel?" [1]
2. Query Transformation [2]
•
Concept: Refining user queries to improve search result quality [2].
•
Raw Query: "Is Delhi or London more expensive to visit?" [2, 3]
•
Rewritten Query: "Compare the average daily cost of visiting Delhi and London, including accommodation, food, transportation, and attractions." [2, 3]
3. Query Expansion [4]
•
Concept: Broadening the original query to capture more relevant information using LLMs [4].
•
Raw Query: "What are some tourist attractions in Delhi?" [4]
•
Expanded Queries:
◦
"What are the historical landmarks to visit in Delhi?" [4]
◦
"What are the popular markets and shopping areas in Delhi?" [4]
◦
"What are the cultural and religious sites in Delhi?" [4]
◦
"What are some museums and art galleries in Delhi?" [4]
4. Query Routing [5]
•
Concept: Directing queries to specific pipelines based on content and intent [5].
•
Example: A system might route fact-based questions about Delhi and London to a pipeline that uses a knowledge graph, while routing questions requiring summarization or interpretation to another pipeline that uses web search capabilities [5, 6].
•
Fact-Based Query: "What is the population of Delhi?" [5] (Routed to a knowledge graph)
•
Interpretive Query: "How has the cultural exchange between Delhi and London influenced their respective art scenes?" [5] (Routed to a web search and summarization pipeline)
5. Metadata Filtering [7]
•
Concept: Using additional information attached to documents/chunks to refine retrieval [7].
•
Example: When searching for information on recent events in Delhi and London, filter by timestamp to prioritize articles published in the last year [8].
•
Query: "What are the recent developments in urban planning in Delhi and London?" [8]
•
Metadata Filter: Timestamps (last 12 months) [8]
6. Excluding Vector Search Outliers [9]
•
Concept: Managing the number of search results by excluding irrelevant matches [9].
•
Example: Setting a distance threshold to filter out results that are too dissimilar to the query [10].
•
Query: "Compare the street food of Delhi and London." [10]
•
Distance Threshold: Any result with a distance score above a set threshold is filtered out [10].
7. Hybrid Search [11]
•
Concept: Combining vector-based semantic search with keyword-based methods [11].
•
Query: "Best Indian restaurants in London Michelin star" [12]
•
Semantic Search: Identifies articles discussing highly-rated Indian cuisine and dining experiences in London [12].
•
Keyword Search: Ensures results contain the terms "Indian restaurants," "London," and "Michelin star." [12]
•
Alpha Parameter: Adjust the alpha parameter to balance the importance of semantic understanding and keyword matching [11].
These examples demonstrate how various advanced RAG techniques can be applied to queries about Delhi and London to improve the relevance and accuracy of the retrieved information [13].
