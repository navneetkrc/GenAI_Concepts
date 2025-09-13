# 📂 **Part 2: RAG Pipeline & Evaluation Metrics**

---

## 🔄 **Basic RAG Pipeline Overview**

<img width="961" height="938" alt="Screenshot 2025-09-13 at 3 05 24 PM" src="https://github.com/user-attachments/assets/49a52292-fae7-4c15-8d64-bda2c7f869b8" />


A typical Retrieval-Augmented Generation (RAG) system follows three key stages that together ensure your LLM is both informed and accurate:

### 1️⃣ **Ingestion**

* ✅ Documents are split into meaningful chunks
* ✅ Each chunk is converted into embeddings
* ✅ Organized into an index for efficient search
  
  📂 **Goal:** Prepare data so it can be easily retrieved when needed

---

### 2️⃣ **Retrieval**

* ✅ The user query triggers a search
* ✅ The system fetches the most relevant chunks (Top-K results)

  
  📂 **Goal:** Supply the LLM with context directly related to the question

---

### 3️⃣ **Synthesis**

* ✅ The LLM processes retrieved context
* ✅ Generates a response based on the information
* ✅ Evaluated using RAG metrics

  
  📂 **Goal:** Ensure the response is accurate, relevant, and trustworthy

---

## 📊 **The RAG Triad: Core Evaluation Metrics**

To build a robust system, you need to measure how well it’s performing. That’s where the **RAG Triad** comes in.

---
<img width="962" height="996" alt="Screenshot 2025-09-13 at 4 34 40 PM" src="https://github.com/user-attachments/assets/e04df145-7ceb-4210-b603-66636e8252ff" />


---

### 📌 **Context Relevance**

Measures how closely the retrieved information aligns with the user’s query

✔ Helps identify retrieval issues

✔ Ensures the LLM is getting useful data

---

### 📌 **Answer Relevance**

Evaluates if the generated answer properly addresses the question

✔ Reflects the quality of the LLM’s reasoning

✔ Ensures user satisfaction

---

### 📌 **Groundedness**

Assesses whether the response is backed by the retrieved context

✔ Reduces hallucinations

✔ Builds trustworthiness in AI-generated responses

---

## 🚀 **Improving Retrieval & Evaluation Scores**

✅ Optimize document chunking and embeddings during ingestion

✅ Refine retrieval logic to boost context relevance

✅ Ensure the LLM synthesizes grounded and relevant answers

✅ Use the RAG triad for systematic, targeted improvements

---

## 📈 **Why This Matters**

* Streamlined workflows for question answering
* Enhanced accuracy and trust in AI responses
* Efficient experimentation and iteration
* Scalable, production-ready systems powered by metrics-driven insights

---
---

# 📚 **Advanced Retrieval Techniques: Notes Summary**

Learn how retrieval strategies impact LLM performance by improving **context relevance (CR), groundedness(G), and answer relevance (AR)**.

---

## 🔍 **1. Default Retrieval – Baseline Approach**

➡ **Definition:**
Top‑K independent chunks retrieved by vector similarity, without neighboring sentences or hierarchy.
📊 Corresponds to “Direct Query Engine” in leaderboard.

➡ **Observed Behavior:**

* Provides minimal context
* Lowest coherence and groundedness
* Example metrics:
  **CR:** 0.2550 | **G:** 0.80125 | **AR:** 0.930

➡ **Takeaway:**
Fastest but fragmented context → prone to hallucinations and weak grounding.

---

## 🟠 **2. Sentence‑Window Retrieval – Context Expansion**

---

<img width="968" height="994" alt="Screenshot 2025-09-13 at 4 40 13 PM" src="https://github.com/user-attachments/assets/2479cb85-eb4c-4300-8129-79f6b1afd151" />

---

➡ **Definition:**
After matching a sentence via embedding, retrieve it along with surrounding sentences to form a coherent passage.

➡ **Observed Behavior:**

* Better context flow
* Higher relevance and grounding
* Example metrics:
  **CR:** 0.3675 | **G:** 0.8780 | **AR:** 0.925

➡ **Takeaway:**
Improves local coherence by reducing boundary cuts.

---


## 🔵 **3. Auto‑Merging Retrieval – Hierarchical Context**


---

<img width="960" height="1013" alt="Screenshot 2025-09-13 at 4 43 14 PM" src="https://github.com/user-attachments/assets/b596496a-dedf-4b1a-b8f8-341dd649ba37" />

---
➡ **Definition:**
Documents are structured as a tree; small “child” chunks are dynamically merged into larger “parent” chunks when relevant.

➡ **Observed Behavior:**

* Most coherent, complete context
* Balances detail and brevity
* Example metrics:
  **CR:** 0.4350 | **G:** 1.0000 | **AR:** 0.940

➡ **Takeaway:**
Adaptive context sizing delivers the highest grounding with similar latency.

---

## 📈 **Why These Methods Work**

✅ **More Complete Context:**
Neighboring sentences and parent nodes reduce fragmentation → LLM sees a full semantic span.

✅ **Adaptive Retrieval:**
Auto‑merging adjusts context length dynamically → supports better answer relevance.

✅ **Measurable Improvement:**
Leaderboard example:

➡ CR rises from **0.2550 → 0.3675 → 0.4350**

➡ G rises from **0.80125 → 0.8780 → 1.0000**

---


## 📊 **Quick Comparison Table**

| 🔢 **Method**          | 📖 **How it Works**                                   | 📊 **Example Metrics (G / AR / CR)** | ✅ **Effect**                                        |
| ---------------------- | ----------------------------------------------------- | ------------------------------------ | --------------------------------------------------- |
| ✅ **Default**          | Top‑K chunks via embedding; no neighbors or hierarchy | 0.80125 / 0.930 / 0.2550             | Fast baseline, fragmented context                   |
| 🟠 **Sentence‑window** | Retrieve hit sentence plus neighbors for coherence    | 0.8780 / 0.925 / 0.3675              | Better local context, improved grounding            |
| 🔵 **Auto‑merging**    | Tree of chunks; promote relevant child to parent      | 1.0000 / 0.940 / 0.4350              | Largest, most coherent context with similar latency |

---
<img width="962" height="645" alt="Screenshot 2025-09-13 at 4 43 51 PM" src="https://github.com/user-attachments/assets/29b98804-4f0b-499d-af7a-1a5796068134" />

---
## 🎯 **Infographic Takeaways**

* **Goal:** Improve Context Relevance, Groundedness, and Answer Relevance by enriching retrieved information.
* **Default Retrieval:** Quick but incomplete → weak grounding.
* **Sentence‑Window:** Adds local neighbors → smoother passages → higher CR and G.
* **Auto‑Merging:** Dynamically expands context → full passages → top performance.
* **Key Insight:** Thoughtful context enrichment drastically boosts AI trustworthiness and answer quality.

---


