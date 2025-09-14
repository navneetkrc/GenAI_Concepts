---
<img width="959" height="883" alt="Screenshot 2025-09-14 at 1 01 21 PM" src="https://github.com/user-attachments/assets/c94de7be-f3b8-488e-9b9d-0df4723701bb" />

---

---

<img width="963" height="996" alt="Screenshot 2025-09-14 at 1 01 43 PM" src="https://github.com/user-attachments/assets/ca95bc77-8afb-4f74-b328-1d08c1d175e3" />

---

# 📚 **Sentence-Window Retrieval – Infographic Notes**

---

## ✅ **What is Sentence-Window Retrieval?**

📖 It expands a retrieved sentence by including neighboring sentences above and below, forming a coherent, self-contained passage.
🎯 Helps LLMs by preserving discourse, qualifiers, timelines, or definitions that would otherwise be lost in isolated fragments.

---

## ⚙ **How It Works**

1. 🔍 **Embedding Lookup:** Find Top‑K sentences using standard embedding search.
2. ➕ **Add Local Context:** For each hit, attach adjacent sentences within a configurable window.
3. 📦 **Feed to LLM:** Augmented chunks are passed to the LLM for synthesis without altering index structure.

---

## 📈 **Impact on RAG Metrics**

✔ **Context Relevance:**
Nearby details enrich the topic match, aligning better with the query.

✔ **Groundedness:**
Fuller passages supply clear evidence, reducing hallucinations.

✔ **Answer Relevance:**
Contiguous context keeps the model on-topic for more accurate responses.

---

## 🔧 **Tuning Tips**

* 📏 **Window Size:**
  Start small → expand until key qualifiers, definitions, or timelines are consistently included.

* ⚖ **Balance Top‑K & Window:**
  Larger windows may require reducing Top‑K to stay within token limits while preserving quality.

* 🧩 **Chunking Strategy:**
  Use smaller base chunks before windowing to avoid redundant information.

---

## ✅ **When to Prefer It**

✔ Queries relying on qualifiers, caveats, or time-sensitive phrases
✔ Multi-sentence facts or definitions spread across several lines
✔ Cases where single-sentence retrieval strips away context necessary for reasoning

---

## 🔄 **Quick Workflow**

1. 🔍 Retrieve Top‑K sentences using embeddings
2. ➕ Expand each with neighboring sentences
3. 📦 Pass augmented contexts to the LLM
4. 📊 Score outputs using context relevance, groundedness, and answer relevance

---

## 🟠 **Key Takeaways**

* Sentence-window retrieval enriches context without changing index structures.
* It’s ideal for queries requiring nuanced or multi-sentence understanding.
* Careful tuning of window size and chunk balance ensures coherent, on-topic outputs within token constraints.

