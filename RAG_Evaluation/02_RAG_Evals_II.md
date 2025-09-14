---
<img width="966" height="783" alt="Screenshot 2025-09-13 at 6 32 46 PM" src="https://github.com/user-attachments/assets/9b6f553b-7126-4b11-b1cd-8e00e49f63bc" />

---

---
<img width="960" height="1045" alt="Screenshot 2025-09-13 at 6 38 46 PM" src="https://github.com/user-attachments/assets/9ff901ee-dc55-4bf6-aff2-e38fed7057e8" />

---

---
<img width="963" height="1040" alt="Screenshot 2025-09-13 at 6 42 33 PM" src="https://github.com/user-attachments/assets/6c52455c-79ac-4f99-a195-2a8e1bf2bb9d" />

---



---
<img width="957" height="1023" alt="Screenshot 2025-09-13 at 6 31 56 PM" src="https://github.com/user-attachments/assets/be7f1152-3b58-4b29-88bd-f7ed84927ffa" />

---

<img width="960" height="1035" alt="Screenshot 2025-09-13 at 6 54 24 PM" src="https://github.com/user-attachments/assets/90000a37-09fd-40f2-9a2a-9d9b1527f54f" />

---

---

# 📊 **Evaluating RAG Retrievals and Responses — Infographic Notes**

---

## ✅ **Overview**

* Evaluation tracks **three core signals** from the Retrieval-Augmented Generation (RAG) pipeline:

  1. **Context relevance** – how useful retrieved passages are to the query
  2. **Groundedness** – whether the answer is backed by retrieved content
  3. **Answer relevance** – how well the final response addresses the question
* 📸 Screenshots (per‑chunk, mean context, final answer scoring) help guide targeted improvements to both retrieval and generation.

---

## 📚 **Context Relevance**

🔑 **Definition:**
Measures how relevant each retrieved chunk is to the query.
Example: Scores like 0.5 and 0.7 are averaged → mean context relevance of 0.6.

📝 **Example:**
A passage on job-seeking strategies scored 0.7 for answering “How can altruism benefit career growth?”

🚀 **Use:**
Refine chunking, indexing, and retriever parameters to maximize this score → improves the quality of inputs to the LLM.

---

## ✅ **Groundedness**

🔑 **Definition:**
Checks whether the response is supported by retrieved evidence, reducing hallucinations by requiring alignment.

🚀 **Use:**
Ensure retrieval focuses on relevant, trustworthy chunks and that answers explicitly reference or use supporting content.

---

## 🎯 **Answer Relevance**

🔑 **Definition:**
Evaluates whether the final answer addresses the question’s intent, regardless of whether support is cited.

📝 **Example:**
A response is scored 0.9 for directly answering the query with clear reasoning.

🚀 **Use:**
Tune prompts and rerank results so the LLM remains on-topic and effectively synthesizes retrieved context.

---

## ⚙ **What is a Feedback Function?**

📌 **Definition:**
A programmatic function that scores specific behaviors (e.g. chunk relevance, factual support) and returns both a numeric value and rationale.

✅ **Practical Use:**

* Scores per chunk → track retrieval quality
* Aggregates mean relevance → detect weak retrievals
* Evaluates final answers → ensure relevance and correctness

---

## 📈 **Trade-Off: Scalability vs Meaning**

| ➕ Scalable      | ✅ Automated evals like lexical overlap or MLM-based scoring | ❌ Less semantically rich      |
| --------------- | ----------------------------------------------------------- | ----------------------------- |
| ➖ Less scalable | ✅ Human or ground-truth evaluations with detailed insight   | ❌ Slow and resource-intensive |

💡 **Practical takeaway:**

✔ Use fast automated feedback for wide coverage

✔ Combine with occasional human reviews for depth and nuance

---

## 🛠 **Using Signals to Improve the System**

✅ **Boost Context Relevance**

* Refine chunking strategies
* Improve retrieval windows
* Rerank to highlight high-confidence passages

✅ **Enhance Groundedness**

* Ensure answers are evidence-backed
* Prompt models to cite or condition on retrieved content

✅ **Improve Answer Relevance**

* Align synthesis with query intent
* Constrain generation to retrieved material
* Encourage structured, on-topic responses

---

## 📋 **Quick Checklist**

✔ Define feedback functions for:

* Context relevance
* Groundedness
* Answer relevance
  Each returning a score and rationale.

✔ Track per‑chunk scores and aggregate mean relevance → identify weak retrievals early.

✔ Review answer-level feedback → ensure responses are on-topic and supported.

---

## 🏁 **Key Takeaways**

✅ Automated feedback ensures coverage and speed, while human or ground-truth evaluations add meaning where needed.

✅ Focusing on the triad — context relevance, groundedness, and answer relevance — with feedback loops improves both retrieval and generation.

✅ Balanced feedback frameworks make RAG systems more reliable, trustworthy, and aligned with user queries.

---
