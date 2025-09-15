---

# 📊 **Maximizing RAG Triad Scores – Iterative Retrieval Strategy**

---

## ✅ **Goal**

➡️ Start with a simple retrieval baseline
➡️ Systematically vary sentence-window size
➡️ Track token cost vs relevance tradeoffs
➡️ Improve **context relevance**, **groundedness**, and **answer relevance**

> 💬 *“In interviews, I always stress starting simple. A well-defined baseline helps in isolating the real impact of any change, especially when dealing with complex models like RAG.”*

---

## 🧪 **Experiment Design**

📂 **Multiple Question Sets**
Run all app variants on the same evals → ensure fair comparisons across experiments.

📊 **Window Size Sweep**
Test sentence windows (e.g. 1, 3, 5) → study how more context affects model responses.

📈 **Use RAG Triad Metrics**
Measure context relevance, groundedness, and answer relevance → maintain consistent evaluation.

> 💬 *“Whenever I experiment, I treat evaluation as data-driven. Consistency in measurement avoids false positives and ensures improvements are reproducible.”*

---

## 🔄 **Iteration Loop**

1️⃣ **Start Simple**
Use LlamaIndex basic retrieval → find failure modes from limited context.

2️⃣ **Introduce Sentence-Window Retrieval**
Augment context with neighboring sentences → re-evaluate improvements.

3️⃣ **Keep Experimenting**
Adjust window sizes → identify configurations that maximize triad scores.

> 💬 *“Iteration is key — observing how changes in context affect downstream metrics helps us intelligently refine the approach rather than blindly tuning parameters.”*

---

## 📋 **What to Track**

🗂 **Experiment Logs**
Record window size, question sets, scores, latency, token usage → pick the best setup confidently.

⚖ **Token vs Relevance Tradeoff**
Balance longer windows (higher cost) against gains in context relevance → find the efficient frontier.

🔍 **Context vs Groundedness**
Track how improvements in context relevance often boost groundedness → better support for answers.

> 💬 *“I always recommend logging every run — even failures. They help trace performance trends and guide informed decisions in production pipelines.”*

---

## 💡 **Practical Tips**

▶ **Start Small**
Begin with window size 1 → gradually increase → avoid unnecessary cost early on.

🔄 **Compare After Every Change**
Re-run the triad metrics → side-by-side comparison ensures improvements are real.

📦 **Use Fixed Datasets**
Keep the evaluation harness and question pool constant → attribute differences to retrieval methods, not dataset drift.

> 💬 *“Starting small and verifying changes systematically helps build scalable solutions without wasting compute or introducing noise into experiments.”*

---

## ✅ **Key Takeaways**

* Iterative improvements rooted in data and metrics beat guesswork.
* Sentence-window retrieval can dramatically enhance context relevance and groundedness.
* Tracking token cost alongside accuracy helps optimize both performance and efficiency.
* Systematic experimentation with clear records leads to reliable, reproducible results.

> 💬 *“In interviews, I explain that optimization isn’t about brute force — it’s about structured experimentation and knowing where to invest effort for maximum impact.”*

---

<img width="956" height="864" alt="1" src="https://github.com/user-attachments/assets/6745e01d-9678-4a87-a97c-97ba671ef9cc" />

---

<img width="958" height="728" alt="2" src="https://github.com/user-attachments/assets/e1f646bc-8cd8-4380-91cc-4da013c053f8" />

---

<img width="958" height="877" alt="3" src="https://github.com/user-attachments/assets/4201e4e3-e3d5-48e1-ab4c-489845196656" />

---
