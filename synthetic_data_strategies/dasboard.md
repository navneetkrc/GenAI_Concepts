
---

## ✍️ Synthetic Query Types
- **Chunk-Level Queries**: Capture document-wide facts
- **Sentence-Level Queries**: Capture fine-grained details
- **Constraints-Based Queries**: Inject metadata/domain info

---

## 🧠 Context2ID vs Query2ID

| Data Type | What it Trains | Purpose |
|:---------|:---------------|:--------|
| Context2ID | Memorizing document content | Enables the model to "know" the corpus |
| Query2ID | Query understanding & retrieval | Trains matching user queries to correct documents |

---

## 🔥 Hard Negative Mining in Preference Learning
- Hard negatives = **Top-ranked wrong documents**.
- Preference Optimization Loss (RPO) **favors positives** over negatives.
- Avoid random negatives ➔ hurts model quality!

---

## 🧠 LLMs Used for Synthetic Data
| Task | LLM |
|:----|:---|
| Synthetic Query Generation | Mixtral 8x7B |
| Semantic Identifier Generation | Claude 3 Sonnet |

---

# 📋 4. Final Takeaways Dashboard

## ✅ Top 5 Key Learnings
- **Sentence-level queries** and **multi-granularity** matter.
- **Domain-specific constraints** enhance retrieval.
- **Context memorization** boosts performance.
- **Hard negatives** are crucial for ranking learning.
- **Synthetic data training** alone can reach SOTA levels!

---

## 🚀 Future Work Ideas
- **Incremental updates** to models without full retraining.
- **More complex queries**: multi-hop or multi-evidence.
- **Apply strategies to dense retrieval** domain adaptation.
- **Automatic metadata extraction** to create better constraint-based queries.

---

# ✨ End of Dashboard

> Made for better readability and professional presentation on GitHub.  
> Customize it with your repo theme/colors if needed! 🎨🚀
