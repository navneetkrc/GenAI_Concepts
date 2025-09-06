# 📝 Short Notes on DPO (Direct Preference Optimization)
---
<img width="1429" height="799" alt="Screenshot 2025-09-06 at 7 46 08 PM" src="https://github.com/user-attachments/assets/356126e6-fc37-4401-882f-80bca345a3f8" />

**Core Idea:**

* DPO fine-tunes LLMs by learning from **comparison data** (positive vs negative responses).
* Instead of reward models (RLHF), it directly optimizes model likelihood to prefer human-labeled *better answers* over worse ones.
* Uses **contrastive learning**: pushes model toward the preferred response, away from the rejected one.

**Formula Intuition:**

* Loss compares model’s log-probability of positive vs negative responses, normalized against a reference model.
* Effect: More stable than RLHF, easier to train, avoids reward hacking.

---

# 🏭 Industry Use Cases

### 🔹 E-commerce Search (Samsung Example)

1. **Query Disambiguation**

   * *User query:* “Samsung fold”
   * ❌ Negative response: Maps only to “Galaxy Z Fold 2 (outdated)”
   * ✅ Positive response: Suggests the latest **Galaxy Z Fold 6**, with context on upgrades.
   * 👉 DPO ensures search results prefer the **most relevant, updated SKU**.

2. **Answering Product Questions**

   * *Query:* “Does this TV support Netflix?”
   * ❌ Negative: “This TV has apps.” (vague, unhelpful)
   * ✅ Positive: “Yes, Samsung Neo QLED TVs support Netflix pre-installed.”
   * 👉 DPO improves **answer helpfulness** in product Q\&A.

3. **Personalized Recommendations**

   * *Query:* “Best phone for students”
   * ❌ Negative: Suggests high-end **Galaxy S Ultra** (expensive, irrelevant)
   * ✅ Positive: Suggests **Galaxy A series** with price–performance balance.
   * 👉 DPO aligns results with **human preferences**, not just popularity.

---

# 🎤 Interview Soundbite

👉 *“DPO helps align Samsung’s e-commerce search to **human preferences** — preferring answers that are accurate, helpful, and contextual. For example, when users search ‘Samsung fold’, DPO ensures the model suggests the **latest Galaxy Z Fold** instead of outdated models. This makes search more reliable and user-friendly compared to raw ranking models.”*

---
---

# ⚖️ RLHF vs DPO in E-commerce Search

---

## 🔹 **Training Setup**

* **RLHF** 🌀: Multi-step → SFT → Reward Model → PPO.
* **DPO** 🎯: Single-step → Directly optimize using preference pairs.

---

## 🔹 **Complexity & Cost**

* **RLHF** 💸: Expensive, needs reward model + online sampling.
* **DPO** ⚡: Cheaper, stable, no reward model required.

---

## 🔹 **E-commerce Example (Samsung Search)**

* **RLHF**:
  🔍 Query: *“Galaxy Fold”* → model overfits to **older Z Fold 2** (higher historical clicks).
* **DPO**:
  ✅ With preference pairs, model learns *Z Fold 6 > Z Fold 2* → ranks **latest Z Fold 6** higher.

---

## 🔹 **Answer Quality**

* **RLHF** 🗣️: Improves helpfulness/safety, but may get verbose.
* **DPO** 📏: Concise, directly aligned with user preference.

---

## 🔹 **Best Use**

* **RLHF** 🛡️: Needed when optimizing **safety, fairness, or long-horizon behavior**.
* **DPO** 🚀: Perfect for **fast, stable alignment** → search ranking, product Q\&A.

---

## 🎤 **Interview Soundbite**

👉 *“RLHF is powerful but heavy — it needs a reward model and online training. DPO is lighter: it directly learns from preference pairs. In Samsung e-commerce search, DPO ensures that when a user types ‘Samsung fold,’ the model surfaces the **latest Galaxy Z Fold 6** over outdated models. That’s stable alignment at lower cost.”*

---

# 🚀 Interview Prep Notes: Samsung Ecommerce Search – Cross-Encoder Latency vs Embedding Optimization

## 🎯 Scenario (Failure → Solution)

👉 **Failure**:

* We built a **Cross-Encoder model** to boost search relevance.
* But embedding-based retrieval is **a must-step** in ecommerce search pipelines.
* Adding a cross-encoder layer on top meant **two calls** → **retrieval + re-ranking**, leading to **latency issues**.
* ❌ Couldn’t deploy in production even though accuracy improved.

👉 **Solution**:

* Went with **only one call to embedding-based retrieval** (bi-encoder / two-tower).
* Used **cross-encoders only for offline metrics & relevance evaluation**.
* Improved embeddings directly via **SFT + DPO**, using click-driven positive/negative pairs → so embeddings alone could produce **high-quality retrieval results**.

---

## 🛒 Example in Samsung Ecommerce Search

* Query: `"best 5 star refrigerator 2024"`
* Plain embeddings: retrieved **older / irrelevant refrigerators**.
* Cross-encoder: accurate, but too slow for live search.
* ✅ Fine-tuned embeddings with click logs & curated relevance pairs →

  * Retrieved **newly launched 5★ models** first.
  * Brought **business-priority SKUs** (new launches, featured products) into top ranks.

---

## 📌 Training Signal – Loss Functions

We trained the embedding model using relevance signals from **clicks & curated pairs**.

| 🔧 Loss Function                    | ⚡ What it Does                                 | 📘 Mini Explanation                                                          | 🎯 Suitability for Our Case                     |
| ----------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------- |
| **Contrastive Loss**                | Pulls positives closer, pushes negatives apart | Query & correct product → closer in vector space, irrelevant ones → farther. | ✅ Works well with positive/negative click data. |
| **Triplet Loss**                    | Margin between pos & neg                       | Enforce query closer to pos than neg by margin.                              | Higher cost, not ideal at scale.                |
| **InfoNCE (Softmax Cross-Entropy)** | Multi-class discrimination                     | Query → “pick the clicked product” among candidates.                         | ✅ Scales well with click logs.                  |
| **Margin MSE Loss**                 | Learn relative ordering                        | Minimize score difference mismatch vs labels.                                | Useful for ranked data, but heavier.            |

---

## 🎯 Top 2 Choices for Samsung Use Case

1. **InfoNCE (Softmax Cross-Entropy)** → Matches click-data setup, scalable.
2. **Contrastive Loss** → Simple, effective for positive/negative pairs.

*(Triplet & Margin losses require complex sampling or larger compute, not ideal for production retraining cycles.)*

---

## 🎤 Interview Soundbite

> “In Samsung ecommerce search, we initially explored cross-encoders for ranking. But embedding-based retrieval is mandatory, and adding a second cross-encoder step increased latency beyond acceptable limits. Instead, we optimized embeddings directly using **click-driven fine-tuning with Contrastive and InfoNCE loss**. Cross-encoders are now used offline for evaluation, while embeddings alone power real-time retrieval. This way, we achieved strong improvements in relevance without sacrificing latency.”

---

⚡ This narrative shows:

* **Awareness of architectural trade-offs** (latency vs accuracy).
* **Practical pivot** to embeddings + PEFT/DPO.
* **Business impact** (faster retrieval + better relevance).

---
---

# ⚡ Samsung Ecommerce Search – Failure → Fix → Outcome

---

## ❌ The Challenge (Failure)

* Cross-Encoders gave **great accuracy** but:

  * Embedding retrieval is **mandatory** anyway.
  * Adding CE meant **2 calls → high latency**.
  * Not acceptable for **real-time ecommerce search**.

---

## 🔄 The Pivot (Fix)

* **Kept only one call** → embedding-based retrieval (bi-encoder / two-tower).
* **Cross-Encoders moved offline** → used only for **metrics & evaluation**.
* **Improved embeddings** directly using:

  * **Click logs** (positive/negative pairs).
  * **Business-priority SKUs** (new launches to top).
  * **SFT + DPO** for alignment.

---

## 🛒 Example (Samsung Context)

**Query:** `"best 5 star refrigerator 2024"`

* Before: Old/unpopular models ranked high.
* After SFT/DPO:

  * ✅ Latest 5★ refrigerators surfaced.
  * ✅ Business-priority products boosted.

---

## 📊 Training Loss Functions (with Click Data)

| 🔧 Loss         | ⚡ Role                                           | ✅ Why Useful Here                 |
| --------------- | ------------------------------------------------ | --------------------------------- |
| **Contrastive** | Pull positives closer, push negatives apart      | Simple & fits pos/neg click pairs |
| **InfoNCE**     | Softmax over candidates → choose clicked product | Scales well with large click logs |
| **Triplet**     | Enforce margin between pos & neg                 | Heavy sampling, less scalable     |
| **Margin MSE**  | Learn rank differences                           | More compute cost                 |

**🎯 Best for Samsung:**

1. **InfoNCE** (click-driven multi-class).
2. **Contrastive** (pair-based efficiency).

---

## 🎤 Interview Soundbite

> “We initially built cross-encoders for better ranking, but embedding retrieval was mandatory. Adding cross-encoders online doubled latency, so we pivoted to **embedding-only retrieval**, fine-tuned with **click-driven SFT/DPO**. Cross-encoders still help offline evaluation, but embeddings alone now power real-time search — balancing **latency and relevance** effectively.”

---
---
<img width="1429" height="799" alt="Screenshot 2025-09-06 at 7 46 08 PM" src="https://github.com/user-attachments/assets/8ac602cf-9279-4448-b542-2bfdb3d23604" />

# 🎯 DPO + Contrastive Learning in Samsung Ecommerce Search

---

## 🧩 Why DPO for Retrieval?

* Traditional **contrastive loss** (InfoNCE, Triplet) teaches embeddings to **separate positives from negatives**.
* But in **real search**, we often have *preferences* instead of just binary labels.

  * E.g., Product A clicked more than Product B → **A > B**, not just A = positive, B = negative.
* **DPO** directly optimizes embeddings to reflect **pairwise preferences** from clicks, purchases, or business rules.

---

## ⚡ How It Works

1. **Inputs:** Query + 2 products (chosen vs rejected).

   * Example: `"best Galaxy phone for gaming"`

     * ✅ Positive: *Galaxy S24 Ultra \[ギャラクシー S24 ウルトラ]* (clicked/purchased).
     * ❌ Negative: *Galaxy A15 \[ギャラクシー A15]* (not clicked).
2. **Model Objective:**

   * Encode `(query, product)` pairs.
   * Ensure **similarity(query, positive) > similarity(query, negative)** by a margin.
   * DPO formalizes this into a preference loss, grounded in probabilities.
3. **Result:** Embeddings align with **human-like preferences**, not just categorical matches.

---

## 🛒 Samsung Ecommerce Examples

### 1. **Search Query Understanding**

* Query: `"Samsung tablet for drawing"`

  * ✅ Positive: *Galaxy Tab S9 with S-Pen \[ギャラクシー タブ S9 Sペン付き]*
  * ❌ Negative: *Galaxy Tab A7 Lite \[ギャラクシー タブ A7 ライト]*
* **Improvement:** Instead of just matching the word “tablet,” embeddings learn that **S-Pen support is critical**.

---

### 2. **Ranking Newly Launched Products**

* Query: `"latest 8K TV"`

  * ✅ Positive: *Neo QLED 8K 2024 Model \[ネオ QLED 8K 2024モデル]*
  * ❌ Negative: *Older 2022 8K QLED*
* **Improvement:** DPO makes embeddings **time-sensitive**, ensuring **new launches rank higher** naturally.

---

### 3. **Handling Vague Queries**

* Query: `"Samsung washing machine for bachelors"`

  * ✅ Positive: *6kg Front Load SlimFit \[スリムフィット 6kg フロントロード]*
  * ❌ Negative: *12kg Family Size Top Load*
* **Improvement:** Embeddings learn to map **ambiguous lifestyle queries → correct SKU size**.

---

### 4. **Business Priority Boost**

* Query: `"best refrigerator for families"`

  * ✅ Positive: *Bespoke 5-Star 2024 Model \[ビスポーク 冷蔵庫 5つ星]*
  * ❌ Negative: *Older 3-Star model on discount*
* **Improvement:** DPO aligns embeddings with **business-driven SKUs** while keeping them relevant.

---

### 5. **Disambiguation Across Product Lines**

* Query: `"Samsung smart projector"`

  * ✅ Positive: *Freestyle Gen2 \[フリースタイル 第2世代]*
  * ❌ Negative: *4K UHD Monitor*
* **Improvement:** Prevents confusion between **adjacent categories** (projector vs monitor).

---

## 📊 Measurable Gains

1. **Higher CTR (Click-through rate)**

   * Users see the “right” products faster.
2. **Reduced Zero-Results Queries**

   * Even vague queries map to good embeddings.
3. **Improved Recall @ K**

   * More relevant products in top-10 retrieval.
4. **Business Alignment**

   * Embeddings respect **priority SKUs and launches**.

---

## 🎤 Interview Soundbite

> “We couldn’t afford cross-encoder latency, so we enhanced our **bi-encoder embeddings** using **DPO with click-derived preferences**. For example, when a user searched for *‘tablet for drawing’*, the model learned to prioritize **S-Pen supported Galaxy Tabs** over generic budget tablets. Similarly, vague queries like *‘washing machine for bachelors’* now map to slim-fit SKUs. This preference-based fine-tuning gave us \*\*higher CTR, better recall, and ensured that new launches like Bespoke refrigerators surfaced at the top — all within a single embedding retrieval step.”
---
---


# 🎯 DPO + Contrastive Learning in Samsung Ecommerce Search

---

## 🧩 Why DPO for Retrieval?

* Traditional **contrastive loss** (InfoNCE, Triplet) teaches embeddings to **separate positives from negatives**.
* But in **real search**, we often have *preferences* instead of just binary labels.

  * E.g., Product A clicked more than Product B → **A > B**, not just A = positive, B = negative.
* **DPO** directly optimizes embeddings to reflect **pairwise preferences** from clicks, purchases, or business rules.

---

## ⚡ How It Works

1. **Inputs:** Query + 2 products (chosen vs rejected).

   * Example: `"best Galaxy phone for gaming"`

     * ✅ Positive: *Galaxy S24 Ultra \[ギャラクシー S24 ウルトラ]* (clicked/purchased).
     * ❌ Negative: *Galaxy A15 \[ギャラクシー A15]* (not clicked).
2. **Model Objective:**

   * Encode `(query, product)` pairs.
   * Ensure **similarity(query, positive) > similarity(query, negative)** by a margin.
   * DPO formalizes this into a preference loss, grounded in probabilities.
3. **Result:** Embeddings align with **human-like preferences**, not just categorical matches.

---

## 🛒 Samsung Ecommerce Examples

### 1. **Search Query Understanding**

* Query: `"Samsung tablet for drawing"`

  * ✅ Positive: *Galaxy Tab S9 with S-Pen \[ギャラクシー タブ S9 Sペン付き]*
  * ❌ Negative: *Galaxy Tab A7 Lite \[ギャラクシー タブ A7 ライト]*
* **Improvement:** Instead of just matching the word “tablet,” embeddings learn that **S-Pen support is critical**.

---

### 2. **Ranking Newly Launched Products**

* Query: `"latest 8K TV"`

  * ✅ Positive: *Neo QLED 8K 2024 Model \[ネオ QLED 8K 2024モデル]*
  * ❌ Negative: *Older 2022 8K QLED*
* **Improvement:** DPO makes embeddings **time-sensitive**, ensuring **new launches rank higher** naturally.

---

### 3. **Handling Vague Queries**

* Query: `"Samsung washing machine for bachelors"`

  * ✅ Positive: *6kg Front Load SlimFit \[スリムフィット 6kg フロントロード]*
  * ❌ Negative: *12kg Family Size Top Load*
* **Improvement:** Embeddings learn to map **ambiguous lifestyle queries → correct SKU size**.

---

### 4. **Business Priority Boost**

* Query: `"best refrigerator for families"`

  * ✅ Positive: *Bespoke 5-Star 2024 Model \[ビスポーク 冷蔵庫 5つ星]*
  * ❌ Negative: *Older 3-Star model on discount*
* **Improvement:** DPO aligns embeddings with **business-driven SKUs** while keeping them relevant.

---

### 5. **Disambiguation Across Product Lines**

* Query: `"Samsung smart projector"`

  * ✅ Positive: *Freestyle Gen2 \[フリースタイル 第2世代]*
  * ❌ Negative: *4K UHD Monitor*
* **Improvement:** Prevents confusion between **adjacent categories** (projector vs monitor).

---

## 📊 Measurable Gains

1. **Higher CTR (Click-through rate)**

   * Users see the “right” products faster.
2. **Reduced Zero-Results Queries**

   * Even vague queries map to good embeddings.
3. **Improved Recall @ K**

   * More relevant products in top-10 retrieval.
4. **Business Alignment**

   * Embeddings respect **priority SKUs and launches**.

---

## 🎤 Interview Soundbite

> “We couldn’t afford cross-encoder latency, so we enhanced our **bi-encoder embeddings** using **DPO with click-derived preferences**. For example, when a user searched for *‘tablet for drawing’*, the model learned to prioritize **S-Pen supported Galaxy Tabs** over generic budget tablets. Similarly, vague queries like *‘washing machine for bachelors’* now map to slim-fit SKUs. This preference-based fine-tuning gave us \*\*higher CTR, better recall, and ensured that new launches like Bespoke refrigerators surfaced at the top — all within a single embedding retrieval step.”




---
---

# ⚖️ Contrastive Learning vs DPO in Samsung Ecommerce Search
---

# ⚖️ Contrastive Learning vs DPO Contrastive

```markdown
| 🧩 Aspect                     | 🔵 Standard Contrastive (InfoNCE / Triplet)                              | 🟢 DPO Contrastive (Preference-based)                           |
|-------------------------------|--------------------------------------------------------------------------|----------------------------------------------------------------|
| **Training Signal**           | Binary labels → Positive ✅ vs Negative ❌ pairs                          | Pairwise preferences → “A > B” ordering                        |
|                               | (e.g., query ↔ product match or not)                                     | (e.g., click data shows product A is preferred over product B)  |
|                               |                                                                          |                                                                |
| **Example**                   | Query: *“Tablet for drawing”*                                            | Query: *“Tablet for drawing”*                                   |
|                               | ✅ Tab S9 w/ S-Pen vs ❌ Random Tab A7 Lite                               | ✅ Tab S9 w/ S-Pen > ❌ Tab A8 (clicked less)                   |
|                               |                                                                          |                                                                |
| **Query Understanding**       | Captures **semantic similarity only**                                    | Captures **intent + preference context**                       |
|                               |                                                                          |                                                                |
| **Handling Vague Queries**    | “Washing machine for bachelors” → Any washing machine                    | Learns to rank **SlimFit 6kg SKU > 12kg Family SKU**           |
|                               |                                                                          |                                                                |
| **New Product Ranking**       | May still favor older SKUs with higher co-occurrence                     | **Prioritizes new launches** like 2024 Neo QLED 8K             |
|                               |                                                                          |                                                                |
| **Business Alignment**        | No direct encoding of SKU priorities                                     | Embeddings aligned with **priority SKUs** (e.g., Bespoke 5⭐ Fridge) |
|                               |                                                                          |                                                                |
| **Offline vs Online Fit**     | Optimizes generic similarity → may mis-rank                              | Directly optimizes **click / purchase-derived preferences**    |
|                               |                                                                          |                                                                |
| **Performance Impact**        | ✅ Faster training                                                       | ✅ Higher CTR, Recall@K, better query → SKU match               |
|                               | ❌ But weaker retrieval nuance                                            |                                                                |
|                               |                                                                          |                                                                |
| **Cost of Training**          | ✅ Lower compute requirement                                             | Slightly higher (pairwise), but **cheaper than full RLHF**     |
```

---
---

## 🎯 Interview Soundbite

> “With standard contrastive learning, our Samsung search embeddings only captured surface similarity — e.g., all tablets looked similar for a query like *‘tablet for drawing.’* By applying **DPO contrastive**, we trained embeddings on **click-based preferences**: the S-Pen Tab S9 was ranked above generic budget tablets. This approach not only improved **CTR and Recall\@K**, but also allowed us to **prioritize new launches and business-critical SKUs**, without adding latency like cross-encoders.”

---
---

# 🔄 Samsung Ecommerce Search – DPO Contrastive Pipeline

📌 User Query → "Tablet for drawing"

      │
      ▼
🟦 Query Encoder (Bi-Encoder Tower)
      │
      ▼
🟦 Product Encoder (Bi-Encoder Tower)
      │
      ▼
🔍 Embedding Retrieval
   (Top-K Products)

      │
      ▼
⚖️ DPO Contrastive Optimization
   • Positive: Galaxy Tab S9 w/ S-Pen ✅  
   • Negative: Tab A8 ❌  
   • Learned Preference: Tab S9 > Tab A8

      │
      ▼
🎯 Improved Ranking
   • Top-1 = Galaxy Tab S9  
   • Top-3 = Other S-Pen SKUs  
   • Lower Rank = Budget Tabs


---

# 🌟 Key Gains in Samsung Ecommerce Search

* **Better Query Understanding**: “Fridge for 2 people” → Bespoke 230L, not 600L family fridge.
* **New Launch Boosting**: 2024 Neo QLED 8K prioritized over older QLEDs.
* **Business Alignment**: EcoBubble 6kg washing machine surfaced for “washing machine for bachelors.”
* **Reduced Latency**: Retrieval-only pipeline, no cross-encoder at runtime.
* **Offline Evaluation**: Cross-encoders still used for relevance metrics, but embeddings power live search.

---

👉 **Interview Soundbite**:

> “We replaced online cross-encoder reranking with **DPO-optimized embeddings** to cut latency. Embedding retrieval now directly reflects **user preferences and business priorities**, e.g., for *‘tablet for drawing’*, the Tab S9 with S-Pen consistently ranks first. This balanced **performance and speed** in Samsung search.”

---


---
---



---

# 📝 Best Use Cases of DPO in E-commerce Search

---
<img width="1423" height="785" alt="Screenshot 2025-09-06 at 11 00 06 PM" src="https://github.com/user-attachments/assets/975836b4-08de-44c3-b341-8ac4c41494fe" />

## 1️⃣ Changing Model Behaviour

*(Identity, Multilingual, Instruction Following, Safety)*

* **Identity / Business Priorities**

  * *Query:* “best samsung phone under 50k”
  * ✅ *Galaxy S23 FE* (priority launch, great value)
  * ❌ *Older Galaxy S21* (outdated but semantically similar)
  * 👉 DPO shifts model behaviour to **align with Samsung’s launch roadmap**.

* **Multilingual Consistency**

  * *Query (Hindi):* “छोटा फ्रिज बैचलर के लिए” → *Small fridge for bachelors*
  * ✅ 190L Single-door Bespoke
  * ❌ 500L Family-size double door
  * 👉 DPO enforces **consistent preference ranking across languages**.

* **Instruction Following**

  * *Query:* “Samsung 1 ton AC, only inverter models”
  * ✅ Inverter, 5-Star AC
  * ❌ Non-inverter 3-Star
  * 👉 DPO ensures **filters/instructions are respected** during retrieval.

* **Safety / Brand Control**

  * *Query:* “Samsung phone with free charger”
  * ✅ Show standard *Galaxy S23* SKU page
  * ❌ Do not hallucinate “free charger bundle”
  * 👉 DPO adjusts behaviour to **avoid unsafe/wrong commitments**.

---

## 2️⃣ Improving Model Capabilities

*(Contrastive Advantage, Handling Vague/Complex/Multi-intent Queries)*

* **Contrastive Advantage → Better Ranking Nuance**

  * *Query:* “Samsung smart TV for gaming”
  * ✅ Neo QLED 120Hz with low latency
  * ❌ Entry-level Crystal UHD
  * 👉 DPO leverages preference pairs to **capture fine-grained feature importance**.

* **Handling Vague Queries**

  * *Query:* “fridge for bachelors”
  * ✅ Compact 190L Single-door
  * ❌ Large 500L Family fridge
  * 👉 DPO infers **lifestyle context → bachelor = small size**.

* **Complex Queries**

  * *Query:* “Samsung laptop for students under 60k, lightweight”
  * ✅ Galaxy Book3 (thin, affordable)
  * ❌ Galaxy Book Ultra (too costly, overpowered)
  * 👉 DPO captures **budget + use-case tradeoffs**.

* **Multi-intent Queries**

  * *Query:* “Best Samsung AC for summer + low electricity bill”
  * ✅ 1.5 Ton Inverter, 5-Star Energy rating
  * ❌ Cheaper 3-Star model
  * 👉 DPO optimizes for **joint satisfaction of multiple intents**.

---

🎯 **Interview takeaway:**
*“In Samsung e-commerce search, we applied DPO not only to change behaviour (like enforcing business identity, multilingual consistency, safe outputs) but also to expand capabilities—ranking with contrastive precision, and handling vague, complex, or multi-intent queries using click/purchase-derived preferences.”*

---
---
Perfect 🚀 — here’s a **2-page style prep sheet** in Markdown, structured for quick recall and strong storytelling in interviews.

---

# 📘 DPO in E-commerce Search (Samsung Use Cases)

---

## Page 1️⃣ — Changing Model Behaviour

| 🧩 Behaviour Aspect                | 📱 Samsung E-commerce Example                                                                      | 🎯 Why DPO Helps                                          |
| ---------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Identity / Business Priorities** | Query: *“best samsung phone under 50k”* → ✅ Galaxy S23 FE > ❌ Galaxy S21                           | Aligns retrieval with **new launches & priority SKUs**    |
| **Multilingual Consistency**       | Query: *“छोटा फ्रिज बैचलर के लिए”* → ✅ 190L Single-door > ❌ 500L Double-door                       | Ensures **consistent preferences across languages**       |
| **Instruction Following**          | Query: *“Samsung 1 ton AC, only inverter models”* → ✅ Inverter AC > ❌ Non-inverter                 | Embeddings **learn to follow constraints**                |
| **Safety / Brand Control**         | Query: *“Samsung phone with free charger”* → ✅ Galaxy S23 SKU page > ❌ “Free bundle” hallucination | DPO tunes model to **avoid unsafe or misleading outputs** |

👉 **Key Interview Line:**
*"DPO helps us control *how* the model behaves, making sure results reflect Samsung’s business priorities, respect instructions, and remain consistent across markets."*

---

## Page 2️⃣ — Improving Model Capabilities

| 🧩 Capability Aspect               | 📱 Samsung E-commerce Example                                                                                    | 🎯 Why DPO Helps                                                |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Contrastive Advantage (Nuance)** | Query: *“Samsung smart TV for gaming”* → ✅ Neo QLED 120Hz > ❌ Crystal UHD                                        | Learns **fine-grained ranking signals** (refresh rate, latency) |
| **Handling Vague Queries**         | Query: *“fridge for bachelors”* → ✅ 190L Compact fridge > ❌ 500L Family fridge                                   | Infers **contextual lifestyle intent**                          |
| **Complex Queries**                | Query: *“Samsung laptop for students under 60k, lightweight”* → ✅ Galaxy Book3 Thin > ❌ Galaxy Book Ultra        | Learns **budget + use-case tradeoffs**                          |
| **Multi-intent Queries**           | Query: *“Best Samsung AC for summer + low electricity bill”* → ✅ 1.5 Ton Inverter 5-Star > ❌ 3-Star Non-inverter | Optimizes for **joint satisfaction of multiple intents**        |

👉 **Key Interview Line:**
*"DPO enhances *capabilities* — it allows embedding models to capture preferences beyond raw semantic similarity, handling vague, complex, or multi-intent queries directly at retrieval time."*

---

# 🎯 Final Interview Hook

*"Initially, we trained cross-encoders, but due to latency constraints we couldn’t use them in live search. Instead of discarding our learnings, we applied **DPO on the bi-encoder embeddings**. This allowed us to integrate user preferences (clicks, purchases, business priorities) directly into retrieval, improving ranking quality while keeping latency low."*

---
---

### 📝 Explanation (Interview-ready)

* **Cross Encoder Path (Discarded)**

  * ✅ Accurate re-ranking
  * ❌ Too slow for real-time Samsung search

* **Embedding + DPO Path (Kept)**

  * ✅ Direct retrieval aligned with clicks, purchases, and business SKUs
  * ✅ One-shot retrieval → faster results
  * ✅ Learns fine-grained preferences (vague, multi-intent, new launches)
  * ⚡ Perfect trade-off: relevance + latency



---
---

# 📊 Before vs After: Cross Encoder → DPO Embeddings

| ⚡ Metric                                                 | ❌ Cross Encoder (Discarded)                          | ✅ DPO Embedding Retrieval                          |
| -------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------- |
| **Latency per Query**                                    | \~800ms – 1200ms                                     | \~120ms – 200ms ⚡                                  |
| **CTR (Click-Through Rate)**                             | +12% lift vs baseline                                | +10–11% lift (very close)                          |
| **Recall\@10**                                           | 0.72                                                 | 0.70 – 0.71 (slightly lower but stable)            |
| **Business-SKU Placement (New Launches, Priority SKUs)** | Often buried below older high-co-occurrence SKUs     | Surfaced higher thanks to **preference alignment** |
| **Scalability (QPS handled)**                            | ❌ Limited (heavy re-ranking)                         | ✅ Much higher throughput                           |
| **Use Case Fit**                                         | Great for **offline eval** and A/B testing relevance | Best for **online retrieval** in Samsung search    |

---

### 🎯 Interview Framing

*"We measured both paths carefully. The cross encoder gave the highest offline scores but introduced unacceptable latency. By finetuning our embedding model with DPO on click/purchase data, we preserved \~95% of the relevance gains while cutting response time by \~5–6x. This allowed us to scale Samsung’s e-commerce search with **both relevance and speed**."*

---
---

<img width="1424" height="794" alt="Screenshot 2025-09-06 at 11 06 36 PM" src="https://github.com/user-attachments/assets/c7007722-9ab6-4a5f-b44b-556b949f4b65" />

Got it ✅ — here are **short notes** for the slide with **Samsung e-commerce–specific examples** added so you can use them in interviews:

---

# 📝 Principles of DPO Data Curation (Short Notes)

### 🔹 Common Methods for High-Quality DPO Data

1. **Correction**

   * Take model’s weak response → treat as ❌ negative
   * Provide improved response → treat as ✅ positive
   * 📌 *Samsung Example*:

     * Query: *“Best phone for photography”*
     * Model: *“Galaxy A14”* ❌ (generic, weaker camera)
     * Corrected: *“Galaxy S24 Ultra with 200MP Pro Camera”* ✅

2. **Online / On-policy**

   * Generate multiple responses from the model for same query
   * Rank them using clicks / purchases / business rules
   * Pick **best as positive, worst as negative**
   * 📌 *Samsung Example*:

     * Query: *“slim refrigerator for bachelors”*
     * Model outputs:

       1. *“Bespoke Slim 275L 3-Star”* ✅ (preferred by clicks)
       2. *“Side-by-Side 700L”* ❌ (not matching intent)

---

### 🔸 Avoid Overfitting

* DPO may overfit to **shortcuts** in data.
* Example issue: Model always prefers responses with certain **keywords** even if irrelevant.
* 📌 *Samsung Example*:

  * If positive examples always contain the word *“5-Star”*, the model may **over-prioritize any fridge with 5-Star**, even if the query was about *“side-by-side family fridge.”*

---

# 🎯 Key Takeaway (Interview-Safe)

* **Correction + On-policy** → ensures realistic preference pairs for Samsung search.
* **Avoid shortcuts** → prevent model from over-prioritizing buzzwords (*5G, 5-Star, AI Camera*) instead of true relevance.
* **Result**: Samsung’s search embeddings better reflect **customer intent + business goals**, not just surface-level similarity.

---

# 📌 Principles of DPO Data Curation (Samsung E-commerce)

---

## 🔹 Common Methods for High-Quality Data

### 🛠️ Correction
- **Definition:** Take weak/incorrect model responses as negatives, improve them as positives.  
- **Samsung Example:**  
  ❌ Query: *“Best TV for gaming”* → Response: *“Samsung 32-inch Basic LED”*  
  ✅ Corrected Response: *“Samsung Neo QLED 65-inch with 144Hz refresh rate”*

---

### 🔄 Online / On-Policy
- **Definition:** Generate multiple responses from the model → pick best (positive) & worst (negative).  
- **Samsung Example:**  
  Query: *“Budget fridge under 300L”*  
  - Response 1: *“Bespoke 2-door 256L”* ✅  
  - Response 2: *“Family Hub 700L”* ❌  

---

## ⚠️ Avoid Overfitting
- **Problem:** DPO may latch onto shortcuts (e.g., preferring responses with buzzwords only).  
- **Samsung Example:**  
  ❌ Always preferring “Bespoke” because it appears in positive samples  
  ✅ Balanced curation ensures model doesn’t just rank by keyword, but by **query intent** (budget, use-case, features).

---

# 🎯 STAR Interview Practice

### ⭐ Example 1
- **Situation:** Cross-encoders improved search but added heavy latency.  
- **Task:** Deliver low-latency, high-quality retrieval for Samsung e-commerce.  
- **Action:** Applied DPO to fine-tune embedding model using click/preference data.  
- **Result:** Reduced query latency by **30%**, while improving Recall@10 by **15%**.

---

### ⭐ Example 2
- **Situation:** Customers searching vague queries like *“fridge for bachelors”* got irrelevant results.  
- **Task:** Improve retrieval to reflect real customer intent.  
- **Action:** Curated preference pairs (6kg SlimFit washer > 12kg family washer).  
- **Result:** CTR improved by **12%**, bounce rate dropped.

---

### ⭐ Example 3
- **Situation:** New Samsung product launches (e.g., Neo QLED) were buried under older popular SKUs.  
- **Task:** Ensure new products get surfaced appropriately.  
- **Action:** Used DPO contrastive learning (new launch SKUs > old SKUs).  
- **Result:** **Faster adoption** of new launches, aligning search with business needs.

---

### ⭐ Example 4
- **Situation:** Some model outputs overfit to “buzzwords” (always ranking “Bespoke”).  
- **Task:** Prevent shortcut learning.  
- **Action:** Balanced positive/negative samples across product families.  
- **Result:** Search results became **diverse yet relevant**, improving fairness across SKUs.

---

✨ **Key Takeaway:**  
DPO allows us to **align retrieval embeddings with user intent + business goals**, overcoming latency issues of cross-encoders while maintaining high relevance in Samsung e-commerce search.


---

