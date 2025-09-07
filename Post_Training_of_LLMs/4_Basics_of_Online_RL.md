---

# 🔁 Reinforcement Learning for LLMs: Online vs Offline

---

<img width="1433" height="779" alt="Screenshot 2025-09-07 at 2 05 14 PM" src="https://github.com/user-attachments/assets/31bdb1bc-74e3-435d-82e6-bcc7a999f9fe" />




## 🟠 **Online Learning**

✨ *Learns in real-time with continuous feedback*

* ⚡ **How it works:** Model generates new responses → collects rewards → updates weights live.
* 📱 **Samsung Example:** When users start searching for *“Galaxy Z Fold6”*, search instantly adapts to rank foldables higher.
* ✅ **Pros:**

  * 🔥 Adapts quickly to trending products (e.g., festival offers, seasonal sales).
  * 🆕 Keeps recommendations fresh.
* ⚠️ **Cons:**

  * ❌ Risk of reinforcing noisy/incorrect signals.
  * 💻 High computational cost in real-time.

---

## 🔵 **Offline Learning**

📚 *Learns from pre-collected historical data*

* ⚡ **How it works:** Uses fixed prompt–response–reward tuples. No fresh generation during training.
* 📱 **Samsung Example:** Training on past queries like *“Galaxy S23 Ultra” → clicks on accessories* before deployment.
* ✅ **Pros:**

  * 🛡️ Stable & safe (no live instability).
  * 📊 Leverages massive logs (search, clicks, purchases).
* ⚠️ **Cons:**

  * 🐢 Slow to adapt to **new launches** (e.g., Fold6 just released).
  * ⏳ May miss sudden shifts in user behavior.

---

## 📌 **Interview Takeaway**

👉 In Samsung e-commerce, we combine both:

* **Offline RL** for stable learning from billions of historical logs.
* **Online RL** for fast adaptation to **new launches** and **seasonal demand shifts**.

---
---
# 🤖 Online RL: Let Model Explore Better Responses

---

<img width="1056" height="595" alt="Screenshot 2025-09-07 at 2 11 05 PM" src="https://github.com/user-attachments/assets/ba190c81-c62f-40d7-b291-1db468a152b6" />

---


## 🔄 **Core Flow**

1. 📥 **Batch of Prompts**
   → Users searching on Samsung.com (*“Best phone for gaming”*, *“energy-efficient fridge”*).

2. ⚙️ **Language Model Generates Responses**
   → Suggests ranked product lists (e.g., Galaxy S23 Ultra, Galaxy A54).

3. 🎯 **Reward Function Evaluates**
   → Based on **clicks, add-to-cart, purchase rate**.

   * Example: If more users click *Galaxy S23 Ultra* over *Galaxy A54*, that gets **higher reward**.

4. 🔁 **Update Weights**
   → Model learns **real-time preference signals**.

   * Next time, it shows **S23 Ultra higher for gaming queries** automatically.

---

## 📱 **Samsung e-Commerce Example**

* Query: *“Tablet for drawing”* 🎨

  * **Generated Response:** Galaxy Tab A8 + Galaxy Tab S9 Ultra.
  * **Reward Function:** Users overwhelmingly click Tab S9 Ultra (with S-Pen).
  * **Update:** Model learns → Rank **S9 Ultra > A8** for drawing-related queries.

* Query: *“Fridge for bachelors”* 🧑‍🍳

  * **Generated Response:** Large 12kg Bespoke Fridge + SlimFit 6kg Fridge.
  * **Reward Function:** SlimFit gets more clicks/purchases.
  * **Update:** Model prioritizes **SlimFit SKU** in future for “bachelor” queries.

---

## 📌 **Key Insight**

👉 Online RL allows Samsung’s search engine to **self-correct and adapt in real-time**:

* ⚡ New product launches (e.g., Fold6, Bespoke AI Oven) get boosted quickly.
* 📊 Aligns with **true customer intent** from clicks & purchases.

---
---

# 📊 Samsung e-Commerce: Online RL in Action

| ⚡ Scenario                              | ❌ Before RL (Static Search)                | ✅ After RL (Online RL Updates)                                         |
| --------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------- |
| 🎮 **Query: "Best phone for gaming"**   | S23 Ultra ranked below A54 (lexical bias). | S23 Ultra moves up after clicks → optimized for gaming intent.         |
| 🎨 **Query: "Tablet for drawing"**      | Tab A8 shown equally with Tab S9 Ultra.    | Tab S9 Ultra ranked higher (rewarded by user clicks on S-Pen feature). |
| 🧑‍🍳 **Query: "Fridge for bachelors"** | Large Bespoke fridge ranked top.           | SlimFit fridge boosted → aligns with small household preference.       |
| 🛒 **Query: "Budget washing machine"**  | Expensive AI Combo sometimes shown.        | Model adapts: ranks 7kg EcoBubble as top pick (higher conversion).     |
| 🔥 **New Launch: Galaxy Z Fold6**       | Low rank initially due to no history.      | Clicks & early adoption boost Fold6 visibility within days.            |

---

## 🚀 **Key Takeaway for Interviews**

* **Situation:** Samsung search engine struggled with static retrieval.
* **Task:** Improve ranking quality and adapt to real customer behavior.
* **Action:** Applied Online RL → model generated responses, reward functions captured clicks/purchases, weights updated iteratively.
* **Result:** Higher **CTR, conversions, and relevance** across phones, tablets, and appliances.

---
---

<img width="1424" height="792" alt="Screenshot 2025-09-07 at 2 14 18 PM" src="https://github.com/user-attachments/assets/061447d4-f119-4f52-b9a4-2b4f99faaccb" />


---

# 🎯 Reward Function in Online RL

**Option 1: Trained Reward Model**

---

## 📝 How It Works

1. 📄 Two responses (e.g., product rankings or summaries) are **judged by humans/customers**.
2. 🧠 Reward model assigns scores → `r_j` vs `r_k`.
3. ⚖️ Loss computed: `log(σ(r_j - r_k))` → updates model.
4. 🔄 Model learns which response is **preferred**.

---

## ⚡ Pros & Cons

* ✅ Initialized from **instruct model**, fine-tuned on preference data.
* 🌍 Works well for **open-ended tasks** (e.g., ranking, personalization).
* 🔒 Improves **chat quality & safety**.
* ⚠️ Less accurate for **strict correctness domains** (e.g., coding, math).

---

## 📦 Samsung e-Commerce Examples

| 🛒 Query                                | ❌ Less Preferred Response                        | ✅ More Preferred Response                                                 |
| --------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------- |
| **"Phone for photography"** 📸          | Ranks Galaxy A14 higher due to keyword “camera”. | Prioritizes Galaxy S24 Ultra (rewarded by users for pro camera features). |
| **"Budget fridge for bachelors"** 🧑‍🍳 | Suggests 500L Family Fridge (irrelevant).        | Rewards 190L Single-Door model (clicks show bachelor preference).         |
| **"Tablet for drawing"** 🎨             | Suggests Tab A8 (no stylus).                     | Rewards Tab S9 Ultra with S-Pen support.                                  |
| **"Latest Samsung TV"** 📺              | Shows older 2022 QLED models.                    | Rewards 2024 Neo QLED/Frame TV (human preference = newest).               |

---

## 🚀 Interview Takeaway

* **Situation:** Search often misranks Samsung SKUs due to semantic overlap.
* **Task:** Learn **true user intent** from clicks/purchases.
* **Action:** Applied a **trained reward model** → compares positive vs negative responses.
* **Result:** Higher **CTR, conversions, and alignment** with user preferences.

---
---

<img width="1432" height="781" alt="Screenshot 2025-09-07 at 2 16 47 PM" src="https://github.com/user-attachments/assets/e13c3876-418c-4c67-b318-1303ba805c95" />

---

# 🏷️ **Why Verifiable Rewards (Ground Truth) Matter — Short & Sharp**

**Core idea:**
When you can *directly check* a model’s output against a known-correct answer (a ground truth or unit test), the reward signal is deterministic, low-noise, auditable, and safe — often *more useful* than an opaque, large learned reward model.

---

## 🔍 Verifiable vs Learned Reward Models — Quick Comparison

| ✅ Aspect                        |              🔵 **Verifiable (Ground-truth / Unit-tests)** |              🟣 **Learned Reward Models (Large / Complex)** |
| ------------------------------- | ---------------------------------------------------------: | ----------------------------------------------------------: |
| **Signal quality**              |                                Deterministic / low-noise ✅ |                        Noisy, predictive — can be biased ⚠️ |
| **Interpretability**            |                    Transparent: pass/fail or exact score ✅ |                                  Opaque and hard to audit ❓ |
| **Robustness to hacking**       |                 Harder to game if tests cover edge-cases ✅ |                    Easily gamed / adversarially exploited ❌ |
| **Scalability cost**            |                     One-time test creation; cheap to run ✅ |   Requires many labels, continual retraining — expensive 💸 |
| **Handling distribution shift** |                   Traceable failures — easy to add tests ✅ |          Model drifts can break reward alignment silently ❗ |
| **Best fit**                    | Math, coding, factual QA, retrieval, unit-testable tasks ✅ | Complex human-preference signals, nuanced quality judgments |

---

## 🧩 Why prefer verifiable rewards (concrete reasons)

* **Deterministic feedback:** Exact matches (numbers, labels, test-pass) make training signals consistent.
* **Better debugging & audit trails:** Failed tests show *what* broke and *why*.
* **Safety & alignment:** Easier to enforce constraints (e.g., “never return PII”) with tests.
* **Cost-effective at scale:** Unit tests / label sets run cheaply across many examples.
* **Hard guarantees:** For mission-critical logic (pricing, billing, code), verifiability is essential.

**Interview punchline:**
“Where correctness is binary or verifiable (math, unit-tests, facts), ground-truth rewards give reliable, auditable supervision that learned reward models usually can’t match.”

---

## 📚 Verifiable Reward Datasets & Test-Style Suites (by domain)

### 🧮 **Math & Reasoning**

* **GSM8K** — grade-school math word problems with numeric answers (verifiable).
* **MATH** — competition-level math problems with definitive solutions.
* **Synthetic arithmetic sets** — perfect for deterministic checks.

### 💻 **Code & Programming**

* **HumanEval** (OpenAI) — coding problems with unit tests (pass/fail).
* **MBPP** (Mostly Basic Python Problems) — unit-test-based verification.
* **APPS** — programming problems with test suites for automatic scoring.
  **Why:** Unit tests provide direct reward signals (pass = good, fail = bad).

### 📖 **Reading Comprehension & QA**

* **SQuAD / NaturalQuestions** — ground-truth answer spans or exact answers.
* **DROP** — discrete reasoning / numeric answers that can be validated.
* **MS MARCO / TREC / BEIR** — retrieval datasets with relevance labels (verifiable metrics: MRR, NDCG).

### 🔎 **Fact-checking / Factuality**

* **FEVER** — claim verification with labeled evidence (entailed/refuted).
* **Fact-check datasets** (structured claims + verifiable labels).
  **Why:** You can programmatically check claim vs source evidence.

### 🧾 **Structured Business Logic**

* **Pricing & billing test suites** — synthetic or historical cases with known correct outputs.
* **Inventory & reconciliation tests** — deterministic checks (stock math, totals).

---

## 🛠️ Best Practices — Use Verifiable Rewards Effectively

1. **Start with unit-tests / golden answers** for core correctness paths.
2. **Combine:** use verifiable rewards for correctness + learned reward models for soft preferences (fluency, style).
3. **Cover edge cases:** adversarial/rare-case tests reduce brittleness.
4. **Automate continuous testing:** run tests in CI for model updates.
5. **Version & audit tests:** ground-truth datasets are your safety contract — track changes.

---

## ✅ Final Recommendation (one-liner)

**Use verifiable rewards as the foundation for correctness and safety; augment with learned reward models only where human preferences or nuance can’t be strictly specified.**

---

---
---
<img width="1429" height="782" alt="Screenshot 2025-09-07 at 2 35 43 PM" src="https://github.com/user-attachments/assets/dd2968ec-1cc9-43e1-8fbd-f2ceb1a38115" />

---

# 📘 **Policy Training in Online RL**

In online reinforcement learning (RL), models are trained by adjusting policies based on rewards and feedback. This slide introduces two key methods: **PPO** and **GRPO**.

---

## ✅ **PPO – Proximal Policy Optimization**

🎬 **Example – Netflix Movie Recommendations**

* **Policy Model:** Suggests movies based on user query like *“Best sci-fi movies to watch this weekend”*
* **Reference Model:** A pre-trained recommendation system to maintain consistency with past viewing trends
* **Reward Model:** Checks if the user actually clicks “Play” or watches the movie
* **Value Model:** Estimates long-term user engagement based on watch history

🔄 **How it works:**

* Compares new recommendations with previous ones using KL divergence
* Uses Generalized Advantage Estimation (GAE) to assess improvements
* Carefully updates suggestions without making radical changes that could confuse users

---

## ✅ **GRPO – Group Relative Policy Optimization**

🎧 **Example – Spotify Playlist Creation**

* Generates playlists for different user groups:

  * 🎸 Rock fans
  * 🎹 Jazz listeners
  * 🎤 Pop enthusiasts
* **Observations (o₁, o₂...oᴳ):** Inputs tailored to each group’s preferences
* **Group Computation:** Aggregates feedback across all users to enhance playlist suggestions

💡 **Why GRPO is useful:**

* Optimizes recommendations for various audience segments at once
* Enables sharing of learning across groups for faster improvements

---

## 📊 **Key Differences**

| Feature         | ✅ PPO                 | ✅ GRPO                          |
| --------------- | --------------------- | ------------------------------- |
| Scope           | Single user/query     | Multiple user groups            |
| Feedback type   | Personalized          | Collective                      |
| Stability focus | Limits abrupt changes | Encourages group-based learning |
| Applications    | Movie recommendations | Music playlists, news feeds     |

---

## 🔢 **Mathematical Objective (PPO)**

$$
J_{PPO}(\theta) = \mathbb{E}_{q,o}\left[ \frac{1}{|o|}\sum_{t=1}^{|o|} \min \left( \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\text{old}}(o_t|q, o_{<t})} A_t, \text{clip}\left( \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\text{old}}(o_t|q, o_{<t})}, 1-\epsilon, 1+\epsilon \right)A_t \right) \right]
$$

📌 Ensures controlled updates to the recommendation system during training.

---

## ✅ **Why This Matters for Platforms Like Netflix, Spotify, and Amazon**

* 🎯 Personalization → Increases user engagement
* 🔄 Smooth updates → Avoids confusing or irrelevant suggestions
* 🤝 Shared learning → Scales improvements across users
* 🚀 Efficient training → Better user experience without needing massive individual data

---
---

# 📘 **Policy Training in Online RL**

Online Reinforcement Learning (RL) trains models by updating policies based on rewards and feedback. This slide explains two methods: **PPO** and **GRPO**.

---

## ✅ **PPO – Proximal Policy Optimization**

📦 **Use Case – Samsung Product Recommendations**

* **Policy Model:** Generates personalized product suggestions based on user query “Best Samsung phone under ₹50,000?”
* **Reference Model:** A pre-trained recommendation system that ensures suggestions align with past patterns.
* **Reward Model:** Checks if the recommended products are clicked or purchased.
* **Value Model:** Estimates long-term satisfaction from these recommendations.

🔄 **Training Process:**

* Compares current suggestions with older ones using KL divergence.
* Uses Generalized Advantage Estimation (GAE) to assess improvements.
* Updates recommendations carefully without making large disruptive changes.

---

## ✅ **GRPO – Group Relative Policy Optimization**

📦 **Use Case – Samsung Bundle Offers**

* Generates multiple suggestions at once for different customer segments:

  * 📱 Mobile shoppers
  * 📺 TV buyers
  * 🧊 Appliances users
* **Observations (o₁, o₂...oᴳ):** Different inputs based on segment preferences.
* **Group Computation:** Aggregates feedback across segments to fine-tune recommendations.

💡 **Why GRPO?**

* Helps Samsung optimize offers for diverse user groups without needing separate training pipelines.
* Encourages sharing of learning across related tasks.

---

## 📊 **Key Differences**

| Feature         | ✅ PPO                        | ✅ GRPO                            |
| --------------- | ---------------------------- | --------------------------------- |
| Scope           | Single user/query            | Multiple segments                 |
| Feedback type   | Individual                   | Group-based                       |
| Stability focus | Limits large updates         | Encourages cross-segment learning |
| Applications    | Personalized recommendations | Multi-segment offers and bundles  |

---

## 🔢 **Mathematical Objective (PPO)**

$$
J_{PPO}(\theta) = \mathbb{E}_{q,o}\left[ \frac{1}{|o|}\sum_{t=1}^{|o|} \min \left( \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\text{old}}(o_t|q, o_{<t})} A_t, \text{clip}\left( \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\text{old}}(o_t|q, o_{<t})}, 1-\epsilon, 1+\epsilon \right)A_t \right) \right]
$$

📌 Controls how much the model is allowed to change during training.

---

## ✅ **Why This Matters for Samsung**

* 📦 Better product suggestions → More user satisfaction
* 📊 Balanced updates → Avoid sudden, confusing changes
* 🤝 Shared learning → Efficient training across segments
* 🚀 Scalable personalization → Improved shopping experience across devices

---
<img width="1430" height="789" alt="Screenshot 2025-09-07 at 2 57 31 PM" src="https://github.com/user-attachments/assets/f43c30a5-6966-47fa-b216-ebd5a5148883" />

---

### ✅ **PPO – Proximal Policy Optimization**

**Definition:**
PPO is a reinforcement learning algorithm that stabilizes training by limiting how much the policy is allowed to change between updates, using clipped objectives and divergence penalties.

**Interview Punchline:**
“It’s like teaching a model to learn without overreacting — ensuring safer, smoother, and more reliable improvements during training.”

---

### ✅ **GRPO – Group Relative Policy Optimization**

**Definition:**
GRPO extends PPO by aggregating observations and rewards from multiple tasks or user segments, enabling models to learn from shared feedback and improve across groups.

**Interview Punchline:**
“It’s the smart way to scale learning — letting models collaborate and generalize faster while optimizing for diverse tasks at once.”

---
---

# 📊 **Comparison: PPO vs GRPO in Online RL**

| ✅ Feature                | 📦 **PPO – Proximal Policy Optimization**                                                                                                    | 📂 **GRPO – Group Relative Policy Optimization**                                                                                                          |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📖 **Main Issue Before** | Training large models led to unstable updates → recommendations or responses would suddenly change, confusing users or degrading performance | Handling diverse user groups or tasks individually was inefficient → models couldn’t learn from related tasks, leading to slow or suboptimal improvements |
| 🧠 **How It Solved It**  | Carefully limits policy changes using KL divergence and clipped objectives → ensures updates are smooth and safe                             | Shares information across tasks by grouping observations → accelerates learning and improves performance across segments                                  |
| 📦 **Popular Use Cases** | Personalized suggestions, chatbots, reasoning tasks for individuals                                                                          | Multi-task learning, group-based recommendations, shared experiences across users                                                                         |
| 🤖 **Popular LLMs**      | OpenAI’s ChatGPT, Anthropic’s Claude, DeepMind’s Sparrow                                                                                     | Google DeepMind’s Gopher-Cluster variants, Meta’s LLaMA-Group, research prototypes for multi-domain assistants                                            |
| ⚙ **Training Method**    | Single-query focus → optimizes step-by-step learning for one interaction at a time                                                           | Group-query focus → computes group rewards and gradients, enabling collaborative learning                                                                 |
| 📈 **Strengths**         | More stable updates, better safety, great for reasoning tasks                                                                                | Faster convergence for multiple tasks, cross-learning, efficient use of feedback                                                                          |

---

### 📌 Summary

➡ **PPO** was designed to address instability during training when models made abrupt updates that could disrupt user experience.
➡ **GRPO** came later to solve scalability issues when handling multiple tasks or user segments, enabling shared learning across groups while keeping updates smooth.

---
---



