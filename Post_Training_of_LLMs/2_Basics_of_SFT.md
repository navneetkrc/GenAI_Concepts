# 📝 Short Notes: Basics of SFT

### 📌 What is SFT?

* **Supervised Fine-Tuning (SFT)**: The process of taking a **pretrained LLM** and training it further on **curated prompt–response pairs**.
* Goal: Make the model **follow instructions** and **align with desired style/format**.
* Training: Uses standard supervised loss (cross-entropy) where the model learns to predict the target response given a prompt.
* Typical use: First step before preference optimization (DPO / RLHF).

### ⚙️ Key Benefits

* Steers a general LLM into a **task-specific assistant**.
* **Stable, efficient, and interpretable** compared to RL-based methods.
* Requires only **labeled examples** (no preference/reward models).

---

# 🛒 SFT in E-commerce (Samsung Search Example)

### 🎯 Problem

* Users search with diverse queries:

  * **Short/product codes**: *“S23 Ultra”*
  * **Colloquial terms**: *“Samsung big fridge double door”*
  * **Feature-based**: *“AC with inverter and 5 star rating”*
* Base LLM may fail → misinterpret queries, rank irrelevant products, or hallucinate.

### 🔧 How SFT Helps

1. **Training Data**

   * Collect **query → ideal product set/description** pairs from logs, curated rules, or expert labeling.
   * Example:

     * Input: “best Samsung fridge for family of 4”
     * Target: Curated summary → *“Samsung 345L Double Door Refrigerator \[サムスン 345L 冷蔵庫], energy efficient, suitable for medium families.”*

2. **Fine-Tuning Process**

   * Train the LLM with these pairs so it **learns to map queries → correct product context**.
   * Loss function ensures the LLM reproduces high-quality answers instead of generic text.

3. **Result**

   * **Better semantic match**: “galaxy pad for study” → *Samsung Galaxy Tab \[サムスン ギャラクシータブ]* instead of phones.
   * **Reduced no-results queries**: Queries like *“apple ipad alternative”* mapped to *Samsung Galaxy Tab \[サムスン ギャラクシータブ]*.
   * **Improved instruction adherence**: Model always responds with structured product info (JSON, title, price, features).

---

✅ **Takeaway**:
SFT makes the model **e-commerce aware**, ensures **query understanding**, and **aligns responses to product catalog style**. It acts as the **foundation layer** before advanced alignment (DPO/RLHF).

---
---

# 🔑 Supervised Fine-Tuning (SFT): Best Use Cases

### 🎯 Context

SFT is the **first stage of post-training**. It takes a general pre-trained LLM and aligns it with **specific behaviors, tasks, or domains** using labeled input–output pairs.

👉 Think of it as teaching a “generalist” model to behave like a “specialist” in your application.

---

## 🏆 Best Use Cases for SFT

### 1. **Instruction Following & Style Alignment**

* **Goal:** Make the model follow human instructions consistently.
* **Example:** Base models might respond vaguely to *“Summarize this email in 3 bullet points.”*

  * After SFT on instruction–response pairs, the model learns **format control, tone, and brevity**.
* **Interview angle:** “SFT is what turns a base LM into an instruction follower before preference optimization like RLHF/DPO.”

---

### 2. **Domain Adaptation**

* **Goal:** Inject missing domain knowledge into the model.
* **Example:**

  * Fine-tuning on **medical dialogues** → model speaks in clinical terms, cites symptoms, avoids casual tone.
  * Fine-tuning on **legal contracts** → model handles clause extraction, compliance checks.
* **Interview angle:** “Continual pre-training gives broad exposure, but SFT makes models usable in specialized domains like healthcare, finance, or legal.”

---

### 3. **Search & Ranking (E-commerce)**

* **Goal:** Improve **query understanding** and **relevance ranking**.
* **Example (Samsung e-commerce):**

  * Query: *“budget phone with good camera”*
  * Fine-tune on query–product click data → model ranks **Galaxy A series** higher than irrelevant SKUs.
* **Behavior change:** The model learns **Samsung-specific vocabulary, synonyms, and product hierarchy.**

---

### 4. **Product Q\&A & Customer Support**

* **Goal:** Answer **factual, product-specific questions**.
* **Example:**

  * Generic model: *“Does Galaxy S24 support wireless charging?”* → may hallucinate.
  * After SFT on Samsung FAQs & manuals → precise answer: *“Yes, Galaxy S24 supports Qi wireless charging up to 15W.”*
* **Behavior change:** From vague/general → **factual, brand-aware, trustworthy.**

---

### 5. **Mapping Vague Queries to SKUs / Intents**

* **Goal:** Bridge **user intent → product catalog.**
* **Example:**

  * Query: *“best phone for gaming”*
  * Fine-tuned model maps to **Galaxy S24 Ultra** SKUs (with high GPU & battery).
* **Behavior change:** Model learns to resolve **implicit needs** into **explicit SKUs**.

---

### 6. **Safety & Policy Adherence**

* **Goal:** Enforce **company rules & guardrails**.
* **Example:**

  * SFT with filtered data teaches the model:

    * Not to give medical advice.
    * Always output JSON format for APIs.
* **Behavior change:** From “free-form answers” → **policy-compliant responses**.

---

## 💡 Interview Framing Tips

When asked **“Why use SFT?”** or **“Where would you apply it?”**:

* Start with the **general idea:**

  > “SFT is the fastest way to align a general LLM with domain-specific tasks, behaviors, and safety requirements.”

* Then give **concrete use cases with examples:**

  1. Instruction following (general → compliant formatting).
  2. Domain adaptation (general → medical/legal expert).
  3. E-commerce: query understanding, ranking, SKU mapping.
  4. Product Q\&A (brand-specific factuality).
  5. Policy adherence (consistent JSON, avoid restricted topics).

* End with a **behavioral punchline:**

  > “In short, SFT doesn’t just make a model smarter—it makes it act the way *you need it to behave in production*.”

---
---

# 📝 SFT Interview Prep – Infographic Style

---

## ⚡ What is SFT?

➡️ Supervised Fine-Tuning = train pre-trained LLM on **labeled input–output pairs**
➡️ First step before DPO / RLHF
➡️ Goal: **align model behavior** (instruction following, domain knowledge, safety)

---

## 🎯 Top Use Cases

**1. Instruction Following**
📌 JSON formatting, summarization, style control

**2. Domain Adaptation**
📌 Medical, Legal, Finance, E-commerce

**3. Samsung E-commerce Example**

* 🛒 Query understanding → “budget phone with good camera” → Galaxy A series
* 📱 Product QA → “Does S24 support wireless charging?” → Yes, Qi 15W
* 🎯 Vague queries → “best phone for gaming” → Galaxy S24 Ultra

**4. Policy & Safety**
📌 Avoid disallowed content, enforce rules

---

## 🛠️ How SFT Changes Model Behavior

**Before SFT:**

⚪ Generic responses

⚪ Not domain-specific

⚪ Inconsistent format

**After SFT:**

✅ Domain-aware (Samsung products)

✅ Consistent format (JSON, structured)

✅ Maps intent → action/product

✅ Safer & policy aligned

---

## 📌 Interview Quick Points

* 🕒 Fastest + cheapest way to specialize a model
* 📂 Needs **high-quality labeled data** (search logs, FAQs, QA pairs)
* ⚖️ Risk: too narrow = **catastrophic forgetting**
* 🔑 SFT sets **foundation**, DPO/RLHF refine preferences

---

## 💡 Elevator Answer

> “SFT fine-tunes LLMs on labeled pairs to align them with tasks.
> For Samsung e-commerce, it helps rank relevant products, answer FAQs, and map vague queries like *‘best phone for gaming’* to the correct SKUs. It’s the fastest way to make a base LM domain-ready.”

---
# 🎯 Supervised Fine-Tuning (SFT) – Cheat Sheet

## 📝 Basics of SFT
Supervised Fine-Tuning (SFT) = training a base LLM on **task-specific, labeled data**.  
It sets **new behavior** and **capabilities** by showing the model the *desired input → output* mapping.

- 📌 **Goal:** Align model responses with domain-specific requirements  
- ⚡ **Method:** Collect (prompt, response) pairs → fine-tune → validate  
- 🎛️ **Leverage:** Often the *first stage* before preference tuning (DPO / RLHF)  

---

## 🔑 Best Use Cases of SFT

### 1️⃣ Search Query Understanding & Ranking  
- **Problem:** Raw LLM may not rank Samsung products correctly for vague queries.  
- **SFT Solution:** Train on (query → top product SKUs) examples.  
- **Result:** Improves **search precision** for queries like:  
  - "best phone for night photography" → 📱 *Galaxy S24 Ultra [ギャラクシー S24 ウルトラ]*  
  - "budget Samsung fridge" → ❄️ *Samsung 300L Top Freezer [サムスン 300L 冷蔵庫]*  

---

### 2️⃣ Answering Product-Related Queries  
- **Problem:** Users ask detailed questions (“Does Galaxy Book support HDMI 2.1?”).  
- **SFT Solution:** Fine-tune on FAQ-style Q&A from Samsung manuals, specs, support docs.  
- **Result:** Reliable, context-aware answers without hallucination.  

---

### 3️⃣ Mapping Vague Queries → SKU  
- **Problem:** Query like *“Samsung big washing machine for family”* is fuzzy.  
- **SFT Solution:** Train on mapping vague → structured product SKUs.  
- **Result:**  
  - “big washing machine for family” → 🧺 *Samsung 10kg AI EcoBubble Washer [サムスン 10kg 洗濯機]*  
  - “flagship foldable” → 📱 *Galaxy Z Fold6 [ギャラクシー Z フォールド6]*  

---

## 🎨 Why SFT Works Well in E-commerce
- 🎯 Focuses LLM on **business-specific needs** (Samsung catalog, specs, FAQs)  
- 🛒 Reduces **hallucination risk** in customer-facing search & support  
- ⚡ Boosts **ranking relevance** and **answer reliability**  
- 🧩 Creates **foundation for safe alignment** (can layer validators, RAG, or RLHF after SFT)  

---

## 🧠 Interview Tip
👉 Frame SFT as the **first alignment step**: it teaches the model the *language of the domain*.  
Without it, post-training (RLHF/DPO) won’t have a strong base to align.  

---
# 🎯 Supervised Fine-Tuning (SFT) – Cheat Sheet

## 📝 Basics of SFT
Supervised Fine-Tuning (SFT) = training a base LLM on **task-specific, labeled data**.  
It sets **new behavior** and **capabilities** by showing the model the *desired input → output* mapping.

- 📌 **Goal:** Align model responses with domain-specific requirements  
- ⚡ **Method:** Collect (prompt, response) pairs → fine-tune → validate  
- 🎛️ **Leverage:** Often the *first stage* before preference tuning (DPO / RLHF)  

---

## 🔑 Best Use Cases of SFT

### 1️⃣ Search Query Understanding & Ranking  
- **Problem:** Raw LLM may not rank Samsung products correctly for vague queries.  
- **SFT Solution:** Train on (query → top product SKUs) examples.  
- **Result:** Improves **search precision** for queries like:  
  - "best phone for night photography" → 📱 *Galaxy S24 Ultra [ギャラクシー S24 ウルトラ]*  
  - "budget Samsung fridge" → ❄️ *Samsung 300L Top Freezer [サムスン 300L 冷蔵庫]*  

---

### 2️⃣ Answering Product-Related Queries  
- **Problem:** Users ask detailed questions (“Does Galaxy Book support HDMI 2.1?”).  
- **SFT Solution:** Fine-tune on FAQ-style Q&A from Samsung manuals, specs, support docs.  
- **Result:** Reliable, context-aware answers without hallucination.  

---

### 3️⃣ Mapping Vague Queries → SKU  
- **Problem:** Query like *“Samsung big washing machine for family”* is fuzzy.  
- **SFT Solution:** Train on mapping vague → structured product SKUs.  
- **Result:**  
  - “big washing machine for family” → 🧺 *Samsung 10kg AI EcoBubble Washer [サムスン 10kg 洗濯機]*  
  - “flagship foldable” → 📱 *Galaxy Z Fold6 [ギャラクシー Z フォールド6]*  

---

## 🎨 Why SFT Works Well in E-commerce
- 🎯 Focuses LLM on **business-specific needs** (Samsung catalog, specs, FAQs)  
- 🛒 Reduces **hallucination risk** in customer-facing search & support  
- ⚡ Boosts **ranking relevance** and **answer reliability**  
- 🧩 Creates **foundation for safe alignment** (can layer validators, RAG, or RLHF after SFT)  

---

## 🧠 Interview Tip
👉 Frame SFT as the **first alignment step**: it teaches the model the *language of the domain*.  
Without it, post-training (RLHF/DPO) won’t have a strong base to align.  


---
---

# 🧩 Principles of SFT Data Curation

High-quality **data curation** is the backbone of Supervised Fine-Tuning (SFT).  
Bad data → bad alignment.  
Good data → clear, reliable model behavior. ✅

---

## 1️⃣ Distillation 🧪
**What it is:** Use a stronger model (teacher) to generate high-quality training pairs for a smaller/follow-up model (student).

- 🔹 **Example:**  
  Teacher GPT-4 → generates product Q&A  


Q: Which Samsung phone has the best zoom camera?
A: Galaxy S24 Ultra with 100x Space Zoom.


Student LLM (Samsung model) fine-tuned on this distilled data.

- 📌 **Benefit:** Cheap way to transfer reasoning/knowledge from top-tier models → smaller task-specific models.

---

## 2️⃣ Rejection Sampling 🚫
**What it is:** Generate multiple responses per prompt, then keep only the best ones (based on human or automated preferences).

- 🔹 **Example:**  
Prompt: "Suggest a Samsung fridge for a family of 4 under ₹50,000."  
- R1: "Galaxy S24 Ultra is best." ❌ (hallucination)  
- R2: "Samsung 345L Double Door Refrigerator fits a family of 4." ✅  
- R3: "Buy any fridge from Amazon." ❌ (off-domain)  

Keep ✅ R2 → train on it.

- 📌 **Benefit:** Filters noise and improves **response reliability**.

---

## 3️⃣ Filtering 🧹
**What it is:** Automatically clean and prune training data to remove low-quality, toxic, irrelevant, or duplicate entries.

- 🔹 **Example:**  
- Remove ❌ “Buy iPhone 15 Pro” → not Samsung domain.  
- Remove ❌ toxic/offensive queries.  
- Deduplicate → only keep one version of “best Samsung phone for students.”  

- 📌 **Benefit:** Keeps dataset **domain-focused** and **safe**.

---

# 🎯 Why These Matter in Interviews
- 💡 Show you understand **data quality > model size**.  
- ⚖️ Distillation = leverage big models.  
- 🚫 Rejection sampling = enforce quality.  
- 🧹 Filtering = maintain domain focus.  

👉 Together, they ensure **SFT teaches the right behaviors** without introducing harmful or irrelevant outputs.


---

---
---
<img width="1427" height="786" alt="Screenshot 2025-09-05 at 7 43 31 PM" src="https://github.com/user-attachments/assets/2255afcd-1ebb-401d-a551-59c18c1edce5" />


---

# 📊 Full Fine-tuning vs Parameter-Efficient Fine-tuning (PEFT)

---

## 📝 Basics

* **Full Fine-tuning** → Update **all weights (W + ΔW)**
* **PEFT (LoRA, Adapters)** → Keep **original weights frozen**, only train small matrices (BA)

---

## ⚖️ Comparison

| Feature                | 🔵 Full Fine-tuning          | 🟠 PEFT (LoRA)                  |
| ---------------------- | ---------------------------- | ------------------------------- |
| **Parameters Updated** | All model weights            | Only small adapters             |
| **Compute / Memory**   | Very high ⚡💾                | Very low 🪶                     |
| **Learning Depth**     | Strong domain mastery        | Task-specific, lightweight      |
| **Forgetting Risk**    | Higher (may lose generality) | Lower (retains base skills)     |
| **Best Use**           | Critical domain tasks        | Rapid customization, multi-task |

---

## 🛒 Samsung E-commerce Search Use Cases

### 🔵 Full Fine-tuning

* Train on entire **Samsung product catalog + queries**
* Learns domain deeply:

  * Distinguish *Galaxy S24 Ultra vs S24+*
  * Map vague queries → SKUs (*“best Samsung fridge for 4 people under 60k”*)
* **Trade-off**: Expensive retraining when catalog changes

---

### 🟠 PEFT (LoRA)

* Plug-in adapters for **specific tasks/seasons**
* Examples:

  * **Seasonal adapter** → *“Samsung Diwali offers”* 🎉
  * **Query understanding adapter** → handle typos (*“samzung fridge”*)
  * **Comparison adapter** → *“S23 Ultra vs iPhone 15 Pro Max”*
* **Benefit**: Cheap, fast, switchable, safe

---

## 🎯 Interview Takeaway

* Full FT = **Deep, stable domain alignment**
* PEFT = **Agile, efficient, task-focused**
* In **Samsung e-commerce**, use **both**:

  * Full FT for **long-term catalog intelligence**
  * PEFT for **rapid market/seasonal adaptations**

---

👉 **Key Quote**:
*“Full fine-tuning gives deep mastery; PEFT gives agility. The real power is in combining them for scalable, adaptive e-commerce search.”*

---

---

# 🧩 PEFT Approaches – Interview Cheat Sheet

---

## 📊 Comparison of PEFT Methods

| 🔧 **Method**                          | ⚡ **Key Idea**                                        | 📊 **Params Trained** | 💾 **Memory Use**        | ✅ **Pros**                                                     | ⚠️ **Cons**                              | 🛒 **E-commerce Example**                              |
| -------------------------------------- | ----------------------------------------------------- | --------------------- | ------------------------ | -------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------ |
| 🔵 **LoRA** <br> (Low-Rank Adaptation) | Add small low-rank matrices **A,B** to weight updates | 🟢 \~0.1–1%           | 🟠 Moderate              | ✅ Memory-efficient <br> ✅ Adapters are swappable               | ⚠️ Needs full-precision model in GPU RAM | Seasonal adapter: *“Samsung Diwali offers”*            |
| 🟣 **QLoRA** <br> (Quantized LoRA)     | Quantize base to **4-bit** → apply LoRA               | 🟢 \~0.1–1%           | 🟢 Very low (4-bit base) | ✅ Train **65B+ models** on single GPU <br> ✅ Huge cost savings | ⚠️ Slight drop in precision              | Large-scale SKU mapping without cluster GPUs           |
| 🟡 **Prefix / Prompt Tuning**          | Train continuous **prefix tokens** at each layer      | 🟢 Few MBs            | 🟢 Very low              | ✅ Ultra-lightweight <br> ✅ Fast training                       | ⚠️ Limited capacity                      | Campaign queries: *“Summer sale Samsung AC”*           |
| 🟠 **Adapters**                        | Insert **small trainable modules** between layers     | 🟡 1–5%               | 🟠 Medium                | ✅ Modular, composable, reusable                                | ⚠️ Adds latency                          | Structured Q\&A on specs                               |
| 🟢 **BitFit**                          | Fine-tune only **bias terms**                         | 🔴 <0.1%              | 🟢 Negligible            | ✅ Ultra-fast <br> ✅ Very light                                 | ⚠️ Weak improvements                     | Typos like *“samzung fridge”*                          |
| 🔴 **Hybrid (LoRA+Prefix)**            | Combine **low-rank + prefix vectors**                 | 🟡 <5%                | 🟠 Medium                | ✅ Flexible & powerful                                          | ⚠️ Complex to manage                     | Comparison queries: *“S23 Ultra vs iPhone 15 Pro Max”* |

---

## 🎯 Key Takeaways for Interviews

* **LoRA** → The **standard workhorse** of PEFT (efficient, modular).
* **QLoRA** → Makes **large-scale fine-tuning possible on limited GPUs**.
* **Prefix / Prompt Tuning** → Good for **lightweight, temporary campaigns**.
* **Adapters** → Strong choice for **modular, reusable task-specific skills**.
* **BitFit** → Minimal tuning, good for **typos/noise robustness**.
* **Hybrid** → When you need both **capacity + efficiency**.

---

👉 **Interview soundbite:**
*"LoRA is the go-to adapter method, QLoRA pushes efficiency further with 4-bit quantization, and the rest of PEFT methods trade off between size, speed, and task specialization."*

---

---

# 🎤 More Interview Soundbites + Samsung E-commerce Examples

### 🟦 **LoRA (Low-Rank Adaptation)**

👉 *“LoRA hits the sweet spot between efficiency and performance — training only low-rank updates while keeping the base frozen.”*
💡 **Samsung Example**: Fine-tuning the base LLM for **holiday sale-specific queries** (“Diwali TV offers”, “Galaxy Z Flip discount”) without retraining the whole search model.

---

### 🟪 **QLoRA (Quantized LoRA)**

👉 *“QLoRA democratized LLM fine-tuning — you can adapt a 65B model on a single GPU by quantizing to 4-bit and applying LoRA.”*
💡 **Samsung Example**: Running **domain-specific fine-tunes at scale** — e.g., mapping vague queries like *“big screen for cricket world cup”* to **Samsung Neo QLED TVs**.

---

### 🟨 **Prefix / Prompt Tuning**

👉 *“Prefix tuning is like giving the model a soft system prompt baked into its layers — lightweight but limited in capacity.”*
💡 **Samsung Example**: Adding a **temporary prefix** for **seasonal campaigns**, e.g., ensuring the model prioritizes *“Back-to-school Galaxy Book offers”* during July–August.

---

### 🟧 **Adapters**

👉 *“Adapters let us plug in new skills like Lego blocks — modular and reusable, but at the cost of some inference latency.”*
💡 **Samsung Example**: Plugging in **adapters for regional catalogs** (India vs. Europe), so the same model can adapt quickly to **different product lineups** without retraining.

---

### 🟩 **BitFit (Bias Fine-Tuning)**

👉 *“BitFit is the minimalist’s approach — fine-tuning just bias terms; it’s fast, tiny, but only nudges performance.”*
💡 **Samsung Example**: Quick **bias correction for search relevance**, e.g., improving ranking when users in India type *“fridge”* vs. users in the US who type *“refrigerator.”*

---

### 🔴 **Hybrid Approaches (LoRA + Prefix, etc.)**

👉 *“Hybrid methods combine LoRA and prefix tokens to balance efficiency and expressiveness — useful when one method alone falls short.”*
💡 **Samsung Example**: Using **LoRA for general e-commerce tuning** + **Prefix tuning for live events** like *Samsung Galaxy Unpacked launches*, ensuring relevance without degrading other domains.

---

# 🎯 Meta Soundbites with Context

* 👉 *“PEFT is about steering big models with small knobs — in Samsung search, that means adapting to festivals, campaigns, and product launches in days, not months.”*
* 👉 *“Full fine-tuning is like repainting the entire Samsung store; PEFT is putting up banners and rearranging sections for an event.”*
* 👉 *“In e-commerce, PEFT lets us run agile fine-tunes — like a Diwali adapter or a Black Friday prefix — without retraining the entire LLM.”*

---


---
---
