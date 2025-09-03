## Post-training of LLMs
<img width="1901" height="976" alt="Screenshot 2025-09-03 at 12 11 49 PM" src="https://github.com/user-attachments/assets/ae132101-6725-4ca1-a9bd-29afcfaf8ebd" />

**Post-training** is a stage where a pre-trained language model is further refined using curated, often instruction-focused, data to improve its ability to follow user directions and specific behaviors.

### Pre-training
- Starts with a **randomly initialized model** trained on broad data from sources like Wikipedia, Common Crawl, and GitHub.
- Produces a **base model** that predicts the next word/token but is not specialized for instructions or tasks.

### Instruction/Chat Model (Post-training)
- Uses post-training to learn from curated Q&A or instruction datasets.
- Results in a model that can respond to prompts, e.g., answering questions accurately.

### Continual Post-training & Customization
- (Continual) post-training is used to further adapt the model to new data, tasks, or changes, enhancing its capabilities or altering behaviors.
- Produces a **customized model** specialized for specific domains or tasks (e.g., SQL generation).

### Key Takeaways
- **Pre-training:** Broad, general knowledge learning.
- **Post-training:** Targeted fine-tuning for following instructions.
- **Customization:** Specialization for domains, tasks, or behaviors.

These steps enable LLMs to progress from generic language understanding to domain-specialized assistants through sequential training and fine-tuning.

---

<img width="1428" height="785" alt="Screenshot 2025-09-03 at 1 21 03 PM" src="https://github.com/user-attachments/assets/5b0875e9-b6b1-4f80-a7c7-d2bf3394a763" />


---
---

# 🧠 LLM Training & Post-training Overview

---

| 🔹 **Pre-Training**                                                                   | 🔹 **Post-Training (SFT)**                                                          |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 📚 **Data** → Unlabeled text (Wikipedia, Common Crawl, GitHub)                        | 🎯 **Data** → Labeled prompt–response pairs                                         |
| 🔄 **Process** → Predict next word (*“I like cats”*)                                  | 🔄 **Process** → Learn correct response imitation                                   |
| 📏 **Objective** →  <br> $\min_{\pi} -\log \pi(\text{next word} \mid \text{context})$ | 📏 **Objective** →  <br> $\min_{\pi} -\log \pi(\text{Response} \mid \text{Prompt})$ |
| ⚡ **Scale** → Very large datasets ($>>2T$ tokens)                                     | ⚡ **Scale** → Smaller datasets (\~1K–1B tokens)                                     |
| ✅ **Result** → General-purpose language model (no task-specific skills)               | ✅ **Result** → Instruction-following, relevant answers                              |

---

## 🔹 Extra Context: Post-training Path

🔧 **Purpose** → Adapt base LLM to **specific behaviors/domains**

🪜 **Steps** → SFT → RLHF → Continual Fine-tuning

🌐 **Impact** → Generic model 📝 → Assistant / Domain Expert 👩‍⚕️👨‍💻⚖️

---


---

# 🧠 Methods Used During LLM Training

---

## 🔹 Pre-Training (Unsupervised Learning)

📚 **Data** → Wikipedia, Common Crawl, GitHub (unlabeled)
🔄 **Process** → Predict next word (*“I like cats”*)
📏 **Objective** →

$$
\min_{\pi} -\log \pi(I) - \log \pi(\text{like} \mid I) - \log \pi(\text{cats} \mid I\, \text{like})
$$

⚡ **Scale** → $>>2T$ tokens
✅ **Result** → Broad **language understanding**, not task-specific

➡️ **Flow**: 📚 Data → 🔄 Prediction → 📏 Loss Minimization → ✅ General LLM

---

## 🔹 Post-training Method 1: Supervised Fine-tuning (SFT)

🎯 **Data** → Labeled prompt–response pairs

> Prompt: *“Explain LLM to me”*
> Response: *“LLM is …”*

🔄 **Process** → Model learns correct responses
📏 **Objective** →

$$
\min_{\pi} -\log \pi(\text{Response} \mid \text{Prompt})
$$

⚡ **Scale** → \~1K–1B tokens
✅ **Result** → Better **instruction following & relevance**

➡️ **Flow**: 🎯 Labeled Data → 🔄 Response Imitation → 📏 Loss Minimization → ✅ Instruction Follower

---

## 🔹 Extra Context: LLM Post-training


🔧 **Purpose** → Adapt base LLM to **specific behaviors/domains**

🪜 **Steps** → 1️⃣ SFT → 2️⃣ RLHF  → 3️⃣ Continual fine-tuning (coding, medical, legal, etc.)

🌐 **Impact** → Generic text generator 📝 → **Assistant / Domain Expert** 👩‍⚕️👨‍💻⚖️

➡️ **Flow**: 🔧 Adaptation → 🪜 Multi-step Training → 🌐 Domain Expertise

---
---

<img width="1433" height="798" alt="Screenshot 2025-09-03 at 4 58 39 PM" src="https://github.com/user-attachments/assets/b4799faa-e0dc-49bd-9c2f-11fec5d49e1c" />


---

# 🎯 Post-Training Methods for Aligning LLMs

LLMs need **alignment beyond SFT**. 
Two core methods are:
👉 **Direct Preference Optimization (DPO)**
👉 **Online Reinforcement Learning (RL / RLHF)**

---

## 🔹 Direct Preference Optimization (DPO)

* ⚖️ **Idea**: Train directly on *“good vs bad”* response pairs → no explicit reward model.
* 📏 **Objective**:
  $-\log \sigma\!\left(\beta[(\Delta \log \pi) - (\Delta \log \pi_{\text{ref}})]\right)$
* 🗂️ **Training**: Offline, preference datasets (1K–1B tokens).
* ✅ **Pros**: Simple, stable, compute-efficient.

**Models using DPO:**

* 🤖 Zephyr series (Zephyr-7B, dDPO).
* ⚡ StableLM Zephyr 3B (UltraFeedback).

➡️ **Flow**: Prompt → (Good vs Bad) → ⚖️ Preference Loss → ✅ Aligned Policy

---

## 🔹 Online Reinforcement Learning (RLHF)

* ⚖️ **Idea**: Policy → response → reward model → update with PPO.
* 📏 **Objective**:
Got it 👍 — let me rewrite the **Online RLHF objective** in a slide-ready way with **clean LaTeX math blocks** so it renders properly in markdown:

---

### ✅ Online RL (RLHF) Objective

* 🔄 **Training**: Online, large prompt sets, iterative updates.
* ✅ **Pros**: Flexible, supports safety shaping & long-horizon goals.

**Models using Online RL:**

* 🧑‍🏫 InstructGPT / early ChatGPT pipeline.
* 🦙 Llama-2-Chat (SFT → rejection sampling → PPO).
* 🐦 DeepMind Sparrow (RLHF + rules/evidence).
* 🧭 Anthropic Claude (Constitutional AI + RL from AI feedback).

➡️ **Flow**: Prompt → Policy → Reward Model → PPO Update → ✅ Safer/Helpful Policy

---

## ⚖️ When to Use Which?

| **DPO**                           | **Online RL (RLHF)**                |
| --------------------------------- | ----------------------------------- |
| ✅ Offline, preference pairs ready | ✅ Complex goals / safety trade-offs |
| ⚡ Lightweight, compute-efficient  | 🔄 Iterative, requires reward model |
| 🛠️ Easy to implement             | 🧪 More flexible but harder to tune |

🔗 **Hybrid pipelines**: SFT → DPO / rejection sampling → PPO-based RLHF → best balance of **helpfulness + safety + accuracy**.

---

# 🎯 Post-Training Alignment of LLMs

| 🔹 **Direct Preference Optimization (DPO)**                                                                | 🔹 **Online Reinforcement Learning (RLHF)**                                              |   |                         |
| ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | - | ----------------------- |
| ⚖️ **Idea**: Train on *good vs bad* response pairs → no reward model                                       | ⚖️ **Idea**: Policy generates → reward model scores → PPO updates                        |   |                         |
| 🗂️ **Training**: Offline, preference datasets (1K–1B tokens)                                              | 🔄 **Training**: Online, large prompt sets, iterative updates                            |   |                         |
| ✅ **Pros**: Simple, stable, compute-efficient                                                              | ✅ **Pros**: Flexible, supports safety shaping & long-horizon goals                       |   |                         |
| 🤖 **Models**: Zephyr-7B, StableLM Zephyr-3B                                                               | 🤖 **Models**: InstructGPT, Llama-2-Chat, Sparrow, Claude                                |   |                         |
| ➡️ **Flow**: Prompt → (Good vs Bad) → ⚖️ Preference Loss → ✅ Policy                                        | ➡️ **Flow**: Prompt → Policy → Reward Model → PPO Update → ✅ Policy                      |   |                         |

---

## ⚖️ When to Use Which?

* **DPO** → Straightforward, offline, tight compute, good preference data.
* **RLHF** → Complex goals, explicit safety trade-offs, long-horizon behavior.
* **Best practice** → Hybrid: **SFT → DPO/rejection sampling → PPO-based RLHF**.

---

---
<img width="1429" height="791" alt="Screenshot 2025-09-03 at 5 18 48 PM" src="https://github.com/user-attachments/assets/d8a493af-3490-4174-bb74-6bc34b3f7b8a" />

Here’s a **concise infographic-style markdown** for your slide. It puts **algorithms (left)**, **libraries (right)**, and **evaluation (bottom)** into one structured view, plus a practical recipe:

---

# 🔧 Keys to Successful LLM Post-Training

✨ Success = **Data + Algorithm** 🧩 **Libraries** ⚙️ **Evaluation** 📊

---

## 🧩 Three Elements

* 📘 **Data + Algorithm Co-Design** → choose SFT, DPO, RLOO, GRPO, PPO to fit data & compute.
* ⚙️ **Reliable Library** → maintained frameworks that scale training & inference.
* 📊 **Evaluation Suite** → mix automated + LLM-as-judge to track helpfulness, safety, robustness.

---

## 🔹 Algorithms at a Glance

* **SFT** → Supervised prompt-response pairs; first step for instruction following.
* **DPO** → Offline preference alignment; no reward model / online RL needed.
* **REINFORCE / RLOO** → Online RLHF with leave-one-out baselines; efficient vs PPO.
* **GRPO** → Group-relative optimization; less reward-model dependence, stable RLHF.
* **PPO** → Standard RLHF optimizer; reward model + KL control for safe updates.

➡️ **Flow**: 📘 Data → 🎯 Algorithm → ✅ Aligned Policy

---

## ⚙️ Libraries to Know

* **Hugging Face TRL** → Supports SFT, DPO, GRPO, PPO/RLOO; integrates with Transformers, DeepSpeed, vLLM.
* **OpenRLHF** → Scalable PPO/RLHF on Ray + vLLM, multi-GPU ready.
* **veRL** → Production RL stack (GRPO/PPO), modular APIs, LLM infra.
* **NeMo-RL** → NVIDIA’s stack for DPO/GRPO across 1–1000+ GPUs, with TensorRT-LLM.

---

## 📊 Evaluation Suite

* **MT-Bench** → Multi-turn reasoning & dialogue (LLM-as-judge).
* **AlpacaEval (LC)** → Fast head-to-head with length-bias fixes.
* **HELM** → Holistic benchmark: robustness, bias, toxicity, efficiency.
* **Knowledge/Truth Probes** → MMLU (knowledge), TruthfulQA (avoid falsehoods).

➡️ **Flow**: ✅ Aligned Policy → 📊 Benchmarks → 📈 Iteration

---

## 📝 Practical Recipe

1. **SFT** on curated instructions.
2. **DPO** for efficient offline alignment.
3. **RLHF (RLOO / GRPO / PPO)** for iterative goals or safety shaping.
4. **Standardize** training with a mature library.
5. **Gate changes** behind a stable eval dashboard (MT-Bench + AlpacaEval + HELM + safety probes).

---

---

<img width="1426" height="789" alt="Screenshot 2025-09-03 at 5 20 55 PM" src="https://github.com/user-attachments/assets/21660f87-68e8-47a8-823c-ab2ae6e84161" />


Here’s a **slide-ready infographic markdown text** version of your evaluation benchmark summary.
I’ve grouped items into panels, added icons 🎯📊🤖, and used concise bullets so it’s **easy to visualize**:

---

# 📊 LLM Evaluation Benchmarks – Infographic

## 👥 Human Preferences

* **Chatbot Arena** → Crowdsourced head-to-head chat battles, scored with Elo.
  🔹 Live, crowd-driven leaderboard of real user preferences.

---

## 🤖 LLM-as-Judge (Chat Quality)

* **AlpacaEval** → Single-turn, pairwise judging with LLM grader.
  🔹 Outputs win rates, fast iteration & leaderboard tracking.
* **MT-Bench** → Multi-turn prompts scored 1–10 by LLM judge.
  🔹 Measures coherence, helpfulness, reasoning.
* **Arena-Hard V1/V2** → Stress-test benchmark distilled from Arena logs.
  🔹 Offline proxy for Arena rankings.

---

## 📚 Static Instruction Benchmarks

* **LiveCodeBench** → Fresh coding problems, tests execution & self-repair.
* **AIME 2024/25** → Hard math exam-style reasoning challenges.
* **GPQA** → Graduate-level science Qs (physics, chemistry, bio).
* **MMLU-Pro** → Harder MMLU with reasoning-centric items.
* **IFEval** → Instruction-following with verifiable constraints.

---

## 🛠️ Function Calling & Agents

* **BFCL V2/V3** → Tool-use leaderboard (multi, parallel, multi-step).
* **NexusBench V1/V2** → Open suite for tool use & agents, reports accuracy.
* **TauBench** → Airline/retail workflows via APIs, pass\@k metrics.
* **ToolSandbox** → Multi-turn, stateful tool-use with simulated users.

---

## 🎤 Interview Cheat Sheet

* **Positioning**:

  * Arena → “real-world chat quality”
  * MT-Bench / AlpacaEval / Arena-Hard → “fast iteration”
  * MMLU-Pro / GPQA / AIME / IFEval → “reasoning, knowledge, compliance”
* **Agents & Tools**:

  * Cite BFCL / NexusBench / TauBench / ToolSandbox → reliability, planning, recovery.

---

⚡ **Pro-tip:** In interviews, map each eval to **what it measures** (preferences, chat, static reasoning, tools) and **why it matters** (iteration speed, trust, robustness).

---
---


<img width="1431" height="781" alt="Screenshot 2025-09-03 at 5 25 41 PM" src="https://github.com/user-attachments/assets/46f22ba5-d449-402f-b95b-cd689001450e" />



---

# 📊 LLM Evaluation Benchmarks – Infographic

## 👥 Human Preferences

* **Chatbot Arena** → Crowdsourced head-to-head chat battles, scored with Elo.
  🔹 Live, crowd-driven leaderboard of real user preferences.

---

## 🤖 LLM-as-Judge (Chat Quality)

* **AlpacaEval** → Single-turn, pairwise judging with LLM grader.
  🔹 Outputs win rates, fast iteration & leaderboard tracking.
* **MT-Bench** → Multi-turn prompts scored 1–10 by LLM judge.
  🔹 Measures coherence, helpfulness, reasoning.
* **Arena-Hard V1/V2** → Stress-test benchmark distilled from Arena logs.
  🔹 Offline proxy for Arena rankings.

---

## 📚 Static Instruction Benchmarks

* **LiveCodeBench** → Fresh coding problems, tests execution & self-repair.
* **AIME 2024/25** → Hard math exam-style reasoning challenges.
* **GPQA** → Graduate-level science Qs (physics, chemistry, bio).
* **MMLU-Pro** → Harder MMLU with reasoning-centric items.
* **IFEval** → Instruction-following with verifiable constraints.

---

## 🛠️ Function Calling & Agents

* **BFCL V2/V3** → Tool-use leaderboard (multi, parallel, multi-step).
* **NexusBench V1/V2** → Open suite for tool use & agents, reports accuracy.
* **TauBench** → Airline/retail workflows via APIs, pass\@k metrics.
* **ToolSandbox** → Multi-turn, stateful tool-use with simulated users.

---

## 🎤 Interview Cheat Sheet

* **Positioning**:

  * Arena → “real-world chat quality”
  * MT-Bench / AlpacaEval / Arena-Hard → “fast iteration”
  * MMLU-Pro / GPQA / AIME / IFEval → “reasoning, knowledge, compliance”
* **Agents & Tools**:

  * Cite BFCL / NexusBench / TauBench / ToolSandbox → reliability, planning, recovery.

---

⚡ **Pro-tip:** In interviews, map each eval to **what it measures** (preferences, chat, static reasoning, tools) and **why it matters** (iteration speed, trust, robustness).

---
---
<img width="1431" height="781" alt="Screenshot 2025-09-03 at 5 25 41 PM" src="https://github.com/user-attachments/assets/5b245c43-fc6c-438e-96d7-c7f05031516e" />


---

# 🎯 When to Use Post-Training vs Alternatives

👉 **Short answer**:
Post-training is needed when **reliably changing behavior** or **boosting targeted capabilities**.
Otherwise, prompting, RAG, or continual pre-training may be **simpler, cheaper, safer**.

---

## 📝 Few Rules Only

* **Method**: Prompting / system prompts → enforce small policies (“don’t discuss X”, fixed output format).
* ⚠️ **Caveat**: Simple but brittle → may break under paraphrase, long context, or adversarial input.
* ✅ Use **validators / guardrails** for safety.

---

## 🔎 Real-Time Knowledge

* **Method**: Retrieval-Augmented Generation (RAG) pipes in current / user / proprietary data.
* ⭐ **Strength**: Instant adaptation, no weight updates, preserves parametric knowledge.
* 📊 **Eval**: Use RAG-specific evals to check hallucination & grounding.

---

## 📚 Domain Models

* **Method**: Continual pre-training + post-training on large domain corpora.
* 📏 **Scale**: Hundreds of millions → billions of tokens.
* 🚀 **Gain**: Bridges domain gap, boosts style/format & safety.
* 🧠 **Tip**: Careful data selection prevents forgetting.

---

## 🤖 Tight Instruction Following

* **Method**: Post-training pipeline (SFT → DPO / RLHF).
* 🎯 **Use Case**: Multi-policy compliance (20+ rules), strict formats (JSON), advanced tracks (SQL, function calling, reasoning).
* ⚖️ **Trade-off**: Reliable behavior change, but risks regressions if data/objectives are weak → gate with evals.

---

## ✅ Practical Decision Rubric

* **Prompting** → light tweaks, add validators when data scarce.
* **RAG** → freshness, traceability, proprietary knowledge dominates.
* **Continual pre-training** → large domain gap, need exposure to new tokens.
* **Post-training (SFT/DPO/RLHF)** → enforce strict multi-policy compliance or advanced capabilities.

---

⚡ **Interview Tip**: Frame post-training as a **last-mile alignment lever**, while Prompting, RAG, and continual pre-training are **lighter, targeted interventions**.

---
