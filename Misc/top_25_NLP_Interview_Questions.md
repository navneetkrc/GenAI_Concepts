# 🔝 Top 25 NLP Interview Questions

---

## 🔹 Core Concepts

### ✅ What is Natural Language Processing (NLP)?

NLP is a field of AI that enables machines to understand, interpret, generate, and interact using human language.
It combines **linguistics** and **machine learning** to process text or speech data.

---

### ✅ What is the difference between NLP and NLU (Natural Language Understanding)?

| **Aspect**  | **NLP**                              | **NLU**                              |
| ----------- | ------------------------------------ | ------------------------------------ |
| **Goal**    | General text processing              | Understanding meaning & intent       |
| **Scope**   | Includes NLU, NLG, and preprocessing | Subset of NLP                        |
| **Example** | Tokenization, POS tagging            | Intent detection, sentiment analysis |

**🔁 Note:** NLU focuses on meaning and semantics within NLP.

---

### ✅ What is tokenization and why is it important?

Tokenization splits text into smaller units called **tokens** (words, subwords, characters).
It's the **first step** in most NLP pipelines, making raw text processable for models.

---

### ✅ What are stop words, and should we always remove them?

Stop words are common words like *"the"*, *"is"*, *"in"* that often carry little semantic value.
**Should we remove them?**

✔ Often removed to reduce noise and improve efficiency.
⚠ But retained for tasks like translation or question answering where every word matters.

---

### ✅ Explain stemming vs lemmatization with examples.

| **Feature** | **Stemming**          | **Lemmatization**      |
| ----------- | --------------------- | ---------------------- |
| **Method**  | Rule-based chopping   | Dictionary + POS-aware |
| **Output**  | Not always real words | Real base words        |
| **Example** | `"running"` → `runn`  | `"running"` → `run`    |

🧠 *Stemming is faster but less accurate; lemmatization gives better linguistic correctness.*

---

## 🔹 Embeddings & Representations

### ✅ What are word embeddings? How do they work?

**Short Answer:**
Word embeddings are dense vector representations capturing word meanings via context and relationships.

**Detailed Explanation:**
They map words to fixed-size vectors where similar meanings have similar vectors.
Learned from large corpora by models like **Word2Vec** or **GloVe**, they capture semantic relationships beyond simple one-hot encoding.

---

### ✅ How do Word2Vec, GloVe, and FastText differ?

| **Model**    | **Key Idea**                                                               |
| ------------ | -------------------------------------------------------------------------- |
| **Word2Vec** | Predicts words using local context (Skip-gram/CBOW)                        |
| **GloVe**    | Builds word vectors from global co-occurrence statistics                   |
| **FastText** | Uses subword n-grams to represent words, handling rare/unseen words better |

**Infographic Suggestion:**
Illustrate three pipelines showing Word2Vec (local window), GloVe (corpus-wide matrix), and FastText (subword composition).

---

### ✅ What is the Out-of-Vocabulary (OOV) problem and how can it be handled?

**OOV:** When a word wasn't seen during training, making it impossible to generate its embedding.

**Solutions:**
✔ FastText's subword n-grams compose unseen words.
✔ Contextual models (e.g., **BERT**) generate dynamic embeddings based on sentence context.
✔ `<UNK>` token for unknown words in simpler models.

---

### ✅ What are sentence and document embeddings? How do they differ from word embeddings?

| **Level**             | **Representation**                                   |
| --------------------- | ---------------------------------------------------- |
| **Word**              | Individual word vectors                              |
| **Sentence/Document** | Vector summarizing full sentence or document meaning |

**Methods:**
✔ Averaging word embeddings (basic)
✔ Pretrained models like **Sentence-BERT**, **Universal Sentence Encoder**

**Use Cases:** Classification, Semantic Search, Clustering.

---

### ✅ Explain static vs contextual word embeddings.

| **Aspect**     | **Static**                | **Contextual**                          |
| -------------- | ------------------------- | --------------------------------------- |
| **Definition** | One fixed vector per word | Vectors vary based on sentence context  |
| **Examples**   | Word2Vec, GloVe           | BERT, RoBERTa                           |
| **Limitation** | Cannot resolve ambiguity  | Handles polysemy, captures true meaning |

---

## 🔹 Attention & Transformers

### ✅ What is the attention mechanism? How does it work?

**Short Answer:**
Attention allows models to focus on relevant input parts during predictions.

**Detailed Steps:**

1. Convert tokens to **Queries (Q)**, **Keys (K)**, **Values (V)**

2. Compute attention weights:

   $$
   \text{Attention} = \text{softmax} \left( \frac{Q \cdot K^T}{\sqrt{d_k}} \right)
   $$

3. Output: Weighted sum of **V** emphasizing important tokens.

**Infographic Suggestion:** Visual showing token interactions with varying attention weights.

---

### ✅ Self-attention vs Cross-attention

| **Type**            | **Description**                                         | **Use Case**                          |
| ------------------- | ------------------------------------------------------- | ------------------------------------- |
| **Self-attention**  | Tokens attend to others in the same sequence            | Captures intra-sequence relationships |
| **Cross-attention** | One sequence attends to another (e.g., encoder-decoder) | Translation, Summarization            |

---

### ✅ What are Transformers and why are they important in NLP?

**Short Answer:**
Transformers use self-attention to model long-range dependencies efficiently.

**Key Features:**
✔ Parallelizable
✔ Capture complex dependencies
✔ Backbone of models like **BERT**, **GPT**, **T5**

**Infographic Suggestion:** Block diagram showing encoder-decoder structure with attention layers.

---

### ✅ Compare RNN, LSTM, GRU, Transformer models

| **Model**       | **Pros**                        | **Cons**                          |
| --------------- | ------------------------------- | --------------------------------- |
| **RNN**         | Simple, sequential processing   | Struggles with long dependencies  |
| **LSTM**        | Handles long-range dependencies | Slow, step-by-step                |
| **GRU**         | Faster than LSTM                | Still sequential, less expressive |
| **Transformer** | Parallel, long-range context    | Heavy for very long sequences     |

---

### ✅ Local vs Global Attention

| **Type**   | **Scope**                | **Use Case**                      |
| ---------- | ------------------------ | --------------------------------- |
| **Local**  | Focuses on nearby tokens | Long documents (e.g., Longformer) |
| **Global** | Attends to all tokens    | Full-sequence understanding       |

**Tradeoff:**
Local attention improves efficiency; global captures complete relationships.

---

## 🔹 Language Models

### ✅ What is a language model? How is it trained?

Predicts sequence likelihood:

$$
P(w_1, w_2, ..., w_n)
$$

**Training:**
✔ Next-word prediction (Causal LM, e.g., GPT)
✔ Masked word prediction (MLM, e.g., BERT)

Forms the foundation for generation, search, translation, etc.

---

### ✅ MLM vs CLM

| **Aspect**   | **MLM (Masked)**     | **CLM (Causal)**   |
| ------------ | -------------------- | ------------------ |
| **Goal**     | Predict masked words | Predict next word  |
| **Context**  | Uses left & right    | Only left-to-right |
| **Examples** | BERT, RoBERTa        | GPT, GPT-3         |

**Example:**
MLM: `"The cat sat on the [MASK]"` → Predict `"mat"`
CLM: `"The cat sat on the"` → Predict `"mat"`

---

### ✅ How does BERT work? Why bidirectional?

BERT uses:
✔ **Bidirectional self-attention**
✔ **Masked Language Modeling (MLM)**
✔ **Next Sentence Prediction (NSP)**

Processes entire sentences, capturing rich context for classification, QA, etc.

---

### ✅ GPT vs BERT

| **Feature**      | **GPT**               | **BERT**               |
| ---------------- | --------------------- | ---------------------- |
| **Training**     | Causal LM (next word) | Masked LM + NSP        |
| **Architecture** | Decoder stack         | Encoder stack          |
| **Focus**        | Text generation       | Language understanding |
| **Context**      | Left-to-right         | Bidirectional          |

---

### ✅ What is fine-tuning?

Adapting pretrained models to downstream tasks:

**Steps:**
✔ Add task-specific layers (e.g., classifier, QA head)
✔ Train on labeled task data
✔ Benefits: Few labeled examples required, leverages pretrained knowledge

---

## 🔹 Evaluation & Applications

### ✅ What is BLEU score?

Measures machine translation quality via n-gram overlap:

$$
\text{BLEU} = \text{Brevity Penalty} \times \text{Geometric Mean of n-gram Precisions}
$$

**Range:** 0 to 1 (higher = better)
⚠ Limitation: Focuses on surface overlap, may miss true semantic quality.

---

### ✅ What is perplexity?

Quantifies model's uncertainty:

$$
\text{Perplexity} = 2^{-\text{average log probability of tokens}}
$$

Lower perplexity = better predictions (used in evaluating generative models).

---

### ✅ NER and POS Tagging

| **Task**        | **Function**                        | **Example**                      |
| --------------- | ----------------------------------- | -------------------------------- |
| **NER**         | Detects entities (names, locations) | `"Barack Obama"` → PERSON        |
| **POS Tagging** | Labels grammatical roles            | `"fox"` → NOUN, `"jumps"` → VERB |

Essential for syntactic and semantic understanding.

---

### ✅ How does semantic search work using embeddings?

**Process:**
✔ Convert queries/documents to embeddings (e.g., Sentence-BERT)
✔ Compute similarity (cosine similarity)
✔ Retrieve conceptually similar content, beyond keyword matches

**Example:**
Query: `"Symptoms of flu"`
Retrieves: `"How to treat viral fever"`

---

### ✅ End-to-End Text Classification Pipeline

**Steps:**

1. Data Collection
2. Preprocessing: Cleaning, tokenization
3. Embedding: Word2Vec, Sentence-BERT, or Transformers
4. Model Training: Classifier or fine-tuned language model
5. Evaluation: Metrics like Accuracy, F1
6. Deployment: API exposure
7. Monitoring & Retraining

---

## 🎯 **Infographics Recommendations**

* **Embedding Models Comparison:** Word2Vec vs GloVe vs FastText visual

 ![image](https://github.com/user-attachments/assets/f6f63692-a7e9-4cbb-b496-a0729e4563eb)

* **Attention Mechanism:** Query-Key-Value flow diagram
https://medium.com/data-science/what-are-query-key-and-value-in-the-transformer-architecture-and-why-are-they-used-acbe73f731f2

  ![image](https://github.com/user-attachments/assets/d8ba1980-3e79-424e-95aa-e4067c31fcae)

* **Transformer Block Diagram:** Encoder, Decoder, Attention layers
* **Local vs Global Attention:** Token connectivity illustration
* **Semantic Search Workflow:** Query-to-Embedding to Result flow

---

Would you like me to generate ready-to-use visuals for any of these? 🎨
