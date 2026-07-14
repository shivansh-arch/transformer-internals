# 🧠 Transformer Internals (TextScope)

> **An interactive visualization tool that explains how Transformer models process text—from tokenization to self-attention.**

Transformer Internals (TextScope) is an educational project built with **Python** and **Streamlit** that helps users understand what happens inside modern Transformer-based language models.

Instead of treating models like GPT or BERT as black boxes, this project visualizes every major stage of the Transformer pipeline, allowing users to explore tokenization, embeddings, positional encoding, and attention mechanisms interactively.

---

# 🚀 Features

* 🔤 Byte Pair Encoding (BPE) Tokenization
* 🆔 Token IDs Visualization
* 📖 GloVe Word Embeddings
* 🔍 Similar Word Search
* 📍 Positional Encoding Heatmaps
* 🎯 Self-Attention Visualization
* 📊 Interactive Heatmaps
* ⚡ Streamlit-based User Interface
* 🎓 Designed for Learning Transformer Internals

---

# 🏗️ Project Pipeline

```text
Input Sentence
      │
      ▼
Tokenization (BPE)
      │
      ▼
Token IDs
      │
      ▼
Word Embeddings
      │
      ▼
Positional Encoding
      │
      ▼
Query • Key • Value
      │
      ▼
Scaled Dot Product Attention
      │
      ▼
Attention Weights
      │
      ▼
Visualization
```

---

# 📂 Project Structure

```text
transformer-internals/
│
├── app.py
├── requirements.txt
│
├── modules/
│   ├── tokenizer.py
│   ├── embeddings.py
│   ├── positional_encoding.py
│   ├── attention.py
│   └── utils.py
│
├── assets/
│
├── images/
│
└── README.md
```

*(Update the structure if your folder names differ.)*

---

# 🧠 What You'll Learn

This project demonstrates the core concepts behind Transformer models:

## 1️⃣ Tokenization

* Byte Pair Encoding (BPE)
* Token IDs
* Vocabulary mapping

---

## 2️⃣ Word Embeddings

Visualize how words are converted into dense vectors.

Learn:

* semantic similarity
* embedding dimensions
* nearest words

---

## 3️⃣ Positional Encoding

Understand how Transformers preserve word order.

Visualize

* sine waves
* cosine waves
* positional embedding matrix

---

## 4️⃣ Self Attention

Explore how each word attends to every other word.

See:

* Query matrix
* Key matrix
* Value matrix
* Attention Scores
* Softmax
* Attention Heatmap

---

# 📐 Attention Formula

The project demonstrates the fundamental Transformer equation:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √dk)V
```

Where:

* **Query (Q)** → What information is this token searching for?
* **Key (K)** → What information does this token contain?
* **Value (V)** → What information should this token pass to others?

---

# 🛠️ Technologies Used

* Python
* Streamlit
* NumPy
* Matplotlib
* tiktoken
* Gensim
* GloVe Embeddings

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/shivansh-arch/transformer-internals.git
cd transformer-internals
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📸 Screenshots

Add screenshots here for each stage.

### Tokenization

```
images/tokenization.png
```

---

### Embeddings

```
images/embeddings.png
```

---

### Positional Encoding

```
images/position.png
```

---

### Attention Heatmap

```
images/attention.png
```

---

# 🎥 Demo

Add a short GIF demonstrating:

* entering text
* tokenization
* embeddings
* positional encoding
* attention visualization

A 15–20 second GIF greatly improves the repository.

---

# 🎯 Why This Project?

Most tutorials explain Transformers mathematically but provide little intuition.

TextScope bridges that gap by letting users see how every stage transforms the input text, making complex concepts easier to understand.

---

# 📈 Future Improvements

* Multi-Head Attention Visualization
* Rotary Positional Embeddings (RoPE)
* Flash Attention
* Decoder Block Visualization
* Feed Forward Network Visualization
* Residual Connection Animation
* Layer Normalization Visualization
* Interactive Q, K, V Editing
* MiniGPT Integration
* Support for Multiple Transformer Architectures

---

# 🎓 Ideal For

* Students learning NLP
* Deep Learning beginners
* Machine Learning enthusiasts
* AI educators
* Interview preparation
* Understanding LLM architecture

---

# ⚠️ Limitations

* Educational visualization only
* Uses static GloVe embeddings
* Demonstrates a simplified Transformer pipeline
* Not intended for production inference

---

# 👨‍💻 Author

**Shivansh Gupta**

Computer Science Student | Machine Learning & AI Enthusiast

GitHub: https://github.com/shivansh-arch

---

# ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub. It helps others discover the project and motivates future improvements.
