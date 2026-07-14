# 🧠 Transformer Visualizer

An interactive educational application that visualizes the core building blocks of the Transformer architecture used in modern Large Language Models (LLMs) like GPT, LLaMA, and BERT.

Instead of treating Transformers as a black box, this project lets you explore each stage of the pipeline—from tokenization to self-attention—through intuitive visualizations.

---

## ✨ Features

### 🔤 1. BPE Tokenization

* Visualizes Byte Pair Encoding (BPE) tokens
* Displays token IDs
* Compares BPE tokens with word-level tokens
* Explains why modern LLMs use subword tokenization

---

### 📖 2. Word Embeddings

* Uses **GloVe (100-dimensional)** embeddings
* Displays semantically similar words
* Shows embedding statistics
* Demonstrates how words are represented as vectors

---

### 📍 3. Positional Encoding

* Visualizes sinusoidal positional encodings
* Interactive heatmap
* Explains why Transformers need positional information

---

### 🎯 4. Self-Attention

* Implements Scaled Dot-Product Attention from scratch using NumPy
* Visualizes the attention matrix
* Displays Query, Key, and Value computations
* Interactive attention heatmap

---

### 🔍 5. Attention Explorer

Select any word and inspect how much attention it gives to every other word in the sentence.

This provides an intuitive understanding of contextual relationships learned by attention.

---

### 📊 6. Transformer Pipeline

The application visualizes the complete data flow:

```text
Input Sentence
      │
      ▼
BPE Tokenization
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
Contextualized Output
```

---

# 🏗️ Project Structure

```text
transformer-visualizer/
│
├── app.py                  # Streamlit application
│
├── src/
│   ├── tokenizer.py        # BPE tokenization
│   ├── embeddings.py       # GloVe embeddings
│   ├── positional.py       # Positional Encoding
│   ├── attention.py        # Self Attention
│   └── pipeline.py         # Pipeline orchestration
│
├── weights/
│
├── requirements.txt
│
└── README.md
```

---

# 🧮 Mathematics

### Query, Key and Value

[
Q=XW_Q
]

[
K=XW_K
]

[
V=XW_V
]

---

### Attention

[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
]

---

### Positional Encoding

[
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)
]

[
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
]

---

# 🚀 Getting Started

## Clone the repository

```bash
git clone https://github.com/your-username/transformer-visualizer.git

cd transformer-visualizer
```

---

## Create a virtual environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the application

```bash
streamlit run app.py
```

---

# 📸 Screenshots

## Tokenization

> Add a screenshot here

---

## Word Embeddings

> Add a screenshot here

---

## Positional Encoding

> Add a screenshot here

---

## Self-Attention Heatmap

> Add a screenshot here

---

## Attention Explorer

> Add a screenshot here

---

# 💻 Technologies Used

* Python
* NumPy
* Streamlit
* Matplotlib
* Pandas
* Gensim
* tiktoken

---

# 🎯 Learning Objectives

This project helps understand:

* Byte Pair Encoding (BPE)
* Word Embeddings
* Positional Encoding
* Query, Key and Value vectors
* Scaled Dot-Product Attention
* Attention Heatmaps
* Contextual Word Representations

---

# ⚠️ Limitations

This project is designed for **education and visualization**.

It is **not** a complete Transformer implementation and does not perform language generation or model training.

Instead, it focuses on explaining the fundamental concepts that power modern Transformer-based language models.

---

# 🔮 Future Improvements

* Multi-Head Attention visualization
* Rotary Positional Embeddings (RoPE)
* RMSNorm visualization
* SwiGLU visualization
* Decoder causal masking
* Layer-by-layer Transformer visualization
* Interactive attention animations

---

# 🤝 Contributing

Contributions are welcome.

If you'd like to improve the visualizations, optimize performance, or add new Transformer components, feel free to open an issue or submit a pull request.

---

# 📜 License

This project is released under the MIT License.

---

# ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

It helps others discover the project and supports future improvements.
