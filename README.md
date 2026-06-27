# 🔍 TextScope — Transformer Internals Visualizer

A learning project that shows you what happens inside a transformer when you give it a sentence. Built from scratch using numpy, tiktoken, and GloVe embeddings.

---

## 💡 What It Does

You type a sentence → TextScope runs it through the full transformer pipeline and visualizes every step:

1. **Tokenization** — how your sentence gets split into subword pieces
2. **Word Embeddings** — what words are semantically similar to each word in your sentence
3. **Positional Encoding** — how position information gets added to embeddings
4. **Self-Attention** — which words attend to which, shown as a heatmap

---

## 🧠 Concepts Covered

- Byte Pair Encoding (BPE) tokenization via `tiktoken`
- GloVe word vectors and cosine similarity
- Sinusoidal positional encoding (implemented from scratch)
- Scaled dot-product self-attention: `softmax(QKᵀ / √dₖ) · V`

---

## 🗂️ Project Structure

```
textscope/
├── src/
│   ├── tokenizer.py       # tiktoken wrapper
│   ├── embeddings.py      # GloVe loader + cosine similarity
│   ├── positional.py      # sinusoidal positional encoding
│   ├── attention.py       # self-attention from scratch
│   └── pipeline.py        # chains everything together
├── notebooks/
│   └── exploration.ipynb  # step-by-step explanation notebook
├── app.py                 # Streamlit UI
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/textscope.git
cd textscope
```

**2. Create and activate virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

> Note: First run downloads GloVe vectors (~128MB). This takes about 1 minute and is cached after that.

---

## 📦 Dependencies

- `tiktoken` — tokenization
- `gensim` — GloVe word vectors
- `numpy` — all math (attention, positional encoding)
- `matplotlib` — heatmap visualizations
- `streamlit` — UI
- `pandas` — tables
- `scikit-learn` — PCA (for future embedding visualization)

---

## 📚 What I Learned

- How transformers actually process text at each stage
- Why tokenization matters (token boundaries, efficiency, subwords)
- How cosine similarity measures semantic closeness in vector space
- Why sinusoidal positional encoding works (bounded values, generalizes to any length)
- The full QKV attention mechanism — built with only numpy

---

## 🔭 Planned Extensions

- [ ] Multi-head attention visualization
- [ ] UMAP/PCA plot of word embeddings
- [ ] Causal masking (decoder-style attention)
- [ ] Compare attention patterns across different sentences

---

## 👤 Author

**Shivansh Gupta**  
B.Tech CSE (AI/ML) — Lovely Professional University  
GitHub: )
