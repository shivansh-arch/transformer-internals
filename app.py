import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.pipeline import Pipeline


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Transformer Visualizer",
    page_icon="🧠",
    layout="wide"
)


# =====================================================
# Sidebar
# =====================================================

st.sidebar.title("⚙️ Settings")

mode = st.sidebar.radio(
    "Attention Weights",
    [
        "Random (Default)",
        "Trained (if available)"
    ]
)

attention_mode = (
    "trained"
    if mode.startswith("Trained")
    else "random"
)


@st.cache_resource
def load_pipeline(attention_mode):
    return Pipeline(attention_mode=attention_mode)


# =====================================================
# Title
# =====================================================

st.title("🧠 Transformer Visualizer")

st.markdown("""
This application visualizes the complete Transformer pipeline.

The goal is educational—you can inspect every stage from
tokenization to self-attention.
""")

st.divider()


# =====================================================
# User Input
# =====================================================

text = st.text_input(
    "Enter a sentence",
    value="The cat sat on the mat."
)

run = st.button(
    "🚀 Run Visualization",
    use_container_width=True
)


# =====================================================
# Run Pipeline
# =====================================================

if run:

    if not text.strip():

        st.warning("Please enter a sentence.")

    else:

        try:

            with st.spinner("Loading Transformer pipeline (and GloVe embeddings)..."):
                pipeline = load_pipeline(attention_mode)
                results = pipeline.run(text)

            # =====================================================
            # Summary Metrics
            # =====================================================

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "BPE Tokens",
                    len(results["token_pieces"])
                )

            with c2:
                st.metric(
                    "Word Tokens",
                    len(results["words"])
                )

            with c3:
                st.metric(
                    "Embedding Size",
                    results["embeddings"].shape[1]
                )

            st.divider()

            # =====================================================
            # Tokenization
            # =====================================================

            st.header("1️⃣ Tokenization")

            left, right = st.columns(2)

            with left:

                st.subheader("BPE Tokens")

                token_df = pd.DataFrame({
                    "Position": list(range(len(results["token_pieces"]))),
                    "Token": results["token_pieces"],
                    "Token ID": results["token_ids"]
                })

                st.dataframe(
                    token_df,
                    use_container_width=True
                )

            with right:

                st.subheader("Word-Level Tokens")

                word_df = pd.DataFrame({
                    "Position": list(range(len(results["words"]))),
                    "Word": results["words"]
                })

                st.dataframe(
                    word_df,
                    use_container_width=True
                )

            st.info(results["note"])

            with st.expander("Why are these different?"):

                st.markdown("""
**BPE Tokenization**

Large Language Models tokenize text into **sub-word units**.

Example:

```
playing

↓

play + ing
```

This reduces vocabulary size and handles unknown words better.

---

**GloVe Embeddings**

GloVe stores **one vector per word**.

Because of that, this educational project uses

- BPE for visualization
- GloVe for embeddings

Modern LLMs instead learn embeddings directly for BPE tokens.
""")

            st.divider()

            # =====================================================
            # Embeddings
            # =====================================================

            st.header("2️⃣ Word Embeddings")

            norm_table = []

            for i, word in enumerate(results["words"]):

                similar = results["similar_words"][word]

                norm_table.append({

                    "Word": word,

                    "Vector Norm":
                        round(results["embedding_norms"][i], 3),

                    "Top Similar Words":
                        ", ".join(
                            [x[0] for x in similar]
                        )

                })

            st.dataframe(
                pd.DataFrame(norm_table),
                use_container_width=True
            )
            st.divider()

            # =====================================================
            # Positional Encoding
            # =====================================================

            st.header("3️⃣ Positional Encoding")

            st.markdown("""
Each embedding is combined with a positional encoding so the model
knows the order of words.
""")

            fig, ax = plt.subplots(figsize=(12, 4))

            im = ax.imshow(
                results["positional_encoding"],
                aspect="auto",
                cmap="viridis"
            )

            plt.colorbar(im)

            ax.set_xlabel("Embedding Dimension")
            ax.set_ylabel("Token Position")

            st.pyplot(fig)

            plt.close(fig)

            with st.expander("Mathematics of Positional Encoding"):

                st.latex(
                    r"PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)"
                )

                st.latex(
                    r"PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)"
                )

            st.divider()

            # =====================================================
            # Self Attention
            # =====================================================

            st.header("4️⃣ Self Attention")

            st.markdown("""
Each word generates three vectors:

- Query (Q)
- Key (K)
- Value (V)

Attention decides how much every word should focus on every other word.
""")

            with st.expander("Attention Equations"):

                st.latex(r"Q=XW_Q")
                st.latex(r"K=XW_K")
                st.latex(r"V=XW_V")

                st.latex(
                    r"Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V"
                )

            fig, ax = plt.subplots(figsize=(7, 7))

            im = ax.imshow(
                results["attention_weights"],
                cmap="Blues"
            )

            plt.colorbar(im)

            words = results["words"]
            unique_words = [f"{i}: {w}" for i, w in enumerate(words)]

            ax.set_xticks(range(len(words)))
            ax.set_yticks(range(len(words)))

            ax.set_xticklabels(
                unique_words,
                rotation=45,
                ha="right"
            )

            ax.set_yticklabels(unique_words)

            ax.set_xlabel("Keys")

            ax.set_ylabel("Queries")

            st.pyplot(fig)

            plt.close(fig)

            st.subheader("Attention Matrix")

            attention_df = pd.DataFrame(
                results["attention_weights"],
                index=unique_words,
                columns=unique_words
            )

            st.dataframe(
                attention_df.round(3),
                use_container_width=True
            )

            csv = attention_df.to_csv().encode("utf-8")

            st.download_button(
                label="📥 Download Attention Matrix",
                data=csv,
                file_name="attention_matrix.csv",
                mime="text/csv"
            )

            st.divider()

            # =====================================================
            # Attention Explorer
            # =====================================================

            st.header("5️⃣ Attention Explorer")

            selected = st.selectbox(
                "Choose a query token",
                unique_words
            )

            idx = unique_words.index(selected)

            attention_scores = results["attention_weights"][idx]

            explorer = pd.DataFrame({
                "Key Word": unique_words,
                "Attention Score": np.round(attention_scores, 4)
            })

            explorer = explorer.sort_values(
                by="Attention Score",
                ascending=False
            )

            st.dataframe(
                explorer,
                use_container_width=True
            )

            st.bar_chart(
                explorer.set_index("Key Word")
            )

            st.divider()
                        # =====================================================
            # Transformer Pipeline
            # =====================================================

            st.header("6️⃣ Transformer Pipeline")

            st.markdown("""
```text
                Input Sentence
                      │
                      ▼
             BPE Tokenization
                      │
                      ▼
          Word Embeddings (GloVe)
                      │
                      ▼
          Positional Encoding Added
                      │
                      ▼
        Query / Key / Value Projection
                      │
                      ▼
      Scaled Dot Product Attention
                      │
                      ▼
            Attention Output
```
""")

            st.divider()

            # =====================================================
            # Intermediate Shapes
            # =====================================================

            st.header("📊 Intermediate Tensor Shapes")

            shapes = pd.DataFrame(
                {
                    "Component": [
                        "BPE Tokens",
                        "Word Tokens",
                        "Embeddings",
                        "Positional Encoding",
                        "Transformer Input",
                        "Attention Output",
                        "Attention Matrix",
                    ],
                    "Shape": [
                        f"{len(results['token_pieces'])}",
                        f"{len(results['words'])}",
                        str(results["embeddings"].shape),
                        str(results["positional_encoding"].shape),
                        str(results["transformer_input"].shape),
                        str(results["attention_output"].shape),
                        str(results["attention_weights"].shape),
                    ],
                }
            )

            st.dataframe(
                shapes,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            # =====================================================
            # Educational Notes
            # =====================================================

            st.header("📖 How This Visualizer Works")

            with st.expander("Step 1 — Tokenization", expanded=False):
                st.write("""
Large Language Models first convert text into **tokens**.

Example:

```
playing

↓

play + ing
```

These sub-word units are called **BPE tokens**.
""")

            with st.expander("Step 2 — Word Embeddings"):
                st.write("""
Each word is converted into a dense vector.

This project uses **GloVe 100-dimensional vectors**.

Words with similar meanings have nearby vectors.
""")

            with st.expander("Step 3 — Positional Encoding"):
                st.write("""
Attention itself has no notion of word order.

Positional encodings inject sequence information so the model knows
which word comes first, second, and so on.
""")

            with st.expander("Step 4 — Self Attention"):
                st.write("""
Every word creates:

• Query

• Key

• Value

Attention scores determine which words should influence each other.
""")

            with st.expander("Step 5 — Output"):
                st.write("""
The weighted value vectors become contextualized representations.

These representations are then passed to later transformer blocks in
modern LLMs.
""")

            st.divider()

            # =====================================================
            # About
            # =====================================================

            st.header("ℹ️ About This Project")

            st.info(
                """
This project is an educational visualization of the Transformer
architecture.

Included Components

• BPE Tokenization

• GloVe Embeddings

• Sinusoidal Positional Encoding

• Scaled Dot-Product Attention

Purpose

To help students understand how modern Transformer models process
language internally.

This is **not** a complete Large Language Model.
It focuses on visualizing the core building blocks.
"""
            )

            st.success("✅ Visualization completed successfully.")

        except Exception as e:

            st.exception(e)
