import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from src.pipeline import Pipeline


# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Transformer Visualizer",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Transformer Visualizer")
st.markdown(
    """
Visualize every stage of a Transformer:

- 🔤 Tokenization
- 📖 Word Embeddings
- 📍 Positional Encoding
- 🎯 Self Attention
"""
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Attention Weights",
    ["Random", "Trained"]
)

attention_mode = (
    "trained"
    if mode == "Trained"
    else "random"
)


@st.cache_resource
def load_pipeline(attention_mode):
    return Pipeline(attention_mode=attention_mode)


pipeline = load_pipeline(attention_mode)


# ----------------------------
# Input
# ----------------------------
text = st.text_input(
    "Enter a sentence",
    value="The cat sat on the mat"
)

run = st.button("Run Visualization")

if run:

    try:

        results = pipeline.run(text)

        # ===================================================
        # TOKENIZATION
        # ===================================================

        st.header("1️⃣ Tokenization")

        left, right = st.columns(2)

        with left:

            st.subheader("BPE Tokens")

            token_df = pd.DataFrame({
                "Token": results["token_pieces"],
                "Token ID": results["token_ids"]
            })

            st.dataframe(
                token_df,
                use_container_width=True
            )

        with right:

            st.subheader("Word Tokens")

            word_df = pd.DataFrame({
                "Word": results["words"]
            })

            st.dataframe(
                word_df,
                use_container_width=True
            )

        st.info(results["note"])

        st.divider()

        # ===================================================
        # EMBEDDINGS
        # ===================================================

        st.header("2️⃣ Word Embeddings")

        norms = results["embedding_norms"]

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Embedding Dimension",
                results["embeddings"].shape[1]
            )

        with c2:

            st.metric(
                "Words",
                len(results["words"])
            )

        table = []

        for word in results["words"]:

            similar = results["similar_words"][word]

            table.append({
                "Word": word,
                "Vector Norm": round(
                    norms[results["words"].index(word)],
                    3
                ),
                "Similar Words": ", ".join(
                    [x[0] for x in similar]
                )
            })

        st.dataframe(
            pd.DataFrame(table),
            use_container_width=True
        )

        st.divider()

        # ===================================================
        # POSITIONAL ENCODING
        # ===================================================

        st.header("3️⃣ Positional Encoding")

        fig, ax = plt.subplots(figsize=(12, 4))

        im = ax.imshow(
            results["positional_encoding"],
            cmap="viridis",
            aspect="auto"
        )

        plt.colorbar(im)

        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Token Position")

        st.pyplot(fig)

        plt.close(fig)

        st.divider()

        # ===================================================
        # ATTENTION
        # ===================================================

        st.header("4️⃣ Self Attention")

        fig, ax = plt.subplots(figsize=(7, 7))

        im = ax.imshow(
            results["attention_weights"],
            cmap="Blues"
        )

        plt.colorbar(im)

        words = results["words"]

        ax.set_xticks(range(len(words)))
        ax.set_yticks(range(len(words)))

        ax.set_xticklabels(
            words,
            rotation=45,
            ha="right"
        )

        ax.set_yticklabels(words)

        ax.set_xlabel("Keys")
        ax.set_ylabel("Queries")

        st.pyplot(fig)

        plt.close(fig)

        st.subheader("Attention Matrix")

        attention_df = pd.DataFrame(
            results["attention_weights"],
            index=words,
            columns=words,
        )

        st.dataframe(
            attention_df.round(3),
            use_container_width=True
        )

        csv = attention_df.to_csv().encode("utf-8")

        st.download_button(
            "Download Attention Matrix",
            csv,
            "attention_matrix.csv",
            "text/csv"
        )

        st.divider()

        # ===================================================
        # PIPELINE
        # ===================================================

        st.header("5️⃣ Transformer Pipeline")

        st.markdown("""
