import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from src.pipeline import Pipeline


# ----------------------------
# Cache Pipeline
# ----------------------------
@st.cache_resource
def load_pipeline():
    return Pipeline()


pipeline = load_pipeline()


# ----------------------------
# Title
# ----------------------------
st.title("Transformer Visualizer")
st.write(
    "Visualize Tokenization, Embeddings, Positional Encoding, and Self-Attention."
)


# ----------------------------
# Input
# ----------------------------
text = st.text_input("Enter a sentence:")


# ----------------------------
# Run
# ----------------------------
if st.button("Run Pipeline"):

    if not text.strip():
        st.warning("Please enter a sentence.")

    else:

        try:

            results = pipeline.run(text)

            # =====================================================
            # 1. Tokenization
            # =====================================================
            st.subheader("1. Tokenization (BPE)")

            token_df = pd.DataFrame({
                "BPE Tokens": results["token_pieces"]
            })

            st.table(token_df)

            st.info(results["note"])

            # =====================================================
            # Word Information
            # =====================================================
            st.subheader("Word-Level Tokenization")

            st.write("**Input Words:**")
            st.write(results["words"])

            st.write("**Embedded Words (found in GloVe):**")
            st.write(results["embedded_words"])

            if results["skipped_words"]:
                st.warning(
                    "Skipped Words (not present in GloVe): "
                    + ", ".join(results["skipped_words"])
                )

            # =====================================================
            # 2. Embeddings
            # =====================================================
            st.subheader("2. Similar Words (GloVe)")

            embedding_table = []

            for word in results["embedded_words"]:

                similar = pipeline.embeddings.find_similar_words(word)

                embedding_table.append({
                    "Word": word,
                    "Similar Words": ", ".join(
                        [w for w, score in similar]
                    )
                })

            st.table(pd.DataFrame(embedding_table))

            # =====================================================
            # 3. Positional Encoding
            # =====================================================
            st.subheader("3. Positional Encoding")

            fig, ax = plt.subplots(figsize=(10, 4))

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

            # =====================================================
            # 4. Attention
            # =====================================================
            st.subheader("4. Self-Attention")

            fig, ax = plt.subplots(figsize=(6, 6))

            im = ax.imshow(
                results["attention_weights"],
                cmap="Blues"
            )

            plt.colorbar(im)

            words = results["embedded_words"]

            ax.set_xticks(range(len(words)))
            ax.set_yticks(range(len(words)))

            ax.set_xticklabels(words, rotation=45)
            ax.set_yticklabels(words)

            ax.set_xlabel("Keys")
            ax.set_ylabel("Queries")

            st.pyplot(fig)

            plt.close(fig)

        except Exception as e:
            st.error(f"Error: {e}")
