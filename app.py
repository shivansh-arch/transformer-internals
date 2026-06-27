import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from src.pipeline import Pipeline


# ----------------------------
# Cache the pipeline
# ----------------------------
@st.cache_resource
def load_pipeline():
    return Pipeline()


pipeline = load_pipeline()


# ----------------------------
# App Title
# ----------------------------
st.title("Transformer Visualizer")
st.write("Visualize Tokenization, Embeddings, Positional Encoding, and Self-Attention.")


# ----------------------------
# Text Input
# ----------------------------
text = st.text_input("Enter a sentence:")


# ----------------------------
# Run Pipeline
# ----------------------------
if st.button("Run Pipeline"):

    if text.strip() == "":
        st.warning("Please enter a sentence.")
    else:

        try:
            results = pipeline.run(text)

            # ===================================
            # Tokenization
            # ===================================
            st.subheader("1. Tokenization")

            token_df = pd.DataFrame({
                "Token": results["token_pieces"]
            })

            st.table(token_df)

            # ===================================
            # Embeddings
            # ===================================
            st.subheader("2. Similar Words (GloVe)")

            embedding_table = []

            for word in results["words"]:
                similar = pipeline.embeddings.find_similar_words(word)

                embedding_table.append({
                    "Word": word,
                    "Similar Words": ", ".join(
                        [w for w, score in similar]
                    )
                })

            st.table(pd.DataFrame(embedding_table))

            # ===================================
            # Positional Encoding Heatmap
            # ===================================
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

            # ===================================
            # Attention Heatmap
            # ===================================
            st.subheader("4. Attention Weights")

            fig, ax = plt.subplots(figsize=(6, 6))

            im = ax.imshow(
                results["attention_weights"],
                cmap="Blues"
            )

            plt.colorbar(im)

            ax.set_xticks(range(len(results["words"])))
            ax.set_yticks(range(len(results["words"])))

            ax.set_xticklabels(results["words"], rotation=45)
            ax.set_yticklabels(results["words"])

            ax.set_xlabel("Keys")
            ax.set_ylabel("Queries")

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")