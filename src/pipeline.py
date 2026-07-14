import numpy as np

from src.tokenizer import Tokenizer
from src.embeddings import Embeddings
from src.positional import PositionalEncoding
from src.attention import Attention


class Pipeline:
    def __init__(
        self,
        d_model=100,
        d_k=64,
        d_v=64,
        attention_mode="random",
    ):
        self.tokenizer = Tokenizer()
        self.embeddings = Embeddings()

        self.positional_encoding = PositionalEncoding(d_model)

        weights_path = (
            "weights/trained_attention.npz"
            if attention_mode == "trained"
            else None
        )

        self.attention = Attention(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            weights_path=weights_path,
        )

    def run(self, text):

        # -----------------------------
        # Stage 1: Tokenization (BPE)
        # -----------------------------
        token_pieces = self.tokenizer.tokenize(text)

        token_ids = self.tokenizer.encode(text)

        # -----------------------------
        # Stage 2: Word Tokenization
        # -----------------------------
        words = text.lower().split()

        embedding_vectors = []

        for word in words:
            embedding_vectors.append(
                self.embeddings.get_vector(word)
            )

        embeddings = np.array(embedding_vectors)

        # -----------------------------
        # Stage 3: Positional Encoding
        # -----------------------------
        seq_len = embeddings.shape[0]

        positional_encoding = self.positional_encoding.encode(seq_len)

        transformer_input = embeddings + positional_encoding

        # -----------------------------
        # Stage 4: Attention
        # -----------------------------
        attention_output, attention_weights = self.attention.attend(
            transformer_input
        )

        # -----------------------------
        # Stage 5: Similar Words
        # -----------------------------
        similar_words = {}

        for word in words:
            similar_words[word] = self.embeddings.find_similar_words(word)

        # -----------------------------
        # Stage 6: Statistics
        # -----------------------------
        embedding_norms = np.linalg.norm(
            embeddings,
            axis=1,
        )

        return {
            # Tokenization
            "token_pieces": token_pieces,
            "token_ids": token_ids,
            "words": words,

            # Embeddings
            "embeddings": embeddings,
            "embedding_norms": embedding_norms,
            "similar_words": similar_words,

            # Positional Encoding
            "positional_encoding": positional_encoding,
            "transformer_input": transformer_input,

            # Attention
            "attention_output": attention_output,
            "attention_weights": attention_weights,

            # Explanation
            "note": (
                "BPE tokenization is used for visualization, while "
                "GloVe provides word-level embeddings. Modern LLMs "
                "learn embeddings directly for BPE/subword tokens, "
                "whereas this educational project intentionally "
                "combines both to demonstrate the complete pipeline."
            ),
        }
