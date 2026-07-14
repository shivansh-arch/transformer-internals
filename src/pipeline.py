import numpy as np

from src.tokenizer import Tokenizer
from src.embeddings import Embeddings
from src.positional import PositionalEncoding
from src.attention import Attention


class Pipeline:
    def __init__(self, d_model=100, d_k=64, d_v=64):
        self.tokenizer = Tokenizer()
        self.embeddings = Embeddings()

        self.positional_encoding = PositionalEncoding(d_model)
        self.attention = Attention(d_model, d_k, d_v)

    def run(self, text):
        # -------------------------------
        # 1. BPE tokenization (visualization)
        # -------------------------------
        token_pieces = self.tokenizer.tokenize(text)

        # -------------------------------
        # 2. Word tokenization (GloVe)
        # -------------------------------
        words = text.lower().split()

        embeddings = []

        for word in words:
            vector = self.embeddings.get_vector(word)
            if vector is not None:
                embeddings.append(vector)

        embeddings = np.array(embeddings)

        if embeddings.size == 0:
            raise ValueError("No valid embeddings found.")

        # -------------------------------
        # 3. Positional Encoding
        # -------------------------------
        seq_len = embeddings.shape[0]
        positional_encoding = self.positional_encoding.encode(seq_len)

        X = embeddings + positional_encoding

        # -------------------------------
        # 4. Self Attention
        # -------------------------------
        attention_output, attention_weights = self.attention.attend(X)

        return {
            "token_pieces": token_pieces,
            "words": words,
            "embeddings": embeddings,
            "positional_encoding": positional_encoding,
            "attention_output": attention_output,
            "attention_weights": attention_weights,
            "note": (
                "Displayed tokens use BPE tokenization (similar to GPT models), "
                "while the embedding pipeline uses whole-word GloVe vectors. "
                "This mismatch exists because GloVe provides embeddings only for "
                "complete words, whereas modern transformers learn embeddings "
                "directly for subword tokens."
            ),
        }
