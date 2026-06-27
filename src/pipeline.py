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
        # Tokenize (for visualization)
        token_pieces = self.tokenizer.tokenize(text)

        # GloVe works on words
        words = text.lower().split()

        embeddings = np.array([
            self.embeddings.get_vector(word)
            for word in words
            if self.embeddings.get_vector(word) is not None
        ])

        if embeddings.size == 0:
            raise ValueError("No valid embeddings found.")

        # Positional Encoding
        seq_len = embeddings.shape[0]
        positional_encoding = self.positional_encoding.encode(seq_len)

        X = embeddings + positional_encoding

        # Attention
        attention_output, attention_weights = self.attention.attend(X)

        return {
        "token_pieces": token_pieces,
        "words": words,
        "embeddings": embeddings,
        "positional_encoding": positional_encoding,
        "attention_output": attention_output,
        "attention_weights": attention_weights
}