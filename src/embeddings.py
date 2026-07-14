import hashlib
import os
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "GENSIM_DATA_DIR",
    str(Path(__file__).resolve().parents[1] / ".cache" / "gensim-data"),
)

from gensim import downloader as api


class Embeddings:
    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.word_vectors = None

        try:
            print("Loading word vectors...", flush=True)
            self.word_vectors = api.load("glove-wiki-gigaword-100")
            print("Loaded!", flush=True)
        except Exception as exc:
            print(f"Could not load GloVe vectors. Using fallback embeddings: {exc}", flush=True)

    def _fallback_vector(self, word):
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self.vector_size)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def get_vector(self, word):
        if self.word_vectors is not None and word in self.word_vectors:
            return self.word_vectors[word]
        return self._fallback_vector(word)

    def find_similar_words(self, word, topn=5):
        if self.word_vectors is not None and word in self.word_vectors:
            return self.word_vectors.most_similar(word, topn=topn)
        return []

    def similarity(self, word1, word2):
        v1 = self.get_vector(word1)
        v2 = self.get_vector(word2)

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))