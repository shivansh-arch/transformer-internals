import re

import tiktoken as token


class Tokenizer:
    def __init__(self, encoding_name="cl100k_base"):
        try:
            self.enc = token.get_encoding(encoding_name)
        except Exception:
            self.enc = None
            self._fallback_vocab = {}
            self._fallback_inverse_vocab = {}

    def _fallback_tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]|\s+", text)

    def encode(self, text):
        if self.enc is not None:
            return self.enc.encode(text)

        token_ids = []
        for piece in self._fallback_tokenize(text):
            if piece not in self._fallback_vocab:
                token_id = len(self._fallback_vocab) + 1
                self._fallback_vocab[piece] = token_id
                self._fallback_inverse_vocab[token_id] = piece
            token_ids.append(self._fallback_vocab[piece])
        return token_ids

    def decode(self, token_list):
        if self.enc is not None:
            return self.enc.decode(token_list)

        return "".join(
            self._fallback_inverse_vocab.get(token_id, "")
            for token_id in token_list
        )

    def get_token_count(self, text):
        return len(self.encode(text))

    def tokenize(self, text):
        if self.enc is not None:
            token_ids = self.enc.encode(text)
            return [self.enc.decode([token_id]) for token_id in token_ids]

        return self._fallback_tokenize(text)