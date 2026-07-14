import tiktoken


class Tokenizer:
    def __init__(self, encoding_name="cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def tokenize(self, text):
        """
        Returns BPE token strings.
        """
        ids = self.encoding.encode(text)

        return [
            self.encoding.decode([token_id])
            for token_id in ids
        ]

    def encode(self, text):
        """
        Returns token ids.
        """
        return self.encoding.encode(text)

    def decode(self, token_ids):
        """
        Decode token ids back to text.
        """
        return self.encoding.decode(token_ids)

    def token_info(self, text):
        """
        Educational helper.
        Returns token ids together with decoded token pieces.
        """

        ids = self.encode(text)

        info = []

        for token_id in ids:

            info.append({
                "token": self.encoding.decode([token_id]),
                "token_id": token_id,
            })

        return info

    def vocab_size(self):
        """
        Vocabulary size.
        """
        return self.encoding.n_vocab

    def special_tokens(self):
        """
        Returns all special tokens.
        """
        return self.encoding.special_tokens_set
