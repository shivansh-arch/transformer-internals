import numpy as np


class Attention:
    def __init__(self, d_model, d_k, d_v):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        # Initialize weight matrices
        self.W_Q = np.random.randn(d_model, d_k) * 0.1
        self.W_K = np.random.randn(d_model, d_k) * 0.1
        self.W_V = np.random.randn(d_model, d_v) * 0.1

    def softmax(self, x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def attend(self, X):
        """
        X shape: (seq_len, d_model)
        """

        # Create Query, Key, and Value matrices
        Q = np.dot(X, self.W_Q)  # (seq_len, d_k)
        K = np.dot(X, self.W_K)  # (seq_len, d_k)
        V = np.dot(X, self.W_V)  # (seq_len, d_v)

        # Compute attention scores
        scores = np.matmul(Q, K.T) / np.sqrt(self.d_k)  # (seq_len, seq_len)

        # Convert scores to probabilities
        attention_weights = self.softmax(scores)

        # Compute weighted sum of values
        output = np.matmul(attention_weights, V)  # (seq_len, d_v)

        return output, attention_weights