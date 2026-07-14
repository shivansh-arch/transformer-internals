import os
import numpy as np


class Attention:
    def __init__(self, d_model, d_k, d_v, weights_path=None):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        if weights_path is not None and os.path.exists(weights_path):
            data = np.load(weights_path)

            self.W_Q = data["W_Q"]
            self.W_K = data["W_K"]
            self.W_V = data["W_V"]

            print(f"Loaded attention weights from {weights_path}")

        else:
            self.W_Q = np.random.randn(d_model, d_k) * 0.1
            self.W_K = np.random.randn(d_model, d_k) * 0.1
            self.W_V = np.random.randn(d_model, d_v) * 0.1

            if weights_path is not None:
                print(
                    f"Warning: '{weights_path}' not found. "
                    "Using randomly initialized attention weights."
                )

    def softmax(self, x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def attend(self, X):
        """
        Parameters
        ----------
        X : ndarray
            Shape: (seq_len, d_model)

        Returns
        -------
        output : ndarray
            Shape: (seq_len, d_v)

        attention_weights : ndarray
            Shape: (seq_len, seq_len)
        """

        # Query, Key, Value projections
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        # Scaled Dot-Product Attention
        scores = (Q @ K.T) / np.sqrt(self.d_k)

        attention_weights = self.softmax(scores)

        output = attention_weights @ V

        return output, attention_weights

    def save_weights(self, path):
        """
        Save learned attention weights.
        """

        os.makedirs(os.path.dirname(path), exist_ok=True)

        np.savez(
            path,
            W_Q=self.W_Q,
            W_K=self.W_K,
            W_V=self.W_V,
        )

        print(f"Attention weights saved to {path}")

    def load_weights(self, path):
        """
        Load attention weights after initialization.
        """

        data = np.load(path)

        self.W_Q = data["W_Q"]
        self.W_K = data["W_K"]
        self.W_V = data["W_V"]

        print(f"Attention weights loaded from {path}")
