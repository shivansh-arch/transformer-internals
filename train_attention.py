import os
import numpy as np

from src.attention import Attention


# -----------------------------
# Configuration
# -----------------------------
D_MODEL = 100
D_K = 64
D_V = 64

LR = 0.01
EPOCHS = 500

SAVE_PATH = "weights/trained_attention.npz"

np.random.seed(42)


# -----------------------------
# Create Attention
# -----------------------------
attention = Attention(D_MODEL, D_K, D_V)


# -----------------------------
# Fake Training Dataset
# -----------------------------
# 100 random sequences
# shape:
# (batch, seq_len, d_model)
dataset = np.random.randn(100, 6, D_MODEL)


def mse(pred, target):
    return np.mean((pred - target) ** 2)


print("Training...")

for epoch in range(EPOCHS):

    total_loss = 0

    for X in dataset:

        # -----------------------
        # Forward
        # -----------------------
        output, weights = attention.attend(X)

        # Toy objective:
        # reconstruct original embeddings
        target = X[:, :D_V]

        loss = mse(output, target)

        total_loss += loss

        # -------------------------------------------------
        # VERY SIMPLE UPDATE
        #
        # This is NOT the final attention backprop.
        # It simply nudges weights toward lower loss.
        # We'll replace this later with real gradients.
        # -------------------------------------------------

        noise_scale = LR * loss

        attention.W_Q -= noise_scale * np.random.randn(*attention.W_Q.shape)
        attention.W_K -= noise_scale * np.random.randn(*attention.W_K.shape)
        attention.W_V -= noise_scale * np.random.randn(*attention.W_V.shape)

    if epoch % 25 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Loss = {total_loss / len(dataset):.6f}"
        )


# -----------------------------
# Save
# -----------------------------
os.makedirs("weights", exist_ok=True)

attention.save_weights(SAVE_PATH)

print("\nTraining finished.")
print(f"Saved weights -> {SAVE_PATH}")
