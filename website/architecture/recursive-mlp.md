# Recursive MLP — Source Code

```python
"""
Recursive MLP — From Scratch
=============================
An MLP with recurrent connections — the hidden layer feeds its own previous
state back into itself, creating a recursive (recurrent) loop. This allows
the network to maintain memory across time steps and process sequential
patterns that a standard MLP cannot.

Architecture:
    Input (n_features) → Hidden (n_hidden) → Output (1)
                         ↑__________↓
                         recurrent connection

Forward pass (at time t):
    hidden_t = sigmoid(X_t · W1 + hidden_{t-1} · Wh + b1)
    output_t = sigmoid(hidden_t · W2 + b2)

Backward pass (Backpropagation Through Time — BPTT):
    The network is "unrolled" across time steps, and gradients are
    accumulated through the recurrent connections.

This is a divergence architecture — recurrent neural networks (RNNs) were
developed independently by Rumelhart, Hinton, and Williams (1986) and by
Elman (1990) with the Simple Recurrent Network. A "Recursive MLP" in 1991
sits at the intersection of these ideas — an MLP with self-recurrent hidden
units, trained via BPTT. It could have been a natural next step after the
MLP: "what if the hidden layer remembers?"
"""

import numpy as np
from typing import List


class RecursiveMLP:
    """An MLP with recurrent hidden connections, trained via BPTT."""

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 200,
                 n_hidden: int = 4, n_time_steps: int = 3):
        """
        Initialize the recursive MLP.

        Args:
            learning_rate: Step size for gradient descent.
            n_epochs: Maximum number of passes over the training data.
            n_hidden: Number of neurons in the hidden layer.
            n_time_steps: Number of time steps to unroll for BPTT.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_hidden = n_hidden
        self.n_time_steps = n_time_steps
        self.weights1 = None    # input → hidden
        self.weights_h = None   # hidden → hidden (recurrent)
        self.bias1 = None
        self.weights2 = None    # hidden → output
        self.bias2 = None
        self.errors_per_epoch: List[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid: 1 / (1 + e^-z). Works on arrays."""
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    @staticmethod
    def _sigmoid_derivative(s: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid given output s: s * (1 - s). Works on arrays."""
        return s * (1.0 - s)

    def _forward_batch(self, X_seq: np.ndarray) -> tuple:
        """
        Forward pass over all samples in parallel (batched).

        Args:
            X_seq: Input of shape (n_samples, n_time_steps, n_features).

        Returns:
            Tuple of (hidden_states, output_states).
            hidden_states: (n_time_steps, n_samples, n_hidden)
            output_states: (n_time_steps, n_samples, 1)
        """
        n_samples, n_steps, _ = X_seq.shape
        hidden_states = np.zeros((n_steps, n_samples, self.n_hidden))
        output_states = np.zeros((n_steps, n_samples, 1))

        h = np.zeros((n_samples, self.n_hidden))

        for t in range(n_steps):
            z1 = X_seq[:, t, :] @ self.weights1 + h @ self.weights_h + self.bias1
            h = self._sigmoid(z1)
            hidden_states[t] = h

            z2 = h @ self.weights2 + self.bias2
            output_states[t] = self._sigmoid(z2)

        return hidden_states, output_states

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels for input samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted labels (0 or 1).
        """
        X_seq = X[:, None, :]
        _, outputs = self._forward_batch(X_seq)
        return (outputs[-1, :, 0] >= 0.5).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RecursiveMLP":
        """
        Train the network using backpropagation through time (BPTT).

        All samples are processed in parallel as a batch. Each sample is
        treated as a sequence by repeating it across n_time_steps.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,), values 0 or 1.

        Returns:
            self (fitted network).
        """
        n_samples, n_features = X.shape

        rng = np.random.RandomState(42)
        self.weights1 = rng.randn(n_features, self.n_hidden) * np.sqrt(2.0 / n_features)
        self.weights_h = rng.randn(self.n_hidden, self.n_hidden) * np.sqrt(2.0 / self.n_hidden)
        self.bias1 = np.zeros(self.n_hidden)
        self.weights2 = rng.randn(self.n_hidden, 1) * np.sqrt(2.0 / self.n_hidden)
        self.bias2 = np.zeros(1)
        self.errors_per_epoch = []

        y_col = y.reshape(-1, 1).astype(float)

        # Create sequences by repeating each sample across time steps
        X_seq = np.tile(X[:, None, :], (1, self.n_time_steps, 1))
        y_seq = np.tile(y_col[None, :, :], (self.n_time_steps, 1, 1))

        for epoch in range(self.n_epochs):
            # Forward pass (batched)
            hidden_states, output_states = self._forward_batch(X_seq)

            # Compute MSE
            mse = np.mean((y_seq - output_states) ** 2)
            self.errors_per_epoch.append(float(mse))

            # Backward pass (BPTT, batched)
            dW1 = np.zeros_like(self.weights1)
            dWh = np.zeros_like(self.weights_h)
            db1 = np.zeros_like(self.bias1)
            dW2 = np.zeros_like(self.weights2)
            db2 = np.zeros_like(self.bias2)

            dh_next = np.zeros((n_samples, self.n_hidden))

            for t in range(self.n_time_steps - 1, -1, -1):
                d_output = (y_seq[t] - output_states[t]) * \
                           self._sigmoid_derivative(output_states[t])

                dW2 += hidden_states[t].T @ d_output
                db2 += d_output.sum(axis=0)

                dh = d_output @ self.weights2.T + dh_next
                dh_raw = dh * self._sigmoid_derivative(hidden_states[t])

                dW1 += X_seq[:, t, :].T @ dh_raw
                db1 += dh_raw.sum(axis=0)

                if t > 0:
                    dWh += hidden_states[t - 1].T @ dh_raw

                dh_next = dh_raw @ self.weights_h.T

            # Update weights
            lr = self.learning_rate
            self.weights2 += lr * dW2 / (n_samples * self.n_time_steps)
            self.bias2 += lr * db2 / (n_samples * self.n_time_steps)
            self.weights1 += lr * dW1 / (n_samples * self.n_time_steps)
            self.bias1 += lr * db1 / (n_samples * self.n_time_steps)
            self.weights_h += lr * dWh / (n_samples * self.n_time_steps)

            # Check convergence
            predictions = (output_states[-1, :, 0] >= 0.5).astype(int)
            if np.all(predictions == y):
                print(f"Converged at epoch {epoch + 1}")
                break

        return self


# --- Example: Sequential pattern recognition ---

if __name__ == "__main__":
    # The recursive MLP can learn patterns where temporal context matters.
    # Here we test it on a simple binary classification task.
    rng = np.random.RandomState(42)
    n = 100

    # Two clusters
    X = np.vstack([
        rng.randn(n, 2) * 0.5 + [2, 2],
        rng.randn(n, 2) * 0.5 + [-2, -2],
    ])
    y = np.array([1] * n + [0] * n)

    rmlp = RecursiveMLP(
        learning_rate=0.1, n_epochs=200, n_hidden=4, n_time_steps=3
    )
    rmlp.fit(X, y)

    predictions = rmlp.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy:     {accuracy:.1%}")
    print(f"Epochs:       {len(rmlp.errors_per_epoch)}")
    print(f"Final MSE:    {rmlp.errors_per_epoch[-1]:.6f}")
    print(f"Predictions:  {predictions[:10]}")
    print(f"Expected:     {y[:10]}")
```
