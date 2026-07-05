# Boltzmann Machine — Source Code

```python
"""
Boltzmann Machine — From Scratch
=================================
A stochastic neural network with visible and hidden units, symmetric
connections, and an energy function inspired by statistical physics.
Published by Ackley, Hinton, and Sejnowski in 1985.

Key properties:
  - Symmetric weights: W_ij = W_ji, no self-connections
  - Binary units: each neuron is 0 or 1 (Bernoulli)
  - Energy: E = -Σ_i b_i * v_i - Σ_j c_j * h_j - Σ_ij W_ij * v_i * h_j
  - Probability: P(unit=1) = sigmoid(activation)
  - Training: Contrastive Divergence (CD-1) — a simplified approximation
    of maximum likelihood learning

The Boltzmann Machine is the ancestor of:
  - Restricted Boltzmann Machines (RBMs, 2002)
  - Deep Belief Networks (2006)
  - Modern energy-based models

For classification, we train the BM as a feature extractor (unsupervised),
then train a simple linear classifier on the learned hidden representations.

Reference:
    Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. (1985).
    "A learning algorithm for Boltzmann machines." Cognitive Science,
    9(1), 147–169.
"""

import numpy as np
from typing import List


class BoltzmannMachine:
    """A Boltzmann Machine with visible and hidden units, trained via CD-1."""

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 100,
                 n_hidden: int = 4, cd_steps: int = 1):
        """
        Initialize the Boltzmann Machine.

        Args:
            learning_rate: Step size for contrastive divergence.
            n_epochs: Number of training passes.
            n_hidden: Number of hidden units.
            cd_steps: Number of Gibbs sampling steps for contrastive divergence.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_hidden = n_hidden
        self.cd_steps = cd_steps
        self.weights = None      # visible-hidden weights
        self.visible_bias = None
        self.hidden_bias = None
        self.classifier_weights = None
        self.classifier_bias = 0.0
        self.errors_per_epoch: List[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                        np.exp(z) / (1.0 + np.exp(z)))

    def _sample_hidden(self, visible: np.ndarray) -> tuple:
        """
        Sample hidden units given visible units.

        Args:
            visible: Visible units of shape (n_visible,) or (n_samples, n_visible).

        Returns:
            Tuple of (probabilities, binary samples).
        """
        if visible.ndim == 1:
            visible = visible[None, :]

        activation = visible @ self.weights + self.hidden_bias
        prob = self._sigmoid(activation)
        sample = (prob > np.random.random(prob.shape)).astype(float)
        return prob, sample

    def _sample_visible(self, hidden: np.ndarray) -> tuple:
        """
        Sample visible units given hidden units.

        Args:
            hidden: Hidden units of shape (n_hidden,) or (n_samples, n_hidden).

        Returns:
            Tuple of (probabilities, binary samples).
        """
        if hidden.ndim == 1:
            hidden = hidden[None, :]

        activation = hidden @ self.weights.T + self.visible_bias
        prob = self._sigmoid(activation)
        sample = (prob > np.random.random(prob.shape)).astype(float)
        return prob, sample

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform inputs to hidden representations (deterministic).

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Hidden activations of shape (n_samples, n_hidden).
        """
        binary_X = (X > np.median(X, axis=0)).astype(float)
        hidden_prob, _ = self._sample_hidden(binary_X)
        return hidden_prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using the trained linear classifier on hidden features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted labels (0 or 1).
        """
        features = self._transform(X)
        linear = features @ self.classifier_weights + self.classifier_bias
        return (linear >= 0.5).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BoltzmannMachine":
        """
        Train the Boltzmann Machine using Contrastive Divergence (CD-1),
        then train a linear classifier on the learned features.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,), values 0 or 1.

        Returns:
            self (fitted network).
        """
        n_samples, n_features = X.shape
        rng = np.random.RandomState(42)

        # Binarize inputs (Boltzmann machines use binary visible units)
        threshold = np.median(X, axis=0)
        X_binary = (X > threshold).astype(float)

        # Initialize parameters
        self.weights = rng.randn(n_features, self.n_hidden) * 0.01
        self.visible_bias = np.zeros(n_features)
        self.hidden_bias = np.zeros(self.n_hidden)
        self.errors_per_epoch = []

        # Phase 1: Unsupervised training via Contrastive Divergence
        batch_size = min(32, n_samples)

        for epoch in range(self.n_epochs):
            # Shuffle data
            perm = rng.permutation(n_samples)
            epoch_error = 0.0

            for batch_start in range(0, n_samples, batch_size):
                batch = X_binary[perm[batch_start:batch_start + batch_size]]
                bs = batch.shape[0]

                # Positive phase: sample hidden from real data
                pos_hidden_prob, pos_hidden = self._sample_hidden(batch)

                # Positive associations
                pos_assoc = batch.T @ pos_hidden_prob  # (n_visible, n_hidden)

                # Negative phase: Gibbs sampling
                neg_visible = batch.copy()
                for _ in range(self.cd_steps):
                    _, neg_hidden = self._sample_hidden(neg_visible)
                    neg_visible_prob, neg_visible = self._sample_visible(neg_hidden)

                neg_hidden_prob, _ = self._sample_hidden(neg_visible)

                # Negative associations
                neg_assoc = neg_visible.T @ neg_hidden_prob

                # Update weights
                lr = self.learning_rate / bs
                self.weights += lr * (pos_assoc - neg_assoc)
                self.visible_bias += lr * (batch.sum(axis=0) - neg_visible.sum(axis=0))
                self.hidden_bias += lr * (pos_hidden_prob.sum(axis=0) - neg_hidden_prob.sum(axis=0))

                # Reconstruction error
                recon_prob, _ = self._sample_visible(pos_hidden_prob)
                epoch_error += np.mean((batch - recon_prob) ** 2)

            self.errors_per_epoch.append(float(epoch_error / (n_samples / batch_size)))

        # Phase 2: Train linear classifier on learned features
        features = self._transform(X)
        n_feat = features.shape[1]
        self.classifier_weights = np.zeros(n_feat)
        self.classifier_bias = 0.0

        for epoch in range(self.n_epochs):
            errors = 0
            for i in range(n_samples):
                linear = features[i] @ self.classifier_weights + self.classifier_bias
                pred = 1 if linear >= 0.5 else 0
                update = self.learning_rate * (y[i] - pred)
                self.classifier_weights += update * features[i]
                self.classifier_bias += update
                if update != 0.0:
                    errors += 1

            if errors == 0:
                print(f"Classifier converged at epoch {epoch + 1}")
                break

        return self


# --- Example ---

if __name__ == "__main__":
    rng = np.random.RandomState(42)
    n = 80

    X = np.vstack([
        rng.randn(n, 6) * 0.5 + [2, 2, 2, -1, -1, -1],
        rng.randn(n, 6) * 0.5 + [-1, -1, -1, 2, 2, 2],
    ])
    y = np.array([1] * n + [0] * n)

    model = BoltzmannMachine(
        learning_rate=0.1, n_epochs=50, n_hidden=4
    )
    model.fit(X, y)

    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy:     {accuracy:.1%}")
    print(f"Epochs:       {len(model.errors_per_epoch)}")
    print(f"Final recon error: {model.errors_per_epoch[-1]:.6f}")
```
