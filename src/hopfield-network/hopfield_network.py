"""
Hopfield Network — From Scratch
================================
A recurrent neural network with symmetric weights that stores patterns
as attractors (local minima of an energy function). Invented by John
Hopfield in 1982, this paper is credited with reviving interest in
neural networks by connecting them to physics (spin glasses, Ising models).

Key properties:
  - Symmetric weights: W_ij = W_ji, no self-connections: W_ii = 0
  - Binary units: each neuron is +1 or -1 (bipolar)
  - Energy function: E = -0.5 * Σ_ij W_ij * s_i * s_j
  - Update rule: s_i = sign(Σ_j W_ij * s_j)
  - Learning (Hebbian): W_ij = (1/N) * Σ_μ x_i^μ * x_j^μ

The network converges to the stored pattern most similar to the input.
For classification, we store one prototype pattern per class (computed
as the mean of training samples) and classify by finding which stored
pattern the input converges to.

Reference:
    Hopfield, J. J. (1982). "Neural networks and physical systems with
    emergent collective computational abilities." Proceedings of the
    National Academy of Sciences, 79(8), 2554–2558.
"""

import numpy as np
from typing import List


class HopfieldNetwork:
    """A Hopfield associative memory network for binary classification."""

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 100,
                 max_recall_steps: int = 50):
        """
        Initialize the Hopfield network.

        Args:
            learning_rate: Used for interface compatibility (Hebbian learning
                doesn't use learning rate, but benchmark expects this param).
            n_epochs: Used for interface compatibility (storage is one-shot).
            max_recall_steps: Maximum asynchronous update steps during recall.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.max_recall_steps = max_recall_steps
        self.weights = None
        self.prototypes = None
        self.errors_per_epoch: List[float] = []

    @staticmethod
    def _to_bipolar(x: np.ndarray) -> np.ndarray:
        """Convert {0, 1} or continuous values to bipolar {-1, +1}."""
        return np.where(x >= np.mean(x), 1, -1)

    @staticmethod
    def _to_binary(x: np.ndarray) -> np.ndarray:
        """Convert bipolar {-1, +1} to binary {0, 1}."""
        return np.where(x > 0, 1, 0)

    def _energy(self, state: np.ndarray) -> float:
        """Compute the Hopfield energy of a state."""
        return -0.5 * state @ self.weights @ state

    def _recall(self, pattern: np.ndarray) -> np.ndarray:
        """
        Asynchronously update the state until convergence or max steps.

        Args:
            pattern: Bipolar input pattern.

        Returns:
            Converged bipolar state.
        """
        state = pattern.copy()
        n = len(state)

        for step in range(self.max_recall_steps):
            updated = False
            order = np.random.RandomState(42).permutation(n)
            for i in order:
                field = np.dot(self.weights[i], state)
                new_val = 1 if field >= 0 else -1
                if new_val != state[i]:
                    state[i] = new_val
                    updated = True

            if not updated:
                break

        return state

    def _hamming_distance(self, a: np.ndarray, b: np.ndarray) -> int:
        """Compute Hamming distance between two bipolar patterns."""
        return int(np.sum(a != b))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels by recalling and matching to stored prototypes.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted labels (0 or 1).
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            pattern = self._to_bipolar(X[i])
            recalled = self._recall(pattern)

            # Find closest prototype
            best_dist = float('inf')
            best_label = 0
            for label, proto in enumerate(self.prototypes):
                dist = self._hamming_distance(recalled, proto)
                if dist < best_dist:
                    best_dist = dist
                    best_label = label

            predictions[i] = best_label

        return predictions

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HopfieldNetwork":
        """
        Store class prototypes using Hebbian learning.

        A prototype is computed as the bipolar mean of all samples in a class.
        The weight matrix is the sum of outer products of prototypes.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,), values 0 or 1.

        Returns:
            self (fitted network).
        """
        n_samples, n_features = X.shape

        # Convert all samples to bipolar
        X_bipolar = np.array([self._to_bipolar(X[i]) for i in range(n_samples)])

        # Create one prototype per class
        classes = np.unique(y)
        self.prototypes = []
        for cls in classes:
            mask = y == cls
            mean_pattern = np.mean(X_bipolar[mask], axis=0)
            proto = np.where(mean_pattern >= 0, 1, -1)
            self.prototypes.append(proto)

        # Hebbian learning: W = Σ outer products of prototypes
        n = n_features
        self.weights = np.zeros((n, n))
        for proto in self.prototypes:
            self.weights += np.outer(proto, proto)

        # Zero diagonal (no self-connections)
        np.fill_diagonal(self.weights, 0)

        # Normalize
        self.weights /= len(self.prototypes)

        self.errors_per_epoch = [0.0]  # One-shot learning

        return self


# --- Example ---

if __name__ == "__main__":
    rng = np.random.RandomState(42)
    n = 50

    # Create patterns
    X = np.vstack([
        rng.randn(n, 8) * 0.3 + [1, 1, 1, 1, -1, -1, -1, -1],
        rng.randn(n, 8) * 0.3 + [-1, -1, -1, -1, 1, 1, 1, 1],
    ])
    y = np.array([1] * n + [0] * n)

    model = HopfieldNetwork(max_recall_steps=30)
    model.fit(X, y)

    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy:     {accuracy:.1%}")
    print(f"Prototypes:   {len(model.prototypes)}")
