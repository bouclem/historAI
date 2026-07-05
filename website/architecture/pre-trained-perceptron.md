# Pre-Trained Perceptron — Source Code

```python
"""
Pre-Trained Perceptron — From Scratch
=====================================
A perceptron that starts with weights learned on a previous task and fine-tunes
on a new one — transfer learning, decades before the concept had a name.

The standard perceptron (1957) initializes weights to zero and learns from
scratch. The Pre-Trained Perceptron accepts pre-trained weights from a source
task and adapts them to a target task. The intuition: if the source and target
tasks share structure (similar features, related decision boundaries), the
pre-trained weights provide a better starting point than zeros — leading to
faster convergence and potentially better solutions.

Learning rule (same as perceptron, but starting from pre-trained weights):
    w_i(t+1) = w_i(t) + r * (d_j - y_j) * x_j,i
    b(t+1)   = b(t)   + r * (d_j - y_j)

The key question is: does transfer help? When the source task is similar to
the target, the pre-trained weights are already close to a good solution —
fewer updates are needed. When the source task is unrelated, the pre-trained
weights can be worse than random initialization, and the perceptron must
"unlearn" before it can learn.

This is a divergence architecture — transfer learning wasn't formally studied
until the 1990s, and pre-training became prominent with deep belief networks
in 2006. But the idea is simple enough to have been explored in 1974, during
the AI winter, when researchers were looking for ways to make perceptrons
more efficient with limited compute.
"""

import numpy as np
from typing import List, Optional


class PreTrainedPerceptron:
    """A perceptron that starts from pre-trained weights and fine-tunes."""

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 100,
                 pretrained_weights: Optional[np.ndarray] = None,
                 pretrained_bias: float = 0.0):
        """
        Initialize the pre-trained perceptron.

        Args:
            learning_rate: Step size for weight updates.
            n_epochs: Maximum number of passes over the training data.
            pretrained_weights: Weights learned on a source task. If None,
                falls back to zero initialization (standard perceptron).
            pretrained_bias: Bias learned on a source task.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.pretrained_weights = pretrained_weights
        self.pretrained_bias = pretrained_bias
        self.weights = None
        self.bias = 0.0
        self.errors_per_epoch: List[int] = []

    def activation(self, z: float) -> int:
        """Step function: returns 1 if z >= 0, else 0."""
        return 1 if z >= 0 else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels for input samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted labels (0 or 1).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation(z) for z in linear_output])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PreTrainedPerceptron":
        """
        Fine-tune the pre-trained weights on the target task.

        If no pre-trained weights are provided, self-generates them by training
        on a perturbed version of the data (simulating a related source task).

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,), values 0 or 1.

        Returns:
            self (fitted perceptron).
        """
        n_samples, n_features = X.shape

        # If no pre-trained weights, self-generate from a perturbed source task
        if self.pretrained_weights is None:
            rng = np.random.RandomState(99)
            # Perturb the data slightly to create a related but different task
            X_source = X + rng.randn(*X.shape) * 0.3
            y_source = y.copy()
            # Flip a few labels to make it imperfect (realistic pre-training)
            flip_idx = rng.choice(n_samples, size=max(1, n_samples // 10), replace=False)
            y_source[flip_idx] = 1 - y_source[flip_idx]

            # Train a quick perceptron on the source task
            source_weights = np.zeros(n_features)
            source_bias = 0.0
            for epoch in range(20):
                errors = 0
                for i in range(n_samples):
                    z = np.dot(X_source[i], source_weights) + source_bias
                    pred = 1 if z >= 0 else 0
                    update = self.learning_rate * (y_source[i] - pred)
                    source_weights += update * X_source[i]
                    source_bias += update
                    if update != 0.0:
                        errors += 1
                if errors == 0:
                    break

            self.pretrained_weights = source_weights
            self.pretrained_bias = source_bias

        # Start from pre-trained weights
        self.weights = self.pretrained_weights.copy().astype(float)
        self.bias = float(self.pretrained_bias)
        self.errors_per_epoch = []

        for epoch in range(self.n_epochs):
            errors = 0
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = self.activation(linear_output)
                update = self.learning_rate * (y[i] - prediction)

                self.weights += update * X[i]
                self.bias += update

                if update != 0.0:
                    errors += 1

            self.errors_per_epoch.append(errors)

            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break

        return self


# --- Example: Transfer learning between related tasks ---

if __name__ == "__main__":
    # Source task: separate points around [2, 2] from [-2, -2]
    rng = np.random.RandomState(42)
    n = 50
    X_source = np.vstack([
        rng.randn(n, 2) * 0.5 + [2, 2],
        rng.randn(n, 2) * 0.5 + [-2, -2],
    ])
    y_source = np.array([1] * n + [0] * n)

    # Train a standard perceptron on the source task
    from src.perceptron.perceptron import Perceptron
    source_model = Perceptron(learning_rate=0.1, n_epochs=100)
    source_model.fit(X_source, y_source)
    print(f"Source weights: {source_model.weights}")
    print(f"Source bias:    {source_model.bias}")

    # Target task: similar boundary but shifted slightly
    X_target = np.vstack([
        rng.randn(n, 2) * 0.5 + [2.5, 1.5],
        rng.randn(n, 2) * 0.5 + [-1.5, -2.5],
    ])
    y_target = np.array([1] * n + [0] * n)

    # Pre-trained perceptron: starts from source weights
    ptp = PreTrainedPerceptron(
        learning_rate=0.1, n_epochs=100,
        pretrained_weights=source_model.weights,
        pretrained_bias=source_model.bias,
    )
    ptp.fit(X_target, y_target)
    print(f"\nPre-trained converged in {len(ptp.errors_per_epoch)} epochs")
    print(f"Final weights: {ptp.weights}")
    print(f"Errors per epoch: {ptp.errors_per_epoch}")

    # Standard perceptron: starts from zeros
    std = Perceptron(learning_rate=0.1, n_epochs=100)
    std.fit(X_target, y_target)
    print(f"\nStandard converged in {len(std.errors_per_epoch)} epochs")
    print(f"Errors per epoch: {std.errors_per_epoch}")
```
