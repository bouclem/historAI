# Perceptron — Source Code

```python
"""
Perceptron — From Scratch
=========================
A from-scratch implementation of Frank Rosenblatt's perceptron algorithm (1957-1958).

The perceptron is a binary linear classifier. It learns by adjusting weights
based on prediction errors, using the perceptron learning rule:

    w_i(t+1) = w_i(t) + r * (d_j - y_j) * x_j,i

where:
    w  = weights
    r  = learning rate
    d  = desired (true) output
    y  = predicted output
    x  = input features

References:
    - Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for
      Information Storage and Organization in the Brain."
    - Rosenblatt, F. (1962). "Principles of Neurodynamics."
"""

import numpy as np
from typing import List, Tuple


class Perceptron:
    """A single-layer perceptron classifier built from scratch."""

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 100):
        """
        Initialize the perceptron.

        Args:
            learning_rate: Step size for weight updates (0 < r <= 1).
            n_epochs: Maximum number of passes over the training data.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Train the perceptron on the given data.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,), values 0 or 1.

        Returns:
            self (fitted perceptron).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
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

            # Convergence: no errors means perfect classification
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break

        return self

    def decision_boundary(self) -> Tuple[float, float]:
        """
        Return the slope and intercept of the decision boundary line
        (valid for 2D inputs only).

        Returns:
            (slope, intercept) of the line w1*x1 + w2*x2 + b = 0.
        """
        if len(self.weights) != 2:
            raise ValueError("Decision boundary only available for 2D inputs.")
        slope = -self.weights[0] / self.weights[1]
        intercept = -self.bias / self.weights[1]
        return slope, intercept


# --- Example: Learning the AND gate ---

if __name__ == "__main__":
    # Training data: AND logic gate
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y_train = np.array([0, 0, 0, 1])  # AND outputs

    # Create and train the perceptron
    p = Perceptron(learning_rate=0.1, n_epochs=100)
    p.fit(X_train, y_train)

    # Test predictions
    predictions = p.predict(X_train)
    print(f"Weights: {p.weights}")
    print(f"Bias:    {p.bias}")
    print(f"Predictions: {predictions}")
    print(f"Expected:    {y_train}")
    print(f"Errors per epoch: {p.errors_per_epoch}")

    # Show decision boundary
    slope, intercept = p.decision_boundary()
    print(f"Decision boundary: y = {slope:.4f}x + {intercept:.4f}")
```
