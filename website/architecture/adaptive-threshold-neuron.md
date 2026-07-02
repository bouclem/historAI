# Adaptive Threshold Neuron — Source Code

```python
"""
Adaptive Threshold Neuron — From Scratch
=========================================
A neuron model where the threshold adapts automatically during learning.

Unlike the McCulloch-Pitts neuron (1943) which has a fixed threshold, and the
perceptron (1957) which adapts weights but treats threshold as bias, this model
treats the threshold itself as a learnable parameter that adjusts based on
prediction errors.

The learning rule for the threshold:
    θ(t+1) = θ(t) - r * (d_j - y_j)

When the neuron fires too often (predicts 1 when it should predict 0),
the threshold increases (becomes harder to fire).
When the neuron doesn't fire enough (predicts 0 when it should predict 1),
the threshold decreases (becomes easier to fire).

Weights also adapt:
    w_i(t+1) = w_i(t) + r * (d_j - y_j) * x_j,i

This is a divergence architecture — it could have existed in 1948, between
the McCulloch-Pitts neuron and the perceptron.
"""

import numpy as np
from typing import List, Tuple


class AdaptiveThresholdNeuron:
    """A neuron with an automatically adapting threshold."""

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 100,
                 initial_threshold: float = 0.0):
        """
        Initialize the adaptive threshold neuron.

        Args:
            learning_rate: Step size for weight and threshold updates.
            n_epochs: Maximum number of passes over the training data.
            initial_threshold: Starting threshold value.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.threshold = initial_threshold
        self.weights = None
        self.errors_per_epoch: List[int] = []
        self.threshold_history: List[float] = []

    def activation(self, z: float) -> int:
        """Step function: fires (1) if z >= threshold, else 0."""
        return 1 if z >= self.threshold else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels for input samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted labels (0 or 1).
        """
        linear_output = np.dot(X, self.weights)
        return np.array([self.activation(z) for z in linear_output])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveThresholdNeuron":
        """
        Train the neuron — both weights and threshold adapt.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,), values 0 or 1.

        Returns:
            self (fitted neuron).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.errors_per_epoch = []
        self.threshold_history = []

        for epoch in range(self.n_epochs):
            errors = 0
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights)
                prediction = self.activation(linear_output)
                error = y[i] - prediction

                # Update weights
                self.weights += self.learning_rate * error * X[i]

                # Update threshold — moves opposite to error
                # If predicting too many 1s (error < 0), threshold increases
                # If predicting too many 0s (error > 0), threshold decreases
                self.threshold -= self.learning_rate * error

                if error != 0.0:
                    errors += 1

            self.errors_per_epoch.append(errors)
            self.threshold_history.append(self.threshold)

            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                print(f"Final threshold: {self.threshold:.4f}")
                break

        return self

    def decision_boundary(self) -> Tuple[float, float]:
        """
        Return the slope and intercept of the decision boundary line
        (valid for 2D inputs only).

        The boundary is where w·x = θ, i.e. w1*x1 + w2*x2 = θ.

        Returns:
            (slope, intercept) of the decision boundary.
        """
        if len(self.weights) != 2:
            raise ValueError("Decision boundary only available for 2D inputs.")
        slope = -self.weights[0] / self.weights[1]
        intercept = self.threshold / self.weights[1]
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

    # Create and train the adaptive threshold neuron
    neuron = AdaptiveThresholdNeuron(learning_rate=0.1, n_epochs=100)
    neuron.fit(X_train, y_train)

    # Test predictions
    predictions = neuron.predict(X_train)
    print(f"Weights:    {neuron.weights}")
    print(f"Threshold:  {neuron.threshold:.4f}")
    print(f"Predictions: {predictions}")
    print(f"Expected:    {y_train}")
    print(f"Errors per epoch: {neuron.errors_per_epoch}")
    print(f"Threshold history: {[f'{t:.4f}' for t in neuron.threshold_history]}")

    # Show decision boundary
    slope, intercept = neuron.decision_boundary()
    print(f"Decision boundary: y = {slope:.4f}x + {intercept:.4f}")
```
