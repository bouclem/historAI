# Multi-Threshold Perceptron — Source Code

```python
"""
Multi-Threshold Perceptron — From Scratch
=========================================
A perceptron variant where the activation function uses multiple thresholds
instead of a single one, creating a multi-step function that divides the
real line into alternating firing and non-firing regions.

The standard perceptron (1957) uses a single threshold: fire if z >= θ.
This model uses K thresholds θ_0 < θ_1 < ... < θ_{K-1}, creating K+1 regions.
The neuron fires in alternating regions (odd-indexed regions fire, even ones
don't). This allows a single neuron to learn non-linearly-separable patterns
— like concentric circles or band-shaped distributions — that the standard
perceptron cannot.

Learning rules:
    Weights (same as perceptron):
        w_i(t+1) = w_i(t) + r * (d_j - y_j) * x_j,i

    Thresholds (nearest threshold moves toward z):
        θ_nearest += r * sign(z - θ_nearest) * |d_j - y_j|

When the neuron misclassifies, the threshold nearest to the current linear
output moves toward it — shrinking the region that caused the error and
expanding the adjacent region that would correct it.

This is a divergence architecture — it could have existed in 1958, right
after Rosenblatt's perceptron, as a natural extension to multiple thresholds.
"""

import numpy as np
from typing import List, Tuple


class MultiThresholdPerceptron:
    """A perceptron with multiple thresholds forming a multi-step activation."""

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 100,
                 n_thresholds: int = 2, initial_thresholds: List[float] = None):
        """
        Initialize the multi-threshold perceptron.

        Args:
            learning_rate: Step size for weight and threshold updates.
            n_epochs: Maximum number of passes over the training data.
            n_thresholds: Number of thresholds defining the activation regions.
            initial_thresholds: Optional list of starting threshold values.
                If None, thresholds are evenly spaced around zero.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_thresholds = n_thresholds
        if initial_thresholds is not None:
            self.thresholds = sorted(initial_thresholds)
        else:
            spread = n_thresholds
            self.thresholds = list(np.linspace(-spread, spread, n_thresholds))
        self.weights = None
        self.errors_per_epoch: List[int] = []
        self.threshold_history: List[List[float]] = []

    def activation(self, z: float) -> int:
        """
        Multi-step activation: fires in alternating regions.

        Thresholds divide the real line into K+1 regions. The neuron fires
        (returns 1) in odd-indexed regions, doesn't fire (returns 0) in
        even-indexed regions.

        Args:
            z: Linear output (weighted sum of inputs).

        Returns:
            1 if z falls in a firing region, 0 otherwise.
        """
        region = 0
        for t in self.thresholds:
            if z >= t:
                region += 1
            else:
                break
        return region % 2

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiThresholdPerceptron":
        """
        Train the neuron — weights and all thresholds adapt.

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
                z = np.dot(X[i], self.weights)
                prediction = self.activation(z)
                error = y[i] - prediction

                if error != 0:
                    # Update weights (perceptron rule)
                    self.weights += self.learning_rate * error * X[i]

                    # Update nearest threshold — move it toward z
                    distances = [abs(z - t) for t in self.thresholds]
                    nearest_idx = int(np.argmin(distances))
                    nearest = self.thresholds[nearest_idx]
                    direction = 1.0 if z > nearest else -1.0
                    self.thresholds[nearest_idx] = (
                        nearest + self.learning_rate * direction * abs(error)
                    )
                    self.thresholds.sort()

                    errors += 1

            self.errors_per_epoch.append(errors)
            self.threshold_history.append(list(self.thresholds))

            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                print(f"Final thresholds: {[f'{t:.4f}' for t in self.thresholds]}")
                break

        return self

    def decision_boundaries(self) -> List[Tuple[float, float]]:
        """
        Return slope and intercept for each threshold boundary line
        (valid for 2D inputs only).

        Each threshold θ_i defines a line: w1*x1 + w2*x2 = θ_i.

        Returns:
            List of (slope, intercept) tuples, one per threshold.
        """
        if len(self.weights) != 2:
            raise ValueError("Decision boundaries only available for 2D inputs.")
        boundaries = []
        for t in self.thresholds:
            slope = -self.weights[0] / self.weights[1]
            intercept = t / self.weights[1]
            boundaries.append((slope, intercept))
        return boundaries


# --- Example: Learning the XOR gate ---

if __name__ == "__main__":
    # XOR is not linearly separable — a standard perceptron cannot learn it.
    # The multi-threshold perceptron can, because its multi-step activation
    # creates a non-linear decision boundary with a single neuron.
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y_train = np.array([0, 1, 1, 0])  # XOR outputs

    # Create and train the multi-threshold perceptron
    mtp = MultiThresholdPerceptron(
        learning_rate=0.1, n_epochs=100, n_thresholds=2
    )
    mtp.fit(X_train, y_train)

    # Test predictions
    predictions = mtp.predict(X_train)
    print(f"Weights:      {mtp.weights}")
    print(f"Thresholds:   {[f'{t:.4f}' for t in mtp.thresholds]}")
    print(f"Predictions:  {predictions}")
    print(f"Expected:     {y_train}")
    print(f"Errors per epoch: {mtp.errors_per_epoch}")
    print(f"Threshold history (last 5): "
          f"{[[f'{t:.3f}' for t in epoch] for epoch in mtp.threshold_history[-5:]]}")

    # Show decision boundaries
    boundaries = mtp.decision_boundaries()
    for i, (slope, intercept) in enumerate(boundaries):
        print(f"Boundary {i}: y = {slope:.4f}x + {intercept:.4f}")
```
