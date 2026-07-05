"""
Multi-Layer Perceptron — From Scratch
=====================================
A multi-layer perceptron (MLP) with one hidden layer, sigmoid activation,
and backpropagation learning — implemented entirely from scratch using
only NumPy for matrix operations.

The MLP overcomes the fundamental limitation of the single-layer perceptron:
it can learn non-linearly-separable patterns like XOR, concentric circles,
and other complex decision boundaries. This is achieved through:

1. A hidden layer with non-linear activation (sigmoid)
2. Backpropagation — gradient descent through the hidden layer

Architecture:
    Input layer (n_features) → Hidden layer (n_hidden) → Output layer (1)

Forward pass:
    hidden = sigmoid(X · W1 + b1)
    output = sigmoid(hidden · W2 + b2)

Backward pass:
    delta_output = (y - output) * sigmoid'(output)
    delta_hidden = (delta_output · W2^T) * sigmoid'(hidden)
    W2 += lr * hidden^T · delta_output
    W1 += lr * X^T · delta_hidden

This architecture was popularized in 1986 by Rumelhart, Hinton, and Williams
in their landmark paper "Learning representations by back-propagating errors."
It solved the fundamental problem that had held neural networks back since
Minsky and Papert's 1969 critique: how to train hidden layers when you can't
directly observe their errors.
"""

import numpy as np
from typing import List


class MultiLayerPerceptron:
    """A multi-layer perceptron with one hidden layer and backpropagation."""

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 200,
                 n_hidden: int = 4):
        """
        Initialize the multi-layer perceptron.

        Args:
            learning_rate: Step size for gradient descent.
            n_epochs: Maximum number of passes over the training data.
            n_hidden: Number of neurons in the hidden layer.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_hidden = n_hidden
        self.weights1 = None  # input → hidden
        self.bias1 = None
        self.weights2 = None  # hidden → output
        self.bias2 = None
        self.errors_per_epoch: List[float] = []

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Sigmoid activation function: 1 / (1 + e^-z)."""
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        else:
            ez = np.exp(z)
            return ez / (1.0 + ez)

    @staticmethod
    def _sigmoid_derivative(s: float) -> float:
        """Derivative of sigmoid, given the sigmoid output s: s * (1 - s)."""
        return s * (1.0 - s)

    def _forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass through the network.

        Args:
            X: Input matrix of shape (n_samples, n_features).

        Returns:
            Tuple of (hidden_activations, output_activations).
        """
        # Hidden layer: sigmoid(X · W1 + b1)
        z1 = np.dot(X, self.weights1) + self.bias1
        hidden = np.vectorize(self._sigmoid)(z1)

        # Output layer: sigmoid(hidden · W2 + b2)
        z2 = np.dot(hidden, self.weights2) + self.bias2
        output = np.vectorize(self._sigmoid)(z2)

        return hidden, output

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels for input samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted labels (0 or 1).
        """
        _, output = self._forward(X)
        return (output >= 0.5).astype(int).flatten()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiLayerPerceptron":
        """
        Train the network using backpropagation (gradient descent).

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,), values 0 or 1.

        Returns:
            self (fitted network).
        """
        n_samples, n_features = X.shape

        # Initialize weights with small random values (Xavier-like)
        rng = np.random.RandomState(42)
        self.weights1 = rng.randn(n_features, self.n_hidden) * np.sqrt(2.0 / n_features)
        self.bias1 = np.zeros(self.n_hidden)
        self.weights2 = rng.randn(self.n_hidden, 1) * np.sqrt(2.0 / self.n_hidden)
        self.bias2 = np.zeros(1)
        self.errors_per_epoch = []

        # Reshape y to column vector
        y_col = y.reshape(-1, 1).astype(float)

        for epoch in range(self.n_epochs):
            # --- Forward pass ---
            hidden, output = self._forward(X)

            # --- Backward pass (backpropagation) ---
            # Output layer error: (y - output) * sigmoid'(output)
            output_error = (y_col - output) * np.vectorize(self._sigmoid_derivative)(output)

            # Hidden layer error: (output_error · W2^T) * sigmoid'(hidden)
            hidden_error = np.dot(output_error, self.weights2.T) * np.vectorize(self._sigmoid_derivative)(hidden)

            # --- Weight updates ---
            self.weights2 += self.learning_rate * np.dot(hidden.T, output_error)
            self.bias2 += self.learning_rate * np.sum(output_error, axis=0)
            self.weights1 += self.learning_rate * np.dot(X.T, hidden_error)
            self.bias1 += self.learning_rate * np.sum(hidden_error, axis=0)

            # Track mean squared error
            mse = np.mean((y_col - output) ** 2)
            self.errors_per_epoch.append(float(mse))

            # Check convergence
            predictions = (output >= 0.5).astype(int).flatten()
            errors = np.sum(predictions != y)
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break

        return self


# --- Example: Learning the XOR gate ---

if __name__ == "__main__":
    # XOR is the classic non-linearly-separable problem.
    # A single-layer perceptron cannot learn it. An MLP can.
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y_train = np.array([0, 1, 1, 0])  # XOR outputs

    # Create and train the MLP
    mlp = MultiLayerPerceptron(
        learning_rate=0.5, n_epochs=2000, n_hidden=4
    )
    mlp.fit(X_train, y_train)

    # Test predictions
    predictions = mlp.predict(X_train)
    print(f"Weights1:\n{mlp.weights1}")
    print(f"Bias1:   {mlp.bias1}")
    print(f"Weights2: {mlp.weights2.flatten()}")
    print(f"Bias2:   {mlp.bias2}")
    print(f"Predictions:  {predictions}")
    print(f"Expected:     {y_train}")
    print(f"Final MSE:    {mlp.errors_per_epoch[-1]:.6f}")
    print(f"Epochs:       {len(mlp.errors_per_epoch)}")
