# Neocognitron — Source Code

```python
"""
Neocognitron — From Scratch
============================
A hierarchical neural network inspired by the visual cortex, invented by
Kunihiko Fukushima in 1979. It introduced the concepts of:
  - Simple cells (S-cells): feature detectors with local receptive fields
  - Complex cells (C-cells): spatial pooling for shift invariance
  - Hierarchical alternation: S-layers and C-layers alternate

This is the direct ancestor of convolutional neural networks (CNNs). The
key insight — that feature hierarchies with local receptive fields and
spatial pooling can achieve shift-invariant pattern recognition — predates
LeCun's LeNet (1989) by a decade.

Architecture (simplified for this implementation):
    Input → S1 (feature extraction) → C1 (pooling) → S2 → C2 → Output

S-cells use inhibitory competition (lateral inhibition) and excitatory
inputs from their receptive field. C-cells average over spatial positions
to provide tolerance to shifts.

Training: The original Neocognitron used unsupervised competitive learning.
For this implementation, we use a simplified supervised approach where
S-cell receptive fields are learned via Hebbian updates, and the output
layer is trained with the perceptron learning rule.

Reference:
    Fukushima, K. (1979). "Neocognitron: A self-organizing neural network
    model for a mechanism of pattern recognition unaffected by shift in
    position." Biological Cybernetics, 36(4), 193–202.
"""

import numpy as np
from typing import List


class Neocognitron:
    """A simplified Neocognitron with S-cell and C-cell layers."""

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 100,
                 n_s1_kernels: int = 4, kernel_size: int = 3,
                 pool_size: int = 2):
        """
        Initialize the Neocognitron.

        Args:
            learning_rate: Step size for weight updates.
            n_epochs: Maximum number of training passes.
            n_s1_kernels: Number of S1 feature detectors.
            kernel_size: Size of the receptive field (square).
            pool_size: Size of the spatial pooling window.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_s1_kernels = n_s1_kernels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.s1_weights = None
        self.s2_weights = None
        self.output_weights = None
        self.errors_per_epoch: List[float] = []

    def _convolve(self, X: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        2D convolution (valid mode) on a single feature map.

        Args:
            X: Input feature map of shape (h, w).
            kernel: Convolution kernel of shape (kh, kw).

        Returns:
            Output feature map of shape (h - kh + 1, w - kw + 1).
        """
        kh, kw = kernel.shape
        h, w = X.shape
        oh, ow = h - kh + 1, w - kw + 1
        out = np.zeros((oh, ow))
        for i in range(oh):
            for j in range(ow):
                out[i, j] = np.sum(X[i:i+kh, j:j+kw] * kernel)
        return out

    def _pool(self, feature_map: np.ndarray) -> np.ndarray:
        """
        Average pooling over the feature map.

        Args:
            feature_map: 2D array.

        Returns:
            Pooled feature map.
        """
        h, w = feature_map.shape
        ph, pw = self.pool_size, self.pool_size
        oh, ow = h // ph, w // pw
        out = np.zeros((oh, ow))
        for i in range(oh):
            for j in range(ow):
                out[i, j] = np.mean(feature_map[i*ph:(i+1)*ph, j*pw:(j+1)*pw])
        return out

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            X: Input of shape (n_samples, n_features). Features are
               reshaped to a square feature map.

        Returns:
            Flattened C2 output features for classification.
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        side = int(np.ceil(np.sqrt(n_features)))

        all_c2_features = []

        for s in range(n_samples):
            # Pad to square if needed
            x_map = np.zeros((side, side))
            x_flat = X[s]
            x_map.flat[:n_features] = x_flat
            c2_features = []

            for k in range(self.n_s1_kernels):
                # S1: convolve with learned kernel
                s1_out = self._convolve(x_map, self.s1_weights[k])
                s1_out = np.maximum(s1_out, 0)  # ReLU-like activation

                # C1: spatial pooling
                c1_out = self._pool(s1_out)

                # S2: another convolution with a 1x1 "combination" weight
                s2_out = c1_out * self.s2_weights[k]
                s2_out = np.maximum(s2_out, 0)

                # C2: final pooling
                c2_out = self._pool(s2_out)
                c2_features.append(c2_out.flatten())

            all_c2_features.append(np.concatenate(c2_features))

        return np.array(all_c2_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted labels (0 or 1).
        """
        features = self._forward(X)
        linear_output = features @ self.output_weights
        return (linear_output >= 0.5).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Neocognitron":
        """
        Train the Neocognitron using Hebbian learning for S-cells
        and perceptron rule for the output layer.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,), values 0 or 1.

        Returns:
            self (fitted network).
        """
        n_samples, n_features = X.shape
        side = int(np.ceil(np.sqrt(n_features)))

        rng = np.random.RandomState(42)

        # Adapt kernel size to feature map dimensions
        effective_kernel = min(self.kernel_size, side)
        if effective_kernel < 1:
            effective_kernel = 1

        # Initialize S1 kernels (random small values)
        self.s1_weights = rng.randn(self.n_s1_kernels, effective_kernel,
                                    effective_kernel) * 0.1

        # Initialize S2 combination weights
        c1_side = max(1, (side - effective_kernel + 1) // self.pool_size)
        self.s2_weights = rng.randn(self.n_s1_kernels, c1_side, c1_side) * 0.1

        self.errors_per_epoch = []

        # Phase 1: Hebbian learning for S1 kernels (unsupervised)
        for epoch in range(min(20, self.n_epochs)):
            for s in range(n_samples):
                x_map = np.zeros((side, side))
                x_map.flat[:n_features] = X[s]
                for k in range(self.n_s1_kernels):
                    s1_out = self._convolve(x_map, self.s1_weights[k])
                    activation = np.mean(np.maximum(s1_out, 0))
                    # Hebbian update: strengthen active connections
                    update = self.learning_rate * activation * x_map[
                        :effective_kernel, :effective_kernel
                    ]
                    self.s1_weights[k] += update
                    # Normalize kernel
                    norm = np.linalg.norm(self.s1_weights[k])
                    if norm > 0:
                        self.s1_weights[k] /= norm

        # Phase 2: Train output layer (perceptron on extracted features)
        features = self._forward(X)
        n_output_features = features.shape[1]
        self.output_weights = np.zeros(n_output_features)
        output_bias = 0.0

        for epoch in range(self.n_epochs):
            errors = 0
            for i in range(n_samples):
                linear = features[i] @ self.output_weights + output_bias
                pred = 1 if linear >= 0.5 else 0
                update = self.learning_rate * (y[i] - pred)
                self.output_weights += update * features[i]
                output_bias += update
                if update != 0.0:
                    errors += 1

            self.errors_per_epoch.append(float(errors))

            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break

        return self


# --- Example ---

if __name__ == "__main__":
    rng = np.random.RandomState(42)
    n = 80

    # Create simple 4x4 "images" (16 features)
    X = np.vstack([
        rng.randn(n, 16) * 0.3 + np.array([1, 1, 0, 0, 1, 1, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0]),
        rng.randn(n, 16) * 0.3 + np.array([0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 1, 1, 0, 0, 1, 1]),
    ])
    y = np.array([1] * n + [0] * n)

    model = Neocognitron(
        learning_rate=0.1, n_epochs=100,
        n_s1_kernels=4, kernel_size=2, pool_size=2
    )
    model.fit(X, y)

    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy:     {accuracy:.1%}")
    print(f"Epochs:       {len(model.errors_per_epoch)}")
```
