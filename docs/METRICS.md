# Metrics

## Benchmark Results — 2025-07-03

### Architectures: 2
- Perceptron (1957)
- Adaptive Threshold Neuron (1948, divergence)

### Datasets: 6
| Dataset | Samples | Features | Separable |
|---|---|---|---|
| Linear Separable | 200 | 2 | Yes |
| Noisy Linear | 200 | 2 | No |
| Concentric Circles | 200 | 2 | No |
| High-Dimensional | 300 | 10 | Yes |
| Large Scale | 1000 | 5 | Yes |
| Extended XOR | 200 | 2 | No |

### Results Summary

| Architecture | Dataset | Accuracy | F1 | Epochs | Time |
|---|---|---|---|---|---|
| Perceptron | Linear Separable | 100.0% | 100.0% | 2 | 1.9ms |
| Adaptive Threshold Neuron | Linear Separable | 100.0% | 100.0% | 2 | 2.2ms |
| Perceptron | Noisy Linear | 82.0% | 82.0% | 200 | 118.1ms |
| Adaptive Threshold Neuron | Noisy Linear | 82.0% | 82.0% | 200 | 194.1ms |
| Perceptron | Concentric Circles | 54.5% | 53.3% | 200 | 118.7ms |
| Adaptive Threshold Neuron | Concentric Circles | 54.5% | 53.3% | 200 | 191.3ms |
| Perceptron | High-Dimensional | 100.0% | 100.0% | 2 | 1.8ms |
| Adaptive Threshold Neuron | High-Dimensional | 100.0% | 100.0% | 2 | 3.1ms |
| Perceptron | Large Scale | 100.0% | 100.0% | 2 | 6.0ms |
| Adaptive Threshold Neuron | Large Scale | 100.0% | 100.0% | 2 | 10.3ms |
| Perceptron | Extended XOR | 48.0% | 54.8% | 200 | 113.3ms |
| Adaptive Threshold Neuron | Extended XOR | 48.0% | 54.8% | 200 | 189.3ms |

### Verdict: Adaptive Threshold Neuron
- **Accuracy**: Identical to Perceptron on all datasets
- **Speed**: 1.6x slower on average due to threshold update overhead
- **Classification**: WORSE than real history (no benefit, slower)

### Code Metrics
| Metric | Value |
|---|---|
| Python LOC (src/) | ~250 |
| Website files | 12 |
| Dependencies | 2 (numpy, matplotlib) |
| Test datasets | 6 |
| Runs per test | 5 |
