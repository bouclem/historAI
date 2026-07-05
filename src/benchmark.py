"""
Benchmark — Tests & Benchmarks for All Architectures
====================================================
Runs every architecture in src/ on multiple synthetic datasets of varying
difficulty, measures convergence speed, accuracy, precision, recall, F1,
and stability across repeated runs. Generates comparison charts as PNG
files using matplotlib (Agg backend — no GUI).

Usage:
    python src/benchmark.py

Output:
    PNG charts in src/benchmark_results/
"""

import sys
import os
import time
import importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no window opens
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path so we can import architectures
sys.path.insert(0, str(Path(__file__).parent))

from perceptron.perceptron import Perceptron

# Hyphenated folder names can't be imported normally — use importlib
_atn_spec = importlib.util.spec_from_file_location(
    "adaptive_threshold_neuron",
    Path(__file__).parent / "adaptive-threshold-neuron" / "adaptive_threshold_neuron.py",
)
_atn_module = importlib.util.module_from_spec(_atn_spec)
_atn_spec.loader.exec_module(_atn_module)
AdaptiveThresholdNeuron = _atn_module.AdaptiveThresholdNeuron

_mtp_spec = importlib.util.spec_from_file_location(
    "multi_threshold_perceptron",
    Path(__file__).parent / "multi-threshold-perceptron" / "multi_threshold_perceptron.py",
)
_mtp_module = importlib.util.module_from_spec(_mtp_spec)
_mtp_spec.loader.exec_module(_mtp_module)
MultiThresholdPerceptron = _mtp_module.MultiThresholdPerceptron

_mlp_spec = importlib.util.spec_from_file_location(
    "multi_layer_perceptron",
    Path(__file__).parent / "multi-layer-perceptron" / "multi_layer_perceptron.py",
)
_mlp_module = importlib.util.module_from_spec(_mlp_spec)
_mlp_spec.loader.exec_module(_mlp_module)
MultiLayerPerceptron = _mlp_module.MultiLayerPerceptron

_ptp_spec = importlib.util.spec_from_file_location(
    "pre_trained_perceptron",
    Path(__file__).parent / "pre-trained-perceptron" / "pre_trained_perceptron.py",
)
_ptp_module = importlib.util.module_from_spec(_ptp_spec)
_ptp_spec.loader.exec_module(_ptp_module)
PreTrainedPerceptron = _ptp_module.PreTrainedPerceptron

_rmlp_spec = importlib.util.spec_from_file_location(
    "recursive_mlp",
    Path(__file__).parent / "recursive-mlp" / "recursive_mlp.py",
)
_rmlp_module = importlib.util.module_from_spec(_rmlp_spec)
_rmlp_spec.loader.exec_module(_rmlp_module)
RecursiveMLP = _rmlp_module.RecursiveMLP


# --- Dataset generators ---

def make_linear_separable(n_samples=200, n_features=2, noise=0.3, seed=42):
    """Two linearly separable Gaussian clusters."""
    rng = np.random.RandomState(seed)
    n_per_class = n_samples // 2
    X1 = rng.randn(n_per_class, n_features) * noise + np.array([2.0] * n_features)
    X2 = rng.randn(n_per_class, n_features) * noise + np.array([-2.0] * n_features)
    X = np.vstack([X1, X2])
    y = np.array([1] * n_per_class + [0] * n_per_class)
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def make_noisy_linear(n_samples=200, n_features=2, noise=1.5, seed=42):
    """Overlapping clusters — not perfectly separable."""
    rng = np.random.RandomState(seed)
    n_per_class = n_samples // 2
    X1 = rng.randn(n_per_class, n_features) * noise + np.array([1.0] * n_features)
    X2 = rng.randn(n_per_class, n_features) * noise + np.array([-1.0] * n_features)
    X = np.vstack([X1, X2])
    y = np.array([1] * n_per_class + [0] * n_per_class)
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def make_circles(n_samples=200, noise=0.15, seed=42):
    """Concentric circles — non-linearly separable."""
    rng = np.random.RandomState(seed)
    n_per_class = n_samples // 2
    angles1 = rng.uniform(0, 2 * np.pi, n_per_class)
    r1 = 2.0 + rng.randn(n_per_class) * noise
    X1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    angles2 = rng.uniform(0, 2 * np.pi, n_per_class)
    r2 = 0.8 + rng.randn(n_per_class) * noise
    X2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    X = np.vstack([X1, X2])
    y = np.array([1] * n_per_class + [0] * n_per_class)
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def make_high_dimensional(n_samples=300, n_features=10, noise=0.8, seed=42):
    """High-dimensional linearly separable data."""
    rng = np.random.RandomState(seed)
    n_per_class = n_samples // 2
    center1 = rng.randn(n_features) * 2
    center2 = rng.randn(n_features) * 2
    center2 = center2 + np.sign(center2 - center1) * 3
    X1 = rng.randn(n_per_class, n_features) * noise + center1
    X2 = rng.randn(n_per_class, n_features) * noise + center2
    X = np.vstack([X1, X2])
    y = np.array([1] * n_per_class + [0] * n_per_class)
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def make_large_scale(n_samples=1000, n_features=5, noise=0.6, seed=42):
    """Large dataset — tests scalability."""
    rng = np.random.RandomState(seed)
    n_per_class = n_samples // 2
    X1 = rng.randn(n_per_class, n_features) * noise + np.array([1.5] * n_features)
    X2 = rng.randn(n_per_class, n_features) * noise + np.array([-1.5] * n_features)
    X = np.vstack([X1, X2])
    y = np.array([1] * n_per_class + [0] * n_per_class)
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def make_xor_extended(n_samples=200, noise=0.2, seed=42):
    """Extended XOR — 2D non-linearly separable with continuous features."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3, 3, (n_samples, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    flip_idx = rng.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    return X, y


# --- All datasets ---

DATASETS = {
    "Linear Separable": {
        "data": make_linear_separable,
        "params": {"n_samples": 200, "n_features": 2, "noise": 0.3},
        "separable": True,
    },
    "Noisy Linear": {
        "data": make_noisy_linear,
        "params": {"n_samples": 200, "n_features": 2, "noise": 1.5},
        "separable": False,
    },
    "Concentric Circles": {
        "data": make_circles,
        "params": {"n_samples": 200, "noise": 0.15},
        "separable": False,
    },
    "High-Dimensional (10D)": {
        "data": make_high_dimensional,
        "params": {"n_samples": 300, "n_features": 10, "noise": 0.8},
        "separable": True,
    },
    "Large Scale (1000)": {
        "data": make_large_scale,
        "params": {"n_samples": 1000, "n_features": 5, "noise": 0.6},
        "separable": True,
    },
    "Extended XOR": {
        "data": make_xor_extended,
        "params": {"n_samples": 200, "noise": 0.2},
        "separable": False,
    },
}

# Architectures to benchmark
ARCHITECTURES = [
    {
        "name": "Perceptron",
        "class": Perceptron,
        "color": "#4a9eff",
    },
    {
        "name": "Adaptive Threshold Neuron",
        "class": AdaptiveThresholdNeuron,
        "color": "#f5a623",
    },
    {
        "name": "Multi-Threshold Perceptron",
        "class": MultiThresholdPerceptron,
        "color": "#b388ff",
    },
    {
        "name": "Multi-Layer Perceptron",
        "class": MultiLayerPerceptron,
        "color": "#26c6da",
    },
    {
        "name": "Pre-Trained Perceptron",
        "class": PreTrainedPerceptron,
        "color": "#ff8a65",
    },
    {
        "name": "Recursive MLP",
        "class": RecursiveMLP,
        "color": "#66bb6a",
    },
]

N_RUNS = 5  # Number of repeated runs for stability measurement


# --- Metrics ---

def compute_metrics(y_true, y_pred):
    """Compute precision, recall, F1 for binary classification."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# --- Benchmark runner ---

def benchmark_single(arch_class, X, y, learning_rate=0.01, n_epochs=200, seed=42):
    """Run a single benchmark and return metrics."""
    model = arch_class(learning_rate=learning_rate, n_epochs=n_epochs)

    start_time = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - start_time

    predictions = model.predict(X)
    metrics = compute_metrics(y, predictions)
    metrics["epochs_to_converge"] = len(model.errors_per_epoch)
    metrics["errors_per_epoch"] = model.errors_per_epoch
    metrics["time_elapsed"] = elapsed
    metrics["converged"] = metrics["accuracy"] == 1.0

    return metrics


def benchmark_with_stability(arch_class, X, y, learning_rate=0.01, n_epochs=200):
    """Run multiple times to measure stability. Returns averaged metrics."""
    all_results = []

    for run in range(N_RUNS):
        result = benchmark_single(arch_class, X, y, learning_rate, n_epochs, seed=42 + run)
        all_results.append(result)

    avg = {
        "accuracy_mean": np.mean([r["accuracy"] for r in all_results]),
        "accuracy_std": np.std([r["accuracy"] for r in all_results]),
        "precision_mean": np.mean([r["precision"] for r in all_results]),
        "precision_std": np.std([r["precision"] for r in all_results]),
        "recall_mean": np.mean([r["recall"] for r in all_results]),
        "recall_std": np.std([r["recall"] for r in all_results]),
        "f1_mean": np.mean([r["f1"] for r in all_results]),
        "f1_std": np.std([r["f1"] for r in all_results]),
        "epochs_mean": np.mean([r["epochs_to_converge"] for r in all_results]),
        "epochs_std": np.std([r["epochs_to_converge"] for r in all_results]),
        "time_mean": np.mean([r["time_elapsed"] for r in all_results]),
        "time_std": np.std([r["time_elapsed"] for r in all_results]),
        "converged_count": sum(1 for r in all_results if r["converged"]),
        "n_runs": N_RUNS,
        "errors_per_epoch": all_results[0]["errors_per_epoch"],
    }

    return avg


def run_all_benchmarks():
    """Run all architectures on all datasets."""
    results = {}

    for ds_name, ds_info in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}  (separable: {ds_info['separable']})")
        print(f"{'='*70}")

        X, y = ds_info["data"](**ds_info["params"])
        n_samples, n_features = X.shape
        print(f"  Samples: {n_samples}  Features: {n_features}")

        results[ds_name] = {}

        for arch in ARCHITECTURES:
            name = arch["name"]
            print(f"\n  -> {name} ({N_RUNS} runs)")

            avg = benchmark_with_stability(arch["class"], X, y)

            results[ds_name][name] = avg

            print(f"    Accuracy:   {avg['accuracy_mean']:.1%} +/- {avg['accuracy_std']:.1%}")
            print(f"    Precision:  {avg['precision_mean']:.1%} +/- {avg['precision_std']:.1%}")
            print(f"    Recall:     {avg['recall_mean']:.1%} +/- {avg['recall_std']:.1%}")
            print(f"    F1:         {avg['f1_mean']:.1%} +/- {avg['f1_std']:.1%}")
            print(f"    Epochs:     {avg['epochs_mean']:.1f} +/- {avg['epochs_std']:.1f}")
            print(f"    Time:       {avg['time_mean']*1000:.2f}ms +/- {avg['time_std']*1000:.2f}ms")
            print(f"    Converged:  {avg['converged_count']}/{avg['n_runs']} runs")

    return results


# --- Chart generation ---

def plot_error_history(results, output_dir):
    """Error-per-epoch comparison charts for each dataset."""
    n_datasets = len(DATASETS)
    cols = 3
    rows = (n_datasets + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for idx, (ds_name, ds_results) in enumerate(results.items()):
        ax = axes[idx]
        for arch in ARCHITECTURES:
            name = arch["name"]
            metrics = ds_results[name]
            errors = metrics["errors_per_epoch"]
            epochs = range(1, len(errors) + 1)
            ax.plot(epochs, errors, label=name, color=arch["color"], linewidth=2)

        ax.set_title(ds_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Misclassifications")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Error History - All Datasets", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = output_dir / "error_history.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {path}")


def plot_metric_comparison(results, output_dir):
    """Bar charts comparing accuracy, F1, epochs, time across datasets."""
    arch_names = [a["name"] for a in ARCHITECTURES]
    ds_names = list(DATASETS.keys())
    n_archs = len(arch_names)
    n_ds = len(ds_names)

    metrics_to_plot = [
        ("accuracy_mean", "Accuracy (%)", True),
        ("f1_mean", "F1 Score (%)", True),
        ("epochs_mean", "Epochs to Converge", False),
        ("time_mean", "Training Time (ms)", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    bar_width = 0.35
    x = np.arange(n_ds)

    for plot_idx, (metric_key, title, as_percent) in enumerate(metrics_to_plot):
        ax = axes[plot_idx]

        for i, arch in enumerate(ARCHITECTURES):
            values = []
            errors = []
            for ds_name in ds_names:
                m = results[ds_name][arch["name"]]
                val = m[metric_key]
                if as_percent:
                    val *= 100
                values.append(val)
                std_key = metric_key.replace("_mean", "_std")
                std_val = m.get(std_key, 0)
                if as_percent:
                    std_val *= 100
                errors.append(std_val)

            ax.bar(x + i * bar_width, values, bar_width,
                   label=arch["name"], color=arch["color"],
                   yerr=errors, capsize=3, edgecolor="white", linewidth=0.5)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x + bar_width * (n_archs - 1) / 2)
        ax.set_xticklabels(ds_names, rotation=25, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Architecture Comparison - All Metrics", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = output_dir / "metric_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_datasets(output_dir):
    """Visualize the 2D datasets."""
    ds_2d = {
        "Linear Separable": DATASETS["Linear Separable"],
        "Noisy Linear": DATASETS["Noisy Linear"],
        "Concentric Circles": DATASETS["Concentric Circles"],
        "Extended XOR": DATASETS["Extended XOR"],
    }
    n = len(ds_2d)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for idx, (name, info) in enumerate(ds_2d.items()):
        ax = axes[idx]
        X, y = info["data"](**info["params"])
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#ff6b6b", label="Class 0", s=15, alpha=0.7)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#4a9eff", label="Class 1", s=15, alpha=0.7)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(ds_2d), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Benchmark Datasets", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = output_dir / "datasets.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_radar(results, output_dir):
    """Radar chart comparing architectures on key metrics (averaged across datasets)."""
    arch_names = [a["name"] for a in ARCHITECTURES]
    metrics_keys = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    avg_values = {}
    for arch in ARCHITECTURES:
        name = arch["name"]
        values = []
        for ds_name in DATASETS:
            m = results[ds_name][name]
            values.append([m[k] for k in metrics_keys])
        avg_values[name] = np.mean(values, axis=0)

    n_metrics = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for arch in ARCHITECTURES:
        name = arch["name"]
        values = avg_values[name].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=arch["color"])
        ax.fill(angles, values, alpha=0.15, color=arch["color"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Average Performance Across All Datasets", fontsize=13, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    path = output_dir / "radar_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# --- Main ---

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  historAI - Architecture Benchmark Suite")
    print("=" * 70)
    print(f"  Architectures: {len(ARCHITECTURES)}")
    print(f"  Datasets:      {len(DATASETS)}")
    print(f"  Runs per test: {N_RUNS}")

    # Run benchmarks
    results = run_all_benchmarks()

    # Generate charts
    output_dir = Path(__file__).parent.parent / "website" / "benchmark-results"
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print("  Generating charts")
    print(f"{'='*70}")

    plot_datasets(output_dir)
    plot_error_history(results, output_dir)
    plot_metric_comparison(results, output_dir)
    plot_radar(results, output_dir)

    # Summary table
    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    header = f"  {'Architecture':<30} {'Dataset':<22} {'Acc':<14} {'F1':<14} {'Epochs':<12} {'Time':<12}"
    print(header)
    print(f"  {'-'*30} {'-'*22} {'-'*14} {'-'*14} {'-'*12} {'-'*12}")

    for ds_name in DATASETS:
        for arch in ARCHITECTURES:
            name = arch["name"]
            m = results[ds_name][name]
            acc = f"{m['accuracy_mean']:.1%}+/-{m['accuracy_std']:.1%}"
            f1 = f"{m['f1_mean']:.1%}+/-{m['f1_std']:.1%}"
            ep = f"{m['epochs_mean']:.0f}+/-{m['epochs_std']:.0f}"
            tm = f"{m['time_mean']*1000:.1f}ms"
            print(f"  {name:<30} {ds_name:<22} {acc:<14} {f1:<14} {ep:<12} {tm:<12}")

    print(f"\n  Charts saved to: {output_dir}/")
    print(f"  - datasets.png")
    print(f"  - error_history.png")
    print(f"  - metric_comparison.png")
    print(f"  - radar_comparison.png")
    print()
