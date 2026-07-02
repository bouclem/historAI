# historAI

The history of artificial intelligence, remade from scratch.

## What is this?

historAI is a project that tells the real history of AI — from the first neuron model in 1943 to modern architectures. At certain points in the timeline, we introduce **divergence points**: new architectures that could have existed at that moment in history, fully implemented from scratch.

Each divergence is classified based on benchmark testing:
- **Blue** — Real history (as it actually happened)
- **Red** — Divergence that performed worse than the real-history equivalent
- **Green** — Divergence that performed better than the real-history equivalent
- **Amber** — Divergence that hasn't been tested yet

## Structure

```
historAI/
├── src/                    # AI architecture implementations (Python)
│   ├── perceptron/         # Rosenblatt's perceptron (1957)
│   ├── adaptive-threshold-neuron/  # Divergence architecture (1948)
│   ├── benchmark.py        # Benchmark suite — run with `python src/benchmark.py`
│   └── benchmark_results/  # Generated PNG charts
├── website/                # Website (HTML/CSS/JS)
│   ├── index.html          # Timeline homepage
│   ├── explore.html        # Architecture listing
│   ├── benchmarks.html     # Public benchmark results
│   ├── benchmark-results/  # PNG charts for website
│   ├── pages/              # Architecture & event pages
│   ├── architecture/       # .md files with code (loaded by JS)
│   ├── styles.css
│   └── script.js
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies (numpy, matplotlib)
└── .gitignore
```

## Getting Started

### Run benchmarks

```bash
pip install -r requirements.txt
python src/benchmark.py
```

Results are saved to `src/benchmark_results/` as PNG files.

### Preview website

```bash
cd website
python -m http.server 8000
```

Open `http://localhost:8000` in your browser.

### Deploy

The website is deployed on Vercel. Set the output directory to `website/`.

## Architectures

| Architecture | Year | Type |
|---|---|---|
| Perceptron | 1957 | Real history |
| Adaptive Threshold Neuron | 1948 | Divergence (worse) |

## License

MIT
