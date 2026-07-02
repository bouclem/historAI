# Lessons Learned

## Python
- **Hyphenated folder names can't be imported** — `adaptive-threshold-neuron` is not a valid Python package name. Use `importlib.util.spec_from_file_location()` to import from hyphenated paths.
- **matplotlib Agg backend** — Always set `matplotlib.use("Agg")` before importing `pyplot` when generating charts without a display. Prevents GUI errors on headless systems.

## Benchmarking
- **Logic gates are too simple** — AND/OR/XOR with 4 samples don't stress-test architectures. Use synthetic datasets with 200-1000+ samples, varying noise, and non-linear patterns.
- **Run multiple times** — Single runs can be misleading. 5 runs with mean ± std gives stability information.
- **Same accuracy != same architecture** — The Adaptive Threshold Neuron matched the Perceptron's accuracy exactly but was 1.6x slower, proving the threshold adaptation provides no benefit over bias.

## Web Development
- **fetch() needs a server** — Loading `.md` files via `fetch()` doesn't work with `file://` protocol. Always use `python -m http.server` for local testing.
- **Relative paths matter** — When moving pages between directories (e.g., `pages/` to root), all relative links, CSS, and JS paths must be updated.
