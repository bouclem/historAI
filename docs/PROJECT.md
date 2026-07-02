# Project Overview

## What

historAI is an educational project that tells the history of artificial intelligence while implementing each architecture from scratch. At certain points in the timeline, we introduce **divergence architectures** — new models that could have existed at that moment in history, which we design, implement, and benchmark.

## Why

To understand AI deeply by rebuilding it from nothing. To explore "what if" paths in AI history — could alternative architectures have changed the course of the field?

## Architecture

### Source code (`src/`)
Each architecture lives in its own folder: `src/[name]/[name].py`. A benchmark suite (`src/benchmark.py`) tests all architectures on 6 datasets and generates comparison charts.

### Website (`website/`)
Static HTML/CSS/JS site. No framework. Pages load architecture code from `.md` files in `website/architecture/` via JavaScript `fetch()`. Deployed on Vercel.

### Design system
- Dark modern theme with OKLCH colors
- Blue = real history, Red = worse divergence, Green = better divergence, Amber = untested
- Fonts: Space Grotesk (display), Inter (body), JetBrains Mono (code)

## Direction

1. Implement more real architectures (MLP, Adaline, Hopfield, CNNs, etc.)
2. Design and test more divergence architectures
3. Expand the timeline to cover modern AI (transformers, diffusion models)
4. Add interactive visualizations for decision boundaries
