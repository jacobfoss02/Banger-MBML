# Bayesian GP Decomposition of Norwegian Live Births

Additive Gaussian Process model for decomposing and forecasting monthly live births in Norway (1966–2025). Built with Pyro and PyTorch, fitted via Stochastic Variational Inference (SVI).

---

## What it does

The model separates the birth-rate time series into three interpretable components:

- **Trend:** long-term demographic shifts (Matérn-5/2 kernel)
- **Seasonal:** recurring within-year patterns (Periodic × RBF kernel)
- **Short-term:** residual shocks and transient fluctuations (RBF kernel)

A Negative-Binomial likelihood handles count overdispersion. The model is benchmarked against a seasonal naive and a Bayesian Structural Time Series (BSTS) baseline.

---

## Notebooks

| # | Notebook | Description |
|-|-|-|
| 01 | `01_baselines.ipynb` | Data loading & preprocessing, seasonal naive and BSTS baselines |
| 02 | `02_synthetic_recovery.ipynb` | Synthetic data validation — confirms the model can recover known hyperparameters before touching real data |
| 03 | `03_gp_svi.ipynb` | Full model fit on 1966–2024 data, component decomposition, 2025 forecast, and evaluation |

Run them in order. Notebook 01 produces `norway_births_monthly.csv` and `01_eval_baselines.csv` which are consumed by the later notebooks.

---

## Setup

```bash
pip install torch pyro-ppl numpy pandas matplotlib scipy
```

Tested with Python 3.11, Pyro 1.9.1, PyTorch 2.8.

---

## Key results

- **Trend lengthscale:** ~3.5 years — captures medium-term demographic cycles (1969 peak, 1970s decline, 1990s rebound, post-2009 decline)
- **Seasonal amplitude (`sig_seas`):** ~0.038 — stable spring-peak / winter-trough pattern
- **Short-term component:** near-zero amplitude, meaning the real data is smooth and doesn't require a shock component
- **NegBin concentration (`alpha`):** ~615 — low overdispersion relative to a Poisson, consistent with well-recorded vital statistics

---

## Limitations

- SVI with a mean-field (`AutoNormal`) guide ignores posterior correlations between neighbouring time points, so uncertainty in the smooth component functions is likely underestimated
- Full NUTS/HMC would give more accurate posteriors but is out of scope

---

## Data source

Monthly live births in Norway from Statistics Norway (SSB), 1966–2025.