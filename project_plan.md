# MBML Project Plan — Bayesian decomposition of Norwegian live births, 1966–2025

*Course: 42186 Model-Based Machine Learning, DTU, Spring 2026*
*Author: Jacob — revised plan, 21 April 2026*

---

## 1. Elevator pitch

Adapt the PyMC *Baby Births Modelling with HSGPs* notebook (Vehtari's birthdays case study) to **Norwegian monthly live birth counts 1966–2025** (SSB table 05531, 720 observations), implementing everything from scratch in **Pyro**. We build a Bayesian additive-GP decomposition of the time series into a long-term trend, an annual seasonal shape that evolves slowly over decades, and structural shocks (1970s fertility transition, post-2009 decline, COVID-2020), and use it for (i) interpretation of *why* Norwegian births change over time and (ii) short-horizon forecasting with calibrated uncertainty.

**Project type (from project description, slide 5–8):** *data-driven* (rich dataset → research question → PGM).

---

## 2. Research question

> Can we separate the variation in Norwegian monthly live births 1966–2025 into a small number of interpretable, independently-inferred Bayesian components — a long-term demographic trend, a slowly-evolving annual seasonal pattern, and short-term residual fluctuations — and use the resulting generative model to forecast 2026 births with honest uncertainty and to quantify the size and recovery time of the 2020 COVID shock?

Two concrete sub-questions grade the project:

1. **Interpretation.** How much of the variance in monthly births is explained by the trend vs. the seasonal vs. short-term components, and does the seasonal shape in 2020–2025 differ from 1966–1975?
2. **Forecast.** Does a Bayesian GP decomposition give better-calibrated 12-month-ahead forecasts than a seasonal-naive baseline and a Bayesian structural time-series (BSTS) baseline?

---

## 3. Dataset

Source: Statistics Norway (SSB), table 05531, *"Levendefødte, etter måned og år"*. Local file: `05531_20260420-195045.csv`.

After reshaping: 720 monthly observations, `year ∈ {1966,…,2025}`, `month ∈ {1,…,12}`, `births ∈ [3420, 6390]` (integer counts, no missing values).

![EDA](computer:///sessions/cool-charming-franklin/mnt/MBML/eda_overview.png)

What the EDA confirms (motivates the model):

- **Long-term trend is non-monotonic.** Peak 1969 (~67.7k/yr), sharp 1970s fertility-transition decline, trough 1983 (~49.9k), partial rebound 1990–2009, then a long decline back to ~50k, with COVID-2020 visible and a mild 2024–2025 recovery. A smooth non-parametric trend is appropriate.
- **Strong annual seasonality.** Ratio plot shows a reproducible shape: peak Mar–May (≈1.06–1.08× yearly mean), trough Nov–Dec and Feb (≈0.91–0.95×), with modest cross-year variation.
- **Potential seasonal drift.** The standard-deviation band on the seasonal ratio suggests the shape is *not* perfectly stationary — worth modelling the seasonal as `periodic × slow-trend`, as in the PyMC notebook.
- **No weekly effect, no day-of-year effect.** We only have monthly aggregates, so the weekly/day-of-year/floating-holiday components in the PyMC example are *not* modelled. This simplifies the port to Pyro considerably.

Train / test split: 1966-01 to 2024-12 for training; 2025-01 to 2025-12 held out for forecast evaluation (a full year of already-observed ground truth).

---

## 4. Generative model (PGM)

Let $t_i$ be the time of observation $i$ in decimal years (e.g., 1966.0, 1966.0833, …, 2025.9166), $m_i \in \{1,\dots,12\}$ the month, and $y_i$ the live-birth count.

Decompose the *log-rate* as an additive sum of three latent Gaussian-process components plus a residual:

$$
\log \mu_i \;=\; \underbrace{f_\mathrm{trend}(t_i)}_{\text{slow demographic}} \;+\; \underbrace{f_\mathrm{seas}(m_i,\,t_i)}_{\text{annual pattern, drifting}} \;+\; \underbrace{f_\mathrm{short}(t_i)}_{\text{residual shocks (COVID, etc.)}} \;+\; c,
$$

with

- **Trend** $f_\mathrm{trend} \sim \mathcal{GP}(0, k_\mathrm{trend})$ using a Matérn-5/2 kernel with a long lengthscale prior (years-scale): $\ell_\mathrm{trend} \sim \mathrm{InvGamma}(a, b)$ calibrated so 95% mass covers ~5–30 years; amplitude $\sigma_\mathrm{trend} \sim \mathrm{HalfNormal}(0.2)$ (operating on log-rate, so 0.2 ≈ ±20%).
- **Seasonal** modelled as the *product* of a periodic kernel (period = 1 year) and a long-lengthscale warping component, so the 12-month shape evolves slowly:
  $$ k_\mathrm{seas}(t,t') = k_\mathrm{periodic}(t,t';\,p=1,\ell_p) \cdot k_\mathrm{RBF}(t,t';\ell_s) $$
  with $\ell_p \sim \mathrm{HalfNormal}(1)$, $\ell_s \sim \mathrm{InvGamma}$ covering 10–40 years, amplitude $\sigma_\mathrm{seas} \sim \mathrm{HalfNormal}(0.1)$.
- **Short-term** $f_\mathrm{short} \sim \mathcal{GP}(0, k_\mathrm{RBF}(\ell_\mathrm{short}))$ with $\ell_\mathrm{short}$ on months-scale (0.1–1 year) to absorb COVID-type shocks.
- **Intercept** $c \sim \mathcal{N}(\log \bar y, 1)$.
- **Likelihood.** Count data → **Negative-Binomial**:
  $$ y_i \mid \mu_i, \alpha \sim \mathrm{NegBin}(\mathrm{mean}=\mu_i,\ \mathrm{dispersion}=\alpha),\quad \alpha \sim \mathrm{Gamma}(2,0.1). $$
  A Gaussian on log counts is an OK first approximation but a Negative-Binomial is cleaner for counts and appears in the Bayesian-Spatial-Count-Models slide of the project description (slide 24) — it's in the same family the instructors expect.

### Plate diagram

```
        ℓ_trend  σ_trend          ℓ_p  ℓ_s  σ_seas        ℓ_short  σ_short        c           α
           │        │              │    │    │               │        │           │           │
           └──►  f_trend(t)        └────┴────┴──► f_seas(m,t) └────────┴──► f_short(t)         │
                     │                              │                         │               │
                     └──────────────┬───────────────┴──────────── + ──────────┘               │
                                    ▼                                                         │
                                log μ_i  ──────────────────────►  y_i  ~  NegBin(μ_i, α)  ◄───┘
                                                                     (i = 1..720)
```

Observed nodes are $y_i$ (shaded). All other nodes are latent (priors listed above). This is exactly the style requested in the project advice slides: priors only on unobserved variables, information flows through the latent sum into the observed count.

---

## 5. Pyro implementation strategy

The PyMC notebook uses **HSGP** (Hilbert-Space Gaussian Process) approximations to make the GP tractable for ~7000 daily data points. With only **720 monthly** data points, *exact GPs are feasible* — the covariance matrix is 720×720, factorisable in milliseconds. HSGP is therefore not needed for inference speed and is out of scope for this project; it is noted as a natural extension in the report conclusion.

### Plan

1. **Baselines.** Seasonal-naive and BSTS (local-linear-trend + fixed monthly seasonals), inferred with SVI in Pyro. Establishes the comparison target. *(done — `01_baselines.ipynb`)*

2. **Synthetic recovery.** Sample from the additive-GP prior with known hyperparameters; verify that SVI recovers them from synthetic counts. Prior predictive check, ELBO convergence, component decomposition on synthetic data. *(done — `02_synthetic_recovery.ipynb`)*

3. **Full model on real data — SVI.** Fit the exact additive GP to 1966–2024 training data with `AutoNormal` (mean-field) guide and `ClippedAdam`. Produce component decomposition plot, variance breakdown, COVID-2020 shock quantification, 2025 forecast, and evaluation table vs. baselines. Consolidated evaluation table added as a final section of this same notebook. *(done — `03_gp_svi.ipynb`)*


**Inference note.** SVI with `AutoNormal` is the primary inference method throughout. Mean-field posteriors underestimate uncertainty in the component functions due to ignored cross-time correlations; this is acknowledged as a limitation in the report. Full NUTS is noted as future work and is not a deliverable.

### Baselines (slide 17: "always try a simple baseline")

1. **Seasonal-naive.** $\hat y_{t} = y_{t-12}$.
2. **Bayesian structural time-series.** Local-linear-trend + fixed seasonal (monthly dummies), implemented in Pyro — much simpler than the GP model, good sanity check.
3. **No-trend GP.** Just seasonal + short-term, to show the trend component earns its keep (optional, can be a brief ablation in the report).

---

## 6. Evaluation

- **Posterior predictive checks.** Plot 95% credible bands over the 1966–2025 fit; visual residual diagnostics; compare empirical vs. posterior-predictive monthly mean per month-of-year.
- **Held-out 2025.** Compute log-predictive-density, CRPS, and 80/95% coverage on the 12 held-out months. Compare baselines vs. GP decomposition.
- **Component decomposition plot** (the money plot, exactly as in the PyMC notebook). Four panels: data, posterior mean ± 95% of trend, seasonal, short-term.
- **Variance-decomposition table.** Fraction of total variance on log-rate explained by each component.
- **COVID-shock quantification.** Report posterior over the 2020–2021 short-term component integrated across the year, with credible interval.
- **Prior-predictive check.** Sample from prior — do the generated curves look like births data? This is an important sanity step that the project description calls out.

---

## 7. Risks & open decisions

1. **Is monthly resolution rich enough for a GP-decomposition story?** Yes — 720 points, clear components, and with Bayesian uncertainty it's the *quality of the decomposition and the forecast*, not the volume, that matters.
2. **Is SVI sufficient as the only inference method?** Yes, with caveats. Mean-field AutoNormal gives approximate marginals; uncertainty in smooth components like the trend is likely underestimated. This is acknowledged in the report as a limitation and NUTS is mentioned as the natural next step. The decomposition results and forecast ranking are not expected to change qualitatively.
3. **Likelihood: NegBin vs. Gaussian on log y.** NegBin — cleaner story, honest handling of counts, and it echoes the project-description spatial-count-model slide (slide 24). Already implemented.
4. **Extra covariates?** Could add unemployment, COVID-lockdown-stringency, or parental-leave policy indicators as extensions. Out of scope; good "future work" mention.
5. **Discrete latents?** Slide 14 warns that discrete latents need special treatment in Pyro. Our design has *no* discrete latents, by choice.

---

## 8. One-paragraph abstract (draft for the report)

> Norwegian live-birth counts over 1966–2025 reflect a complex interplay of demographic trends, stable annual seasonality, and occasional shocks. We build a Bayesian additive-GP decomposition in Pyro, with a slow Matérn trend, a periodic × slowly-warping seasonal, and an RBF short-term component, observed through a Negative-Binomial likelihood on 720 monthly counts. Inference is performed with SVI using a mean-field guide; the model is first validated on synthetic data before fitting to real observations. The fitted model cleanly separates the 1970s fertility transition, a slowly-drifting seasonal shape, and a quantified COVID-2020 dip, and its 12-month-ahead forecast for 2025 outperforms a seasonal-naive and a Bayesian structural-time-series baseline in log-predictive density, CRPS, and coverage.

---

*All code notebooks are complete. Remaining work: consolidated evaluation table (minor addition to `03_gp_svi.ipynb`) and the written report.*
