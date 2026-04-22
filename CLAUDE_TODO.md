# CLAUDE TODO — Fix list for notebooks 01, 02, 03
# Work through these in order. Each item names the file, the exact problem seen
# in the plot, the root cause, and the precise fix to apply.

---

## NOTEBOOK 01 — `01_baselines.ipynb`

---

### BUG 1 — `01_data_overview.png` bottom panel is empty/broken
**What the plot shows:** The annual births bar chart (bottom panel) is invisible.
The red dashed train/test split line appears near 1975 instead of 2025.

**Root cause:** The subplots are created with `sharex=True`. The top panel plots
`df["date"]` which is a datetime axis. The bottom panel plots `annual.index`
which is integer years (1966, 1967 … 2025). Because the x-axis is shared as
datetime, matplotlib interprets those integers as days since the epoch — placing
all bars and the axvline near 5–6 years after 1970, i.e. around 1975, which is
off the visible date range. The bars are essentially plotted at the wrong x
coordinates.

**Fix:** In the cell that creates `fig, axes = plt.subplots(2, 1, ...)`:
1. Remove `sharex=True` from the subplots call.
2. Convert `annual.index` to mid-year datetime values for the bar chart:
   ```python
   annual_dates = pd.to_datetime(annual.index.astype(str) + "-07-01")
   axes[1].bar(annual_dates, annual.values, width=pd.Timedelta(days=300),
               color="steelblue", alpha=0.7)
   axes[1].axvline(pd.Timestamp("2025-01-01"), color="firebrick", ls="--", lw=1.2)
   axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
   ```
3. Re-save the figure.

---

### BUG 2 — `01_bsts_prior_predictive.png` — prior explodes to 70,000
**What the plot shows:** The prior 95% PI grows from ~10,000 in 1966 to ~70,000
by 2025. The data line sits in the lower portion throughout but the band grows
enormously over time.

**Root cause:** The local-linear trend is a cumulative random walk over 59 years
on the log-births scale. With `sigma_slope ~ HalfNormal(0.002)`, the slope can
drift by ±0.002/yr × sqrt(59) ≈ ±0.015 per year by 2025, and cumulative level
drift compounds over time. When exponentiated back to counts, this produces
huge uncertainty bands by the late years.

**Fix:** Tighten `sigma_slope` prior in the `bsts_model` function:
```python
# CHANGE:
sigma_slope = pyro.sample("sigma_slope", dist.HalfNormal(0.002))
# TO:
sigma_slope = pyro.sample("sigma_slope", dist.HalfNormal(0.0005))
```
This keeps the slope drift small enough that the level stays near a plausible
range over 59 years. Re-run the prior predictive check cell to verify the band
stays within 0–15,000 across the full time range.

---

### BUG 3 — `01_bsts_posterior_fit.png` — posterior predictive explodes to 1.75 million
**What the plot shows:** The 95% CI reaches 1.75 million births by 2025. The
actual data and posterior median appear as a flat line at the bottom of the plot.

**Root cause:** Same as BUG 2 — the random walk has not been constrained tightly
enough, so even after SVI the posterior over level values drifts to log-scale
values of 10–15, and exp(14) ≈ 1.2 million. The SVI has not fully constrained
the slope noise.

**Fix:**
1. Apply the `sigma_slope` prior tightening from BUG 2 first.
2. After re-running SVI, if the posterior still has a few extreme outlier samples,
   clip on the log scale before exponentiation:
   ```python
   # CHANGE:
   obs_tr = np.exp(post_pred_tr["obs"].numpy())
   # TO:
   obs_tr = np.exp(np.clip(post_pred_tr["obs"].numpy(), 6.5, 10.0))
   # (6.5 ≈ exp → 665 births, 10 ≈ exp → 22,000 births — safe outer range)
   ```
3. Re-save the posterior fit figure.

---

### BUG 4 — `01_forecast_comparison.png` — BSTS forecast panel shows 95% PI up to 1.5 million
**What the plot shows:** Bottom panel (BSTS forecast): 95% PI reaches 1.5
million, actual 2025 data and BSTS median appear flat at 0 on this scale.
Top panel (seasonal naive): ✅ looks correct, no fix needed.

**Root cause:** Same slope-drift issue as BUG 2/3 propagated into the
one-step-ahead forecast. After fixing BUG 2, re-run BUG 3 fix, then re-run the
forecast cell and re-save. The BSTS forecast should then show a reasonable
~3,500–6,000 range.

---

## NOTEBOOK 02 — `02_synthetic_recovery.ipynb`

---

### BUG 5 — `02_prior_predictive.png` — prior too wide, data at bottom of band
**What the plot shows:** The prior 80% PI spans 0–20,000 and 95% PI spans
0–60,000. The synthetic data (~3,000–7,000) sits in the very bottom of the
band. The prior median is around 15,000–20,000 — roughly 4× the actual data.

**Root cause:** The amplitude priors `HalfNormal(0.3)` for `sig_trend` and
`HalfNormal(0.15)` for `sig_seas` and `sig_short` allow the prior GP to produce
very large log-rate deviations. Combined they can easily sum to ±1.5 on log
scale, giving counts up to exp(8.47 + 1.5) ≈ 20,000.

**Fix:** In `additive_gp_model` inside notebook 02, tighten the amplitude priors:
```python
# CHANGE:
sig_trend = pyro.sample("sig_trend", dist.HalfNormal(torch.tensor(0.3)))
sig_seas  = pyro.sample("sig_seas",  dist.HalfNormal(torch.tensor(0.15)))
sig_short = pyro.sample("sig_short", dist.HalfNormal(torch.tensor(0.15)))
# TO:
sig_trend = pyro.sample("sig_trend", dist.HalfNormal(torch.tensor(0.2)))
sig_seas  = pyro.sample("sig_seas",  dist.HalfNormal(torch.tensor(0.05)))
sig_short = pyro.sample("sig_short", dist.HalfNormal(torch.tensor(0.05)))
```
Re-run the prior predictive cell. The 80% PI should now stay mostly within
0–12,000, with the data visibly in the middle of the band.

---

### BUG 6 — `02_component_decomposition.png` — seasonal oscillates around ~0.8 not 0
**What the plot shows:** The seasonal component (panel 3) oscillates around
~0.8 on the log-rate scale instead of 0. The trend (panel 2) appears nearly
flat at ~4.3 instead of near 0. These offsets suggest the intercept `c` is not
being cleanly separated from the component functions.

**Root cause:** In the `post_components` function, the component functions
`f_trend`, `f_seas`, `f_short` are reconstructed as `L @ z` — but the
posterior z values for the synthetic run have absorbed part of the intercept
into the seasonal/trend baseline. This is an identification issue inherent to
additive GPs on short data (10 years). Not a code bug per se, but the plot
is misleading.

**Fix:** Add a zero-centering step when plotting the component decomposition,
so each component is plotted as deviation from its own posterior mean:
```python
# In the component plot cell, add before plotting:
f_tr_centered = f_tr_post - f_tr_post.mean(axis=0, keepdims=True)
f_se_centered = f_se_post - f_se_post.mean(axis=0, keepdims=True)
f_sh_centered = f_sh_post - f_sh_post.mean(axis=0, keepdims=True)
# Then plot f_tr_centered, f_se_centered, f_sh_centered instead.
```
Add a note in the markdown cell: "Components are plotted as deviations from
their posterior mean to remove the intercept ambiguity."

---

### BUG 7 — `02_parameter_recovery.png` — most hyperparameters poorly recovered
**What the plot shows:** The posterior medians are far from the true values for
most parameters. Most notably: `sig_seas` posterior ≈ 0.30 vs true 0.08 (4×
overestimate), `c` posterior ≈ 3.7 vs true 8.47 (completely wrong), `alpha`
posterior ≈ 25 vs true 30.

**Root cause:** This is expected behaviour for mean-field SVI on additive GPs
with identification ambiguity — the components can trade off amplitude against
each other. The `c` intercept being at 3.7 instead of 8.47 suggests the guide
has absorbed part of the intercept into the z components. After applying BUG 5
(tighter amplitude priors), recovery should improve.

**Fix:**
1. Apply BUG 5 fix first, then re-run the full notebook B training.
2. In the summary markdown (cell 9), add an honest note: "Recovery of `c` and
   amplitude hyperparameters is imprecise under mean-field SVI due to component
   identification ambiguity — the prior on `sig_seas` was tightened to partially
   address this. Lengthscale parameters are better identified."
3. No code change needed beyond the prior tightening. Re-save the figure.

---

## NOTEBOOK 03 — `03_gp_svi.ipynb`

This notebook has the most serious issues. The SVI has converged to a bad local
minimum. The root cause is that the amplitude priors are too loose AND the
whitened parameterization with 2160 latent dimensions creates a very difficult
optimization landscape for AutoNormal. The fixes below address both.

---

### BUG 8 — `03_prior_predictive.png` — prior too wide, data at bottom of band
**What the plot shows:** Identical to BUG 5. The prior 95% PI spans 0–60,000;
real data (~4,000–6,500) is near the bottom.

**Fix:** In `additive_gp_model` in notebook 03, tighten amplitude priors:
```python
# CHANGE:
sig_trend = pyro.sample("sig_trend", dist.HalfNormal(torch.tensor(0.3)))
sig_seas  = pyro.sample("sig_seas",  dist.HalfNormal(torch.tensor(0.15)))
sig_short = pyro.sample("sig_short", dist.HalfNormal(torch.tensor(0.15)))
# TO:
sig_trend = pyro.sample("sig_trend", dist.HalfNormal(torch.tensor(0.2)))
sig_seas  = pyro.sample("sig_seas",  dist.HalfNormal(torch.tensor(0.05)))
sig_short = pyro.sample("sig_short", dist.HalfNormal(torch.tensor(0.05)))
```
Also tighten `ell_short` to prevent it from drifting to multi-year values:
```python
# CHANGE:
ell_short = pyro.sample("ell_short", dist.InverseGamma(torch.tensor(3.0), torch.tensor(0.5)))
# TO:
ell_short = pyro.sample("ell_short", dist.InverseGamma(torch.tensor(5.0), torch.tensor(0.5)))
# InvGamma(5, 0.5) → mean = 0.5/4 = 0.125 years ≈ 1.5 months, keeps short-term truly short
```
Re-run the prior predictive cell after this change to verify the band is tighter.

---

### BUG 9 — `03_svi_convergence.png` — initial spike dominates, convergence invisible
**What the plot shows:** The y-axis (−ELBO) is scaled to 2.5 million due to the
very first step's spike. After step ~100 the curve appears flat near 0 and the
actual convergence behaviour (including the LR drop effect at step 4000) is
completely invisible.

**Fix:** After the training loop, change the convergence plot cell to use a
log y-axis AND to zoom in separately:
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 3))

# Left: full run on log scale
axes[0].plot(losses, lw=0.5, alpha=0.6, color="steelblue")
axes[0].set_yscale("log")
axes[0].axvline(N_STEPS // 2, color="grey", ls="--", lw=0.8, label="LR drop")
axes[0].set_xlabel("SVI step")
axes[0].set_ylabel("-ELBO (log scale)")
axes[0].set_title("Full training run")
axes[0].legend()

# Right: zoom on steps 500–8000 (skip initial spike)
smooth = pd.Series(losses).rolling(200, center=True).mean()
axes[1].plot(range(500, N_STEPS), losses[500:], lw=0.5, alpha=0.5, color="steelblue")
axes[1].plot(range(500, N_STEPS), smooth[500:], lw=1.5, color="firebrick",
             label="200-step rolling mean")
axes[1].axvline(N_STEPS // 2, color="grey", ls="--", lw=0.8, label="LR drop")
axes[1].set_xlabel("SVI step")
axes[1].set_ylabel("-ELBO")
axes[1].set_title("Convergence (steps 500+)")
axes[1].legend()

plt.tight_layout()
plt.savefig(PROJ / "03_svi_convergence.png", dpi=120, bbox_inches="tight")
plt.show()
```

---

### BUG 10 — `03_component_decomposition.png` — seasonal amplitude wildly wrong, trend monotonic
**What the plot shows:**
- Seasonal component oscillates at ±3 on log-rate scale. Real seasonal variation
  is ±0.08. This is 40× too large.
- Trend is monotonically declining from 1966 to 2025 — does not show the 1969
  peak, the 1983 trough, or the 1990s rebound.
- Short-term component is slow-varying (absorbing decade-scale variation).

**Root cause (confirmed by BUG 11):** The SVI converged to a local minimum
where `sig_seas ≈ 0.73` and `ell_short ≈ 3.8 years`. With a huge seasonal
amplitude and a long short-term lengthscale, the "seasonal" component is
explaining everything and the other components are poorly identified.

**Fix:** This plot will fix itself once BUG 8 (prior tightening) is applied
and the model is re-run. After re-running, verify:
- Seasonal oscillations are in the ±0.05–0.15 range on log-rate
- Trend shows visible non-monotonic structure (peak 1969 → decline → rebound)
- Short-term component shows a dip around 2020

No separate code change needed here — it's a downstream consequence of BUG 8.
Add a dashed horizontal line at y=0 on all three component panels if not already
present, to make deviations easier to read:
```python
for ax in axes[1:]:
    ax.axhline(0, color="grey", lw=0.6, ls=":", zorder=0)
```

---

### BUG 11 — `03_hyperparameter_posteriors.png` — all hyperparameters converged to wrong values
**What the plot shows:**
- `ell_trend` posterior ≈ 250–270 years (should be 5–30 years)
- `ell_s` posterior ≈ 750–2000 years (should be 10–40 years)
- `ell_short` posterior ≈ 3.4–4.2 years (should be 0.1–1 year)
- `sig_seas` posterior ≈ 0.727–0.732 (should be ~0.05–0.10)
- All posteriors are extremely narrow (SVI guide collapsed to a point mass)

**Root cause:** SVI converged to a bad local optimum. The narrow posterior
width confirms the AutoNormal guide has "over-fitted" to a single bad solution
with essentially zero variance — a known failure mode of mean-field SVI on
high-dimensional whitened GP models.

**Fix:** After applying BUG 8, also add a 3-phase learning rate schedule to
help escape local minima:
```python
# CHANGE the SVI training loop to:
pyro.clear_param_store()
torch.manual_seed(42)
guide = autoguide.AutoNormal(additive_gp_model)

N_STEPS = 10000
losses  = []
t0 = time.time()

# Phase 1: warm-up with low LR
svi = SVI(additive_gp_model, guide, ClippedAdam({"lr": 0.005, "clip_norm": 5.0}), loss=Trace_ELBO())
for step in range(2000):
    loss = svi.step(t_all, y_train, N_TRAIN)
    losses.append(loss)
    if step % 500 == 0:
        print(f"[warm-up] step {step:5d}  ELBO = {-loss:10.1f}")

# Phase 2: main training with higher LR
svi = SVI(additive_gp_model, guide, ClippedAdam({"lr": 0.02, "clip_norm": 10.0}), loss=Trace_ELBO())
for step in range(2000, 8000):
    loss = svi.step(t_all, y_train, N_TRAIN)
    losses.append(loss)
    if step % 1000 == 0:
        print(f"[main]    step {step:5d}  ELBO = {-loss:10.1f}")

# Phase 3: fine-tune
svi = SVI(additive_gp_model, guide, ClippedAdam({"lr": 0.003, "clip_norm": 5.0}), loss=Trace_ELBO())
for step in range(8000, N_STEPS):
    loss = svi.step(t_all, y_train, N_TRAIN)
    losses.append(loss)
    if step % 500 == 0:
        elapsed = time.time() - t0
        print(f"[fine]    step {step:5d}  ELBO = {-loss:10.1f}  ({elapsed:.0f}s)")

print(f"\nTotal SVI time: {time.time()-t0:.0f}s")
```
After re-running, the hyperparameter posteriors should show:
- `ell_trend` in the range 5–30
- `sig_seas` in the range 0.04–0.12
- `ell_short` below 1.0

---

### BUG 12 — `03_covid_shock.png` — COVID shock estimated at only -0.9%, indistinguishable from zero
**What the plot shows:** The posterior median COVID-2020 short-term effect is
-0.9% with a very wide distribution spanning -8% to +5%. This is essentially
noise — there's barely any signal of a 2020 dip.

**Root cause:** With `ell_short ≈ 3.8 years` (from the bad local minimum),
the short-term component is far too smooth to capture a single-year shock.
It cannot resolve a 2020 dip — the 3.8-year lengthscale smooths it away.

**Fix:** This will improve once BUG 8 and BUG 11 are applied (ell_short forced
below 1 year by the tighter prior). After re-running, the COVID plot should show
a clearly negative posterior (expected around -5% to -10% based on data).
No separate code change needed.

---

### BUG 13 — `03_forecast_2025.png` — GP median too flat, misses seasonal peak
**What the plot shows:** GP median peaks at ~4,600 in May, but actual 2025
births peak at ~5,200 in August-September. The forecast misses the seasonal
amplitude and the timing of the peak. Actuals are within the 95% PI but at
the very top edge through summer.

**Root cause:** Consequence of the blown-out seasonal amplitude in the bad
local minimum — when the seasonal is at ±3 log-rate, the forecast samples
span an enormous range and the median averages out to something flat. After
fixing BUG 8/11, the seasonal should be well-identified and the 2025 forecast
should track the actual seasonal shape more closely.

**Fix:** This should resolve after BUG 8 and BUG 11 fixes and re-running.
No separate code change needed. After re-running, verify:
- GP median shows a clear spring/summer peak matching the actual seasonal shape
- 80% PI width is roughly ±300–500 births (currently it's ±1000+)

---

## EXECUTION ORDER

Fix and re-run in this exact order to avoid cascading issues:

1. Fix `01_baselines.ipynb`:
   - Apply BUG 1 (bottom panel axis fix) — re-run data overview cell only
   - Apply BUG 2 (sigma_slope prior) — re-run from the bsts_model cell onward
   - BUG 3 and BUG 4 should resolve automatically after BUG 2

2. Fix `02_synthetic_recovery.ipynb`:
   - Apply BUG 5 (amplitude priors) in additive_gp_model
   - Re-run from the prior predictive cell onward (Step B only, Step A is fine)
   - Apply BUG 6 (zero-centering in component plot)
   - BUG 7 should partially resolve after BUG 5

3. Fix `03_gp_svi.ipynb`:
   - Apply BUG 8 (amplitude + ell_short priors) in additive_gp_model
   - Apply BUG 9 (convergence plot fix)
   - Apply BUG 11 (3-phase LR schedule)
   - Re-run from the prior predictive cell all the way through
   - BUG 10, 12, 13 should resolve as downstream consequences

---

## QUICK SANITY CHECKS AFTER RE-RUNNING

After completing all fixes, verify these conditions before writing the report:

- [ ] `01_data_overview.png`: annual bar chart visible with bars from 1966–2025, dashed line at 2025
- [ ] `01_bsts_posterior_fit.png`: 95% CI stays within 2,000–15,000 births
- [ ] `01_forecast_comparison.png`: both panels on same ~3,000–7,000 scale
- [ ] `03_prior_predictive.png`: data visibly in middle third of prior 80% PI
- [ ] `03_svi_convergence.png`: post-spike convergence clearly visible
- [ ] `03_component_decomposition.png`: seasonal oscillations ≤ ±0.15 log-rate
- [ ] `03_hyperparameter_posteriors.png`: ell_trend 5–30, sig_seas 0.04–0.12, ell_short < 1.0
- [ ] `03_covid_shock.png`: posterior median clearly negative (around -5% to -10%)
- [ ] `03_forecast_2025.png`: GP median tracks seasonal shape, PI width < ±600 births
