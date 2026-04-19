# Project Notes & Improvement Plan

Reconstruction notes for the Non-DROS v2 smart-irrigation prototype — what the code currently does, how faithful it is to the Preite & Vignali (2024) paper, and the honest list of things that must change before this can be called a serious system.

> TL;DR: The pipeline runs end-to-end, matches the paper's recipe closely, and reports ~99% accuracy plus ~big water/energy savings. **Those numbers are almost meaningless in the current setup** because the training target is computed deterministically from two of the six input features. The model is mostly re-learning a threshold rule. This is the paper's flaw, inherited faithfully. See §5 for how to fix it.

---

## 1. What this project is (and isn't)

**What it is.** A supervised multi-class classifier that looks at one instant of sensor readings + one weather sample and outputs one of four irrigation actions:

| Class | Label | Meaning |
|-------|-------|---------|
| 0 | OFF | Soil moisture above upper confidence limit — close valve |
| 1 | ON | Soil moisture below lower confidence limit, no rain coming — open valve |
| 2 | No adjustment | Soil moisture inside the confidence interval — do nothing |
| 3 | Alert | Critically dry AND rain coming — page the farmer |

**What it isn't.**
- **Not reinforcement learning.** No agent, no reward signal, no environment-in-the-loop training. It's a static classifier.
- **Not a time-series forecaster.** No LSTM, no transformer, no autoregression. Each decision is made from a single 10-minute snapshot. "Three-day forecast" here means "we include *right now's* rainfall/ET forecast value as a feature" — we never consume the full 3-day horizon.
- **Not predictive in the future-state sense.** It predicts the *action*, not tomorrow's soil moisture.

**How the paper frames it.** The paper calls this a "predictive algorithm-based irrigation management system", but the prediction is of the control class, not of soil state. The architecture is a rule-based policy encoded into an MLP. The MLP is a stand-in for a decision tree that happens to have been trained on labeled snapshots.

---

## 2. File-by-file walkthrough

### `src/01_weather_api_fetcher.py`
- Pulls hourly weather from Open-Meteo's **archive** endpoint (not forecast — we're reconstructing the paper's season, July 28 – Sept 3 2023).
- Coords: `44.1125, 10.411` (paper's living lab).
- Saves: `data/open_meteo_forecast_data.csv` — columns `datetime`, rainfall, RH, ET0.

Status: fine. Only note: the paper reports "three-day forecast" as a feature, but the archive returns the hour's *actual* value. So in training we are pretending the forecast was perfect. Same as the paper — they did this too.

### `src/02_data_preprocessing.py`
- Loads three raw CSVs from the Stuard living-lab dataset (environmental, soil, water meter).
- Restricts everything to `2023-07-28 → 2023-09-03`.
- Rounds every sample to the nearest 10-min bin.
- Builds a per-irrigation-line 10-min grid and merges the three streams.
- Forward-fills gaps within each line; drops rows that are still missing.
- Computes `Irrigation (ON/OFF)` as `volume_diff > 0` (positive jumps in cumulative volume meter).
- Saves: `data/merged_sensor_data.csv`.

Status: solid. One subtle thing: `Irrigation (ON/OFF)` here is the **real** irrigation history from the water meter, not the target the model is trained on — that comes later and is synthetic.

### `src/03_soil_capacity_calculator.py`
This is the most important file. It does two things:
1. **Estimates soil capacity μ and σ** by looking at moisture *derivative* in slow-drainage regions (the flat tail after an irrigation or rain event). Uses `RECESSION_LOOKBACK_STEPS=12` (2 hours) and `MAX_SLOW_DRAINAGE_DROP=-0.5`. These constants are our own — the paper didn't publish numbers.
2. **Synthesizes the training labels** (`Irrigation_Decision`) row-by-row from the policy logic:

```
default                                              → 2 (No adj)
moisture > upper_limit_standard (μ + σ/2)            → 0 (OFF)
moisture < lower_limit_standard (μ − σ/2) and not rain → 1 (ON)
moisture < lower_limit_standard and rain and not critical → 0 (OFF)
moisture < lower_limit_standard and rain and ≤ critical (μ − σ) → 3 (ALERT)
```

- Saves: `data/processed_dataset.csv` (6 features + target, for training), `data/modeling_dataset_full.csv` (same + datetime + line + raw irrigation, for sim), `data/soil_capacity_metadata.json`.

Current metadata values:
```
mu=25.25%, sigma=4.31%, lower=23.10%, upper=27.41%, critical=20.94%
class counts: {0: 2689, 1: 8641, 2: 4992, 3: 91}
```
Paper counts: `{0: 3467, 1: 8875, 2: 3780, 3: 65}` — same order of magnitude, different exact split because our μ/σ differ slightly.

**This is where label leakage is baked in.** See §3.

### `src/04_train_neural_network.py`
- MLP via scikit-learn: `hidden_layer_sizes=(6,12,6)`, `activation="tanh"`, `solver="adam"`, `alpha=0.01`, `max_iter=5000`. Exactly the paper spec.
- 70/30 stratified split. No normalization (paper skipped it too).
- Saves `{model, feature_columns, notes}` bundle to `models/mlp_irrigation_model.pkl`.
- Prints accuracy, confusion matrix, full classification report.

### `src/05_monte_carlo_simulation.py`
Two versions here — the active one (top) is simple, the commented-out one at the bottom is a much more careful stateful simulator that was abandoned.

**Active version**:
- Runs 1000 simulated "seasons" of 10,080 samples each (70 days × 144 steps).
- Samples soil moisture from `Normal(μ, σ/3)`, clipped to observed min/max.
- Holds every other feature constant at its training-set mean.
- Feeds into the classifier. Counts class=1 outputs. Each class=1 = 50 L water + 0.667 kWh.
- Compares to IRRIFRAME baseline (324.5 mm × 132 m² = 42,834 L, 571 kWh).
- Reports savings ± 2σ as a 95% CI.

**Why this is misleading**: Soil moisture is being sampled *right around μ*, which is the center of the confidence band where the policy's default is class 2 (No adj) or class 0 (OFF). The classifier almost never outputs ON, because iid samples from `N(μ, σ/3)` almost never dip below `μ − σ/2`. So "water savings" is dominated by the sampling distribution, not by any intelligence in the model. This is how the paper gets up to 57% energy savings — they stack the deck.

### `main_controller.py`
- Mock hardware. Reads sensors, fetches live Open-Meteo, loads the model, predicts, prints the command.
- **Bug:** `joblib.load('models/mlp_irrigation_model.pkl')` returns the bundle dict, but the code calls `model.predict(df_live)` directly. That will crash with `AttributeError: 'dict' object has no attribute 'predict'`. Fix: `bundle = joblib.load(...); model = bundle["model"]; feature_columns = bundle["feature_columns"]; df_live = df_live[feature_columns]`.
- **Feature-order bug (latent):** even once unboxed, `df_live` columns may not match the order the model was trained on. Reordering via `feature_columns` fixes both at once.
- Missing feature: `Weather Forecast Environmental humidity [RH %]` was in the weather CSV but `read_live_sensors()` doesn't include it — this is OK because it's also not in `FEATURE_COLUMNS`. Just worth knowing.

### Data files
- `stuard_*.csv` — raw IoT data from Mendeley (see `data/README.TXT`). Source: https://data.mendeley.com/datasets/35wh56287y/2.
- `indicators.csv` — daily GDD / Heat Units. **Not currently used.** Paper doesn't use them either, but they're great candidates for feature engineering.
- `desc_tree.py` — orphaned exploratory script that tries to classify which irrigation line a row belongs to. Not wired into the pipeline. Harmless; consider deleting.

---

## 3. The central problem: label leakage

The training target `Irrigation_Decision` is computed as a **deterministic function** of two of the six input features:

```
target = f(Soil Moisture [RH%], Weather Forecast Rainfall [mm])
```

The other four features (`Soil Temperature`, `Environmental Temperature`, `Environmental Humidity`, `Evapotranspiration`) appear in X but are *completely absent from y*. The network is being asked to learn a fixed piecewise-linear decision rule over two of its six inputs.

Consequences:
- **Accuracy is tautological.** A decision tree of depth 3 can hit 100% here. Any reasonable classifier will. ~99% is unimpressive.
- **The remaining four features are noise** as far as training is concerned. The model will just learn to ignore them. Feature-importance analysis would confirm this.
- **No generalization claim is valid.** If soil capacity shifts (different soil, different crop stage), the synthetic labels shift too — the model doesn't learn anything transferable.
- **The Monte Carlo savings are a function of the labeling rule**, not the model. You could replace the MLP with the literal if/else and get the same (or better) result.

This problem **is inherited from the paper** — Section 2.3.2 describes exactly this process. The paper is presenting rule-following with extra steps. But inheriting that flaw doesn't absolve the project of it.

---

## 4. How faithful is this to the paper?

High faithfulness where it matters:

| Paper element | This project |
|---|---|
| MLP (6,12,6), tanh, Adam, α=0.01, constant LR | ✅ `04_train_neural_network.py` |
| No input normalization | ✅ (Pipeline wrapper commented out) |
| 70/30 stratified split | ✅ |
| 4 classes with the OFF/ON/NoAdj/Alert semantics | ✅ |
| Soil capacity via moisture derivative slow-drainage region | ✅ with our own numeric thresholds |
| Confidence band at μ ± σ/2, critical at μ − σ | ✅ |
| 1000-season Monte Carlo, 70 days, 10-min ticks, 300 L/hr, 4 kW | ✅ |
| IRRIFRAME baseline: 324.5 mm × 132 m² | ✅ |

Divergences / honest notes:

- **Paper trains three classifiers (KNN, SVM, MLP) and compares.** This project only does MLP.
- **Paper reports water savings 14.5–27.6% and energy 49.2–57%.** The active MC samples differently from the paper (we use `N(μ, σ/3)` because `N(μ, μ−σ/2)` as literally written in the paper is numerically nonsense). The commented-out stateful sim in `05_monte_carlo_simulation.py` is a more physically realistic alternative that was never activated.
- **Paper's archive vs forecast data.** Both projects train on archive "reconstructed" forecasts. Nobody is validating with a live forecast ever actually used.
- **Paper's class counts:** 3467 / 8875 / 3780 / 65 vs ours 2689 / 8641 / 4992 / 91. Close; the difference comes from our heuristic choices in `03_soil_capacity_calculator.py` (lookback 12 steps, drop cutoff −0.5).

---

## 5. Improvement plan (ranked by leverage)

### 🔴 Priority 1 — Fix the label leakage (biggest fake-accuracy problem)

Pick one of these redesigns:

**Option A — predict ground-truth irrigation actions, not synthetic labels.**
- Use `Irrigation (ON/OFF)` from the water meter as a weak supervision signal instead of/alongside the synthetic `Irrigation_Decision`.
- Trade-off: the meter only tells you ON vs OFF — the four-class scheme disappears.
- Best if you care about *replicating what a good farmer actually does*.

**Option B — predict future soil moisture, and derive the action from the prediction.**
- Turn this into a regression / forecasting problem: given the last N 10-minute samples, predict soil moisture over the next h hours.
- Then run the paper's confidence-interval policy on the predicted trajectory to emit an action.
- Model candidates: GRU, LSTM, 1-D CNN, temporal fusion transformer, or even gradient-boosted trees over lag features.
- This is what "predictive algorithm" actually means. The paper doesn't do this; the rest of the literature does.

**Option C — keep the classifier but make labels not derivable from inputs.**
- Have an agronomist hand-label a subset of events. Or use the water meter ON/OFF as ground truth and learn the rest.
- Effortful but the only fully honest classifier framing.

I'd recommend Option B as the most interesting and defensible direction.

### 🔴 Priority 2 — Fix the Monte Carlo

The active MC samples moisture iid from a narrow Gaussian and holds everything else at mean. This tells us very little about real-world savings.

Activate the commented-out stateful simulator (bottom of `05_monte_carlo_simulation.py`) as a starting point. Improvements from there:

1. Replace the empirical transition pools (`on_pool`, `off_dry_pool`, `off_rain_pool`) with a **physical soil-moisture balance model**:
   ```
   M[t+1] = M[t] + k_irr·ON[t] + k_rain·rain[t] − k_et·ET[t] − k_drain·max(M[t]−μ, 0)
   ```
   Fit `k_*` against the observed data.
2. Use a **realistic weather generator** (bootstrap multi-day blocks conditioned on season, as the commented-out code already does) so exogenous inputs have realistic autocorrelation.
3. **Compare against the correct baselines**, not just IRRIFRAME total. At minimum: (a) a dumb threshold-only policy using μ ± σ/2 and (b) the actual water meter history. If the ML model doesn't beat the pure-threshold policy, it has no value.

### 🟡 Priority 3 — Fix `main_controller.py`

Already described in §2: unbox the bundle and reorder features before `predict`. Also add a soft retry around the Open-Meteo call, a log file, and a safety interlock (e.g. no more than X minutes ON per hour).

### 🟡 Priority 4 — Add what the paper left out

- **Crop growth stage.** Paper itself flags this in its future-work section. `indicators.csv` has GDD — use it. Water needs for a tomato at flowering ≠ at ripening.
- **Deeper soil profile.** Paper used one 20 cm probe per line. Adding 40 cm and 60 cm probes (if available) would let you model water percolating past the root zone — which is literally what the paper claims to prevent.
- **Full three-day forecast horizon.** Currently we use just `hourly[0]`. Either consume the full 72 values as features (gives ~72 extra columns) or summarize into statistics (mean, max, sum of rain over 24/48/72h).

### 🟢 Priority 5 — Engineering cleanup

- Delete `data/desc_tree.py` and `data/raw_sensor_data.csv` (empty).
- Delete the triple-quoted dead code at the bottom of `05_monte_carlo_simulation.py` once it's either activated or confirmed obsolete.
- Add `make` / `just` targets or a single `run_pipeline.py` that runs 01 → 02 → 03 → 04 → 05 in order.
- Add **unit tests** for `03_soil_capacity_calculator.py` label logic — it's the most error-prone step and has no tests.
- Pin versions in `requirements.txt` — currently it's unpinned.
- Consider an `sklearn.pipeline.Pipeline(StandardScaler, MLPClassifier)` even though the paper skipped it. For any future non-trivial features it will matter.

### 🟢 Priority 6 — Reproduce the paper's full comparison

Train KNN (k=13, distance weights) and SVM (RBF, C=1000, γ=0.1) alongside the MLP. The paper's whole argument for MLP is "decision boundaries are cleaner" — you can only show that if you train the others.

---

## 6. What to do right now, if you had one afternoon

1. Fix the `main_controller.py` bundle-unbox bug (5 min).
2. Add a feature-importance or permutation-importance printout to `04_train_neural_network.py`. Confirm that Soil Temperature / Env Temperature / Env Humidity / ET are ~zero importance. Once you see it, the label-leakage problem will be obvious and motivating.
3. Prototype Option B (soil-moisture forecasting) on a single irrigation line with a simple GRU. Even a crappy forecaster that beats persistence is more defensible than the current setup.

---

## 7. Things I'm not sure about

- Whether the Stuard water-meter `volume_diff > 0` heuristic catches all real irrigation events — it depends on sensor reporting cadence and whether the meter sometimes rolls back.
- Whether our slow-drainage filter (`RECESSION_LOOKBACK_STEPS=12`, `MAX_SLOW_DRAINAGE_DROP=-0.5`) produces a μ that matches the paper's unpublished value. Our μ = 25.25%, which is plausible for a sandy-loam tomato field but not verifiable without a figure from the paper.
- Whether `Open-Meteo archive` actually returns the same values the paper's forecast API returned during the 2023 season — archives are usually reanalysis, not the forecast that was issued at the time.

If you touch those three assumptions, document the new values in `soil_capacity_metadata.json` and re-run 03 → 04 → 05.
