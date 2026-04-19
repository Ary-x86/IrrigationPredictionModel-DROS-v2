# Phase 0 honesty check — legacy Preite MLP leakage audit

Bundle: `/home/aryan/Projects/Non-DROS v2/models/mlp_irrigation_model.pkl`
Data:   `/home/aryan/Projects/Non-DROS v2/data/processed_dataset.csv`

## Headline

- MLP test accuracy (70/30 stratified split, seed 42): **0.9868**
- Shallow (max_depth=4) decision tree on **only** `Soil Moisture [RH%]` + `Weather Forecast Rainfall [mm]`: **1.0000**

If the two-feature tree matches the MLP, the label is recoverable from the two features the labeling rule uses. That is the definition of label leakage.

## Permutation importance (accuracy drop on held-out 30%, 5 shuffles, mean)

| feature | accuracy_drop |
| --- | ---: |
| Soil Moisture [RH%] | +0.5742 |
| Environmental Temperature [ C] | +0.1703 |
| Environmental Humidity [RH %] | +0.1307 |
| Weather Forecast Rainfall [mm] | +0.0178 |
| Soil Temperature [C] | +0.0126 |
| Crop Data Evapotranspiration [mm] | +0.0047 |

## Classification report

```
precision    recall  f1-score   support

         OFF       0.98      0.96      0.97       807
          ON       0.99      1.00      1.00      2591
       NoAdj       0.98      0.98      0.98      1498
       Alert       0.71      0.81      0.76        27

    accuracy                           0.99      4923
   macro avg       0.92      0.94      0.93      4923
weighted avg       0.99      0.99      0.99      4923
```

## Confusion matrix (rows=true, cols=pred; labels [OFF, ON, NoAdj, Alert])

```
[[ 775    0   23    9]
 [   0 2587    4    0]
 [   9   15 1474    0]
 [   3    2    0   22]]
```

## Why this matters

`src/03_soil_capacity_calculator.py` computes `Irrigation_Decision` with explicit threshold rules on `Soil Moisture [RH%]` and `Weather Forecast Rainfall [mm]`. Those two columns are then fed in as features for training in `src/04_train_neural_network.py`. The MLP is re-learning the threshold rule, not discovering an irrigation policy. Headline accuracy and any downstream Monte Carlo water-savings numbers are therefore tautological.

This motivates the Track B rebuild: reframe the task as future-VWC regression, drop threshold-derived classification labels entirely.