# NASA battery data

Place the original NASA `.mat` files in `data/raw/`:

```text
data/raw/
├── B0005.mat
├── B0006.mat
├── B0007.mat
└── B0018.mat
```

The default experiment trains on **B0005, B0006, and B0007** and evaluates on the unseen **B0018** cell.

## Target

For each cell, capacity-based State of Health is defined as:

`SOH_t = Capacity_t / Capacity_initial`

`capacity_ah` is retained only for constructing/evaluating the target. It is **not** included in the predictor list, avoiding direct target leakage.

## Predictors

Cycle-level statistics are extracted from voltage, current, temperature, and discharge time. Windows are created independently within each cell and never cross battery boundaries.

Raw NASA files are intentionally not redistributed by this repository. Users should obtain the data from the original data provider and respect its terms.
