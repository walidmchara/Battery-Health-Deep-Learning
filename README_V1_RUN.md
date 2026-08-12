# V1 NASA SOH experiment

This code release implements a leakage-aware cross-battery baseline:

- Train cells: B0005, B0006, B0007
- Unseen test cell: B0018
- Models: LSTM or Transformer
- Target: cell-relative capacity SOH
- Metrics: MAE, RMSE, MAPE, R²

## Run

```bash
pip install -r requirements.txt
python -m src.train --config configs/nasa.yaml
python -m src.evaluate --config configs/nasa.yaml
```

To benchmark the LSTM, change `model.type` in `configs/nasa.yaml` from `transformer` to `lstm`, then train/evaluate again.

The implementation deliberately excludes measured capacity from model inputs because capacity is used to define the SOH target.
