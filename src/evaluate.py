from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.dataset import make_sequences
from src.train import build_model
from src.preprocessing import FEATURE_COLUMNS, load_batteries


def safe_mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-8
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    output_dir = Path(cfg["output"]["directory"])
    model_type = cfg["model"]["type"].lower()
    checkpoint_path = output_dir / f"{model_type}_best.pt"
    scaler = joblib.load(output_dir / "scaler.joblib")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    test_frame = load_batteries(cfg["data"]["data_dir"], cfg["data"]["test_batteries"])
    x_test, y_test, metadata = make_sequences(
        test_frame,
        FEATURE_COLUMNS,
        int(cfg["data"]["sequence_length"]),
        scaler,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, len(FEATURE_COLUMNS), int(cfg["data"]["sequence_length"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    with torch.no_grad():
        pred = model(torch.as_tensor(x_test, dtype=torch.float32, device=device))
        y_pred = pred.cpu().numpy()

    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "MAPE_percent": safe_mape(y_test, y_pred),
        "R2": float(r2_score(y_test, y_pred)),
    }

    print(json.dumps(metrics, indent=2))
    with open(output_dir / f"{model_type}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    prediction_frame = pd.DataFrame(metadata)
    prediction_frame["soh_actual"] = y_test
    prediction_frame["soh_predicted"] = y_pred
    prediction_frame.to_csv(output_dir / f"{model_type}_predictions.csv", index=False)

    for battery_id, group in prediction_frame.groupby("battery_id"):
        plt.figure(figsize=(8, 4.5))
        plt.plot(group["cycle_index"], group["soh_actual"], label="Actual SOH")
        plt.plot(group["cycle_index"], group["soh_predicted"], label="Predicted SOH")
        plt.xlabel("Discharge cycle")
        plt.ylabel("SOH")
        plt.title(f"{battery_id}: Actual vs Predicted SOH")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_type}_{battery_id}_soh.png", dpi=180)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nasa.yaml")
    args = parser.parse_args()
    main(args.config)
