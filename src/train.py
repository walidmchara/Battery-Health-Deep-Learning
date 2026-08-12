from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import SOHSequenceDataset, make_sequences
from src.models.lstm import LSTMRegressor
from src.models.transformer import TransformerRegressor
from src.preprocessing import FEATURE_COLUMNS, load_batteries


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg, input_size, sequence_length):
    model_cfg = cfg["model"]
    common = dict(
        input_size=input_size,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    )
    if model_cfg["type"].lower() == "lstm":
        return LSTMRegressor(**common)
    if model_cfg["type"].lower() == "transformer":
        return TransformerRegressor(
            **common,
            num_heads=model_cfg["num_heads"],
            max_length=max(512, sequence_length),
        )
    raise ValueError("model.type must be 'lstm' or 'transformer'")


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    set_seed(cfg["seed"])
    data_cfg = cfg["data"]
    sequence_length = int(data_cfg["sequence_length"])
    frame = load_batteries(data_cfg["data_dir"], data_cfg["train_batteries"])

    # Chronological split is performed independently for each training cell.
    train_parts, val_parts = [], []
    val_fraction = float(cfg["training"]["validation_fraction"])
    for _, group in frame.groupby("battery_id", sort=False):
        group = group.sort_values("cycle_index")
        cut = max(sequence_length, int(len(group) * (1.0 - val_fraction)))
        cut = min(cut, len(group) - sequence_length)
        train_parts.append(group.iloc[:cut].copy())
        # overlap by sequence_length-1 so first validation target has full history
        val_parts.append(group.iloc[max(0, cut - sequence_length + 1) :].copy())

    import pandas as pd
    train_frame = pd.concat(train_parts, ignore_index=True)
    val_frame = pd.concat(val_parts, ignore_index=True)

    scaler = StandardScaler().fit(train_frame[FEATURE_COLUMNS].to_numpy())
    x_train, y_train, _ = make_sequences(
        train_frame, FEATURE_COLUMNS, sequence_length, scaler
    )
    x_val, y_val, _ = make_sequences(
        val_frame, FEATURE_COLUMNS, sequence_length, scaler
    )

    train_loader = DataLoader(
        SOHSequenceDataset(x_train, y_train),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        SOHSequenceDataset(x_val, y_val),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, len(FEATURE_COLUMNS), sequence_length).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    loss_fn = nn.MSELoss()

    output_dir = Path(cfg["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f'{cfg["model"]["type"].lower()}_best.pt'

    best_val, stale = float("inf"), 0
    history = []

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_losses.append(loss_fn(model(x), y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val, stale = val_loss, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_columns": FEATURE_COLUMNS,
                    "config": cfg,
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= int(cfg["training"]["patience"]):
                print("Early stopping.")
                break

    joblib.dump(scaler, output_dir / "scaler.joblib")
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Saved best checkpoint to {checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nasa.yaml")
    args = parser.parse_args()
    main(args.config)
