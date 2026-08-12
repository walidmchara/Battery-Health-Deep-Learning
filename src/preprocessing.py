from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat


FEATURE_COLUMNS = [
    "cycle_index",
    "mean_voltage",
    "min_voltage",
    "max_voltage",
    "voltage_std",
    "mean_current",
    "current_std",
    "mean_temperature",
    "max_temperature",
    "temperature_std",
    "discharge_duration",
]


def _as_1d(value):
    return np.asarray(value, dtype=np.float64).reshape(-1)


def extract_discharge_cycles(mat_path: str | Path, battery_id: str | None = None) -> pd.DataFrame:
    """Extract one row of leakage-free predictors per NASA discharge cycle."""
    mat_path = Path(mat_path)
    battery_id = battery_id or mat_path.stem
    raw = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if battery_id not in raw:
        raise KeyError(f"{battery_id!r} not found in {mat_path}")

    battery = raw[battery_id]
    rows = []
    discharge_index = 0

    for cycle in np.atleast_1d(battery.cycle):
        if str(cycle.type).strip().lower() != "discharge":
            continue

        data = cycle.data
        voltage = _as_1d(data.Voltage_measured)
        current = _as_1d(data.Current_measured)
        temperature = _as_1d(data.Temperature_measured)
        time = _as_1d(data.Time)
        capacity = float(np.asarray(data.Capacity).squeeze())

        discharge_index += 1
        duration = float(time[-1] - time[0]) if len(time) > 1 else 0.0

        rows.append(
            {
                "battery_id": battery_id,
                "cycle_index": discharge_index,
                "capacity_ah": capacity,  # target construction only; never a predictor
                "mean_voltage": float(np.mean(voltage)),
                "min_voltage": float(np.min(voltage)),
                "max_voltage": float(np.max(voltage)),
                "voltage_std": float(np.std(voltage)),
                "mean_current": float(np.mean(current)),
                "current_std": float(np.std(current)),
                "mean_temperature": float(np.mean(temperature)),
                "max_temperature": float(np.max(temperature)),
                "temperature_std": float(np.std(temperature)),
                "discharge_duration": duration,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No discharge cycles found in {mat_path}")

    # Cell-relative capacity SOH. Capacity itself is deliberately excluded from FEATURE_COLUMNS.
    reference_capacity = float(frame.loc[0, "capacity_ah"])
    frame["soh"] = frame["capacity_ah"] / reference_capacity
    return frame


def load_batteries(data_dir: str | Path, battery_ids: list[str]) -> pd.DataFrame:
    frames = []
    for battery_id in battery_ids:
        path = Path(data_dir) / f"{battery_id}.mat"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Copy the NASA .mat file to {Path(data_dir)}."
            )
        frames.append(extract_discharge_cycles(path, battery_id))
    return pd.concat(frames, ignore_index=True)
