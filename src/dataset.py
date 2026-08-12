from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def make_sequences(frame, feature_columns, sequence_length, scaler):
    """Create within-cell windows; windows never cross battery boundaries."""
    xs, ys, metadata = [], [], []

    for battery_id, group in frame.groupby("battery_id", sort=False):
        group = group.sort_values("cycle_index").reset_index(drop=True)
        features = scaler.transform(group[feature_columns].to_numpy(dtype=np.float32))
        targets = group["soh"].to_numpy(dtype=np.float32)

        for end in range(sequence_length - 1, len(group)):
            start = end - sequence_length + 1
            xs.append(features[start : end + 1])
            ys.append(targets[end])
            metadata.append(
                {
                    "battery_id": battery_id,
                    "cycle_index": int(group.loc[end, "cycle_index"]),
                    "capacity_ah": float(group.loc[end, "capacity_ah"]),
                }
            )

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        metadata,
    )


class SOHSequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
