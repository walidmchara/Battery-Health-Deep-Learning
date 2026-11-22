from __future__ import annotations


def validate_battery_splits(train_batteries, test_batteries):
    """Reject ambiguous or overlapping battery-level splits."""
    normalized = []
    for name, batteries in (("train", train_batteries), ("test", test_batteries)):
        if not isinstance(batteries, (list, tuple)) or not batteries:
            raise ValueError(f"{name} batteries must be a non-empty list")
        if any(not isinstance(battery, str) or not battery.strip() for battery in batteries):
            raise ValueError(f"{name} battery identifiers must be non-empty strings")
        ids = [battery.strip().casefold() for battery in batteries]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name} batteries contain duplicate identifiers")
        normalized.append(set(ids))
    overlap = normalized[0] & normalized[1]
    if overlap:
        raise ValueError(f"Train and test batteries must be disjoint: {sorted(overlap)}")
