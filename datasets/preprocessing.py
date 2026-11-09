"""Explicit, auditable preprocessing protocols for GFM4MTSAD experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


TRAIN_ONLY_PROTOCOL = "train_only"
PGRF_FAITHFUL_PROTOCOL = "pgrf_faithful"
SUPPORTED_PROTOCOLS = (TRAIN_ONLY_PROTOCOL, PGRF_FAITHFUL_PROTOCOL)


def first_order_difference(values: np.ndarray) -> np.ndarray:
    """Match PGRF-Net's length-preserving first-order differencing exactly."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(
            "first-order differencing requires a non-empty [time, feature] array"
        )
    return np.diff(values, axis=0, prepend=values[0:1, :])


def _validate_splits(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits = tuple(np.asarray(value) for value in (train, validation, test))
    if any(value.ndim != 2 or value.shape[0] == 0 for value in splits):
        raise ValueError("preprocessing requires non-empty two-dimensional splits")
    feature_counts = {int(value.shape[1]) for value in splits}
    if len(feature_counts) != 1:
        raise ValueError("preprocessing splits must have the same feature count")
    return splits


def _scaler(scale: str):
    if scale in ("standard", "instance"):
        return StandardScaler(), "standard"
    if scale == "min-max":
        return MinMaxScaler(), "min_max"
    if scale == "none":
        return None, "none"
    raise ValueError(f"unsupported DATA.SCALE value: {scale}")


def preprocess_splits(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    *,
    scale: str,
    protocol: str = TRAIN_ONLY_PROTOCOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Transform train/validation/test while preserving their original lengths.

    ``train_only`` is the historical GFM4MTSAD behavior: fit the requested scaler on
    the normal training prefix and transform validation/test with that frozen
    scaler. ``pgrf_faithful`` mirrors PGRF-Net ``data_loader.py``: reconstruct
    the original training sequence as train+validation, append test, difference
    the entire concatenation, and fit a StandardScaler on all resulting values.
    The latter intentionally uses test-distribution statistics and is therefore
    an official-code reproduction arm, not an information-isolated protocol.
    """
    train, validation, test = _validate_splits(train, validation, test)
    protocol = str(protocol).strip().lower()
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(
            f"unsupported preprocessing protocol {protocol!r}; "
            f"expected one of {SUPPORTED_PROTOCOLS}"
        )
    scaler, scaler_name = _scaler(str(scale).strip().lower())
    lengths = tuple(int(value.shape[0]) for value in (train, validation, test))

    if protocol == TRAIN_ONLY_PROTOCOL:
        if scaler is None:
            transformed = (train, validation, test)
        else:
            transformed = (
                scaler.fit_transform(train),
                scaler.transform(validation),
                scaler.transform(test),
            )
        diagnostics = {
            "protocol": TRAIN_ONLY_PROTOCOL,
            "difference_order": 0,
            "difference_scope": "none",
            "scaler": scaler_name,
            "scaler_fit_scope": "train_prefix_only",
            "scaler_fit_timestamps": lengths[0] if scaler is not None else 0,
            "uses_validation_distribution": False,
            "uses_test_distribution": False,
            "split_lengths": {
                "train": lengths[0],
                "validation": lengths[1],
                "test": lengths[2],
            },
        }
        return (*transformed, diagnostics)

    if scaler_name != "standard":
        raise ValueError(
            "pgrf_faithful requires DATA.SCALE=standard because the official "
            "PGRF-Net input scaler is StandardScaler"
        )
    full = np.concatenate((train, validation, test), axis=0)
    differenced = first_order_difference(full)
    scaled = scaler.fit_transform(differenced)
    train_end = lengths[0]
    validation_end = train_end + lengths[1]
    transformed = (
        scaled[:train_end],
        scaled[train_end:validation_end],
        scaled[validation_end:],
    )
    diagnostics = {
        "protocol": PGRF_FAITHFUL_PROTOCOL,
        "difference_order": 1,
        "difference_scope": "concatenated_train_validation_test",
        "scaler": "standard",
        "scaler_fit_scope": "concatenated_train_validation_test",
        "scaler_fit_timestamps": int(full.shape[0]),
        "uses_validation_distribution": True,
        "uses_test_distribution": True,
        "split_lengths": {
            "train": lengths[0],
            "validation": lengths[1],
            "test": lengths[2],
        },
    }
    return (*transformed, diagnostics)
