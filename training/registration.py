from __future__ import annotations
from pathlib import Path
import torch
from datasets.preprocessing import PGRF_FAITHFUL_PROTOCOL, TRAIN_ONLY_PROTOCOL
from training.assets import sha256
FRONTEND_WEIGHT_MARKERS = ("linear_seasonal", "linear_trend")

def inspect_frontend_checkpoint(path: str | Path, expected_window: int) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"frontend checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("frontend checkpoint must contain a dictionary")
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("frontend checkpoint model state must be a dictionary")
    weights = {
        name: value
        for name, value in state.items()
        if name.startswith("imlp.")
        and name.endswith(".weight")
        and any(marker in name for marker in FRONTEND_WEIGHT_MARKERS)
        and torch.is_tensor(value)
        and value.dim() == 2
    }
    if not weights:
        raise RuntimeError("checkpoint has no DLinear seasonal/trend frontend weights")
    inputs = sorted({int(value.shape[-1]) for value in weights.values()})
    outputs = sorted({int(value.shape[0]) for value in weights.values()})
    if inputs != [int(expected_window)]:
        raise ValueError(
            "DLinear checkpoint input dimension disagrees with window: "
            f"observed={inputs}, expected={[int(expected_window)]}"
        )
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256(path),
        "checkpoint_size_bytes": int(path.stat().st_size),
        "observed_input_dimensions": inputs,
        "observed_output_dimensions": outputs,
        "frontend_weight_count": len(weights),
    }



def build_registration(
    checkpoint: str | Path,
    *,
    entity: str,
    window: int,
    source_config: str | Path,
    preprocessing_protocol: str = PGRF_FAITHFUL_PROTOCOL,
    source_template_entity: str | None = None,
    seed: int | None = None,
    backend: str | None = None,
) -> dict:
    if preprocessing_protocol not in {
        PGRF_FAITHFUL_PROTOCOL,
        TRAIN_ONLY_PROTOCOL,
    }:
        raise ValueError(
            f"unsupported frontend preprocessing protocol: {preprocessing_protocol}"
        )
    report = inspect_frontend_checkpoint(checkpoint, window)
    source_config = Path(source_config)
    if not source_config.is_file():
        raise FileNotFoundError(f"source config not found: {source_config}")
    registration = {
        **report,
        "entity": str(entity),
        "window": int(window),
        "preprocessing_protocol": preprocessing_protocol,
        "data_scale": "standard",
        "source_config": str(source_config.resolve()),
        "source_config_sha256": sha256(source_config),
        "source_template_entity": str(source_template_entity or entity),
        "test_distribution_used_by_preprocessing": (
            preprocessing_protocol == PGRF_FAITHFUL_PROTOCOL
        ),
        "test_labels_used_by_preprocessing": False,
    }
    if seed is not None:
        if int(seed) < 0:
            raise ValueError("frontend seed must be nonnegative")
        registration["seed"] = int(seed)
    if backend is not None:
        registration["backend"] = str(backend)
    return registration
