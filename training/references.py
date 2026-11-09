import torch
from training.assets import sha256 as sha

def save_evaluation_reference(model, path):
    """Persist calibration state that legacy checkpoints omitted."""
    head = model.head
    names = (
        "evidence_reference_mean",
        "evidence_reference_scale",
        "evidence_reference_values",
        "evidence_reference_count",
        "evidence_reference_fitted",
        "mil_reference_mean",
        "mil_reference_scale",
        "mil_reference_count",
        "mil_reference_fitted",
        "source_reliability_prior",
        "source_reliability_count",
        "source_reliability_fitted",
    )
    state = {
        name: getattr(head, name).detach().cpu()
        for name in names
        if hasattr(head, name)
    }
    if not state:
        raise RuntimeError("model head exposes no evaluation reference state")
    torch.save(state, path)
    return {
        "path": str(path.resolve()),
        "sha256": sha(path),
        "tensor_names": sorted(state),
        "evidence_reference_count": int(
            state.get("evidence_reference_count", torch.tensor(0)).item()
        ),
        "mil_reference_count": int(
            state.get("mil_reference_count", torch.tensor(0)).item()
        ),
    }
