"""Normalized temporal and graph distances between paired views."""

import math

import torch
import torch.nn.functional as F


def _validate_pair(anchor, candidate, dimensions, name):
    if not isinstance(anchor, torch.Tensor) or not isinstance(
        candidate, torch.Tensor
    ):
        raise TypeError(f"{name} inputs must be torch.Tensor values")
    if anchor.dim() != dimensions or candidate.dim() != dimensions:
        raise ValueError(f"{name} inputs must have {dimensions} dimensions")
    if anchor.shape != candidate.shape:
        raise ValueError(
            f"{name} inputs must have identical shape: "
            f"anchor={tuple(anchor.shape)}, candidate={tuple(candidate.shape)}"
        )
    if not torch.isfinite(anchor).all() or not torch.isfinite(candidate).all():
        raise ValueError(f"{name} inputs must be finite")


def normalized_temporal_drift(anchor, candidate, eps=1e-6):
    """Return per-window RMS drift normalized by anchor temporal variation."""
    _validate_pair(anchor, candidate, 3, "temporal drift")
    eps = float(eps)
    if eps <= 0.0:
        raise ValueError("temporal drift eps must be positive")
    delta = (candidate - anchor).square().mean(dim=(1, 2)).sqrt()
    centered = anchor - anchor.mean(dim=1, keepdim=True)
    reference = centered.square().mean(dim=(1, 2)).sqrt()
    return delta / reference.clamp_min(eps)


def normalized_graph_drift(anchor, candidate, eps=1e-6):
    """Return per-window mean absolute graph drift on the DLinear graph."""
    _validate_pair(anchor, candidate, 3, "graph drift")
    if anchor.size(1) != anchor.size(2):
        raise ValueError("graph drift inputs must have shape [B, N, N]")
    eps = float(eps)
    if eps <= 0.0:
        raise ValueError("graph drift eps must be positive")
    delta = (candidate - anchor).abs().mean(dim=(1, 2))
    reference = anchor.abs().mean(dim=(1, 2))
    return delta / reference.clamp_min(eps)
