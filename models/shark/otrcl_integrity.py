"""Read-only scientific integrity diagnostics for OTR-CL.

The functions in this module never participate in a loss and never mutate a
view.  They turn the four-view method contract into quantities that can be
saved beside a training run:

* W+ should remain close in temporal and relational space;
* Wc should move farther in temporal space than Wf;
* Wf should move farther in relational space than Wc.
"""

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


def native_relational_residual(node_embeddings, adjacency, eps=1e-6):
    """Compute normalized native-GFM residual ``z_i - sum_j A_ij z_j``."""
    if node_embeddings.dim() != 3 or adjacency.dim() != 3:
        raise ValueError(
            "native relational residual expects [B, N, D] and [B, N, N]"
        )
    batch_size, node_count, _ = node_embeddings.shape
    if tuple(adjacency.shape) != (batch_size, node_count, node_count):
        raise ValueError("adjacency does not match native node embeddings")
    if not torch.isfinite(node_embeddings).all() or not torch.isfinite(
        adjacency
    ).all():
        raise ValueError("native relational residual inputs must be finite")
    if (adjacency < 0).any():
        raise ValueError("native relational residual requires nonnegative graph")
    eye = torch.eye(
        node_count,
        device=adjacency.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    weights = adjacency.masked_fill(eye, 0.0)
    degree = weights.sum(dim=-1, keepdim=True)
    context = torch.bmm(weights / degree.clamp_min(eps), node_embeddings)
    return F.normalize(
        node_embeddings - context,
        p=2,
        dim=-1,
        eps=float(eps),
    )


def relational_cosine_drift(anchor, candidate):
    """Return per-window mean node-wise cosine drift in relational space."""
    _validate_pair(anchor, candidate, 3, "relational drift")
    similarity = F.cosine_similarity(anchor, candidate, dim=-1, eps=1e-6)
    return (1.0 - similarity).mean(dim=1)


def _masked_summary(values, valid_mask):
    values = values.detach()
    valid_mask = valid_mask.to(device=values.device, dtype=torch.bool)
    if valid_mask.shape != values.shape:
        raise ValueError("view validity mask must have shape [B]")
    finite = torch.isfinite(values)
    accepted = valid_mask & finite
    all_finite = values[finite]
    accepted_values = values[accepted]
    return {
        "all_mean": (
            float(all_finite.mean().cpu()) if all_finite.numel() else None
        ),
        "accepted_mean": (
            float(accepted_values.mean().cpu())
            if accepted_values.numel()
            else None
        ),
        "finite_count": int(finite.sum().item()),
        "accepted_count": int(accepted.sum().item()),
        "sample_count": int(values.numel()),
    }


def _finite_difference(left, right):
    if left is None or right is None:
        return None
    value = float(left) - float(right)
    return value if math.isfinite(value) else None


@torch.no_grad()
def summarize_otrcl_view_integrity(
    anchor_windows,
    view_windows,
    anchor_outputs,
    view_outputs,
    valid_masks,
    direct_node_mask,
    propagated_node_mask,
    consistency_exclusion_mask,
    eps=1e-6,
):
    """Summarize W+/Wc/Wf semantics without changing the training graph."""
    names = ("W+", "Wc", "Wf")
    for name in names:
        if name not in view_windows or name not in view_outputs:
            raise KeyError(f"missing OTR-CL integrity view: {name}")
        if name not in valid_masks:
            raise KeyError(f"missing OTR-CL integrity validity mask: {name}")
    required_outputs = ("adjacency", "native_node_embeddings")
    for name, outputs in (("W", anchor_outputs),) + tuple(
        (view_name, view_outputs[view_name]) for view_name in names
    ):
        missing = [key for key in required_outputs if key not in outputs]
        if missing:
            raise KeyError(f"{name} outputs missing: {', '.join(missing)}")

    anchor_relation = native_relational_residual(
        anchor_outputs["native_node_embeddings"],
        anchor_outputs["adjacency"],
        eps=eps,
    )
    report = {"views": {}}
    for name in names:
        outputs = view_outputs[name]
        candidate_relation = native_relational_residual(
            outputs["native_node_embeddings"],
            outputs["adjacency"],
            eps=eps,
        )
        report["views"][name] = {
            "temporal_drift": _masked_summary(
                normalized_temporal_drift(
                    anchor_windows,
                    view_windows[name],
                    eps=eps,
                ),
                valid_masks[name],
            ),
            "graph_drift": _masked_summary(
                normalized_graph_drift(
                    anchor_outputs["adjacency"],
                    outputs["adjacency"],
                    eps=eps,
                ),
                valid_masks[name],
            ),
            "relational_drift": _masked_summary(
                relational_cosine_drift(
                    anchor_relation,
                    candidate_relation,
                ),
                valid_masks[name],
            ),
        }

    wc_temporal = report["views"]["Wc"]["temporal_drift"][
        "accepted_mean"
    ]
    wf_temporal = report["views"]["Wf"]["temporal_drift"][
        "accepted_mean"
    ]
    wc_relational = report["views"]["Wc"]["relational_drift"][
        "accepted_mean"
    ]
    wf_relational = report["views"]["Wf"]["relational_drift"][
        "accepted_mean"
    ]
    temporal_separation = _finite_difference(wc_temporal, wf_temporal)
    relational_separation = _finite_difference(wf_relational, wc_relational)
    report["directional_separation"] = {
        "wc_minus_wf_temporal_drift": temporal_separation,
        "wf_minus_wc_relational_drift": relational_separation,
        "temporal_direction_pass": bool(
            temporal_separation is not None and temporal_separation > 0.0
        ),
        "relational_direction_pass": bool(
            relational_separation is not None and relational_separation > 0.0
        ),
    }

    direct = direct_node_mask.bool()
    propagated = propagated_node_mask.bool()
    exclusion = consistency_exclusion_mask.bool()
    if not (
        direct.shape == propagated.shape == exclusion.shape
        and direct.dim() == 2
    ):
        raise ValueError("OTR-CL node masks must share shape [B, N]")
    unchanged = ~(direct | propagated | exclusion)
    positive_valid = valid_masks["W+"].bool()
    report["mask_counts"] = {
        "positive_valid": int(positive_valid.sum().item()),
        "positive_invalid": int((~positive_valid).sum().item()),
        "coarse_valid": int(valid_masks["Wc"].bool().sum().item()),
        "fine_valid": int(valid_masks["Wf"].bool().sum().item()),
        "direct_nodes": int(direct.sum().item()),
        "propagated_nodes": int(propagated.sum().item()),
        "consistency_excluded_nodes": int(exclusion.sum().item()),
        "unchanged_nodes": int(unchanged.sum().item()),
    }
    report["view_contract"] = {
        "Wc_exposes_node_mask": False,
        "Wf_exposes_direct_node_mask": bool(direct.any().item()),
    }
    return report


def gradient_clip_diagnostics(
    named_parameters,
    clip_enabled,
    clip_threshold,
    temporal_representation_enters_inference,
    eps=1e-12,
):
    """Measure pre-clip gradient norms by method-relevant parameter group."""
    named_parameters = list(named_parameters)
    groups = {
        "temporal_encoder": ("temporal_regime_encoder.",),
        "native_gfm_path": (
            "gfm_adapter.",
            "sensor_graph_prompt.",
            "gfm.",
        ),
        "ipra": ("information_residual.",),
        "direct_head": ("head.direct_node_head.",),
    }

    def norm_for(prefixes=None, exclude_prefixes=()):
        squared = 0.0
        parameter_count = 0
        gradient_tensor_count = 0
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                continue
            if prefixes is not None and not name.startswith(prefixes):
                continue
            if exclude_prefixes and name.startswith(exclude_prefixes):
                continue
            parameter_count += int(parameter.numel())
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if not torch.isfinite(gradient).all():
                return {
                    "norm": None,
                    "finite": False,
                    "parameter_count": parameter_count,
                    "gradient_tensor_count": gradient_tensor_count + 1,
                }
            squared += float(gradient.double().square().sum().cpu())
            gradient_tensor_count += 1
        return {
            "norm": math.sqrt(squared),
            "finite": True,
            "parameter_count": parameter_count,
            "gradient_tensor_count": gradient_tensor_count,
        }

    total = norm_for()
    deployment = norm_for(exclude_prefixes=("temporal_regime_encoder.",))
    group_reports = {
        name: norm_for(prefixes=prefixes)
        for name, prefixes in groups.items()
    }
    total_norm = total["norm"]
    threshold = float(clip_threshold)
    enabled = bool(clip_enabled)
    coefficient = None
    if total_norm is not None:
        coefficient = (
            min(1.0, threshold / max(total_norm, float(eps)))
            if enabled
            else 1.0
        )
    temporal_norm = group_reports["temporal_encoder"]["norm"]
    return {
        "clip_enabled": enabled,
        "clip_threshold": threshold,
        "total_trainable_gradient_norm": total_norm,
        "deployment_path_gradient_norm": deployment["norm"],
        "estimated_clip_coefficient": coefficient,
        "would_clip": bool(
            enabled
            and total_norm is not None
            and math.isfinite(total_norm)
            and total_norm > threshold
        ),
        "temporal_branch_participates_in_global_norm": bool(
            temporal_norm is not None and temporal_norm > 0.0
        ),
        "temporal_representation_enters_inference": bool(
            temporal_representation_enters_inference
        ),
        "groups": group_reports,
        "total": total,
        "deployment_path": deployment,
    }
