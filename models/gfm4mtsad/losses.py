import math

import torch
import torch.nn.functional as F


def _weighted_sample_mean(per_sample_loss, sample_weights, zero_reference):
    if per_sample_loss.numel() == 0:
        return zero_reference.sum() * 0.0
    if sample_weights is None:
        return per_sample_loss.mean()

    weights = sample_weights.detach().to(
        device=per_sample_loss.device,
        dtype=per_sample_loss.dtype,
    ).reshape(-1)
    if weights.numel() != per_sample_loss.numel():
        raise ValueError(
            "sample_weights must match the per-sample loss count: "
            f"weights={weights.numel()}, losses={per_sample_loss.numel()}"
        )
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("sample_weights must be finite and non-negative")
    weight_sum = weights.sum()
    if float(weight_sum.detach()) <= 0.0:
        return zero_reference.sum() * 0.0
    return (per_sample_loss * weights).sum() / weight_sum


def carots_style_contrastive_loss(
    anchor,
    positive,
    negative,
    temperature,
    sample_weights=None,
):
    if anchor.size(0) < 1:
        return anchor.sum() * 0.0

    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)
    pos_sim = torch.sum(anchor * positive, dim=1, keepdim=True)
    neg_sim = torch.sum(anchor * negative, dim=1, keepdim=True)
    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    return _weighted_sample_mean(per_sample, sample_weights, anchor)


def batch_hard_normal_contrastive_loss(
    anchor,
    positive,
    negative,
    normal_scores,
    temperature,
    hard_normal_fraction,
    hard_negative_count,
):
    """Contrast high-scoring normal anchors against batch-hard anomalies.

    High-scoring normal windows remain normal: their relation-preserving view
    is the positive, while the most similar accepted synthetic anomalies in
    the batch form the negative set.  Score selection is detached so this
    term changes representation geometry rather than directly suppressing
    the selected window scores.
    """
    if anchor.shape != positive.shape or anchor.ndim != 2:
        raise ValueError(
            "anchor and positive must have the same [B, D] shape: "
            f"anchor={tuple(anchor.shape)}, positive={tuple(positive.shape)}"
        )
    if negative.ndim != 2 or negative.size(1) != anchor.size(1):
        raise ValueError(
            "negative must have shape [V, D] with the shared embedding dim: "
            f"anchor={tuple(anchor.shape)}, negative={tuple(negative.shape)}"
        )
    scores = normal_scores.reshape(-1)
    if scores.numel() != anchor.size(0):
        raise ValueError(
            "normal_scores must contain one score per normal anchor: "
            f"scores={scores.numel()}, anchors={anchor.size(0)}"
        )
    hard_normal_fraction = float(hard_normal_fraction)
    if not 0.0 < hard_normal_fraction <= 1.0:
        raise ValueError("hard_normal_fraction must be in (0, 1]")
    hard_negative_count = int(hard_negative_count)
    if hard_negative_count < 1:
        raise ValueError("hard_negative_count must be positive")
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if anchor.size(0) == 0 or negative.size(0) == 0:
        return anchor.sum() * 0.0

    hard_normal_count = max(
        1,
        int(math.ceil(hard_normal_fraction * anchor.size(0))),
    )
    hard_indices = torch.topk(
        scores.detach(),
        k=hard_normal_count,
        largest=True,
        sorted=False,
    ).indices
    hard_anchor = F.normalize(anchor[hard_indices], p=2, dim=1)
    hard_positive = F.normalize(positive[hard_indices], p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)

    positive_logits = torch.sum(
        hard_anchor * hard_positive,
        dim=1,
        keepdim=True,
    )
    all_negative_logits = hard_anchor @ negative.transpose(0, 1)
    negative_count = min(hard_negative_count, negative.size(0))
    hard_negative_logits = torch.topk(
        all_negative_logits,
        k=negative_count,
        dim=1,
        largest=True,
        sorted=False,
    ).values
    logits = torch.cat([positive_logits, hard_negative_logits], dim=1)
    logits = logits / temperature
    labels = torch.zeros(
        hard_anchor.size(0),
        dtype=torch.long,
        device=hard_anchor.device,
    )
    loss = F.cross_entropy(logits, labels)
    chance_scale = math.log(2.0) / math.log(float(negative_count + 1))
    return loss * chance_scale


def node_aware_hard_contrastive_loss(
    anchor_nodes,
    positive_nodes,
    negative_nodes,
    direct_node_mask,
    normal_node_scores,
    temperature,
    hard_normal_fraction,
    hard_negative_count,
    sample_weights=None,
):
    """Contrast hard normal nodes against same-sensor direct interventions.

    Candidate anchors are the normal counterparts of accepted directly
    intervened nodes.  Their relation-preserving counterparts are positives.
    Negatives are mined only from directly intervened nodes with the same
    sensor identity, so cross-sensor semantics cannot become a shortcut.
    Score-based hard-anchor selection is detached; gradients act on the
    shared node representation rather than on the selecting node score.
    """
    if (
        anchor_nodes.shape != positive_nodes.shape
        or anchor_nodes.shape != negative_nodes.shape
        or anchor_nodes.ndim != 3
    ):
        raise ValueError(
            "node views must share shape [B, N, D]: "
            f"anchor={tuple(anchor_nodes.shape)}, "
            f"positive={tuple(positive_nodes.shape)}, "
            f"negative={tuple(negative_nodes.shape)}"
        )
    expected_node_shape = anchor_nodes.shape[:2]
    if direct_node_mask.shape != expected_node_shape:
        raise ValueError(
            "direct_node_mask must have shape [B, N]: "
            f"mask={tuple(direct_node_mask.shape)}, "
            f"nodes={tuple(expected_node_shape)}"
        )
    if normal_node_scores.shape != expected_node_shape:
        raise ValueError(
            "normal_node_scores must have shape [B, N]: "
            f" scores={tuple(normal_node_scores.shape)}, "
            f"nodes={tuple(expected_node_shape)}"
        )
    hard_normal_fraction = float(hard_normal_fraction)
    if not 0.0 < hard_normal_fraction <= 1.0:
        raise ValueError("hard_normal_fraction must be in (0, 1]")
    hard_negative_count = int(hard_negative_count)
    if hard_negative_count < 1:
        raise ValueError("hard_negative_count must be positive")
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    direct_mask = direct_node_mask.bool()
    direct_indices = torch.nonzero(direct_mask, as_tuple=False)
    if direct_indices.numel() == 0:
        return anchor_nodes.sum() * 0.0

    candidate_scores = normal_node_scores[direct_mask]
    hard_normal_count = max(
        1,
        int(math.ceil(hard_normal_fraction * candidate_scores.numel())),
    )
    selected = torch.topk(
        candidate_scores.detach(),
        k=hard_normal_count,
        largest=True,
        sorted=False,
    ).indices
    selected_indices = direct_indices[selected]

    anchor_nodes = F.normalize(anchor_nodes, p=2, dim=-1)
    positive_nodes = F.normalize(positive_nodes, p=2, dim=-1)
    negative_nodes = F.normalize(negative_nodes, p=2, dim=-1)
    per_anchor_losses = []
    selected_weights = []
    for batch_idx, sensor_idx in selected_indices.tolist():
        anchor = anchor_nodes[batch_idx, sensor_idx]
        positive = positive_nodes[batch_idx, sensor_idx]
        same_sensor_mask = direct_mask[:, sensor_idx]
        negative_pool = negative_nodes[same_sensor_mask, sensor_idx]
        negative_count = min(hard_negative_count, negative_pool.size(0))
        negative_logits = torch.topk(
            negative_pool @ anchor,
            k=negative_count,
            largest=True,
            sorted=False,
        ).values
        positive_logit = torch.sum(anchor * positive).reshape(1)
        logits = torch.cat([positive_logit, negative_logits]) / temperature
        per_anchor = torch.logsumexp(logits, dim=0) - logits[0]
        chance_scale = math.log(2.0) / math.log(float(negative_count + 1))
        per_anchor_losses.append(per_anchor * chance_scale)
        if sample_weights is not None:
            selected_weights.append(sample_weights[batch_idx])

    per_anchor_losses = torch.stack(per_anchor_losses)
    weights = (
        torch.stack(selected_weights)
        if sample_weights is not None
        else None
    )
    return _weighted_sample_mean(per_anchor_losses, weights, anchor_nodes)


def intervention_gain_node_loss(
    anchor_scores,
    anomaly_scores,
    node_mask,
    margin,
    detach_anchor=True,
    sample_weights=None,
):
    if anchor_scores.shape != anomaly_scores.shape or anomaly_scores.shape != node_mask.shape:
        raise ValueError(
            "intervention_gain_node_loss got inconsistent shapes: "
            f"anchor={tuple(anchor_scores.shape)}, anomaly={tuple(anomaly_scores.shape)}, "
            f"mask={tuple(node_mask.shape)}"
        )

    baseline = anchor_scores.detach() if detach_anchor else anchor_scores
    gains = anomaly_scores - baseline
    node_mask = node_mask.bool()
    losses = []
    selected_weights = []

    for sample_idx, (sample_gains, mask) in enumerate(zip(gains, node_mask)):
        direct_gains = sample_gains[mask]
        unmodified_gains = sample_gains[~mask]
        if direct_gains.numel() == 0 or unmodified_gains.numel() == 0:
            continue
        pairwise_margin = margin + unmodified_gains.unsqueeze(0) - direct_gains.unsqueeze(1)
        losses.append(F.relu(pairwise_margin).mean())
        if sample_weights is not None:
            selected_weights.append(sample_weights[sample_idx])

    if not losses:
        return anomaly_scores.sum() * 0.0
    selected_weights = (
        torch.stack(selected_weights) if sample_weights is not None else None
    )
    return _weighted_sample_mean(
        torch.stack(losses),
        selected_weights,
        anomaly_scores,
    )


def masked_node_consistency_loss(
    reference_scores,
    view_scores,
    mask=None,
    detach_reference=True,
    sample_weights=None,
):
    if reference_scores.shape != view_scores.shape:
        raise ValueError(
            "masked_node_consistency_loss got inconsistent score shapes: "
            f"reference={tuple(reference_scores.shape)}, view={tuple(view_scores.shape)}"
        )
    if mask is None:
        mask = torch.ones_like(view_scores, dtype=torch.bool)
    if mask.shape != view_scores.shape:
        raise ValueError(
            "masked_node_consistency_loss got inconsistent mask shape: "
            f"scores={tuple(view_scores.shape)}, mask={tuple(mask.shape)}"
        )

    target = reference_scores.detach() if detach_reference else reference_scores
    elementwise = F.smooth_l1_loss(view_scores, target, reduction="none")
    mask = mask.bool()
    if sample_weights is None:
        selected = elementwise.masked_select(mask)
        if selected.numel() == 0:
            return view_scores.sum() * 0.0
        return selected.mean()

    per_sample = []
    selected_weights = []
    for sample_idx, (sample_loss, sample_mask) in enumerate(zip(elementwise, mask)):
        selected = sample_loss.masked_select(sample_mask)
        if selected.numel() == 0:
            continue
        per_sample.append(selected.mean())
        selected_weights.append(sample_weights[sample_idx])
    if not per_sample:
        return view_scores.sum() * 0.0
    return _weighted_sample_mean(
        torch.stack(per_sample),
        torch.stack(selected_weights),
        view_scores,
    )


def ranking_loss(normal_scores, anomaly_scores, margin, sample_weights=None):
    per_sample = F.relu(margin + normal_scores - anomaly_scores)
    return _weighted_sample_mean(per_sample, sample_weights, anomaly_scores)


def hard_normal_tail_ranking_loss(
    normal_scores,
    anomaly_scores,
    margin,
    hard_fraction,
    positive_normal_scores=None,
    sample_weights=None,
):
    """Rank synthetic anomalies above the hardest normal-score tail.

    The normal pool contains the original normal windows and, when supplied,
    their relation-preserving positive views.  The mean of the highest fixed
    fraction is used instead of a label-selected threshold.
    """
    hard_fraction = float(hard_fraction)
    if not 0.0 < hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be in (0, 1]")

    normal_parts = [normal_scores.reshape(-1)]
    if positive_normal_scores is not None:
        normal_parts.append(positive_normal_scores.reshape(-1))
    normal_pool = torch.cat(normal_parts, dim=0)
    anomalies = anomaly_scores.reshape(-1)
    if normal_pool.numel() == 0 or anomalies.numel() == 0:
        return anomaly_scores.sum() * 0.0

    hard_count = max(1, int(math.ceil(hard_fraction * normal_pool.numel())))
    hard_reference = torch.topk(
        normal_pool,
        k=hard_count,
        largest=True,
        sorted=False,
    ).values.mean()
    per_sample = F.relu(margin + hard_reference - anomalies)
    return _weighted_sample_mean(per_sample, sample_weights, anomaly_scores)


def window_bce_loss(
    normal_scores,
    anomaly_scores,
    positive_scores=None,
    sample_weights=None,
):
    losses = [
        F.binary_cross_entropy_with_logits(
            normal_scores,
            torch.zeros_like(normal_scores),
            reduction="none",
        ),
        F.binary_cross_entropy_with_logits(
            anomaly_scores,
            torch.ones_like(anomaly_scores),
            reduction="none",
        ),
    ]
    if positive_scores is not None:
        losses.append(
            F.binary_cross_entropy_with_logits(
                positive_scores,
                torch.zeros_like(positive_scores),
                reduction="none",
            )
        )
    per_sample = torch.stack(losses, dim=0).mean(dim=0)
    return _weighted_sample_mean(per_sample, sample_weights, anomaly_scores)


def head_fidelity_loss(node_scores_calibrated, raw_node_scores, eps=1e-6):
    raw_log = torch.log(raw_node_scores.clamp_min(eps))
    return F.smooth_l1_loss(node_scores_calibrated, raw_log)
