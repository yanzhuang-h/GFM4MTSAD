import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gfm4mtsad.losses import _weighted_sample_mean


class RelationalResidualProjector(nn.Module):
    """从 GFM 原生节点表示中提取相对邻域上下文的关系残差。"""

    def __init__(self, exclude_self=True, eps=1e-6):
        super().__init__()
        self.exclude_self = bool(exclude_self)
        self.eps = float(eps)
        if self.eps <= 0.0:
            raise ValueError("Relational residual eps must be positive")

    @classmethod
    def from_config(cls, cfg):
        settings = cfg.GFM4MTSAD.OTRCL
        return cls(
            exclude_self=bool(settings.RELATIONAL_EXCLUDE_SELF),
            eps=float(settings.RELATIONAL_EPS),
        )

    def forward(self, node_embeddings, adjacency):
        if node_embeddings.dim() != 3:
            raise ValueError("node_embeddings must have shape [B, N, D]")
        if adjacency.dim() != 3:
            raise ValueError("adjacency must have shape [B, N, N]")
        batch_size, node_count, _ = node_embeddings.shape
        if tuple(adjacency.shape) != (batch_size, node_count, node_count):
            raise ValueError(
                "adjacency does not match node embeddings: "
                f"nodes={tuple(node_embeddings.shape)}, "
                f"adjacency={tuple(adjacency.shape)}"
            )
        if not torch.isfinite(node_embeddings).all():
            raise ValueError("node_embeddings must be finite")
        if not torch.isfinite(adjacency).all():
            raise ValueError("adjacency must be finite")
        if (adjacency < 0).any():
            raise ValueError(
                "Relational residual requires nonnegative adjacency weights"
            )

        weights = adjacency
        if self.exclude_self:
            eye = torch.eye(
                node_count,
                dtype=torch.bool,
                device=adjacency.device,
            ).unsqueeze(0)
            weights = adjacency.masked_fill(eye, 0.0)
        degree = weights.sum(dim=-1, keepdim=True)
        normalized_weights = weights / degree.clamp_min(self.eps)
        relation_context = torch.bmm(normalized_weights, node_embeddings)
        relational_residual_raw = node_embeddings - relation_context
        relational_residual = F.normalize(
            relational_residual_raw,
            p=2,
            dim=-1,
            eps=self.eps,
        )
        return {
            "relation_context": relation_context,
            "relational_residual_raw": relational_residual_raw,
            "relational_residual": relational_residual,
            "relation_neighbor_degree": degree.squeeze(-1),
        }


def _masked_per_sample_mean(values, mask):
    mask = mask.to(device=values.device, dtype=values.dtype)
    counts = mask.sum(dim=1)
    means = (values * mask).sum(dim=1) / counts.clamp_min(1.0)
    return means, counts > 0


def relational_node_contrastive_loss(
    anchor_residual,
    positive_residual,
    fine_residual,
    direct_node_mask,
    unchanged_node_mask,
    temperature=0.2,
    unchanged_weight=0.25,
    sample_weights=None,
):
    """节点关系反事实对比，不读取后端专有 prototype 或异常分数。"""
    views = (anchor_residual, positive_residual, fine_residual)
    if any(view.dim() != 3 for view in views):
        raise ValueError(
            "Relational contrast expects three [B, N, D] tensors"
        )
    if (
        anchor_residual.shape != positive_residual.shape
        or anchor_residual.shape != fine_residual.shape
    ):
        raise ValueError("Relational contrast views must have identical shape")
    expected_mask_shape = anchor_residual.shape[:2]
    for name, mask in (
        ("direct_node_mask", direct_node_mask),
        ("unchanged_node_mask", unchanged_node_mask),
    ):
        if tuple(mask.shape) != expected_mask_shape:
            raise ValueError(
                f"{name} must have shape {tuple(expected_mask_shape)}, "
                f"got {tuple(mask.shape)}"
            )
    if (direct_node_mask.bool() & unchanged_node_mask.bool()).any():
        raise ValueError("direct and unchanged node masks must be disjoint")
    temperature = float(temperature)
    unchanged_weight = float(unchanged_weight)
    if temperature <= 0.0:
        raise ValueError("Relational contrast temperature must be positive")
    if unchanged_weight < 0.0:
        raise ValueError("Relational unchanged weight must be nonnegative")

    anchor = F.normalize(anchor_residual, p=2, dim=-1)
    positive = F.normalize(positive_residual, p=2, dim=-1)
    fine = F.normalize(fine_residual, p=2, dim=-1)
    positive_similarity = (anchor * positive).sum(dim=-1)
    fine_similarity = (anchor * fine).sum(dim=-1)
    logits = torch.stack(
        (positive_similarity, fine_similarity),
        dim=-1,
    ) / temperature
    direct_per_node = F.cross_entropy(
        logits.reshape(-1, 2),
        torch.zeros(
            logits.numel() // 2,
            dtype=torch.long,
            device=logits.device,
        ),
        reduction="none",
    ).reshape(logits.shape[:2])
    direct_per_sample, direct_valid = _masked_per_sample_mean(
        direct_per_node,
        direct_node_mask.bool(),
    )

    positive_drift = 1.0 - positive_similarity
    fine_drift = 1.0 - fine_similarity
    unchanged_per_node = 0.5 * (positive_drift + fine_drift)
    unchanged_per_sample, unchanged_valid = _masked_per_sample_mean(
        unchanged_per_node,
        unchanged_node_mask.bool(),
    )

    if sample_weights is None:
        weights = None
    else:
        weights = sample_weights.reshape(-1)
        if weights.numel() != anchor.size(0):
            raise ValueError("sample_weights must have shape [B]")
    direct_weights = weights[direct_valid] if weights is not None else None
    unchanged_weights = (
        weights[unchanged_valid] if weights is not None else None
    )
    direct_loss = _weighted_sample_mean(
        direct_per_sample[direct_valid],
        direct_weights,
        anchor,
    )
    unchanged_loss = _weighted_sample_mean(
        unchanged_per_sample[unchanged_valid],
        unchanged_weights,
        anchor,
    )
    total = direct_loss + unchanged_weight * unchanged_loss

    direct_gap_values = (
        positive_similarity - fine_similarity
    )[direct_node_mask.bool()]
    unchanged_drift_values = unchanged_per_node[
        unchanged_node_mask.bool()
    ]
    zero = total.detach() * 0.0
    return {
        "total": total,
        "direct": direct_loss,
        "unchanged": unchanged_loss,
        "weighted_unchanged": unchanged_weight * unchanged_loss,
        "direct_similarity_gap": (
            direct_gap_values.detach().mean()
            if direct_gap_values.numel()
            else zero
        ),
        "unchanged_drift": (
            unchanged_drift_values.detach().mean()
            if unchanged_drift_values.numel()
            else zero
        ),
        "direct_node_count": direct_node_mask.bool().sum().detach(),
        "unchanged_node_count": unchanged_node_mask.bool().sum().detach(),
        "sample_count": torch.as_tensor(
            anchor.size(0),
            device=anchor.device,
            dtype=torch.long,
        ),
    }
