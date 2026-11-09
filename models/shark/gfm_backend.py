from dataclasses import dataclass
from typing import Dict, Optional

import torch


NODE_ANOMALY_EVIDENCE_SEMANTICS = (
    "nonnegative_higher_is_more_anomalous"
)


@dataclass(frozen=True)
class GFMBackendSpec:
    name: str
    token_dim: int
    embedding_dim: int
    uses_source_node_features: bool
    adaptation_strategy: str
    uses_declared_node_tokens: bool = True
    # 默认为 unspecified，避免把任意表征型 GFM 自动视为具有节点异常语义。
    # 只有后端显式声明统一语义后，才能参加原生证据机制对比。
    node_score_semantics: str = "unspecified"


@dataclass(frozen=True)
class GFMBackendInput:
    node_tokens: torch.Tensor
    adjacency: torch.Tensor
    source_node_features: Optional[torch.Tensor] = None
    source_projection_weight: Optional[torch.Tensor] = None
    time_trend_features: Optional[torch.Tensor] = None
    time_seasonal_features: Optional[torch.Tensor] = None


REQUIRED_BACKEND_OUTPUTS = (
    "node_scores",
    "node_embeddings",
    "graph_embedding",
)


class GFMBackendContract:
    """Interface implemented by each GFM-specific thin wrapper."""

    def backend_spec(self):
        raise NotImplementedError

    def forward_backend(self, inputs):
        raise NotImplementedError

    def adaptation_parameter_names(self):
        raise NotImplementedError

    def adaptation_is_identity(self):
        raise NotImplementedError


def validate_backend_output(
    outputs: Dict[str, torch.Tensor],
    inputs: GFMBackendInput,
    spec: GFMBackendSpec,
):
    if inputs.node_tokens.dim() != 3:
        raise ValueError("GFM backend node tokens must have shape [B, N, D]")
    if inputs.node_tokens.size(-1) != spec.token_dim:
        raise ValueError(
            f"GFM backend expected {spec.token_dim}D node tokens, "
            f"got {inputs.node_tokens.size(-1)}D"
        )
    if inputs.adjacency.dim() != 3:
        raise ValueError("GFM backend adjacency must have shape [B, N, N]")
    if tuple(inputs.adjacency.shape) != (
        inputs.node_tokens.size(0),
        inputs.node_tokens.size(1),
        inputs.node_tokens.size(1),
    ):
        raise ValueError("GFM backend adjacency does not match node tokens")

    missing = [name for name in REQUIRED_BACKEND_OUTPUTS if name not in outputs]
    if missing:
        raise ValueError(f"GFM backend output is missing: {missing}")

    batch_size, node_count = inputs.node_tokens.shape[:2]
    expected = {
        "node_scores": (batch_size, node_count),
        "node_embeddings": (batch_size, node_count, spec.embedding_dim),
        "graph_embedding": (batch_size, spec.embedding_dim),
    }
    for name, shape in expected.items():
        value = outputs[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"GFM backend output {name} must be a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(
                f"GFM backend output {name} has shape {tuple(value.shape)}, "
                f"expected {shape}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"GFM backend output {name} must be finite")
    if (
        spec.node_score_semantics == NODE_ANOMALY_EVIDENCE_SEMANTICS
        and (outputs["node_scores"] < 0).any()
    ):
        raise ValueError(
            "GFM backend declared nonnegative anomaly evidence but returned "
            "negative node scores"
        )
    return outputs


def require_node_anomaly_evidence(spec: GFMBackendSpec):
    """拒绝把没有统一节点异常证据语义的 GFM 静默接入局部对比。"""
    if spec.node_score_semantics != NODE_ANOMALY_EVIDENCE_SEMANTICS:
        raise ValueError(
            "Native evidence contrast requires backend node scores with "
            f"semantics={NODE_ANOMALY_EVIDENCE_SEMANTICS}, got "
            f"{spec.node_score_semantics!r} from {spec.name}"
        )
    return spec
