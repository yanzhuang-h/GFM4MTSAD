import torch
import torch.nn as nn
import torch.nn.functional as F

from models.anomalygfm.modeling_anomalygfm import (
    AnomalyGFM,
    normalize_adjacency,
)


class AdapterNullspaceInputResidual(nn.Module):
    """Project only the 128D information excluded by the learned 8D adapter."""

    def __init__(self, input_dim, output_dim, scale=1.0, eps=1e-6):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.scale = float(scale)
        self.eps = float(eps)
        self.weight = nn.Parameter(
            torch.zeros(self.output_dim, self.input_dim)
        )

    def adapter_nullspace(self, node_features, adapter_weight):
        if node_features.dim() != 3:
            raise ValueError(
                "High-dimensional entry expects node features [B, N, F], "
                f"got {tuple(node_features.shape)}"
            )
        if node_features.size(-1) != self.input_dim:
            raise ValueError(
                "High-dimensional entry input mismatch: "
                f"got={node_features.size(-1)}, expected={self.input_dim}"
            )
        if adapter_weight.dim() != 2 or adapter_weight.size(1) != self.input_dim:
            raise ValueError(
                "Adapter weight must be [D, F] with the same F as node features"
            )

        # The adapter remains trainable in v8.15, but it is frozen in this
        # candidate. Detaching makes that experimental boundary explicit.
        rows = adapter_weight.detach().to(
            device=node_features.device,
            dtype=node_features.dtype,
        )
        gram_inverse = torch.linalg.pinv(rows @ rows.transpose(0, 1))
        coordinates = F.linear(node_features, rows)
        dual_coordinates = torch.matmul(coordinates, gram_inverse)
        retained = torch.matmul(dual_coordinates, rows)
        return node_features - retained, retained

    def forward(self, node_features, adapter_weight):
        nullspace, retained = self.adapter_nullspace(
            node_features,
            adapter_weight,
        )
        rms = nullspace.square().mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        normalized = nullspace / rms
        residual = self.scale * F.linear(normalized, self.weight)
        return {
            "residual_300": residual,
            "nullspace_rms": nullspace.square().mean(dim=-1).sqrt(),
            "retained_rms": retained.square().mean(dim=-1).sqrt(),
        }


class HighDimResidualAnomalyGFM(AnomalyGFM):
    """AnomalyGFM with a zero-initialized residual at its first GCN layer."""

    def __init__(self, cfg, high_dim_input):
        initialization = str(
            getattr(cfg.ANOMALYGFM, "INITIALIZATION", "pretrained")
        )
        super().__init__(cfg, initialization=initialization)
        settings = cfg.SHARK.HIGH_DIM_GFM_ENTRY
        self.high_dim_entry = AdapterNullspaceInputResidual(
            input_dim=int(high_dim_input),
            output_dim=int(self.embedding_dim),
            scale=float(settings.SCALE),
            eps=float(settings.EPS),
        )

    def configure_tuning(self):
        super().configure_tuning()
        if hasattr(self, "high_dim_entry"):
            enabled = bool(self.cfg.SHARK.HIGH_DIM_GFM_ENTRY.ENABLE)
            for parameter in self.high_dim_entry.parameters():
                parameter.requires_grad = enabled

    def forward_with_high_dim(
        self,
        node_features,
        high_dim_features,
        adapter_weight,
        adjacency,
    ):
        if node_features.size(0) != high_dim_features.size(0):
            raise ValueError("Low- and high-dimensional batches must match")
        if node_features.size(1) != high_dim_features.size(1):
            raise ValueError("Low- and high-dimensional node counts must match")
        if node_features.size(-1) != self.input_dim:
            raise ValueError(
                f"AnomalyGFM expects {self.input_dim}D base node features"
            )

        entry = self.high_dim_entry(high_dim_features, adapter_weight)
        residual = entry["residual_300"]
        adjacency_norm = normalize_adjacency(adjacency)
        outputs = []
        for index in range(node_features.size(0)):
            outputs.append(
                self._forward_single_high_dim(
                    node_features[index:index + 1],
                    residual[index:index + 1],
                    adjacency_norm[index:index + 1],
                    adjacency[index],
                )
            )
        return {
            "node_scores": torch.cat(
                [output["node_scores"] for output in outputs], dim=0
            ),
            "graph_embedding": torch.cat(
                [output["graph_embedding"] for output in outputs], dim=0
            ),
            "node_embeddings": torch.cat(
                [output["node_embeddings"] for output in outputs], dim=0
            ),
            "high_dim_input_residual": residual,
            "high_dim_nullspace_rms": entry["nullspace_rms"],
            "high_dim_retained_rms": entry["retained_rms"],
        }

    def _forward_single_high_dim(
        self,
        node_features,
        input_residual,
        adjacency_norm,
        raw_adjacency,
    ):
        gcn1 = self.model.gcn1
        first_layer_features = gcn1.fc(node_features) + input_residual
        hidden = torch.bmm(adjacency_norm, first_layer_features)
        if gcn1.bias is not None:
            hidden = hidden + gcn1.bias
        hidden = gcn1.act(hidden)
        embedding = self.model.gcn2(hidden, adjacency_norm, sparse=False)

        normal_prompt = self.model.act(
            self.model.fc_normal_prompt(self.normal_prompt_raw)
        )
        abnormal_prompt = self.model.act(
            self.model.fc_abnormal_prompt(self.abnormal_prompt_raw)
        )

        raw_adjacency = raw_adjacency * (
            1
            - torch.eye(
                raw_adjacency.size(0),
                device=raw_adjacency.device,
                dtype=raw_adjacency.dtype,
            )
        )
        column_sum = raw_adjacency.sum(1, keepdim=True)
        residual_adjacency = raw_adjacency.div(column_sum)
        residual_adjacency[torch.isinf(residual_adjacency)] = 0
        residual_adjacency[torch.isnan(residual_adjacency)] = 0
        neighbors = torch.bmm(
            torch.unsqueeze(residual_adjacency, 0),
            embedding,
        )
        embedding_residual = self.node_adapter(embedding - neighbors)
        embedding_residual = F.normalize(
            embedding_residual,
            p=2,
            dim=-1,
        )
        normal_prompt = F.normalize(normal_prompt, p=2, dim=0)
        abnormal_prompt = F.normalize(abnormal_prompt, p=2, dim=0)
        score_normal = torch.matmul(embedding_residual, normal_prompt)
        score_abnormal = torch.matmul(embedding_residual, abnormal_prompt)
        node_scores = (
            torch.exp(score_abnormal)
            + self.beta * torch.exp(-score_normal)
        )
        if node_scores.dim() == 1:
            node_scores = node_scores.unsqueeze(0)
        return {
            "node_scores": node_scores,
            "node_embeddings": embedding_residual,
            "graph_embedding": embedding_residual.mean(dim=1),
        }

    def high_dim_entry_diagnostics(self):
        weight = self.high_dim_entry.weight.detach().float()
        return {
            "high_dim_entry_enabled": bool(
                self.cfg.SHARK.HIGH_DIM_GFM_ENTRY.ENABLE
            ),
            "high_dim_entry_input_dim": self.high_dim_entry.input_dim,
            "high_dim_entry_output_dim": self.high_dim_entry.output_dim,
            "high_dim_entry_scale": self.high_dim_entry.scale,
            "high_dim_entry_weight_norm": float(weight.norm().cpu()),
            "high_dim_entry_nonzero_count": int(
                torch.count_nonzero(weight).cpu()
            ),
        }
