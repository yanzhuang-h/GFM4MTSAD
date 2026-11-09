import torch
import torch.nn.functional as F

from models.anomalygfm.modeling_anomalygfm import normalize_adjacency

from models.gfm4mtsad.gfm_backend import (
    GFMBackendContract,
    GFMBackendInput,
    GFMBackendSpec,
    NODE_ANOMALY_EVIDENCE_SEMANTICS,
    validate_backend_output,
)
from models.gfm4mtsad.high_dim_gfm_entry import HighDimResidualAnomalyGFM


class AnomalyGFMHighDimBackend(
    HighDimResidualAnomalyGFM,
    GFMBackendContract,
):
    """Thin backend preserving the existing AnomalyGFM candidate exactly."""

    def backend_spec(self):
        native_peft = bool(
            getattr(
                getattr(self.cfg.GFM4MTSAD, "GFM_BACKEND", None),
                "TRAIN_NATIVE_PEFT",
                False,
            )
        )
        high_dim = bool(self.cfg.GFM4MTSAD.HIGH_DIM_GFM_ENTRY.ENABLE)
        if native_peft and high_dim:
            strategy = "native_prompt_plus_high_dim_residual"
        elif native_peft:
            strategy = "native_normal_abnormal_prompt"
        elif high_dim:
            strategy = "adapter_nullspace_first_hidden_residual"
        else:
            strategy = "frozen"
        return GFMBackendSpec(
            name=(
                f"anomalygfm_{self.initialization}_high_dim_entry"
                if high_dim
                else f"anomalygfm_{self.initialization}_native_prompt"
            ),
            token_dim=int(self.input_dim),
            embedding_dim=int(self.embedding_dim),
            uses_source_node_features=high_dim,
            adaptation_strategy=strategy,
            node_score_semantics=NODE_ANOMALY_EVIDENCE_SEMANTICS,
        )

    def forward_backend(self, inputs: GFMBackendInput):
        high_dim = bool(self.cfg.GFM4MTSAD.HIGH_DIM_GFM_ENTRY.ENABLE)
        if high_dim:
            if inputs.source_node_features is None:
                raise ValueError("AnomalyGFM backend requires source node features")
            if inputs.source_projection_weight is None:
                raise ValueError(
                    "AnomalyGFM backend requires source projection weight"
                )
            native = self.forward_with_high_dim(
                inputs.node_tokens,
                inputs.source_node_features,
                inputs.source_projection_weight,
                inputs.adjacency,
            )
            outputs = {
                "node_scores": native["node_scores"],
                "node_embeddings": native["node_embeddings"],
                "graph_embedding": native["graph_embedding"],
                "gfm_high_dim_input_residual": native[
                    "high_dim_input_residual"
                ],
                "gfm_high_dim_nullspace_rms": native[
                    "high_dim_nullspace_rms"
                ],
                "gfm_high_dim_retained_rms": native[
                    "high_dim_retained_rms"
                ],
            }
        else:
            native = self(inputs.node_tokens, inputs.adjacency)
            outputs = {
                "node_scores": native["node_scores"],
                "node_embeddings": native["node_embeddings"],
                "graph_embedding": native["graph_embedding"],
            }
        return validate_backend_output(outputs, inputs, self.backend_spec())

    def forward_backend_with_topologies(
        self,
        inputs: GFMBackendInput,
        gcn1_adjacency: torch.Tensor,
        downstream_adjacency: torch.Tensor,
    ):
        """Run the registered AnomalyGFM with explicit layer topologies.

        This is the production counterpart of the frozen E4 reliability audit:
        GCN1 may use a different graph from GCN2, while GCN2 and the native
        post-GCN neighbor residual share one downstream topology.
        """
        if bool(self.cfg.GFM4MTSAD.HIGH_DIM_GFM_ENTRY.ENABLE):
            raise ValueError(
                "Topology reliability routing requires the registered "
                "zero-input-IPRA AnomalyGFM path"
            )
        expected = tuple(inputs.adjacency.shape)
        for name, adjacency in (
            ("gcn1", gcn1_adjacency),
            ("downstream", downstream_adjacency),
        ):
            if tuple(adjacency.shape) != expected:
                raise ValueError(
                    f"{name} topology shape {tuple(adjacency.shape)} does not "
                    f"match registered adjacency {expected}"
                )
            if not torch.isfinite(adjacency).all():
                raise ValueError(f"{name} topology must be finite")

        hidden = self.model.gcn1(
            inputs.node_tokens,
            normalize_adjacency(gcn1_adjacency),
        )
        embedding = self.model.gcn2(
            hidden,
            normalize_adjacency(downstream_adjacency),
        )

        node_count = downstream_adjacency.size(1)
        eye = torch.eye(
            node_count,
            device=downstream_adjacency.device,
            dtype=downstream_adjacency.dtype,
        ).unsqueeze(0)
        off_diagonal = downstream_adjacency * (1.0 - eye)
        row_sum = off_diagonal.sum(dim=-1, keepdim=True)
        row_normalized = off_diagonal / row_sum.clamp_min(1e-12)
        row_normalized = torch.where(
            row_sum > 0,
            row_normalized,
            torch.zeros_like(row_normalized),
        )
        neighbors = torch.bmm(row_normalized, embedding)
        embedding_residual = self.node_adapter(embedding - neighbors)
        embedding_residual = F.normalize(embedding_residual, p=2, dim=-1)

        normal_prompt = F.normalize(
            self.model.act(
                self.model.fc_normal_prompt(self.normal_prompt_raw)
            ),
            p=2,
            dim=0,
        )
        abnormal_prompt = F.normalize(
            self.model.act(
                self.model.fc_abnormal_prompt(self.abnormal_prompt_raw)
            ),
            p=2,
            dim=0,
        )
        score_normal = torch.matmul(embedding_residual, normal_prompt)
        score_abnormal = torch.matmul(embedding_residual, abnormal_prompt)
        node_scores = (
            torch.exp(score_abnormal)
            + self.beta * torch.exp(-score_normal)
        )
        outputs = {
            "node_scores": node_scores,
            "node_embeddings": embedding_residual,
            "graph_embedding": embedding_residual.mean(dim=1),
        }
        return validate_backend_output(outputs, inputs, self.backend_spec())

    def adaptation_parameter_names(self):
        names = []
        if bool(self.cfg.GFM4MTSAD.HIGH_DIM_GFM_ENTRY.ENABLE):
            names.append("high_dim_entry.weight")
        if bool(
            getattr(
                getattr(self.cfg.GFM4MTSAD, "GFM_BACKEND", None),
                "TRAIN_NATIVE_PEFT",
                False,
            )
        ):
            names.extend(("normal_prompt_raw", "abnormal_prompt_raw"))
        return tuple(names)

    def adaptation_is_identity(self):
        return int(torch.count_nonzero(self.high_dim_entry.weight)) == 0

    def train(self, mode=True):
        # Native prompt tensors have no train/eval-specific behavior. Keep the
        # frozen GFM BatchNorm state fixed while gradients update only PEFT.
        super().train(False)
        return self
