import torch
import torch.nn as nn
import torch.nn.functional as F


class InformationPreservingResidualAdapter(nn.Module):
    """Recover Bridge-excluded source information at the GFM output boundary."""

    def __init__(
        self,
        source_dim,
        embedding_dim,
        rank,
        scale=1.0,
        eps=1e-6,
        gate_init=0.0,
    ):
        super().__init__()
        self.source_dim = int(source_dim)
        self.embedding_dim = int(embedding_dim)
        self.rank = int(rank)
        self.scale = float(scale)
        self.eps = float(eps)
        if not 0 < self.rank <= self.source_dim:
            raise ValueError(
                "Residual rank must be in [1, source_dim], "
                f"got rank={self.rank}, source_dim={self.source_dim}"
            )

        self.down = nn.Linear(self.source_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.embedding_dim, bias=False)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        nn.init.orthogonal_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def bridge_nullspace(self, source_node_features, bridge_weight):
        if source_node_features.dim() != 3:
            raise ValueError(
                "Residual adapter expects source features [B, N, F], "
                f"got {tuple(source_node_features.shape)}"
            )
        if source_node_features.size(-1) != self.source_dim:
            raise ValueError(
                f"Residual adapter expected {self.source_dim}D source features, "
                f"got {source_node_features.size(-1)}D"
            )
        if bridge_weight.dim() != 2 or bridge_weight.size(1) != self.source_dim:
            raise ValueError(
                "Bridge weight must have shape [token_dim, source_dim]"
            )

        # The Bridge has its own supervised path. Detaching its row space keeps
        # this side channel from changing that optimization objective.
        rows = bridge_weight.detach().to(
            device=source_node_features.device,
            dtype=source_node_features.dtype,
        )
        gram_inverse = torch.linalg.pinv(rows @ rows.transpose(0, 1))
        coordinates = F.linear(source_node_features, rows)
        retained = torch.matmul(coordinates, gram_inverse) @ rows
        return source_node_features - retained, retained

    def forward(
        self,
        source_node_features,
        bridge_weight,
        backend_node_embeddings,
    ):
        if backend_node_embeddings.dim() != 3:
            raise ValueError(
                "Backend embeddings must have shape [B, N, H]"
            )
        if backend_node_embeddings.shape[:2] != source_node_features.shape[:2]:
            raise ValueError(
                "Source features and backend embeddings must share B and N"
            )
        if backend_node_embeddings.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Residual adapter expected {self.embedding_dim}D embeddings, "
                f"got {backend_node_embeddings.size(-1)}D"
            )

        nullspace, retained = self.bridge_nullspace(
            source_node_features,
            bridge_weight,
        )
        nullspace_mean_square = nullspace.square().mean(dim=-1, keepdim=True)
        nullspace_rms = nullspace_mean_square.sqrt()
        # Clamping after sqrt preserves the forward value but still backprops
        # through sqrt(0), whose derivative can poison a trainable frontend.
        # Clamp the squared value first so zero-nullspace nodes have finite
        # gradients while the nonzero forward contract stays unchanged.
        normalized = nullspace / nullspace_mean_square.clamp_min(self.eps ** 2).sqrt()
        latent = F.gelu(self.down(normalized))
        gate = torch.sigmoid(self.gate_logit)
        residual = self.scale * gate * self.up(latent)
        compensated = backend_node_embeddings + residual
        return {
            "node_embeddings": compensated,
            "graph_embedding": compensated.mean(dim=1),
            "information_residual": residual,
            "information_residual_nullspace_rms": nullspace_rms.squeeze(-1),
            "information_residual_retained_rms": (
                retained.square().mean(dim=-1).sqrt()
            ),
            "information_residual_gate": gate,
        }

    def is_zero_output(self):
        return int(torch.count_nonzero(self.up.weight.detach())) == 0

    def diagnostics(self):
        return {
            "information_residual_source_dim": self.source_dim,
            "information_residual_embedding_dim": self.embedding_dim,
            "information_residual_rank": self.rank,
            "information_residual_scale": self.scale,
            "information_residual_gate": float(
                torch.sigmoid(self.gate_logit.detach()).cpu()
            ),
            "information_residual_down_norm": float(
                self.down.weight.detach().float().norm().cpu()
            ),
            "information_residual_up_norm": float(
                self.up.weight.detach().float().norm().cpu()
            ),
            "information_residual_zero_output": self.is_zero_output(),
        }
