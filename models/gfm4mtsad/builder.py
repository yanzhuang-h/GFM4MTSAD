import torch
import torch.nn as nn


class Pearson_G(nn.Module):
    def __init__(self, cfg):
        super(Pearson_G, self).__init__()
        self.top_k = cfg.GRAPHBUILDER.TOP_K
        self.self_loop = cfg.GRAPHBUILDER.SELF_LOOP
        self.absolute = cfg.GRAPHBUILDER.ABSOLUTE
        self.eps = cfg.GRAPHBUILDER.EPS
        self.reference_prior_enable = bool(
            getattr(cfg.GRAPHBUILDER, "REFERENCE_PRIOR_ENABLE", False)
        )
        self.reference_prior_blend = float(
            getattr(cfg.GRAPHBUILDER, "REFERENCE_PRIOR_BLEND", 0.25)
        )
        if not 0.0 <= self.reference_prior_blend <= 1.0:
            raise ValueError(
                "GRAPHBUILDER.REFERENCE_PRIOR_BLEND must be in [0, 1]"
            )

        # Keep the new state out of legacy models.  This is important because
        # old checkpoints are loaded strictly and must remain reproducible.
        if self.reference_prior_enable:
            node_count = int(getattr(cfg.DATA, "N_VAR", 0))
            if node_count <= 0:
                raise ValueError(
                    "GRAPHBUILDER.REFERENCE_PRIOR_ENABLE requires DATA.N_VAR"
                )
            self.reference_node_count = node_count
            self.register_buffer(
                "reference_edge_frequency",
                torch.zeros(node_count, node_count),
            )
            self.register_buffer(
                "reference_edge_count",
                torch.tensor(0, dtype=torch.long),
            )
            self.register_buffer(
                "reference_fitted",
                torch.tensor(False, dtype=torch.bool),
            )
        else:
            self.reference_node_count = None

    def _validate_node_features(self, node_features):
        if node_features.ndim != 3:
            raise ValueError(
                "Pearson graph expects node features shaped [B, N, T], "
                f"got {tuple(node_features.shape)}"
            )
        if self.reference_prior_enable:
            observed = int(node_features.size(1))
            if observed != self.reference_node_count:
                raise ValueError(
                    "Graph reference node count mismatch: "
                    f"expected={self.reference_node_count}, observed={observed}"
                )

    def _correlation(self, node_features):
        self._validate_node_features(node_features)
        x = node_features - node_features.mean(dim=-1, keepdim=True)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        corr = torch.bmm(x, x.transpose(1, 2))
        if self.absolute:
            corr = corr.abs()
        n_nodes = corr.size(1)
        eye = torch.eye(
            n_nodes,
            dtype=torch.bool,
            device=corr.device,
        ).unsqueeze(0)
        corr = corr.masked_fill(eye, 0.0)
        return corr, eye

    def _select(self, corr, eye):
        n_nodes = corr.size(1)
        if self.top_k > 0 and self.top_k < n_nodes:
            _, indices = torch.topk(corr, k=self.top_k, dim=-1)
            adj = torch.zeros_like(corr)
            adj.scatter_(dim=-1, index=indices, src=corr.gather(dim=-1, index=indices))
            adj = torch.maximum(adj, adj.transpose(1, 2))
            selected = torch.zeros_like(corr, dtype=torch.bool)
            selected.scatter_(dim=-1, index=indices, value=True)
            selected = selected | selected.transpose(1, 2)
        else:
            adj = corr
            selected = (~eye).expand(corr.size(0), -1, -1)

        if self.self_loop:
            adj = adj + eye.to(dtype=adj.dtype)
        return adj, selected

    @torch.no_grad()
    def reset_reference(self):
        """Clear the normal train-window edge-frequency reference."""
        if not self.reference_prior_enable:
            return
        self.reference_edge_frequency.zero_()
        self.reference_edge_count.zero_()
        self.reference_fitted.fill_(False)

    @torch.no_grad()
    def update_reference(self, node_features):
        """Update the edge-frequency reference from normal windows only."""
        if not self.reference_prior_enable:
            return
        corr, eye = self._correlation(node_features.detach())
        _, selected = self._select(corr, eye)
        batch_frequency = selected.float().mean(dim=0)
        batch_count = int(selected.size(0))
        if batch_count <= 0:
            return
        old_count = int(self.reference_edge_count.item())
        total_count = old_count + batch_count
        old_weight = old_count / total_count
        new_weight = batch_count / total_count
        self.reference_edge_frequency.mul_(old_weight).add_(
            batch_frequency.to(
                device=self.reference_edge_frequency.device,
                dtype=self.reference_edge_frequency.dtype,
            )
            * new_weight
        )
        self.reference_edge_count.fill_(total_count)
        self.reference_fitted.fill_(True)

    def reference_prior_diagnostics(self):
        if not self.reference_prior_enable:
            return {
                "reference_prior_enabled": False,
                "reference_prior_fitted": False,
            }
        frequency = self.reference_edge_frequency.detach().float()
        n_nodes = frequency.size(0)
        non_self = ~torch.eye(n_nodes, dtype=torch.bool, device=frequency.device)
        values = frequency[non_self]
        return {
            "reference_prior_enabled": True,
            "reference_prior_fitted": bool(self.reference_fitted.item()),
            "reference_prior_blend": self.reference_prior_blend,
            "reference_prior_node_count": n_nodes,
            "reference_prior_window_count": int(self.reference_edge_count.item()),
            "reference_prior_edge_frequency_mean": float(values.mean().cpu())
            if values.numel()
            else 0.0,
            "reference_prior_edge_frequency_max": float(values.max().cpu())
            if values.numel()
            else 0.0,
        }

    def forward(self, node_features):
        corr, eye = self._correlation(node_features)
        if self.reference_prior_enable and bool(self.reference_fitted.item()):
            prior = self.reference_edge_frequency.detach().to(
                device=corr.device,
                dtype=corr.dtype,
            )
            adjusted = (1.0 - self.reference_prior_blend) + (
                self.reference_prior_blend * prior
            )
            # The diagonal is inserted as a self-loop after edge selection;
            # keep it neutral even if a reference was partially fitted.
            adjusted = adjusted.clone()
            adjusted.fill_diagonal_(1.0)
            corr = corr * adjusted.unsqueeze(0)
        adj, _ = self._select(corr, eye)
        return adj


def build_graph(cfg):
    method = cfg.GRAPHBUILDER.METHOD
    if method == "pearson":
        return Pearson_G(cfg)
    raise ValueError(f"Unsupported GRAPHBUILDER.METHOD: {method}")
