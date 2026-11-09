import torch

from models.gfm4mtsad.differentiable_token_bridge import (
    build_differentiable_token_bridge,
)
from models.gfm4mtsad.modeling_gfm_backend import GFMBackendGFM4MTSAD


class UnifiedGFM4MTSAD(GFMBackendGFM4MTSAD):
    """The single three-module GFM4MTSAD path used by the formal benchmark."""

    def __init__(self, cfg):
        super().__init__(cfg)
        spec = self.gfm.backend_spec()
        self.gfm_adapter = build_differentiable_token_bridge(
            cfg,
            source_dim=int(cfg.GFM4MTSAD.TOKEN_BRIDGE.SOURCE_DIM),
            token_dim=int(spec.token_dim),
        )
        self._bridge_initialized = False
        self.freeze_for_backend_tuning()

    @torch.no_grad()
    def fit_feature_adapter(self, loader, max_batches=None):
        """Initialize the shared Bridge from normal Time-to-Node features."""
        device = next(self.parameters()).device
        count = 0
        feature_sum = torch.zeros(
            self.gfm_adapter.source_dim,
            dtype=torch.float64,
        )
        feature_cross_product = torch.zeros(
            self.gfm_adapter.source_dim,
            self.gfm_adapter.source_dim,
            dtype=torch.float64,
        )
        was_training = self.training
        self.eval()
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            windows = batch[0] if isinstance(batch, (tuple, list)) else batch
            windows = windows.to(device, non_blocking=True).float()
            rows = self.imlp(windows).reshape(-1, self.gfm_adapter.source_dim)
            rows = rows[torch.isfinite(rows).all(dim=1)].double().cpu()
            if rows.numel() == 0:
                continue
            count += int(rows.size(0))
            feature_sum += rows.sum(dim=0)
            feature_cross_product += rows.transpose(0, 1) @ rows
        if count < 2:
            raise RuntimeError("Unified Token Bridge received fewer than two rows")
        self.gfm_adapter.fit_pca_from_moments(
            count,
            feature_sum,
            feature_cross_product,
        )
        self._bridge_initialized = True
        self.train(was_training)
        self.freeze_for_backend_tuning()
        return {
            "method": "normal_only_streaming_pca",
            "row_count": count,
            "source_dim": self.gfm_adapter.source_dim,
            "token_dim": self.gfm_adapter.token_dim,
        }

    def forward(self, x, labels=None, return_dict=True):
        if not self._bridge_initialized:
            raise RuntimeError("Unified Token Bridge must be initialized first")
        outputs = super().forward(x, labels=labels, return_dict=True)
        outputs["formal_module_path"] = (
            "sensor_graph_encoder",
            "universal_gfm_adapter",
            "normal_referenced_node_to_time",
        )
        if return_dict:
            return outputs
        return outputs["window_score"]

    def trainable_parameter_summary(self):
        summary = dict(super().trainable_parameter_summary())
        summary.update({
            "formal_architecture": "three_module_unified_gfm4mtsad_v1",
            "bridge_initialized": self._bridge_initialized,
            "bridge_parameter_count": sum(
                parameter.numel()
                for parameter in self.gfm_adapter.parameters()
            ),
            "time_to_node_frozen": not any(
                parameter.requires_grad for parameter in self.imlp.parameters()
            ),
            "gfm_backbone_frozen": all(
                not parameter.requires_grad
                for name, parameter in self.gfm.named_parameters()
                if name not in self.gfm.adaptation_parameter_names()
            ),
        })
        return summary
