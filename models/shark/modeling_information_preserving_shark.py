from models.shark.information_preserving_residual import (
    InformationPreservingResidualAdapter,
)
from models.shark.modeling_unified_shark import UnifiedSHARK


class InformationPreservingUnifiedSHARK(UnifiedSHARK):
    """Unified SHARK plus one backend-neutral information residual."""

    def __init__(self, cfg):
        super().__init__(cfg)
        settings = cfg.SHARK.INFORMATION_PRESERVING_RESIDUAL
        spec = self.gfm.backend_spec()
        self.information_residual = InformationPreservingResidualAdapter(
            source_dim=int(settings.SOURCE_DIM),
            embedding_dim=int(spec.embedding_dim),
            rank=int(settings.RANK),
            scale=float(settings.SCALE),
            eps=float(settings.EPS),
            gate_init=float(settings.GATE_INIT),
        )
        self.freeze_for_backend_tuning()

    def adapt_backend_outputs(self, graph_outputs, backend_inputs):
        if backend_inputs.source_node_features is None:
            raise RuntimeError(
                "Information residual requires shared source node features"
            )
        if backend_inputs.source_projection_weight is None:
            raise RuntimeError(
                "Information residual requires the Universal Bridge weight"
            )
        residual_outputs = self.information_residual(
            backend_inputs.source_node_features,
            backend_inputs.source_projection_weight,
            graph_outputs["node_embeddings"],
        )
        outputs = dict(graph_outputs)
        outputs.update(residual_outputs)
        return outputs

    def forward(self, x, labels=None, return_dict=True):
        outputs = super().forward(x, labels=labels, return_dict=True)
        return self._add_information_preserving_contract(outputs, return_dict)

    def forward_from_frozen_frontend(
        self,
        frontend,
        labels=None,
        return_dict=True,
    ):
        """Run the trainable SHARK path from one cached frozen frontend."""
        if not self._bridge_initialized:
            raise RuntimeError("Unified Token Bridge must be initialized first")
        outputs = self._forward_from_frontend(
            frontend,
            labels=labels,
            return_dict=True,
        )
        outputs["formal_module_path"] = (
            "sensor_graph_encoder",
            "universal_gfm_adapter",
            "normal_referenced_node_to_time",
        )
        return self._add_information_preserving_contract(outputs, return_dict)

    @staticmethod
    def _add_information_preserving_contract(outputs, return_dict):
        outputs["universal_adapter_components"] = (
            "differentiable_token_bridge",
            "information_preserving_residual_adapter",
            "gfm_native_peft",
        )
        if return_dict:
            return outputs
        return outputs["window_score"]

    def trainable_parameter_summary(self):
        summary = dict(super().trainable_parameter_summary())
        summary.update(self.information_residual.diagnostics())
        summary.update({
            "formal_architecture": "three_module_unified_shark_ipra_v1",
            "information_residual_location": "shared_gfm_output_boundary",
        })
        return summary
