from pathlib import Path

import torch
import torch.nn.functional as F

from models.gfm4mtsad.high_dim_gfm_entry import HighDimResidualAnomalyGFM
from models.gfm4mtsad.modeling_gfm4mtsad import GFM4MTSAD


class HighDimGFMEntryGFM4MTSAD(GFM4MTSAD):
    """Frozen v8.15 with one trainable adapter-nullspace GFM input residual."""

    def __init__(self, cfg):
        self._frozen_v815_loaded = False
        self._frozen_v815_checkpoint = None
        super().__init__(cfg)
        self.freeze_for_high_dim_entry_tuning()

    def _init_gfm(self, cfg):
        if str(cfg.GFM4MTSAD.GFM) != "AnomalyGFM":
            raise ValueError("High-dimensional entry requires AnomalyGFM")
        return HighDimResidualAnomalyGFM(
            cfg,
            high_dim_input=int(cfg.DLINEAR.FEATURE_DIM),
        )

    def freeze_for_high_dim_entry_tuning(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        if bool(self.cfg.GFM4MTSAD.HIGH_DIM_GFM_ENTRY.ENABLE):
            self.gfm.high_dim_entry.weight.requires_grad = True

    def load_frozen_v815_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Frozen v8.15 checkpoint not found: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint["model_state"]
        message = self.load_state_dict(state, strict=False)
        expected_missing = {"gfm.high_dim_entry.weight"}
        if set(message.missing_keys) != expected_missing:
            raise RuntimeError(
                "Frozen v8.15 checkpoint has unexpected missing keys: "
                f"{message.missing_keys}"
            )
        if message.unexpected_keys:
            raise RuntimeError(
                "Frozen v8.15 checkpoint has unexpected keys: "
                f"{message.unexpected_keys}"
            )
        if int(torch.count_nonzero(self.gfm.high_dim_entry.weight)) != 0:
            raise RuntimeError("High-dimensional entry must remain zero at load")

        self._frozen_v815_loaded = True
        self._frozen_v815_checkpoint = str(checkpoint_path.resolve())
        if hasattr(self.gfm_adapter, "fitted"):
            self.gfm_adapter.fitted = True
        self.freeze_for_high_dim_entry_tuning()
        return {
            "checkpoint": self._frozen_v815_checkpoint,
            "checkpoint_epoch_index": int(checkpoint.get("epoch", -1)),
            "missing_keys": sorted(message.missing_keys),
            "unexpected_keys": sorted(message.unexpected_keys),
        }

    @torch.no_grad()
    def fit_feature_adapter(self, loader, max_batches=None):
        if not self._frozen_v815_loaded:
            return super().fit_feature_adapter(loader, max_batches=max_batches)
        return {
            "status": "PRESERVED_FROM_FROZEN_V815_CHECKPOINT",
            "checkpoint": self._frozen_v815_checkpoint,
        }

    def trainable_parameter_summary(self):
        summary = dict(super().trainable_parameter_summary())
        diagnostics = self.gfm.high_dim_entry_diagnostics()
        summary.update(diagnostics)
        summary["candidate_training_mode"] = "frozen_v815_entry_only"
        summary["frozen_v815_loaded"] = bool(self._frozen_v815_loaded)
        summary["frozen_v815_checkpoint"] = self._frozen_v815_checkpoint
        return summary

    def peft_diagnostics(self):
        diagnostics = dict(super().peft_diagnostics())
        diagnostics.update(self.gfm.high_dim_entry_diagnostics())
        diagnostics["candidate_training_mode"] = "frozen_v815_entry_only"
        diagnostics["frozen_v815_loaded"] = bool(self._frozen_v815_loaded)
        diagnostics["frozen_v815_checkpoint"] = self._frozen_v815_checkpoint
        return diagnostics

    def forward(self, x, labels=None, return_dict=True):
        x = x.float()
        node_features = self.imlp(x)
        adjacency = self.graph_builder(node_features.detach())
        base_gfm_node_features = self.gfm_adapter(node_features)
        gfm_node_features = self.sensor_graph_prompt(
            base_gfm_node_features,
            adjacency,
        )
        graph_outputs = self.gfm.forward_with_high_dim(
            gfm_node_features,
            node_features,
            self.gfm_adapter.proj.weight,
            adjacency,
        )
        raw_node_scores = graph_outputs["node_scores"]
        node_embeddings = graph_outputs["node_embeddings"]
        graph_embedding = graph_outputs["graph_embedding"]
        head_outputs = self.head(
            node_embeddings=node_embeddings,
            raw_node_scores=raw_node_scores,
            graph_embedding=graph_embedding,
        )
        window_score = head_outputs["window_score"]

        if labels is None:
            loss = window_score.sum() * 0.0
        else:
            labels = labels.float().view_as(window_score)
            loss = F.binary_cross_entropy_with_logits(window_score, labels)

        outputs = dict(head_outputs)
        outputs.update({
            "logits": window_score,
            "window_score": window_score,
            "loss": loss,
            "losses": (loss,),
            "metrics": (loss.detach(),),
            "node_features": node_features,
            "graph_node_features": node_features,
            "gfm_node_features_base": base_gfm_node_features,
            "gfm_node_features": gfm_node_features,
            "gfm_high_dim_input_residual": graph_outputs[
                "high_dim_input_residual"
            ],
            "gfm_high_dim_nullspace_rms": graph_outputs[
                "high_dim_nullspace_rms"
            ],
            "gfm_high_dim_retained_rms": graph_outputs[
                "high_dim_retained_rms"
            ],
            "adjacency": adjacency,
            "raw_node_scores": raw_node_scores,
            "node_scores_raw": raw_node_scores,
            "native_node_scores": raw_node_scores,
            "node_embeddings": node_embeddings,
            "graph_embedding": graph_embedding,
            "backend_name": "anomalygfm_high_dim_entry",
        })
        if return_dict:
            return outputs
        return window_score
