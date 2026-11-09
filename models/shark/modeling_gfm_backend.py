from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.shark.adapter import build_adapter, build_sensor_graph_prompt
from models.shark.builder import build_graph
from models.shark.gfm_backend import GFMBackendInput
from models.shark.gfm_backend_registry import build_gfm_backend
from models.shark.head import SharkHead
from models.shark.modeling_anchored_mil_scale import AnchoredMILScaleSHARK
from models.shark.modeling_shark import SHARK


class GFMBackendSHARK(AnchoredMILScaleSHARK):
    """Backend-neutral SHARK shell with compatible state-dict boundaries."""

    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        self._frozen_backend_loaded = False
        self._frozen_backend_checkpoint = None
        self._frozen_v815_loaded = False
        self._frozen_v815_checkpoint = None
        self._shared_frontend_loaded = False
        self._shared_frontend_checkpoint = None
        if str(cfg.SHARK.HEAD.NAME) != "SharkHead":
            raise ValueError("Backend SHARK requires SHARK.HEAD.NAME=SharkHead")

        self.imlp = self._init_imlp()
        self.graph_builder = build_graph(cfg)
        self.gfm = self._init_gfm(cfg)
        spec = self.gfm.backend_spec()
        self.gfm_adapter = build_adapter(
            cfg,
            input_dim=int(cfg.DLINEAR.FEATURE_DIM),
            output_dim=spec.token_dim,
        )
        self.sensor_graph_prompt = build_sensor_graph_prompt(
            cfg,
            feature_dim=spec.token_dim,
        )
        self.head = SharkHead(cfg, embedding_dim=spec.embedding_dim)
        self._freeze_backbone()
        self.freeze_for_backend_tuning()

        sensor_count = int(cfg.DATA.N_VAR)
        self.register_buffer("mil_scale_anchor", torch.ones(sensor_count))
        self.register_buffer(
            "mil_scale_anchor_fitted",
            torch.tensor(False, dtype=torch.bool),
        )
        self._last_mil_scale_action = "uninitialized"

    def _init_gfm(self, cfg):
        return build_gfm_backend(
            cfg,
            source_feature_dim=int(cfg.DLINEAR.FEATURE_DIM),
        )

    def freeze_for_backend_tuning(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        backend_cfg = getattr(self.cfg.SHARK, "GFM_BACKEND", None)
        training_mode = str(
            getattr(
                backend_cfg,
                "TRAINING_MODE",
                "backend_adaptation_only",
            )
        )
        if training_mode == "frozen_backend_shared_outer":
            prefixes = tuple(
                str(prefix)
                for prefix in getattr(
                    backend_cfg,
                    "TRAINABLE_PREFIXES",
                    (),
                )
            )
            if not prefixes:
                raise ValueError(
                    "Shared-outer training requires explicit trainable prefixes"
                )
            if any(prefix.startswith("gfm.") for prefix in prefixes):
                raise ValueError("Shared-outer training must keep the GFM frozen")
            matched = []
            for name, parameter in self.named_parameters():
                if name.startswith(prefixes):
                    parameter.requires_grad = True
                    matched.append(name)
            if not matched:
                raise ValueError("Shared-outer training matched no parameters")
            if bool(
                getattr(
                    backend_cfg,
                    "TRAIN_BACKEND_ADAPTATION",
                    False,
                )
            ):
                backend_parameters = dict(self.gfm.named_parameters())
                for name in self.gfm.adaptation_parameter_names():
                    backend_parameters[name].requires_grad = True
            return
        if training_mode != "backend_adaptation_only":
            raise ValueError(
                f"Unsupported GFM backend training mode: {training_mode}"
            )
        declared = tuple(self.gfm.adaptation_parameter_names())
        backend_parameters = dict(self.gfm.named_parameters())
        missing = sorted(set(declared).difference(backend_parameters))
        if missing:
            raise RuntimeError(
                f"GFM backend declared unknown adaptation parameters: {missing}"
            )
        for name in declared:
            backend_parameters[name].requires_grad = True

    def freeze_for_high_dim_entry_tuning(self):
        self.freeze_for_backend_tuning()

    def _mil_scale_policy(self):
        return str(
            getattr(
                getattr(self.cfg.SHARK, "GFM_BACKEND", None),
                "MIL_SCALE_POLICY",
                "anchored_pre_adaptation",
            )
        )

    @torch.no_grad()
    def fit_feature_adapter(self, loader, max_batches=None):
        if not self.gfm.backend_spec().uses_declared_node_tokens:
            return None
        return SHARK.fit_feature_adapter(
            self,
            loader,
            max_batches=max_batches,
        )

    @torch.no_grad()
    def fit_evidence_calibrator(self, loader, max_batches=None):
        policy = self._mil_scale_policy()
        if policy == "refit_each_epoch":
            return SHARK.fit_evidence_calibrator(
                self,
                loader,
                max_batches=max_batches,
            )
        if policy != "anchored_pre_adaptation":
            raise ValueError(f"Unsupported MIL scale policy: {policy}")
        return super().fit_evidence_calibrator(
            loader,
            max_batches=max_batches,
        )

    def mil_reference_diagnostics(self):
        if self._mil_scale_policy() == "refit_each_epoch":
            return SHARK.mil_reference_diagnostics(self)
        return super().mil_reference_diagnostics()

    def load_frozen_backend_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Frozen backend baseline not found: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        message = self.load_state_dict(checkpoint["model_state"], strict=False)
        expected_missing = {
            *(f"gfm.{name}" for name in self.gfm.adaptation_parameter_names()),
            "mil_scale_anchor",
            "mil_scale_anchor_fitted",
        }
        if set(message.missing_keys) != expected_missing:
            raise RuntimeError(
                "Frozen backend baseline has unexpected missing keys: "
                f"{message.missing_keys}"
            )
        if message.unexpected_keys:
            raise RuntimeError(
                "Frozen backend baseline has unexpected keys: "
                f"{message.unexpected_keys}"
            )
        if not self.gfm.adaptation_is_identity():
            raise RuntimeError("GFM backend adaptation must be identity at load")
        if bool(self.mil_scale_anchor_fitted):
            raise RuntimeError("MIL scale anchor must be empty at baseline load")

        self._frozen_backend_loaded = True
        self._frozen_backend_checkpoint = str(checkpoint_path.resolve())
        self._frozen_v815_loaded = True
        self._frozen_v815_checkpoint = self._frozen_backend_checkpoint
        if hasattr(self.gfm_adapter, "fitted"):
            self.gfm_adapter.fitted = True
        self.freeze_for_backend_tuning()
        return {
            "checkpoint": self._frozen_backend_checkpoint,
            "checkpoint_epoch_index": int(checkpoint.get("epoch", -1)),
            "missing_keys": sorted(message.missing_keys),
            "unexpected_keys": sorted(message.unexpected_keys),
            "backend": self.gfm.backend_spec().name,
        }

    def load_frozen_v815_checkpoint(self, checkpoint_path):
        return self.load_frozen_backend_checkpoint(checkpoint_path)

    def load_shared_frontend_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Shared MTS frontend checkpoint not found: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint.get("model_state", {})
        prefix = "imlp."
        frontend_state = {
            name[len(prefix):]: value
            for name, value in model_state.items()
            if name.startswith(prefix)
        }
        if not frontend_state:
            raise RuntimeError("Shared checkpoint contains no DLinear frontend state")
        self.imlp.load_state_dict(frontend_state, strict=True)
        self._shared_frontend_loaded = True
        self._shared_frontend_checkpoint = str(checkpoint_path.resolve())
        self.freeze_for_backend_tuning()
        return {
            "checkpoint": self._shared_frontend_checkpoint,
            "parameter_count": len(frontend_state),
        }

    def trainable_parameter_summary(self):
        summary = dict(SHARK.trainable_parameter_summary(self))
        if self._mil_scale_policy() == "anchored_pre_adaptation":
            summary.update(self.anchored_mil_scale_diagnostics())
        spec = self.gfm.backend_spec()
        summary.update({
            "backend_name": spec.name,
            "backend_token_dim": spec.token_dim,
            "backend_embedding_dim": spec.embedding_dim,
            "backend_adaptation_strategy": spec.adaptation_strategy,
            "backend_adaptation_parameter_names": list(
                self.gfm.adaptation_parameter_names()
            ),
        })
        summary["candidate_training_mode"] = (
            str(
                getattr(
                    getattr(self.cfg.SHARK, "GFM_BACKEND", None),
                    "TRAINING_MODE",
                    "backend_adaptation_only",
                )
            )
        )
        summary["mil_scale_policy"] = self._mil_scale_policy()
        summary["shared_frontend_loaded"] = self._shared_frontend_loaded
        summary["shared_frontend_checkpoint"] = self._shared_frontend_checkpoint
        return summary

    def peft_diagnostics(self):
        diagnostics = dict(self.trainable_parameter_summary())
        if hasattr(self.gfm, "prompt_diagnostics"):
            diagnostics.update(self.gfm.prompt_diagnostics())
        diagnostics.update(self._sensor_graph_prompt_diagnostics())
        return diagnostics

    def adapt_backend_outputs(self, graph_outputs, backend_inputs):
        """Shared post-backend boundary; the unified v1 control is identity."""
        return graph_outputs

    def _forward_backend(self, backend_inputs):
        """Extension point for SIP-GFM graph routing experiments."""
        return self.gfm.forward_backend(backend_inputs)

    def combine_time_components(self, seasonal_features, trend_features):
        """公共图输入汇合点；OTR-CL 默认保持原 seasonal+trend。"""
        return seasonal_features + trend_features

    def _encode_frontend(self, x):
        x = x.float()
        if hasattr(self.imlp, "forward_components"):
            time_seasonal_features, time_trend_features = (
                self.imlp.forward_components(x)
            )
            node_features = self.combine_time_components(
                time_seasonal_features,
                time_trend_features,
            )
        else:
            node_features = self.imlp(x)
            time_seasonal_features = None
            time_trend_features = None
        adjacency = self.graph_builder(node_features.detach())
        return {
            "node_features": node_features,
            "adjacency": adjacency,
            "time_seasonal_features": time_seasonal_features,
            "time_trend_features": time_trend_features,
        }

    def encode_frozen_frontend(self, x):
        """Encode a frozen Time-to-Node frontend once for safe reuse.

        Training callers may cache these tensors only when neither the DLinear
        frontend nor graph builder carries trainable parameters. The explicit
        guard prevents a future configuration from silently losing gradients.
        """
        trainable = [
            f"{prefix}.{name}"
            for prefix, module in (
                ("imlp", self.imlp),
                ("graph_builder", self.graph_builder),
            )
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise RuntimeError(
                "Frozen frontend reuse requires non-trainable frontend "
                f"parameters; found={trainable}"
            )
        return self._encode_frontend(x)

    def frozen_frontend_reuse_available(self):
        return not any(
            parameter.requires_grad
            for module in (self.imlp, self.graph_builder)
            for parameter in module.parameters()
        )

    def _forward_from_frontend(self, frontend, labels=None, return_dict=True):
        node_features = frontend["node_features"]
        adjacency = frontend["adjacency"]
        time_seasonal_features = frontend["time_seasonal_features"]
        time_trend_features = frontend["time_trend_features"]
        spec = self.gfm.backend_spec()
        if spec.uses_declared_node_tokens:
            base_gfm_node_features = self.gfm_adapter(node_features)
            gfm_node_features = self.sensor_graph_prompt(
                base_gfm_node_features,
                adjacency,
            )
            source_projection_weight = self.gfm_adapter.proj.weight
        else:
            gfm_node_features = node_features.new_zeros(
                node_features.size(0),
                node_features.size(1),
                spec.token_dim,
            )
            base_gfm_node_features = gfm_node_features
            source_projection_weight = None
        backend_inputs = GFMBackendInput(
            node_tokens=gfm_node_features,
            adjacency=adjacency,
            source_node_features=node_features,
            source_projection_weight=source_projection_weight,
            time_trend_features=time_trend_features,
            time_seasonal_features=time_seasonal_features,
        )
        graph_outputs = self._forward_backend(backend_inputs)
        native_node_embeddings = graph_outputs["node_embeddings"]
        graph_outputs = self.adapt_backend_outputs(
            graph_outputs,
            backend_inputs,
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
            "time_trend_features": time_trend_features,
            "time_seasonal_features": time_seasonal_features,
            "gfm_node_features_base": base_gfm_node_features,
            "gfm_node_features": gfm_node_features,
            "adjacency": adjacency,
            "raw_node_scores": raw_node_scores,
            "node_scores_raw": raw_node_scores,
            "native_node_scores": raw_node_scores,
            "native_node_embeddings": native_node_embeddings,
            "adapted_node_embeddings": node_embeddings,
            "node_embeddings": node_embeddings,
            "graph_embedding": graph_embedding,
            "backend_name": self.gfm.backend_spec().name,
        })
        outputs.update({
            name: value
            for name, value in graph_outputs.items()
            if name not in {
                "node_scores",
                "node_embeddings",
                "graph_embedding",
                "native_node_embeddings",
                "adapted_node_embeddings",
            }
        })
        if return_dict:
            return outputs
        return window_score

    def forward(self, x, labels=None, return_dict=True):
        return self._forward_from_frontend(
            self._encode_frontend(x),
            labels=labels,
            return_dict=return_dict,
        )
