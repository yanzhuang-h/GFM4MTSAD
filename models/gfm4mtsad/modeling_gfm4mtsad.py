import torch
import torch.nn as nn
import torch.nn.functional as F

from models.anomalygfm.modeling_anomalygfm import AnomalyGFM
from models.gfm4mtsad.adapter import build_adapter, build_sensor_graph_prompt
from models.gfm4mtsad.builder import build_graph
from models.gfm4mtsad.head import GFM4MTSADHead
from models.gfm4mtsad.imlp import DLinear_MLP


class GFM4MTSAD(nn.Module):
    def __init__(self, cfg):
        super(GFM4MTSAD, self).__init__()
        self.cfg = cfg
        if str(cfg.GFM4MTSAD.HEAD.NAME) != "GFM4MTSADHead":
            raise ValueError("GFM4MTSAD v8.15 only supports GFM4MTSAD.HEAD.NAME=GFM4MTSADHead")
        self.imlp = self._init_imlp()
        self.graph_builder = build_graph(cfg)
        self.gfm_adapter = build_adapter(
            cfg,
            input_dim=cfg.DLINEAR.FEATURE_DIM,
            output_dim=cfg.ANOMALYGFM.INPUT_DIM,
        )
        self.sensor_graph_prompt = build_sensor_graph_prompt(
            cfg,
            feature_dim=cfg.ANOMALYGFM.INPUT_DIM,
        )
        self.gfm = self._init_gfm(cfg)
        self.head = GFM4MTSADHead(cfg, embedding_dim=self.gfm.embedding_dim)

        self._freeze_backbone()

    def _init_imlp(self):
        mlp = self.cfg.GFM4MTSAD.IMLP
        mlp_classes = {
            "DLinear": DLinear_MLP,
        }

        if mlp not in mlp_classes:
            raise ValueError(f"Unsupported GFM4MTSAD.IMLP: {mlp}")

        imlp = mlp_classes[mlp](self.cfg)
        return imlp

    def _init_gfm(self, cfg):
        name = cfg.GFM4MTSAD.GFM
        if name == "AnomalyGFM":
            return AnomalyGFM(cfg)
        raise ValueError(f"Unsupported GFM4MTSAD.GFM: {name}")

    def _freeze_backbone(self):
        if self.cfg.DLINEAR.FREEZE:
            self.imlp.eval()
            for param in self.imlp.parameters():
                param.requires_grad = False

        if self.cfg.GFM4MTSAD.ADAPTER.FREEZE:
            self.gfm_adapter.eval()
            for param in self.gfm_adapter.parameters():
                param.requires_grad = False

        if hasattr(self.gfm, "freeze"):
            self.gfm.freeze()
        else:
            self.gfm.eval()
            for param in self.gfm.parameters():
                param.requires_grad = False
        if hasattr(self.gfm, "configure_tuning"):
            self.gfm.configure_tuning()

        if (
            hasattr(self.sensor_graph_prompt, "enabled")
            and not self.sensor_graph_prompt.enabled
        ):
            self.sensor_graph_prompt.freeze()

    def train(self, mode=True):
        super(GFM4MTSAD, self).train(mode)
        if self.cfg.DLINEAR.FREEZE:
            self.imlp.eval()
        if self.cfg.GFM4MTSAD.ADAPTER.FREEZE:
            self.gfm_adapter.eval()
        if (
            hasattr(self.sensor_graph_prompt, "enabled")
            and not self.sensor_graph_prompt.enabled
        ):
            self.sensor_graph_prompt.eval()
        if self._gfm_has_trainable_parameters():
            self.gfm.train(mode)
        else:
            self.gfm.eval()
        return self

    def _gfm_has_trainable_parameters(self):
        return any(param.requires_grad for param in self.gfm.parameters())

    def trainable_parameter_summary(self):
        total = 0
        trainable = 0
        names = []
        gfm_total = 0
        gfm_trainable = 0
        sensor_prompt_total = 0
        sensor_prompt_trainable = 0
        for name, param in self.named_parameters():
            count = param.numel()
            total += count
            if name.startswith("gfm."):
                gfm_total += count
            if name.startswith("sensor_graph_prompt."):
                sensor_prompt_total += count
            if param.requires_grad:
                trainable += count
                names.append(name)
                if name.startswith("gfm."):
                    gfm_trainable += count
                if name.startswith("sensor_graph_prompt."):
                    sensor_prompt_trainable += count
        peft_trainable = gfm_trainable + sensor_prompt_trainable
        peft_total = gfm_total + sensor_prompt_total
        return {
            "strategy": getattr(self.cfg.GFM4MTSAD.GFM_TUNING, "STRATEGY", "frozen"),
            "total_params": total,
            "trainable_params": trainable,
            "trainable_ratio": trainable / max(total, 1),
            "gfm_total_params": gfm_total,
            "gfm_trainable_params": gfm_trainable,
            "gfm_trainable_ratio": gfm_trainable / max(gfm_total, 1),
            "sensor_graph_prompt_total_params": sensor_prompt_total,
            "sensor_graph_prompt_trainable_params": sensor_prompt_trainable,
            "peft_trainable_params": peft_trainable,
            "peft_trainable_ratio": peft_trainable / max(peft_total, 1),
            "trainable_parameter_names": names,
            "gfm_trainable_parameter_names": self.gfm.trainable_parameter_names()
            if hasattr(self.gfm, "trainable_parameter_names")
            else [],
        }

    def peft_diagnostics(self):
        summary = self.trainable_parameter_summary()
        prompt_stats = {}
        if hasattr(self.gfm, "prompt_diagnostics"):
            prompt_stats.update(self.gfm.prompt_diagnostics())
        prompt_stats.update(self._sensor_graph_prompt_diagnostics())
        prompt_stats.update(self.graph_reference_diagnostics())
        return {
            **summary,
            **prompt_stats,
        }

    def mil_reference_diagnostics(self):
        if not hasattr(self.head, "mil_reference_diagnostics"):
            return {}
        return self.head.mil_reference_diagnostics()

    def graph_reference_diagnostics(self):
        if not hasattr(self.graph_builder, "reference_prior_diagnostics"):
            return {}
        return self.graph_builder.reference_prior_diagnostics()

    def _sensor_graph_prompt_diagnostics(self):
        if not hasattr(self, "sensor_graph_prompt"):
            return {}
        module = self.sensor_graph_prompt
        stats = {
            "sensor_graph_prompt_enabled": bool(getattr(module, "enabled", False)),
            "sensor_graph_prompt_scale": float(getattr(module, "scale", 0.0)),
        }
        for name in ("node_prompt", "graph_prompt"):
            param = getattr(module, name, None)
            if param is not None:
                stats[f"sensor_graph_prompt_{name}_norm"] = float(param.detach().norm().cpu().item())
        return stats

    @torch.no_grad()
    def fit_feature_adapter(self, loader, max_batches=None):
        features = []
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            windows = batch[0] if isinstance(batch, (tuple, list)) else batch
            windows = windows.to(device, non_blocking=True).float()
            features.append(self.imlp(windows).detach().cpu())

        if features:
            node_features = torch.cat(features, dim=0).to(device)
            self.gfm_adapter.fit(node_features)

        self.train(was_training)

    @torch.no_grad()
    def fit_graph_reference(self, loader, max_batches=None):
        """Fit the Pearson edge-frequency prior from normal train windows.

        The loader is intentionally consumed without labels.  The frozen
        DLinear frontend is the same frontend used by ``forward``; no GFM or
        anomaly score is evaluated while fitting this reference.
        """
        if not hasattr(self.graph_builder, "update_reference"):
            return {
                "reference_prior_enabled": False,
                "reference_prior_fitted": False,
            }

        self.graph_builder.reset_reference()
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        batch_count = 0
        window_count = 0
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            windows = batch[0] if isinstance(batch, (tuple, list)) else batch
            windows = windows.to(device, non_blocking=True).float()
            if windows.numel() == 0:
                continue
            node_features = self.imlp(windows)
            self.graph_builder.update_reference(node_features.detach())
            batch_count += 1
            window_count += int(windows.size(0))

        self.train(was_training)
        diagnostics = self.graph_builder.reference_prior_diagnostics()
        diagnostics.update({
            "reference_prior_fit_batches": batch_count,
            "reference_prior_fit_windows": window_count,
            "reference_prior_used_labels": False,
        })
        if diagnostics.get("reference_prior_enabled") and window_count == 0:
            raise RuntimeError(
                "Graph reference fitting received no normal train windows"
            )
        return diagnostics

    @torch.no_grad()
    def fit_evidence_calibrator(self, loader, max_batches=None):
        supports_evidence_reference = all(
            hasattr(self.head, name)
            for name in (
                "set_evidence_reference",
                "reset_evidence_reference",
                "evidence_reference_fitted",
                "evidence_reference_mean",
                "evidence_reference_scale",
                "evidence_reference_values",
                "evidence_reference_count",
            )
        )
        supports_mil_reference = all(
            hasattr(self.head, name)
            for name in (
                "set_mil_node_reference",
                "reset_mil_node_reference",
                "mil_reference_fitted",
                "mil_reference_mean",
                "mil_reference_scale",
            )
        )
        fit_evidence_reference = supports_evidence_reference and bool(
            getattr(self.cfg.GFM4MTSAD.HEAD, "NORMAL_CALIBRATION_ENABLE", False)
        )
        fit_mil_reference = (
            supports_mil_reference
            and (
                bool(getattr(self.head, "uses_mil_reference", False))
                or getattr(self.cfg.GFM4MTSAD.HEAD, "EVIDENCE_FUSION_MODE", "")
                == "counterfactual_mil"
            )
        )
        if not fit_evidence_reference and not fit_mil_reference:
            return

        evidence_batches = []
        mil_node_batches = []
        was_training = self.training
        device = next(self.parameters()).device
        old_evidence_reference = None
        if fit_evidence_reference:
            old_evidence_reference = {
                "fitted": bool(self.head.evidence_reference_fitted.item()),
                "mean": self.head.evidence_reference_mean.detach().clone(),
                "scale": self.head.evidence_reference_scale.detach().clone(),
                "values": self.head.evidence_reference_values.detach().clone(),
                "count": int(self.head.evidence_reference_count.item()),
            }
        old_mil_reference = None
        if fit_mil_reference:
            old_mil_reference = {
                "fitted": bool(self.head.mil_reference_fitted.item()),
                "mean": self.head.mil_reference_mean.detach().clone(),
                "scale": self.head.mil_reference_scale.detach().clone(),
                "values": self.head.mil_reference_values.detach().clone()
                if hasattr(self.head, "mil_reference_values")
                else None,
                "count": int(self.head.mil_reference_count.item())
                if hasattr(self.head, "mil_reference_count")
                else 0,
            }

        self.eval()
        if fit_evidence_reference:
            self.head.reset_evidence_reference()
        if fit_mil_reference:
            self.head.reset_mil_node_reference()
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            windows, labels = batch if isinstance(batch, (tuple, list)) else (batch, None)
            windows = windows.to(device, non_blocking=True).float()
            if labels is not None:
                labels = labels.to(device, non_blocking=True).view(-1)
                normal_mask = labels == 0
                if normal_mask.any():
                    windows = windows[normal_mask]
                else:
                    continue
            outputs = self.forward(windows)
            if fit_evidence_reference:
                evidence_batches.append(outputs["conservative_scores_raw"].detach())
            if fit_mil_reference:
                mil_node_batches.append(outputs["mil_direct_raw"].detach())

        if fit_evidence_reference and evidence_batches:
            evidence = torch.cat(evidence_batches, dim=0)
            mean = evidence.mean(dim=0)
            scale = evidence.std(dim=0, unbiased=False).clamp_min(
                float(self.cfg.GFM4MTSAD.HEAD.NORMAL_CALIBRATION_MIN_SCALE)
            )
            self.head.set_evidence_reference(mean, scale, evidence)
        elif fit_evidence_reference and old_evidence_reference["fitted"]:
            count = old_evidence_reference["count"]
            old_reference_values = (
                old_evidence_reference["values"][:count]
                if count > 0
                else None
            )
            self.head.set_evidence_reference(
                old_evidence_reference["mean"],
                old_evidence_reference["scale"],
                old_reference_values,
            )
        elif fit_evidence_reference:
            self.head.reset_evidence_reference()

        if fit_mil_reference and mil_node_batches:
            mil_nodes = torch.cat(mil_node_batches, dim=0)
            mil_mean = mil_nodes.mean(dim=0)
            mil_scale = mil_nodes.std(dim=0, unbiased=False).clamp_min(
                float(self.cfg.GFM4MTSAD.HEAD.NORMAL_CALIBRATION_MIN_SCALE)
            )
            self.head.set_mil_node_reference(mil_mean, mil_scale, mil_nodes)
        elif fit_mil_reference and old_mil_reference["fitted"]:
            old_values = old_mil_reference["values"]
            if old_values is not None:
                old_values = old_values[: old_mil_reference["count"]]
            self.head.set_mil_node_reference(
                old_mil_reference["mean"],
                old_mil_reference["scale"],
                old_values,
            )
        elif fit_mil_reference:
            self.head.reset_mil_node_reference()

        self.train(was_training)

    def forward(self, x, labels=None, return_dict=True):
        x = x.float()
        node_features = self.imlp(x)
        adjacency = self.graph_builder(node_features.detach())
        gfm_node_features = self.gfm_adapter(node_features)
        gfm_node_features = self.sensor_graph_prompt(
            gfm_node_features,
            adjacency,
        )
        graph_outputs = self.gfm(gfm_node_features, adjacency)
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
            "gfm_node_features": gfm_node_features,
            "adjacency": adjacency,
            "raw_node_scores": raw_node_scores,
            "node_scores_raw": raw_node_scores,
            "native_node_scores": raw_node_scores,
            "node_embeddings": node_embeddings,
            "graph_embedding": graph_embedding,
            "backend_name": "anomalygfm",
        })

        if return_dict:
            return outputs
        return window_score
