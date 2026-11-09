from pathlib import Path

import torch

from models.gfm4mtsad.modeling_gfm4mtsad import GFM4MTSAD


class AnchoredMILScaleGFM4MTSAD(GFM4MTSAD):
    """High-dimensional entry with a fixed pre-adaptation MIL scale."""

    def __init__(self, cfg):
        super().__init__(cfg)
        sensor_count = int(cfg.DATA.N_VAR)
        self.register_buffer(
            "mil_scale_anchor",
            torch.ones(sensor_count),
        )
        self.register_buffer(
            "mil_scale_anchor_fitted",
            torch.tensor(False, dtype=torch.bool),
        )
        self._last_mil_scale_action = "uninitialized"


    @torch.no_grad()
    def fit_evidence_calibrator(self, loader, max_batches=None):
        super().fit_evidence_calibrator(loader, max_batches=max_batches)
        if not bool(self.head.mil_reference_fitted):
            raise RuntimeError("MIL reference must be fitted before scale anchoring")
        fitted_scale = self.head.mil_reference_scale.detach()
        if not torch.isfinite(fitted_scale).all() or bool((fitted_scale <= 0).any()):
            raise RuntimeError("MIL reference scale must be finite and positive")

        if not bool(self.mil_scale_anchor_fitted):
            self.mil_scale_anchor.copy_(
                fitted_scale.to(
                    device=self.mil_scale_anchor.device,
                    dtype=self.mil_scale_anchor.dtype,
                )
            )
            self.mil_scale_anchor_fitted.fill_(True)
            self._last_mil_scale_action = "captured_pre_adaptation_scale"
        else:
            self._last_mil_scale_action = "restored_pre_adaptation_scale"

        self.head.mil_reference_scale.copy_(
            self.mil_scale_anchor.to(
                device=self.head.mil_reference_scale.device,
                dtype=self.head.mil_reference_scale.dtype,
            )
        )
        return self.anchored_mil_scale_diagnostics()

    def anchored_mil_scale_diagnostics(self):
        anchor = self.mil_scale_anchor.detach().float()
        active = self.head.mil_reference_scale.detach().float()
        return {
            "mil_scale_anchor_fitted": bool(self.mil_scale_anchor_fitted),
            "mil_scale_anchor_min": float(anchor.min().cpu()),
            "mil_scale_anchor_mean": float(anchor.mean().cpu()),
            "mil_scale_anchor_max": float(anchor.max().cpu()),
            "mil_scale_anchor_active_max_abs": float(
                (active - anchor.to(active.device)).abs().max().cpu()
            ),
            "mil_scale_policy": "pre_adaptation_anchor_mean_refit_only",
            "mil_scale_last_action": self._last_mil_scale_action,
        }

    def mil_reference_diagnostics(self):
        diagnostics = dict(super().mil_reference_diagnostics())
        diagnostics.update(self.anchored_mil_scale_diagnostics())
        return diagnostics

    def trainable_parameter_summary(self):
        summary = dict(super().trainable_parameter_summary())
        summary.update(self.anchored_mil_scale_diagnostics())
        summary["candidate_training_mode"] = (
            "frozen_v815_entry_only_anchored_mil_scale"
        )
        return summary

    def peft_diagnostics(self):
        diagnostics = dict(super().peft_diagnostics())
        diagnostics.update(self.anchored_mil_scale_diagnostics())
        diagnostics["candidate_training_mode"] = (
            "frozen_v815_entry_only_anchored_mil_scale"
        )
        return diagnostics
