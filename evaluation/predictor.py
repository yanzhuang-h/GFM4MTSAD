import numpy as np
import torch

from models.shark.causal_rolling_calibration import (
    causal_rolling_robust_calibration,
)
from models.shark.node_to_time_reliability_gate import (
    normal_validity_reliability_gate,
)
from models.shark.predictor_shark import SharkPredictor
from models.shark.unified_node_to_time import (
    event_preserving_temporal_consensus,
)


class UnifiedNodeToTimePredictor(SharkPredictor):
    """The fixed, backend-neutral final score used by unified SHARK v1."""

    def _apply_node_to_time(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        causal = causal_rolling_robust_calibration(raw)
        diagnostics = self.augmentor.normal_mechanism_diagnostics()
        if not diagnostics or not diagnostics.get("fitted", False):
            raise RuntimeError(
                "Unified Node-to-Time requires a fitted normal mechanism"
            )
        gated, gate = normal_validity_reliability_gate(
            raw,
            causal,
            diagnostics["valid_node_ratio"],
        )
        gate["selected_source"] = (
            "unified_causal" if gate["causal_selected"] else "unified_raw"
        )
        width = int(self.cfg.SHARK.TEMPORAL_CONSENSUS.WIDTH)
        final = event_preserving_temporal_consensus(gated, width=width)
        return final, causal, gated, gate

    @torch.no_grad()
    def _get_test_outputs(self):
        raw, labels, node_scores = super()._get_test_outputs()
        raw = np.asarray(raw, dtype=np.float64)
        final, causal, gated, gate = self._apply_node_to_time(raw)
        self.test_scores_before_causal_calibration = raw.copy()
        self.test_scores_after_causal_calibration = causal
        self.test_scores_after_reliability_gate = gated
        self.node_to_time_reliability_gate = gate
        self.test_scores_before_temporal_consensus = np.asarray(
            gated,
            dtype=np.float64,
        ).copy()
        self.test_scores_after_temporal_consensus = final
        self.test_score_sources["unified_raw"] = raw
        self.test_score_sources["unified_causal"] = causal
        self.test_score_sources["unified_reliability_gated"] = gated
        self.test_score_sources["unified_node_to_time"] = final
        self.cfg.SHARK.EVAL.SCORE_SOURCE = "unified_node_to_time"
        return final, labels, node_scores

    @torch.no_grad()
    def _get_reference_outputs(self, loader, description, score_source=None):
        raw, labels = super()._get_reference_outputs(
            loader,
            description,
            score_source="shark_head",
        )
        final, causal, gated, _ = self._apply_node_to_time(raw)
        split = {
            "shark train scores": "train",
            "shark validation scores": "validation",
        }.get(description)
        if split is not None:
            setattr(self, f"{split}_scores_before_causal_calibration", raw.copy())
            setattr(self, f"{split}_scores_after_causal_calibration", causal.copy())
            setattr(self, f"{split}_scores_after_reliability_gate", gated.copy())
            setattr(self, f"{split}_scores_after_temporal_consensus", final.copy())
        return final, labels

    def save_to_npy(self, **kwargs):
        for split in ("train", "validation"):
            for stage in (
                "before_causal_calibration",
                "after_causal_calibration",
                "after_reliability_gate",
                "after_temporal_consensus",
            ):
                name = f"{split}_scores_{stage}"
                if hasattr(self, name):
                    kwargs[name] = getattr(self, name)
        kwargs["test_scores_before_causal_calibration"] = (
            self.test_scores_before_causal_calibration
        )
        kwargs["test_scores_after_causal_calibration"] = (
            self.test_scores_after_causal_calibration
        )
        kwargs["test_scores_after_reliability_gate"] = (
            self.test_scores_after_reliability_gate
        )
        kwargs["test_node_to_time_valid_node_ratio"] = np.asarray(
            [self.node_to_time_reliability_gate["valid_node_ratio"]],
            dtype=np.float64,
        )
        kwargs["test_node_to_time_selected_source"] = np.asarray(
            [self.node_to_time_reliability_gate["selected_source"]]
        )
        kwargs["test_scores_before_temporal_consensus"] = (
            self.test_scores_before_temporal_consensus
        )
        kwargs["test_scores_after_temporal_consensus"] = (
            self.test_scores_after_temporal_consensus
        )
        super().save_to_npy(**kwargs)

    def get_score_source_sweep(self):
        sweep = dict(super().get_score_source_sweep())
        for source in (
            "unified_raw",
            "unified_causal",
            "unified_reliability_gated",
            "unified_node_to_time",
        ):
            scores = self.test_score_sources[source]
            threshold = self._get_threshold(scores, self.test_labels)
            pred = (scores > threshold).astype(int)
            pred_pa = (
                self.point_adjust(pred.copy(), self.test_labels)
                if self.cfg.TEST.POINT_ADJUST
                else None
            )
            metrics = self.get_results(scores, pred_pa, self.test_labels)
            sweep[source] = {
                "Threshold": threshold,
                "AUROC": metrics["AUROC"],
                "AUPRC": metrics["AUPRC"],
                "F1": metrics["F1"],
                "F1_PA": metrics.get("F1_PA", float("nan")),
            }
        return sweep


class InformationPreservingPredictor(UnifiedNodeToTimePredictor):
    """Registered information-preserving model predictor."""

    def predict(self):
        # The inherited helper picks CUDA whenever it is available. Respect the
        # explicitly selected model device, including CPU on a GPU host.
        from unittest.mock import patch
        from training.data import prepare_on_device

        device = next(self.model.parameters()).device
        with patch(
            "models.shark.predictor_shark.prepare_inputs",
            lambda values: prepare_on_device(values, device),
        ):
            return super().predict()
