import math
import sys
import time

import torch
from tqdm import tqdm

from trainer import Trainer, prepare_inputs
import models.optimizer as optim
from models.shark.augment import SharkAugmentor
from models.shark.losses import (
    batch_hard_normal_contrastive_loss,
    carots_style_contrastive_loss,
    decoupled_local_global_contrastive_loss,
    head_fidelity_loss,
    intervention_gain_node_loss,
    hard_normal_tail_ranking_loss,
    masked_node_consistency_loss,
    mechanism_conditioned_local_global_contrastive_loss,
    mechanism_matched_relation_contrastive_loss,
    node_aware_hard_contrastive_loss,
    native_evidence_mechanism_contrastive_loss,
    ranking_loss,
    tail_preserving_mechanism_window_contrastive_loss,
    temporal_regime_contrastive_loss,
    window_bce_loss,
)
from models.shark.gfm_backend import require_node_anomaly_evidence
from models.shark.relation_residual import RELATION_PATH_PARAMETER_PREFIXES
from models.shark.relational_contrast import relational_node_contrastive_loss
from models.shark.otrcl_integrity import (
    gradient_clip_diagnostics,
    summarize_otrcl_view_integrity,
)
from models.shark.temporal_counterfactual import (
    TemporalCounterfactualGenerator,
)
from utils.misc import mkdir


class SharkTrainer(Trainer):
    def __init__(self, cfg, model, negative_teacher=None):
        super().__init__(cfg, model)
        self.augmentor = SharkAugmentor(cfg)
        self.negative_teacher = negative_teacher
        self.otrcl_settings = getattr(cfg.SHARK, "OTRCL", None)
        self.otrcl_enabled = bool(
            getattr(self.otrcl_settings, "ENABLE", False)
        )
        self.temporal_counterfactual = (
            TemporalCounterfactualGenerator.from_config(cfg)
            if self.otrcl_enabled
            else None
        )
        self.last_otrcl_diagnostics = {}
        self.last_otrcl_gradient_diagnostics = {}
        if self.otrcl_enabled and not hasattr(
            self.model,
            "temporal_regime_encoder",
        ):
            raise ValueError(
                "SHARK.OTRCL.ENABLE requires OTRCLUnifiedSHARK"
            )
        if self.augmentor.requires_normal_mechanism:
            print(
                f"[SHARK setup] dataset={cfg.DATA.NAME} fitting normal mechanism",
                flush=True,
            )
            self.augmentor.fit_normal_mechanism(self.train_loader)
            self.model._shark_normal_mechanism = self.augmentor.normal_mechanism
            self._print_normal_mechanism_diagnostics()
        if self.augmentor.requires_frozen_gfm_teacher:
            if self.negative_teacher is None:
                raise ValueError(
                    "frozen_gfm_teacher screening requires a separate teacher"
                )
            teacher_diagnostics = self.augmentor.bind_frozen_gfm_teacher(
                self.negative_teacher
            )
            print(
                "[SHARK frozen teacher selector] "
                f"bound={teacher_diagnostics.get('bound', False)} "
                f"valid_nodes={teacher_diagnostics.get('valid_node_count', 0)} "
                "teacher_trainable=False",
                flush=True,
            )
        print(f"[SHARK setup] dataset={cfg.DATA.NAME} fitting Sensor-to-GFM Adapter", flush=True)
        self.model.fit_feature_adapter(
            self.train_loader,
            max_batches=cfg.SHARK.ADAPTER.FIT_MAX_BATCHES,
        )
        if bool(
            getattr(
                getattr(cfg, "GRAPHBUILDER", None),
                "REFERENCE_PRIOR_ENABLE",
                False,
            )
        ):
            print(
                f"[SHARK setup] dataset={cfg.DATA.NAME} "
                "fitting normal DLinear graph reference",
                flush=True,
            )
            graph_reference = self.model.fit_graph_reference(
                self.train_loader,
                max_batches=getattr(
                    cfg.GRAPHBUILDER,
                    "REFERENCE_PRIOR_MAX_BATCHES",
                    None,
                ),
            )
            print(
                "[SHARK graph reference] "
                f"fitted={graph_reference.get('reference_prior_fitted', False)} "
                f"windows={graph_reference.get('reference_prior_fit_windows', 0)} "
                f"blend={graph_reference.get('reference_prior_blend', 0.0):.3f}",
                flush=True,
            )
        if self.augmentor.requires_frontend_graph_selector:
            print(
                f"[SHARK setup] dataset={cfg.DATA.NAME} fitting train-normal "
                "frontend-effective selector scale",
                flush=True,
            )
            selector = self.augmentor.fit_frontend_graph_selector(
                self.model.imlp,
                self.model.graph_builder,
                self.train_loader,
            )
            print(
                "[SHARK v2.2 selector] "
                f"fitted={selector.get('fitted', False)} "
                f"windows={selector.get('fit_windows', 0)} "
                f"valid_nodes={selector.get('valid_node_count', 0)} "
                f"scale_floor={selector.get('scale_floor', 0.0):.6g}",
                flush=True,
            )
        print(f"[SHARK setup] dataset={cfg.DATA.NAME} fitting normal reference", flush=True)
        self.model.fit_evidence_calibrator(
            self.train_loader,
            max_batches=cfg.SHARK.HEAD.NORMAL_CALIBRATION_MAX_BATCHES,
        )
        self._print_mil_reference_diagnostics()
        print(f"[SHARK setup] dataset={cfg.DATA.NAME} ready", flush=True)
        self._print_peft_diagnostics()

    @torch.no_grad()
    def _apply_frozen_teacher_selection(
        self,
        windows,
        adjacency,
        anomaly_type_ids,
    ):
        selector = self.augmentor.frozen_gfm_teacher_selector
        if selector is None:
            raise RuntimeError("Frozen GFM teacher selector is not bound")
        selected = selector.select_batch(
            windows,
            self.augmentor.last_candidate_records,
        )
        batch_size, _, node_count = windows.shape
        negative = windows.clone()
        direct_mask = windows.new_zeros(batch_size, node_count)
        propagated_mask = windows.new_zeros(batch_size, node_count)
        valid_mask = torch.zeros(
            batch_size,
            device=windows.device,
            dtype=torch.bool,
        )
        sample_weights = windows.new_zeros(batch_size)
        acceptance_tier = torch.full(
            (batch_size,),
            self.augmentor.ACCEPTANCE_INVALID,
            device=windows.device,
            dtype=torch.long,
        )
        rms_rescue_mask = torch.zeros_like(valid_mask)
        delta_rms = windows.new_zeros(batch_size)
        seed_ids = torch.full(
            (batch_size,),
            -1,
            device=windows.device,
            dtype=torch.long,
        )
        direct_type_ids = torch.full_like(seed_ids, -1)
        consistency_exclusion = torch.zeros(
            batch_size,
            node_count,
            device=windows.device,
            dtype=torch.bool,
        )
        type_to_id = {
            name: index
            for index, name in enumerate(self.augmentor.synthetic_types)
        }
        for batch_index, record in selected.items():
            seed = int(record["seed"])
            start = int(record["start"])
            end = int(record["end"])
            anomaly_type = str(record["anomaly_type"])
            values = record["values"].to(
                device=windows.device,
                dtype=windows.dtype,
            )
            negative[batch_index, start:end, seed] = values
            direct_mask[batch_index, seed] = 1.0
            valid_mask[batch_index] = True
            sample_weights[batch_index] = float(record["sample_weight"])
            acceptance_tier[batch_index] = int(record["acceptance_tier"])
            rms_rescue_mask[batch_index] = bool(
                record["rms_rescue_eligible"]
            )
            delta_rms[batch_index] = float(record["perturbation_rms"])
            seed_ids[batch_index] = seed
            type_id = type_to_id[anomaly_type]
            direct_type_ids[batch_index] = type_id
            anomaly_type_ids[batch_index] = type_id
            if self.augmentor.normal_mechanism is not None:
                consistency_exclusion[batch_index] = (
                    self.augmentor.normal_mechanism.affected_context_mask(
                        seed,
                        windows.device,
                    )
                )
            neighbors = self.augmentor._graph_neighbor_candidates(
                adjacency[batch_index],
                seed,
                node_count,
            )
            propagated = self.augmentor._scope_aware_negative_propagation(
                negative,
                windows,
                propagated_mask,
                batch_index,
                seed,
                neighbors,
                adjacency[batch_index, seed],
                start,
                end,
                values,
            )
            if propagated.numel() > 0:
                consistency_exclusion[batch_index, propagated] = True
            elif self.augmentor.scope_aware_negative_enable:
                sample_weights[batch_index] *= (
                    self.augmentor.scope_aware_negative_singleton_weight
                )
        self.augmentor.last_valid_mask = valid_mask
        self.augmentor.last_sample_weights = sample_weights
        self.augmentor.last_acceptance_tier = acceptance_tier
        self.augmentor.last_rms_rescue_mask = rms_rescue_mask
        self.augmentor.last_selected_delta_rms = delta_rms
        self.augmentor.last_seed_ids = seed_ids
        self.augmentor.last_direct_type_ids = direct_type_ids
        self.augmentor.last_edit_source_ids = torch.where(
            valid_mask,
            torch.full_like(seed_ids, self.augmentor.EDIT_SOURCE_DIRECT),
            torch.full_like(seed_ids, self.augmentor.EDIT_SOURCE_INVALID),
        )
        self.augmentor.last_fallback_type_ids = torch.full_like(seed_ids, -1)
        self.augmentor.last_consistency_exclusion_mask = consistency_exclusion
        return negative, direct_mask, propagated_mask, anomaly_type_ids

    def create_optimizer(self):
        self.optimizer = optim.construct_optimizer(self.model, self.cfg)
        self._print_optimizer_groups()

    def train(self):
        metric_best = self.cfg.TRAIN.METRIC_BEST
        for cur_epoch in range(self.cfg.SOLVER.START_EPOCH, self.cfg.SOLVER.MAX_EPOCH):
            self.cur_epoch = cur_epoch
            epoch_start = time.time()
            print(
                f"[SHARK epoch] dataset={self.cfg.DATA.NAME} "
                f"epoch={cur_epoch + 1}/{self.cfg.SOLVER.MAX_EPOCH} start",
                flush=True,
            )
            self.train_epoch()

            if self._is_eval_epoch(cur_epoch):
                print(
                    f"[SHARK validation] dataset={self.cfg.DATA.NAME} "
                    f"epoch={cur_epoch + 1}/{self.cfg.SOLVER.MAX_EPOCH} start",
                    flush=True,
                )
                tracking_meter = self.eval_epoch()
                if self._check_improvement(tracking_meter.avg, metric_best):
                    with open(mkdir(self.cfg.RESULT_DIR) / "best_result.txt", "w") as f:
                        f.write(
                            f"Val/{tracking_meter.name}: {tracking_meter.avg}"
                            f"\tEpoch: {self.cur_epoch}"
                        )
                    print(
                        f"[current best] Val/{tracking_meter.name}: "
                        f"{tracking_meter.avg}\tEpoch: {self.cur_epoch}",
                        flush=True,
                    )
                    self.save_best_model()
                    metric_best = tracking_meter.avg
                elif self.cfg.TRAIN.SAVE_EVERY_EVAL:
                    self.save_best_model()
                print(
                    f"[SHARK validation] dataset={self.cfg.DATA.NAME} "
                    f"epoch={cur_epoch + 1}/{self.cfg.SOLVER.MAX_EPOCH} "
                    f"{tracking_meter.name}={tracking_meter.avg:.6f} complete",
                    flush=True,
                )

            print(
                f"[SHARK epoch] dataset={self.cfg.DATA.NAME} "
                f"epoch={cur_epoch + 1}/{self.cfg.SOLVER.MAX_EPOCH} "
                f"elapsed={self._format_duration(time.time() - epoch_start)} complete",
                flush=True,
            )

    def _print_peft_diagnostics(self):
        if not hasattr(self.model, "peft_diagnostics"):
            return
        diagnostics = self.model.peft_diagnostics()
        names = diagnostics.get("trainable_parameter_names", [])
        preview = ", ".join(str(name) for name in names[:12])
        if len(names) > 12:
            preview += f", ... (+{len(names) - 12})"
        print(
            "[SHARK PEFT] "
            f"strategy={diagnostics.get('strategy')} "
            f"trainable={diagnostics.get('trainable_params')}/"
            f"{diagnostics.get('total_params')} "
            f"ratio={diagnostics.get('trainable_ratio', 0.0):.6f} "
            f"gfm_trainable={diagnostics.get('gfm_trainable_params')} "
            f"sensor_prompt_trainable={diagnostics.get('sensor_graph_prompt_trainable_params')} "
            f"names=[{preview}]"
        )

    def _print_optimizer_groups(self):
        if not hasattr(self.optimizer, "param_groups"):
            return
        parts = []
        for idx, group in enumerate(self.optimizer.param_groups):
            name = group.get("name", f"group{idx}")
            n_params = sum(param.numel() for param in group.get("params", []))
            parts.append(
                f"{name}:params={n_params},lr={group.get('lr', 0.0):.3e},"
                f"lr_scale={group.get('lr_scale', 1.0):.3g},"
                f"wd={group.get('weight_decay', self.cfg.SOLVER.WEIGHT_DECAY):.3g}"
            )
        print("[SHARK optimizer] " + " | ".join(parts))

    def _print_mil_reference_diagnostics(self):
        if not hasattr(self.model, "mil_reference_diagnostics"):
            return
        diagnostics = self.model.mil_reference_diagnostics()
        if not diagnostics:
            return
        print(
            "[SHARK MIL reference] "
            f"fitted={diagnostics.get('mil_reference_fitted', False)} "
            f"scale_min={diagnostics.get('mil_reference_scale_min', float('nan')):.6f} "
            f"scale_mean={diagnostics.get('mil_reference_scale_mean', float('nan')):.6f} "
            f"scale_max={diagnostics.get('mil_reference_scale_max', float('nan')):.6f}",
            flush=True,
        )

    def _print_normal_mechanism_diagnostics(self):
        diagnostics = self.augmentor.normal_mechanism_diagnostics()
        if not diagnostics:
            return
        print(
            "[SHARK normal mechanism] "
            f"fitted={diagnostics.get('fitted', False)} "
            f"samples={diagnostics.get('n_samples', 0)} "
            f"fit_samples={diagnostics.get('fit_samples', 0)} "
            f"valid_nodes={diagnostics.get('valid_node_count', 0)}/"
            f"{diagnostics.get('n_nodes', 0)} "
            f"valid_ratio={diagnostics.get('valid_node_ratio', 0.0):.4f} "
            f"continuous={diagnostics.get('continuous_node_count', 0)} "
            f"discrete={diagnostics.get('discrete_node_count', 0)} "
            f"constant={diagnostics.get('constant_node_count', 0)} "
            f"parents={diagnostics.get('parent_count_mean', 0.0):.2f} "
            f"residual_scale="
            f"{diagnostics.get('residual_scale_min', float('nan')):.4f}/"
            f"{diagnostics.get('residual_scale_mean', float('nan')):.4f}/"
            f"{diagnostics.get('residual_scale_max', float('nan')):.4f}",
            flush=True,
        )

    def train_epoch(self):
        if not getattr(self.cfg.SHARK, "PROGRESS_ENABLE", True):
            super().train_epoch()
        else:
            self._train_epoch_with_progress()

        if self.cfg.SHARK.HEAD.NORMAL_CALIBRATION_REFIT_EACH_EPOCH:
            print(
                f"[SHARK calibration] dataset={self.cfg.DATA.NAME} "
                f"epoch={self.cur_epoch + 1}/{self.cfg.SOLVER.MAX_EPOCH}",
                flush=True,
            )
            self.model.fit_evidence_calibrator(
                self.train_loader,
                max_batches=self.cfg.SHARK.HEAD.NORMAL_CALIBRATION_MAX_BATCHES,
            )
            self._print_mil_reference_diagnostics()

    def _train_epoch_with_progress(self):
        metric_meters = self._get_metric_meters()
        loss_meters = self._get_loss_meters()
        self.model.train()
        data_size = len(self.train_loader)
        update_interval = max(int(getattr(self.cfg.SHARK, "PROGRESS_UPDATE_INTERVAL", 10)), 1)
        interactive = bool(getattr(sys.stderr, "isatty", lambda: False)())
        progress = tqdm(
            self.train_loader,
            total=data_size,
            desc=f"{self.cfg.DATA.NAME} train {self.cur_epoch + 1}/{self.cfg.SOLVER.MAX_EPOCH}",
            dynamic_ncols=True,
            leave=True,
            mininterval=0.5,
            disable=not interactive,
        )
        epoch_start = time.time()
        self._reset_synthetic_epoch_diagnostics()

        for idx, inputs in enumerate(progress):
            cur_iter = idx + 1
            self.cur_iter = cur_iter
            lr = optim.get_epoch_lr(self.cur_epoch + float(cur_iter) / max(data_size, 1), self.cfg)
            if isinstance(self.optimizer, tuple):
                [optim.set_lr(optimizer, lr) for optimizer in self.optimizer]
            else:
                optim.set_lr(self.optimizer, lr)

            outputs = self.train_step(inputs)
            self._accumulate_synthetic_epoch_diagnostics()
            batch_size = self._find_batch_size(inputs)
            self._update_metric_meters(metric_meters, outputs["metrics"], batch_size)
            self._update_loss_meters(loss_meters, outputs["losses"], batch_size)

            if cur_iter == 1 or cur_iter % update_interval == 0 or cur_iter == data_size:
                loss_values = outputs["losses"]
                progress_values = self._progress_values(loss_values, lr)
                if interactive:
                    progress.set_postfix(**progress_values, refresh=False)
                else:
                    self._print_progress_snapshot(
                        cur_iter,
                        data_size,
                        progress_values,
                        time.time() - epoch_start,
                    )

        progress.close()
        self._print_synthetic_epoch_diagnostics()

    def _reset_synthetic_epoch_diagnostics(self):
        self._synthetic_epoch_diagnostics = {
            "attempted_count": 0,
            "valid_count": 0,
            "fitted_window_count": 0,
            "surrogate_accepted_count": 0,
            "fallback_count": 0,
            "fallback_attempt_count": 0,
            "fallback_rejected_count": 0,
            "fallback_mechanism_gain_sum": 0.0,
            "fallback_relation_gain_sum": 0.0,
            "fallback_temporal_gain_sum": 0.0,
            "direct_count": 0,
            "direct_attempt_count": 0,
            "invalid_count": 0,
            "hard_count": 0,
            "soft_count": 0,
            "sample_weight_sum": 0.0,
            "candidate_total_count": 0,
            "candidate_finite_count": 0,
            "candidate_plausible_count": 0,
            "candidate_below_gain_floor_count": 0,
            "candidate_above_cap_count": 0,
            "candidate_hard_eligible_count": 0,
            "candidate_soft_eligible_count": 0,
            "candidate_rms_rescue_count": 0,
            "reference_floor_check_count": 0,
            "reference_floor_hit_count": 0,
            "reference_scale_fallback_count": 0,
            "boundary_attempt_count": 0,
            "boundary_valid_count": 0,
            "boundary_insufficient_count": 0,
            "boundary_context_invalid_count": 0,
            "boundary_nonfinite_count": 0,
            "boundary_collapsed_count": 0,
            "boundary_unknown_count": 0,
            "candidate_normalized_gain_count": 0,
            "candidate_score_gain_count": 0,
            "above_cap_normalized_gain_count": 0,
            "rms_rescue_sample_count": 0,
            "support_eligible_count": 0,
            "support_collapsed_count": 0,
            "support_direction_clipped_count": 0,
            "candidate_support_clipped_count": 0,
            "candidate_support_violation_count": 0,
            "candidate_context_valid_count": 0,
            "candidate_positive_gain_count": 0,
            "candidate_eligible_count": 0,
            "selected_orientation_count": 0,
            "candidate_normalized_gain_sum": 0.0,
            "candidate_score_gain_sum": 0.0,
            "above_cap_normalized_gain_sum": 0.0,
            "candidate_rms_sum": 0.0,
            "selected_delta_rms_sum": 0.0,
            "rms_rescue_delta_rms_sum": 0.0,
            "selected_orientation_sum": 0.0,
            "attempted_seed_counts": None,
            "valid_seed_counts": None,
            "fallback_type_counts": {},
            "fallback_type_mechanism_gain_sums": {},
            "direct_type_attempted_counts": {},
            "direct_type_valid_counts": {},
            "direct_type_hard_counts": {},
            "direct_type_soft_counts": {},
            "direct_type_invalid_counts": {},
            "direct_type_sample_weight_sums": {},
            "direct_type_candidate_total_counts": {},
            "direct_type_candidate_finite_counts": {},
            "direct_type_candidate_plausible_counts": {},
            "direct_type_candidate_below_floor_counts": {},
            "direct_type_candidate_above_cap_counts": {},
            "direct_type_candidate_hard_eligible_counts": {},
            "direct_type_candidate_soft_eligible_counts": {},
            "direct_type_candidate_rms_rescue_counts": {},
            "direct_type_reference_floor_checks": {},
            "direct_type_reference_floor_hits": {},
            "direct_type_reference_scale_fallback_counts": {},
            "direct_type_boundary_attempt_counts": {},
            "direct_type_boundary_valid_counts": {},
            "direct_type_boundary_insufficient_counts": {},
            "direct_type_boundary_context_invalid_counts": {},
            "direct_type_boundary_nonfinite_counts": {},
            "direct_type_boundary_collapsed_counts": {},
            "direct_type_boundary_unknown_counts": {},
            "direct_type_candidate_normalized_gain_counts": {},
            "direct_type_candidate_score_gain_counts": {},
            "direct_type_above_cap_normalized_gain_counts": {},
            "direct_type_support_eligible_counts": {},
            "direct_type_support_collapsed_counts": {},
            "direct_type_support_direction_clipped_counts": {},
            "direct_type_candidate_support_clipped_counts": {},
            "direct_type_candidate_support_violation_counts": {},
            "direct_type_candidate_context_valid_counts": {},
            "direct_type_candidate_positive_gain_counts": {},
            "direct_type_candidate_eligible_counts": {},
            "direct_type_selected_orientation_counts": {},
            "direct_type_candidate_normalized_gain_sums": {},
            "direct_type_candidate_score_gain_sums": {},
            "direct_type_above_cap_normalized_gain_sums": {},
            "direct_type_candidate_rms_sums": {},
            "direct_type_selected_orientation_sums": {},
            "direct_type_score_sums": {},
            "direct_type_target_sums": {},
            "direct_type_reference_scale_sums": {},
            "direct_type_normalized_gain_sums": {},
            "direct_segment_attempt_counts": {},
            "direct_segment_selected_counts": {},
            "direct_segment_attempt_length_sums": {},
            "direct_segment_selected_length_sums": {},
            "direct_type_segment_attempt_counts": {},
            "direct_type_segment_selected_counts": {},
            "model_diagnostic_count": 0,
            "model_diagnostic_sums": {},
            "source_model_diagnostics": {},
            "positive_selection_batch_count": 0,
            "positive_selection_sums": {},
            "scope_propagated_sample_count": 0,
            "scope_propagated_node_count": 0,
            "frontend_effective_candidate_count": 0,
            "frontend_effective_eligible_count": 0,
            "frontend_effective_below_count": 0,
            "frontend_effective_above_count": 0,
            "frontend_effective_graph_nonlocal_count": 0,
            "frontend_effective_rescue_used_count": 0,
            "frontend_primary_nonlegacy_pool_count": 0,
            "frontend_primary_nonlegacy_eligible_count": 0,
            "frontend_primary_nonlegacy_selected_count": 0,
            "frontend_effective_selected_d_fd_sum": 0.0,
            "frontend_effective_selected_graph_drift_sum": 0.0,
        }

    def _accumulate_synthetic_epoch_diagnostics(self):
        positive_selection = getattr(
            self.augmentor, "last_positive_selection_diagnostics", {}
        )
        if positive_selection:
            totals = self._synthetic_epoch_diagnostics
            totals["positive_selection_batch_count"] += 1
            for key in (
                "candidate_graph_drift_mean",
                "selected_graph_drift_mean",
                "candidate_temporal_drift_mean",
                "selected_temporal_drift_mean",
                "temporal_floor_mean",
                "graph_floor_mean",
            ):
                value = float(positive_selection.get(key, float("nan")))
                if math.isfinite(value):
                    totals["positive_selection_sums"][key] = (
                        totals["positive_selection_sums"].get(key, 0.0) + value
                    )
        diagnostics = getattr(self.augmentor, "last_coverage_diagnostics", {})
        if not diagnostics:
            return
        totals = self._synthetic_epoch_diagnostics
        for key in (
            "attempted_count",
            "valid_count",
            "fitted_window_count",
            "surrogate_accepted_count",
            "fallback_count",
            "fallback_attempt_count",
            "fallback_rejected_count",
            "direct_count",
            "direct_attempt_count",
            "invalid_count",
            "hard_count",
            "soft_count",
            "candidate_total_count",
            "candidate_finite_count",
            "candidate_plausible_count",
            "candidate_below_gain_floor_count",
            "candidate_above_cap_count",
            "candidate_hard_eligible_count",
            "candidate_soft_eligible_count",
            "candidate_rms_rescue_count",
            "reference_floor_check_count",
            "reference_floor_hit_count",
            "reference_scale_fallback_count",
            "boundary_attempt_count",
            "boundary_valid_count",
            "boundary_insufficient_count",
            "boundary_context_invalid_count",
            "boundary_nonfinite_count",
            "boundary_collapsed_count",
            "boundary_unknown_count",
            "candidate_normalized_gain_count",
            "candidate_score_gain_count",
            "above_cap_normalized_gain_count",
            "rms_rescue_sample_count",
            "support_eligible_count",
            "support_collapsed_count",
            "support_direction_clipped_count",
            "candidate_support_clipped_count",
            "candidate_support_violation_count",
            "candidate_context_valid_count",
            "candidate_positive_gain_count",
            "candidate_eligible_count",
            "selected_orientation_count",
            "scope_propagated_sample_count",
            "scope_propagated_node_count",
            "frontend_effective_candidate_count",
            "frontend_effective_eligible_count",
            "frontend_effective_below_count",
            "frontend_effective_above_count",
            "frontend_effective_graph_nonlocal_count",
            "frontend_effective_rescue_used_count",
            "frontend_primary_nonlegacy_pool_count",
            "frontend_primary_nonlegacy_eligible_count",
            "frontend_primary_nonlegacy_selected_count",
        ):
            totals[key] += int(diagnostics.get(key, 0))
        for key in (
            "fallback_mechanism_gain_sum",
            "fallback_relation_gain_sum",
            "fallback_temporal_gain_sum",
            "sample_weight_sum",
            "candidate_normalized_gain_sum",
            "candidate_score_gain_sum",
            "above_cap_normalized_gain_sum",
            "candidate_rms_sum",
            "selected_delta_rms_sum",
            "rms_rescue_delta_rms_sum",
            "selected_orientation_sum",
            "frontend_effective_selected_d_fd_sum",
            "frontend_effective_selected_graph_drift_sum",
        ):
            totals[key] += float(diagnostics.get(key, 0.0))
        for key in ("attempted_seed_counts", "valid_seed_counts"):
            values = diagnostics.get(key)
            if values is None:
                continue
            values = torch.as_tensor(values, dtype=torch.float64)
            if totals[key] is None:
                totals[key] = torch.zeros_like(values)
            totals[key] += values
        for name, count in diagnostics.get("fallback_type_counts", {}).items():
            totals["fallback_type_counts"][name] = (
                totals["fallback_type_counts"].get(name, 0) + int(count)
            )
        for name, value in diagnostics.get("fallback_type_mechanism_gain_sums", {}).items():
            totals["fallback_type_mechanism_gain_sums"][name] = (
                totals["fallback_type_mechanism_gain_sums"].get(name, 0.0)
                + float(value)
            )
        for key in (
            "direct_type_attempted_counts",
            "direct_type_valid_counts",
            "direct_type_hard_counts",
            "direct_type_soft_counts",
            "direct_type_invalid_counts",
            "direct_type_candidate_total_counts",
            "direct_type_candidate_finite_counts",
            "direct_type_candidate_plausible_counts",
            "direct_type_candidate_below_floor_counts",
            "direct_type_candidate_above_cap_counts",
            "direct_type_candidate_hard_eligible_counts",
            "direct_type_candidate_soft_eligible_counts",
            "direct_type_candidate_rms_rescue_counts",
            "direct_type_reference_floor_checks",
            "direct_type_reference_floor_hits",
            "direct_type_reference_scale_fallback_counts",
            "direct_type_boundary_attempt_counts",
            "direct_type_boundary_valid_counts",
            "direct_type_boundary_insufficient_counts",
            "direct_type_boundary_context_invalid_counts",
            "direct_type_boundary_nonfinite_counts",
            "direct_type_boundary_collapsed_counts",
            "direct_type_boundary_unknown_counts",
            "direct_type_candidate_normalized_gain_counts",
            "direct_type_candidate_score_gain_counts",
            "direct_type_above_cap_normalized_gain_counts",
            "direct_type_support_eligible_counts",
            "direct_type_support_collapsed_counts",
            "direct_type_support_direction_clipped_counts",
            "direct_type_candidate_support_clipped_counts",
            "direct_type_candidate_support_violation_counts",
            "direct_type_candidate_context_valid_counts",
            "direct_type_candidate_positive_gain_counts",
            "direct_type_candidate_eligible_counts",
            "direct_type_selected_orientation_counts",
        ):
            for name, value in diagnostics.get(key, {}).items():
                totals[key][name] = totals[key].get(name, 0) + int(value)
        for key in (
            "direct_type_score_sums",
            "direct_type_target_sums",
            "direct_type_reference_scale_sums",
            "direct_type_normalized_gain_sums",
            "direct_type_sample_weight_sums",
            "direct_type_candidate_normalized_gain_sums",
            "direct_type_candidate_score_gain_sums",
            "direct_type_above_cap_normalized_gain_sums",
            "direct_type_candidate_rms_sums",
            "direct_type_selected_orientation_sums",
        ):
            for name, value in diagnostics.get(key, {}).items():
                totals[key][name] = totals[key].get(name, 0.0) + float(value)
        for key in (
            "direct_segment_attempt_counts",
            "direct_segment_selected_counts",
        ):
            for scale, value in diagnostics.get(key, {}).items():
                totals[key][scale] = totals[key].get(scale, 0) + int(value)
        for key in (
            "direct_segment_attempt_length_sums",
            "direct_segment_selected_length_sums",
        ):
            for scale, value in diagnostics.get(key, {}).items():
                totals[key][scale] = totals[key].get(scale, 0.0) + float(value)
        for key in (
            "direct_type_segment_attempt_counts",
            "direct_type_segment_selected_counts",
        ):
            for anomaly_type, scale_counts in diagnostics.get(key, {}).items():
                destination = totals[key].setdefault(anomaly_type, {})
                for scale, value in scale_counts.items():
                    destination[scale] = destination.get(scale, 0) + int(value)
        valid_count = int(diagnostics.get("valid_count", 0))
        model_diagnostics = getattr(self.augmentor, "last_diagnostics", {})
        if valid_count <= 0 or not model_diagnostics:
            return
        model_keys = (
            "direct_score_gap",
            "direct_gain_gap",
            "direct_hr1",
            "direct_hrk",
            "consistency_exclusion_ratio",
            "positive_evidence_drift",
            "negative_evidence_change",
            "positive_adjacency_drift",
            "negative_adjacency_drift",
            "window_sensor_coupling_error",
            "mil_pooling_entropy",
            "mil_effective_node_count",
            "direct_node_calibrated_gain",
            "unmodified_node_drift",
            "positive_node_drift",
            "normal_reference_scale_min",
            "normal_reference_scale_mean",
            "normal_reference_scale_max",
        )
        finite_values = {}
        for key in model_keys:
            value = float(model_diagnostics.get(key, float("nan")))
            if math.isfinite(value):
                finite_values[key] = value
        if not finite_values:
            return
        totals["model_diagnostic_count"] += valid_count
        for key, value in finite_values.items():
            totals["model_diagnostic_sums"][key] = (
                totals["model_diagnostic_sums"].get(key, 0.0) + value * valid_count
            )
        for source_name, source_values in model_diagnostics.get(
            "source_model_diagnostics", {}
        ).items():
            source_count = int(source_values.get("count", 0))
            if source_count <= 0:
                continue
            source_total = totals["source_model_diagnostics"].setdefault(
                source_name,
                {"count": 0, "sums": {}},
            )
            source_total["count"] += source_count
            for key in ("direct_score_gap", "direct_gain_gap", "direct_hr1", "direct_hrk"):
                value = float(source_values.get(key, float("nan")))
                if math.isfinite(value):
                    source_total["sums"][key] = (
                        source_total["sums"].get(key, 0.0) + value * source_count
                    )

    def _print_synthetic_epoch_diagnostics(self):
        totals = getattr(self, "_synthetic_epoch_diagnostics", None)
        if not totals:
            return
        positive_batches = totals.get("positive_selection_batch_count", 0)
        if positive_batches > 0:
            positive_means = {
                key: value / positive_batches
                for key, value in totals["positive_selection_sums"].items()
            }
            print(
                "[SHARK positive selection] "
                f"batches={positive_batches} "
                f"candidate_graph_drift={positive_means.get('candidate_graph_drift_mean', float('nan')):.4f} "
                f"selected_graph_drift={positive_means.get('selected_graph_drift_mean', float('nan')):.4f} "
                f"candidate_temporal_drift={positive_means.get('candidate_temporal_drift_mean', float('nan')):.4f} "
                f"selected_temporal_drift={positive_means.get('selected_temporal_drift_mean', float('nan')):.4f} "
                f"temporal_floor={positive_means.get('temporal_floor_mean', float('nan')):.4f} "
                f"graph_floor={positive_means.get('graph_floor_mean', float('nan')):.4f}",
                flush=True,
            )
        if totals["attempted_count"] <= 0:
            return
        attempted = totals["attempted_count"]
        attempted_counts = totals["attempted_seed_counts"]
        valid_counts = totals["valid_seed_counts"]
        attempted_entropy = self._normalized_seed_entropy(attempted_counts)
        valid_entropy = self._normalized_seed_entropy(valid_counts)
        seed_coverage = (
            float((valid_counts > 0).double().mean()) if valid_counts is not None else 0.0
        )
        selector = self.augmentor.frontend_graph_effective_selector
        if selector is not None:
            selector_diagnostics = selector.running_diagnostics()
            selector_proposals = torch.as_tensor(
                selector_diagnostics["frontend_effective_proposal_counts"]
            )
            selector_selected = torch.as_tensor(
                selector_diagnostics["frontend_effective_selected_counts"]
            )
            selector_valid = selector.valid_nodes.detach().bool().cpu()
            proposal_coverage = float(
                (selector_proposals[selector_valid] > 0).float().mean()
            )
            selected_coverage = float(
                (selector_selected[selector_valid] > 0).float().mean()
            )
            print(
                "[SHARK frontend-effective selector epoch] "
                f"dataset={self.cfg.DATA.NAME} epoch={self.cur_epoch + 1} "
                f"candidates={totals['frontend_effective_candidate_count']} "
                f"eligible={totals['frontend_effective_eligible_count']} "
                f"eligible_ratio={totals['frontend_effective_eligible_count'] / max(totals['frontend_effective_candidate_count'], 1):.3f} "
                f"below={totals['frontend_effective_below_count']} "
                f"above={totals['frontend_effective_above_count']} "
                f"graph_nonlocal={totals['frontend_effective_graph_nonlocal_count']} "
                f"coverage_rescue={totals.get('frontend_effective_rescue_used_count', 0)} "
                f"nonlegacy_pool={totals.get('frontend_primary_nonlegacy_pool_count', 0)} "
                f"nonlegacy_eligible={totals.get('frontend_primary_nonlegacy_eligible_count', 0)} "
                f"nonlegacy_selected={totals.get('frontend_primary_nonlegacy_selected_count', 0)} "
                f"selected_D_fd={totals['frontend_effective_selected_d_fd_sum'] / max(totals['valid_count'], 1):.3f} "
                f"selected_D_adj_nonincident={totals['frontend_effective_selected_graph_drift_sum'] / max(totals['valid_count'], 1):.4f} "
                f"proposal_node_coverage={proposal_coverage:.3f} "
                f"selected_node_coverage={selected_coverage:.3f}",
                flush=True,
            )
        fallback_types = ",".join(
            f"{name}:{count / attempted:.3f}"
            for name, count in sorted(totals["fallback_type_counts"].items())
        ) or "none"
        diagnostic_count = max(totals["model_diagnostic_count"], 1)
        model_means = {
            key: value / diagnostic_count
            for key, value in totals["model_diagnostic_sums"].items()
        }
        source_parts = []
        for source_name in (
            "surrogate",
            "fallback",
            "direct",
            "acceptance_hard",
            "acceptance_soft",
            "within_bound_soft",
            "rms_rescue",
        ):
            source_total = totals["source_model_diagnostics"].get(source_name)
            if not source_total or source_total["count"] <= 0:
                continue
            count = source_total["count"]
            means = {
                key: value / count for key, value in source_total["sums"].items()
            }
            source_parts.append(
                f"{source_name}:ratio={count / max(totals['valid_count'], 1):.3f}/"
                f"gap={means.get('direct_score_gap', float('nan')):.3f}/"
                f"gain={means.get('direct_gain_gap', float('nan')):.3f}/"
                f"hr1={means.get('direct_hr1', float('nan')):.3f}/"
                f"hrk={means.get('direct_hrk', float('nan')):.3f}"
            )
        source_text = ",".join(source_parts) or "none"
        direct_type_parts = []
        for name, type_attempted in sorted(
            totals["direct_type_attempted_counts"].items()
        ):
            type_valid = totals["direct_type_valid_counts"].get(name, 0)
            divisor = max(type_valid, 1)
            candidate_total = totals["direct_type_candidate_total_counts"].get(
                name, 0
            )
            floor_checks = totals["direct_type_reference_floor_checks"].get(
                name, 0
            )
            boundary_attempts = totals["direct_type_boundary_attempt_counts"].get(
                name, 0
            )
            candidate_gain_count = totals[
                "direct_type_candidate_normalized_gain_counts"
            ].get(name, 0)
            candidate_score_gain_count = totals[
                "direct_type_candidate_score_gain_counts"
            ].get(name, 0)
            candidate_eligible = totals[
                "direct_type_candidate_eligible_counts"
            ].get(name, 0)
            support_eligible = totals[
                "direct_type_support_eligible_counts"
            ].get(name, 0)
            part = (
                f"{name}:attempt={type_attempted / attempted:.3f}/"
                f"valid={type_valid / max(type_attempted, 1):.3f}/"
                f"hard={totals['direct_type_hard_counts'].get(name, 0) / max(type_attempted, 1):.3f}/"
                f"soft={totals['direct_type_soft_counts'].get(name, 0) / max(type_attempted, 1):.3f}/"
                f"invalid={totals['direct_type_invalid_counts'].get(name, 0) / max(type_attempted, 1):.3f}/"
                f"weight={totals['direct_type_sample_weight_sums'].get(name, 0.0) / divisor:.3f}/"
                f"score={totals['direct_type_score_sums'].get(name, 0.0) / divisor:.3f}/"
                f"target={totals['direct_type_target_sums'].get(name, 0.0) / divisor:.3f}/"
                f"gain={totals['direct_type_normalized_gain_sums'].get(name, 0.0) / divisor:.3f}/"
                f"cand_finite={totals['direct_type_candidate_finite_counts'].get(name, 0) / max(candidate_total, 1):.3f}/"
                f"cand_plausible={totals['direct_type_candidate_plausible_counts'].get(name, 0) / max(candidate_total, 1):.3f}/"
                f"cand_below={totals['direct_type_candidate_below_floor_counts'].get(name, 0)}/"
                f"cand_above={totals['direct_type_candidate_above_cap_counts'].get(name, 0)}/"
                f"cand_hard={totals['direct_type_candidate_hard_eligible_counts'].get(name, 0)}/"
                f"cand_soft={totals['direct_type_candidate_soft_eligible_counts'].get(name, 0)}/"
                f"cand_rescue={totals['direct_type_candidate_rms_rescue_counts'].get(name, 0)}/"
                f"cand_gain={totals['direct_type_candidate_normalized_gain_sums'].get(name, 0.0) / max(candidate_gain_count, 1):.3f}/"
                f"cand_score_gain={totals['direct_type_candidate_score_gain_sums'].get(name, 0.0) / max(candidate_score_gain_count, 1):.3f}/"
                f"context_valid={totals['direct_type_candidate_context_valid_counts'].get(name, 0) / max(candidate_total, 1):.3f}/"
                f"positive_gain={totals['direct_type_candidate_positive_gain_counts'].get(name, 0) / max(candidate_eligible, 1):.3f}/"
                f"eligible_accept={type_valid / max(support_eligible, 1):.3f}/"
                f"support_collapsed={totals['direct_type_support_collapsed_counts'].get(name, 0) / max(type_attempted, 1):.3f}/"
                f"support_clipped={totals['direct_type_candidate_support_clipped_counts'].get(name, 0) / max(candidate_total, 1):.3f}/"
                f"support_violation={totals['direct_type_candidate_support_violation_counts'].get(name, 0)}/"
                f"orientation={totals['direct_type_selected_orientation_sums'].get(name, 0.0) / max(totals['direct_type_selected_orientation_counts'].get(name, 0), 1):.3f}/"
                f"boundary_valid={totals['direct_type_boundary_valid_counts'].get(name, 0) / max(boundary_attempts, 1):.3f}/"
                f"boundary_context_invalid={totals['direct_type_boundary_context_invalid_counts'].get(name, 0) / max(boundary_attempts, 1):.3f}/"
                f"boundary_collapsed={totals['direct_type_boundary_collapsed_counts'].get(name, 0) / max(boundary_attempts, 1):.3f}/"
                f"scale_fallback={totals['direct_type_reference_scale_fallback_counts'].get(name, 0) / max(totals['direct_type_boundary_valid_counts'].get(name, 0), 1):.3f}/"
                f"floor_hit={totals['direct_type_reference_floor_hits'].get(name, 0) / max(floor_checks, 1):.3f}"
            )
            source_total = totals["source_model_diagnostics"].get(f"direct_{name}")
            if source_total and source_total["count"] > 0:
                source_count = source_total["count"]
                source_means = {
                    key: value / source_count
                    for key, value in source_total["sums"].items()
                }
                part += (
                    f"/node_gap={source_means.get('direct_score_gap', float('nan')):.3f}"
                    f"/node_gain={source_means.get('direct_gain_gap', float('nan')):.3f}"
                    f"/hr1={source_means.get('direct_hr1', float('nan')):.3f}"
                    f"/hrk={source_means.get('direct_hrk', float('nan')):.3f}"
                )
            direct_type_parts.append(part)
        direct_type_text = ",".join(direct_type_parts) or "none"
        segment_scales = []
        for scale, attempt_count in sorted(
            totals["direct_segment_attempt_counts"].items()
        ):
            selected_count = totals["direct_segment_selected_counts"].get(
                scale, 0
            )
            attempt_length = totals["direct_segment_attempt_length_sums"].get(
                scale, 0.0
            )
            selected_length = totals["direct_segment_selected_length_sums"].get(
                scale, 0.0
            )
            segment_scales.append(
                f"{scale}:attempt={attempt_count}/selected={selected_count}/"
                f"mean_len={selected_length / max(selected_count, 1):.1f}/"
                f"attempt_len={attempt_length / max(attempt_count, 1):.1f}"
            )
        segment_scale_text = ",".join(segment_scales) or "none"
        type_scale_parts = []
        for anomaly_type, scale_counts in sorted(
            totals["direct_type_segment_attempt_counts"].items()
        ):
            selected_counts = totals["direct_type_segment_selected_counts"].get(
                anomaly_type,
                {},
            )
            scales = "/".join(
                f"{scale}:{count}>{selected_counts.get(scale, 0)}"
                for scale, count in sorted(scale_counts.items())
            )
            type_scale_parts.append(f"{anomaly_type}[{scales}]")
        type_scale_text = ",".join(type_scale_parts) or "none"
        fallback_count = max(totals["fallback_count"], 1)
        fallback_attempt_count = max(totals["fallback_attempt_count"], 1)
        direct_attempt_count = max(totals["direct_attempt_count"], 1)
        print(
            "[SHARK synthetic epoch] "
            f"dataset={self.cfg.DATA.NAME} epoch={self.cur_epoch + 1} "
            f"attempted={attempted} valid={totals['valid_count']} "
            f"valid_ratio={totals['valid_count'] / attempted:.3f} "
            f"fitted_ratio={totals['fitted_window_count'] / attempted:.3f} "
            f"surrogate_ratio={totals['surrogate_accepted_count'] / attempted:.3f} "
            f"fallback_ratio={totals['fallback_count'] / attempted:.3f} "
            f"fallback_validity={totals['fallback_count'] / fallback_attempt_count:.3f} "
            f"fallback_mechanism_gain={totals['fallback_mechanism_gain_sum'] / fallback_count:.3f} "
            f"direct_ratio={totals['direct_count'] / attempted:.3f} "
            f"direct_validity={totals['direct_count'] / direct_attempt_count:.3f} "
            f"hard_coverage={totals['hard_count'] / attempted:.3f} "
            f"soft_coverage={totals['soft_count'] / attempted:.3f} "
            f"effective_weighted_count={totals['sample_weight_sum']:.3f} "
            f"mean_sample_weight={totals['sample_weight_sum'] / max(totals['valid_count'], 1):.3f} "
            f"scope_samples={totals['scope_propagated_sample_count']} "
            f"scope_nodes={totals['scope_propagated_node_count']} "
            f"scope_node_mean={totals['scope_propagated_node_count'] / max(totals['valid_count'], 1):.3f} "
            f"rms_rescue_coverage={totals['rms_rescue_sample_count'] / attempted:.3f} "
            f"rms_rescue_rms={totals['rms_rescue_delta_rms_sum'] / max(totals['rms_rescue_sample_count'], 1):.3f} "
            f"selected_rms={totals['selected_delta_rms_sum'] / max(totals['valid_count'], 1):.3f} "
            f"invalid_ratio={totals['invalid_count'] / attempted:.3f} "
            f"candidate_finite={totals['candidate_finite_count'] / max(totals['candidate_total_count'], 1):.3f} "
            f"candidate_plausible={totals['candidate_plausible_count'] / max(totals['candidate_total_count'], 1):.3f} "
            f"candidate_below_floor={totals['candidate_below_gain_floor_count']} "
            f"candidate_above_cap={totals['candidate_above_cap_count']} "
            f"candidate_hard={totals['candidate_hard_eligible_count']} "
            f"candidate_soft={totals['candidate_soft_eligible_count']} "
            f"candidate_rescue={totals['candidate_rms_rescue_count']} "
            f"candidate_gain={totals['candidate_normalized_gain_sum'] / max(totals['candidate_normalized_gain_count'], 1):.3f} "
            f"candidate_score_gain={totals['candidate_score_gain_sum'] / max(totals['candidate_score_gain_count'], 1):.3f} "
            f"context_valid={totals['candidate_context_valid_count'] / max(totals['candidate_total_count'], 1):.3f} "
            f"positive_gain={totals['candidate_positive_gain_count'] / max(totals['candidate_eligible_count'], 1):.3f} "
            f"eligible_acceptance={totals['valid_count'] / max(totals['support_eligible_count'], 1):.3f} "
            f"support_collapsed={totals['support_collapsed_count'] / attempted:.3f} "
            f"support_clipped={totals['candidate_support_clipped_count'] / max(totals['candidate_total_count'], 1):.3f} "
            f"support_violation={totals['candidate_support_violation_count']} "
            f"orientation={totals['selected_orientation_sum'] / max(totals['selected_orientation_count'], 1):.3f} "
            f"above_cap_gain={totals['above_cap_normalized_gain_sum'] / max(totals['above_cap_normalized_gain_count'], 1):.3f} "
            f"boundary_valid={totals['boundary_valid_count'] / max(totals['boundary_attempt_count'], 1):.3f} "
            f"boundary_context_invalid={totals['boundary_context_invalid_count'] / max(totals['boundary_attempt_count'], 1):.3f} "
            f"boundary_collapsed={totals['boundary_collapsed_count'] / max(totals['boundary_attempt_count'], 1):.3f} "
            f"scale_fallback={totals['reference_scale_fallback_count'] / max(totals['boundary_valid_count'], 1):.3f} "
            f"reference_floor_hit={totals['reference_floor_hit_count'] / max(totals['reference_floor_check_count'], 1):.3f} "
            f"seed_coverage={seed_coverage:.3f} "
            f"attempted_seed_entropy={attempted_entropy:.3f} "
            f"valid_seed_entropy={valid_entropy:.3f} "
            f"direct_gap={model_means.get('direct_score_gap', float('nan')):.3f} "
            f"direct_gain_gap={model_means.get('direct_gain_gap', float('nan')):.3f} "
            f"direct_hr1={model_means.get('direct_hr1', float('nan')):.3f} "
            f"direct_hrk={model_means.get('direct_hrk', float('nan')):.3f} "
            f"consistency_exclusion={model_means.get('consistency_exclusion_ratio', 0.0):.3f} "
            f"positive_drift={model_means.get('positive_evidence_drift', float('nan')):.3f} "
            f"negative_change={model_means.get('negative_evidence_change', float('nan')):.3f} "
            f"coupling_error={model_means.get('window_sensor_coupling_error', float('nan')):.3e} "
            f"mil_entropy={model_means.get('mil_pooling_entropy', float('nan')):.3f} "
            f"mil_effective_nodes={model_means.get('mil_effective_node_count', float('nan')):.3f} "
            f"direct_cal_gain={model_means.get('direct_node_calibrated_gain', float('nan')):.3f} "
            f"unmodified_drift={model_means.get('unmodified_node_drift', float('nan')):.3f} "
            f"positive_node_drift={model_means.get('positive_node_drift', float('nan')):.3f} "
            f"ref_scale={model_means.get('normal_reference_scale_min', float('nan')):.3f}/"
            f"{model_means.get('normal_reference_scale_mean', float('nan')):.3f}/"
            f"{model_means.get('normal_reference_scale_max', float('nan')):.3f} "
            f"fallback_types=({fallback_types}) "
            f"event_scales=({segment_scale_text}) "
            f"type_scales=({type_scale_text}) "
            f"direct_types=({direct_type_text}) "
            f"sources=({source_text})",
            flush=True,
        )

    @staticmethod
    def _normalized_seed_entropy(counts):
        if counts is None or counts.numel() <= 1 or counts.sum() <= 0:
            return 0.0
        probs = counts[counts > 0] / counts.sum()
        return float((-(probs * probs.log()).sum() / torch.log(counts.new_tensor(counts.numel()))).item())

    def _format_progress_loss(self, losses, name):
        try:
            index = tuple(self.loss_names).index(name)
        except ValueError:
            return "n/a"
        if index >= len(losses):
            return "n/a"
        return f"{float(losses[index].detach()):.3f}"

    def _progress_values(self, losses, lr):
        return {
            "total": self._format_progress_loss(losses, "Total"),
            "direct": self._format_progress_loss(losses, "DirectNode"),
            "contrast": self._format_progress_loss(losses, "Contrastive"),
            "rank": self._format_progress_loss(losses, "Rank"),
            "lr": f"{self._current_group_lr('base', lr):.1e}",
            "peft_lr": f"{self._current_group_lr('peft', lr):.1e}",
        }

    def _print_progress_snapshot(self, current, total, values, elapsed):
        fraction = current / max(total, 1)
        width = 20
        complete = min(int(round(width * fraction)), width)
        bar = "#" * complete + "-" * (width - complete)
        eta = elapsed / max(current, 1) * max(total - current, 0)
        metrics = " ".join(f"{key}={value}" for key, value in values.items())
        print(
            f"[SHARK progress] {self.cfg.DATA.NAME} "
            f"epoch={self.cur_epoch + 1}/{self.cfg.SOLVER.MAX_EPOCH} "
            f"batch={current}/{total} [{bar}] {100.0 * fraction:5.1f}% "
            f"{metrics} elapsed={self._format_duration(elapsed)} "
            f"eta={self._format_duration(eta)}",
            flush=True,
        )

    @staticmethod
    def _format_duration(seconds):
        seconds = max(int(round(seconds)), 0)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _current_group_lr(self, group_name, fallback):
        if not hasattr(self.optimizer, "param_groups"):
            return fallback
        for group in self.optimizer.param_groups:
            if group.get("name") == group_name:
                return float(group.get("lr", fallback))
        return fallback

    def _maybe_audit_weighted_gradients(self, losses, ready=True):
        epoch = int(getattr(self, "cur_epoch", -1)) + 1
        if (
            not self.model.training
            or not bool(ready)
            or epoch not in (1, 5, 10, 20, 30)
        ):
            return
        completed = getattr(self, "_gradient_audit_epochs", set())
        if epoch in completed:
            return

        named_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        parameter_groups = {
            "peft": [
                parameter
                for name, parameter in named_parameters
                if (
                    name.startswith("sensor_graph_prompt.")
                    or name.startswith("gfm.")
                )
            ],
            "direct_head": [
                parameter
                for name, parameter in named_parameters
                if name.startswith("head.direct_node_head.")
            ],
        }
        relation_residual_enabled = bool(
            getattr(
                getattr(self.cfg.SHARK, "RELATION_RESIDUAL_CONTRAST", None),
                "ENABLE",
                False,
            )
        )
        if relation_residual_enabled:
            parameter_groups["relation_path"] = [
                parameter
                for name, parameter in named_parameters
                if name.startswith(RELATION_PATH_PARAMETER_PREFIXES)
            ]
        local_global_enabled = bool(
            getattr(
                getattr(self.cfg.SHARK, "LOCAL_GLOBAL_CONTRAST", None),
                "ENABLE",
                False,
            )
        )
        if local_global_enabled:
            parameter_groups["local_global_path"] = [
                parameter
                for name, parameter in named_parameters
                if name.startswith(RELATION_PATH_PARAMETER_PREFIXES)
            ]
        decoupled_enabled = bool(
            getattr(
                getattr(
                    self.cfg.SHARK,
                    "DECOUPLED_LOCAL_GLOBAL_CONTRAST",
                    None,
                ),
                "ENABLE",
                False,
            )
        )
        if decoupled_enabled:
            parameter_groups["decoupled_local_global_path"] = [
                parameter
                for name, parameter in named_parameters
                if name.startswith(RELATION_PATH_PARAMETER_PREFIXES)
            ]
        native_evidence_contrast_enabled = bool(
            getattr(
                getattr(
                    self.cfg.SHARK,
                    "NATIVE_EVIDENCE_MECHANISM_CONTRAST",
                    None,
                ),
                "ENABLE",
                False,
            )
        )
        if native_evidence_contrast_enabled:
            parameter_groups["native_evidence_backend"] = [
                parameter
                for name, parameter in named_parameters
                if name.startswith("gfm.")
            ]
            parameter_groups["native_evidence_shared_path"] = [
                parameter
                for name, parameter in named_parameters
                if name.startswith(RELATION_PATH_PARAMETER_PREFIXES)
            ]
        tail_window_contrast_enabled = bool(
            getattr(
                getattr(
                    self.cfg.SHARK,
                    "TAIL_PRESERVING_WINDOW_CONTRAST",
                    None,
                ),
                "ENABLE",
                False,
            )
        )
        if tail_window_contrast_enabled:
            window_risk_prefixes = RELATION_PATH_PARAMETER_PREFIXES + (
                "gfm.",
                "sensor_graph_prompt.",
                "head.direct_node_head.",
            )
            parameter_groups["tail_window_risk_path"] = [
                parameter
                for name, parameter in named_parameters
                if name.startswith(window_risk_prefixes)
            ]
        if self.otrcl_enabled:
            if bool(self.otrcl_settings.TEMPORAL_ENABLE):
                parameter_groups["otrcl_temporal_path"] = [
                    parameter
                    for name, parameter in named_parameters
                    if name.startswith("temporal_regime_encoder.")
                ]
            if bool(self.otrcl_settings.RELATIONAL_ENABLE):
                parameter_groups["otrcl_relation_path"] = [
                    parameter
                    for name, parameter in named_parameters
                    if (
                        name.startswith("gfm_adapter.")
                        or name.startswith("sensor_graph_prompt.")
                        or name.startswith("gfm.")
                    )
                ]
            parameter_groups["otrcl_native_gfm_path"] = [
                parameter
                for name, parameter in named_parameters
                if (
                    name.startswith("gfm_adapter.")
                    or name.startswith("sensor_graph_prompt.")
                    or name.startswith("gfm.")
                )
            ]
            parameter_groups["otrcl_ipra_path"] = [
                parameter
                for name, parameter in named_parameters
                if name.startswith("information_residual.")
            ]
            parameter_groups["otrcl_direct_head_path"] = [
                parameter
                for name, parameter in named_parameters
                if name.startswith("head.direct_node_head.")
            ]
            parameter_groups["otrcl_inference_shared_path"] = [
                parameter
                for name, parameter in named_parameters
                if not name.startswith("temporal_regime_encoder.")
            ]
        epoch_records = {}
        for group_name, parameters in parameter_groups.items():
            if not parameters:
                continue
            gradients = {}
            for loss_name, loss in losses.items():
                if not loss.requires_grad:
                    gradients[loss_name] = tuple(None for _ in parameters)
                    continue
                gradients[loss_name] = torch.autograd.grad(
                    loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )

            norms = {
                loss_name: self._gradient_norm(group_gradients)
                for loss_name, group_gradients in gradients.items()
            }
            gradient_health = {
                loss_name: self._gradient_health(group_gradients)
                for loss_name, group_gradients in gradients.items()
            }
            cosine_pairs = [
                ("Contrastive", "Node"),
                ("Contrastive", "Rank"),
                ("Node", "Rank"),
            ]
            if "GlobalContrastive" in gradients:
                cosine_pairs.extend(
                    (
                        ("GlobalContrastive", "Node"),
                        ("GlobalContrastive", "Rank"),
                    )
                )
            if "LocalContrastive" in gradients:
                cosine_pairs.extend(
                    (
                        ("LocalContrastive", "Node"),
                        ("LocalContrastive", "Rank"),
                    )
                )
            if "EvidenceContrastive" in gradients:
                cosine_pairs.extend(
                    (
                        ("EvidenceContrastive", "Node"),
                        ("EvidenceContrastive", "Rank"),
                    )
                )
            if (
                "GlobalContrastive" in gradients
                and "EvidenceContrastive" in gradients
            ):
                cosine_pairs.append(
                    ("GlobalContrastive", "EvidenceContrastive")
                )
            for auxiliary_name in ("WindowMechanism", "NormalTail"):
                if auxiliary_name in gradients:
                    cosine_pairs.extend(
                        (
                            (auxiliary_name, "Node"),
                            (auxiliary_name, "Rank"),
                            (auxiliary_name, "WindowBCE"),
                        )
                    )
                    if "GlobalContrastive" in gradients:
                        cosine_pairs.append(
                            ("GlobalContrastive", auxiliary_name)
                        )
            if (
                "GlobalContrastive" in gradients
                and "LocalContrastive" in gradients
            ):
                cosine_pairs.append(
                    ("GlobalContrastive", "LocalContrastive")
                )
            cosines = {
                f"{left}-{right}": self._gradient_cosine(
                    gradients[left],
                    gradients[right],
                )
                for left, right in cosine_pairs
            }
            cosine_health = {
                f"{left}-{right}": self._gradient_cosine_health(
                    gradients[left],
                    gradients[right],
                )
                for left, right in cosine_pairs
            }
            epoch_records[group_name] = {
                "parameter_count": sum(
                    int(parameter.numel()) for parameter in parameters
                ),
                "gradient_norms": {
                    name: (
                        float(value) if math.isfinite(float(value)) else None
                    )
                    for name, value in norms.items()
                },
                "gradient_cosines": {
                    name: (
                        float(value) if math.isfinite(float(value)) else None
                    )
                    for name, value in cosines.items()
                },
                "gradient_health": gradient_health,
                "cosine_health": cosine_health,
            }
            decoupled_text = ""
            if "GlobalContrastive" in norms:
                decoupled_text += (
                    f" global_contrastive={norms['GlobalContrastive']:.6e}"
                    " cos_global_node="
                    f"{cosines['GlobalContrastive-Node']:.6f}"
                    " cos_global_rank="
                    f"{cosines['GlobalContrastive-Rank']:.6f}"
                )
            if "LocalContrastive" in norms:
                decoupled_text += (
                    f" local_contrastive={norms['LocalContrastive']:.6e}"
                    " cos_local_node="
                    f"{cosines['LocalContrastive-Node']:.6f}"
                    " cos_local_rank="
                    f"{cosines['LocalContrastive-Rank']:.6f}"
                )
            if "GlobalContrastive-LocalContrastive" in cosines:
                decoupled_text += (
                    " cos_global_local="
                    f"{cosines['GlobalContrastive-LocalContrastive']:.6f}"
                )
            if "EvidenceContrastive" in norms:
                decoupled_text += (
                    " evidence_contrastive="
                    f"{norms['EvidenceContrastive']:.6e}"
                    " cos_evidence_node="
                    f"{cosines['EvidenceContrastive-Node']:.6f}"
                    " cos_evidence_rank="
                    f"{cosines['EvidenceContrastive-Rank']:.6f}"
                )
            if "WindowMechanism" in norms:
                decoupled_text += (
                    " window_mechanism="
                    f"{norms['WindowMechanism']:.6e}"
                    " cos_window_rank="
                    f"{cosines['WindowMechanism-Rank']:.6f}"
                    " cos_window_bce="
                    f"{cosines['WindowMechanism-WindowBCE']:.6f}"
                )
            if "NormalTail" in norms:
                decoupled_text += (
                    f" normal_tail={norms['NormalTail']:.6e}"
                    " cos_tail_rank="
                    f"{cosines['NormalTail-Rank']:.6f}"
                    " cos_tail_bce="
                    f"{cosines['NormalTail-WindowBCE']:.6f}"
                )
            print(
                "[SHARK gradient audit] "
                f"dataset={self.cfg.DATA.NAME} epoch={epoch} group={group_name} "
                f"contrastive={norms['Contrastive']:.6e} "
                f"node={norms['Node']:.6e} "
                f"rank={norms['Rank']:.6e} "
                f"window_bce={norms['WindowBCE']:.6e} "
                f"cos_contrastive_node={cosines['Contrastive-Node']:.6f} "
                f"cos_contrastive_rank={cosines['Contrastive-Rank']:.6f} "
                f"cos_node_rank={cosines['Node-Rank']:.6f}"
                f"{decoupled_text}",
                flush=True,
            )
        completed.add(epoch)
        self._gradient_audit_epochs = completed
        records = getattr(self, "_gradient_audit_records", {})
        records[str(epoch)] = epoch_records
        self._gradient_audit_records = records

    def gradient_audit_records(self):
        return dict(getattr(self, "_gradient_audit_records", {}))

    def _record_local_global_contrast(self, result, training):
        """累计局部—全局目标的运行证据，避免“能跑”等同于“已生效”。"""
        stats = getattr(
            self,
            "_local_global_contrast_stats",
            {
                "calls": 0,
                "training_calls": 0,
                "active_calls": 0,
                "training_active_calls": 0,
                "sample_count": 0,
                "training_sample_count": 0,
                "direct_node_count": 0,
                "unchanged_node_count": 0,
                "training_direct_node_count": 0,
                "training_unchanged_node_count": 0,
                "weighted_global_gap_sum": 0.0,
                "weighted_direct_gap_sum": 0.0,
                "weighted_unchanged_drift_sum": 0.0,
                "weighted_counterfactual_gap_sum": 0.0,
            },
        )
        stats["calls"] += 1
        if training:
            stats["training_calls"] += 1
        sample_count = int(result["sample_count"].detach().cpu())
        direct_count = int(result["direct_node_count"].detach().cpu())
        unchanged_count = int(result["unchanged_node_count"].detach().cpu())
        if sample_count > 0:
            stats["active_calls"] += 1
            stats["sample_count"] += sample_count
            stats["direct_node_count"] += direct_count
            stats["unchanged_node_count"] += unchanged_count
            if training:
                stats["training_active_calls"] += 1
                stats["training_sample_count"] += sample_count
                stats["training_direct_node_count"] += direct_count
                stats["training_unchanged_node_count"] += unchanged_count
                for source, target in (
                    ("global_gap", "weighted_global_gap_sum"),
                    ("direct_gap", "weighted_direct_gap_sum"),
                    ("unchanged_drift", "weighted_unchanged_drift_sum"),
                    ("counterfactual_gap", "weighted_counterfactual_gap_sum"),
                ):
                    stats[target] += (
                        float(result[source].detach().cpu()) * sample_count
                    )
        self._local_global_contrast_stats = stats

        if training and int(getattr(self, "cur_iter", -1)) in (0, 1):
            settings = self.cfg.SHARK.LOCAL_GLOBAL_CONTRAST
            print(
                "[SHARK local-global contrast] "
                f"dataset={self.cfg.DATA.NAME} "
                f"arm={settings.EXPERIMENT_ARM} "
                f"valid_samples={sample_count} direct_nodes={direct_count} "
                f"unchanged_nodes={unchanged_count} "
                f"global_gap={float(result['global_gap'].detach().cpu()):.6f} "
                f"direct_gap={float(result['direct_gap'].detach().cpu()):.6f} "
                "unchanged_drift="
                f"{float(result['unchanged_drift'].detach().cpu()):.6f} "
                "combined_gap="
                f"{float(result['counterfactual_gap'].detach().cpu()):.6f}",
                flush=True,
            )

    def local_global_contrast_diagnostics(self):
        """返回可写入 smoke/primary 审计文件的累计诊断。"""
        stats = dict(
            getattr(
                self,
                "_local_global_contrast_stats",
                {
                    "calls": 0,
                    "training_calls": 0,
                    "active_calls": 0,
                    "training_active_calls": 0,
                    "sample_count": 0,
                    "training_sample_count": 0,
                    "direct_node_count": 0,
                    "unchanged_node_count": 0,
                    "training_direct_node_count": 0,
                    "training_unchanged_node_count": 0,
                    "weighted_global_gap_sum": 0.0,
                    "weighted_direct_gap_sum": 0.0,
                    "weighted_unchanged_drift_sum": 0.0,
                    "weighted_counterfactual_gap_sum": 0.0,
                },
            )
        )
        count = max(int(stats["training_sample_count"]), 1)
        settings = getattr(self.cfg.SHARK, "LOCAL_GLOBAL_CONTRAST", None)
        return {
            "enabled": bool(getattr(settings, "ENABLE", False)),
            "experiment_arm": str(
                getattr(settings, "EXPERIMENT_ARM", "disabled")
            ),
            "calls": int(stats["calls"]),
            "training_calls": int(stats["training_calls"]),
            "active_calls": int(stats["active_calls"]),
            "training_active_calls": int(stats["training_active_calls"]),
            "sample_count": int(stats["sample_count"]),
            "training_sample_count": int(stats["training_sample_count"]),
            "direct_node_count": int(stats["direct_node_count"]),
            "unchanged_node_count": int(stats["unchanged_node_count"]),
            "training_direct_node_count": int(
                stats["training_direct_node_count"]
            ),
            "training_unchanged_node_count": int(
                stats["training_unchanged_node_count"]
            ),
            "global_gap_mean": stats["weighted_global_gap_sum"] / count,
            "direct_gap_mean": stats["weighted_direct_gap_sum"] / count,
            "unchanged_drift_mean": (
                stats["weighted_unchanged_drift_sum"] / count
            ),
            "counterfactual_gap_mean": (
                stats["weighted_counterfactual_gap_sum"] / count
            ),
        }

    def _record_decoupled_local_global_contrast(self, result, training):
        """累计解耦全局/局部分量，供 Smoke 判断分支是否真正激活。"""
        stats = getattr(
            self,
            "_decoupled_local_global_contrast_stats",
            {
                "calls": 0,
                "training_calls": 0,
                "active_calls": 0,
                "training_active_calls": 0,
                "sample_count": 0,
                "training_sample_count": 0,
                "local_sample_count": 0,
                "training_local_sample_count": 0,
                "direct_node_count": 0,
                "training_direct_node_count": 0,
                "global_weight_sum": 0.0,
                "training_global_weight_sum": 0.0,
                "local_weight_sum": 0.0,
                "training_local_weight_sum": 0.0,
                "weighted_global_loss_sum": 0.0,
                "weighted_local_loss_sum": 0.0,
                "weighted_local_contribution_sum": 0.0,
                "weighted_global_gap_sum": 0.0,
                "weighted_direct_gap_sum": 0.0,
                "local_gap_measured": False,
            },
        )
        stats["calls"] += 1
        if training:
            stats["training_calls"] += 1
        sample_count = int(result["sample_count"].detach().cpu())
        local_sample_count = int(
            result["local_sample_count"].detach().cpu()
        )
        direct_node_count = int(result["direct_node_count"].detach().cpu())
        global_weight_sum = float(result["global_weight_sum"].detach().cpu())
        local_weight_sum = float(result["local_weight_sum"].detach().cpu())
        local_measured = bool(result.get("local_measured", False))
        if sample_count > 0:
            stats["active_calls"] += 1
            stats["sample_count"] += sample_count
            stats["local_sample_count"] += local_sample_count
            stats["direct_node_count"] += direct_node_count
            stats["global_weight_sum"] += global_weight_sum
            stats["local_weight_sum"] += local_weight_sum
            if training:
                stats["training_active_calls"] += 1
                stats["training_sample_count"] += sample_count
                stats["training_local_sample_count"] += local_sample_count
                stats["training_direct_node_count"] += direct_node_count
                stats["training_global_weight_sum"] += global_weight_sum
                stats["training_local_weight_sum"] += local_weight_sum
                stats["weighted_global_loss_sum"] += (
                    float(result["global"].detach().cpu()) * global_weight_sum
                )
                stats["weighted_global_gap_sum"] += (
                    float(result["global_gap"].detach().cpu()) * global_weight_sum
                )
                if local_measured and local_weight_sum > 0.0:
                    stats["local_gap_measured"] = True
                    stats["weighted_local_loss_sum"] += (
                        float(result["local"].detach().cpu())
                        * local_weight_sum
                    )
                    stats["weighted_local_contribution_sum"] += (
                        float(result["weighted_local"].detach().cpu())
                        * local_weight_sum
                    )
                    stats["weighted_direct_gap_sum"] += (
                        float(result["direct_gap"].detach().cpu())
                        * local_weight_sum
                    )
        self._decoupled_local_global_contrast_stats = stats

        if training and int(getattr(self, "cur_iter", -1)) in (0, 1):
            settings = self.cfg.SHARK.DECOUPLED_LOCAL_GLOBAL_CONTRAST
            print(
                "[SHARK 解耦局部-全局对比] "
                f"dataset={self.cfg.DATA.NAME} "
                f"arm={settings.EXPERIMENT_ARM} "
                f"valid_samples={sample_count} "
                f"local_samples={local_sample_count} "
                f"direct_nodes={direct_node_count} "
                f"global_loss={float(result['global'].detach().cpu()):.6f} "
                f"local_loss={float(result['local'].detach().cpu()):.6f} "
                "weighted_local="
                f"{float(result['weighted_local'].detach().cpu()):.6f} "
                f"global_gap={float(result['global_gap'].detach().cpu()):.6f} "
                f"direct_gap={float(result['direct_gap'].detach().cpu()):.6f}",
                flush=True,
            )

    def decoupled_local_global_contrast_diagnostics(self):
        """返回解耦目标的累计损失、间隔和有效监督数量。"""
        stats = dict(
            getattr(
                self,
                "_decoupled_local_global_contrast_stats",
                {
                    "calls": 0,
                    "training_calls": 0,
                    "active_calls": 0,
                    "training_active_calls": 0,
                    "sample_count": 0,
                    "training_sample_count": 0,
                    "local_sample_count": 0,
                    "training_local_sample_count": 0,
                    "direct_node_count": 0,
                    "training_direct_node_count": 0,
                    "global_weight_sum": 0.0,
                    "training_global_weight_sum": 0.0,
                    "local_weight_sum": 0.0,
                    "training_local_weight_sum": 0.0,
                    "weighted_global_loss_sum": 0.0,
                    "weighted_local_loss_sum": 0.0,
                    "weighted_local_contribution_sum": 0.0,
                    "weighted_global_gap_sum": 0.0,
                    "weighted_direct_gap_sum": 0.0,
                    "local_gap_measured": False,
                },
            )
        )
        global_count = max(float(stats["training_global_weight_sum"]), 1.0)
        local_count = max(float(stats["training_local_weight_sum"]), 1.0)
        settings = getattr(
            self.cfg.SHARK,
            "DECOUPLED_LOCAL_GLOBAL_CONTRAST",
            None,
        )
        return {
            "enabled": bool(getattr(settings, "ENABLE", False)),
            "experiment_arm": str(
                getattr(settings, "EXPERIMENT_ARM", "disabled")
            ),
            "calls": int(stats["calls"]),
            "training_calls": int(stats["training_calls"]),
            "active_calls": int(stats["active_calls"]),
            "training_active_calls": int(stats["training_active_calls"]),
            "sample_count": int(stats["sample_count"]),
            "training_sample_count": int(stats["training_sample_count"]),
            "local_sample_count": int(stats["local_sample_count"]),
            "training_local_sample_count": int(
                stats["training_local_sample_count"]
            ),
            "direct_node_count": int(stats["direct_node_count"]),
            "training_direct_node_count": int(
                stats["training_direct_node_count"]
            ),
            "global_weight_sum": float(stats["global_weight_sum"]),
            "training_global_weight_sum": float(
                stats["training_global_weight_sum"]
            ),
            "local_weight_sum": float(stats["local_weight_sum"]),
            "training_local_weight_sum": float(
                stats["training_local_weight_sum"]
            ),
            "local_gap_measured": bool(stats["local_gap_measured"]),
            "global_loss_mean": (
                stats["weighted_global_loss_sum"] / global_count
            ),
            "local_loss_mean": (
                stats["weighted_local_loss_sum"] / local_count
            ),
            "weighted_local_loss_mean": (
                stats["weighted_local_contribution_sum"] / local_count
            ),
            "global_gap_mean": (
                stats["weighted_global_gap_sum"] / global_count
            ),
            "direct_gap_mean": (
                stats["weighted_direct_gap_sum"] / local_count
            ),
        }

    @staticmethod
    def _native_evidence_contrast_stat_defaults():
        return {
            "calls": 0,
            "training_calls": 0,
            "training_active_calls": 0,
            "sample_count": 0,
            "training_sample_count": 0,
            "local_sample_count": 0,
            "training_local_sample_count": 0,
            "direct_node_count": 0,
            "training_direct_node_count": 0,
            "training_global_weight_sum": 0.0,
            "training_local_weight_sum": 0.0,
            "weighted_global_loss_sum": 0.0,
            "weighted_local_loss_sum": 0.0,
            "weighted_local_contribution_sum": 0.0,
            "weighted_pair_loss_sum": 0.0,
            "weighted_global_gap_sum": 0.0,
            "weighted_pair_gap_sum": 0.0,
            "weighted_reference_evidence_sum": 0.0,
            "weighted_negative_evidence_sum": 0.0,
            "weighted_pair_satisfaction_sum": 0.0,
            "local_measured": False,
        }

    def _record_native_evidence_mechanism_contrast(self, result, training):
        """累计后端通用节点异常证据对比的损失和证据方向。"""
        stats = getattr(
            self,
            "_native_evidence_mechanism_contrast_stats",
            self._native_evidence_contrast_stat_defaults(),
        )
        stats["calls"] += 1
        if training:
            stats["training_calls"] += 1
        sample_count = int(result["sample_count"].detach().cpu())
        local_sample_count = int(
            result["local_sample_count"].detach().cpu()
        )
        direct_node_count = int(result["direct_node_count"].detach().cpu())
        global_weight_sum = float(result["global_weight_sum"].detach().cpu())
        local_weight_sum = float(result["local_weight_sum"].detach().cpu())
        local_measured = bool(result.get("local_measured", False))
        stats["sample_count"] += sample_count
        stats["local_sample_count"] += local_sample_count
        stats["direct_node_count"] += direct_node_count
        if training and sample_count > 0:
            stats["training_active_calls"] += 1
            stats["training_sample_count"] += sample_count
            stats["training_local_sample_count"] += local_sample_count
            stats["training_direct_node_count"] += direct_node_count
            stats["training_global_weight_sum"] += global_weight_sum
            stats["weighted_global_loss_sum"] += (
                float(result["global"].detach().cpu()) * global_weight_sum
            )
            stats["weighted_global_gap_sum"] += (
                float(result["global_gap"].detach().cpu()) * global_weight_sum
            )
            if local_measured and local_weight_sum > 0.0:
                stats["local_measured"] = True
                stats["training_local_weight_sum"] += local_weight_sum
                for source, target in (
                    ("local", "weighted_local_loss_sum"),
                    ("weighted_local", "weighted_local_contribution_sum"),
                    ("local_pair", "weighted_pair_loss_sum"),
                    ("pair_gap", "weighted_pair_gap_sum"),
                    ("reference_evidence", "weighted_reference_evidence_sum"),
                    ("negative_evidence", "weighted_negative_evidence_sum"),
                    ("pair_satisfaction", "weighted_pair_satisfaction_sum"),
                ):
                    stats[target] += (
                        float(result[source].detach().cpu())
                        * local_weight_sum
                    )
        self._native_evidence_mechanism_contrast_stats = stats

        if training and int(getattr(self, "cur_iter", -1)) in (0, 1):
            settings = self.cfg.SHARK.NATIVE_EVIDENCE_MECHANISM_CONTRAST
            print(
                "[SHARK 原生证据机制对比] "
                f"dataset={self.cfg.DATA.NAME} arm={settings.EXPERIMENT_ARM} "
                f"valid_samples={sample_count} local_samples={local_sample_count} "
                f"direct_nodes={direct_node_count} "
                f"pair_gap={float(result['pair_gap'].detach().cpu()):.6f} "
                "reference_evidence="
                f"{float(result['reference_evidence'].detach().cpu()):.6f} "
                "negative_evidence="
                f"{float(result['negative_evidence'].detach().cpu()):.6f} "
                f"local_loss={float(result['local'].detach().cpu()):.6f}",
                flush=True,
            )

    def native_evidence_mechanism_contrast_diagnostics(self):
        stats = dict(
            getattr(
                self,
                "_native_evidence_mechanism_contrast_stats",
                self._native_evidence_contrast_stat_defaults(),
            )
        )
        global_count = max(float(stats["training_global_weight_sum"]), 1.0)
        local_count = max(float(stats["training_local_weight_sum"]), 1.0)
        settings = getattr(
            self.cfg.SHARK,
            "NATIVE_EVIDENCE_MECHANISM_CONTRAST",
            None,
        )
        return {
            "enabled": bool(getattr(settings, "ENABLE", False)),
            "experiment_arm": str(
                getattr(settings, "EXPERIMENT_ARM", "disabled")
            ),
            "calls": int(stats["calls"]),
            "training_calls": int(stats["training_calls"]),
            "training_active_calls": int(stats["training_active_calls"]),
            "sample_count": int(stats["sample_count"]),
            "training_sample_count": int(stats["training_sample_count"]),
            "local_sample_count": int(stats["local_sample_count"]),
            "training_local_sample_count": int(
                stats["training_local_sample_count"]
            ),
            "direct_node_count": int(stats["direct_node_count"]),
            "training_direct_node_count": int(
                stats["training_direct_node_count"]
            ),
            "training_global_weight_sum": float(
                stats["training_global_weight_sum"]
            ),
            "training_local_weight_sum": float(
                stats["training_local_weight_sum"]
            ),
            "local_measured": bool(stats["local_measured"]),
            "global_loss_mean": (
                stats["weighted_global_loss_sum"] / global_count
            ),
            "local_loss_mean": (
                stats["weighted_local_loss_sum"] / local_count
            ),
            "weighted_local_loss_mean": (
                stats["weighted_local_contribution_sum"] / local_count
            ),
            "pair_loss_mean": (
                stats["weighted_pair_loss_sum"] / local_count
            ),
            "global_gap_mean": (
                stats["weighted_global_gap_sum"] / global_count
            ),
            "pair_gap_mean": stats["weighted_pair_gap_sum"] / local_count,
            "reference_evidence_mean": (
                stats["weighted_reference_evidence_sum"] / local_count
            ),
            "negative_evidence_mean": (
                stats["weighted_negative_evidence_sum"] / local_count
            ),
            "pair_satisfaction_mean": (
                stats["weighted_pair_satisfaction_sum"] / local_count
            ),
        }

    @staticmethod
    def _tail_window_contrast_stat_defaults():
        return {
            "calls": 0,
            "training_calls": 0,
            "training_active_calls": 0,
            "training_sample_count": 0,
            "training_hard_tail_count": 0,
            "training_weight_sum": 0.0,
            "weighted_global_loss_sum": 0.0,
            "weighted_window_loss_sum": 0.0,
            "weighted_window_contribution_sum": 0.0,
            "weighted_tail_loss_sum": 0.0,
            "weighted_tail_contribution_sum": 0.0,
            "weighted_global_gap_sum": 0.0,
            "weighted_normal_variation_sum": 0.0,
            "weighted_normal_reference_sum": 0.0,
            "weighted_negative_risk_sum": 0.0,
            "weighted_mechanism_gain_sum": 0.0,
            "weighted_required_gain_sum": 0.0,
            "weighted_window_pair_gap_sum": 0.0,
            "weighted_pair_satisfaction_sum": 0.0,
            "weighted_normal_center_sum": 0.0,
            "weighted_hard_tail_reference_sum": 0.0,
            "weighted_hard_tail_mean_sum": 0.0,
            "weighted_tail_excess_sum": 0.0,
            "weighted_tail_violation_sum": 0.0,
            "window_measured": False,
            "tail_measured": False,
        }

    def _record_tail_preserving_window_contrast(self, result, training):
        """累计窗口机制间隔与正常尾部压缩的运行证据。"""
        stats = getattr(
            self,
            "_tail_preserving_window_contrast_stats",
            self._tail_window_contrast_stat_defaults(),
        )
        stats["calls"] += 1
        if training:
            stats["training_calls"] += 1
        sample_count = int(result["sample_count"].detach().cpu())
        hard_tail_count = int(result["hard_tail_count"].detach().cpu())
        weight_sum = float(result["weight_sum"].detach().cpu())
        if training and sample_count > 0 and weight_sum > 0.0:
            stats["training_active_calls"] += 1
            stats["training_sample_count"] += sample_count
            stats["training_hard_tail_count"] += hard_tail_count
            stats["training_weight_sum"] += weight_sum
            for source, target in (
                ("global", "weighted_global_loss_sum"),
                ("window", "weighted_window_loss_sum"),
                ("weighted_window", "weighted_window_contribution_sum"),
                ("tail", "weighted_tail_loss_sum"),
                ("weighted_tail", "weighted_tail_contribution_sum"),
                ("global_gap", "weighted_global_gap_sum"),
                ("normal_variation", "weighted_normal_variation_sum"),
                ("normal_reference", "weighted_normal_reference_sum"),
                ("negative_risk", "weighted_negative_risk_sum"),
                ("mechanism_gain", "weighted_mechanism_gain_sum"),
                ("required_gain", "weighted_required_gain_sum"),
                ("window_pair_gap", "weighted_window_pair_gap_sum"),
                ("pair_satisfaction", "weighted_pair_satisfaction_sum"),
                ("normal_center_reference", "weighted_normal_center_sum"),
                (
                    "hard_normal_tail_reference",
                    "weighted_hard_tail_reference_sum",
                ),
                ("hard_normal_tail_mean", "weighted_hard_tail_mean_sum"),
                ("tail_excess_mean", "weighted_tail_excess_sum"),
                ("tail_violation_rate", "weighted_tail_violation_sum"),
            ):
                stats[target] += float(result[source].detach().cpu()) * weight_sum
            stats["window_measured"] = bool(
                stats["window_measured"] or result["window_measured"]
            )
            stats["tail_measured"] = bool(
                stats["tail_measured"] or result["tail_measured"]
            )
        self._tail_preserving_window_contrast_stats = stats

        if training and int(getattr(self, "cur_iter", -1)) in (0, 1):
            settings = self.cfg.SHARK.TAIL_PRESERVING_WINDOW_CONTRAST
            print(
                "[SHARK 尾部保持窗口机制对比] "
                f"dataset={self.cfg.DATA.NAME} arm={settings.EXPERIMENT_ARM} "
                f"valid_samples={sample_count} hard_tail={hard_tail_count} "
                "normal_variation="
                f"{float(result['normal_variation'].detach().cpu()):.6f} "
                f"mechanism_gain={float(result['mechanism_gain'].detach().cpu()):.6f} "
                f"window_pair_gap={float(result['window_pair_gap'].detach().cpu()):.6f} "
                f"tail_excess={float(result['tail_excess_mean'].detach().cpu()):.6f} "
                "tail_violation_rate="
                f"{float(result['tail_violation_rate'].detach().cpu()):.6f}",
                flush=True,
            )

    def tail_preserving_window_contrast_diagnostics(self):
        stats = dict(
            getattr(
                self,
                "_tail_preserving_window_contrast_stats",
                self._tail_window_contrast_stat_defaults(),
            )
        )
        count = max(float(stats["training_weight_sum"]), 1.0)
        settings = getattr(
            self.cfg.SHARK,
            "TAIL_PRESERVING_WINDOW_CONTRAST",
            None,
        )
        mean_fields = {
            "global_loss_mean": "weighted_global_loss_sum",
            "window_loss_mean": "weighted_window_loss_sum",
            "weighted_window_loss_mean": "weighted_window_contribution_sum",
            "tail_loss_mean": "weighted_tail_loss_sum",
            "weighted_tail_loss_mean": "weighted_tail_contribution_sum",
            "global_gap_mean": "weighted_global_gap_sum",
            "normal_variation_mean": "weighted_normal_variation_sum",
            "normal_reference_mean": "weighted_normal_reference_sum",
            "negative_risk_mean": "weighted_negative_risk_sum",
            "mechanism_gain_mean": "weighted_mechanism_gain_sum",
            "required_gain_mean": "weighted_required_gain_sum",
            "window_pair_gap_mean": "weighted_window_pair_gap_sum",
            "pair_satisfaction_mean": "weighted_pair_satisfaction_sum",
            "normal_center_reference_mean": "weighted_normal_center_sum",
            "hard_normal_tail_reference_mean": (
                "weighted_hard_tail_reference_sum"
            ),
            "hard_normal_tail_mean": "weighted_hard_tail_mean_sum",
            "tail_excess_mean": "weighted_tail_excess_sum",
            "tail_violation_rate_mean": "weighted_tail_violation_sum",
        }
        diagnostics = {
            "enabled": bool(getattr(settings, "ENABLE", False)),
            "experiment_arm": str(
                getattr(settings, "EXPERIMENT_ARM", "disabled")
            ),
            "calls": int(stats["calls"]),
            "training_calls": int(stats["training_calls"]),
            "training_active_calls": int(stats["training_active_calls"]),
            "training_sample_count": int(stats["training_sample_count"]),
            "training_hard_tail_count": int(
                stats["training_hard_tail_count"]
            ),
            "training_weight_sum": float(stats["training_weight_sum"]),
            "window_measured": bool(stats["window_measured"]),
            "tail_measured": bool(stats["tail_measured"]),
        }
        diagnostics.update(
            {
                output_name: stats[source_name] / count
                for output_name, source_name in mean_fields.items()
            }
        )
        return diagnostics

    @staticmethod
    def _gradient_norm(gradients):
        squared_norm = None
        for gradient in gradients:
            if gradient is None:
                continue
            value = gradient.detach().float().square().sum()
            squared_norm = value if squared_norm is None else squared_norm + value
        if squared_norm is None:
            return 0.0
        return float(squared_norm.sqrt().cpu())

    @staticmethod
    def _gradient_health(gradients):
        """区分缺失、零值、有限非零和非有限梯度。

        ``_gradient_norm`` 为了兼容旧审计会把缺失梯度记为 0、把非有限
        梯度转成 ``None``。这里保留更细的状态，避免 NaN/Inf 被误判为
        “没有梯度”而放过运行审计。
        """
        present = [
            gradient.detach().float()
            for gradient in gradients
            if gradient is not None
        ]
        if not present:
            return {
                "status": "missing",
                "tensor_count": len(gradients),
                "present_tensor_count": 0,
                "finite_tensor_count": 0,
                "nonfinite_tensor_count": 0,
                "norm": 0.0,
                "has_gradient": False,
                "finite": True,
                "nonzero": False,
            }
        finite_count = sum(
            int(torch.isfinite(gradient).all()) for gradient in present
        )
        nonfinite_count = len(present) - finite_count
        if nonfinite_count:
            return {
                "status": "nonfinite",
                "tensor_count": len(gradients),
                "present_tensor_count": len(present),
                "finite_tensor_count": finite_count,
                "nonfinite_tensor_count": nonfinite_count,
                "norm": None,
                "has_gradient": True,
                "finite": False,
                "nonzero": False,
            }
        squared_norm = sum(
            gradient.double().square().sum() for gradient in present
        )
        norm = float(squared_norm.sqrt().cpu())
        if not math.isfinite(norm):
            status = "nonfinite"
            finite = False
        elif norm <= 0.0:
            status = "finite_zero"
            finite = True
        else:
            status = "finite_nonzero"
            finite = True
        return {
            "status": status,
            "tensor_count": len(gradients),
            "present_tensor_count": len(present),
            "finite_tensor_count": finite_count,
            "nonfinite_tensor_count": nonfinite_count,
            "norm": norm if math.isfinite(norm) else None,
            "has_gradient": True,
            "finite": finite,
            "nonzero": bool(status == "finite_nonzero"),
        }

    @staticmethod
    def _gradient_cosine(left_gradients, right_gradients):
        dot = None
        left_norm = None
        right_norm = None
        for left, right in zip(left_gradients, right_gradients):
            if left is None or right is None:
                continue
            left = left.detach().float()
            right = right.detach().float()
            pair_dot = (left * right).sum()
            pair_left_norm = left.square().sum()
            pair_right_norm = right.square().sum()
            dot = pair_dot if dot is None else dot + pair_dot
            left_norm = (
                pair_left_norm if left_norm is None else left_norm + pair_left_norm
            )
            right_norm = (
                pair_right_norm
                if right_norm is None
                else right_norm + pair_right_norm
            )
        if dot is None or float(left_norm) <= 0.0 or float(right_norm) <= 0.0:
            return float("nan")
        cosine = dot / (left_norm.sqrt() * right_norm.sqrt()).clamp_min(1e-12)
        return float(cosine.clamp(min=-1.0, max=1.0).cpu())

    @staticmethod
    def _gradient_cosine_health(left_gradients, right_gradients):
        """报告梯度余弦是否可定义，并显式标记非有限输入。"""
        pairs = [
            (left.detach().float(), right.detach().float())
            for left, right in zip(left_gradients, right_gradients)
            if left is not None and right is not None
        ]
        if not pairs:
            return {
                "status": "missing",
                "pair_count": 0,
                "finite_pair_count": 0,
                "nonfinite_pair_count": 0,
                "defined": False,
                "finite": True,
                "left_nonzero": False,
                "right_nonzero": False,
            }
        finite_pairs = [
            (left, right)
            for left, right in pairs
            if torch.isfinite(left).all() and torch.isfinite(right).all()
        ]
        nonfinite_count = len(pairs) - len(finite_pairs)
        if nonfinite_count:
            return {
                "status": "nonfinite",
                "pair_count": len(pairs),
                "finite_pair_count": len(finite_pairs),
                "nonfinite_pair_count": nonfinite_count,
                "defined": False,
                "finite": False,
                "left_nonzero": False,
                "right_nonzero": False,
            }
        left_squared = sum(
            left.double().square().sum() for left, _ in finite_pairs
        )
        right_squared = sum(
            right.double().square().sum() for _, right in finite_pairs
        )
        left_norm = float(left_squared.sqrt().cpu())
        right_norm = float(right_squared.sqrt().cpu())
        if (
            not math.isfinite(left_norm)
            or not math.isfinite(right_norm)
        ):
            status = "nonfinite"
            finite = False
            defined = False
        elif left_norm <= 0.0 or right_norm <= 0.0:
            status = "zero_norm"
            finite = True
            defined = False
        else:
            status = "finite"
            finite = True
            defined = True
        return {
            "status": status,
            "pair_count": len(pairs),
            "finite_pair_count": len(finite_pairs),
            "nonfinite_pair_count": nonfinite_count,
            "defined": defined,
            "finite": finite,
            "left_nonzero": (
                bool(left_norm > 0.0) if math.isfinite(left_norm) else False
            ),
            "right_nonzero": (
                bool(right_norm > 0.0) if math.isfinite(right_norm) else False
            ),
        }

    @torch.no_grad()
    def _otrcl_graph_evaluator(self, windows):
        if not hasattr(self.model.imlp, "forward_components"):
            raise RuntimeError(
                "OTR-CL graph orthogonality requires explicit DLinear components"
            )
        seasonal, trend = self.model.imlp.forward_components(windows.float())
        return self.model.graph_builder((seasonal + trend).detach())

    def _otrcl_model_forward(self, windows, **kwargs):
        if self.otrcl_enabled:
            kwargs["compute_otrcl"] = bool(
                getattr(self.otrcl_settings, "TEMPORAL_ENABLE", False)
                or getattr(self.otrcl_settings, "RELATIONAL_ENABLE", False)
            )
        return self.model(windows, **kwargs)

    def _forward_loss(self, windows):
        frozen_frontend = None
        with torch.no_grad():
            reuse_frozen_frontend = (
                hasattr(self.model, "encode_frozen_frontend")
                and hasattr(self.model, "frozen_frontend_reuse_available")
                and self.model.frozen_frontend_reuse_available()
            )
            if reuse_frozen_frontend:
                frozen_frontend = self.model.encode_frozen_frontend(
                    windows.float()
                )
                base_node_features = frozen_frontend["node_features"]
                base_adjacency = frozen_frontend["adjacency"]
            else:
                base_node_features = self.model.imlp(windows.float())
                base_adjacency = self.model.graph_builder(base_node_features)

        token_bridge = getattr(self.cfg.SHARK, "TOKEN_BRIDGE", None)
        joint_bridge_training = bool(
            getattr(token_bridge, "JOINT_TRAINING", False)
        )
        if self.otrcl_enabled and joint_bridge_training:
            raise ValueError(
                "OTR-CL v1 does not combine with joint Token Bridge retention"
            )
        can_reuse_frozen_anchor = (
            frozen_frontend is not None
            and hasattr(self.model, "forward_from_frozen_frontend")
            and not self.otrcl_enabled
            and not joint_bridge_training
        )
        if can_reuse_frozen_anchor:
            outputs_anchor = self.model.forward_from_frozen_frontend(
                frozen_frontend
            )
        elif joint_bridge_training:
            outputs_anchor = self._otrcl_model_forward(
                windows,
                compute_bridge_retention=True,
            )
        else:
            outputs_anchor = self._otrcl_model_forward(windows)
        node_evidence = self._build_counterfactual_node_evidence(outputs_anchor, base_adjacency)
        progress = self._training_progress()
        collect_diagnostics = (
            getattr(self.cfg.SHARK.SYNTHETIC, "PRINT_DIAGNOSTICS", False)
            and self.model.training
            and self.cur_iter in (0, 1)
        )
        relation_residual_enabled = bool(
            getattr(
                getattr(self.cfg.SHARK, "RELATION_RESIDUAL_CONTRAST", None),
                "ENABLE",
                False,
            )
        )
        local_global_settings = getattr(
            self.cfg.SHARK,
            "LOCAL_GLOBAL_CONTRAST",
            None,
        )
        local_global_enabled = bool(
            getattr(local_global_settings, "ENABLE", False)
        )
        decoupled_local_global_settings = getattr(
            self.cfg.SHARK,
            "DECOUPLED_LOCAL_GLOBAL_CONTRAST",
            None,
        )
        decoupled_local_global_enabled = bool(
            getattr(decoupled_local_global_settings, "ENABLE", False)
        )
        native_evidence_contrast_settings = getattr(
            self.cfg.SHARK,
            "NATIVE_EVIDENCE_MECHANISM_CONTRAST",
            None,
        )
        native_evidence_contrast_enabled = bool(
            getattr(native_evidence_contrast_settings, "ENABLE", False)
        )
        tail_window_contrast_settings = getattr(
            self.cfg.SHARK,
            "TAIL_PRESERVING_WINDOW_CONTRAST",
            None,
        )
        tail_window_contrast_enabled = bool(
            getattr(tail_window_contrast_settings, "ENABLE", False)
        )
        if native_evidence_contrast_enabled:
            backend = getattr(self.model, "gfm", None)
            if backend is None or not hasattr(backend, "backend_spec"):
                raise ValueError(
                    "原生证据机制对比要求模型暴露 GFM backend_spec"
                )
            require_node_anomaly_evidence(backend.backend_spec())
        active_contrast_candidates = [
            name
            for name, enabled in (
                ("RELATION_RESIDUAL_CONTRAST", relation_residual_enabled),
                ("LOCAL_GLOBAL_CONTRAST", local_global_enabled),
                (
                    "DECOUPLED_LOCAL_GLOBAL_CONTRAST",
                    decoupled_local_global_enabled,
                ),
                (
                    "NATIVE_EVIDENCE_MECHANISM_CONTRAST",
                    native_evidence_contrast_enabled,
                ),
                (
                    "TAIL_PRESERVING_WINDOW_CONTRAST",
                    tail_window_contrast_enabled,
                ),
                ("OTRCL", self.otrcl_enabled),
            )
            if enabled
        ]
        if len(active_contrast_candidates) > 1:
            raise ValueError(
                "局部对比候选必须保持单因素互斥，当前同时启用: "
                + ", ".join(active_contrast_candidates)
            )
        if decoupled_local_global_enabled and (
            relation_residual_enabled or local_global_enabled
        ):
            active = []
            if relation_residual_enabled:
                active.append("RELATION_RESIDUAL_CONTRAST")
            if local_global_enabled:
                active.append("LOCAL_GLOBAL_CONTRAST")
            raise ValueError(
                "解耦局部—全局对比是隔离单因素实验，不能同时启用: "
                + ", ".join(active)
            )
        if relation_residual_enabled and local_global_enabled:
            raise ValueError(
                "关系残差对比与局部—全局反事实对比是互斥的单因素候选"
            )
        temporal_views = None
        negative_schedule = self.augmentor.next_coverage_balanced_schedule(
            windows.size(0)
        )
        selector_reference = (
            None
            if self.augmentor.requires_frozen_gfm_teacher
            else base_node_features
        )
        if self.otrcl_enabled:
            temporal_views = self.temporal_counterfactual(
                windows,
                graph_evaluator=self._otrcl_graph_evaluator,
            )
            positive_view = temporal_views["positive_view"]
            negative_view, direct_node_mask, propagated_node_mask, anomaly_type_ids = self.augmentor.relation_function_breaking_view(
                windows,
                base_adjacency,
                node_evidence=node_evidence,
                progress=progress,
                collect_diagnostics=collect_diagnostics,
                orthogonality_reference_view=temporal_views[
                    "coarse_view"
                ],
                frontend_reference=selector_reference,
                **negative_schedule,
            )
        elif relation_residual_enabled:
            positive_view, negative_view, direct_node_mask, propagated_node_mask, anomaly_type_ids = (
                self.augmentor.mechanism_matched_views(
                    windows,
                    base_adjacency,
                    node_evidence=node_evidence,
                    progress=progress,
                    collect_diagnostics=collect_diagnostics,
                )
            )
        else:
            positive_view = self.augmentor.relation_preserving_view(
                windows,
                base_adjacency,
            )
            negative_view, direct_node_mask, propagated_node_mask, anomaly_type_ids = self.augmentor.relation_function_breaking_view(
                windows,
                base_adjacency,
                node_evidence=node_evidence,
                progress=progress,
                collect_diagnostics=collect_diagnostics,
                frontend_reference=selector_reference,
                **negative_schedule,
            )
        if self.augmentor.requires_frozen_gfm_teacher:
            (
                negative_view,
                direct_node_mask,
                propagated_node_mask,
                anomaly_type_ids,
            ) = self._apply_frozen_teacher_selection(
                windows,
                base_adjacency,
                anomaly_type_ids,
            )
        valid_negative_mask = getattr(self.augmentor, "last_valid_mask", None)
        if valid_negative_mask is None:
            valid_negative_mask = torch.ones(
                windows.size(0),
                device=windows.device,
                dtype=torch.bool,
            )
        else:
            valid_negative_mask = valid_negative_mask.bool()
        sample_weights = getattr(self.augmentor, "last_sample_weights", None)
        if sample_weights is None:
            sample_weights = valid_negative_mask.to(dtype=windows.dtype)
        else:
            sample_weights = sample_weights.detach().to(
                device=windows.device,
                dtype=windows.dtype,
            )
        if sample_weights.shape != valid_negative_mask.shape:
            raise ValueError(
                "Counterfactual sample weights must have shape [B]: "
                f"weights={tuple(sample_weights.shape)}, batch={windows.size(0)}"
            )
        if not torch.isfinite(sample_weights).all() or (sample_weights < 0).any():
            raise ValueError("Counterfactual sample weights must be finite and non-negative")
        valid_negative_mask = valid_negative_mask & (sample_weights > 0)
        fine_valid_mask = valid_negative_mask.clone()
        if temporal_views is not None:
            valid_negative_mask = valid_negative_mask & temporal_views[
                "positive_valid_mask"
            ].bool()
        valid_sample_weights = sample_weights[valid_negative_mask]
        consistency_exclusion_mask = getattr(
            self.augmentor,
            "last_consistency_exclusion_mask",
            None,
        )
        if consistency_exclusion_mask is None:
            consistency_exclusion_mask = propagated_node_mask.bool()
        else:
            consistency_exclusion_mask = consistency_exclusion_mask.to(
                device=windows.device,
                dtype=torch.bool,
            )
            if consistency_exclusion_mask.shape != direct_node_mask.shape:
                raise ValueError(
                    "Counterfactual consistency exclusion mask must have shape [B, N]: "
                    f"mask={tuple(consistency_exclusion_mask.shape)}, "
                    f"direct={tuple(direct_node_mask.shape)}"
                )

        outputs_positive = self._otrcl_model_forward(positive_view)
        outputs_negative = self._otrcl_model_forward(negative_view)
        outputs_coarse = (
            self._otrcl_model_forward(temporal_views["coarse_view"])
            if temporal_views is not None
            else None
        )
        otrcl_integrity = None
        if temporal_views is not None:
            otrcl_integrity = summarize_otrcl_view_integrity(
                anchor_windows=windows,
                view_windows={
                    "W+": positive_view,
                    "Wc": temporal_views["coarse_view"],
                    "Wf": negative_view,
                },
                anchor_outputs=outputs_anchor,
                view_outputs={
                    "W+": outputs_positive,
                    "Wc": outputs_coarse,
                    "Wf": outputs_negative,
                },
                valid_masks={
                    "W+": temporal_views["positive_valid_mask"],
                    "Wc": temporal_views["coarse_valid_mask"],
                    "Wf": fine_valid_mask,
                },
                direct_node_mask=direct_node_mask,
                propagated_node_mask=propagated_node_mask,
                consistency_exclusion_mask=consistency_exclusion_mask,
                eps=float(self.otrcl_settings.ORTHOGONALITY_EPS),
            )
            positive_invalid_count = otrcl_integrity["mask_counts"][
                "positive_invalid"
            ]
            otrcl_integrity["supervision_audit"] = {
                "Wc_window_supervision_only": True,
                "Wc_has_node_mask": False,
                "Wf_window_anomaly_supervision": True,
                "Wf_direct_node_supervision": True,
                "primary_Wf_losses_require_valid_Wplus": True,
                "primary_Wc_losses_require_valid_Wplus": True,
                "positive_node_consistency_is_validity_masked": False,
                "positive_fidelity_is_validity_masked": False,
                "positive_invalid_enters_node_consistency": bool(
                    positive_invalid_count > 0
                    and float(
                        self.cfg.SHARK.LOSS.POSITIVE_NODE_CONSISTENCY_WEIGHT
                    )
                    > 0.0
                ),
                "positive_invalid_enters_fidelity": bool(
                    positive_invalid_count > 0
                    and float(self.cfg.SHARK.LOSS.FIDELITY_WEIGHT) > 0.0
                ),
                "fine_valid_count_before_Wplus_gate": int(
                    fine_valid_mask.sum().item()
                ),
                "fine_valid_count_after_Wplus_gate": int(
                    valid_negative_mask.sum().item()
                ),
            }
        self._update_counterfactual_model_diagnostics(
            outputs_anchor,
            outputs_positive,
            outputs_negative,
            direct_node_mask,
            propagated_node_mask,
            anomaly_type_ids,
            valid_negative_mask,
            consistency_exclusion_mask,
        )
        self._maybe_print_counterfactual_diagnostics()
        relation_contrast_target = str(
            getattr(
                getattr(
                    self.cfg.SHARK,
                    "RELATION_RESIDUAL_CONTRAST",
                    None,
                ),
                "CONTRAST_TARGET",
                "relation_residual",
            )
        )
        temporal_loss = None
        loss_temporal_contrastive = outputs_anchor["window_score"].sum() * 0.0
        loss_coarse_rank = loss_temporal_contrastive
        loss_coarse_bce = loss_temporal_contrastive
        coarse_valid_mask = None
        if temporal_views is not None:
            coarse_valid_mask = temporal_views["coarse_valid_mask"].bool()
            coarse_valid_mask = coarse_valid_mask & temporal_views[
                "positive_valid_mask"
            ].bool()
            if coarse_valid_mask.any():
                if bool(self.otrcl_settings.TEMPORAL_ENABLE):
                    temporal_loss = temporal_regime_contrastive_loss(
                        outputs_anchor["temporal_regime_embedding"][
                            coarse_valid_mask
                        ],
                        outputs_positive["temporal_regime_embedding"][
                            coarse_valid_mask
                        ],
                        outputs_coarse["temporal_regime_embedding"][
                            coarse_valid_mask
                        ],
                        temperature=float(
                            self.otrcl_settings.TEMPORAL_TEMPERATURE
                        ),
                    )
                    loss_temporal_contrastive = temporal_loss["total"]
                loss_coarse_rank = ranking_loss(
                    outputs_anchor["window_score"][coarse_valid_mask],
                    outputs_coarse["window_score"][coarse_valid_mask],
                    self.cfg.SHARK.LOSS.RANK_MARGIN,
                )
                loss_coarse_bce = window_bce_loss(
                    outputs_anchor["window_score"][coarse_valid_mask],
                    outputs_coarse["window_score"][coarse_valid_mask],
                    outputs_positive["window_score"][coarse_valid_mask],
                )
            elif bool(self.otrcl_settings.TEMPORAL_ENABLE):
                temporal_loss = {
                    "total": loss_temporal_contrastive,
                    "positive_similarity": loss_temporal_contrastive.detach(),
                    "coarse_similarity": loss_temporal_contrastive.detach(),
                    "similarity_gap": loss_temporal_contrastive.detach(),
                    "sample_count": torch.zeros(
                        (), device=windows.device, dtype=torch.long
                    ),
                }
        tail_window_loss = None
        if tail_window_contrast_enabled:
            if valid_negative_mask.any():
                tail_window_loss = (
                    tail_preserving_mechanism_window_contrastive_loss(
                        outputs_anchor["graph_embedding"][valid_negative_mask],
                        outputs_positive["graph_embedding"][valid_negative_mask],
                        outputs_negative["graph_embedding"][valid_negative_mask],
                        outputs_anchor["window_score"][valid_negative_mask],
                        outputs_positive["window_score"][valid_negative_mask],
                        outputs_negative["window_score"][valid_negative_mask],
                        global_temperature=float(
                            tail_window_contrast_settings.GLOBAL_TEMPERATURE
                        ),
                        window_temperature=float(
                            tail_window_contrast_settings.WINDOW_TEMPERATURE
                        ),
                        mechanism_margin=float(
                            tail_window_contrast_settings.MECHANISM_MARGIN
                        ),
                        normal_variation_weight=float(
                            tail_window_contrast_settings.NORMAL_VARIATION_WEIGHT
                        ),
                        window_loss_weight=float(
                            tail_window_contrast_settings.WINDOW_LOSS_WEIGHT
                        ),
                        tail_loss_weight=float(
                            tail_window_contrast_settings.TAIL_LOSS_WEIGHT
                        ),
                        tail_fraction=float(
                            tail_window_contrast_settings.TAIL_FRACTION
                        ),
                        tail_temperature=float(
                            tail_window_contrast_settings.TAIL_TEMPERATURE
                        ),
                        tail_budget=float(
                            tail_window_contrast_settings.TAIL_BUDGET
                        ),
                        sample_weights=valid_sample_weights,
                    )
                )
            else:
                graph_zero = (
                    outputs_anchor["graph_embedding"].sum()
                    + outputs_positive["graph_embedding"].sum()
                    + outputs_negative["graph_embedding"].sum()
                ) * 0.0
                risk_zero = (
                    outputs_anchor["window_score"].sum()
                    + outputs_positive["window_score"].sum()
                    + outputs_negative["window_score"].sum()
                ) * 0.0
                count_zero = torch.zeros(
                    (), device=windows.device, dtype=torch.long
                )
                tail_window_loss = {
                    "total": graph_zero,
                    "global": graph_zero,
                    "window": risk_zero,
                    "weighted_window": risk_zero,
                    "tail": risk_zero,
                    "weighted_tail": risk_zero,
                    "global_gap": graph_zero.detach().clone(),
                    "normal_variation": risk_zero.detach().clone(),
                    "normal_reference": risk_zero.detach().clone(),
                    "negative_risk": risk_zero.detach().clone(),
                    "mechanism_gain": risk_zero.detach().clone(),
                    "required_gain": risk_zero.detach().clone(),
                    "window_pair_gap": risk_zero.detach().clone(),
                    "pair_satisfaction": risk_zero.detach().clone(),
                    "normal_center_reference": risk_zero.detach().clone(),
                    "hard_normal_tail_reference": risk_zero.detach().clone(),
                    "hard_normal_tail_mean": risk_zero.detach().clone(),
                    "tail_excess_mean": risk_zero.detach().clone(),
                    "tail_violation_rate": risk_zero.detach().clone(),
                    "weight_sum": risk_zero.detach().clone(),
                    "sample_count": count_zero,
                    "hard_tail_count": count_zero.clone(),
                    "window_measured": False,
                    "tail_measured": False,
                }
            self._record_tail_preserving_window_contrast(
                tail_window_loss,
                training=bool(self.model.training),
            )
            relation_loss = None
            loss_paired_contrastive = tail_window_loss["total"]
        native_evidence_loss = None
        if native_evidence_contrast_enabled:
            if valid_negative_mask.any():
                native_evidence_loss = native_evidence_mechanism_contrastive_loss(
                    outputs_anchor["graph_embedding"][valid_negative_mask],
                    outputs_positive["graph_embedding"][valid_negative_mask],
                    outputs_negative["graph_embedding"][valid_negative_mask],
                    outputs_anchor["native_node_scores"][valid_negative_mask],
                    outputs_positive["native_node_scores"][valid_negative_mask],
                    outputs_negative["native_node_scores"][valid_negative_mask],
                    direct_node_mask[valid_negative_mask],
                    global_temperature=float(
                        native_evidence_contrast_settings.GLOBAL_TEMPERATURE
                    ),
                    local_temperature=float(
                        native_evidence_contrast_settings.LOCAL_TEMPERATURE
                    ),
                    pair_margin=float(
                        native_evidence_contrast_settings.PAIR_MARGIN
                    ),
                    local_loss_weight=float(
                        native_evidence_contrast_settings.LOCAL_LOSS_WEIGHT
                    ),
                    sample_weights=valid_sample_weights,
                    eps=float(native_evidence_contrast_settings.EPS),
                    min_reference_scale=float(
                        native_evidence_contrast_settings.MIN_REFERENCE_SCALE
                    ),
                )
            else:
                graph_zero = (
                    outputs_anchor["graph_embedding"].sum()
                    + outputs_positive["graph_embedding"].sum()
                    + outputs_negative["graph_embedding"].sum()
                ) * 0.0
                evidence_zero = (
                    outputs_anchor["native_node_scores"][..., :0].sum()
                    + outputs_positive["native_node_scores"][..., :0].sum()
                    + outputs_negative["native_node_scores"][..., :0].sum()
                ) * 0.0
                native_evidence_loss = {
                    "total": graph_zero,
                    "global": graph_zero,
                    "local": evidence_zero,
                    "weighted_local": evidence_zero,
                    "local_pair": evidence_zero,
                    "global_gap": graph_zero.detach().clone(),
                    "pair_gap": evidence_zero.detach().clone(),
                    "reference_evidence": evidence_zero.detach().clone(),
                    "negative_evidence": evidence_zero.detach().clone(),
                    "pair_satisfaction": evidence_zero.detach().clone(),
                    "global_weight_sum": graph_zero.detach().clone(),
                    "local_weight_sum": evidence_zero.detach().clone(),
                    "local_measured": False,
                    "sample_count": torch.zeros(
                        (), device=windows.device, dtype=torch.long
                    ),
                    "local_sample_count": torch.zeros(
                        (), device=windows.device, dtype=torch.long
                    ),
                    "direct_node_count": torch.zeros(
                        (), device=windows.device, dtype=torch.long
                    ),
                }
            self._record_native_evidence_mechanism_contrast(
                native_evidence_loss,
                training=bool(self.model.training),
            )
            relation_loss = None
            loss_paired_contrastive = native_evidence_loss["total"]
        decoupled_loss = None
        if decoupled_local_global_enabled:
            if valid_negative_mask.any():
                decoupled_loss = decoupled_local_global_contrastive_loss(
                    outputs_anchor["graph_embedding"][valid_negative_mask],
                    outputs_positive["graph_embedding"][valid_negative_mask],
                    outputs_negative["graph_embedding"][valid_negative_mask],
                    outputs_anchor["node_embeddings"][valid_negative_mask],
                    outputs_positive["node_embeddings"][valid_negative_mask],
                    outputs_negative["node_embeddings"][valid_negative_mask],
                    direct_node_mask[valid_negative_mask],
                    global_temperature=float(
                        decoupled_local_global_settings.GLOBAL_TEMPERATURE
                    ),
                    local_temperature=float(
                        decoupled_local_global_settings.LOCAL_TEMPERATURE
                    ),
                    global_margin=float(
                        decoupled_local_global_settings.GLOBAL_MARGIN
                    ),
                    local_margin=float(
                        decoupled_local_global_settings.LOCAL_MARGIN
                    ),
                    local_loss_weight=float(
                        decoupled_local_global_settings.LOCAL_LOSS_WEIGHT
                    ),
                    sample_weights=valid_sample_weights,
                )
            else:
                graph_zero = (
                    outputs_anchor["graph_embedding"].sum()
                    + outputs_positive["graph_embedding"].sum()
                    + outputs_negative["graph_embedding"].sum()
                ) * 0.0
                node_zero = (
                    outputs_anchor["node_embeddings"].sum()
                    + outputs_positive["node_embeddings"].sum()
                    + outputs_negative["node_embeddings"].sum()
                ) * 0.0
                decoupled_loss = {
                    "total": graph_zero,
                    "global": graph_zero,
                    "local": node_zero,
                    "weighted_local": node_zero,
                    "global_gap": graph_zero.detach().clone(),
                    "direct_gap": node_zero.detach().clone(),
                    "global_weight_sum": graph_zero.detach().clone(),
                    "local_weight_sum": node_zero.detach().clone(),
                    "local_measured": False,
                    "sample_count": torch.zeros(
                        (), device=windows.device, dtype=torch.long
                    ),
                    "local_sample_count": torch.zeros(
                        (), device=windows.device, dtype=torch.long
                    ),
                    "direct_node_count": torch.zeros(
                        (), device=windows.device, dtype=torch.long
                    ),
                }
            self._record_decoupled_local_global_contrast(
                decoupled_loss,
                training=bool(self.model.training),
            )
            relation_loss = None
            loss_paired_contrastive = decoupled_loss["total"]
        if local_global_enabled:
            # 只在已通过 Graph-VAR 接受门槛的样本上比较局部—全局增量；
            # propagated 与额外 consistency exclusion 均不冒充“未变节点”。
            unchanged_node_mask = ~(
                direct_node_mask.bool()
                | propagated_node_mask.bool()
                | consistency_exclusion_mask.bool()
            )
            if valid_negative_mask.any():
                local_global_loss = (
                    mechanism_conditioned_local_global_contrastive_loss(
                        outputs_anchor["graph_embedding"][valid_negative_mask],
                        outputs_positive["graph_embedding"][valid_negative_mask],
                        outputs_negative["graph_embedding"][valid_negative_mask],
                        outputs_anchor["node_embeddings"][valid_negative_mask],
                        outputs_positive["node_embeddings"][valid_negative_mask],
                        outputs_negative["node_embeddings"][valid_negative_mask],
                        direct_node_mask[valid_negative_mask],
                        unchanged_node_mask=unchanged_node_mask[
                            valid_negative_mask
                        ],
                        temperature=float(local_global_settings.TEMPERATURE),
                        margin=float(local_global_settings.MARGIN),
                        direct_weight=float(
                            local_global_settings.DIRECT_WEIGHT
                        ),
                        unchanged_weight=float(
                            local_global_settings.UNCHANGED_WEIGHT
                        ),
                        sample_weights=valid_sample_weights,
                    )
                )
            else:
                zero = outputs_anchor["graph_embedding"].sum() * 0.0
                local_global_loss = {
                    "total": zero,
                    "global_gap": zero,
                    "direct_gap": zero,
                    "unchanged_drift": zero,
                    "counterfactual_gap": zero,
                    "sample_count": torch.zeros(
                        (), device=windows.device, dtype=torch.long
                    ),
                    "direct_node_count": torch.zeros(
                        (), device=windows.device, dtype=torch.long
                    ),
                    "unchanged_node_count": torch.zeros(
                        (), device=windows.device, dtype=torch.long
                    ),
                }
            self._record_local_global_contrast(
                local_global_loss,
                training=bool(self.model.training),
            )
            relation_loss = None
            loss_paired_contrastive = local_global_loss["total"]
        elif self.otrcl_enabled and bool(
            self.otrcl_settings.RELATIONAL_ENABLE
        ):
            required = "native_relational_residual"
            if not all(
                required in outputs
                for outputs in (
                    outputs_anchor,
                    outputs_positive,
                    outputs_negative,
                )
            ):
                raise RuntimeError(
                    "OTR-CL relation branch did not receive native GFM "
                    "relational residuals"
                )
            unchanged_node_mask = ~(
                direct_node_mask.bool()
                | propagated_node_mask.bool()
                | consistency_exclusion_mask.bool()
            )
            if valid_negative_mask.any():
                relation_loss = relational_node_contrastive_loss(
                    outputs_anchor[required][valid_negative_mask],
                    outputs_positive[required][valid_negative_mask],
                    outputs_negative[required][valid_negative_mask],
                    direct_node_mask[valid_negative_mask],
                    unchanged_node_mask[valid_negative_mask],
                    temperature=float(
                        self.otrcl_settings.RELATIONAL_TEMPERATURE
                    ),
                    unchanged_weight=float(
                        self.otrcl_settings.RELATIONAL_UNCHANGED_WEIGHT
                    ),
                    sample_weights=valid_sample_weights,
                )
            else:
                zero = (
                    outputs_anchor[required].sum()
                    + outputs_positive[required].sum()
                    + outputs_negative[required].sum()
                ) * 0.0
                relation_loss = {
                    "total": zero,
                    "direct": zero,
                    "unchanged": zero,
                    "weighted_unchanged": zero,
                    "direct_similarity_gap": zero.detach(),
                    "unchanged_drift": zero.detach(),
                    "direct_node_count": torch.zeros(
                        (), device=windows.device, dtype=torch.long
                    ),
                    "unchanged_node_count": torch.zeros(
                        (), device=windows.device, dtype=torch.long
                    ),
                    "sample_count": torch.zeros(
                        (), device=windows.device, dtype=torch.long
                    ),
                }
            loss_paired_contrastive = relation_loss["total"]
        elif relation_residual_enabled:
            if not all(
                "relation_residual" in outputs
                for outputs in (outputs_anchor, outputs_positive, outputs_negative)
            ):
                raise RuntimeError(
                    "机制关系对比已启用，但时间—图关系残差没有拟合或没有输出"
                )
            if relation_contrast_target == "relation_residual":
                relation_loss = mechanism_matched_relation_contrastive_loss(
                    outputs_anchor["relation_residual"][valid_negative_mask],
                    outputs_positive["relation_residual"][valid_negative_mask],
                    outputs_negative["relation_residual"][valid_negative_mask],
                    direct_node_mask[valid_negative_mask],
                    unchanged_node_mask=(
                        ~direct_node_mask[valid_negative_mask].bool()
                        & ~propagated_node_mask[valid_negative_mask].bool()
                    ),
                    temperature=float(
                        self.cfg.SHARK.RELATION_RESIDUAL_CONTRAST.TEMPERATURE
                    ),
                    unchanged_weight=float(
                        self.cfg.SHARK.RELATION_RESIDUAL_CONTRAST.UNCHANGED_WEIGHT
                    ),
                    sample_weights=valid_sample_weights,
                )
                loss_paired_contrastive = relation_loss["total"]
            elif relation_contrast_target == "graph_embedding":
                # 严格目标对照：视图和其他监督保持不变，只恢复旧图级对比目标。
                relation_loss = None
                loss_paired_contrastive = carots_style_contrastive_loss(
                    outputs_anchor["graph_embedding"][valid_negative_mask],
                    outputs_positive["graph_embedding"][valid_negative_mask],
                    outputs_negative["graph_embedding"][valid_negative_mask],
                    self.cfg.SHARK.CONTRAST.TEMPERATURE,
                    sample_weights=valid_sample_weights,
                )
            else:
                raise ValueError(
                    "未知机制匹配对比目标: "
                    f"{relation_contrast_target}; 可选 relation_residual 或 "
                    "graph_embedding"
                )
        elif (
            not decoupled_local_global_enabled
            and not native_evidence_contrast_enabled
            and not tail_window_contrast_enabled
        ):
            relation_loss = None
            loss_paired_contrastive = carots_style_contrastive_loss(
                outputs_anchor["graph_embedding"][valid_negative_mask],
                outputs_positive["graph_embedding"][valid_negative_mask],
                outputs_negative["graph_embedding"][valid_negative_mask],
                self.cfg.SHARK.CONTRAST.TEMPERATURE,
                sample_weights=valid_sample_weights,
            )
        batch_hard_contrast_enabled = bool(
            getattr(
                self.cfg.SHARK.LOSS,
                "BATCH_HARD_CONTRAST_ENABLE",
                False,
            )
        )
        node_aware_hard_contrast_enabled = bool(
            getattr(
                self.cfg.SHARK.LOSS,
                "NODE_AWARE_HARD_CONTRAST_ENABLE",
                False,
            )
        )
        if (
            relation_residual_enabled
            or local_global_enabled
            or decoupled_local_global_enabled
            or native_evidence_contrast_enabled
            or tail_window_contrast_enabled
            or self.otrcl_enabled
        ) and (
            batch_hard_contrast_enabled or node_aware_hard_contrast_enabled
        ):
            raise ValueError(
                "局部关系对比候选不能与旧 hard contrast 单因素候选同时启用"
            )
        if batch_hard_contrast_enabled and node_aware_hard_contrast_enabled:
            raise ValueError(
                "Batch-hard graph contrast and node-aware hard contrast "
                "are mutually exclusive single-factor candidates"
            )
        if node_aware_hard_contrast_enabled:
            loss_node_aware_hard_contrastive = (
                node_aware_hard_contrastive_loss(
                    outputs_anchor["node_embeddings"][valid_negative_mask],
                    outputs_positive["node_embeddings"][valid_negative_mask],
                    outputs_negative["node_embeddings"][valid_negative_mask],
                    direct_node_mask[valid_negative_mask],
                    outputs_anchor["direct_node_score_logits"][
                        valid_negative_mask
                    ],
                    self.cfg.SHARK.CONTRAST.TEMPERATURE,
                    self.cfg.SHARK.LOSS.NODE_AWARE_HARD_NORMAL_FRACTION,
                    self.cfg.SHARK.LOSS.NODE_AWARE_HARD_NEGATIVE_COUNT,
                    sample_weights=valid_sample_weights,
                )
                if valid_negative_mask.any()
                else outputs_anchor["node_embeddings"].sum() * 0.0
            )
            node_contrast_blend = float(
                self.cfg.SHARK.LOSS.NODE_AWARE_HARD_CONTRAST_BLEND
            )
            if not 0.0 <= node_contrast_blend <= 1.0:
                raise ValueError(
                    "NODE_AWARE_HARD_CONTRAST_BLEND must be in [0, 1]"
                )
            loss_contrastive = (
                (1.0 - node_contrast_blend) * loss_paired_contrastive
                + node_contrast_blend * loss_node_aware_hard_contrastive
            )
            loss_batch_hard_contrastive = None
        elif batch_hard_contrast_enabled:
            loss_batch_hard_contrastive = batch_hard_normal_contrastive_loss(
                outputs_anchor["graph_embedding"],
                outputs_positive["graph_embedding"],
                outputs_negative["graph_embedding"][valid_negative_mask],
                outputs_anchor["window_score"],
                self.cfg.SHARK.CONTRAST.TEMPERATURE,
                self.cfg.SHARK.LOSS.BATCH_HARD_NORMAL_FRACTION,
                self.cfg.SHARK.LOSS.BATCH_HARD_NEGATIVE_COUNT,
            )
            contrast_blend = float(
                self.cfg.SHARK.LOSS.BATCH_HARD_CONTRAST_BLEND
            )
            if not 0.0 <= contrast_blend <= 1.0:
                raise ValueError(
                    "BATCH_HARD_CONTRAST_BLEND must be in [0, 1]"
                )
            loss_contrastive = (
                (1.0 - contrast_blend) * loss_paired_contrastive
                + contrast_blend * loss_batch_hard_contrastive
            )
            loss_node_aware_hard_contrastive = None
        else:
            loss_batch_hard_contrastive = None
            loss_node_aware_hard_contrastive = None
            loss_contrastive = loss_paired_contrastive
        if temporal_views is not None:
            generator_diagnostics = temporal_views["diagnostics"]
            integrity_views = otrcl_integrity["views"]
            integrity_direction = otrcl_integrity[
                "directional_separation"
            ]

            def integrity_value(view_name, metric_name):
                value = integrity_views[view_name][metric_name][
                    "accepted_mean"
                ]
                return float(value) if value is not None else 0.0

            def directional_value(name):
                value = integrity_direction[name]
                return float(value) if value is not None else 0.0

            coarse_score_gap = outputs_anchor["window_score"].sum() * 0.0
            if coarse_valid_mask is not None and coarse_valid_mask.any():
                coarse_score_gap = (
                    outputs_coarse["window_score"][coarse_valid_mask]
                    - outputs_anchor["window_score"][coarse_valid_mask]
                ).detach().mean()
            self.last_otrcl_diagnostics = {
                "batch_size": int(windows.size(0)),
                "positive_valid_count": int(
                    temporal_views["positive_valid_mask"].sum().item()
                ),
                "coarse_valid_count": int(
                    temporal_views["coarse_valid_mask"].sum().item()
                ),
                "positive_temporal_drift_mean": float(
                    generator_diagnostics["positive_temporal_drift"].mean()
                ),
                "positive_graph_drift_mean": float(
                    generator_diagnostics["positive_graph_drift"].mean()
                ),
                "coarse_temporal_drift_mean": float(
                    generator_diagnostics["coarse_temporal_drift"].clamp_min(0.0).mean()
                ),
                "coarse_graph_drift_mean": float(
                    generator_diagnostics["coarse_graph_drift"][
                        torch.isfinite(
                            generator_diagnostics["coarse_graph_drift"]
                        )
                    ].mean()
                    if torch.isfinite(
                        generator_diagnostics["coarse_graph_drift"]
                    ).any()
                    else 0.0
                ),
                "coarse_score_gap": float(coarse_score_gap),
                "temporal_similarity_gap": float(
                    temporal_loss["similarity_gap"]
                    if temporal_loss is not None
                    else 0.0
                ),
                "relational_similarity_gap": float(
                    relation_loss["direct_similarity_gap"]
                    if relation_loss is not None
                    and "direct_similarity_gap" in relation_loss
                    else 0.0
                ),
                "relational_unchanged_drift": float(
                    relation_loss["unchanged_drift"]
                    if relation_loss is not None
                    and "unchanged_drift" in relation_loss
                    else 0.0
                ),
                "temporal_loss": float(loss_temporal_contrastive.detach()),
                "relational_loss": float(loss_contrastive.detach())
                if bool(self.otrcl_settings.RELATIONAL_ENABLE)
                else 0.0,
                "coarse_rank_loss": float(loss_coarse_rank.detach()),
                "coarse_bce_loss": float(loss_coarse_bce.detach()),
                "positive_relational_drift_mean": integrity_value(
                    "W+", "relational_drift"
                ),
                "coarse_relational_drift_mean": integrity_value(
                    "Wc", "relational_drift"
                ),
                "fine_temporal_drift_mean": integrity_value(
                    "Wf", "temporal_drift"
                ),
                "fine_graph_drift_mean": integrity_value(
                    "Wf", "graph_drift"
                ),
                "fine_relational_drift_mean": integrity_value(
                    "Wf", "relational_drift"
                ),
                "wc_minus_wf_temporal_drift": directional_value(
                    "wc_minus_wf_temporal_drift"
                ),
                "wf_minus_wc_relational_drift": directional_value(
                    "wf_minus_wc_relational_drift"
                ),
                "fine_valid_count": int(
                    fine_valid_mask.sum().item()
                ),
                "direct_node_count": int(
                    direct_node_mask.bool().sum().item()
                ),
                "unchanged_node_count": int(
                    otrcl_integrity["mask_counts"]["unchanged_nodes"]
                ),
                "method_integrity": otrcl_integrity,
            }
        loss_positive_node_consistency = masked_node_consistency_loss(
            outputs_anchor["direct_node_score_logits"],
            outputs_positive["direct_node_score_logits"],
        )
        if valid_negative_mask.any():
            loss_node = self._node_supervision_loss(
                self._select_output_rows(outputs_anchor, valid_negative_mask),
                self._select_output_rows(outputs_negative, valid_negative_mask),
                direct_node_mask[valid_negative_mask],
                sample_weights=valid_sample_weights,
            )
            loss_direct_node = loss_node
            loss_pairwise_rank = ranking_loss(
                outputs_anchor["window_score"][valid_negative_mask],
                outputs_negative["window_score"][valid_negative_mask],
                self.cfg.SHARK.LOSS.RANK_MARGIN,
                sample_weights=valid_sample_weights,
            )
            tail_rank_enabled = bool(
                getattr(self.cfg.SHARK.LOSS, "TAIL_RANK_ENABLE", False)
            )
            if tail_rank_enabled:
                loss_tail_rank = hard_normal_tail_ranking_loss(
                    outputs_anchor["window_score"],
                    outputs_negative["window_score"][valid_negative_mask],
                    self.cfg.SHARK.LOSS.RANK_MARGIN,
                    self.cfg.SHARK.LOSS.TAIL_RANK_HARD_FRACTION,
                    positive_normal_scores=outputs_positive["window_score"],
                    sample_weights=valid_sample_weights,
                )
                tail_blend = float(self.cfg.SHARK.LOSS.TAIL_RANK_BLEND)
                if not 0.0 <= tail_blend <= 1.0:
                    raise ValueError("TAIL_RANK_BLEND must be in [0, 1]")
                loss_rank = (
                    (1.0 - tail_blend) * loss_pairwise_rank
                    + tail_blend * loss_tail_rank
                )
            else:
                loss_tail_rank = None
                loss_rank = loss_pairwise_rank
            loss_window_bce = window_bce_loss(
                outputs_anchor["window_score"][valid_negative_mask],
                outputs_negative["window_score"][valid_negative_mask],
                outputs_positive["window_score"][valid_negative_mask],
                sample_weights=valid_sample_weights,
            )
            unmodified_node_mask = ~(
                direct_node_mask[valid_negative_mask].bool()
                | propagated_node_mask[valid_negative_mask].bool()
                | consistency_exclusion_mask[valid_negative_mask]
            )
            loss_unmodified_node_consistency = masked_node_consistency_loss(
                outputs_anchor["direct_node_score_logits"][valid_negative_mask],
                outputs_negative["direct_node_score_logits"][valid_negative_mask],
                mask=unmodified_node_mask,
                sample_weights=valid_sample_weights,
            )
        else:
            loss_rank = outputs_anchor["window_score"].sum() * 0.0
            loss_tail_rank = (
                loss_rank
                if bool(getattr(self.cfg.SHARK.LOSS, "TAIL_RANK_ENABLE", False))
                else None
            )
            loss_window_bce = loss_rank
            loss_node = loss_rank
            loss_direct_node = loss_rank
            loss_unmodified_node_consistency = loss_rank
        negative_fidelity = (
            head_fidelity_loss(
                outputs_negative["node_scores_calibrated"][valid_negative_mask],
                outputs_negative["raw_node_scores"][valid_negative_mask],
                self.cfg.SHARK.HEAD.EPS,
            )
            if valid_negative_mask.any()
            else outputs_negative["node_scores_calibrated"].sum() * 0.0
        )
        loss_fidelity = (
            head_fidelity_loss(
                outputs_anchor["node_scores_calibrated"],
                outputs_anchor["raw_node_scores"],
                self.cfg.SHARK.HEAD.EPS,
            )
            + head_fidelity_loss(
                outputs_positive["node_scores_calibrated"],
                outputs_positive["raw_node_scores"],
                self.cfg.SHARK.HEAD.EPS,
            )
            + negative_fidelity
        ) / 3.0
        gradient_audit_losses = {
            "Contrastive": loss_contrastive,
            "Node": loss_node,
            "Rank": loss_rank,
            "WindowBCE": loss_window_bce,
        }
        if temporal_views is not None:
            gradient_audit_losses.update({
                "TemporalContrastive": loss_temporal_contrastive,
                "CoarseRank": loss_coarse_rank,
                "CoarseBCE": loss_coarse_bce,
            })
        if decoupled_loss is not None:
            # 记录实际进入总目标的局部加权分量，便于直接判断它是否压过
            # 窗口主目标或与节点/排序监督持续冲突。
            gradient_audit_losses["GlobalContrastive"] = decoupled_loss[
                "global"
            ]
            gradient_audit_losses["LocalContrastive"] = decoupled_loss[
                "weighted_local"
            ]
        if native_evidence_loss is not None:
            gradient_audit_losses["GlobalContrastive"] = (
                native_evidence_loss["global"]
            )
            gradient_audit_losses["EvidenceContrastive"] = (
                native_evidence_loss["weighted_local"]
            )
        if tail_window_loss is not None:
            gradient_audit_losses["GlobalContrastive"] = tail_window_loss[
                "global"
            ]
            gradient_audit_losses["WindowMechanism"] = tail_window_loss[
                "weighted_window"
            ]
            gradient_audit_losses["NormalTail"] = tail_window_loss[
                "weighted_tail"
            ]
        gradient_audit_ready = bool(valid_negative_mask.any().detach().cpu())
        if decoupled_loss is not None:
            local_required = float(
                decoupled_local_global_settings.LOCAL_LOSS_WEIGHT
            ) > 0.0
            gradient_audit_ready = gradient_audit_ready and (
                not local_required
                or bool(decoupled_loss.get("local_measured", False))
            )
        if native_evidence_loss is not None:
            local_required = float(
                native_evidence_contrast_settings.LOCAL_LOSS_WEIGHT
            ) > 0.0
            gradient_audit_ready = gradient_audit_ready and (
                not local_required
                or bool(native_evidence_loss.get("local_measured", False))
            )
        if tail_window_loss is not None:
            window_required = float(
                tail_window_contrast_settings.WINDOW_LOSS_WEIGHT
            ) > 0.0
            tail_required = float(
                tail_window_contrast_settings.TAIL_LOSS_WEIGHT
            ) > 0.0
            gradient_audit_ready = gradient_audit_ready and (
                not window_required
                or bool(tail_window_loss.get("window_measured", False))
            ) and (
                not tail_required
                or bool(tail_window_loss.get("tail_measured", False))
            )
        self._maybe_audit_weighted_gradients(
            gradient_audit_losses,
            ready=gradient_audit_ready,
        )

        loss_bridge_retention = outputs_anchor.get(
            "bridge_retention_loss",
            outputs_anchor["window_score"].sum() * 0.0,
        )
        retention_weight = float(
            getattr(token_bridge, "RETENTION_WEIGHT", 0.0)
            if token_bridge is not None
            else 0.0
        )
        contrastive_weight = float(self.cfg.SHARK.LOSS.CONTRASTIVE_WEIGHT)
        if self.otrcl_enabled and bool(
            self.otrcl_settings.RELATIONAL_ENABLE
        ):
            contrastive_weight = float(
                self.otrcl_settings.RELATIONAL_LOSS_WEIGHT
            )
        loss = (
            contrastive_weight * loss_contrastive
            + self.cfg.SHARK.LOSS.NODE_WEIGHT * loss_node
            + self.cfg.SHARK.LOSS.RANK_WEIGHT * loss_rank
            + self.cfg.SHARK.LOSS.WINDOW_BCE_WEIGHT * loss_window_bce
            + self.cfg.SHARK.LOSS.FIDELITY_WEIGHT * loss_fidelity
            + self.cfg.SHARK.LOSS.POSITIVE_NODE_CONSISTENCY_WEIGHT
            * loss_positive_node_consistency
            + self.cfg.SHARK.LOSS.UNMODIFIED_NODE_CONSISTENCY_WEIGHT
            * loss_unmodified_node_consistency
            + retention_weight * loss_bridge_retention
        )
        if temporal_views is not None:
            loss = (
                loss
                + float(self.otrcl_settings.TEMPORAL_LOSS_WEIGHT)
                * loss_temporal_contrastive
                + float(self.otrcl_settings.COARSE_RANK_WEIGHT)
                * loss_coarse_rank
                + float(self.otrcl_settings.COARSE_BCE_WEIGHT)
                * loss_coarse_bce
            )
        return self._pack_losses(
            loss,
            loss_contrastive,
            loss_node,
            loss_rank,
            loss_window_bce,
            loss_fidelity,
            loss_direct_node,
            loss_positive_node_consistency,
            loss_unmodified_node_consistency,
            loss_bridge_retention if joint_bridge_training else None,
            loss_tail_rank,
            loss_batch_hard_contrastive,
            loss_node_aware_hard_contrastive,
            loss_temporal_contrastive if temporal_views is not None else None,
            loss_coarse_rank if temporal_views is not None else None,
            loss_coarse_bce if temporal_views is not None else None,
        )

    def _build_counterfactual_node_evidence(self, outputs_anchor, adjacency):
        sources = tuple(getattr(self.cfg.SHARK.SYNTHETIC, "EVIDENCE_SOURCES", ()))
        evidence_parts = []
        for source in sources:
            if source == "direct_node_scores":
                evidence_parts.append(outputs_anchor["direct_node_scores"].detach())
            elif source == "raw_node_scores":
                evidence_parts.append(outputs_anchor["raw_node_scores"].detach())
            elif source == "node_scores":
                evidence_parts.append(outputs_anchor["node_scores"].detach())
            elif source == "relation_strength":
                evidence_parts.append(self._relation_strength(adjacency).detach())
            else:
                raise ValueError(f"Unsupported SHARK.SYNTHETIC.EVIDENCE_SOURCES entry: {source}")

        if not evidence_parts:
            return None
        normalized = [self._normalize_node_evidence(part) for part in evidence_parts]
        return torch.stack(normalized, dim=0).mean(dim=0)

    @staticmethod
    def _select_output_rows(outputs, row_mask):
        selected = {}
        batch_size = row_mask.numel()
        for name, value in outputs.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == batch_size:
                selected[name] = value[row_mask]
            else:
                selected[name] = value
        return selected

    @staticmethod
    def _relation_strength(adjacency):
        adjacency = adjacency.float()
        if adjacency.dim() != 3:
            raise ValueError(f"Expected adjacency [B, N, N], got {tuple(adjacency.shape)}")
        eye = torch.eye(adjacency.size(1), device=adjacency.device, dtype=torch.bool).unsqueeze(0)
        masked = adjacency.masked_fill(eye, 0.0)
        return masked.clamp_min(0.0).sum(dim=-1)

    @staticmethod
    def _normalize_node_evidence(values):
        values = values.float()
        min_value = values.min(dim=1, keepdim=True).values
        max_value = values.max(dim=1, keepdim=True).values
        return (values - min_value) / (max_value - min_value).clamp_min(1e-6)

    def _training_progress(self):
        max_epoch = max(int(self.cfg.SOLVER.MAX_EPOCH) - 1, 1)
        epoch_progress = min(max(float(self.cur_epoch) / max_epoch, 0.0), 1.0)
        return epoch_progress

    @torch.no_grad()
    def _update_counterfactual_model_diagnostics(
        self,
        outputs_anchor,
        outputs_positive,
        outputs_negative,
        direct_node_mask,
        propagated_node_mask,
        anomaly_type_ids,
        valid_negative_mask=None,
        consistency_exclusion_mask=None,
    ):
        if not getattr(self.cfg.SHARK.SYNTHETIC, "PRINT_DIAGNOSTICS", False):
            return
        if not self.model.training:
            return
        diagnostics = getattr(self.augmentor, "last_diagnostics", None)
        if diagnostics is None:
            diagnostics = {}
            self.augmentor.last_diagnostics = diagnostics

        edit_source_ids = getattr(self.augmentor, "last_edit_source_ids", None)
        fallback_type_ids = getattr(self.augmentor, "last_fallback_type_ids", None)
        direct_type_ids = getattr(self.augmentor, "last_direct_type_ids", None)
        acceptance_tier = getattr(self.augmentor, "last_acceptance_tier", None)
        rms_rescue_mask = getattr(self.augmentor, "last_rms_rescue_mask", None)

        if valid_negative_mask is not None:
            valid_negative_mask = valid_negative_mask.bool()
            diagnostics["valid_negative_ratio"] = float(
                valid_negative_mask.float().mean().cpu()
            )
            if not valid_negative_mask.any():
                return
            outputs_anchor = self._select_output_rows(outputs_anchor, valid_negative_mask)
            outputs_positive = self._select_output_rows(outputs_positive, valid_negative_mask)
            outputs_negative = self._select_output_rows(outputs_negative, valid_negative_mask)
            direct_node_mask = direct_node_mask[valid_negative_mask]
            propagated_node_mask = propagated_node_mask[valid_negative_mask]
            if consistency_exclusion_mask is not None:
                consistency_exclusion_mask = consistency_exclusion_mask[
                    valid_negative_mask
                ]
            anomaly_type_ids = anomaly_type_ids[valid_negative_mask]
            if edit_source_ids is not None:
                edit_source_ids = edit_source_ids[valid_negative_mask]
            if fallback_type_ids is not None:
                fallback_type_ids = fallback_type_ids[valid_negative_mask]
            if direct_type_ids is not None:
                direct_type_ids = direct_type_ids[valid_negative_mask]
            if acceptance_tier is not None:
                acceptance_tier = acceptance_tier[valid_negative_mask]
            if rms_rescue_mask is not None:
                rms_rescue_mask = rms_rescue_mask[valid_negative_mask]

        direct_scores_anchor = outputs_anchor["direct_node_scores"].detach()
        direct_scores_positive = outputs_positive["direct_node_scores"].detach()
        direct_scores_negative = outputs_negative["direct_node_scores"].detach()
        direct_mask = direct_node_mask.bool()
        if consistency_exclusion_mask is None:
            consistency_exclusion_mask = propagated_node_mask.bool()
        unmodified_mask = ~(
            direct_mask
            | propagated_node_mask.bool()
            | consistency_exclusion_mask.bool()
        )

        direct_count = direct_mask.sum().clamp_min(1)
        unmodified_count = unmodified_mask.sum().clamp_min(1)
        direct_mean = direct_scores_negative.masked_select(direct_mask).sum() / direct_count
        unmodified_mean = direct_scores_negative.masked_select(unmodified_mask).sum() / unmodified_count
        score_gain = direct_scores_negative - direct_scores_anchor
        direct_gain = score_gain.masked_select(direct_mask).sum() / direct_count
        unmodified_gain = score_gain.masked_select(unmodified_mask).sum() / unmodified_count

        top1 = direct_scores_negative.argmax(dim=1, keepdim=True)
        k = min(max(int(self.cfg.SHARK.HEAD.TOPK), 1), direct_scores_negative.size(1))
        topk = torch.topk(direct_scores_negative, k=k, dim=1).indices
        diagnostics.update({
            "direct_score_gap": float((direct_mean - unmodified_mean).cpu()),
            "direct_gain_gap": float((direct_gain - unmodified_gain).cpu()),
            "direct_hr1": float(direct_mask.gather(1, top1).any(dim=1).float().mean().cpu()),
            "direct_hrk": float(direct_mask.gather(1, topk).any(dim=1).float().mean().cpu()),
            "consistency_exclusion_ratio": float(
                consistency_exclusion_mask.float().mean().cpu()
            ),
            "positive_evidence_drift": float(
                (direct_scores_positive - direct_scores_anchor).abs().mean().cpu()
            ),
            "negative_evidence_change": float(
                (direct_scores_negative - direct_scores_anchor).abs().mean().cpu()
            ),
            "positive_adjacency_drift": float(
                (outputs_positive["adjacency"] - outputs_anchor["adjacency"]).abs().mean().cpu()
            ),
            "negative_adjacency_drift": float(
                (outputs_negative["adjacency"] - outputs_anchor["adjacency"]).abs().mean().cpu()
            ),
        })

        if all(
            "mil_sensor_risk" in outputs
            for outputs in (outputs_anchor, outputs_positive, outputs_negative)
        ):
            mil_anchor = outputs_anchor["mil_sensor_risk"].detach()
            mil_positive = outputs_positive["mil_sensor_risk"].detach()
            mil_negative = outputs_negative["mil_sensor_risk"].detach()
            calibrated_gain = mil_negative - mil_anchor
            direct_calibrated_gain = (
                calibrated_gain.masked_select(direct_mask).sum() / direct_count
            )
            unmodified_node_drift = (
                calibrated_gain.abs().masked_select(unmodified_mask).sum()
                / unmodified_count
            )
            pooling_weights = outputs_negative["mil_pooling_weights"].detach()
            stable_weights = pooling_weights.clamp_min(self.cfg.SHARK.HEAD.EPS)
            pooling_entropy = -(stable_weights * stable_weights.log()).sum(dim=1)

            coupling_errors = []
            for outputs in (outputs_anchor, outputs_positive, outputs_negative):
                recomputed, _ = self.model.head._pool_mil_sensor_risk(
                    outputs["sensor_score"].detach(),
                    pooling="normalized_logsumexp",
                )
                coupling_errors.append(
                    (outputs["window_score"].detach() - recomputed).abs().max()
                )

            reference_diagnostics = self.model.mil_reference_diagnostics()
            diagnostics.update({
                "window_sensor_coupling_error": float(
                    torch.stack(coupling_errors).max().cpu()
                ),
                "mil_pooling_entropy": float(pooling_entropy.mean().cpu()),
                "mil_effective_node_count": float(pooling_entropy.exp().mean().cpu()),
                "direct_node_calibrated_gain": float(direct_calibrated_gain.cpu()),
                "unmodified_node_drift": float(unmodified_node_drift.cpu()),
                "positive_node_drift": float(
                    (mil_positive - mil_anchor).abs().mean().cpu()
                ),
                "normal_reference_scale_min": reference_diagnostics.get(
                    "mil_reference_scale_min", float("nan")
                ),
                "normal_reference_scale_mean": reference_diagnostics.get(
                    "mil_reference_scale_mean", float("nan")
                ),
                "normal_reference_scale_max": reference_diagnostics.get(
                    "mil_reference_scale_max", float("nan")
                ),
            })

        source_model_diagnostics = {}

        def add_source_diagnostics(name, sample_mask):
            if sample_mask is None or not sample_mask.any():
                return
            source_scores = direct_scores_negative[sample_mask]
            source_anchor = direct_scores_anchor[sample_mask]
            source_direct = direct_mask[sample_mask]
            source_unmodified = unmodified_mask[sample_mask]
            source_direct_count = source_direct.sum().clamp_min(1)
            source_unmodified_count = source_unmodified.sum().clamp_min(1)
            source_direct_mean = (
                source_scores.masked_select(source_direct).sum() / source_direct_count
            )
            source_unmodified_mean = (
                source_scores.masked_select(source_unmodified).sum()
                / source_unmodified_count
            )
            source_gain = source_scores - source_anchor
            source_direct_gain = (
                source_gain.masked_select(source_direct).sum() / source_direct_count
            )
            source_unmodified_gain = (
                source_gain.masked_select(source_unmodified).sum()
                / source_unmodified_count
            )
            source_top1 = source_scores.argmax(dim=1, keepdim=True)
            source_topk = torch.topk(source_scores, k=k, dim=1).indices
            source_model_diagnostics[name] = {
                "count": int(sample_mask.sum().item()),
                "direct_score_gap": float(
                    (source_direct_mean - source_unmodified_mean).cpu()
                ),
                "direct_gain_gap": float(
                    (source_direct_gain - source_unmodified_gain).cpu()
                ),
                "direct_hr1": float(
                    source_direct.gather(1, source_top1)
                    .any(dim=1)
                    .float()
                    .mean()
                    .cpu()
                ),
                "direct_hrk": float(
                    source_direct.gather(1, source_topk)
                    .any(dim=1)
                    .float()
                    .mean()
                    .cpu()
                ),
            }

        if edit_source_ids is not None:
            add_source_diagnostics(
                "surrogate",
                edit_source_ids == self.augmentor.EDIT_SOURCE_SURROGATE,
            )
            fallback_mask = edit_source_ids == self.augmentor.EDIT_SOURCE_FALLBACK
            add_source_diagnostics("fallback", fallback_mask)
            if fallback_type_ids is not None:
                for type_idx, type_name in enumerate(self.augmentor.hybrid_fallback_types):
                    add_source_diagnostics(
                        f"fallback_{type_name}",
                        fallback_mask & (fallback_type_ids == type_idx),
                    )
            direct_mask_source = edit_source_ids == self.augmentor.EDIT_SOURCE_DIRECT
            add_source_diagnostics("direct", direct_mask_source)
            if direct_type_ids is not None:
                for type_idx, type_name in enumerate(self.augmentor.synthetic_types):
                    add_source_diagnostics(
                        f"direct_{type_name}",
                        direct_mask_source & (direct_type_ids == type_idx),
                    )
        if acceptance_tier is not None:
            hard_mask = acceptance_tier == self.augmentor.ACCEPTANCE_HARD
            soft_mask = acceptance_tier == self.augmentor.ACCEPTANCE_SOFT
            add_source_diagnostics(
                "acceptance_hard",
                hard_mask,
            )
            add_source_diagnostics(
                "acceptance_soft",
                soft_mask,
            )
            if rms_rescue_mask is not None:
                add_source_diagnostics("rms_rescue", rms_rescue_mask)
                add_source_diagnostics(
                    "within_bound_soft",
                    soft_mask & ~rms_rescue_mask,
                )
        diagnostics["source_model_diagnostics"] = source_model_diagnostics

        type_direct_gap = []
        type_direct_hr1 = []
        type_direct_hrk = []
        for type_idx in range(len(self.augmentor.synthetic_types)):
            sample_mask = anomaly_type_ids == type_idx
            if not sample_mask.any():
                type_direct_gap.append(float("nan"))
                type_direct_hr1.append(float("nan"))
                type_direct_hrk.append(float("nan"))
                continue
            sample_scores = direct_scores_negative[sample_mask]
            sample_direct = direct_mask[sample_mask]
            sample_unmodified = unmodified_mask[sample_mask]
            sample_direct_mean = sample_scores.masked_select(sample_direct).mean()
            if sample_unmodified.any():
                sample_unmodified_mean = sample_scores.masked_select(sample_unmodified).mean()
            else:
                sample_unmodified_mean = sample_direct_mean
            sample_top1 = sample_scores.argmax(dim=1, keepdim=True)
            sample_topk = torch.topk(sample_scores, k=k, dim=1).indices
            type_direct_gap.append(float((sample_direct_mean - sample_unmodified_mean).cpu()))
            type_direct_hr1.append(
                float(sample_direct.gather(1, sample_top1).any(dim=1).float().mean().cpu())
            )
            type_direct_hrk.append(
                float(sample_direct.gather(1, sample_topk).any(dim=1).float().mean().cpu())
            )
        diagnostics["type_direct_gap"] = type_direct_gap
        diagnostics["type_direct_hr1"] = type_direct_hr1
        diagnostics["type_direct_hrk"] = type_direct_hrk

    def _maybe_print_counterfactual_diagnostics(self):
        if not getattr(self.cfg.SHARK.SYNTHETIC, "PRINT_DIAGNOSTICS", False):
            return
        if not self.model.training:
            return
        if self.cur_iter not in (0, 1):
            return
        diagnostics = getattr(self.augmentor, "last_diagnostics", {})
        if not diagnostics:
            return
        type_ratio = diagnostics.get("type_ratio", [])
        type_delta_rms = diagnostics.get("type_delta_rms", [])
        type_direct_gap = diagnostics.get("type_direct_gap", [])
        type_direct_hr1 = diagnostics.get("type_direct_hr1", [])
        type_direct_hrk = diagnostics.get("type_direct_hrk", [])
        type_boundary_score = diagnostics.get("type_boundary_score", [])
        type_boundary_acceptance = diagnostics.get("type_boundary_acceptance", [])
        type_valid_ratio = diagnostics.get("type_valid_ratio", [])
        type_hard_ratio = diagnostics.get("type_hard_ratio", [])
        type_soft_ratio = diagnostics.get("type_soft_ratio", [])
        type_invalid_ratio = diagnostics.get("type_invalid_ratio", [])
        type_sample_weight = diagnostics.get("type_sample_weight", [])
        type_direct_score = diagnostics.get("type_direct_score", [])
        type_direct_target = diagnostics.get("type_direct_target", [])
        type_normalized_gain = diagnostics.get("type_normalized_gain", [])
        type_parts = []
        for type_idx, name in enumerate(self.augmentor.synthetic_types):
            ratio = type_ratio[type_idx] if type_idx < len(type_ratio) else float("nan")
            delta = type_delta_rms[type_idx] if type_idx < len(type_delta_rms) else float("nan")
            gap = type_direct_gap[type_idx] if type_idx < len(type_direct_gap) else float("nan")
            hr1 = type_direct_hr1[type_idx] if type_idx < len(type_direct_hr1) else float("nan")
            hrk = type_direct_hrk[type_idx] if type_idx < len(type_direct_hrk) else float("nan")
            part = (
                f"{name}:ratio={ratio:.2f}/delta={delta:.3f}/"
                f"gap={gap:.3f}/hr1={hr1:.2f}/hrk={hrk:.2f}"
            )
            if type_idx < len(type_boundary_score):
                acceptance = (
                    type_boundary_acceptance[type_idx]
                    if type_idx < len(type_boundary_acceptance)
                    else float("nan")
                )
                part += f"/boundary={type_boundary_score[type_idx]:.3f}/accept={acceptance:.2f}"
            if type_idx < len(type_valid_ratio):
                score = (
                    type_direct_score[type_idx]
                    if type_idx < len(type_direct_score)
                    else float("nan")
                )
                target = (
                    type_direct_target[type_idx]
                    if type_idx < len(type_direct_target)
                    else float("nan")
                )
                gain = (
                    type_normalized_gain[type_idx]
                    if type_idx < len(type_normalized_gain)
                    else float("nan")
                )
                part += (
                    f"/valid={type_valid_ratio[type_idx]:.2f}/"
                    f"hard={type_hard_ratio[type_idx] if type_idx < len(type_hard_ratio) else float('nan'):.2f}/"
                    f"soft={type_soft_ratio[type_idx] if type_idx < len(type_soft_ratio) else float('nan'):.2f}/"
                    f"invalid={type_invalid_ratio[type_idx] if type_idx < len(type_invalid_ratio) else float('nan'):.2f}/"
                    f"weight={type_sample_weight[type_idx] if type_idx < len(type_sample_weight) else float('nan'):.2f}/"
                    f"score={score:.3f}/target={target:.3f}/gain={gain:.3f}"
                )
            type_parts.append(part)
        type_text = ", ".join(type_parts)
        boundary_text = ""
        if "boundary_quantile" in diagnostics:
            boundary_text = (
                f"boundary_q={diagnostics.get('boundary_quantile', float('nan')):.3f} "
                f"boundary_target={diagnostics.get('boundary_target', float('nan')):.3f} "
                f"boundary_original={diagnostics.get('boundary_original_score', float('nan')):.3f} "
                f"boundary_accept={diagnostics.get('boundary_acceptance_ratio', 0.0):.2f} "
                f"fitted_window={diagnostics.get('fitted_window_ratio', 0.0):.2f} "
                f"surrogate_ratio={diagnostics.get('surrogate_accepted_ratio', 0.0):.2f} "
                f"hybrid_fallback={diagnostics.get('hybrid_fallback_ratio', 0.0):.2f} "
                f"fallback_validity={diagnostics.get('hybrid_fallback_validity_ratio', 0.0):.2f} "
                f"fallback_mechanism_gain={diagnostics.get('hybrid_fallback_mechanism_gain', float('nan')):.3f} "
                f"invalid={diagnostics.get('invalid_ratio', 0.0):.2f} "
                f"seed_cov={diagnostics.get('seed_coverage', 0.0):.2f} "
                f"seed_entropy={diagnostics.get('seed_entropy', 0.0):.2f} "
                f"surrogate_r2={diagnostics.get('surrogate_r2', float('nan')):.3f} "
                f"surrogate_delta_r2={diagnostics.get('surrogate_delta_r2', float('nan')):.3f} "
                f"candidate_fit={diagnostics.get('candidate_fit_rate', 0.0):.2f} "
                f"fit_ratio={diagnostics.get('surrogate_fit_ratio', 0.0):.2f} "
                f"fallback_ratio={diagnostics.get('fallback_ratio', 0.0):.2f} "
                f"marginal_delta={diagnostics.get('marginal_delta_rms', float('nan')):.3f} "
            )
        elif "direct_target_quantile" in diagnostics:
            boundary_text = (
                f"direct_q={diagnostics.get('direct_target_quantile', float('nan')):.3f} "
                f"acceptance={diagnostics.get('direct_acceptance_mode', 'hard')} "
                f"reference_scale={diagnostics.get('direct_reference_scale_mode', 'iqr')} "
                f"upper_bound={diagnostics.get('direct_upper_bound_mode', 'mechanism_cap')} "
                f"direct_validity={diagnostics.get('direct_validity_ratio', 0.0):.2f} "
                f"hard_coverage={diagnostics.get('hard_coverage', 0.0):.2f} "
                f"soft_coverage={diagnostics.get('soft_coverage', 0.0):.2f} "
                f"rms_rescue={diagnostics.get('rms_rescue_coverage', 0.0):.2f} "
                f"rescue_rms={diagnostics.get('rms_rescue_delta_rms_mean', 0.0):.3f} "
                f"mean_weight={diagnostics.get('accepted_sample_weight_mean', 0.0):.3f} "
                f"effective_weight={diagnostics.get('effective_weighted_count', 0.0):.2f} "
                f"candidate_finite={diagnostics.get('candidate_finite_ratio', 0.0):.2f} "
                f"below_floor={diagnostics.get('candidate_below_gain_floor_count', 0)} "
                f"above_cap={diagnostics.get('candidate_above_cap_count', 0)} "
                f"hard_eligible={diagnostics.get('candidate_hard_eligible_count', 0)} "
                f"soft_eligible={diagnostics.get('candidate_soft_eligible_count', 0)} "
                f"rescue_eligible={diagnostics.get('candidate_rms_rescue_count', 0)} "
                f"scale_fallback={diagnostics.get('reference_scale_fallback_ratio', 0.0):.2f} "
                f"floor_hit={diagnostics.get('reference_floor_hit_ratio', 0.0):.2f} "
                f"invalid={diagnostics.get('invalid_ratio', 0.0):.2f} "
                f"seed_cov={diagnostics.get('seed_coverage', 0.0):.2f} "
                f"seed_entropy={diagnostics.get('seed_entropy', 0.0):.2f} "
                f"marginal_delta={diagnostics.get('marginal_delta_rms', float('nan')):.3f} "
            )
        print(
            "[SHARK synthetic] "
            f"progress={self._training_progress():.3f} "
            f"mode={diagnostics.get('edit_mode', 'legacy')} "
            f"strength={diagnostics.get('edit_strength', float('nan')):.3f} "
            f"magnitude={diagnostics.get('magnitude', float('nan')):.3f} "
            f"seed_evidence={diagnostics.get('seed_evidence_mean', float('nan')):.3f} "
            f"evidence_ratio={diagnostics.get('evidence_sampling_ratio', 0.0):.2f} "
            f"random_ratio={diagnostics.get('random_sampling_ratio', 0.0):.2f} "
            f"valid_ratio={diagnostics.get('valid_negative_ratio', diagnostics.get('boundary_acceptance_ratio', 1.0)):.2f} "
            f"direct_gap={diagnostics.get('direct_score_gap', float('nan')):.3f} "
            f"direct_gain_gap={diagnostics.get('direct_gain_gap', float('nan')):.3f} "
            f"direct_hr1={diagnostics.get('direct_hr1', float('nan')):.2f} "
            f"direct_hrk={diagnostics.get('direct_hrk', float('nan')):.2f} "
            f"consistency_exclusion={diagnostics.get('consistency_exclusion_ratio', 0.0):.3f} "
            f"positive_drift={diagnostics.get('positive_evidence_drift', float('nan')):.3f} "
            f"negative_change={diagnostics.get('negative_evidence_change', float('nan')):.3f} "
            f"adj_pos={diagnostics.get('positive_adjacency_drift', float('nan')):.3f} "
            f"adj_neg={diagnostics.get('negative_adjacency_drift', float('nan')):.3f} "
            f"coupling_error={diagnostics.get('window_sensor_coupling_error', float('nan')):.3e} "
            f"mil_entropy={diagnostics.get('mil_pooling_entropy', float('nan')):.3f} "
            f"mil_effective_nodes={diagnostics.get('mil_effective_node_count', float('nan')):.3f} "
            f"direct_cal_gain={diagnostics.get('direct_node_calibrated_gain', float('nan')):.3f} "
            f"unmodified_drift={diagnostics.get('unmodified_node_drift', float('nan')):.3f} "
            f"positive_node_drift={diagnostics.get('positive_node_drift', float('nan')):.3f} "
            f"{boundary_text}"
            f"types=({type_text})"
        )

    @staticmethod
    def _pack_losses(
        total,
        contrastive,
        node,
        rank,
        window_bce,
        fidelity,
        direct_node,
        positive_node_consistency,
        unmodified_node_consistency,
        bridge_retention=None,
        tail_rank=None,
        batch_hard_contrastive=None,
        node_aware_hard_contrastive=None,
        temporal_contrastive=None,
        coarse_rank=None,
        coarse_bce=None,
    ):
        losses = (
            total,
            contrastive,
            node,
            rank,
            window_bce,
            fidelity,
            direct_node,
            positive_node_consistency,
            unmodified_node_consistency,
        )
        if bridge_retention is not None:
            losses = losses + (bridge_retention,)
        if tail_rank is not None:
            losses = losses + (tail_rank,)
        if batch_hard_contrastive is not None:
            losses = losses + (batch_hard_contrastive,)
        if node_aware_hard_contrastive is not None:
            losses = losses + (node_aware_hard_contrastive,)
        if temporal_contrastive is not None:
            losses = losses + (temporal_contrastive,)
        if coarse_rank is not None:
            losses = losses + (coarse_rank,)
        if coarse_bce is not None:
            losses = losses + (coarse_bce,)
        components = {
            "contrastive": contrastive,
            "node": node,
            "rank": rank,
            "window_bce": window_bce,
            "fidelity": fidelity,
            "positive_node_consistency": positive_node_consistency,
            "unmodified_node_consistency": unmodified_node_consistency,
        }
        if temporal_contrastive is not None:
            components["temporal_contrastive"] = temporal_contrastive
        if coarse_rank is not None:
            components["coarse_rank"] = coarse_rank
        if coarse_bce is not None:
            components["coarse_bce"] = coarse_bce
        return {
            "total": total,
            "losses": losses,
            # 为实验级梯度路由提供具名损失，不改变默认联合反向路径。
            "components": components,
        }

    def _node_supervision_loss(
        self,
        outputs_anchor,
        outputs_negative,
        direct_node_mask,
        sample_weights=None,
    ):
        if self.cfg.SHARK.LOSS.NODE_LOSS_TYPE != "intervention_gain_rank":
            raise ValueError("SHARK v8.15 requires intervention_gain_rank node supervision")
        return intervention_gain_node_loss(
            outputs_anchor["direct_node_score_logits"],
            outputs_negative["direct_node_score_logits"],
            direct_node_mask,
            self.cfg.SHARK.LOSS.NODE_GAIN_MARGIN,
            detach_anchor=self.cfg.SHARK.LOSS.NODE_GAIN_DETACH_ANCHOR,
            sample_weights=sample_weights,
        )

    def train_step(self, inputs):
        windows, _ = prepare_inputs(inputs)
        loss_outputs = self._forward_loss(windows)
        loss = loss_outputs["total"]

        self.optimizer.zero_grad()
        loss.backward()
        if self.otrcl_enabled:
            self.last_otrcl_gradient_diagnostics = (
                gradient_clip_diagnostics(
                    self.model.named_parameters(),
                    clip_enabled=bool(self.cfg.SOLVER.GRADIENT_CLIP),
                    clip_threshold=float(
                        self.cfg.SOLVER.GRADIENT_CLIP_NORM
                    ),
                    temporal_representation_enters_inference=bool(
                        self.otrcl_settings.PROGRESSIVE_GRAPH_FUSION
                    ),
                )
            )
            self.last_otrcl_diagnostics["gradient_clipping"] = dict(
                self.last_otrcl_gradient_diagnostics
            )
        if self.cfg.SOLVER.GRADIENT_CLIP:
            torch.nn.utils.clip_grad_norm_(
                optim.get_trainable_params(self.model),
                self.cfg.SOLVER.GRADIENT_CLIP_NORM,
            )
        self.optimizer.step()

        return {
            "metrics": (loss.detach(),),
            "losses": loss_outputs["losses"],
        }

    @torch.no_grad()
    def eval_step(self, inputs):
        assert not self.model.training
        windows, _ = prepare_inputs(inputs)
        loss_outputs = self._forward_loss(windows)
        loss = loss_outputs["total"]

        return {
            "metrics": (loss.detach(),),
            "losses": loss_outputs["losses"],
        }
