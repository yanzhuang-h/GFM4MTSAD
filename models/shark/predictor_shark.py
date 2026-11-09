import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from threshold import Thresholder
from datasets.loader import (
    get_test_dataloader,
    get_train_dataloader,
    get_val_dataloader,
)
from trainer import prepare_inputs
from utils.misc import mkdir
from utils.oraclead_metrics import (
    display_metrics,
    evaluate_oraclead_metrics,
    evaluate_oraclead_paper_metrics,
)
from evaluation.comprehensive_metrics import (
    pgrf_metrics,
)
from models.shark.augment import SharkAugmentor


class SharkPredictor:
    def __init__(self, cfg, model):
        self.cfg = cfg.clone()

        self.model = model

        self.threshold = None
        self.result_dir = os.path.join(cfg.SHARK.EVAL.RESULT_DIR, cfg.DATA.NAME)

        self.test_loader = get_test_dataloader(cfg)
        self.augmentor = SharkAugmentor(cfg)
        if self.augmentor.requires_normal_mechanism:
            fitted_mechanism = getattr(model, "_shark_normal_mechanism", None)
            if fitted_mechanism is not None and fitted_mechanism.fitted:
                self.augmentor.normal_mechanism = fitted_mechanism
                diagnostics = fitted_mechanism.diagnostics()
            else:
                diagnostics = self.augmentor.fit_normal_mechanism(
                    self.test_loader.dataset.train
                )
            print(
                "[SHARK predictor mechanism] "
                f"valid_nodes={diagnostics.get('valid_node_count', 0)}/"
                f"{diagnostics.get('n_nodes', 0)} "
                f"valid_ratio={diagnostics.get('valid_node_ratio', 0.0):.4f}",
                flush=True,
            )
        if self.augmentor.requires_frontend_graph_selector:
            selector = self.augmentor.fit_frontend_graph_selector(
                self.model.imlp,
                self.model.graph_builder,
                get_train_dataloader(cfg),
            )
            print(
                "[SHARK predictor v2.2 selector] "
                f"fitted={selector.get('fitted', False)} "
                f"windows={selector.get('fit_windows', 0)} "
                f"valid_nodes={selector.get('valid_node_count', 0)}",
                flush=True,
            )

    @torch.no_grad()
    def predict(self):
        self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        test_start = time.perf_counter()
        self.test_scores, self.test_labels, self.test_node_scores = self._get_test_outputs()
        self.test_inference_seconds = time.perf_counter() - test_start
        self.test_peak_gpu_memory_bytes = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        )
        reference_start = time.perf_counter()
        self.train_scores, self.train_labels = self._get_reference_outputs(
            get_train_dataloader(self.cfg), "shark train scores"
        )
        self.validation_scores, self.validation_labels = self._get_reference_outputs(
            get_val_dataloader(self.cfg), "shark validation scores"
        )
        self.reference_score_seconds = time.perf_counter() - reference_start

        assert len(self.test_scores) == len(self.test_labels)

        self.threshold = self._get_threshold(self.test_scores, self.test_labels)
        pred = (self.test_scores > self.threshold).astype(int)

        if self.cfg.TEST.POINT_ADJUST:
            pred_pa = self.point_adjust(pred.copy(), self.test_labels)
        else:
            pred_pa = None

        results = self.get_results(self.test_scores, pred_pa, self.test_labels)
        try:
            self.oraclead_metric_report = evaluate_oraclead_metrics(
                self.test_labels,
                self.test_scores,
            )
            results.update(display_metrics(self.oraclead_metric_report))
        except (ArithmeticError, ImportError, IndexError, KeyError, TypeError, ValueError, RuntimeError) as error:
            # 评价扩展不应改变模型分数；缺少可选官方依赖时明确写出 NA。
            self.oraclead_metric_report = {
                "schema_version": "SHARK-ORACLEAD-METRICS-V1",
                "available": False,
                "error": f"{type(error).__name__}: {error}",
            }
            results.update({
                "AP": "NA",
                "R-F1": "NA",
                "Aff-F1": "NA",
                "V-ROC": "NA",
                "V-PR": "NA",
            })
        try:
            self.oraclead_paper_metric_report = evaluate_oraclead_paper_metrics(
                self.test_labels,
                self.test_scores,
            )
            oracle_paper = self.oraclead_paper_metric_report
            standard = oracle_paper["standard_f1"]
            range_report = oracle_paper["range_f1"]
            affiliation = oracle_paper["affiliation_f1"]
            vus = oracle_paper["vus"]
            results.update({
                "OracleAD200_P": standard.get("precision", "NA"),
                "OracleAD200_R": standard.get("recall", "NA"),
                "OracleAD200_F1": standard.get("f1", "NA"),
                "OracleAD200_R-P": range_report.get("precision", "NA"),
                "OracleAD200_R-R": range_report.get("recall", "NA"),
                "OracleAD200_R-F1": range_report.get("f1", "NA"),
                "OracleAD200_Aff-P": affiliation.get("precision", "NA"),
                "OracleAD200_Aff-R": affiliation.get("recall", "NA"),
                "OracleAD200_Aff-F1": affiliation.get("f1", "NA"),
                "OracleAD200_A-ROC": oracle_paper.get("a_roc", "NA"),
                "OracleAD200_A-PR": oracle_paper.get("a_pr", "NA"),
                "OracleAD200_V-ROC": vus.get("v_roc", "NA"),
                "OracleAD200_V-PR": vus.get("v_pr", "NA"),
            })
        except (ArithmeticError, ImportError, IndexError, KeyError, TypeError, ValueError, RuntimeError) as error:
            self.oraclead_paper_metric_report = {
                "schema_version": "SHARK-ORACLEAD-PAPER-200-V1",
                "available": False,
                "error": f"{type(error).__name__}: {error}",
            }
        try:
            self.pgrf_metric_report = pgrf_metrics(
                self.test_labels,
                self.test_scores,
            )
            pgrf = self.pgrf_metric_report["appendix_oracle_1000"]
            pa = pgrf["point_adjusted"]
            raw = pgrf["raw_point"]
            event = pgrf["range_event"]
            results.update({
                "PGRF_PA_P": pa.get("precision", "NA"),
                "PGRF_PA_R": pa.get("recall", "NA"),
                "PGRF_PA_F1": pa.get("f1", "NA"),
                "PGRF_Raw_P": raw.get("precision", "NA"),
                "PGRF_Raw_R": raw.get("recall", "NA"),
                "PGRF_Raw_F1": raw.get("f1", "NA"),
                "PGRF_R-P": event.get("r_precision", "NA"),
                "PGRF_R-R": event.get("r_recall", "NA"),
                "PGRF_R-F1": event.get("r_f1", "NA"),
            })
        except (ArithmeticError, ImportError, IndexError, KeyError, TypeError, ValueError, RuntimeError) as error:
            self.pgrf_metric_report = {
                "schema_version": "SHARK-PGRF-METRICS-V2",
                "available": False,
                "error": f"{type(error).__name__}: {error}",
            }
        explanation_results = self.get_explanation_results(pred_pa if pred_pa is not None else pred)
        results.update(explanation_results)
        results.update(self.get_msl_target_channel_results())
        controlled_results = self.get_controlled_directprop_results()
        results.update(controlled_results)
        results.update(self.get_normal_mechanism_results())
        results.update(self.get_evidence_alignment_results())
        sweep_results = self.get_score_source_sweep()
        peft_diagnostics = self.get_peft_diagnostics()

        self.save_results(results, sweep_results, peft_diagnostics)
        self.save_metric_protocol(results)
        self.save_to_npy(
            train_scores=self.train_scores,
            train_labels=self.train_labels,
            validation_scores=self.validation_scores,
            validation_labels=self.validation_labels,
            test_scores=self.test_scores,
            test_labels=self.test_labels,
            test_window_scores=self.test_window_scores,
            test_node_scores=self.test_node_scores,
            test_direct_node_scores=self.test_direct_node_scores,
            test_root_node_scores=self.test_root_node_scores,
            test_affected_node_scores=self.test_affected_node_scores,
            test_node_scores_raw=self.test_node_scores_raw,
            test_sensor_score=self.test_sensor_score,
            test_sensor_importance=self.test_sensor_importance,
            test_mil_sensor_risk=self.test_mil_sensor_risk,
            test_mil_pooling_weights=self.test_mil_pooling_weights,
            test_mil_direct_raw=self.test_mil_direct_raw,
            test_mil_reference_mean=self.test_mil_reference_mean,
            test_mil_reference_scale=self.test_mil_reference_scale,
            test_head_residual=self.test_head_residual,
            test_fusion_weights=self.test_fusion_weights,
            test_expert_scores=self.test_expert_scores,
            test_gate_weights=self.test_gate_weights,
            test_gate_logits=self.test_gate_logits,
            test_conservative_scores=self.test_conservative_scores,
            test_conservative_scores_raw=self.test_conservative_scores_raw,
            test_conservative_weights=self.test_conservative_weights,
            test_conservative_logits=self.test_conservative_logits,
            test_reliable_source_weights=self.test_reliable_source_weights,
            test_reliable_source_names=self.test_reliable_source_names,
            test_evidence_protocol_scores=self.test_evidence_protocol_scores,
            test_evidence_protocol_scores_raw=self.test_evidence_protocol_scores_raw,
            test_evidence_protocol_weights=self.test_evidence_protocol_weights,
            test_evidence_protocol_names=self.test_evidence_protocol_names,
            test_source_reliability_prior=self.test_source_reliability_prior,
            test_source_reliability_names=self.test_source_reliability_names,
            test_sensor_importance_direct=self.test_sensor_importance_components.get("direct"),
            test_sensor_importance_local=self.test_sensor_importance_components.get("local"),
            test_sensor_importance_relation=self.test_sensor_importance_components.get("relation"),
            test_sensor_importance_context=self.test_sensor_importance_components.get("context"),
            test_sensor_importance_spike=self.test_sensor_importance_components.get("spike"),
            test_sensor_importance_distribution=self.test_sensor_importance_components.get("distribution"),
            test_sensor_importance_graph_context=self.test_sensor_importance_components.get("graph_context"),
        )

    def get_normal_mechanism_results(self):
        diagnostics = self.augmentor.normal_mechanism_diagnostics()
        if not diagnostics:
            return {}
        return {
            "NormalMechanism_Fitted": bool(diagnostics.get("fitted", False)),
            "NormalMechanism_ValidNodeCount": diagnostics.get("valid_node_count", 0),
            "NormalMechanism_ValidNodeRatio": diagnostics.get("valid_node_ratio", 0.0),
            "NormalMechanism_ContinuousNodeCount": diagnostics.get("continuous_node_count", 0),
            "NormalMechanism_DiscreteNodeCount": diagnostics.get("discrete_node_count", 0),
            "NormalMechanism_ConstantNodeCount": diagnostics.get("constant_node_count", 0),
            "NormalMechanism_NodesWithParentCount": diagnostics.get("nodes_with_parent_count", 0),
            "NormalMechanism_ParentCountMean": diagnostics.get("parent_count_mean", 0.0),
            "NormalMechanism_ResidualScaleMin": diagnostics.get("residual_scale_min", float("nan")),
            "NormalMechanism_ResidualScaleMean": diagnostics.get("residual_scale_mean", float("nan")),
            "NormalMechanism_ResidualScaleMax": diagnostics.get("residual_scale_max", float("nan")),
        }

    def get_peft_diagnostics(self):
        if not hasattr(self.model, "peft_diagnostics"):
            return {}
        diagnostics = self.model.peft_diagnostics()
        if hasattr(self.model, "mil_reference_diagnostics"):
            diagnostics.update(self.model.mil_reference_diagnostics())
        return {
            key: value
            for key, value in diagnostics.items()
            if isinstance(value, (str, int, float, bool, list, tuple))
        }

    def get_evidence_alignment_results(self):
        sources = getattr(self, "test_score_sources", {})
        diagnostics = {}
        pairs = (
            ("direct_topk", "raw_topk"),
            ("direct_topk", "cal_topk"),
            ("direct_topk", "shark_head"),
            ("raw_topk", "shark_head"),
            ("cal_topk", "shark_head"),
        )
        for left, right in pairs:
            if left not in sources or right not in sources:
                continue
            diagnostics[f"EvidenceCorr_{left}_vs_{right}"] = self._safe_corr(sources[left], sources[right])
            diagnostics[f"EvidenceGap_{left}_minus_{right}"] = float(np.mean(sources[left] - sources[right]))
        if "mil_window_score" in sources and "shark_head" in sources:
            diagnostics["MILWindowSensorCouplingError"] = float(
                np.max(np.abs(sources["mil_window_score"] - sources["shark_head"]))
            )
        pooling_weights = getattr(self, "test_mil_pooling_weights", None)
        if pooling_weights is not None and pooling_weights.size > 0:
            stable_weights = np.clip(pooling_weights, self.cfg.SHARK.HEAD.EPS, 1.0)
            entropy = -(stable_weights * np.log(stable_weights)).sum(axis=1)
            diagnostics["MILPoolingEntropy"] = float(entropy.mean())
            diagnostics["MILEffectiveNodeCount"] = float(np.exp(entropy).mean())
        return diagnostics

    @staticmethod
    def _safe_corr(left, right):
        left = np.asarray(left).reshape(-1)
        right = np.asarray(right).reshape(-1)
        if left.size == 0 or right.size == 0 or left.size != right.size:
            return float("nan")
        left_std = float(np.std(left))
        right_std = float(np.std(right))
        if left_std <= 1e-12 or right_std <= 1e-12:
            return float("nan")
        return float(np.corrcoef(left, right)[0, 1])

    @torch.no_grad()
    def _get_reference_outputs(self, loader, description, score_source=None):
        loader = torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            drop_last=False,
            collate_fn=loader.collate_fn,
            worker_init_fn=getattr(loader, "worker_init_fn", None),
            generator=getattr(loader, "generator", None),
        )
        score_source = self._normalize_score_source(
            score_source or self.cfg.SHARK.EVAL.SCORE_SOURCE
        )
        scores_all = []
        labels_all = []
        self.model.eval()
        for inputs in tqdm(loader, desc=description):
            windows, labels = prepare_inputs(inputs)
            outputs = self.model(windows)
            window_scores = outputs["window_score"].detach().flatten().cpu().numpy()
            fallback_node_scores = outputs["node_scores"]
            raw_node_scores = outputs.get(
                "raw_node_scores",
                outputs.get("node_scores_raw", fallback_node_scores),
            ).detach().cpu().numpy()
            calibrated_node_scores = outputs.get(
                "node_scores_calibrated", fallback_node_scores
            ).detach().cpu().numpy()
            sources = self._build_score_sources(
                window_scores, raw_node_scores, calibrated_node_scores
            )
            if "direct_node_scores" in outputs or "root_node_scores" in outputs:
                direct = outputs.get("direct_node_scores")
                if direct is None:
                    direct = outputs["root_node_scores"]
                direct = direct.detach().cpu().numpy()
                sources.update(self._node_score_sources("direct", direct))
                sources.update(self._node_score_sources("root", direct))
            if "affected_node_scores" in outputs:
                affected = outputs["affected_node_scores"].detach().cpu().numpy()
                sources.update(self._node_score_sources("affected", affected))
            for key, value in outputs.get("score_breakdown", {}).items():
                if torch.is_tensor(value):
                    sources[key] = value.detach().flatten().cpu().numpy()
            if score_source not in sources:
                raise ValueError(
                    f"Unsupported SHARK reference score source: {score_source}. "
                    f"Available: {', '.join(sources.keys())}"
                )
            scores_all.append(sources[score_source])
            labels_all.append(labels.detach().flatten().cpu().numpy())
        scores = np.concatenate(scores_all, axis=0)
        labels = np.concatenate(labels_all, axis=0)
        if scores.shape[0] != labels.shape[0]:
            raise ValueError("SHARK reference scores and labels have different lengths")
        return scores, labels

    @torch.no_grad()
    def _get_test_outputs(self):
        window_scores_all = []
        labels_all = []
        node_scores_all = []
        direct_node_scores_all = []
        root_node_scores_all = []
        affected_node_scores_all = []
        node_scores_raw_all = []
        sensor_importance_all = []
        head_residual_all = []
        expert_scores_all = []
        gate_weights_all = []
        gate_logits_all = []
        conservative_scores_all = []
        conservative_scores_raw_all = []
        conservative_weights_all = []
        conservative_logits_all = []
        reliable_source_weights_all = []
        reliable_source_names = None
        evidence_protocol_scores_all = []
        evidence_protocol_scores_raw_all = []
        evidence_protocol_weights_all = []
        evidence_protocol_names = None
        source_reliability_prior = None
        source_reliability_names = None
        sensor_score_all = []
        mil_sensor_risk_all = []
        mil_pooling_weights_all = []
        mil_direct_raw_all = []
        sensor_importance_component_all = {
            "direct": [],
            "local": [],
            "relation": [],
            "context": [],
            "spike": [],
            "distribution": [],
            "graph_context": [],
        }
        self._score_breakdown_all = {}
        self.test_fusion_weights = None

        self.model.eval()

        for inputs in tqdm(self.test_loader, desc="shark test scores"):
            windows, labels = prepare_inputs(inputs)
            outputs = self.model(windows)
            window_scores = outputs["window_score"].detach()
            score_breakdown = {
                key: value.detach()
                for key, value in outputs.get("score_breakdown", {}).items()
                if torch.is_tensor(value)
            }
            node_scores = outputs.get("node_scores_calibrated", outputs["node_scores"]).detach()
            direct_node_scores = outputs.get("direct_node_scores", outputs.get("root_node_scores", node_scores)).detach()
            root_node_scores = outputs.get("root_node_scores", direct_node_scores).detach()
            affected_node_scores = outputs.get("affected_node_scores", node_scores).detach()
            node_scores_raw = outputs.get("raw_node_scores", outputs.get("node_scores_raw", outputs["node_scores"])).detach()
            sensor_score = outputs.get("sensor_score", outputs["sensor_importance"]).detach()
            sensor_importance = outputs["sensor_importance"].detach()
            mil_sensor_risk = outputs.get("mil_sensor_risk")
            mil_pooling_weights = outputs.get("mil_pooling_weights")
            mil_direct_raw = outputs.get("mil_direct_raw")
            head_residual = outputs["head_residual"] if "head_residual" in outputs else outputs["bridge_residual"]
            head_residual = head_residual.detach()
            fusion_weights = outputs.get("fusion_weights")
            expert_scores = outputs.get("expert_scores")
            gate_weights = outputs.get("gate_weights")
            gate_logits = outputs.get("gate_logits")
            conservative_scores = outputs.get("conservative_scores")
            conservative_scores_raw = outputs.get("conservative_scores_raw")
            conservative_weights = outputs.get("conservative_weights")
            conservative_logits = outputs.get("conservative_logits")
            reliable_source_weights = outputs.get("reliable_source_weights")
            if reliable_source_names is None and "reliable_source_names" in outputs:
                reliable_source_names = outputs["reliable_source_names"]
            evidence_protocol_scores = outputs.get("evidence_protocol_scores")
            evidence_protocol_scores_raw = outputs.get("evidence_protocol_scores_raw")
            evidence_protocol_weights = outputs.get("evidence_protocol_weights")
            if evidence_protocol_names is None and "evidence_protocol_names" in outputs:
                evidence_protocol_names = outputs["evidence_protocol_names"]
            if source_reliability_prior is None and "source_reliability_prior" in outputs:
                source_reliability_prior = outputs["source_reliability_prior"].detach().cpu().numpy()
            if source_reliability_names is None and "source_reliability_names" in outputs:
                source_reliability_names = outputs["source_reliability_names"]
            sensor_importance_components = outputs.get("sensor_importance_components", {})

            window_scores_all.append(window_scores)
            labels_all.append(labels.detach())
            node_scores_all.append(node_scores)
            direct_node_scores_all.append(direct_node_scores)
            root_node_scores_all.append(root_node_scores)
            affected_node_scores_all.append(affected_node_scores)
            node_scores_raw_all.append(node_scores_raw)
            sensor_score_all.append(sensor_score)
            sensor_importance_all.append(sensor_importance)
            if mil_sensor_risk is not None:
                mil_sensor_risk_all.append(mil_sensor_risk.detach())
            if mil_pooling_weights is not None:
                mil_pooling_weights_all.append(mil_pooling_weights.detach())
            if mil_direct_raw is not None:
                mil_direct_raw_all.append(mil_direct_raw.detach())
            head_residual_all.append(head_residual)
            if expert_scores is not None:
                expert_scores_all.append(expert_scores.detach())
            if gate_weights is not None:
                gate_weights_all.append(gate_weights.detach())
            if gate_logits is not None:
                gate_logits_all.append(gate_logits.detach())
            if conservative_scores is not None:
                conservative_scores_all.append(conservative_scores.detach())
            if conservative_scores_raw is not None:
                conservative_scores_raw_all.append(conservative_scores_raw.detach())
            if conservative_weights is not None:
                conservative_weights_all.append(conservative_weights.detach())
            if conservative_logits is not None:
                conservative_logits_all.append(conservative_logits.detach())
            if reliable_source_weights is not None:
                reliable_source_weights_all.append(reliable_source_weights.detach())
            if evidence_protocol_scores is not None:
                evidence_protocol_scores_all.append(evidence_protocol_scores.detach())
            if evidence_protocol_scores_raw is not None:
                evidence_protocol_scores_raw_all.append(evidence_protocol_scores_raw.detach())
            if evidence_protocol_weights is not None:
                evidence_protocol_weights_all.append(evidence_protocol_weights.detach())
            for key in sensor_importance_component_all:
                if key in sensor_importance_components:
                    sensor_importance_component_all[key].append(sensor_importance_components[key].detach())
            for key, value in score_breakdown.items():
                self._score_breakdown_all.setdefault(key, []).append(value)

        window_scores = torch.flatten(torch.concat(window_scores_all, dim=0)).cpu().numpy()
        labels = torch.flatten(torch.concat(labels_all, dim=0)).cpu().numpy()
        node_scores = torch.concat(node_scores_all, dim=0).cpu().numpy()
        direct_node_scores = torch.concat(direct_node_scores_all, dim=0).cpu().numpy()
        root_node_scores = torch.concat(root_node_scores_all, dim=0).cpu().numpy()
        affected_node_scores = torch.concat(affected_node_scores_all, dim=0).cpu().numpy()
        node_scores_raw = torch.concat(node_scores_raw_all, dim=0).cpu().numpy()
        sensor_score = torch.concat(sensor_score_all, dim=0).cpu().numpy()
        sensor_importance = torch.concat(sensor_importance_all, dim=0).cpu().numpy()
        head_residual = torch.concat(head_residual_all, dim=0).cpu().numpy()

        self.test_window_scores = window_scores
        self.test_direct_node_scores = direct_node_scores
        self.test_root_node_scores = root_node_scores
        self.test_affected_node_scores = affected_node_scores
        self.test_node_scores_raw = node_scores_raw
        self.test_sensor_score = sensor_score
        self.test_sensor_importance = sensor_importance
        self.test_mil_sensor_risk = self._concat_optional(mil_sensor_risk_all)
        self.test_mil_pooling_weights = self._concat_optional(mil_pooling_weights_all)
        self.test_mil_direct_raw = self._concat_optional(mil_direct_raw_all)
        self.test_mil_reference_mean = (
            self.model.head.mil_reference_mean.detach().cpu().numpy()
            if hasattr(self.model.head, "mil_reference_mean")
            else None
        )
        self.test_mil_reference_scale = (
            self.model.head.mil_reference_scale.detach().cpu().numpy()
            if hasattr(self.model.head, "mil_reference_scale")
            else None
        )
        self.test_head_residual = head_residual
        self.test_expert_scores = self._concat_optional(expert_scores_all)
        self.test_gate_weights = self._concat_optional(gate_weights_all)
        self.test_gate_logits = self._concat_optional(gate_logits_all)
        self.test_conservative_scores = self._concat_optional(conservative_scores_all)
        self.test_conservative_scores_raw = self._concat_optional(conservative_scores_raw_all)
        self.test_conservative_weights = self._concat_optional(conservative_weights_all)
        self.test_conservative_logits = self._concat_optional(conservative_logits_all)
        self.test_reliable_source_weights = self._concat_optional(reliable_source_weights_all)
        self.test_reliable_source_names = (
            np.asarray(reliable_source_names, dtype=object)
            if reliable_source_names is not None
            else None
        )
        self.test_evidence_protocol_scores = self._concat_optional(evidence_protocol_scores_all)
        self.test_evidence_protocol_scores_raw = self._concat_optional(evidence_protocol_scores_raw_all)
        self.test_evidence_protocol_weights = self._concat_optional(evidence_protocol_weights_all)
        self.test_evidence_protocol_names = (
            np.asarray(evidence_protocol_names, dtype=object)
            if evidence_protocol_names is not None
            else None
        )
        self.test_source_reliability_prior = source_reliability_prior
        self.test_source_reliability_names = (
            np.asarray(source_reliability_names, dtype=object)
            if source_reliability_names is not None
            else None
        )
        self.test_fusion_weights = self.test_gate_weights
        self.test_sensor_importance_components = {
            key: self._concat_optional(values)
            for key, values in sensor_importance_component_all.items()
        }
        self.test_score_sources = self._build_score_sources(window_scores, node_scores_raw, node_scores)
        self.test_score_sources.update(self._node_score_sources("direct", direct_node_scores))
        self.test_score_sources.update(self._node_score_sources("root", root_node_scores))
        self.test_score_sources.update(self._node_score_sources("affected", affected_node_scores))
        for key, values in getattr(self, "_score_breakdown_all", {}).items():
            self.test_score_sources[key] = torch.flatten(torch.concat(values, dim=0)).cpu().numpy()
        source = self._normalize_score_source(self.cfg.SHARK.EVAL.SCORE_SOURCE)
        if source not in self.test_score_sources:
            raise ValueError(
                f"Unsupported SHARK.EVAL.SCORE_SOURCE: {self.cfg.SHARK.EVAL.SCORE_SOURCE}. "
                f"Available: {', '.join(self.test_score_sources.keys())}"
            )
        scores = self.test_score_sources[source]

        if len(scores) != len(labels):
            raise ValueError("SHARK test scores and labels have different lengths")
        if node_scores.shape[0] != len(scores):
            raise ValueError("SHARK node scores and window scores have different lengths")

        return scores, labels, node_scores

    def _build_score_sources(self, window_scores, raw_node_scores, cal_node_scores):
        sources = {"shark_head": window_scores, "aggregator": window_scores, "logits": window_scores}
        sources.update(self._node_score_sources("raw", raw_node_scores))
        sources.update(self._node_score_sources("cal", cal_node_scores))

        sources["node_mean"] = sources["cal_mean"]
        sources["node_max"] = sources["cal_max"]
        sources["node_topk_mean"] = sources["cal_topk"]
        sources["node_std"] = sources["cal_std"]
        return sources

    def _node_score_sources(self, prefix, node_scores):
        k = min(max(int(self.cfg.SHARK.HEAD.TOPK), 1), node_scores.shape[1])
        sorted_scores = np.sort(node_scores, axis=1)
        return {
            f"{prefix}_mean": node_scores.mean(axis=1),
            f"{prefix}_max": node_scores.max(axis=1),
            f"{prefix}_topk": sorted_scores[:, -k:].mean(axis=1),
            f"{prefix}_std": node_scores.std(axis=1),
        }

    @staticmethod
    def _normalize_score_source(source):
        aliases = {
            "node_mean": "cal_mean",
            "node_max": "cal_max",
            "node_topk_mean": "cal_topk",
            "node_std": "cal_std",
            "aggregator": "shark_head",
            "logits": "aggregator",
        }
        normalized = aliases.get(source, source)
        return aliases.get(normalized, normalized)

    def _get_threshold(self, test_scores, test_labels):
        if self.cfg.TEST.THRESHOLD.TYPE == "best_f1":
            return Thresholder(self.cfg, test_scores, test_labels).threshold
        return self.cfg.SHARK.EVAL.THRESHOLD

    @staticmethod
    def point_adjust(pred, gt):
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        return pred

    @staticmethod
    def get_results(scores, pred_pa, labels):
        results = {}

        #! AUROC
        auroc = float(roc_auc_score(labels, scores))
        results.update({"AUROC": auroc})

        #! AUPRC
        precision, recall, _ = precision_recall_curve(labels, scores)
        auprc = float(auc(recall, precision))
        results.update({"AUPRC": auprc})

        #! Precision, Recall, F1
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
        best_idx = np.argmax(f1)
        results.update({
            "Precision": precision[best_idx],
            "Recall": recall[best_idx],
            "F1": np.nanmax(f1),
        })

        #! Precision_PA, Recall_PA, F1_PA
        if pred_pa is not None:
            tp = np.sum((pred_pa == 1) & (labels == 1))
            fp = np.sum((pred_pa == 1) & (labels == 0))
            fn = np.sum((pred_pa == 0) & (labels == 1))
            precision_pa = tp / (tp + fp + 1e-12)
            recall_pa = tp / (tp + fn + 1e-12)
            f1_pa = 2 * (precision_pa * recall_pa) / (precision_pa + recall_pa + 1e-12)
            results.update({
                "Precision_PA": precision_pa,
                "Recall_PA": recall_pa,
                "F1_PA": f1_pa,
            })

        return results

    def get_explanation_results(self, pred):
        if "SMD" not in self.cfg.DATA.NAME:
            return {}

        interpretation = self._load_smd_interpretation_labels(end_inclusive=False)
        if not interpretation:
            return {}
        legacy_interpretation = self._load_smd_interpretation_labels(
            end_inclusive=True
        )

        k = min(max(int(self.cfg.SHARK.HEAD.TOPK), 1), self.test_node_scores.shape[1])
        pred_indices = np.where(pred == 1)[0]

        predicted = len(pred_indices) * k
        true_total = sum(
            len(nodes)
            for window_idx in range(len(self.test_scores))
            for nodes in [interpretation.get(self._window_end_time(window_idx))]
            if nodes
        )
        legacy_true_total = sum(
            len(nodes)
            for window_idx in range(len(self.test_scores))
            for nodes in [
                legacy_interpretation.get(self._window_end_time(window_idx))
            ]
            if nodes
        )

        sensor_hit, sensor_recall = self._sensor_metrics_for_scores(
            self.test_sensor_score,
            interpretation,
            pred_indices,
            predicted,
            true_total,
            k,
        )
        raw_hit, raw_recall = self._sensor_metrics_for_scores(
            self.test_node_scores_raw,
            interpretation,
            pred_indices,
            predicted,
            true_total,
            k,
        )
        cal_hit, cal_recall = self._sensor_metrics_for_scores(
            self.test_node_scores,
            interpretation,
            pred_indices,
            predicted,
            true_total,
            k,
        )
        legacy_hit, legacy_recall = self._sensor_metrics_for_scores(
            self.test_sensor_score,
            legacy_interpretation,
            pred_indices,
            predicted,
            legacy_true_total,
            k,
        )
        localization_hit, localization_precision, localization_recall = (
            self._localization_metrics_for_scores(
                self.test_sensor_score,
                interpretation,
                k,
            )
        )

        return {
            "Sensor_HitRate": sensor_hit,
            "Sensor_Recall": sensor_recall,
            "Sensor_HitRate_Protocol": sensor_hit,
            "Sensor_Recall_Protocol": sensor_recall,
            "Sensor_HitRate_RawTopK": raw_hit,
            "Sensor_Recall_RawTopK": raw_recall,
            "Sensor_HitRate_CalTopK": cal_hit,
            "Sensor_Recall_CalTopK": cal_recall,
            "Sensor_HitRate_LegacyInclusive": legacy_hit,
            "Sensor_Recall_LegacyInclusive": legacy_recall,
            "Sensor_Localization_HitRate@K": localization_hit,
            "Sensor_Localization_Precision@K": localization_precision,
            "Sensor_Localization_Recall@K": localization_recall,
        }

    def _sensor_metrics_for_scores(self, node_scores, interpretation, pred_indices, predicted, true_total, k):
        top_nodes = np.argsort(node_scores, axis=1)[:, -k:]
        hits = 0

        for window_idx in pred_indices:
            time_idx = self._window_end_time(window_idx)
            true_nodes = interpretation.get(time_idx)
            if not true_nodes:
                continue
            predicted_nodes = set(int(node) for node in top_nodes[window_idx])
            hits += len(predicted_nodes & true_nodes)

        return hits / (predicted + 1e-12), hits / (true_total + 1e-12)

    def _localization_metrics_for_scores(self, node_scores, interpretation, k):
        top_nodes = np.argsort(node_scores, axis=1)[:, -k:]
        annotated_windows = 0
        hit_windows = 0
        node_hits = 0
        true_total = 0
        for window_idx in range(node_scores.shape[0]):
            true_nodes = interpretation.get(self._window_end_time(window_idx))
            if not true_nodes:
                continue
            annotated_windows += 1
            predicted_nodes = set(int(node) for node in top_nodes[window_idx])
            overlap = len(predicted_nodes & true_nodes)
            hit_windows += int(overlap > 0)
            node_hits += overlap
            true_total += len(true_nodes)
        if annotated_windows == 0:
            return 0.0, 0.0, 0.0
        return (
            hit_windows / annotated_windows,
            node_hits / (annotated_windows * k),
            node_hits / max(true_total, 1),
        )

    def _window_end_time(self, window_idx):
        return int(
            window_idx * self.cfg.DATA.TEST_STEP
            + self.cfg.DATA.WIN_SIZE
            - 1
        )

    def get_msl_target_channel_results(self):
        if "MSL" not in self.cfg.DATA.NAME:
            return {}
        scores = np.asarray(self.test_sensor_score)
        labels = np.asarray(self.test_labels).reshape(-1)
        if scores.ndim != 2 or scores.shape[0] != labels.size or scores.shape[1] == 0:
            return {}
        anomaly_scores = scores[labels == 1]
        if anomaly_scores.shape[0] == 0:
            return {}
        k = min(max(int(self.cfg.SHARK.HEAD.TOPK), 1), scores.shape[1])
        order = np.argsort(-anomaly_scores, axis=1)
        target_positions = np.argmax(order == 0, axis=1)
        return {
            "MSL_TargetChannel_HR@1": float(np.mean(target_positions < 1)),
            "MSL_TargetChannel_HR@K": float(np.mean(target_positions < k)),
            "MSL_TargetChannel_MRR": float(
                np.mean(1.0 / (target_positions.astype(float) + 1.0))
            ),
        }

    @torch.no_grad()
    def get_controlled_directprop_results(self):
        root_prop_cfg = getattr(self.cfg.SHARK, "ROOT_PROP", None)
        if root_prop_cfg is None or not root_prop_cfg.CONTROLLED_EVAL_ENABLE:
            return {}

        k = min(max(int(self.cfg.SHARK.HEAD.TOPK), 1), self.test_node_scores.shape[1])
        max_windows = int(root_prop_cfg.CONTROLLED_EVAL_MAX_WINDOWS)
        n_nodes = max(int(self.test_node_scores.shape[1]), 1)
        results = {
            "Controlled_Random_HR@1": 1.0 / n_nodes,
            "Controlled_Random_HR@K": k / n_nodes,
        }

        direct_metrics = []
        for anomaly_type in self.augmentor.synthetic_types:
            metrics = self._evaluate_controlled_edit(anomaly_type, max_windows, k)
            if not metrics:
                continue
            direct_metrics.append(metrics)
            prefix = f"Controlled_{anomaly_type}"
            results[f"{prefix}_Direct_HR@1"] = metrics["direct_hr1"]
            results[f"{prefix}_Direct_HR@K"] = metrics["direct_hrk"]
            results[f"{prefix}_CoverageAdjusted_HR@1"] = metrics[
                "coverage_adjusted_hr1"
            ]
            results[f"{prefix}_CoverageAdjusted_HR@K"] = metrics[
                "coverage_adjusted_hrk"
            ]
            results[f"{prefix}_Direct_ScoreGap"] = metrics["direct_score_gap"]
            results[f"{prefix}_Direct_GainGap"] = metrics["direct_gain_gap"]
            results[f"{prefix}_Perturbation_RMS"] = metrics["perturbation_rms"]
            results[f"{prefix}_Attempted"] = metrics["attempted"]
            results[f"{prefix}_Evaluated"] = metrics["evaluated"]
            results[f"{prefix}_Coverage"] = metrics["coverage"]
            results[f"{prefix}_HardCoverage"] = metrics["hard_coverage"]
            results[f"{prefix}_SoftCoverage"] = metrics["soft_coverage"]
            results[f"{prefix}_InvalidCoverage"] = metrics["invalid_coverage"]
            results[f"{prefix}_WeightedCoverage"] = metrics["weighted_coverage"]
            results[f"{prefix}_AcceptedSampleWeightMean"] = metrics[
                "accepted_sample_weight_mean"
            ]
            results[f"{prefix}_MechanismScoreGain"] = metrics[
                "mechanism_score_gain"
            ]
            results[f"{prefix}_MechanismPositiveGainRatio"] = metrics[
                "mechanism_positive_gain_ratio"
            ]
            results[f"{prefix}_MechanismContextValidRatio"] = metrics[
                "mechanism_context_valid_ratio"
            ]
            results[f"{prefix}_EligibleAcceptance"] = metrics[
                "eligible_acceptance"
            ]
            results[f"{prefix}_SupportCollapsedRatio"] = metrics[
                "support_collapsed_ratio"
            ]
            results[f"{prefix}_SupportClippedRatio"] = metrics[
                "support_clipped_ratio"
            ]
            results[f"{prefix}_SupportViolationCount"] = metrics[
                "support_violation_count"
            ]
            results[f"{prefix}_RelationOrientation"] = metrics[
                "relation_orientation"
            ]
            results[f"{prefix}_Weighted_Direct_HR@1"] = metrics[
                "weighted_direct_hr1"
            ]
            results[f"{prefix}_Weighted_Direct_HR@K"] = metrics[
                "weighted_direct_hrk"
            ]
            results[f"{prefix}_Weighted_Direct_ScoreGap"] = metrics[
                "weighted_direct_score_gap"
            ]
            results[f"{prefix}_Weighted_Direct_GainGap"] = metrics[
                "weighted_direct_gain_gap"
            ]
            results[f"{prefix}_SurrogateRatio"] = metrics["surrogate_ratio"]
            results[f"{prefix}_FallbackRatio"] = metrics["fallback_ratio"]
            results[f"{prefix}_DirectRatio"] = metrics["direct_ratio"]
            results[f"{prefix}_FallbackMechanismGain"] = metrics[
                "fallback_mechanism_gain"
            ]
            for source_name, source_metrics in metrics.get("source_metrics", {}).items():
                source_prefix = f"{prefix}_{source_name}"
                results[f"{source_prefix}_Count"] = source_metrics["count"]
                results[f"{source_prefix}_Ratio"] = source_metrics["ratio"]
                results[f"{source_prefix}_Direct_HR@1"] = source_metrics["direct_hr1"]
                results[f"{source_prefix}_Direct_HR@K"] = source_metrics["direct_hrk"]
                results[f"{source_prefix}_Direct_ScoreGap"] = source_metrics[
                    "direct_score_gap"
                ]
                results[f"{source_prefix}_Direct_GainGap"] = source_metrics[
                    "direct_gain_gap"
                ]

        if direct_metrics:
            for key, output_name in (
                ("direct_hr1", "Controlled_Direct_Macro_HR@1"),
                ("direct_hrk", "Controlled_Direct_Macro_HR@K"),
                ("coverage_adjusted_hr1", "Controlled_Direct_Macro_CoverageAdjusted_HR@1"),
                ("coverage_adjusted_hrk", "Controlled_Direct_Macro_CoverageAdjusted_HR@K"),
                ("direct_score_gap", "Controlled_Direct_Macro_ScoreGap"),
                ("direct_gain_gap", "Controlled_Direct_Macro_GainGap"),
                ("perturbation_rms", "Controlled_Direct_Macro_Perturbation_RMS"),
                ("coverage", "Controlled_Direct_Macro_Coverage"),
                ("hard_coverage", "Controlled_Direct_Macro_HardCoverage"),
                ("soft_coverage", "Controlled_Direct_Macro_SoftCoverage"),
                ("invalid_coverage", "Controlled_Direct_Macro_InvalidCoverage"),
                ("weighted_coverage", "Controlled_Direct_Macro_WeightedCoverage"),
                (
                    "accepted_sample_weight_mean",
                    "Controlled_Direct_Macro_AcceptedSampleWeightMean",
                ),
                (
                    "mechanism_positive_gain_ratio",
                    "Controlled_Direct_Macro_MechanismPositiveGainRatio",
                ),
                (
                    "mechanism_context_valid_ratio",
                    "Controlled_Direct_Macro_MechanismContextValidRatio",
                ),
                (
                    "eligible_acceptance",
                    "Controlled_Direct_Macro_EligibleAcceptance",
                ),
                (
                    "support_violation_count",
                    "Controlled_Direct_Macro_SupportViolationCount",
                ),
                ("weighted_direct_hr1", "Controlled_Direct_Macro_Weighted_HR@1"),
                ("weighted_direct_hrk", "Controlled_Direct_Macro_Weighted_HR@K"),
                (
                    "weighted_direct_score_gap",
                    "Controlled_Direct_Macro_Weighted_ScoreGap",
                ),
                (
                    "weighted_direct_gain_gap",
                    "Controlled_Direct_Macro_Weighted_GainGap",
                ),
                ("surrogate_ratio", "Controlled_Direct_Macro_SurrogateRatio"),
                ("fallback_ratio", "Controlled_Direct_Macro_FallbackRatio"),
                ("direct_ratio", "Controlled_Direct_Macro_DirectRatio"),
                (
                    "fallback_mechanism_gain",
                    "Controlled_Direct_Macro_FallbackMechanismGain",
                ),
            ):
                values = [
                    float(metrics[key])
                    for metrics in direct_metrics
                    if math.isfinite(float(metrics[key]))
                ]
                results[output_name] = (
                    sum(values) / len(values) if values else float("nan")
                )

        propagation_metrics = []
        for anomaly_type in self.augmentor.diagnostic_types:
            metrics = self._evaluate_controlled_edit(anomaly_type, max_windows, k)
            if not metrics:
                continue
            propagation_metrics.append(metrics)
            prefix = f"Controlled_{anomaly_type}_Diagnostic"
            results[f"{prefix}_Direct_HR@1"] = metrics["direct_hr1"]
            results[f"{prefix}_Direct_HR@K"] = metrics["direct_hrk"]
            results[f"{prefix}_Affected_HR@K"] = metrics["affected_hrk"]
            results[f"{prefix}_Propagated_HR@K"] = metrics["propagated_hrk"]
            results[f"{prefix}_DirectProp_Confusion@K"] = metrics["direct_prop_confusion"]

        if propagation_metrics:
            metrics = propagation_metrics[0]
            results.update({
                "Controlled_Direct_HR@K": metrics["direct_hrk"],
                "Controlled_Affected_Diagnostic_HR@K": metrics["affected_hrk"],
                "Controlled_Propagated_Diagnostic_HR@K": metrics["propagated_hrk"],
                "Controlled_DirectProp_Confusion@K": metrics["direct_prop_confusion"],
            })
        return results

    @torch.no_grad()
    def _evaluate_controlled_edit(self, anomaly_type, max_windows, k):
        controlled_types = (anomaly_type,)
        totals = {
            "direct_hr1": 0.0,
            "direct_hrk": 0.0,
            "direct_score_gap": 0.0,
            "direct_gain_gap": 0.0,
            "perturbation_rms": 0.0,
            "affected_hits": 0.0,
            "affected_total": 0.0,
            "propagated_hits": 0.0,
            "propagated_total": 0.0,
            "confusion_hits": 0.0,
            "evaluated": 0,
            "attempted": 0,
            "surrogate_evaluated": 0,
            "fallback_evaluated": 0,
            "direct_evaluated": 0,
            "hard_evaluated": 0,
            "soft_evaluated": 0,
            "sample_weight_sum": 0.0,
            "weighted_direct_hr1": 0.0,
            "weighted_direct_hrk": 0.0,
            "weighted_direct_score_gap": 0.0,
            "weighted_direct_gain_gap": 0.0,
            "fallback_mechanism_gain": 0.0,
            "fallback_mechanism_gain_count": 0,
            "mechanism_score_gain_sum": 0.0,
            "mechanism_score_gain_count": 0,
            "mechanism_positive_gain_count": 0,
            "mechanism_candidate_eligible_count": 0,
            "mechanism_context_valid_count": 0,
            "mechanism_candidate_total_count": 0,
            "mechanism_support_eligible_count": 0,
            "mechanism_support_collapsed_count": 0,
            "mechanism_support_clipped_count": 0,
            "mechanism_support_violation_count": 0,
            "mechanism_orientation_sum": 0.0,
            "mechanism_orientation_count": 0,
            "source_totals": {},
        }

        self.model.eval()
        for inputs in tqdm(self.test_loader, desc=f"controlled {anomaly_type} diagnostic"):
            windows, labels = prepare_inputs(inputs)
            normal_mask = labels.view(-1) == 0
            if not normal_mask.any():
                continue
            windows = windows[normal_mask]

            if totals["attempted"] >= max_windows:
                break
            if totals["attempted"] + windows.size(0) > max_windows:
                windows = windows[:max_windows - totals["attempted"]]
            totals["attempted"] += windows.size(0)

            base_node_features = self.model.imlp(windows.float())
            base_adjacency = self.model.graph_builder(base_node_features)
            original_types = self.augmentor.synthetic_types
            self.augmentor.synthetic_types = controlled_types
            try:
                synthetic, direct_mask, propagated_mask, _ = self.augmentor.relation_function_breaking_view(
                    windows,
                    base_adjacency,
                    progress=1.0,
                    frontend_reference=base_node_features,
                )
            finally:
                self.augmentor.synthetic_types = original_types
            generator_diagnostics = getattr(
                self.augmentor,
                "last_coverage_diagnostics",
                {},
            )
            for source_key, total_key in (
                ("candidate_score_gain_count", "mechanism_score_gain_count"),
                ("candidate_positive_gain_count", "mechanism_positive_gain_count"),
                ("candidate_eligible_count", "mechanism_candidate_eligible_count"),
                ("candidate_context_valid_count", "mechanism_context_valid_count"),
                ("candidate_total_count", "mechanism_candidate_total_count"),
                ("support_eligible_count", "mechanism_support_eligible_count"),
                ("support_collapsed_count", "mechanism_support_collapsed_count"),
                ("candidate_support_clipped_count", "mechanism_support_clipped_count"),
                ("candidate_support_violation_count", "mechanism_support_violation_count"),
                ("selected_orientation_count", "mechanism_orientation_count"),
            ):
                totals[total_key] += int(generator_diagnostics.get(source_key, 0))
            totals["mechanism_score_gain_sum"] += float(
                generator_diagnostics.get("candidate_score_gain_sum", 0.0)
            )
            totals["mechanism_orientation_sum"] += float(
                generator_diagnostics.get("selected_orientation_sum", 0.0)
            )
            valid_mask = getattr(self.augmentor, "last_valid_mask", None)
            sample_weights = getattr(self.augmentor, "last_sample_weights", None)
            acceptance_tier = getattr(self.augmentor, "last_acceptance_tier", None)
            edit_source_ids = getattr(self.augmentor, "last_edit_source_ids", None)
            fallback_type_ids = getattr(self.augmentor, "last_fallback_type_ids", None)
            direct_type_ids = getattr(self.augmentor, "last_direct_type_ids", None)
            acceptance_invalid = getattr(
                self.augmentor,
                "ACCEPTANCE_INVALID",
                SharkAugmentor.ACCEPTANCE_INVALID,
            )
            acceptance_soft = getattr(
                self.augmentor,
                "ACCEPTANCE_SOFT",
                SharkAugmentor.ACCEPTANCE_SOFT,
            )
            acceptance_hard = getattr(
                self.augmentor,
                "ACCEPTANCE_HARD",
                SharkAugmentor.ACCEPTANCE_HARD,
            )
            if valid_mask is None:
                valid_mask = torch.ones(
                    windows.size(0),
                    device=windows.device,
                    dtype=torch.bool,
                )
            else:
                valid_mask = valid_mask.bool()
            if sample_weights is None:
                sample_weights = valid_mask.to(dtype=windows.dtype)
            else:
                sample_weights = sample_weights.to(
                    device=windows.device,
                    dtype=windows.dtype,
                )
            if acceptance_tier is None:
                acceptance_tier = torch.where(
                    valid_mask,
                    torch.full_like(
                        valid_mask,
                        acceptance_hard,
                        dtype=torch.long,
                    ),
                    torch.full_like(
                        valid_mask,
                        acceptance_invalid,
                        dtype=torch.long,
                    ),
                )
            totals["hard_evaluated"] += int(
                (acceptance_tier == acceptance_hard).sum().item()
            )
            totals["soft_evaluated"] += int(
                (acceptance_tier == acceptance_soft).sum().item()
            )
            if valid_mask is not None:
                if not valid_mask.any():
                    continue
                synthetic = synthetic[valid_mask]
                direct_mask = direct_mask[valid_mask]
                propagated_mask = propagated_mask[valid_mask]
                windows = windows[valid_mask]
                sample_weights = sample_weights[valid_mask]
                acceptance_tier = acceptance_tier[valid_mask]
                if edit_source_ids is not None:
                    edit_source_ids = edit_source_ids[valid_mask]
                if fallback_type_ids is not None:
                    fallback_type_ids = fallback_type_ids[valid_mask]
                if direct_type_ids is not None:
                    direct_type_ids = direct_type_ids[valid_mask]
            outputs_anchor = self.model(windows)
            outputs = self.model(synthetic)

            direct_scores = outputs.get("direct_node_scores", outputs.get("root_node_scores", outputs["node_scores"])).detach()
            anchor_direct_scores = outputs_anchor.get(
                "direct_node_scores",
                outputs_anchor.get("root_node_scores", outputs_anchor["node_scores"]),
            ).detach()
            affected_scores = outputs.get("affected_node_scores", outputs["node_scores"]).detach()
            direct_mask = direct_mask.bool()
            propagated_mask = propagated_mask.bool()
            affected_mask = direct_mask | propagated_mask
            unmodified_mask = ~affected_mask

            direct_top1 = direct_scores.argmax(dim=1, keepdim=True)
            direct_topk = torch.topk(direct_scores, k=k, dim=1).indices
            affected_topk = torch.topk(affected_scores, k=k, dim=1).indices
            batch_size = windows.size(0)
            direct_hr1 = direct_mask.gather(1, direct_top1).any(dim=1).float()
            direct_hrk = direct_mask.gather(1, direct_topk).any(dim=1).float()
            direct_mean = (direct_scores * direct_mask.float()).sum(dim=1) / direct_mask.sum(dim=1).clamp_min(1)
            unmodified_mean = (
                (direct_scores * unmodified_mask.float()).sum(dim=1)
                / unmodified_mask.sum(dim=1).clamp_min(1)
            )
            score_gain = direct_scores - anchor_direct_scores
            direct_gain = (
                (score_gain * direct_mask.float()).sum(dim=1)
                / direct_mask.sum(dim=1).clamp_min(1)
            )
            unmodified_gain = (
                (score_gain * unmodified_mask.float()).sum(dim=1)
                / unmodified_mask.sum(dim=1).clamp_min(1)
            )
            sample_score_gap = direct_mean - unmodified_mean
            sample_gain_gap = direct_gain - unmodified_gain

            totals["direct_hr1"] += float(direct_hr1.sum().cpu())
            totals["direct_hrk"] += float(direct_hrk.sum().cpu())
            totals["direct_score_gap"] += float(sample_score_gap.sum().cpu())
            totals["direct_gain_gap"] += float(sample_gain_gap.sum().cpu())
            totals["sample_weight_sum"] += float(sample_weights.sum().cpu())
            totals["weighted_direct_hr1"] += float(
                (direct_hr1 * sample_weights).sum().cpu()
            )
            totals["weighted_direct_hrk"] += float(
                (direct_hrk * sample_weights).sum().cpu()
            )
            totals["weighted_direct_score_gap"] += float(
                (sample_score_gap * sample_weights).sum().cpu()
            )
            totals["weighted_direct_gain_gap"] += float(
                (sample_gain_gap * sample_weights).sum().cpu()
            )
            type_delta_rms = self.augmentor.last_diagnostics.get("type_delta_rms", [float("nan")])
            if type_delta_rms:
                totals["perturbation_rms"] += float(type_delta_rms[0]) * batch_size

            totals["affected_hits"] += float(affected_mask.gather(1, affected_topk).sum().cpu())
            totals["affected_total"] += float(affected_mask.sum().cpu())
            totals["propagated_hits"] += float(propagated_mask.gather(1, affected_topk).sum().cpu())
            totals["propagated_total"] += float(propagated_mask.sum().cpu())
            totals["confusion_hits"] += float(propagated_mask.gather(1, direct_topk).sum().cpu())
            if edit_source_ids is not None:
                totals["surrogate_evaluated"] += int(
                    (edit_source_ids == self.augmentor.EDIT_SOURCE_SURROGATE).sum().item()
                )
                totals["fallback_evaluated"] += int(
                    (edit_source_ids == self.augmentor.EDIT_SOURCE_FALLBACK).sum().item()
                )
                totals["direct_evaluated"] += int(
                    (edit_source_ids == self.augmentor.EDIT_SOURCE_DIRECT).sum().item()
                )
                batch_fallback_count = int(
                    (edit_source_ids == self.augmentor.EDIT_SOURCE_FALLBACK).sum().item()
                )
                mechanism_gain = float(
                    self.augmentor.last_diagnostics.get(
                        "hybrid_fallback_mechanism_gain",
                        float("nan"),
                    )
                )
                if batch_fallback_count > 0 and math.isfinite(mechanism_gain):
                    totals["fallback_mechanism_gain"] += (
                        mechanism_gain * batch_fallback_count
                    )
                    totals["fallback_mechanism_gain_count"] += batch_fallback_count
                source_masks = {
                    "ActualSurrogate": (
                        edit_source_ids == self.augmentor.EDIT_SOURCE_SURROGATE
                    ),
                    "ActualFallback": (
                        edit_source_ids == self.augmentor.EDIT_SOURCE_FALLBACK
                    ),
                    "ActualDirect": (
                        edit_source_ids == self.augmentor.EDIT_SOURCE_DIRECT
                    ),
                    "AcceptanceHard": (
                        acceptance_tier == acceptance_hard
                    ),
                    "AcceptanceSoft": (
                        acceptance_tier == acceptance_soft
                    ),
                }
                if fallback_type_ids is not None:
                    fallback_mask = source_masks["ActualFallback"]
                    for type_idx, type_name in enumerate(
                        self.augmentor.hybrid_fallback_types
                    ):
                        source_masks[f"Fallback_{type_name}"] = (
                            fallback_mask & (fallback_type_ids == type_idx)
                        )
                if direct_type_ids is not None:
                    direct_source_mask = source_masks["ActualDirect"]
                    for type_idx, type_name in enumerate(controlled_types):
                        source_masks[f"Direct_{type_name}"] = (
                            direct_source_mask & (direct_type_ids == type_idx)
                        )
                for source_name, source_mask in source_masks.items():
                    source_count = int(source_mask.sum().item())
                    if source_count <= 0:
                        continue
                    source_total = totals["source_totals"].setdefault(
                        source_name,
                        {
                            "count": 0,
                            "direct_hr1": 0.0,
                            "direct_hrk": 0.0,
                            "direct_score_gap": 0.0,
                            "direct_gain_gap": 0.0,
                        },
                    )
                    source_total["count"] += source_count
                    source_total["direct_hr1"] += float(
                        direct_hr1[source_mask].sum().cpu()
                    )
                    source_total["direct_hrk"] += float(
                        direct_hrk[source_mask].sum().cpu()
                    )
                    source_total["direct_score_gap"] += float(
                        sample_score_gap[source_mask].sum().cpu()
                    )
                    source_total["direct_gain_gap"] += float(
                        sample_gain_gap[source_mask].sum().cpu()
                    )
            totals["evaluated"] += batch_size

        evaluated = totals["evaluated"]
        if evaluated == 0:
            return {
                "direct_hr1": float("nan"),
                "direct_hrk": float("nan"),
                "coverage_adjusted_hr1": 0.0,
                "coverage_adjusted_hrk": 0.0,
                "direct_score_gap": float("nan"),
                "direct_gain_gap": float("nan"),
                "perturbation_rms": float("nan"),
                "attempted": float(totals["attempted"]),
                "evaluated": 0.0,
                "coverage": 0.0,
                "hard_coverage": 0.0,
                "soft_coverage": 0.0,
                "invalid_coverage": 1.0,
                "weighted_coverage": 0.0,
                "accepted_sample_weight_mean": 0.0,
                "mechanism_score_gain": float("nan"),
                "mechanism_positive_gain_ratio": 0.0,
                "mechanism_context_valid_ratio": 0.0,
                "eligible_acceptance": 0.0,
                "support_collapsed_ratio": (
                    totals["mechanism_support_collapsed_count"]
                    / max(totals["attempted"], 1)
                ),
                "support_clipped_ratio": 0.0,
                "support_violation_count": float(
                    totals["mechanism_support_violation_count"]
                ),
                "relation_orientation": float("nan"),
                "weighted_direct_hr1": float("nan"),
                "weighted_direct_hrk": float("nan"),
                "weighted_direct_score_gap": float("nan"),
                "weighted_direct_gain_gap": float("nan"),
                "surrogate_ratio": 0.0,
                "fallback_ratio": 0.0,
                "direct_ratio": 0.0,
                "fallback_mechanism_gain": float("nan"),
                "affected_hrk": float("nan"),
                "propagated_hrk": float("nan"),
                "direct_prop_confusion": float("nan"),
                "source_metrics": {},
            }

        source_metrics = {}
        for source_name, source_total in totals["source_totals"].items():
            source_count = source_total["count"]
            source_metrics[source_name] = {
                "count": float(source_count),
                "ratio": source_count / max(totals["attempted"], 1),
                "direct_hr1": source_total["direct_hr1"] / source_count,
                "direct_hrk": source_total["direct_hrk"] / source_count,
                "direct_score_gap": source_total["direct_score_gap"] / source_count,
                "direct_gain_gap": source_total["direct_gain_gap"] / source_count,
            }
        return {
            "direct_hr1": totals["direct_hr1"] / evaluated,
            "direct_hrk": totals["direct_hrk"] / evaluated,
            "coverage_adjusted_hr1": totals["direct_hr1"] / max(totals["attempted"], 1),
            "coverage_adjusted_hrk": totals["direct_hrk"] / max(totals["attempted"], 1),
            "direct_score_gap": totals["direct_score_gap"] / evaluated,
            "direct_gain_gap": totals["direct_gain_gap"] / evaluated,
            "perturbation_rms": totals["perturbation_rms"] / evaluated,
            "attempted": float(totals["attempted"]),
            "evaluated": float(evaluated),
            "coverage": evaluated / max(totals["attempted"], 1),
            "hard_coverage": totals["hard_evaluated"] / max(totals["attempted"], 1),
            "soft_coverage": totals["soft_evaluated"] / max(totals["attempted"], 1),
            "invalid_coverage": (
                max(totals["attempted"] - evaluated, 0)
                / max(totals["attempted"], 1)
            ),
            "weighted_coverage": totals["sample_weight_sum"] / max(totals["attempted"], 1),
            "accepted_sample_weight_mean": totals["sample_weight_sum"] / max(evaluated, 1),
            "mechanism_score_gain": (
                totals["mechanism_score_gain_sum"]
                / max(totals["mechanism_score_gain_count"], 1)
            ),
            "mechanism_positive_gain_ratio": (
                totals["mechanism_positive_gain_count"]
                / max(totals["mechanism_candidate_eligible_count"], 1)
            ),
            "mechanism_context_valid_ratio": (
                totals["mechanism_context_valid_count"]
                / max(totals["mechanism_candidate_total_count"], 1)
            ),
            "eligible_acceptance": (
                evaluated
                / max(
                    totals["mechanism_support_eligible_count"]
                    or totals["attempted"],
                    1,
                )
            ),
            "support_collapsed_ratio": (
                totals["mechanism_support_collapsed_count"]
                / max(totals["attempted"], 1)
            ),
            "support_clipped_ratio": (
                totals["mechanism_support_clipped_count"]
                / max(totals["mechanism_candidate_total_count"], 1)
            ),
            "support_violation_count": float(
                totals["mechanism_support_violation_count"]
            ),
            "relation_orientation": (
                totals["mechanism_orientation_sum"]
                / totals["mechanism_orientation_count"]
                if totals["mechanism_orientation_count"] > 0
                else float("nan")
            ),
            "weighted_direct_hr1": (
                totals["weighted_direct_hr1"] / max(totals["sample_weight_sum"], 1e-12)
            ),
            "weighted_direct_hrk": (
                totals["weighted_direct_hrk"] / max(totals["sample_weight_sum"], 1e-12)
            ),
            "weighted_direct_score_gap": (
                totals["weighted_direct_score_gap"] / max(totals["sample_weight_sum"], 1e-12)
            ),
            "weighted_direct_gain_gap": (
                totals["weighted_direct_gain_gap"] / max(totals["sample_weight_sum"], 1e-12)
            ),
            "surrogate_ratio": totals["surrogate_evaluated"] / max(totals["attempted"], 1),
            "fallback_ratio": totals["fallback_evaluated"] / max(totals["attempted"], 1),
            "direct_ratio": totals["direct_evaluated"] / max(totals["attempted"], 1),
            "fallback_mechanism_gain": (
                totals["fallback_mechanism_gain"]
                / totals["fallback_mechanism_gain_count"]
                if totals["fallback_mechanism_gain_count"] > 0
                else float("nan")
            ),
            "affected_hrk": totals["affected_hits"] / (totals["affected_total"] + 1e-12),
            "propagated_hrk": totals["propagated_hits"] / (totals["propagated_total"] + 1e-12),
            "direct_prop_confusion": totals["confusion_hits"] / (totals["propagated_total"] + 1e-12),
            "source_metrics": source_metrics,
        }

    def get_score_source_sweep(self):
        sweep_sources = (
            "shark_head",
            "mil_window_score",
            "mil_lse_score",
            "mil_topk_diagnostic_score",
            "head_score",
            "decoupled_window_score",
            "normal_ref_agreement_score",
            "normal_ref_contrast_score",
            "relation_aware_normal_ref_score",
            "relation_aware_base_score",
            "relation_aware_relation_evidence",
            "relation_aware_relation_gate",
            "relation_aware_low_base_gate",
            "relation_aware_relation_bonus",
            "evidence_base_risk",
            "evidence_contrast_base",
            "evidence_contrast_score",
            "evidence_agreement_score",
            "evidence_disagreement_score",
            "moe_score",
            "moe_spike",
            "moe_distribution",
            "moe_graph_context",
            "raw_mean",
            "raw_max",
            "raw_topk",
            "raw_std",
            "raw_log_max",
            "raw_log_topk",
            "raw_log_std",
            "raw_log_std_energy",
            "cal_mean",
            "cal_max",
            "cal_topk",
            "cal_std",
            "cal_std_energy",
            "direct_mean",
            "direct_max",
            "direct_topk",
            "direct_std",
            "direct_evidence_score",
            "local_evidence_score",
            "relation_evidence_score",
            "context_evidence_score",
            "protocol_raw_direct",
            "protocol_raw_local",
            "protocol_raw_relation",
            "protocol_raw_context",
            "protocol_cal_direct",
            "protocol_cal_local",
            "protocol_cal_relation",
            "protocol_cal_context",
            "root_mean",
            "root_max",
            "root_topk",
            "root_std",
            "affected_mean",
            "affected_max",
            "affected_topk",
            "affected_std",
            "evidence_cal_moe",
            "evidence_cal_head",
            "evidence_cal_direct_mean",
            "evidence_cal_direct_max",
            "evidence_cal_direct_topk",
            "evidence_cal_raw_max",
            "evidence_cal_raw_topk",
            "evidence_cal_raw_mean",
            "evidence_cal_cal_mean",
            "evidence_cal_cal_topk",
            "evidence_cal_raw_std",
            "evidence_cal_cal_std_energy",
            "evidence_raw_direct_mean",
            "evidence_raw_raw_mean",
            "evidence_raw_cal_mean",
            "safe_calibrated_score",
            "source_consistent_score",
            "reliable_source_score",
        )
        sweep = {}
        for source in sweep_sources:
            if source not in self.test_score_sources:
                continue
            scores = self.test_score_sources[source]
            threshold = self._get_threshold(scores, self.test_labels)
            pred = (scores > threshold).astype(int)
            pred_pa = self.point_adjust(pred.copy(), self.test_labels) if self.cfg.TEST.POINT_ADJUST else None
            metrics = self.get_results(scores, pred_pa, self.test_labels)
            sweep[source] = {
                "Threshold": threshold,
                "AUROC": metrics["AUROC"],
                "AUPRC": metrics["AUPRC"],
                "F1": metrics["F1"],
                "F1_PA": metrics.get("F1_PA", float("nan")),
            }
        return sweep

    @staticmethod
    def _concat_optional(tensors):
        if not tensors:
            return None
        return torch.concat(tensors, dim=0).cpu().numpy()

    def _load_smd_interpretation_labels(self, end_inclusive=False):
        machine = self.cfg.DATA.NAME.split("_", 1)[1]
        filename = f"machine-{machine}.txt"
        candidates = [
            os.path.join(self.cfg.DATA.BASE_DIR, "smd_raw", "interpretation_label", filename),
            os.path.join(self.cfg.DATA.BASE_DIR, "..", "smd_raw", "interpretation_label", filename),
            os.path.join(self.cfg.DATA.BASE_DIR, "..", "..", "data", "smd_raw", "interpretation_label", filename),
            os.path.join("data", "smd_raw", "interpretation_label", filename),
            os.path.join("..", "data", "smd_raw", "interpretation_label", filename),
        ]
        path = next((os.path.normpath(candidate) for candidate in candidates if os.path.isfile(os.path.normpath(candidate))), None)
        if path is None:
            return {}

        labels = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                span, nodes = line.split(":")
                start, end = [int(value) for value in span.split("-")]
                node_set = set(int(value) - 1 for value in nodes.split(",") if value)
                stop = end + 1 if end_inclusive else end
                for time_idx in range(start, stop):
                    labels.setdefault(time_idx, set()).update(node_set)
        return labels

    def save_results(self, results, sweep_results=None, peft_diagnostics=None):
        results_with_threshold = {
            "ScoreSource": self.cfg.SHARK.EVAL.SCORE_SOURCE,
            "Threshold": self.threshold,
            **results,
        }
        results_string = ", ".join([
            f"{metric}: {value}" if isinstance(value, str) else f"{metric}: {value:.04f}"
            for metric, value in results_with_threshold.items()
        ])
        print(results_string)

        with open(os.path.join(mkdir(self.result_dir) / "test.txt"), "w") as f:
            f.write(results_string)
            if sweep_results:
                f.write("\n\nScore source sweep:\n")
                for source, metrics in sweep_results.items():
                    metric_string = ", ".join([
                        f"{metric}: {value:.04f}"
                        for metric, value in metrics.items()
                    ])
                    f.write(f"{source}: {metric_string}\n")
            if peft_diagnostics:
                f.write("\nPEFT diagnostics:\n")
                for key, value in peft_diagnostics.items():
                    if isinstance(value, (list, tuple)):
                        value_text = ", ".join(str(item) for item in value)
                    elif isinstance(value, float):
                        value_text = f"{value:.06f}"
                    else:
                        value_text = str(value)
                    f.write(f"{key}: {value_text}\n")
            if self.test_source_reliability_prior is not None:
                f.write("\nSource reliability prior:\n")
                names = self.test_source_reliability_names
                if names is None:
                    names = [f"source_{idx}" for idx in range(len(self.test_source_reliability_prior))]
                for name, value in zip(names, self.test_source_reliability_prior):
                    f.write(f"{name}: {float(value):.06f}\n")

    def save_metric_protocol(self, results):
        """保存十项指标及其阈值/公式元数据，避免只审计 test.txt。"""
        import json

        def json_safe(value):
            if isinstance(value, dict):
                return {str(key): json_safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(item) for item in value]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                return "NA"
            return value

        metric_names = (
            "Precision",
            "Recall",
            "F1",
            "Precision_PA",
            "Recall_PA",
            "F1_PA",
            "R-F1",
            "Aff-F1",
            "AUROC",
            "V-ROC",
            "AUPRC",
            "AP",
            "V-PR",
            "PGRF_PA_P",
            "PGRF_PA_R",
            "PGRF_PA_F1",
            "PGRF_Raw_P",
            "PGRF_Raw_R",
            "PGRF_Raw_F1",
            "PGRF_R-P",
            "PGRF_R-R",
            "PGRF_R-F1",
            "OracleAD200_P",
            "OracleAD200_R",
            "OracleAD200_F1",
            "OracleAD200_R-P",
            "OracleAD200_R-R",
            "OracleAD200_R-F1",
            "OracleAD200_Aff-P",
            "OracleAD200_Aff-R",
            "OracleAD200_Aff-F1",
            "OracleAD200_A-ROC",
            "OracleAD200_A-PR",
            "OracleAD200_V-ROC",
            "OracleAD200_V-PR",
        )
        metrics = {}
        for name in metric_names:
            value = results.get(name, "NA")
            metrics[name] = json_safe(value)
        payload = {
            "schema_version": "SHARK-EXTENDED-MTSAD-METRICS-V2",
            "entity": str(self.cfg.DATA.NAME),
            "score_source": str(self.cfg.SHARK.EVAL.SCORE_SOURCE),
            "threshold": float(self.threshold),
            "metrics": metrics,
            "oraclead_protocol": json_safe(self.oraclead_metric_report),
            "oraclead_paper_200_protocol": json_safe(
                self.oraclead_paper_metric_report
            ),
            "pgrf_protocol": json_safe(self.pgrf_metric_report),
            "physdiff_protocol": {
                "status": "MISSING_EXACT_SPOT_PREDICTIONS",
                "required": (
                    "SPOT dynamic predictions calibrated on the paper's normal split; "
                    "reported P/R/F1 are Affiliation metrics"
                ),
                "warning": (
                    "Oracle Aff-F1 and point-adjusted F1 are not directly comparable "
                    "to PhysDiff"
                ),
            },
            "carots_compatibility": {
                "AUROC": "sklearn.roc_auc_score.raw.v1",
                "AUPRC": "sklearn.precision_recall_curve+auc.trapezoid.raw.v1",
                "Precision_Recall_F1": "carots.test_pr_curve_best_f1.v1",
                "point_adjustment": "reported separately as Precision_PA/Recall_PA/F1_PA",
            },
            "label_protocol": {
                "window_label": "any anomaly point inside the input window",
                "window_size": int(self.cfg.DATA.WIN_SIZE),
                "test_step": int(self.cfg.DATA.TEST_STEP),
                "downsample_rate": int(self.cfg.DATA.DOWNSAMPLE_RATE),
                "swat_downsample_label": (
                    "block_any"
                    if "SWaT" in str(self.cfg.DATA.NAME)
                    else "not_applicable"
                ),
                "paper_comparability_warning": (
                    "SHARK/CAROTS evaluates one score per sliding window with "
                    "window-any labels; timestamp-level papers are not strictly "
                    "comparable unless the score/label alignment is replayed."
                ),
            },
        }
        path = mkdir(self.result_dir) / "test_metric_protocol.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

    def save_to_npy(self, **kwargs):
        result_dir = mkdir(self.result_dir)
        for key, value in kwargs.items():
            np.save(os.path.join(result_dir, f"{key}.npy"), value)
