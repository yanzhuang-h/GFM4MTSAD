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
from models.gfm4mtsad.augment import GFM4MTSADAugmentor


class GFM4MTSADPredictor:
    def __init__(self, cfg, model):
        self.cfg = cfg.clone()

        self.model = model

        self.threshold = None
        self.result_dir = os.path.join(cfg.GFM4MTSAD.EVAL.RESULT_DIR, cfg.DATA.NAME)

        self.test_loader = get_test_dataloader(cfg)
        self.augmentor = GFM4MTSADAugmentor(cfg)
        if self.augmentor.requires_normal_mechanism:
            fitted_mechanism = getattr(model, "_gfm4mtsad_normal_mechanism", None)
            if fitted_mechanism is not None and fitted_mechanism.fitted:
                self.augmentor.normal_mechanism = fitted_mechanism
                diagnostics = fitted_mechanism.diagnostics()
            else:
                diagnostics = self.augmentor.fit_normal_mechanism(
                    self.test_loader.dataset.train
                )
            print(
                "[GFM4MTSAD predictor mechanism] "
                f"valid_nodes={diagnostics.get('valid_node_count', 0)}/"
                f"{diagnostics.get('n_nodes', 0)} "
                f"valid_ratio={diagnostics.get('valid_node_ratio', 0.0):.4f}",
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
            get_train_dataloader(self.cfg), "gfm4mtsad train scores"
        )
        self.validation_scores, self.validation_labels = self._get_reference_outputs(
            get_val_dataloader(self.cfg), "gfm4mtsad validation scores"
        )
        self.reference_score_seconds = time.perf_counter() - reference_start

        assert len(self.test_scores) == len(self.test_labels)

        self.threshold = self._get_threshold(self.test_scores, self.test_labels)
        pred = (self.test_scores > self.threshold).astype(int)

        results = self.get_results(self.test_scores, None, self.test_labels)
        results = {name: results[name] for name in ('AUROC', 'AUPRC', 'F1')}
        self.save_results(results)
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
            score_source or self.cfg.GFM4MTSAD.EVAL.SCORE_SOURCE
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
                    f"Unsupported GFM4MTSAD reference score source: {score_source}. "
                    f"Available: {', '.join(sources.keys())}"
                )
            scores_all.append(sources[score_source])
            labels_all.append(labels.detach().flatten().cpu().numpy())
        scores = np.concatenate(scores_all, axis=0)
        labels = np.concatenate(labels_all, axis=0)
        if scores.shape[0] != labels.shape[0]:
            raise ValueError("GFM4MTSAD reference scores and labels have different lengths")
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

        for inputs in tqdm(self.test_loader, desc="gfm4mtsad test scores"):
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
        source = self._normalize_score_source(self.cfg.GFM4MTSAD.EVAL.SCORE_SOURCE)
        if source not in self.test_score_sources:
            raise ValueError(
                f"Unsupported GFM4MTSAD.EVAL.SCORE_SOURCE: {self.cfg.GFM4MTSAD.EVAL.SCORE_SOURCE}. "
                f"Available: {', '.join(self.test_score_sources.keys())}"
            )
        scores = self.test_score_sources[source]

        if len(scores) != len(labels):
            raise ValueError("GFM4MTSAD test scores and labels have different lengths")
        if node_scores.shape[0] != len(scores):
            raise ValueError("GFM4MTSAD node scores and window scores have different lengths")

        return scores, labels, node_scores

    def _build_score_sources(self, window_scores, raw_node_scores, cal_node_scores):
        sources = {"gfm4mtsad_head": window_scores, "aggregator": window_scores, "logits": window_scores}
        sources.update(self._node_score_sources("raw", raw_node_scores))
        sources.update(self._node_score_sources("cal", cal_node_scores))

        sources["node_mean"] = sources["cal_mean"]
        sources["node_max"] = sources["cal_max"]
        sources["node_topk_mean"] = sources["cal_topk"]
        sources["node_std"] = sources["cal_std"]
        return sources

    def _node_score_sources(self, prefix, node_scores):
        k = min(max(int(self.cfg.GFM4MTSAD.HEAD.TOPK), 1), node_scores.shape[1])
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
            "aggregator": "gfm4mtsad_head",
            "logits": "aggregator",
        }
        normalized = aliases.get(source, source)
        return aliases.get(normalized, normalized)

    def _get_threshold(self, test_scores, test_labels):
        if self.cfg.TEST.THRESHOLD.TYPE == "best_f1":
            return Thresholder(self.cfg, test_scores, test_labels).threshold
        return self.cfg.GFM4MTSAD.EVAL.THRESHOLD

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


    @staticmethod
    def _concat_optional(tensors):
        if not tensors:
            return None
        return torch.concat(tensors, dim=0).cpu().numpy()


    def save_to_npy(self, **kwargs):
        result_dir = mkdir(self.result_dir)
        for key, value in kwargs.items():
            np.save(os.path.join(result_dir, f"{key}.npy"), value)

    def save_results(self, results):
        text = ', '.join(f'{name}: {value:.6f}' for name, value in results.items())
        print(text, flush=True)
        (mkdir(self.result_dir) / 'test.txt').write_text(text + '\n', encoding='utf-8')

    def save_metric_protocol(self, results):
        import json
        payload = {
            'schema_version': 'GFM4MTSAD-MAIN-METRICS-V1',
            'entity': str(self.cfg.DATA.NAME),
            'score_source': str(self.cfg.GFM4MTSAD.EVAL.SCORE_SOURCE),
            'threshold': float(self.threshold),
            'metrics': {name: float(results[name]) for name in ('AUROC', 'AUPRC', 'F1')},
            'f1_threshold_uses_test_labels': True,
            'point_adjustment': False,
            'label_protocol': {
                'window_label': 'any anomaly point inside the input window',
                'window_size': int(self.cfg.DATA.WIN_SIZE),
                'test_step': int(self.cfg.DATA.TEST_STEP),
                'downsample_rate': int(self.cfg.DATA.DOWNSAMPLE_RATE),
            },
        }
        (mkdir(self.result_dir) / 'test_metric_protocol.json').write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
