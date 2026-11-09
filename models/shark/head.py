import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharkHead(nn.Module):
    def __init__(self, cfg, embedding_dim=None):
        super(SharkHead, self).__init__()
        if embedding_dim is None:
            embedding_dim = cfg.ANOMALYGFM.EMBEDDING_DIM
        hidden_dim = cfg.SHARK.HEAD.HIDDEN_DIM
        dropout = cfg.SHARK.HEAD.DROPOUT

        self.arch = cfg.SHARK.HEAD.ARCH
        self.topk = cfg.SHARK.HEAD.TOPK
        self.residual_scale = cfg.SHARK.HEAD.RESIDUAL_SCALE
        self.fusion = cfg.SHARK.HEAD.FUSION
        self.eps = cfg.SHARK.HEAD.EPS
        self.expert_names = tuple(cfg.SHARK.HEAD.EXPERT_NAMES)
        self.gate_temperature = cfg.SHARK.HEAD.GATE_TEMPERATURE
        self.moe_residual_enable = bool(
            getattr(cfg.SHARK.HEAD, "MOE_RESIDUAL_ENABLE", False)
        )
        self.moe_residual_scale = max(
            float(getattr(cfg.SHARK.HEAD, "MOE_RESIDUAL_SCALE", 0.1)), 0.0
        )
        self.moe_gate_min_weight = min(
            max(float(getattr(cfg.SHARK.HEAD, "MOE_GATE_MIN_WEIGHT", 0.0)), 0.0),
            1.0,
        )
        self.evidence_source_names = tuple(cfg.SHARK.HEAD.EVIDENCE_SOURCE_NAMES)
        self.conservative_fusion = cfg.SHARK.HEAD.CONSERVATIVE_FUSION
        self.conservative_sources = tuple(cfg.SHARK.HEAD.CONSERVATIVE_SOURCES)
        self.conservative_temperature = cfg.SHARK.HEAD.CONSERVATIVE_TEMPERATURE
        self.detach_raw_evidence = cfg.SHARK.HEAD.DETACH_RAW_EVIDENCE
        self.evidence_fusion = cfg.SHARK.HEAD.EVIDENCE_FUSION
        self.evidence_fusion_mode = cfg.SHARK.HEAD.EVIDENCE_FUSION_MODE
        self.evidence_fusion_topk = cfg.SHARK.HEAD.EVIDENCE_FUSION_TOPK
        self.mil_node_source = cfg.SHARK.HEAD.MIL_NODE_SOURCE
        self.mil_node_calibration = cfg.SHARK.HEAD.MIL_NODE_CALIBRATION
        self.mil_pooling = cfg.SHARK.HEAD.MIL_POOLING
        self.mil_temperature = cfg.SHARK.HEAD.MIL_TEMPERATURE
        self.mil_topk = cfg.SHARK.HEAD.MIL_TOPK
        self.evidence_floor_source = cfg.SHARK.HEAD.EVIDENCE_FLOOR_SOURCE
        self.evidence_floor_weight = cfg.SHARK.HEAD.EVIDENCE_FLOOR_WEIGHT
        self.decoupled_window_temperature = cfg.SHARK.HEAD.DECOUPLED_WINDOW_TEMPERATURE
        self.decoupled_window_source_weight = cfg.SHARK.HEAD.DECOUPLED_WINDOW_SOURCE_WEIGHT
        self.safe_calibrated_sources = tuple(cfg.SHARK.HEAD.SAFE_CALIBRATED_SOURCES)
        self.safe_calibrated_topk = cfg.SHARK.HEAD.SAFE_CALIBRATED_TOPK
        self.safe_calibrated_weight = cfg.SHARK.HEAD.SAFE_CALIBRATED_WEIGHT
        self.safe_calibrated_min_confidence = cfg.SHARK.HEAD.SAFE_CALIBRATED_MIN_CONFIDENCE
        self.safe_calibrated_cap = cfg.SHARK.HEAD.SAFE_CALIBRATED_CAP
        self.source_consistent_sources = tuple(cfg.SHARK.HEAD.SOURCE_CONSISTENT_SOURCES)
        self.source_consistent_topk = cfg.SHARK.HEAD.SOURCE_CONSISTENT_TOPK
        self.source_consistent_agg = cfg.SHARK.HEAD.SOURCE_CONSISTENT_AGG
        self.source_consistent_min_confidence = cfg.SHARK.HEAD.SOURCE_CONSISTENT_MIN_CONFIDENCE
        self.source_consistent_cap = cfg.SHARK.HEAD.SOURCE_CONSISTENT_CAP
        self.agreement_sources = tuple(cfg.SHARK.HEAD.AGREEMENT_SOURCES)
        self.agreement_topk = cfg.SHARK.HEAD.AGREEMENT_TOPK
        self.agreement_agg = cfg.SHARK.HEAD.AGREEMENT_AGG
        self.agreement_bonus_weight = cfg.SHARK.HEAD.AGREEMENT_BONUS_WEIGHT
        self.agreement_min_confidence = cfg.SHARK.HEAD.AGREEMENT_MIN_CONFIDENCE
        self.agreement_cap = cfg.SHARK.HEAD.AGREEMENT_CAP
        self.contrast_sources = tuple(cfg.SHARK.HEAD.CONTRAST_SOURCES)
        self.contrast_topk = cfg.SHARK.HEAD.CONTRAST_TOPK
        self.contrast_agg = cfg.SHARK.HEAD.CONTRAST_AGG
        self.contrast_weight = cfg.SHARK.HEAD.CONTRAST_WEIGHT
        self.contrast_min_confidence = cfg.SHARK.HEAD.CONTRAST_MIN_CONFIDENCE
        self.contrast_cap = cfg.SHARK.HEAD.CONTRAST_CAP
        self.relation_aware_base_source = cfg.SHARK.HEAD.RELATION_AWARE_BASE_SOURCE
        self.relation_aware_sources = tuple(cfg.SHARK.HEAD.RELATION_AWARE_SOURCES)
        self.relation_aware_weight = cfg.SHARK.HEAD.RELATION_AWARE_WEIGHT
        self.relation_aware_gate_temperature = cfg.SHARK.HEAD.RELATION_AWARE_GATE_TEMPERATURE
        self.relation_aware_margin = cfg.SHARK.HEAD.RELATION_AWARE_MARGIN
        self.relation_aware_bonus_cap = cfg.SHARK.HEAD.RELATION_AWARE_BONUS_CAP
        self.relation_aware_min_confidence = cfg.SHARK.HEAD.RELATION_AWARE_MIN_CONFIDENCE
        self.relation_aware_cap = cfg.SHARK.HEAD.RELATION_AWARE_CAP
        self.relation_aware_low_base_gate_enable = cfg.SHARK.HEAD.RELATION_AWARE_LOW_BASE_GATE_ENABLE
        self.relation_aware_low_base_threshold = cfg.SHARK.HEAD.RELATION_AWARE_LOW_BASE_THRESHOLD
        self.relation_aware_low_base_temperature = cfg.SHARK.HEAD.RELATION_AWARE_LOW_BASE_TEMPERATURE
        self.source_reliability_temperature = cfg.SHARK.HEAD.SOURCE_RELIABILITY_TEMPERATURE
        self.source_reliability_consensus_topk = cfg.SHARK.HEAD.SOURCE_RELIABILITY_CONSENSUS_TOPK
        self.source_reliability_min_weight = cfg.SHARK.HEAD.SOURCE_RELIABILITY_MIN_WEIGHT
        self.source_reliability_enable = cfg.SHARK.HEAD.SOURCE_RELIABILITY_ENABLE
        self.source_reliability_prior_weight = cfg.SHARK.HEAD.SOURCE_RELIABILITY_PRIOR_WEIGHT
        self.source_reliability_min = cfg.SHARK.HEAD.SOURCE_RELIABILITY_MIN
        self.source_reliability_max = cfg.SHARK.HEAD.SOURCE_RELIABILITY_MAX
        self.source_reliability_direct_floor = cfg.SHARK.HEAD.SOURCE_RELIABILITY_DIRECT_FLOOR
        self.normal_calibration_enable = cfg.SHARK.HEAD.NORMAL_CALIBRATION_ENABLE
        self.normal_calibration_mode = cfg.SHARK.HEAD.NORMAL_CALIBRATION_MODE
        self.normal_calibration_max_samples = cfg.SHARK.HEAD.NORMAL_CALIBRATION_MAX_SAMPLES
        self.normal_calibration_min_scale = cfg.SHARK.HEAD.NORMAL_CALIBRATION_MIN_SCALE
        self.mil_reference_scale_floor_quantile = float(
            cfg.SHARK.HEAD.MIL_REFERENCE_SCALE_FLOOR_QUANTILE
        )
        self.normal_calibration_ecdf_temperature = cfg.SHARK.HEAD.NORMAL_CALIBRATION_ECDF_TEMPERATURE

        if (
            not math.isfinite(self.mil_reference_scale_floor_quantile)
            or self.mil_reference_scale_floor_quantile < 0.0
            or self.mil_reference_scale_floor_quantile >= 1.0
        ):
            raise ValueError(
                "SHARK.HEAD.MIL_REFERENCE_SCALE_FLOOR_QUANTILE "
                "必须是 [0.0, 1.0) 内的有限数"
            )

        if self.mil_node_source != "direct":
            raise ValueError(f"Unsupported SHARK.HEAD.MIL_NODE_SOURCE: {self.mil_node_source}")
        if self.mil_node_calibration != "sensor_zscore_upper":
            raise ValueError(
                "Unsupported SHARK.HEAD.MIL_NODE_CALIBRATION: "
                f"{self.mil_node_calibration}"
            )
        if self.mil_pooling not in ("normalized_logsumexp", "topk_mean"):
            raise ValueError(f"Unsupported SHARK.HEAD.MIL_POOLING: {self.mil_pooling}")

        num_sources = len(self.conservative_sources)
        num_protocol_sources = len(self.evidence_source_names)
        num_sensors = int(cfg.DATA.N_VAR)
        self.register_buffer("evidence_reference_mean", torch.zeros(num_sources))
        self.register_buffer("evidence_reference_scale", torch.ones(num_sources))
        self.register_buffer(
            "evidence_reference_values",
            torch.zeros(int(self.normal_calibration_max_samples), num_sources),
            persistent=False,
        )
        self.register_buffer("evidence_reference_count", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("evidence_reference_fitted", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("mil_reference_mean", torch.zeros(num_sensors))
        self.register_buffer("mil_reference_scale", torch.ones(num_sensors))
        self.register_buffer("mil_reference_fitted", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("source_reliability_prior", torch.ones(num_protocol_sources))
        self.register_buffer("source_reliability_count", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("source_reliability_fitted", torch.tensor(False, dtype=torch.bool))

        self.node_head = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.direct_node_head = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.window_head = nn.Sequential(
            nn.Linear(embedding_dim + 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.attention_head = nn.Linear(embedding_dim, 1)
        self.spike_head = self._make_expert(6, cfg)
        self.distribution_head = self._make_expert(6, cfg)
        self.graph_context_head = nn.Sequential(
            nn.Linear(embedding_dim + 12, cfg.SHARK.HEAD.EXPERT_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.SHARK.HEAD.EXPERT_HIDDEN_DIM, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(embedding_dim + 12, cfg.SHARK.HEAD.GATE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.SHARK.HEAD.GATE_HIDDEN_DIM, len(self.expert_names)),
        )
        self.conservative_gate = nn.Sequential(
            nn.Linear(embedding_dim + 12, cfg.SHARK.HEAD.GATE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.SHARK.HEAD.GATE_HIDDEN_DIM, len(self.conservative_sources)),
        )

    def forward(self, node_embeddings, raw_node_scores, graph_embedding):
        self._check_shapes(node_embeddings, raw_node_scores, graph_embedding)
        if self.arch != "moe":
            raise ValueError(f"Unsupported SHARK.HEAD.ARCH: {self.arch}")

        raw_log = torch.log(raw_node_scores.clamp_min(self.eps))
        residual = torch.tanh(
            self.node_head(torch.cat([node_embeddings, raw_log.unsqueeze(-1)], dim=-1)).squeeze(-1)
        )
        direct_residual = torch.tanh(
            self.direct_node_head(torch.cat([node_embeddings, raw_log.unsqueeze(-1)], dim=-1)).squeeze(-1)
        )
        node_score_logits = raw_log + self.residual_scale * residual
        node_scores = node_score_logits
        direct_node_score_logits = raw_log + self.residual_scale * direct_residual
        direct_node_scores = direct_node_score_logits
        mil_direct_raw = direct_node_score_logits
        mil_sensor_risk = self._calibrate_mil_node_scores(mil_direct_raw)
        mil_lse_score, mil_pooling_weights = self._pool_mil_sensor_risk(
            mil_sensor_risk,
            pooling="normalized_logsumexp",
        )
        mil_window_score = mil_lse_score
        mil_topk_diagnostic_score, _ = self._pool_mil_sensor_risk(
            mil_sensor_risk.detach(),
            pooling="topk_mean",
        )
        mil_topk_diagnostic_score = mil_topk_diagnostic_score.detach()
        affected_node_score_logits = node_score_logits
        affected_node_scores = node_scores

        raw_stats = self._score_stats(raw_node_scores)
        raw_log_stats = self._score_stats(raw_log)
        cal_stats = self._score_stats(node_scores)
        direct_stats = self._score_stats(direct_node_scores)
        raw_stats_evidence = raw_stats.detach() if self.detach_raw_evidence else raw_stats
        raw_log_stats_evidence = raw_log_stats.detach() if self.detach_raw_evidence else raw_log_stats

        node_scale = node_scores.new_tensor(node_scores.size(1)).sqrt()
        raw_log_std_energy = raw_log_stats[:, 3] * node_scale
        raw_log_std_energy_evidence = (
            raw_log_std_energy.detach() if self.detach_raw_evidence else raw_log_std_energy
        )
        cal_std_energy = cal_stats[:, 3] * node_scale

        cal_gap = cal_stats[:, 1] - cal_stats[:, 2]
        raw_log_gap = raw_log_stats_evidence[:, 1] - raw_log_stats_evidence[:, 2]
        spike_features = torch.stack(
            [
                raw_log_stats_evidence[:, 1],
                raw_log_stats_evidence[:, 2],
                raw_log_gap,
                cal_stats[:, 1],
                cal_stats[:, 2],
                cal_gap,
            ],
            dim=1,
        )
        spike_base = 0.5 * (raw_log_stats_evidence[:, 2] + cal_stats[:, 2])
        spike_score = spike_base + self.residual_scale * self.spike_head(spike_features).squeeze(-1)

        cal_deviation = (node_scores - node_scores.mean(dim=1, keepdim=True)).abs()
        raw_log_for_evidence = raw_log.detach() if self.detach_raw_evidence else raw_log
        raw_log_deviation = (raw_log_for_evidence - raw_log_for_evidence.mean(dim=1, keepdim=True)).abs()
        cal_dev_topk = cal_deviation.topk(min(max(int(self.topk), 1), node_scores.size(1)), dim=1).values.mean(dim=1)
        raw_log_dev_topk = raw_log_deviation.topk(min(max(int(self.topk), 1), raw_log.size(1)), dim=1).values.mean(dim=1)
        distribution_features = torch.stack(
            [
                raw_log_stats_evidence[:, 3],
                raw_log_std_energy_evidence,
                raw_log_dev_topk,
                cal_stats[:, 3],
                cal_std_energy,
                cal_dev_topk,
            ],
            dim=1,
        )
        distribution_base = 0.5 * (raw_log_std_energy_evidence + cal_std_energy)
        distribution_score = distribution_base + self.residual_scale * self.distribution_head(distribution_features).squeeze(-1)

        node_summary = torch.cat([raw_stats_evidence, raw_log_stats_evidence, cal_stats], dim=-1)
        graph_context_features = torch.cat([graph_embedding, node_summary], dim=-1)
        graph_context_score = self.graph_context_head(graph_context_features).squeeze(-1)
        head_features = torch.cat([graph_embedding, raw_stats_evidence, cal_stats], dim=-1)
        head_score = self.window_head(head_features).squeeze(-1)

        expert_scores = torch.stack([spike_score, distribution_score, graph_context_score], dim=1)
        gate_logits = self.gate_head(graph_context_features)
        gate_weights = torch.softmax(gate_logits / max(float(self.gate_temperature), self.eps), dim=1)
        if self.moe_gate_min_weight > 0.0:
            gate_weights = (1.0 - self.moe_gate_min_weight) * gate_weights
            gate_weights = gate_weights + self.moe_gate_min_weight / gate_weights.size(1)
        if self.fusion:
            moe_score = torch.sum(gate_weights * expert_scores, dim=1)
        else:
            gate_weights = torch.zeros_like(gate_weights)
            gate_weights[:, 2] = 1.0
            moe_score = graph_context_score

        attention_weights = torch.softmax(
            self.attention_head(node_embeddings).squeeze(-1),
            dim=1,
        )
        direct_sensor_score = self._minmax(direct_node_scores)
        local_sensor_score = self._minmax(node_scores)
        relation_sensor_score = self._minmax(cal_deviation)
        context_sensor_score = self._minmax(attention_weights)
        evidence_sensor_scores = {
            "direct": direct_sensor_score,
            "local": local_sensor_score,
            "relation": relation_sensor_score,
            "context": context_sensor_score,
        }
        evidence_values = {
            "direct": direct_stats[:, 2],
            "local": cal_stats[:, 0],
            "relation": cal_std_energy,
            "context": graph_context_score,
            "moe": moe_score,
            "head": head_score,
            "raw_mean": raw_stats[:, 0],
            "raw_max": raw_stats[:, 1],
            "raw_topk": raw_stats[:, 2],
            "raw_std": raw_stats[:, 3],
            "raw_log_max": raw_log_stats[:, 1],
            "raw_log_topk": raw_log_stats[:, 2],
            "raw_log_std": raw_log_stats[:, 3],
            "raw_log_std_energy": raw_log_std_energy,
            "cal_mean": cal_stats[:, 0],
            "cal_max": cal_stats[:, 1],
            "cal_topk": cal_stats[:, 2],
            "cal_std": cal_stats[:, 3],
            "cal_std_energy": cal_std_energy,
            "direct_mean": direct_stats[:, 0],
            "direct_max": direct_stats[:, 1],
            "direct_topk": direct_stats[:, 2],
            "direct_std": direct_stats[:, 3],
        }
        if self.detach_raw_evidence:
            evidence_values = {
                name: value.detach() if self._is_raw_evidence(name) else value
                for name, value in evidence_values.items()
            }
        conservative_scores_raw = self._conservative_scores(self.conservative_sources, evidence_values)
        conservative_scores = self._normal_calibrate_evidence(conservative_scores_raw)
        safe_calibrated_score = self._safe_calibrated_score(conservative_scores_raw, conservative_scores)
        source_consistent_score = self._source_consistent_score(conservative_scores_raw, conservative_scores)
        reliable_source_score, reliable_source_weights, reliable_source_names = self._reliable_source_consistent_score(
            conservative_scores_raw,
            conservative_scores,
        )
        normal_ref_agreement_score, evidence_base_risk, evidence_agreement_score, evidence_disagreement_score = (
            self._normal_referenced_agreement_score(
                self.conservative_sources,
                conservative_scores_raw,
                conservative_scores,
            )
        )
        normal_ref_contrast_score, evidence_contrast_base, evidence_contrast_score = (
            self._normal_referenced_contrast_score(
                self.conservative_sources,
                conservative_scores_raw,
                conservative_scores,
            )
        )
        (
            relation_aware_normal_ref_score,
            relation_aware_base_score,
            relation_aware_relation_evidence,
            relation_aware_relation_gate,
            relation_aware_low_base_gate,
            relation_aware_relation_bonus,
        ) = self._relation_aware_normal_ref_score(
            self.conservative_sources,
            conservative_scores_raw,
            conservative_scores,
            normal_ref_contrast_score,
            evidence_contrast_score,
        )
        conservative_logits = self.conservative_gate(graph_context_features)
        conservative_weights = torch.softmax(
            conservative_logits / max(float(self.conservative_temperature), self.eps),
            dim=1,
        )
        if self.conservative_fusion:
            window_score = self._fuse_evidence_scores(
                conservative_scores,
                conservative_weights,
                evidence_values,
                conservative_scores_raw,
                safe_calibrated_score,
                source_consistent_score,
                reliable_source_score,
                normal_ref_agreement_score,
                normal_ref_contrast_score,
                relation_aware_normal_ref_score,
                mil_window_score,
            )
        else:
            conservative_weights = torch.zeros_like(conservative_weights)
            conservative_weights[:, 0] = 1.0
            window_score = moe_score
        if self.evidence_fusion_mode == "counterfactual_mil":
            window_score = mil_window_score
        # Preserve counterfactual-MIL as the canonical backbone.  The
        # optional MoE path is only a bounded residual correction, so it can
        # be enabled for a candidate training arm without replacing the
        # calibrated node-risk route.
        if self.moe_residual_enable:
            moe_residual_score = self.moe_residual_scale * torch.tanh(moe_score)
            window_score = window_score + moe_residual_score
        else:
            moe_residual_score = torch.zeros_like(moe_score)
        decoupled_window_score = self._decoupled_window_score(
            evidence_values["head"],
            source_consistent_score,
        )

        protocol_scores_raw = self._conservative_scores(self.evidence_source_names, evidence_values)
        protocol_scores_calibrated = self._normal_calibrate_selected_evidence(
            self.evidence_source_names,
            protocol_scores_raw,
        )
        protocol_weights, protocol_weight_names = self._project_source_weights(
            reliable_source_weights,
            reliable_source_names,
            self.evidence_source_names,
            fallback_scores=protocol_scores_calibrated,
        )
        source_reliability_prior = self._source_reliability_for_names(
            self.evidence_source_names,
            protocol_weights.device,
            protocol_weights.dtype,
        )
        source_reliability_prior_batch = source_reliability_prior.unsqueeze(0).expand(
            protocol_weights.size(0),
            -1,
        )
        sensor_score = protocol_weights.new_zeros(direct_node_scores.shape)
        for idx, name in enumerate(self.evidence_source_names):
            sensor_score = sensor_score + protocol_weights[:, idx:idx + 1] * evidence_sensor_scores[name]
        if self.evidence_fusion_mode == "counterfactual_mil":
            sensor_score = mil_sensor_risk
        sensor_importance = sensor_score

        return {
            "node_scores": node_scores,
            "node_score_logits": node_score_logits,
            "node_scores_calibrated": node_scores,
            "direct_node_scores": direct_node_scores,
            "direct_node_score_logits": direct_node_score_logits,
            "root_node_scores": direct_node_scores,
            "root_node_score_logits": direct_node_score_logits,
            "affected_node_scores": affected_node_scores,
            "affected_node_score_logits": affected_node_score_logits,
            "raw_node_scores": raw_node_scores,
            "head_residual": residual,
            "direct_head_residual": direct_residual,
            "root_head_residual": direct_residual,
            "bridge_residual": residual,
            "head_score": head_score,
            "window_score": window_score,
            "moe_score": moe_score,
            "moe_residual_score": moe_residual_score,
            "moe_residual_enable": self.moe_residual_enable,
            "moe_residual_scale": self.moe_residual_scale,
            "mil_window_score": mil_window_score,
            "mil_lse_score": mil_lse_score,
            "mil_topk_diagnostic_score": mil_topk_diagnostic_score,
            "mil_sensor_risk": mil_sensor_risk,
            "mil_pooling_weights": mil_pooling_weights,
            "mil_direct_raw": mil_direct_raw,
            "mil_reference_fitted": self.mil_reference_fitted.detach().clone(),
            "decoupled_window_score": decoupled_window_score,
            "normal_ref_agreement_score": normal_ref_agreement_score,
            "normal_ref_contrast_score": normal_ref_contrast_score,
            "relation_aware_normal_ref_score": relation_aware_normal_ref_score,
            "relation_aware_base_score": relation_aware_base_score,
            "relation_aware_relation_evidence": relation_aware_relation_evidence,
            "relation_aware_relation_gate": relation_aware_relation_gate,
            "relation_aware_low_base_gate": relation_aware_low_base_gate,
            "relation_aware_relation_bonus": relation_aware_relation_bonus,
            "evidence_base_risk": evidence_base_risk,
            "evidence_contrast_base": evidence_contrast_base,
            "evidence_contrast_score": evidence_contrast_score,
            "evidence_agreement_score": evidence_agreement_score,
            "evidence_disagreement_score": evidence_disagreement_score,
            "direct_evidence_score": evidence_values["direct"],
            "local_evidence_score": evidence_values["local"],
            "relation_evidence_score": evidence_values["relation"],
            "context_evidence_score": evidence_values["context"],
            "evidence_protocol_scores_raw": protocol_scores_raw,
            "evidence_protocol_scores": protocol_scores_calibrated,
            "evidence_protocol_weights": protocol_weights,
            "evidence_protocol_names": self.evidence_source_names,
            "source_reliability_prior": source_reliability_prior.detach().clone(),
            "source_reliability_fitted": self.source_reliability_fitted.detach().clone(),
            "source_reliability_names": self.evidence_source_names,
            "sensor_score": sensor_score,
            "sensor_importance": sensor_importance,
            "sensor_importance_components": {
                "direct": direct_sensor_score,
                "local": local_sensor_score,
                "relation": relation_sensor_score,
                "context": context_sensor_score,
                "spike": local_sensor_score,
                "distribution": relation_sensor_score,
                "graph_context": context_sensor_score,
            },
            "attention_weights": attention_weights,
            "expert_scores": expert_scores,
            "gate_logits": gate_logits,
            "gate_weights": gate_weights,
            "fusion_weights": gate_weights,
            "conservative_scores": conservative_scores,
            "conservative_scores_raw": conservative_scores_raw,
            "safe_calibrated_score": safe_calibrated_score,
            "source_consistent_score": source_consistent_score,
            "reliable_source_score": reliable_source_score,
            "reliable_source_weights": reliable_source_weights,
            "reliable_source_names": reliable_source_names,
            "conservative_logits": conservative_logits,
            "conservative_weights": conservative_weights,
            "conservative_source_names": self.conservative_sources,
            "evidence_reference_fitted": self.evidence_reference_fitted.detach().clone(),
            "evidence_fusion_mode": self.evidence_fusion_mode,
            "evidence_floor_source": self.evidence_floor_source,
            "expert_names": self.expert_names,
            "fusion_source_names": self.expert_names,
            "score_breakdown": {
                "head_score": head_score,
                "mil_window_score": mil_window_score,
                "mil_lse_score": mil_lse_score,
                "mil_topk_diagnostic_score": mil_topk_diagnostic_score,
                "moe_score": moe_score,
                "moe_residual_score": moe_residual_score,
                "moe_spike": spike_score,
                "moe_distribution": distribution_score,
                "moe_graph_context": graph_context_score,
                "raw_mean": raw_stats[:, 0],
                "raw_max": raw_stats[:, 1],
                "raw_topk": raw_stats[:, 2],
                "raw_std": raw_stats[:, 3],
                "raw_log_mean": raw_log_stats[:, 0],
                "raw_log_max": raw_log_stats[:, 1],
                "raw_log_topk": raw_log_stats[:, 2],
                "raw_log_std": raw_log_stats[:, 3],
                "raw_log_std_energy": raw_log_std_energy,
                "cal_mean": cal_stats[:, 0],
                "cal_max": cal_stats[:, 1],
                "cal_topk": cal_stats[:, 2],
                "cal_std": cal_stats[:, 3],
                "cal_std_energy": cal_std_energy,
                "direct_mean": direct_stats[:, 0],
                "direct_max": direct_stats[:, 1],
                "direct_topk": direct_stats[:, 2],
                "direct_std": direct_stats[:, 3],
                "direct_evidence_score": evidence_values["direct"],
                "local_evidence_score": evidence_values["local"],
                "relation_evidence_score": evidence_values["relation"],
                "context_evidence_score": evidence_values["context"],
                **self._prefixed_evidence_breakdown(
                    "protocol_raw",
                    self.evidence_source_names,
                    protocol_scores_raw,
                ),
                **self._prefixed_evidence_breakdown(
                    "protocol_cal",
                    self.evidence_source_names,
                    protocol_scores_calibrated,
                ),
                **self._prefixed_evidence_breakdown(
                    "protocol_weight",
                    protocol_weight_names,
                    protocol_weights,
                ),
                **self._prefixed_evidence_breakdown(
                    "source_reliability_prior",
                    self.evidence_source_names,
                    source_reliability_prior_batch,
                ),
                **self._prefixed_evidence_breakdown(
                    "evidence_raw",
                    self.conservative_sources,
                    conservative_scores_raw,
                ),
                **self._prefixed_evidence_breakdown(
                    "evidence_cal",
                    self.conservative_sources,
                    conservative_scores,
                ),
                "safe_calibrated_score": safe_calibrated_score,
                "source_consistent_score": source_consistent_score,
                "reliable_source_score": reliable_source_score,
                "decoupled_window_score": decoupled_window_score,
                "normal_ref_agreement_score": normal_ref_agreement_score,
                "normal_ref_contrast_score": normal_ref_contrast_score,
                "relation_aware_normal_ref_score": relation_aware_normal_ref_score,
                "relation_aware_base_score": relation_aware_base_score,
                "relation_aware_relation_evidence": relation_aware_relation_evidence,
                "relation_aware_relation_gate": relation_aware_relation_gate,
                "relation_aware_low_base_gate": relation_aware_low_base_gate,
                "relation_aware_relation_bonus": relation_aware_relation_bonus,
                "evidence_base_risk": evidence_base_risk,
                "evidence_contrast_base": evidence_contrast_base,
                "evidence_contrast_score": evidence_contrast_score,
                "evidence_agreement_score": evidence_agreement_score,
                "evidence_disagreement_score": evidence_disagreement_score,
                "window_score": window_score,
                **self._prefixed_evidence_breakdown(
                    "reliability_weight",
                    reliable_source_names,
                    reliable_source_weights,
                ),
            },
        }

    def _make_expert(self, input_dim, cfg):
        return nn.Sequential(
            nn.Linear(input_dim, cfg.SHARK.HEAD.EXPERT_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.SHARK.HEAD.DROPOUT),
            nn.Linear(cfg.SHARK.HEAD.EXPERT_HIDDEN_DIM, 1),
        )

    def _score_stats(self, scores):
        k = min(max(int(self.topk), 1), scores.size(1))
        return torch.stack(
            [
                scores.mean(dim=1),
                scores.max(dim=1).values,
                scores.topk(k, dim=1).values.mean(dim=1),
                scores.std(dim=1, unbiased=False),
            ],
            dim=1,
        )

    def _minmax(self, values):
        min_value = values.min(dim=1, keepdim=True).values
        max_value = values.max(dim=1, keepdim=True).values
        return (values - min_value) / (max_value - min_value).clamp_min(self.eps)

    @staticmethod
    def _conservative_scores(source_names, values):
        missing = [name for name in source_names if name not in values]
        if missing:
            raise ValueError(f"Unsupported SHARK.HEAD.CONSERVATIVE_SOURCES: {missing}")
        return torch.stack([values[name] for name in source_names], dim=1)

    def _fuse_evidence_scores(
        self,
        conservative_scores,
        conservative_weights,
        evidence_values,
        conservative_scores_raw,
        safe_calibrated_score,
        source_consistent_score,
        reliable_source_score,
        normal_ref_agreement_score,
        normal_ref_contrast_score,
        relation_aware_normal_ref_score,
        mil_window_score,
    ):
        learned_score = torch.sum(conservative_weights * conservative_scores, dim=1)
        if not self.evidence_fusion:
            return learned_score

        if self.evidence_fusion_mode == "softmax":
            return learned_score

        if self.evidence_fusion_mode == "softmax_floor":
            if self.evidence_floor_source not in evidence_values:
                raise ValueError(f"Unsupported SHARK.HEAD.EVIDENCE_FLOOR_SOURCE: {self.evidence_floor_source}")
            floor_weight = min(max(float(self.evidence_floor_weight), 0.0), 1.0)
            if self.evidence_floor_source in self.conservative_sources:
                floor_idx = self.conservative_sources.index(self.evidence_floor_source)
                floor_score = conservative_scores[:, floor_idx]
            else:
                floor_score = evidence_values[self.evidence_floor_source]
            return (1.0 - floor_weight) * learned_score + floor_weight * floor_score

        if self.evidence_fusion_mode == "mean":
            return conservative_scores.mean(dim=1)

        if self.evidence_fusion_mode == "topk_mean":
            k = min(max(int(self.evidence_fusion_topk), 1), conservative_scores.size(1))
            return conservative_scores.topk(k, dim=1).values.mean(dim=1)

        if self.evidence_fusion_mode == "max":
            return conservative_scores.max(dim=1).values

        if self.evidence_fusion_mode == "safe_calibrated":
            return safe_calibrated_score

        if self.evidence_fusion_mode == "source_consistent_calibrated":
            return source_consistent_score

        if self.evidence_fusion_mode == "reliable_source_consistent":
            return reliable_source_score

        if self.evidence_fusion_mode == "normal_referenced_agreement":
            return normal_ref_agreement_score

        if self.evidence_fusion_mode == "normal_referenced_contrast":
            return normal_ref_contrast_score

        if self.evidence_fusion_mode == "relation_aware_normal_ref":
            return relation_aware_normal_ref_score

        if self.evidence_fusion_mode == "counterfactual_mil":
            return mil_window_score

        if self.evidence_fusion_mode == "decoupled_conservative":
            return self._decoupled_window_score(
                evidence_values["head"],
                source_consistent_score,
            )

        raise ValueError(f"Unsupported SHARK.HEAD.EVIDENCE_FUSION_MODE: {self.evidence_fusion_mode}")

    def _calibrate_mil_node_scores(self, node_scores):
        if node_scores.dim() != 2:
            raise ValueError(
                f"MIL node scores must have shape [B, N], got {tuple(node_scores.shape)}"
            )
        if node_scores.size(1) != self.mil_reference_mean.numel():
            raise ValueError(
                "MIL node count must match DATA.N_VAR: "
                f"scores={node_scores.size(1)}, reference={self.mil_reference_mean.numel()}"
            )
        if not bool(self.mil_reference_fitted):
            return node_scores

        mean = self.mil_reference_mean.to(device=node_scores.device, dtype=node_scores.dtype)
        scale = self.mil_reference_scale.to(device=node_scores.device, dtype=node_scores.dtype)
        effective_scale, _, _ = self._effective_mil_reference_scale(scale)
        return (node_scores - mean.unsqueeze(0)) / effective_scale.unsqueeze(0)

    def _effective_mil_reference_scale(self, scale=None):
        """在公共读取点组合旧最小尺度与可选的 MRSF 分位数地板。"""
        if scale is None:
            scale = self.mil_reference_scale
        if scale.dim() != 1 or scale.numel() != self.mil_reference_scale.numel():
            raise ValueError(
                "MIL reference scale 必须是一维 DATA.N_VAR 向量: "
                f"got={tuple(scale.shape)}"
            )
        base_scale = scale.clamp_min(
            float(self.normal_calibration_min_scale)
        )
        quantile = float(self.mil_reference_scale_floor_quantile)
        if quantile == 0.0:
            return base_scale, None, torch.zeros_like(base_scale, dtype=torch.bool)

        if (
            not bool(torch.isfinite(scale).all().item())
            or bool((scale <= 0.0).any().item())
        ):
            raise ValueError("启用 MRSF 时 MIL reference scale 必须全部为有限正数")

        floor_value = torch.quantile(scale, quantile)
        effective_scale = base_scale.clamp_min(floor_value)
        affected_mask = effective_scale > base_scale
        return effective_scale, floor_value, affected_mask

    def _pool_mil_sensor_risk(self, sensor_risk, pooling=None):
        if sensor_risk.dim() != 2 or sensor_risk.size(1) < 1:
            raise ValueError(
                f"MIL sensor risk must have shape [B, N] with N >= 1, got {tuple(sensor_risk.shape)}"
            )

        pooling = self.mil_pooling if pooling is None else pooling
        if pooling == "normalized_logsumexp":
            temperature = max(float(self.mil_temperature), self.eps)
            node_count = sensor_risk.size(1)
            window_risk = temperature * torch.logsumexp(
                sensor_risk / temperature,
                dim=1,
            ) - temperature * sensor_risk.new_tensor(float(node_count)).log()
            pooling_weights = torch.softmax(sensor_risk / temperature, dim=1)
            return window_risk, pooling_weights

        if pooling == "topk_mean":
            k = min(max(int(self.mil_topk), 1), sensor_risk.size(1))
            topk_values, topk_indices = sensor_risk.topk(k, dim=1)
            pooling_weights = torch.zeros_like(sensor_risk)
            pooling_weights.scatter_(1, topk_indices, 1.0 / float(k))
            return topk_values.mean(dim=1), pooling_weights

        raise ValueError(f"Unsupported SHARK.HEAD.MIL_POOLING: {pooling}")

    def _decoupled_window_score(self, head_score, source_consistent_score):
        source_weight = max(float(self.decoupled_window_source_weight), 0.0)
        source_score = source_consistent_score * source_weight
        temperature = max(float(self.decoupled_window_temperature), self.eps)
        stacked = torch.stack([head_score, source_score], dim=1)
        return temperature * torch.logsumexp(stacked / temperature, dim=1)

    def _normal_referenced_agreement_score(self, source_names, raw_scores, calibrated_scores):
        source_indices = [
            source_names.index(source)
            for source in self.agreement_sources
            if source in source_names
        ]
        if not source_indices:
            source_indices = list(range(raw_scores.size(1)))

        scores = calibrated_scores if self.normal_calibration_enable and bool(self.evidence_reference_fitted) else raw_scores
        selected = scores[:, source_indices]
        selected = F.relu(selected - float(self.agreement_min_confidence))
        selected = selected.clamp(max=float(self.agreement_cap))

        base_risk = self._aggregate_source_scores(selected, self.agreement_topk, self.agreement_agg)
        disagreement = selected.std(dim=1, unbiased=False)
        agreement = 1.0 / (1.0 + disagreement)
        bonus = float(self.agreement_bonus_weight) * agreement * base_risk
        return base_risk + bonus, base_risk, agreement, disagreement

    def _normal_referenced_contrast_score(self, source_names, raw_scores, calibrated_scores):
        source_indices = [
            source_names.index(source)
            for source in self.contrast_sources
            if source in source_names
        ]
        if not source_indices:
            source_indices = list(range(raw_scores.size(1)))

        scores = calibrated_scores if self.normal_calibration_enable and bool(self.evidence_reference_fitted) else raw_scores
        selected = scores[:, source_indices]
        selected = F.relu(selected - float(self.contrast_min_confidence))
        selected = selected.clamp(max=float(self.contrast_cap))

        base_risk = self._aggregate_source_scores(selected, self.contrast_topk, self.contrast_agg)
        contrast = selected.std(dim=1, unbiased=False)
        score = base_risk + float(self.contrast_weight) * contrast
        return score, base_risk, contrast

    def _relation_aware_normal_ref_score(
        self,
        source_names,
        raw_scores,
        calibrated_scores,
        normal_ref_contrast_score,
        evidence_contrast_score,
    ):
        source_indices = [
            source_names.index(source)
            for source in self.relation_aware_sources
            if source in source_names
        ]
        if not source_indices and "relation" in source_names:
            source_indices = [source_names.index("relation")]
        if not source_indices:
            base_score = self._relation_aware_base_score(
                normal_ref_contrast_score,
                evidence_contrast_score,
            )
            zeros = torch.zeros_like(base_score)
            return base_score, base_score, zeros, zeros, zeros, zeros

        scores = calibrated_scores if self.normal_calibration_enable and bool(self.evidence_reference_fitted) else raw_scores
        selected = scores[:, source_indices]
        selected = F.relu(selected - float(self.relation_aware_min_confidence))
        selected = selected.clamp(max=float(self.relation_aware_cap))

        relation_evidence = selected.max(dim=1).values
        base_score = self._relation_aware_base_score(
            normal_ref_contrast_score,
            evidence_contrast_score,
        )
        rescue_signal = F.relu(
            relation_evidence - base_score.detach() - float(self.relation_aware_margin)
        )
        temperature = max(float(self.relation_aware_gate_temperature), self.eps)
        relation_gate = 1.0 - torch.exp(-rescue_signal / temperature)
        if self.relation_aware_low_base_gate_enable:
            low_base_temperature = max(
                float(self.relation_aware_low_base_temperature),
                self.eps,
            )
            low_base_gate = torch.sigmoid(
                (float(self.relation_aware_low_base_threshold) - base_score.detach())
                / low_base_temperature
            )
        else:
            low_base_gate = torch.ones_like(base_score)
        gate = relation_gate * low_base_gate
        relation_bonus = float(self.relation_aware_weight) * gate * rescue_signal
        relation_bonus = relation_bonus.clamp(max=float(self.relation_aware_bonus_cap))
        return (
            base_score + relation_bonus,
            base_score,
            relation_evidence,
            gate,
            low_base_gate,
            relation_bonus,
        )

    def _relation_aware_base_score(self, normal_ref_contrast_score, evidence_contrast_score):
        if self.relation_aware_base_source == "normal_ref_contrast":
            return normal_ref_contrast_score
        if self.relation_aware_base_source == "evidence_contrast":
            return evidence_contrast_score
        if self.relation_aware_base_source == "max":
            return torch.maximum(normal_ref_contrast_score, evidence_contrast_score)
        raise ValueError(
            "Unsupported SHARK.HEAD.RELATION_AWARE_BASE_SOURCE: "
            f"{self.relation_aware_base_source}"
        )

    def _safe_calibrated_score(self, raw_scores, calibrated_scores):
        if self.evidence_floor_source in self.conservative_sources:
            base_idx = self.conservative_sources.index(self.evidence_floor_source)
            base_score = raw_scores[:, base_idx]
        else:
            raise ValueError(f"Unsupported SHARK.HEAD.EVIDENCE_FLOOR_SOURCE: {self.evidence_floor_source}")

        if not self.normal_calibration_enable or not bool(self.evidence_reference_fitted):
            return base_score

        source_indices = [
            self.conservative_sources.index(source)
            for source in self.safe_calibrated_sources
            if source in self.conservative_sources
        ]
        if not source_indices:
            return base_score

        selected = calibrated_scores[:, source_indices]
        k = min(max(int(self.safe_calibrated_topk), 1), selected.size(1))
        confidence = selected.topk(k, dim=1).values.mean(dim=1)
        confidence = F.relu(confidence - float(self.safe_calibrated_min_confidence))
        confidence = confidence.clamp(max=float(self.safe_calibrated_cap))
        return base_score + float(self.safe_calibrated_weight) * confidence

    def _source_consistent_score(self, raw_scores, calibrated_scores):
        source_indices = [
            self.conservative_sources.index(source)
            for source in self.source_consistent_sources
            if source in self.conservative_sources
        ]
        if not source_indices:
            return self._aggregate_source_scores(raw_scores, raw_scores.size(1), self.source_consistent_agg)

        scores = calibrated_scores if self.normal_calibration_enable and bool(self.evidence_reference_fitted) else raw_scores
        selected = scores[:, source_indices]
        selected = F.relu(selected - float(self.source_consistent_min_confidence))
        selected = selected.clamp(max=float(self.source_consistent_cap))
        return self._aggregate_source_scores(selected, self.source_consistent_topk, self.source_consistent_agg)

    def _reliable_source_consistent_score(self, raw_scores, calibrated_scores):
        source_indices = [
            self.conservative_sources.index(source)
            for source in self.source_consistent_sources
            if source in self.conservative_sources
        ]
        if not source_indices:
            source_indices = list(range(raw_scores.size(1)))

        scores = calibrated_scores if self.normal_calibration_enable and bool(self.evidence_reference_fitted) else raw_scores
        selected = scores[:, source_indices]
        selected = F.relu(selected - float(self.source_consistent_min_confidence))
        selected = selected.clamp(max=float(self.source_consistent_cap))

        consensus_k = min(max(int(self.source_reliability_consensus_topk), 1), selected.size(1))
        consensus = selected.topk(consensus_k, dim=1).values.mean(dim=1, keepdim=True)
        deviation = (selected - consensus).abs()
        temperature = max(float(self.source_reliability_temperature), self.eps)
        reliability_logits = selected - deviation / temperature
        reliability_weights = torch.softmax(reliability_logits, dim=1)

        min_weight = min(max(float(self.source_reliability_min_weight), 0.0), 1.0)
        if min_weight > 0.0:
            reliability_weights = (1.0 - min_weight) * reliability_weights + min_weight / selected.size(1)

        source_names = tuple(self.conservative_sources[idx] for idx in source_indices)
        reliability_weights = self._apply_source_reliability_prior(reliability_weights, source_names)

        score = torch.sum(reliability_weights * selected, dim=1)
        return score, reliability_weights, source_names

    def _normal_calibrate_selected_evidence(self, source_names, scores):
        if tuple(source_names) == self.conservative_sources:
            return self._normal_calibrate_evidence(scores)
        if not self.normal_calibration_enable or not bool(self.evidence_reference_fitted):
            return scores

        indices = []
        for name in source_names:
            if name not in self.conservative_sources:
                return scores
            indices.append(self.conservative_sources.index(name))

        full_scores = scores.new_zeros(scores.size(0), len(self.conservative_sources))
        full_scores[:, indices] = scores
        calibrated = self._normal_calibrate_evidence(full_scores)
        return calibrated[:, indices]

    def _project_source_weights(self, source_weights, source_names, target_names, fallback_scores):
        target_names = tuple(target_names)
        if source_weights is None or source_names is None:
            weights = torch.softmax(fallback_scores, dim=1)
            weights = self._apply_source_reliability_prior(weights, target_names)
            weights = self._apply_direct_weight_floor(weights, target_names)
            return weights, target_names

        weights = source_weights.new_zeros(source_weights.size(0), len(target_names))
        source_index = {name: idx for idx, name in enumerate(source_names)}
        for target_idx, name in enumerate(target_names):
            if name in source_index:
                weights[:, target_idx] = source_weights[:, source_index[name]]

        row_sum = weights.sum(dim=1, keepdim=True)
        fallback = torch.softmax(fallback_scores, dim=1)
        fallback = self._apply_source_reliability_prior(fallback, target_names)
        weights = torch.where(row_sum > self.eps, weights / row_sum.clamp_min(self.eps), fallback)
        weights = self._apply_direct_weight_floor(weights, target_names)
        return weights, target_names

    def _apply_source_reliability_prior(self, weights, source_names):
        if not self.source_reliability_enable or not bool(self.source_reliability_fitted):
            return weights

        prior_weight = min(max(float(self.source_reliability_prior_weight), 0.0), 1.0)
        if prior_weight <= 0.0:
            return weights

        prior = self._source_reliability_for_names(source_names, weights.device, weights.dtype)
        prior = (1.0 - prior_weight) + prior_weight * prior
        weights = weights * prior.unsqueeze(0)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)

    def _apply_direct_weight_floor(self, weights, source_names):
        if "direct" not in source_names:
            return weights

        floor = min(max(float(self.source_reliability_direct_floor), 0.0), 1.0)
        if floor <= 0.0:
            return weights

        direct_idx = tuple(source_names).index("direct")
        direct = weights[:, direct_idx:direct_idx + 1]
        if weights.size(1) == 1:
            return weights

        other = weights.clone()
        other[:, direct_idx] = 0.0
        other_sum = other.sum(dim=1, keepdim=True).clamp_min(self.eps)
        other = other / other_sum
        adjusted = other * (1.0 - torch.maximum(direct, direct.new_full(direct.shape, floor)))
        adjusted[:, direct_idx:direct_idx + 1] = torch.maximum(
            direct,
            direct.new_full(direct.shape, floor),
        )
        return adjusted / adjusted.sum(dim=1, keepdim=True).clamp_min(self.eps)

    def _source_reliability_for_names(self, source_names, device, dtype):
        prior = torch.ones(len(source_names), device=device, dtype=dtype)
        source_index = {name: idx for idx, name in enumerate(self.evidence_source_names)}
        fitted_prior = self.source_reliability_prior.to(device=device, dtype=dtype)
        for idx, name in enumerate(source_names):
            if name in source_index:
                prior[idx] = fitted_prior[source_index[name]]
        return prior.clamp(
            min=float(self.source_reliability_min),
            max=float(self.source_reliability_max),
        )

    @torch.no_grad()
    def set_source_reliability(self, prior):
        prior = self._normalize_source_reliability(prior)
        prior = prior.to(device=self.source_reliability_prior.device, dtype=self.source_reliability_prior.dtype)
        self.source_reliability_prior.copy_(prior)
        self.source_reliability_count.add_(1)
        self.source_reliability_fitted.fill_(True)

    @torch.no_grad()
    def update_source_reliability(self, prior, momentum=None):
        prior = self._normalize_source_reliability(prior)
        if not bool(self.source_reliability_fitted):
            self.set_source_reliability(prior)
            return

        if momentum is None:
            momentum = 0.9
        momentum = min(max(float(momentum), 0.0), 1.0)
        prior = prior.to(device=self.source_reliability_prior.device, dtype=self.source_reliability_prior.dtype)
        updated = momentum * self.source_reliability_prior + (1.0 - momentum) * prior
        self.source_reliability_prior.copy_(self._normalize_source_reliability(updated))
        self.source_reliability_count.add_(1)
        self.source_reliability_fitted.fill_(True)

    def _normalize_source_reliability(self, prior):
        prior = prior.detach().float().view(-1)
        if prior.numel() != len(self.evidence_source_names):
            raise ValueError(
                "Source reliability size must match SHARK.HEAD.EVIDENCE_SOURCE_NAMES: "
                f"prior={tuple(prior.shape)}, sources={len(self.evidence_source_names)}"
            )
        prior = torch.where(torch.isfinite(prior), prior, torch.ones_like(prior))
        if prior.sum().abs() <= self.eps:
            prior = torch.ones_like(prior)
        prior = prior.clamp_min(float(self.source_reliability_min))
        prior = prior / prior.mean().clamp_min(self.eps)
        return prior.clamp(
            min=float(self.source_reliability_min),
            max=float(self.source_reliability_max),
        )

    @staticmethod
    def _aggregate_source_scores(scores, topk, mode):
        if mode == "mean":
            return scores.mean(dim=1)
        if mode == "max":
            return scores.max(dim=1).values
        if mode == "topk_mean":
            k = min(max(int(topk), 1), scores.size(1))
            return scores.topk(k, dim=1).values.mean(dim=1)
        raise ValueError(f"Unsupported SHARK.HEAD.SOURCE_CONSISTENT_AGG: {mode}")

    @torch.no_grad()
    def set_mil_node_reference(self, mean, scale, values=None):
        mean = mean.detach().view(-1)
        scale = scale.detach().view(-1)
        expected = self.mil_reference_mean.numel()
        if mean.numel() != expected or scale.numel() != expected:
            raise ValueError(
                "MIL reference size must match DATA.N_VAR: "
                f"mean={tuple(mean.shape)}, scale={tuple(scale.shape)}, sensors={expected}"
            )
        mean = mean.to(device=self.mil_reference_mean.device, dtype=self.mil_reference_mean.dtype)
        scale = scale.to(device=self.mil_reference_scale.device, dtype=self.mil_reference_scale.dtype)
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("MIL reference mean and scale must be finite")
        self.mil_reference_mean.copy_(mean)
        self.mil_reference_scale.copy_(
            scale.clamp_min(float(self.normal_calibration_min_scale))
        )
        self.mil_reference_fitted.fill_(True)

    @torch.no_grad()
    def reset_mil_node_reference(self):
        self.mil_reference_mean.zero_()
        self.mil_reference_scale.fill_(1.0)
        self.mil_reference_fitted.fill_(False)

    @torch.no_grad()
    def mil_reference_diagnostics(self):
        scale = self.mil_reference_scale.detach().float()
        effective_scale, floor_value, affected_mask = (
            self._effective_mil_reference_scale(scale)
        )
        affected_indices = torch.nonzero(
            affected_mask,
            as_tuple=False,
        ).view(-1)
        return {
            "mil_reference_fitted": bool(self.mil_reference_fitted.item()),
            "mil_reference_scale_min": float(scale.min().cpu()),
            "mil_reference_scale_mean": float(scale.mean().cpu()),
            "mil_reference_scale_max": float(scale.max().cpu()),
            "mil_reference_scale_values": [
                float(value) for value in scale.cpu().tolist()
            ],
            "mrsf_quantile": float(
                self.mil_reference_scale_floor_quantile
            ),
            "mrsf_floor_value": (
                0.0
                if floor_value is None
                else float(floor_value.detach().cpu())
            ),
            "mrsf_effective_scale_min": float(
                effective_scale.min().cpu()
            ),
            "mrsf_effective_scale_mean": float(
                effective_scale.mean().cpu()
            ),
            "mrsf_effective_scale_max": float(
                effective_scale.max().cpu()
            ),
            "mrsf_effective_scale_values": [
                float(value) for value in effective_scale.cpu().tolist()
            ],
            "mrsf_affected_sensor_count": int(affected_indices.numel()),
            "mrsf_affected_sensor_indices": [
                int(index) for index in affected_indices.cpu().tolist()
            ],
        }

    def set_evidence_reference(self, mean, scale, values=None):
        if mean.numel() != len(self.conservative_sources) or scale.numel() != len(self.conservative_sources):
            raise ValueError(
                "Evidence reference size must match SHARK.HEAD.CONSERVATIVE_SOURCES: "
                f"mean={tuple(mean.shape)}, scale={tuple(scale.shape)}, sources={len(self.conservative_sources)}"
            )
        mean = mean.to(device=self.evidence_reference_mean.device, dtype=self.evidence_reference_mean.dtype)
        scale = scale.to(device=self.evidence_reference_scale.device, dtype=self.evidence_reference_scale.dtype)
        self.evidence_reference_mean.copy_(mean)
        self.evidence_reference_scale.copy_(scale.clamp_min(float(self.normal_calibration_min_scale)))
        self.evidence_reference_values.zero_()
        self.evidence_reference_count.zero_()
        if values is not None:
            if values.dim() != 2 or values.size(1) != len(self.conservative_sources):
                raise ValueError(
                    "Evidence reference values must match SHARK.HEAD.CONSERVATIVE_SOURCES: "
                    f"values={tuple(values.shape)}, sources={len(self.conservative_sources)}"
                )
            values = values.detach()
            max_samples = int(self.normal_calibration_max_samples)
            if values.size(0) > max_samples:
                sample_idx = torch.linspace(
                    0,
                    values.size(0) - 1,
                    steps=max_samples,
                    device=values.device,
                ).round().long()
                values = values.index_select(0, sample_idx)
            values = values.to(
                device=self.evidence_reference_values.device,
                dtype=self.evidence_reference_values.dtype,
            )
            self.evidence_reference_values[: values.size(0)].copy_(values)
            self.evidence_reference_count.fill_(values.size(0))
        self.evidence_reference_fitted.fill_(True)

    def reset_evidence_reference(self):
        self.evidence_reference_mean.zero_()
        self.evidence_reference_scale.fill_(1.0)
        self.evidence_reference_values.zero_()
        self.evidence_reference_count.zero_()
        self.evidence_reference_fitted.fill_(False)

    def _normal_calibrate_evidence(self, scores):
        if not self.normal_calibration_enable or not bool(self.evidence_reference_fitted):
            return scores

        mean = self.evidence_reference_mean.to(device=scores.device, dtype=scores.dtype)
        scale = self.evidence_reference_scale.to(device=scores.device, dtype=scores.dtype)
        z_scores = (scores - mean.unsqueeze(0)) / scale.unsqueeze(0).clamp_min(
            float(self.normal_calibration_min_scale)
        )
        if self.normal_calibration_mode == "zscore":
            return z_scores
        if self.normal_calibration_mode == "abs_zscore":
            return z_scores.abs()
        if self.normal_calibration_mode == "positive_zscore":
            return F.relu(z_scores)
        if self.normal_calibration_mode in {
            "ecdf_upper",
            "ecdf_lower",
            "ecdf_two_sided",
            "soft_ecdf_upper",
            "soft_ecdf_lower",
            "soft_ecdf_two_sided",
        }:
            return self._normal_tail_evidence(scores)
        raise ValueError(f"Unsupported SHARK.HEAD.NORMAL_CALIBRATION_MODE: {self.normal_calibration_mode}")

    def _normal_tail_evidence(self, scores):
        count = int(self.evidence_reference_count.item())
        if count <= 0:
            return scores

        references = self.evidence_reference_values[:count].to(device=scores.device, dtype=scores.dtype)
        mode = self.normal_calibration_mode
        if mode.startswith("soft_"):
            scale = self.evidence_reference_scale.to(device=scores.device, dtype=scores.dtype)
            bandwidth = scale.view(1, 1, -1).clamp_min(float(self.normal_calibration_min_scale))
            bandwidth = bandwidth * max(float(self.normal_calibration_ecdf_temperature), self.eps)
            diff = references.unsqueeze(0) - scores.unsqueeze(1)
            upper_tail = (torch.sigmoid(diff / bandwidth).sum(dim=1) + 1.0) / (count + 1.0)
            lower_tail = (torch.sigmoid(-diff / bandwidth).sum(dim=1) + 1.0) / (count + 1.0)
        else:
            upper_tail = ((references.unsqueeze(0) >= scores.unsqueeze(1)).float().sum(dim=1) + 1.0) / (count + 1.0)
            lower_tail = ((references.unsqueeze(0) <= scores.unsqueeze(1)).float().sum(dim=1) + 1.0) / (count + 1.0)

        if mode.endswith("_upper"):
            p_values = upper_tail
        elif mode.endswith("_lower"):
            p_values = lower_tail
        elif mode.endswith("_two_sided"):
            p_values = 2.0 * torch.minimum(upper_tail, lower_tail)
        else:
            raise ValueError(f"Unsupported SHARK.HEAD.NORMAL_CALIBRATION_MODE: {mode}")

        p_values = p_values.clamp(min=1.0 / (count + 1.0), max=1.0)
        return -torch.log(p_values.clamp_min(self.eps))

    @staticmethod
    def _prefixed_evidence_breakdown(prefix, source_names, scores):
        return {
            f"{prefix}_{name}": scores[:, idx]
            for idx, name in enumerate(source_names)
        }

    @staticmethod
    def _is_raw_evidence(name):
        return name.startswith("raw")

    @staticmethod
    def _check_shapes(node_embeddings, raw_node_scores, graph_embedding):
        if node_embeddings.dim() != 3:
            raise ValueError(f"SharkHead expects node_embeddings [B, N, H], got {tuple(node_embeddings.shape)}")
        if raw_node_scores.dim() != 2:
            raise ValueError(f"SharkHead expects raw_node_scores [B, N], got {tuple(raw_node_scores.shape)}")
        if graph_embedding.dim() != 2:
            raise ValueError(f"SharkHead expects graph_embedding [B, H], got {tuple(graph_embedding.shape)}")
        if node_embeddings.size(0) != raw_node_scores.size(0) or node_embeddings.size(1) != raw_node_scores.size(1):
            raise ValueError(
                "SharkHead got inconsistent node dimensions: "
                f"node_embeddings={tuple(node_embeddings.shape)}, raw_node_scores={tuple(raw_node_scores.shape)}"
            )
        if graph_embedding.size(0) != raw_node_scores.size(0):
            raise ValueError(
                "SharkHead got inconsistent batch dimensions: "
                f"graph_embedding={tuple(graph_embedding.shape)}, raw_node_scores={tuple(raw_node_scores.shape)}"
            )
