import math

import torch


def temporal_topq_rms(residuals, fraction=0.25):
    if residuals.dim() < 2:
        raise ValueError("Temporal residuals must include a time dimension")
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Temporal top-q fraction must be in (0, 1]")
    count = max(1, int(math.ceil(residuals.size(1) * fraction)))
    selected = residuals.abs().topk(count, dim=1).values
    return selected.square().mean(dim=1).sqrt()


class GlobalGraphVARMechanism:
    """Normal-data local mechanisms used to generate and score counterfactuals."""

    SUPPORTED_TYPES = (
        "lag_break",
        "neighbor_break",
        "relation_flip",
        "local_function_shift",
    )

    def __init__(self, cfg, adaptive=False):
        synthetic = cfg.GFM4MTSAD.SYNTHETIC
        self.adaptive = bool(adaptive)
        self.expected_n_nodes = int(cfg.DATA.N_VAR)
        self.lags = max(int(getattr(synthetic, "GRAPH_VAR_LAGS", 2)), 1)
        self.topk = max(int(getattr(synthetic, "GRAPH_VAR_TOPK", 5)), 1)
        self.ridge = max(float(getattr(synthetic, "GRAPH_VAR_RIDGE", 0.01)), 1e-8)
        self.min_parent_strength = max(
            float(getattr(synthetic, "GRAPH_VAR_MIN_PARENT_STRENGTH", 0.01)),
            0.0,
        )
        self.min_samples = max(
            int(getattr(synthetic, "GRAPH_VAR_MIN_SAMPLES", 128)),
            self.lags + 4,
        )
        self.max_samples = max(
            int(getattr(synthetic, "GRAPH_VAR_MAX_SAMPLES", 50000)),
            self.min_samples,
        )
        self.min_node_scale = max(
            float(getattr(synthetic, "GRAPH_VAR_MIN_NODE_SCALE", 1e-3)),
            1e-8,
        )
        self.min_residual_scale = max(
            float(getattr(synthetic, "GRAPH_VAR_MIN_RESIDUAL_SCALE", 1e-3)),
            1e-8,
        )
        self.reference_size = max(
            int(getattr(synthetic, "GRAPH_VAR_REFERENCE_SIZE", 4096)),
            64,
        )
        configured_window = int(getattr(synthetic, "GRAPH_VAR_SCORE_WINDOW", 0))
        self.score_window = max(configured_window, 0)
        self.reference_max_window = max(
            configured_window if configured_window > 0 else int(cfg.DATA.WIN_SIZE),
            1,
        )
        self.temporal_topq_fraction = min(
            max(
                float(
                    getattr(
                        synthetic,
                        "GRAPH_VAR_TEMPORAL_TOPQ_FRACTION",
                        0.25,
                    )
                ),
                1e-6,
            ),
            1.0,
        )
        self.support_low_quantile = float(
            getattr(synthetic, "DIRECT_CANDIDATE_SUPPORT_LOW_QUANTILE", 0.01)
        )
        self.support_high_quantile = float(
            getattr(synthetic, "DIRECT_CANDIDATE_SUPPORT_HIGH_QUANTILE", 0.99)
        )
        self.discrete_max_states = max(
            int(getattr(synthetic, "GRAPH_VAR_DISCRETE_MAX_STATES", 16)),
            2,
        )
        self.discrete_min_state_count = max(
            int(
                getattr(
                    synthetic,
                    "GRAPH_VAR_DISCRETE_MIN_STATE_COUNT",
                    4,
                )
            ),
            2,
        )
        self.lag_offset_steps = max(
            int(getattr(synthetic, "GRAPH_VAR_LAG_OFFSET_STEPS", 1)),
            1,
        )
        min_candidate_rms = max(
            float(getattr(synthetic, "HYBRID_FALLBACK_MIN_RMS", 0.1)),
            0.0,
        )
        max_edit_strength = max(
            float(getattr(synthetic, "EDIT_STRENGTH_END", 0.75)),
            1e-6,
        )
        self.min_lag_offset_effect = min_candidate_rms / max_edit_strength

        self.fitted = False
        self.n_nodes = 0
        self._device_cache = {}
        self._diagnostics = {}

    @staticmethod
    def _extract_train_series(source):
        if hasattr(source, "dataset") and hasattr(source.dataset, "train"):
            source = source.dataset.train
        elif hasattr(source, "train"):
            source = source.train
        series = torch.as_tensor(source, dtype=torch.float32, device="cpu")
        if series.dim() != 2:
            raise ValueError("Normal mechanism training data must have shape [T, N]")
        return series

    @staticmethod
    def _subsample_rows(rows, max_samples):
        if rows.numel() <= max_samples:
            return rows
        positions = torch.linspace(
            0,
            rows.numel() - 1,
            steps=max_samples,
            dtype=torch.float64,
        ).round().long()
        return rows.index_select(0, positions.unique(sorted=True))

    @staticmethod
    def _ridge_fit(features, targets, ridge):
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        ones = torch.ones(
            features.size(0),
            1,
            dtype=features.dtype,
            device=features.device,
        )
        design = torch.cat((features, ones), dim=1)
        normalizer = max(design.size(0), 1)
        gram = design.T @ design / normalizer
        rhs = design.T @ targets / normalizer
        penalty = torch.eye(gram.size(0), dtype=gram.dtype, device=gram.device)
        penalty[-1, -1] = 0.0
        gram = gram + ridge * penalty
        try:
            return torch.linalg.solve(gram, rhs)
        except RuntimeError:
            return torch.linalg.pinv(gram) @ rhs

    @staticmethod
    def _finite_quantiles(values, quantiles):
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return None
        return torch.quantile(
            values,
            torch.as_tensor(quantiles, dtype=values.dtype, device=values.device),
        )

    @staticmethod
    def _reference_sample(values, reference_size):
        values = values[torch.isfinite(values)]
        pre_sort_size = max(reference_size * 4, reference_size)
        if values.numel() > pre_sort_size:
            positions = torch.linspace(
                0,
                values.numel() - 1,
                steps=pre_sort_size,
                dtype=torch.float64,
            ).round().long()
            values = values.index_select(0, positions)
        values = values.sort().values
        if values.numel() > reference_size:
            positions = torch.linspace(
                0,
                values.numel() - 1,
                steps=reference_size,
                dtype=torch.float64,
            ).round().long()
            values = values.index_select(0, positions)
        return values

    def _calibration_reference(self, values, discrete):
        if not self.adaptive or not discrete or values.numel() < 4:
            return values, False
        median = torch.quantile(values, 0.5)
        tolerance = torch.finfo(values.dtype).eps * max(
            abs(float(median.item())),
            1.0,
        )
        positive_tail = values[values > median + tolerance]
        if positive_tail.numel() < self.discrete_min_state_count:
            return values, False
        return positive_tail, True

    def fit(self, source):
        self.fitted = False
        self._device_cache = {}
        self._diagnostics = {}
        series = self._extract_train_series(source)
        n_time, n_nodes = series.shape
        if n_nodes != self.expected_n_nodes:
            raise ValueError(
                "Graph-VAR variable count does not match DATA.N_VAR: "
                f"data={n_nodes}, config={self.expected_n_nodes}"
            )
        if n_time < self.min_samples + self.lags:
            raise ValueError(
                "Not enough normal samples to fit Graph-VAR: "
                f"got {n_time}, need at least {self.min_samples + self.lags}"
            )

        finite = torch.isfinite(series)
        counts = finite.sum(dim=0)
        safe_series = torch.where(finite, series, torch.zeros_like(series))
        means = safe_series.sum(dim=0) / counts.clamp_min(1)
        centered = torch.where(finite, series - means, torch.zeros_like(series))
        scales = (
            centered.square().sum(dim=0) / counts.clamp_min(1)
        ).sqrt()
        safe_scales = scales.clamp_min(self.min_node_scale)

        support_low = torch.full((n_nodes,), float("nan"), dtype=series.dtype)
        support_high = torch.full_like(support_low, float("nan"))
        unique_counts = torch.zeros(n_nodes, dtype=torch.long)
        dominant_ratios = torch.zeros(n_nodes, dtype=series.dtype)
        zero_ratios = torch.zeros(n_nodes, dtype=series.dtype)
        discrete_mask = torch.zeros(n_nodes, dtype=torch.bool)
        discrete_state_count = torch.zeros(n_nodes, dtype=torch.long)
        discrete_states = torch.full(
            (n_nodes, self.discrete_max_states),
            float("nan"),
            dtype=series.dtype,
        )
        for node in range(n_nodes):
            values = series[:, node]
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                continue
            unique, value_counts = torch.unique(
                values,
                sorted=True,
                return_counts=True,
            )
            unique_counts[node] = unique.numel()
            dominant_ratios[node] = value_counts.max() / max(values.numel(), 1)
            zero_ratios[node] = (values == 0).float().mean()
            reliable_states = unique[
                value_counts >= self.discrete_min_state_count
            ]
            is_discrete = bool(
                self.adaptive
                and unique.numel() <= self.discrete_max_states
                and reliable_states.numel() >= 2
            )
            if is_discrete:
                n_states = int(unique.numel())
                discrete_mask[node] = True
                discrete_state_count[node] = n_states
                discrete_states[node, :n_states] = unique
                support_low[node] = unique.min()
                support_high[node] = unique.max()
                continue
            bounds = self._finite_quantiles(
                values,
                (self.support_low_quantile, self.support_high_quantile),
            )
            if bounds is not None:
                support_low[node], support_high[node] = bounds[0], bounds[1]
        support_width = support_high - support_low
        marginal_valid = (
            (counts >= self.min_samples)
            & torch.isfinite(scales)
            & (scales >= self.min_node_scale)
            & torch.isfinite(support_width)
            & (support_width >= self.min_node_scale)
        )

        standardized = torch.where(
            finite,
            (series - means) / safe_scales,
            torch.zeros_like(series),
        )
        all_rows = torch.arange(self.lags, n_time, dtype=torch.long)
        fit_rows = self._subsample_rows(all_rows, self.max_samples)
        lagged = torch.cat(
            [standardized[fit_rows - lag] for lag in range(1, self.lags + 1)],
            dim=1,
        )
        targets = standardized.index_select(0, fit_rows)
        dense_beta = self._ridge_fit(lagged, targets, self.ridge)[:-1]
        dense_beta = dense_beta.reshape(self.lags, n_nodes, n_nodes)
        parent_strength = dense_beta.square().sum(dim=0).sqrt()

        parent_index = torch.full((n_nodes, self.topk), -1, dtype=torch.long)
        self_coef = torch.zeros(n_nodes, self.lags, dtype=series.dtype)
        parent_coef = torch.zeros(
            n_nodes,
            self.topk,
            self.lags,
            dtype=series.dtype,
        )
        intercept = torch.zeros(n_nodes, dtype=series.dtype)
        self_only_coef = torch.zeros(n_nodes, self.lags, dtype=series.dtype)
        self_only_intercept = torch.zeros(n_nodes, dtype=series.dtype)

        for target in range(n_nodes):
            self_features = lagged[:, [
                lag * n_nodes + target for lag in range(self.lags)
            ]]
            self_beta = self._ridge_fit(
                self_features,
                targets[:, target],
                self.ridge,
            )[:, 0]
            self_only_coef[target] = self_beta[:self.lags]
            self_only_intercept[target] = self_beta[-1]

            scores = parent_strength[:, target].clone()
            scores[target] = float("-inf")
            scores[~marginal_valid] = float("-inf")
            scores[scores < self.min_parent_strength] = float("-inf")
            available = int(torch.isfinite(scores).sum().item())
            n_parents = min(self.topk, available)
            if n_parents > 0:
                parents = torch.topk(scores, k=n_parents).indices
                parent_index[target, :n_parents] = parents
            else:
                parents = torch.empty(0, dtype=torch.long)

            feature_indices = [lag * n_nodes + target for lag in range(self.lags)]
            for parent in parents.tolist():
                feature_indices.extend(
                    lag * n_nodes + parent for lag in range(self.lags)
                )
            local_features = lagged[:, feature_indices]
            beta = self._ridge_fit(local_features, targets[:, target], self.ridge)
            beta = beta[:, 0]
            self_coef[target] = beta[:self.lags]
            offset = self.lags
            for slot in range(n_parents):
                parent_coef[target, slot] = beta[offset:offset + self.lags]
                offset += self.lags
            intercept[target] = beta[-1]

        prediction = intercept.unsqueeze(0).expand(all_rows.numel(), -1).clone()
        for lag in range(1, self.lags + 1):
            prediction += standardized[all_rows - lag] * self_coef[:, lag - 1]
        for target in range(n_nodes):
            for slot in range(self.topk):
                parent = int(parent_index[target, slot].item())
                if parent < 0:
                    continue
                for lag in range(1, self.lags + 1):
                    prediction[:, target] += (
                        standardized[all_rows - lag, parent]
                        * parent_coef[target, slot, lag - 1]
                    )

        offset_rows = torch.arange(
            self.lags + self.lag_offset_steps,
            n_time,
            dtype=torch.long,
        )
        delayed_prediction = intercept.unsqueeze(0).expand(
            offset_rows.numel(),
            -1,
        ).clone()
        for lag in range(1, self.lags + 1):
            delayed_prediction += (
                standardized[offset_rows - lag - self.lag_offset_steps]
                * self_coef[:, lag - 1]
            )
        for target in range(n_nodes):
            for slot in range(self.topk):
                parent = int(parent_index[target, slot].item())
                if parent < 0:
                    continue
                for lag in range(1, self.lags + 1):
                    delayed_prediction[:, target] += (
                        standardized[
                            offset_rows - lag - self.lag_offset_steps,
                            parent,
                        ]
                        * parent_coef[target, slot, lag - 1]
                    )
        original_offset_prediction = prediction.index_select(
            0,
            offset_rows - self.lags,
        )
        lag_offset_prediction_delta = (
            delayed_prediction - original_offset_prediction
        ).square().mean(dim=0).sqrt()
        lag_offset_valid = torch.isfinite(lag_offset_prediction_delta) & (
            (
                discrete_mask
                & (lag_offset_prediction_delta > torch.finfo(series.dtype).eps)
            )
            | (
                ~discrete_mask
                & (lag_offset_prediction_delta >= self.min_lag_offset_effect)
            )
        )

        residual = standardized.index_select(0, all_rows) - prediction
        residual_center = torch.zeros(n_nodes, dtype=series.dtype)
        residual_scale = torch.full_like(residual_center, float("nan"))
        residual_valid = torch.zeros(n_nodes, dtype=torch.bool)
        normalized_residual = torch.full_like(residual, float("nan"))
        for node in range(n_nodes):
            values = residual[:, node]
            values = values[torch.isfinite(values)]
            if values.numel() < self.min_samples:
                continue
            median = torch.quantile(values, 0.5)
            q25, q75 = torch.quantile(
                values,
                torch.tensor((0.25, 0.75), dtype=values.dtype),
            )
            robust_scale = (q75 - q25) / 1.349
            std_scale = values.std(unbiased=False)
            scale = robust_scale if robust_scale >= self.min_residual_scale else std_scale
            if not torch.isfinite(scale) or scale < self.min_residual_scale:
                continue
            residual_center[node] = median
            residual_scale[node] = scale
            normalized_residual[:, node] = (residual[:, node] - median) / scale
            residual_valid[node] = True

        self_only_prediction = self_only_intercept.unsqueeze(0).expand(
            all_rows.numel(),
            -1,
        ).clone()
        for lag in range(1, self.lags + 1):
            self_only_prediction += (
                standardized[all_rows - lag]
                * self_only_coef[:, lag - 1]
            )
        self_only_residual = standardized.index_select(0, all_rows) - self_only_prediction
        self_residual_center = torch.zeros(n_nodes, dtype=series.dtype)
        self_residual_scale = torch.full_like(
            self_residual_center,
            float("nan"),
        )
        self_residual_valid = torch.zeros(n_nodes, dtype=torch.bool)
        normalized_self_residual = torch.full_like(
            self_only_residual,
            float("nan"),
        )
        for node in range(n_nodes):
            values = self_only_residual[:, node]
            values = values[torch.isfinite(values)]
            if values.numel() < self.min_samples:
                continue
            median = torch.quantile(values, 0.5)
            q25, q75 = torch.quantile(
                values,
                torch.tensor((0.25, 0.75), dtype=values.dtype),
            )
            robust_scale = (q75 - q25) / 1.349
            std_scale = values.std(unbiased=False)
            scale = (
                robust_scale
                if robust_scale >= self.min_residual_scale
                else std_scale
            )
            if not torch.isfinite(scale) or scale < self.min_residual_scale:
                continue
            self_residual_center[node] = median
            self_residual_scale[node] = scale
            normalized_self_residual[:, node] = (
                self_only_residual[:, node] - median
            ) / scale
            self_residual_valid[node] = True

        reference_max_window = min(
            self.reference_max_window,
            normalized_residual.size(0),
        )
        score_reference = torch.full(
            (reference_max_window, n_nodes, self.reference_size),
            float("nan"),
            dtype=series.dtype,
        )
        score_tail_valid = torch.zeros(
            reference_max_window,
            n_nodes,
            dtype=torch.bool,
        )
        positive_tail_reference = torch.zeros_like(score_tail_valid)
        for score_length in range(1, reference_max_window + 1):
            rolling_scores = normalized_residual.unfold(0, score_length, 1)
            rolling_scores = rolling_scores.square().mean(dim=-1).sqrt()
            for node in range(n_nodes):
                values = self._reference_sample(
                    rolling_scores[:, node],
                    self.reference_size,
                )
                if values.numel() < 4:
                    continue
                score_reference[score_length - 1, node, :values.numel()] = values
                calibration_values, uses_positive_tail = (
                    self._calibration_reference(
                        values,
                        bool(discrete_mask[node].item()),
                    )
                )
                if calibration_values.numel() < 4:
                    continue
                q80, q90, q995 = torch.quantile(
                    calibration_values,
                    torch.tensor(
                        (0.80, 0.90, 0.995),
                        dtype=calibration_values.dtype,
                    ),
                )
                score_tail_valid[score_length - 1, node] = bool(
                    torch.isfinite(q995) and q995 > max(float(q80), float(q90))
                )
                positive_tail_reference[score_length - 1, node] = (
                    uses_positive_tail
                )

        self_score_reference = torch.full_like(score_reference, float("nan"))
        self_score_tail_valid = torch.zeros_like(score_tail_valid)
        self_positive_tail_reference = torch.zeros_like(score_tail_valid)
        for score_length in range(1, reference_max_window + 1):
            rolling_scores = normalized_self_residual.unfold(
                0,
                score_length,
                1,
            )
            rolling_scores = rolling_scores.square().mean(dim=-1).sqrt()
            for node in range(n_nodes):
                values = self._reference_sample(
                    rolling_scores[:, node],
                    self.reference_size,
                )
                if values.numel() < 4:
                    continue
                self_score_reference[
                    score_length - 1,
                    node,
                    :values.numel(),
                ] = values
                calibration_values, uses_positive_tail = (
                    self._calibration_reference(
                        values,
                        bool(discrete_mask[node].item()),
                    )
                )
                if calibration_values.numel() < 4:
                    continue
                q80, q90, q995 = torch.quantile(
                    calibration_values,
                    torch.tensor(
                        (0.80, 0.90, 0.995),
                        dtype=calibration_values.dtype,
                    ),
                )
                self_score_tail_valid[score_length - 1, node] = bool(
                    torch.isfinite(q995) and q995 > max(float(q80), float(q90))
                )
                self_positive_tail_reference[score_length - 1, node] = (
                    uses_positive_tail
                )

        valid_nodes = marginal_valid & residual_valid & score_tail_valid.any(dim=0)
        self_temporal_valid_nodes = (
            marginal_valid
            & self_residual_valid
            & self_score_tail_valid.any(dim=0)
        )
        self.n_nodes = n_nodes
        self.node_mean = means.float()
        self.node_scale = safe_scales.float()
        self.support_low = support_low.float()
        self.support_high = support_high.float()
        self.unique_counts = unique_counts
        self.dominant_ratios = dominant_ratios.float()
        self.zero_ratios = zero_ratios.float()
        self.discrete_mask = discrete_mask
        self.discrete_state_count = discrete_state_count
        self.discrete_states = discrete_states.float()
        self.parent_index = parent_index
        self.self_coef = self_coef.float()
        self.parent_coef = parent_coef.float()
        self.intercept = intercept.float()
        self.self_only_coef = self_only_coef.float()
        self.self_only_intercept = self_only_intercept.float()
        self.residual_center = residual_center.float()
        self.residual_scale = residual_scale.float()
        self.self_residual_center = self_residual_center.float()
        self.self_residual_scale = self_residual_scale.float()
        self.self_residual_valid = self_residual_valid
        self.score_reference = score_reference.float()
        self.positive_tail_reference = positive_tail_reference
        self.self_score_reference = self_score_reference.float()
        self.self_positive_tail_reference = self_positive_tail_reference
        self.self_temporal_valid_nodes = self_temporal_valid_nodes
        self.lag_offset_prediction_delta = lag_offset_prediction_delta.float()
        self.lag_offset_valid = lag_offset_valid
        self.valid_nodes = valid_nodes
        self.fitted = True
        self._device_cache = {}

        valid_residual_scales = self.residual_scale[valid_nodes]
        parent_counts = (parent_index >= 0).sum(dim=1)
        lag_parameter_delta = torch.cat(
            (
                self_coef.roll(1, dims=1) - self_coef,
                (
                    parent_coef.roll(1, dims=2) - parent_coef
                ).reshape(n_nodes, -1),
            ),
            dim=1,
        ).square().mean(dim=1).sqrt()
        has_parent = parent_counts > 0
        relation_eligible = valid_nodes & has_parent
        invalid_reason_counts = {}
        node_diagnostics = []
        for node in range(n_nodes):
            reasons = []
            if int(counts[node].item()) < self.min_samples:
                reasons.append("insufficient_finite_samples")
            if not bool(torch.isfinite(scales[node]).item()) or bool(
                (scales[node] < self.min_node_scale).item()
            ):
                reasons.append("constant_or_low_scale")
            if not bool(torch.isfinite(support_width[node]).item()) or bool(
                (support_width[node] < self.min_node_scale).item()
            ):
                reasons.append("support_collapsed")
            if not bool(residual_valid[node].item()):
                reasons.append("residual_scale_invalid")
            if not bool(score_tail_valid[:, node].any().item()):
                reasons.append("score_tail_collapsed")
            for reason in reasons:
                invalid_reason_counts[reason] = (
                    invalid_reason_counts.get(reason, 0) + 1
                )
            if bool(discrete_mask[node].item()):
                node_kind = "discrete"
            elif int(unique_counts[node].item()) <= 1:
                node_kind = "constant"
            elif (
                self.adaptive
                and int(unique_counts[node].item()) <= self.discrete_max_states
            ):
                node_kind = "discrete_insufficient_support"
            else:
                node_kind = "continuous"
            node_diagnostics.append(
                {
                    "node": node,
                    "kind": node_kind,
                    "finite_count": int(counts[node].item()),
                    "unique_count": int(unique_counts[node].item()),
                    "dominant_ratio": float(dominant_ratios[node].item()),
                    "zero_ratio_normalized": float(zero_ratios[node].item()),
                    "std": float(scales[node].item()),
                    "support_low": self._optional_float(support_low[node]),
                    "support_high": self._optional_float(support_high[node]),
                    "support_width": self._optional_float(support_width[node]),
                    "discrete_state_count": int(
                        discrete_state_count[node].item()
                    ),
                    "residual_scale": self._optional_float(
                        residual_scale[node]
                    ),
                    "lag_parameter_delta": float(
                        lag_parameter_delta[node].item()
                    ),
                    "lag_offset_prediction_delta": float(
                        lag_offset_prediction_delta[node].item()
                    ),
                    "lag_offset_valid": bool(lag_offset_valid[node].item()),
                    "residual_valid": bool(residual_valid[node].item()),
                    "score_tail_valid": bool(
                        score_tail_valid[:, node].any().item()
                    ),
                    "uses_positive_tail": bool(
                        positive_tail_reference[:, node].any().item()
                    ),
                    "parent_count": int(parent_counts[node].item()),
                    "valid": bool(valid_nodes[node].item()),
                    "invalid_reasons": reasons,
                }
            )
        self._diagnostics = {
            "fitted": True,
            "mode": "adaptive_graph_var" if self.adaptive else "global_graph_var",
            "n_nodes": n_nodes,
            "n_samples": n_time,
            "fit_samples": int(fit_rows.numel()),
            "lags": self.lags,
            "topk": self.topk,
            "score_window": self.score_window,
            "temporal_topq_fraction": self.temporal_topq_fraction,
            "score_reference_max_window": reference_max_window,
            "lag_offset_min_effect": self.min_lag_offset_effect,
            "lag_offset_steps": self.lag_offset_steps,
            "valid_node_count": int(valid_nodes.sum().item()),
            "valid_node_ratio": float(valid_nodes.float().mean().item()),
            "continuous_node_count": int(
                sum(item["kind"] == "continuous" for item in node_diagnostics)
            ),
            "discrete_node_count": int(discrete_mask.sum().item()),
            "constant_node_count": int(
                sum(item["kind"] == "constant" for item in node_diagnostics)
            ),
            "positive_tail_node_count": int(
                positive_tail_reference.any(dim=0).sum().item()
            ),
            "nodes_with_parent_count": int((parent_counts > 0).sum().item()),
            "parent_count_mean": float(parent_counts.float().mean().item()),
            "eligible_node_counts": {
                "lag_break": int(
                    (valid_nodes & lag_offset_valid).sum().item()
                ),
                "neighbor_break": int(relation_eligible.sum().item()),
                "relation_flip": int(relation_eligible.sum().item()),
                "local_function_shift": int(valid_nodes.sum().item()),
            },
            "support_collapsed_count": int((~marginal_valid).sum().item()),
            "invalid_reason_counts": invalid_reason_counts,
            "node_diagnostics": node_diagnostics,
            "residual_scale_min": self._safe_stat(valid_residual_scales, "min"),
            "residual_scale_mean": self._safe_stat(valid_residual_scales, "mean"),
            "residual_scale_max": self._safe_stat(valid_residual_scales, "max"),
            "self_residual_valid_node_count": int(
                (valid_nodes & self_residual_valid).sum().item()
            ),
            "self_temporal_valid_node_count": int(
                self_temporal_valid_nodes.sum().item()
            ),
            "self_temporal_fallback_node_count": int(
                (self_temporal_valid_nodes & ~valid_nodes).sum().item()
            ),
        }
        mechanism_relevance = torch.zeros(n_nodes, dtype=series.dtype)
        mechanism_relevance_valid = (
            valid_nodes
            & self_residual_valid
            & (parent_index >= 0).any(dim=1)
            & torch.isfinite(residual_scale)
            & torch.isfinite(self_residual_scale)
            & (self_residual_scale >= self.min_residual_scale)
        )
        mechanism_relevance[mechanism_relevance_valid] = (
            1.0
            - residual_scale[mechanism_relevance_valid]
            / self_residual_scale[mechanism_relevance_valid]
        ).clamp_min(0.0)
        self._diagnostics.update(
            {
                "mechanism_relevance_definition": (
                    "max(0, 1 - full_residual_scale / self_only_residual_scale)"
                ),
                "mechanism_relevance_valid_node_count": int(
                    mechanism_relevance_valid.sum().item()
                ),
                "mechanism_relevance_nonzero_node_count": int(
                    (mechanism_relevance > 0).sum().item()
                ),
                "mechanism_relevance_mean": self._safe_stat(
                    mechanism_relevance[mechanism_relevance_valid],
                    "mean",
                ),
                "mechanism_relevance_max": self._safe_stat(
                    mechanism_relevance[mechanism_relevance_valid],
                    "max",
                ),
            }
        )
        return dict(self._diagnostics)

    def recalibrate(self, source):
        """Refresh normal residual statistics without refitting Graph-VAR."""
        if not self.fitted:
            raise RuntimeError("Graph-VAR must be fitted before recalibration")
        series = self._extract_train_series(source)
        n_time, n_nodes = series.shape
        if n_nodes != self.n_nodes:
            raise ValueError(
                "Graph-VAR recalibration variable count mismatch: "
                f"data={n_nodes}, fitted={self.n_nodes}"
            )
        if n_time < self.min_samples + self.lags:
            raise ValueError(
                "Not enough normal samples to recalibrate Graph-VAR: "
                f"got {n_time}, need at least {self.min_samples + self.lags}"
            )

        finite = torch.isfinite(series)
        standardized = torch.where(
            finite,
            (series - self.node_mean) / self.node_scale,
            torch.zeros_like(series),
        )
        rows = torch.arange(self.lags, n_time, dtype=torch.long)
        prediction = self.intercept.unsqueeze(0).expand(rows.numel(), -1).clone()
        self_prediction = self.self_only_intercept.unsqueeze(0).expand_as(
            prediction
        ).clone()
        for lag in range(1, self.lags + 1):
            history = standardized[rows - lag]
            prediction += history * self.self_coef[:, lag - 1]
            self_prediction += history * self.self_only_coef[:, lag - 1]
        for target in range(n_nodes):
            for slot in range(self.topk):
                parent = int(self.parent_index[target, slot].item())
                if parent < 0:
                    continue
                for lag in range(1, self.lags + 1):
                    prediction[:, target] += (
                        standardized[rows - lag, parent]
                        * self.parent_coef[target, slot, lag - 1]
                    )

        target = standardized.index_select(0, rows)
        target_finite = finite.index_select(0, rows)
        nan = series.new_tensor(float("nan"))
        residual = torch.where(target_finite, target - prediction, nan)
        self_residual = torch.where(
            target_finite,
            target - self_prediction,
            nan,
        )

        def residual_statistics(values):
            center = torch.zeros(n_nodes, dtype=series.dtype)
            scale = torch.full_like(center, float("nan"))
            valid = torch.zeros(n_nodes, dtype=torch.bool)
            normalized = torch.full_like(values, float("nan"))
            for node in range(n_nodes):
                node_values = values[:, node]
                finite_values = node_values[torch.isfinite(node_values)]
                if finite_values.numel() < self.min_samples:
                    continue
                median = torch.quantile(finite_values, 0.5)
                q25, q75 = torch.quantile(
                    finite_values,
                    torch.tensor((0.25, 0.75), dtype=finite_values.dtype),
                )
                robust_scale = (q75 - q25) / 1.349
                std_scale = finite_values.std(unbiased=False)
                node_scale = (
                    robust_scale
                    if robust_scale >= self.min_residual_scale
                    else std_scale
                )
                if (
                    not torch.isfinite(node_scale)
                    or node_scale < self.min_residual_scale
                ):
                    continue
                center[node] = median
                scale[node] = node_scale
                normalized[:, node] = (node_values - median) / node_scale
                valid[node] = True
            return center, scale, valid, normalized

        residual_center, residual_scale, residual_valid, normalized = (
            residual_statistics(residual)
        )
        (
            self_residual_center,
            self_residual_scale,
            self_residual_valid,
            normalized_self,
        ) = residual_statistics(self_residual)

        reference_max_window = min(
            self.reference_max_window,
            normalized.size(0),
        )
        score_reference = torch.full(
            (reference_max_window, n_nodes, self.reference_size),
            float("nan"),
            dtype=series.dtype,
        )
        score_tail_valid = torch.zeros(
            reference_max_window,
            n_nodes,
            dtype=torch.bool,
        )
        positive_tail_reference = torch.zeros_like(score_tail_valid)
        for score_length in range(1, reference_max_window + 1):
            rolling_scores = normalized.unfold(0, score_length, 1)
            rolling_scores = rolling_scores.square().mean(dim=-1).sqrt()
            for node in range(n_nodes):
                values = self._reference_sample(
                    rolling_scores[:, node],
                    self.reference_size,
                )
                if values.numel() < 4:
                    continue
                score_reference[score_length - 1, node, :values.numel()] = values
                calibration_values, uses_positive_tail = (
                    self._calibration_reference(
                        values,
                        bool(self.discrete_mask[node].item()),
                    )
                )
                if calibration_values.numel() < 4:
                    continue
                q80, q90, q995 = torch.quantile(
                    calibration_values,
                    torch.tensor(
                        (0.80, 0.90, 0.995),
                        dtype=calibration_values.dtype,
                    ),
                )
                score_tail_valid[score_length - 1, node] = bool(
                    torch.isfinite(q995) and q995 > max(float(q80), float(q90))
                )
                positive_tail_reference[score_length - 1, node] = (
                    uses_positive_tail
                )

        self_score_reference = torch.full_like(score_reference, float("nan"))
        self_score_tail_valid = torch.zeros_like(score_tail_valid)
        self_positive_tail_reference = torch.zeros_like(score_tail_valid)
        for score_length in range(1, reference_max_window + 1):
            rolling_scores = normalized_self.unfold(0, score_length, 1)
            rolling_scores = rolling_scores.square().mean(dim=-1).sqrt()
            for node in range(n_nodes):
                values = self._reference_sample(
                    rolling_scores[:, node],
                    self.reference_size,
                )
                if values.numel() < 4:
                    continue
                self_score_reference[
                    score_length - 1,
                    node,
                    :values.numel(),
                ] = values
                calibration_values, uses_positive_tail = (
                    self._calibration_reference(
                        values,
                        bool(self.discrete_mask[node].item()),
                    )
                )
                if calibration_values.numel() < 4:
                    continue
                q80, q90, q995 = torch.quantile(
                    calibration_values,
                    torch.tensor(
                        (0.80, 0.90, 0.995),
                        dtype=calibration_values.dtype,
                    ),
                )
                self_score_tail_valid[score_length - 1, node] = bool(
                    torch.isfinite(q995) and q995 > max(float(q80), float(q90))
                )
                self_positive_tail_reference[score_length - 1, node] = (
                    uses_positive_tail
                )

        support_width = self.support_high - self.support_low
        structural_valid = (
            torch.isfinite(self.node_mean)
            & torch.isfinite(self.node_scale)
            & (self.node_scale >= self.min_node_scale)
            & torch.isfinite(support_width)
            & (support_width >= self.min_node_scale)
        )
        previous_valid_nodes = self.valid_nodes.clone()
        valid_nodes = structural_valid & residual_valid & score_tail_valid.any(dim=0)
        self_temporal_valid_nodes = (
            structural_valid
            & self_residual_valid
            & self_score_tail_valid.any(dim=0)
        )
        self.residual_center = residual_center.float()
        self.residual_scale = residual_scale.float()
        self.score_reference = score_reference.float()
        self.positive_tail_reference = positive_tail_reference
        self.self_score_reference = self_score_reference.float()
        self.self_positive_tail_reference = self_positive_tail_reference
        self.self_temporal_valid_nodes = self_temporal_valid_nodes
        self.valid_nodes = valid_nodes
        self._device_cache = {}

        valid_scales = self.residual_scale[valid_nodes]
        parent_counts = (self.parent_index >= 0).sum(dim=1)
        self._diagnostics.update(
            {
                "valid_node_count": int(valid_nodes.sum().item()),
                "valid_node_ratio": float(valid_nodes.float().mean().item()),
                "nodes_with_parent_count": int(
                    (valid_nodes & (parent_counts > 0)).sum().item()
                ),
                "score_reference_max_window": reference_max_window,
                "residual_scale_min": self._safe_stat(valid_scales, "min"),
                "residual_scale_mean": self._safe_stat(valid_scales, "mean"),
                "residual_scale_max": self._safe_stat(valid_scales, "max"),
                "self_residual_valid_node_count": int(
                    (valid_nodes & self.self_residual_valid).sum().item()
                ),
                "self_temporal_valid_node_count": int(
                    self_temporal_valid_nodes.sum().item()
                ),
                "self_temporal_fallback_node_count": int(
                    (self_temporal_valid_nodes & ~valid_nodes).sum().item()
                ),
                "eligible_node_counts": {
                    "lag_break": int(
                        (valid_nodes & self.lag_offset_valid).sum().item()
                    ),
                    "neighbor_break": int(
                        (valid_nodes & (parent_counts > 0)).sum().item()
                    ),
                    "relation_flip": int(
                        (valid_nodes & (parent_counts > 0)).sum().item()
                    ),
                    "local_function_shift": int(valid_nodes.sum().item()),
                },
                "recalibration": {
                    "applied": True,
                    "parameter_update": False,
                    "sample_count": int(n_time),
                    "residual_row_count": int(rows.numel()),
                    "previous_valid_node_count": int(
                        previous_valid_nodes.sum().item()
                    ),
                    "valid_node_count": int(valid_nodes.sum().item()),
                    "gained_valid_nodes": torch.where(
                        valid_nodes & ~previous_valid_nodes
                    )[0].tolist(),
                    "lost_valid_nodes": torch.where(
                        previous_valid_nodes & ~valid_nodes
                    )[0].tolist(),
                },
            }
        )
        return dict(self._diagnostics)

    @staticmethod
    def _safe_stat(values, operation):
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return float("nan")
        if operation == "min":
            return float(values.min().item())
        if operation == "max":
            return float(values.max().item())
        return float(values.mean().item())

    @staticmethod
    def _optional_float(value):
        if not bool(torch.isfinite(value).item()):
            return None
        return float(value.item())

    def diagnostics(self):
        return dict(self._diagnostics)

    def seed_relevance(self, device, dtype=torch.float32):
        """返回只依赖正常训练数据的跨通道机制相关性先验。

        当加入已学习的父节点机制后，节点相对“仅自身历史”预测器的正常残差尺度
        得到降低，该节点才具有正相关性。无效、无父节点或没有改善的节点相关性
        均为零。返回张量在拟合或重新校准后保持固定，不向检测器传播梯度。
        """
        state = self._state(device, dtype)
        if state is None:
            return torch.empty(0, device=device, dtype=dtype)
        full_scale = state["residual_scale"]
        self_scale = state["self_residual_scale"]
        valid = (
            state["valid_nodes"]
            & state["self_residual_valid"]
            & (state["parent_index"] >= 0).any(dim=1)
            & torch.isfinite(full_scale)
            & torch.isfinite(self_scale)
            & (self_scale >= self.min_residual_scale)
        )
        relevance = torch.zeros_like(full_scale)
        relevance[valid] = (
            1.0 - full_scale[valid] / self_scale[valid]
        ).clamp_min(0.0)
        return relevance.detach()

    def _state(self, device, dtype):
        if not self.fitted:
            return None
        key = (device.type, device.index, dtype)
        if key not in self._device_cache:
            self._device_cache[key] = {
                # These fitted indices/flags only control Python branches. Keep
                # host copies alongside the device tensors so autoregressive
                # generation does not synchronize CUDA for every parent/step.
                # fit() and recalibrate() invalidate this entire cache.
                "parent_index_host": tuple(
                    tuple(row) for row in self.parent_index.detach().cpu().tolist()
                ),
                "valid_nodes_host": tuple(self.valid_nodes.detach().cpu().tolist()),
                "self_temporal_valid_nodes_host": tuple(
                    self.self_temporal_valid_nodes.detach().cpu().tolist()
                ),
                "discrete_mask_host": tuple(self.discrete_mask.detach().cpu().tolist()),
                "discrete_state_count_host": tuple(
                    self.discrete_state_count.detach().cpu().tolist()
                ),
                "lag_offset_valid_host": tuple(
                    self.lag_offset_valid.detach().cpu().tolist()
                ),
                "mean": self.node_mean.to(device=device, dtype=dtype),
                "scale": self.node_scale.to(device=device, dtype=dtype),
                "support_low": self.support_low.to(device=device, dtype=dtype),
                "support_high": self.support_high.to(device=device, dtype=dtype),
                "discrete_mask": self.discrete_mask.to(device=device),
                "discrete_state_count": self.discrete_state_count.to(
                    device=device
                ),
                "discrete_states": self.discrete_states.to(
                    device=device,
                    dtype=dtype,
                ),
                "parent_index": self.parent_index.to(device=device),
                "self_coef": self.self_coef.to(device=device, dtype=dtype),
                "parent_coef": self.parent_coef.to(device=device, dtype=dtype),
                "intercept": self.intercept.to(device=device, dtype=dtype),
                "self_only_coef": self.self_only_coef.to(
                    device=device,
                    dtype=dtype,
                ),
                "self_only_intercept": self.self_only_intercept.to(
                    device=device,
                    dtype=dtype,
                ),
                "residual_center": self.residual_center.to(device=device, dtype=dtype),
                "residual_scale": self.residual_scale.to(device=device, dtype=dtype),
                "self_residual_center": self.self_residual_center.to(
                    device=device,
                    dtype=dtype,
                ),
                "self_residual_scale": self.self_residual_scale.to(
                    device=device,
                    dtype=dtype,
                ),
                "self_residual_valid": self.self_residual_valid.to(device=device),
                "score_reference": self.score_reference.to(device=device, dtype=dtype),
                "self_score_reference": self.self_score_reference.to(
                    device=device,
                    dtype=dtype,
                ),
                "lag_offset_prediction_delta": self.lag_offset_prediction_delta.to(
                    device=device,
                    dtype=dtype,
                ),
                "lag_offset_valid": self.lag_offset_valid.to(device=device),
                "valid_nodes": self.valid_nodes.to(device=device),
                "self_temporal_valid_nodes": self.self_temporal_valid_nodes.to(
                    device=device
                ),
            }
        return self._device_cache[key]

    def support(self, seed, device, dtype):
        state = self._state(device, dtype)
        if state is None or seed < 0 or seed >= self.n_nodes:
            nan = torch.tensor(float("nan"), device=device, dtype=dtype)
            return {"low": nan, "high": nan, "collapsed": True}
        low = state["support_low"][seed]
        high = state["support_high"][seed]
        collapsed = not bool(
            torch.isfinite(low)
            and torch.isfinite(high)
            and high - low >= self.min_node_scale
        )
        support = {"low": low, "high": high, "collapsed": collapsed}
        if self.adaptive:
            discrete = state["discrete_mask_host"][seed]
            support["discrete"] = discrete
            support["delta_scale"] = (
                (high - low).abs().clamp_min(self.min_node_scale)
                if discrete
                else state["scale"][seed]
            )
            if discrete:
                n_states = state["discrete_state_count_host"][seed]
                support["states"] = state["discrete_states"][seed, :n_states]
        return support

    def eligible_nodes(
        self,
        anomaly_type,
        device,
        include_self_temporal=False,
    ):
        state = self._state(device, torch.float32)
        if state is None:
            return torch.empty(0, dtype=torch.long, device=device)
        eligible = state["valid_nodes"].clone()
        if anomaly_type == "lag_break" and self.adaptive:
            eligible &= state["lag_offset_valid"]
        elif anomaly_type in ("neighbor_break", "relation_flip"):
            eligible &= (state["parent_index"] >= 0).any(dim=1)
        elif anomaly_type == "local_function_shift" and include_self_temporal:
            eligible |= state["self_temporal_valid_nodes"]
        return torch.where(eligible)[0]

    def uses_self_temporal_fallback(self, seed, device):
        state = self._state(device, torch.float32)
        if state is None or seed < 0 or seed >= self.n_nodes:
            return False
        return bool(
            state["self_temporal_valid_nodes_host"][seed]
            and not state["valid_nodes_host"][seed]
        )

    def affected_context_mask(self, seed, device):
        """Nodes whose local equation consumes the intervened node as a parent."""
        state = self._state(device, torch.float32)
        if state is None or seed < 0 or seed >= self.n_nodes:
            return torch.zeros(self.n_nodes, dtype=torch.bool, device=device)
        affected = (state["parent_index"] == int(seed)).any(dim=1)
        affected &= state["valid_nodes"]
        affected[seed] = False
        return affected

    def _predict_seed(
        self,
        standardized,
        seed,
        positions,
        state,
        coefficients=None,
        lag_offset=0,
    ):
        if coefficients is None:
            self_coef = state["self_coef"][seed]
            parent_coef = state["parent_coef"][seed]
            intercept = state["intercept"][seed]
        else:
            self_coef, parent_coef, intercept = coefficients
        prediction = intercept.expand(
            standardized.size(0), positions.numel()
        ).clone()
        for lag in range(1, self.lags + 1):
            prediction += (
                standardized[:, positions - lag - lag_offset, seed]
                * self_coef[lag - 1]
            )
        parents = state["parent_index_host"][seed]
        for slot in range(self.topk):
            parent = parents[slot]
            if parent < 0:
                continue
            for lag in range(1, self.lags + 1):
                    prediction += (
                        standardized[:, positions - lag - lag_offset, parent]
                        * parent_coef[slot, lag - 1]
                    )
        return prediction

    def _evidence_residual_components(self, windows):
        """Compute the residual terms shared by mechanism evidence paths.

        Positive-view screening only needs ``conditional_rms`` and
        ``self_temporal_rms``. Keeping their construction in one helper makes
        the fast path numerically identical to :meth:`evidence_windows` while
        allowing it to skip peak, top-q and support-distance calculations.
        """
        if windows.dim() == 2:
            windows = windows.unsqueeze(0)
        state = self._state(windows.device, windows.dtype)
        if (
            state is None
            or windows.dim() != 3
            or windows.size(2) != self.n_nodes
        ):
            raise ValueError("Mechanism evidence windows must have shape [B, T, N]")

        score_end = windows.size(1)
        score_start = self.lags
        if self.score_window > 0:
            score_start = max(score_start, score_end - self.score_window)
        if score_end <= score_start:
            raise ValueError("Window is too short for Graph-VAR evidence")

        positions = torch.arange(
            score_start,
            score_end,
            device=windows.device,
        )
        standardized = (windows - state["mean"]) / state["scale"]
        prediction = state["intercept"].view(1, 1, -1).expand(
            windows.size(0),
            positions.numel(),
            -1,
        ).clone()
        self_prediction = state["self_only_intercept"].view(1, 1, -1).expand_as(
            prediction
        ).clone()
        for lag in range(1, self.lags + 1):
            own_history = standardized[:, positions - lag, :]
            prediction += own_history * state["self_coef"][:, lag - 1]
            self_prediction += own_history * state["self_only_coef"][:, lag - 1]
        for slot in range(self.topk):
            parents = state["parent_index"][:, slot]
            parent_valid = parents >= 0
            safe_parents = parents.clamp_min(0)
            parent_history = standardized.index_select(2, safe_parents)
            for lag in range(1, self.lags + 1):
                prediction += (
                    parent_history[:, positions - lag, :]
                    * state["parent_coef"][:, slot, lag - 1]
                    * parent_valid.to(windows.dtype)
                )

        target = standardized[:, positions, :]
        conditional_residual = (
            target
            - prediction
            - state["residual_center"].view(1, 1, -1)
        ) / state["residual_scale"].view(1, 1, -1)
        self_residual = (
            target
            - self_prediction
            - state["self_residual_center"].view(1, 1, -1)
        ) / state["self_residual_scale"].view(1, 1, -1)
        conditional_rms = conditional_residual.square().mean(dim=1).sqrt()
        self_temporal_rms = self_residual.square().mean(dim=1).sqrt()

        return {
            "windows": windows,
            "state": state,
            "positions": positions,
            "conditional_residual": conditional_residual,
            "conditional_rms": conditional_rms,
            "self_temporal_rms": self_temporal_rms,
            "valid": state["valid_nodes"],
            "self_valid": state["valid_nodes"] & state["self_residual_valid"],
        }

    def positive_selection_evidence_windows(self, windows):
        """Return only the train-normal evidence needed for positive selection.

        This path is deliberately restricted to the two quantities consumed
        by GFM4MTSAD positive-view screening. It skips conditional peak/top-q and
        support-distance (including discrete-state) calculations, which are
        still computed by :meth:`evidence_windows` for other consumers.
        """
        components = self._evidence_residual_components(windows)
        conditional_rms = components["conditional_rms"]
        self_temporal_rms = components["self_temporal_rms"]
        valid = components["valid"]
        self_valid = components["self_valid"]
        nan = components["windows"].new_tensor(float("nan"))
        return {
            "conditional_rms": torch.where(
                valid.unsqueeze(0), conditional_rms, nan
            ),
            "relation_increment": torch.where(
                self_valid.unsqueeze(0),
                torch.relu(conditional_rms - self_temporal_rms),
                nan,
            ),
        }

    def evidence_windows(self, windows):
        """Return vectorized normal-mechanism evidence for every graph node."""
        components = self._evidence_residual_components(windows)
        windows = components["windows"]
        state = components["state"]
        positions = components["positions"]
        conditional_residual = components["conditional_residual"]
        conditional_rms = components["conditional_rms"]
        self_temporal_rms = components["self_temporal_rms"]
        valid = components["valid"]
        self_valid = components["self_valid"]
        conditional_peak = conditional_residual.abs().amax(dim=1)
        conditional_topq_rms = temporal_topq_rms(
            conditional_residual,
            self.temporal_topq_fraction,
        )
        relation_increment = torch.relu(conditional_rms - self_temporal_rms)

        values = windows[:, positions, :]
        support_distance = (
            torch.relu(state["support_low"].view(1, 1, -1) - values)
            + torch.relu(values - state["support_high"].view(1, 1, -1))
        ) / state["scale"].view(1, 1, -1).clamp_min(self.min_node_scale)
        discrete_nodes = torch.where(state["discrete_mask"])[0]
        for node in discrete_nodes.tolist():
            n_states = state["discrete_state_count_host"][node]
            states = state["discrete_states"][node, :n_states]
            nearest = (
                values[:, :, node].unsqueeze(-1) - states.view(1, 1, -1)
            ).abs().amin(dim=-1)
            support_distance[:, :, node] = nearest / state["scale"][node].clamp_min(
                self.min_node_scale
            )
        support_distance = support_distance.square().mean(dim=1).sqrt()

        nan = windows.new_tensor(float("nan"))
        evidence = {
            "conditional_rms": torch.where(valid.unsqueeze(0), conditional_rms, nan),
            "conditional_peak": torch.where(valid.unsqueeze(0), conditional_peak, nan),
            "conditional_topq_rms": torch.where(
                valid.unsqueeze(0),
                conditional_topq_rms,
                nan,
            ),
            "self_temporal_rms": torch.where(
                self_valid.unsqueeze(0),
                self_temporal_rms,
                nan,
            ),
            "relation_increment": torch.where(
                self_valid.unsqueeze(0),
                relation_increment,
                nan,
            ),
            "support_distance": torch.where(
                valid.unsqueeze(0),
                support_distance,
                nan,
            ),
        }
        return evidence

    def score_windows(
        self,
        windows,
        seed,
        start,
        end,
        self_temporal=False,
    ):
        if windows.dim() == 2:
            windows = windows.unsqueeze(0)
        state = self._state(windows.device, windows.dtype)
        invalid = windows.new_full((windows.size(0),), float("nan"))
        if (
            state is None
            or windows.size(2) != self.n_nodes
            or seed < 0
            or seed >= self.n_nodes
            or not (
                state["self_temporal_valid_nodes_host"][seed]
                if self_temporal
                else state["valid_nodes_host"][seed]
            )
        ):
            return invalid, {"context_valid": torch.zeros_like(invalid, dtype=torch.bool)}
        score_start = max(int(start), self.lags)
        score_end = min(windows.size(1), int(end) + self.lags)
        if self.score_window > 0:
            score_start = max(score_start, score_end - self.score_window)
        if score_end <= score_start:
            return invalid, {"context_valid": torch.zeros_like(invalid, dtype=torch.bool)}
        positions = torch.arange(score_start, score_end, device=windows.device)
        standardized = (windows - state["mean"]) / state["scale"]
        if self_temporal:
            prediction = state["self_only_intercept"][seed].expand(
                windows.size(0),
                positions.numel(),
            ).clone()
            for lag in range(1, self.lags + 1):
                prediction += (
                    standardized[:, positions - lag, seed]
                    * state["self_only_coef"][seed, lag - 1]
                )
            residual_center = state["self_residual_center"][seed]
            residual_scale = state["self_residual_scale"][seed]
        else:
            prediction = self._predict_seed(standardized, seed, positions, state)
            residual_center = state["residual_center"][seed]
            residual_scale = state["residual_scale"][seed]
        residual = (
            standardized[:, positions, seed]
            - prediction
            - residual_center
        ) / residual_scale
        score = residual.square().mean(dim=1).sqrt()
        valid = torch.isfinite(residual).all(dim=1) & torch.isfinite(score)
        score = torch.where(valid, score, invalid)
        return score, {
            "score": score,
            "base_score": score,
            "context_valid": valid,
            "orientation": torch.zeros_like(score),
            "normalized_residual": residual,
            "score_length": int(positions.numel()),
        }

    def score_counterfactual(
        self,
        original,
        candidate,
        seed,
        start,
        end,
        self_temporal=False,
    ):
        original_score, original_details = self.score_windows(
            original,
            seed,
            start,
            end,
            self_temporal=self_temporal,
        )
        candidate_score, candidate_details = self.score_windows(
            candidate,
            seed,
            start,
            end,
            self_temporal=self_temporal,
        )
        original_residual = original_details.get("normalized_residual")
        candidate_residual = candidate_details.get("normalized_residual")
        if original_residual is None or candidate_residual is None:
            invalid = original_score.new_full(original_score.shape, float("nan"))
            return invalid, {
                "score": invalid,
                "base_score": original_score,
                "raw_candidate_score": candidate_score,
                "intervention_discrepancy": invalid,
                "context_valid": torch.zeros_like(invalid, dtype=torch.bool),
                "orientation": torch.zeros_like(invalid),
            }
        intervention = candidate_residual - original_residual
        discrepancy = intervention.square().mean(dim=1).sqrt()
        score = (
            original_residual.square() + intervention.square()
        ).mean(dim=1).sqrt()
        valid = (
            original_details["context_valid"]
            & candidate_details["context_valid"]
            & torch.isfinite(discrepancy)
            & torch.isfinite(score)
        )
        invalid = score.new_full(score.shape, float("nan"))
        return torch.where(valid, score, invalid), {
            "score": torch.where(valid, score, invalid),
            "base_score": original_score,
            "raw_candidate_score": candidate_score,
            "intervention_discrepancy": torch.where(
                valid,
                discrepancy,
                invalid,
            ),
            "context_valid": valid,
            "orientation": torch.zeros_like(score),
        }

    def boundary(
        self,
        window,
        seed,
        start,
        end,
        target_quantile,
        cap_quantile,
        self_temporal=False,
    ):
        scores, details = self.score_windows(
            window,
            seed,
            start,
            end,
            self_temporal=self_temporal,
        )
        if not bool(details["context_valid"][0].item()):
            return None, "context_invalid"
        state = self._state(window.device, window.dtype)
        score_length = int(details["score_length"])
        reference_key = (
            "self_score_reference" if self_temporal else "score_reference"
        )
        if score_length < 1 or score_length > state[reference_key].size(0):
            return None, "insufficient_reference"
        references = state[reference_key][score_length - 1, seed]
        references = references[torch.isfinite(references)]
        if references.numel() < 4:
            return None, "insufficient_reference"
        references, uses_positive_tail = self._calibration_reference(
            references,
            state["discrete_mask_host"][seed],
        )
        if references.numel() < 4:
            return None, "insufficient_reference"
        target = torch.quantile(references, float(target_quantile))
        cap = torch.quantile(references, float(cap_quantile))
        q25, q75 = torch.quantile(
            references,
            references.new_tensor((0.25, 0.75)),
        )
        iqr_scale = (q75 - q25) / 1.349
        if self.adaptive:
            reference_scale = torch.maximum(
                iqr_scale,
                references.std(unbiased=False),
            ).clamp_min(1e-4)
        else:
            reference_scale = iqr_scale.clamp_min(1e-4)
        if not all(bool(torch.isfinite(value)) for value in (target, cap, reference_scale)):
            return None, "nonfinite"
        if cap <= target:
            return None, "collapsed_tail"
        return {
            "original_score": scores[0].detach(),
            "target": target.detach(),
            "cap": cap.detach(),
            "empirical_cap": cap.detach(),
            "reference_scale": reference_scale.detach(),
            "reference_scale_fallback": False,
            "reference_kind": (
                ("self_temporal_positive_tail" if uses_positive_tail else "self_temporal_full")
                if self_temporal
                else ("positive_tail" if uses_positive_tail else "full")
            ),
        }, "valid"

    def _strongest_parent_slot(self, standardized, seed, start, end, state):
        parents = state["parent_index_host"][seed]
        positions = torch.arange(max(start, self.lags), end, device=standardized.device)
        if positions.numel() == 0:
            return None
        best_slot = None
        best_effect = -math.inf
        for slot in range(self.topk):
            parent = parents[slot]
            if parent < 0:
                continue
            effect = standardized.new_tensor(0.0)
            for lag in range(1, self.lags + 1):
                effect += (
                    standardized[positions - lag, parent]
                    * state["parent_coef"][seed, slot, lag - 1]
                ).abs().mean()
            value = float(effect.item())
            if value > best_effect:
                best_effect = value
                best_slot = slot
        return best_slot

    @staticmethod
    def _discrete_counterfactual(
        original_values,
        raw_values,
        states,
        local_start,
        local_end,
        strength,
    ):
        result = original_values.clone()
        if states.numel() < 2 or local_end <= local_start:
            return result
        original_segment = original_values[local_start:local_end]
        raw_segment = raw_values[local_start:local_end]
        original_state = (
            original_segment.unsqueeze(1) - states.unsqueeze(0)
        ).abs().argmin(dim=1)
        effect = (raw_segment - original_segment).abs()
        if not bool(torch.isfinite(effect).all().item()) or bool(
            (effect.max() <= torch.finfo(effect.dtype).eps).item()
        ):
            return result
        n_positions = effect.numel()
        n_changes = min(
            n_positions,
            max(1, int(math.ceil(max(float(strength), 1e-6) * n_positions))),
        )
        selected = torch.topk(effect, k=n_changes).indices
        for local_idx in selected.tolist():
            current_state = int(original_state[local_idx].item())
            alternatives = torch.arange(states.numel(), device=states.device)
            alternatives = alternatives[alternatives != current_state]
            target = raw_segment[local_idx]
            best = alternatives[
                (states.index_select(0, alternatives) - target).abs().argmin()
            ]
            result[local_start + local_idx] = states[best]
        return result

    def _self_temporal_counterfactual_candidates(
        self,
        window,
        seed,
        start,
        end,
        anomaly_type,
        strengths,
    ):
        if anomaly_type != "local_function_shift":
            return [None] * len(strengths)
        state = self._state(window.device, window.dtype)
        if (
            state is None
            or window.dim() != 2
            or window.size(1) != self.n_nodes
            or seed < 0
            or seed >= self.n_nodes
            or not state["self_temporal_valid_nodes_host"][seed]
        ):
            return [None] * len(strengths)
        generation_start = max(int(start), self.lags)
        generation_end = min(int(end), window.size(0))
        if generation_end <= generation_start:
            return [None] * len(strengths)

        standardized = (window - state["mean"]) / state["scale"]
        positions = torch.arange(
            generation_start,
            generation_end,
            device=window.device,
        )
        original_prediction = state["self_only_intercept"][seed].expand(
            positions.numel()
        ).clone()
        for lag in range(1, self.lags + 1):
            original_prediction += (
                standardized[positions - lag, seed]
                * state["self_only_coef"][seed, lag - 1]
            )
        innovations = standardized[positions, seed] - original_prediction
        direction = -1.0 if float(original_prediction.mean().item()) >= 0.0 else 1.0

        results = []
        for raw_strength in strengths:
            strength = min(max(float(raw_strength), 0.0), 1.0)
            counterfactual_z = standardized.clone()
            shifted_intercept = (
                state["self_only_intercept"][seed] + strength * direction
            )
            for offset, time_idx in enumerate(
                range(generation_start, generation_end)
            ):
                prediction = shifted_intercept.clone()
                for lag in range(1, self.lags + 1):
                    prediction += (
                        counterfactual_z[time_idx - lag, seed]
                        * state["self_only_coef"][seed, lag - 1]
                    )
                counterfactual_z[time_idx, seed] = prediction + innovations[offset]
            values = window[start:end, seed].clone()
            local_start = generation_start - int(start)
            local_end = generation_end - int(start)
            values[local_start:local_end] = (
                counterfactual_z[generation_start:generation_end, seed]
                * state["scale"][seed]
                + state["mean"][seed]
            )
            if self.adaptive and state["discrete_mask_host"][seed]:
                n_states = state["discrete_state_count_host"][seed]
                values = self._discrete_counterfactual(
                    window[start:end, seed],
                    values,
                    state["discrete_states"][seed, :n_states],
                    local_start,
                    local_end,
                    strength,
                )
            results.append(
                values if bool(torch.isfinite(values).all().item()) else None
            )
        return results

    def counterfactual_candidates(self, window, seed, start, end, anomaly_type, strengths,
                                  precompute_parents=False, self_temporal=False):
        """Generate independent strength candidates together, preserving order.

        The recurrence and sum order match counterfactual(). Only the strength
        axis is batched; time remains autoregressive. Kept opt-in until paired
        candidate/gradient/optimizer and GPU checks pass for the target run.
        """
        if isinstance(strengths, torch.Tensor):
            strengths = strengths.detach().cpu().tolist()
        strengths = [min(max(float(s), 0.0), 1.0) for s in strengths]
        count = len(strengths)
        invalid = [None] * count
        if not count or anomaly_type not in self.SUPPORTED_TYPES:
            return invalid
        if self_temporal:
            return self._self_temporal_counterfactual_candidates(
                window,
                seed,
                start,
                end,
                anomaly_type,
                strengths,
            )
        state = self._state(window.device, window.dtype)
        if (state is None or window.dim() != 2 or window.size(1) != self.n_nodes
                or seed < 0 or seed >= self.n_nodes or not state["valid_nodes_host"][seed]):
            return invalid
        adaptive_lag = self.adaptive and anomaly_type == "lag_break"
        generation_start = max(int(start), self.lags + (self.lag_offset_steps if adaptive_lag else 0))
        generation_end = min(int(end), window.size(0))
        if generation_end <= generation_start:
            return invalid
        standardized = (window - state["mean"]) / state["scale"]
        positions = torch.arange(generation_start, generation_end, device=window.device)
        original_prediction = self._predict_seed(standardized.unsqueeze(0), seed, positions, state)[0]
        innovations = standardized[positions, seed] - original_prediction
        self_coef = state["self_coef"][seed].expand(count, -1).clone()
        parent_coef = state["parent_coef"][seed].expand(count, -1, -1).clone()
        intercept = state["intercept"][seed].expand(count).clone()
        strength_tensor = window.new_tensor(strengths)
        if anomaly_type == "lag_break":
            if adaptive_lag:
                if not state["lag_offset_valid_host"][seed]:
                    return invalid
            else:
                if self.lags < 2:
                    return invalid
                self_coef = self_coef + strength_tensor[:, None] * (self_coef.roll(1, dims=1) - self_coef)
                parent_coef = parent_coef + strength_tensor[:, None, None] * (parent_coef.roll(1, dims=2) - parent_coef)
        elif anomaly_type in ("neighbor_break", "relation_flip"):
            slot = self._strongest_parent_slot(standardized, seed, generation_start, generation_end, state)
            if slot is None:
                return invalid
            # Compute scalar multipliers in Python just as the scalar path does.
            multipliers = [1.0 - s if anomaly_type == "neighbor_break" else 1.0 - 2.0*s for s in strengths]
            parent_coef[:, slot] *= window.new_tensor(multipliers)[:, None]
        else:
            direction = -1.0 if float(original_prediction.mean().item()) >= 0.0 else 1.0
            intercept += window.new_tensor([s * direction for s in strengths])

        counterfactual_z = standardized.expand(count, -1, -1).clone()
        parents = state["parent_index_host"][seed]

        if precompute_parents:
            # Only the seed is edited. Other parent histories remain fixed for
            # this invocation; compute products once, never pre-sum them. Keep
            # self-parent terms recurrent too, even for externally fitted states.
            own_coefficients = [self_coef[:, lag, None] for lag in range(self.lags)]
            parent_terms = []
            for slot, parent in enumerate(parents):
                if parent < 0:
                    continue
                for lag in range(1, self.lags + 1):
                    coefficient = parent_coef[:, slot, lag - 1, None]
                    products = None if parent == seed else standardized[:, parent][None, :] * coefficient
                    parent_terms.append((parent, lag, coefficient, products))

            def predict_precomputed(time_idx, lag_offset=0):
                prediction = intercept[:, None].clone()
                for lag, coefficient in enumerate(own_coefficients, 1):
                    position = time_idx - lag - lag_offset
                    prediction += counterfactual_z[:, position:position + 1, seed] * coefficient
                for parent, lag, coefficient, products in parent_terms:
                    position = time_idx - lag - lag_offset
                    if products is None:
                        prediction += counterfactual_z[:, position:position + 1, parent] * coefficient
                    else:
                        prediction += products[:, position:position + 1]
                return prediction[:, 0]

        def predict(single_position, lag_offset=0):
            prediction = intercept[:, None].clone()
            for lag in range(1, self.lags + 1):
                prediction += (counterfactual_z[:, single_position - lag - lag_offset, seed]
                               * self_coef[:, lag - 1, None])
            for slot in range(self.topk):
                parent = parents[slot]
                if parent < 0:
                    continue
                for lag in range(1, self.lags + 1):
                    prediction += (counterfactual_z[:, single_position - lag - lag_offset, parent]
                                   * parent_coef[:, slot, lag - 1, None])
            return prediction[:, 0]

        for offset, time_idx in enumerate(range(generation_start, generation_end)):
            if precompute_parents:
                prediction = predict_precomputed(time_idx)
            else:
                single_position = torch.tensor([time_idx], device=window.device)
                prediction = predict(single_position)
            if adaptive_lag:
                delayed = (predict_precomputed(time_idx, self.lag_offset_steps)
                           if precompute_parents else predict(single_position, self.lag_offset_steps))
                prediction = prediction + strength_tensor * (delayed - prediction)
            counterfactual_z[:, time_idx, seed] = prediction + innovations[offset]
        values = window[start:end, seed].expand(count, -1).clone()
        local_start, local_end = generation_start - int(start), generation_end - int(start)
        values[:, local_start:local_end] = (
            counterfactual_z[:, generation_start:generation_end, seed] * state["scale"][seed]
            + state["mean"][seed])
        results = []
        for index, strength in enumerate(strengths):
            candidate = values[index].clone()
            if self.adaptive and state["discrete_mask_host"][seed]:
                n_states = state["discrete_state_count_host"][seed]
                candidate = self._discrete_counterfactual(window[start:end, seed], candidate,
                    state["discrete_states"][seed, :n_states], local_start, local_end, strength)
            results.append(candidate if bool(torch.isfinite(candidate).all().item()) else None)
        return results

    def counterfactual(
        self,
        window,
        seed,
        start,
        end,
        anomaly_type,
        strength,
        self_temporal=False,
    ):
        if anomaly_type not in self.SUPPORTED_TYPES:
            return None
        if self_temporal:
            return self._self_temporal_counterfactual_candidates(
                window,
                seed,
                start,
                end,
                anomaly_type,
                [strength],
            )[0]
        state = self._state(window.device, window.dtype)
        if (
            state is None
            or window.dim() != 2
            or window.size(1) != self.n_nodes
            or seed < 0
            or seed >= self.n_nodes
            or not state["valid_nodes_host"][seed]
        ):
            return None
        adaptive_lag = self.adaptive and anomaly_type == "lag_break"
        generation_start = max(
            int(start),
            self.lags + (self.lag_offset_steps if adaptive_lag else 0),
        )
        generation_end = min(int(end), window.size(0))
        if generation_end <= generation_start:
            return None

        strength = min(max(float(strength), 0.0), 1.0)
        standardized = (window - state["mean"]) / state["scale"]
        positions = torch.arange(generation_start, generation_end, device=window.device)
        original_prediction = self._predict_seed(
            standardized.unsqueeze(0),
            seed,
            positions,
            state,
        )[0]
        innovations = standardized[positions, seed] - original_prediction

        self_coef = state["self_coef"][seed].clone()
        parent_coef = state["parent_coef"][seed].clone()
        intercept = state["intercept"][seed].clone()
        if anomaly_type == "lag_break":
            if adaptive_lag:
                if not state["lag_offset_valid_host"][seed]:
                    return None
            else:
                if self.lags < 2:
                    return None
                self_coef = self_coef + strength * (
                    self_coef.roll(1) - self_coef
                )
                parent_coef = parent_coef + strength * (
                    parent_coef.roll(1, dims=1) - parent_coef
                )
        elif anomaly_type in ("neighbor_break", "relation_flip"):
            slot = self._strongest_parent_slot(
                standardized,
                seed,
                generation_start,
                generation_end,
                state,
            )
            if slot is None:
                return None
            if anomaly_type == "neighbor_break":
                parent_coef[slot] *= 1.0 - strength
            else:
                parent_coef[slot] *= 1.0 - 2.0 * strength
        else:
            mean_prediction = original_prediction.mean()
            direction = -1.0 if float(mean_prediction.item()) >= 0.0 else 1.0
            intercept += strength * direction

        counterfactual_z = standardized.clone()
        coefficients = (self_coef, parent_coef, intercept)
        for offset, time_idx in enumerate(range(generation_start, generation_end)):
            single_position = torch.tensor([time_idx], device=window.device)
            if adaptive_lag:
                base_prediction = self._predict_seed(
                    counterfactual_z.unsqueeze(0),
                    seed,
                    single_position,
                    state,
                    coefficients=coefficients,
                )[0, 0]
                delayed_prediction = self._predict_seed(
                    counterfactual_z.unsqueeze(0),
                    seed,
                    single_position,
                    state,
                    coefficients=coefficients,
                    lag_offset=self.lag_offset_steps,
                )[0, 0]
                prediction = base_prediction + strength * (
                    delayed_prediction - base_prediction
                )
            else:
                prediction = self._predict_seed(
                    counterfactual_z.unsqueeze(0),
                    seed,
                    single_position,
                    state,
                    coefficients=coefficients,
                )[0, 0]
            counterfactual_z[time_idx, seed] = prediction + innovations[offset]

        values = window[start:end, seed].clone()
        local_start = generation_start - int(start)
        local_end = generation_end - int(start)
        values[local_start:local_end] = (
            counterfactual_z[generation_start:generation_end, seed]
            * state["scale"][seed]
            + state["mean"][seed]
        )
        if self.adaptive and state["discrete_mask_host"][seed]:
            n_states = state["discrete_state_count_host"][seed]
            states = state["discrete_states"][seed, :n_states]
            values = self._discrete_counterfactual(
                window[start:end, seed],
                values,
                states,
                local_start,
                local_end,
                strength,
            )
        if not bool(torch.isfinite(values).all().item()):
            return None
        return values
