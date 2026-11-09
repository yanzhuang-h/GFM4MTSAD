from dataclasses import asdict, dataclass
import math

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from utils.oraclead_metrics import (
    evaluate_oraclead_metrics,
    evaluate_oraclead_paper_metrics,
)


EPS = np.finfo(np.float64).eps
PROTOCOL_METADATA_VERSION = "GFM4MTSAD-METRIC-PROTOCOL-METADATA-V2"
ALLOWED_AGGREGATION_LEVELS = frozenset(
    {"entity", "dataset", "pooled_entities", "unspecified"}
)
ALLOWED_LABEL_STAGES = frozenset(
    {
        "window_any",
        "window_endpoint",
        "timestamp",
        "downsampled_block_any_then_window_any",
        "unspecified",
    }
)


@dataclass(frozen=True)
class ThresholdSelection:
    value: float
    source: str
    uses_test_labels: bool
    comparison: str


def _validate_protocol_context(aggregation_level, score_stage, label_stage):
    aggregation_level = str(aggregation_level)
    score_stage = str(score_stage).strip()
    if aggregation_level not in ALLOWED_AGGREGATION_LEVELS:
        raise ValueError(
            "aggregation_level 必须是 entity、dataset、"
            "pooled_entities 或 unspecified"
        )
    if not score_stage:
        raise ValueError("score_stage 不能为空")
    label_stage = str(label_stage).strip()
    if label_stage not in ALLOWED_LABEL_STAGES:
        raise ValueError(
            "label_stage 必须是 window_any、window_endpoint、timestamp、"
            "downsampled_block_any_then_window_any 或 unspecified"
        )
    return aggregation_level, score_stage, label_stage


def _protocol_entry(
    formula_id,
    threshold_source,
    uses_test_labels_for_threshold,
    point_adjustment,
    aggregation_level,
    score_stage,
    label_stage,
):
    """生成不可省略关键协议字段的指标元数据。"""
    return {
        "formula_id": str(formula_id),
        "threshold_source": str(threshold_source),
        "uses_test_labels_for_threshold": bool(
            uses_test_labels_for_threshold
        ),
        "uses_test_labels_for_metric": True,
        "point_adjustment": str(point_adjustment),
        "aggregation_level": aggregation_level,
        "score_stage": score_stage,
        "label_stage": label_stage,
    }


def build_protocol_metadata(
    oracle_selection,
    deployable_selection=None,
    aggregation_level="unspecified",
    score_stage="unspecified_cached_score",
    label_stage="window_any",
):
    """为每个可混淆的指标族登记公式、阈值和 PA 语义。"""
    if not isinstance(oracle_selection, ThresholdSelection):
        raise TypeError("oracle_selection 必须是 ThresholdSelection")
    if (
        deployable_selection is not None
        and not isinstance(deployable_selection, ThresholdSelection)
    ):
        raise TypeError("deployable_selection 必须是 ThresholdSelection 或 None")
    aggregation_level, score_stage, label_stage = _validate_protocol_context(
        aggregation_level,
        score_stage,
        label_stage,
    )

    def entry(
        formula_id,
        threshold_source="none",
        uses_test_labels_for_threshold=False,
        point_adjustment="none",
    ):
        return _protocol_entry(
            formula_id=formula_id,
            threshold_source=threshold_source,
            uses_test_labels_for_threshold=uses_test_labels_for_threshold,
            point_adjustment=point_adjustment,
            aggregation_level=aggregation_level,
            score_stage=score_stage,
            label_stage=label_stage,
        )

    oracle_source = oracle_selection.source
    metrics = {
        "threshold_free.auroc": entry(
            "sklearn.roc_auc_score.raw.v1"
        ),
        "threshold_free.auprc_trapezoid": entry(
            "sklearn.precision_recall_curve+auc.trapezoid.raw.v1"
        ),
        "threshold_free.average_precision": entry(
            "sklearn.average_precision_score.raw.v1"
        ),
        "pate.continuous_auprc": entry(
            "pate.PATE.continuous.v1"
        ),
        "legacy_oracle.carots_raw_curve": entry(
            "carots.test_pr_curve_best_f1.v1",
            threshold_source=oracle_source,
            uses_test_labels_for_threshold=True,
        ),
        "legacy_oracle.threshold_realized_raw": entry(
            "gfm4mtsad.binary_confusion.v1",
            threshold_source=oracle_source,
            uses_test_labels_for_threshold=True,
        ),
        "legacy_oracle.point_adjusted": entry(
            "gfm4mtsad.binary_event_completion.v1",
            threshold_source=oracle_source,
            uses_test_labels_for_threshold=True,
            point_adjustment="binary_event_completion_after_threshold",
        ),
        "legacy_oracle.event": entry(
            "gfm4mtsad.event_overlap.v1",
            threshold_source=oracle_source,
            uses_test_labels_for_threshold=True,
        ),
        "legacy_oracle.operational": entry(
            "gfm4mtsad.false_alarm_rate.v1",
            threshold_source=oracle_source,
            uses_test_labels_for_threshold=True,
        ),
        "legacy_oracle.affiliation": entry(
            "affiliation.pr_from_events.official.v1",
            threshold_source=oracle_source,
            uses_test_labels_for_threshold=True,
        ),
        "oraclead.range_f1": entry(
            "tsb_ad.range_f1.alpha_0.2.v1",
            threshold_source="oracle_test_uniform_score_grid_100",
            uses_test_labels_for_threshold=True,
        ),
        "oraclead.affiliation_f1": entry(
            "affiliation.pr_from_events.official.oracle_grid.v1",
            threshold_source="oracle_test_uniform_score_grid_100",
            uses_test_labels_for_threshold=True,
        ),
        "oraclead.vus_roc": entry(
            "tsb_ad.vus_roc.window_0_to_100.threshold_250.v1"
        ),
        "oraclead.vus_pr": entry(
            "tsb_ad.vus_pr.window_0_to_100.threshold_250.v1"
        ),
        "pgrf.appendix_oracle_1000": entry(
            "pgrf.independent_pa_raw_binary_event_uniform_1000.v2",
            threshold_source="oracle_test_independent_metric_family_uniform_1000",
            uses_test_labels_for_threshold=True,
            point_adjustment="binary_event_completion_after_threshold",
        ),
        "oraclead.paper_200": entry(
            "oraclead.independent_metric_uniform_200.greater_equal.v1",
            threshold_source="oracle_test_metric_specific_uniform_200",
            uses_test_labels_for_threshold=True,
        ),
    }
    deployable_status = "MISSING_THRESHOLD_SOURCE"
    if deployable_selection is not None:
        deployable_status = "COMPUTED"
        source = deployable_selection.source
        uses_test = deployable_selection.uses_test_labels
        metrics.update({
            "deployable.point": entry(
                "gfm4mtsad.binary_confusion.v1",
                threshold_source=source,
                uses_test_labels_for_threshold=uses_test,
            ),
            "deployable.point_adjusted": entry(
                "gfm4mtsad.binary_event_completion.v1",
                threshold_source=source,
                uses_test_labels_for_threshold=uses_test,
                point_adjustment="binary_event_completion_after_threshold",
            ),
            "deployable.event": entry(
                "gfm4mtsad.event_overlap.v1",
                threshold_source=source,
                uses_test_labels_for_threshold=uses_test,
            ),
            "deployable.operational": entry(
                "gfm4mtsad.false_alarm_rate.v1",
                threshold_source=source,
                uses_test_labels_for_threshold=uses_test,
            ),
            "deployable.affiliation": entry(
                "affiliation.pr_from_events.official.v1",
                threshold_source=source,
                uses_test_labels_for_threshold=uses_test,
            ),
            "pate.binary_f1": entry(
                "pate.PATE.binary.v1",
                threshold_source=source,
                uses_test_labels_for_threshold=uses_test,
            ),
        })
    return {
        "schema_version": PROTOCOL_METADATA_VERSION,
        "aggregation_level": aggregation_level,
        "score_stage": score_stage,
        "label_stage": label_stage,
        "deployable_status": deployable_status,
        "metrics": metrics,
    }


def localization_protocol_metadata(
    aggregation_level="unspecified",
    score_stage="unspecified_cached_score",
    label_stage="window_endpoint",
):
    """登记传感器级指标的公式族，明确其不涉及时间点 PA。"""
    aggregation_level, score_stage, label_stage = _validate_protocol_context(
        aggregation_level,
        score_stage,
        label_stage,
    )
    formulas = {
        "direct_hit_at_k": "gfm4mtsad.localization.direct_hit_at_k.v1",
        "direct_mrr": "gfm4mtsad.localization.direct_mrr.v1",
        "direct_ndcg_at_k": "gfm4mtsad.localization.direct_ndcg_at_k.v1",
        "node_auprc": "gfm4mtsad.localization.node_average_precision.v1",
        "affected_hit_recall_ndcg_at_k": (
            "gfm4mtsad.localization.affected_ranking_at_k.v1"
        ),
    }
    return {
        "schema_version": PROTOCOL_METADATA_VERSION,
        "formula_ids": formulas,
        "threshold_source": "none_rank_based",
        "uses_test_labels_for_threshold": False,
        "uses_test_labels_for_metric": True,
        "point_adjustment": "none",
        "aggregation_level": aggregation_level,
        "score_stage": score_stage,
        "label_stage": label_stage,
    }


def _score_vector(values, name="scores"):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 1:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _binary_vector(values, name="labels"):
    values = np.asarray(values).reshape(-1)
    if values.size < 1:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    rounded = np.rint(values).astype(np.int64, copy=False)
    if not np.array_equal(values, rounded):
        raise ValueError(f"{name} must be binary")
    if not set(np.unique(rounded).tolist()).issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0 and 1")
    return rounded


def _validate_pair(labels, scores):
    labels = _binary_vector(labels)
    scores = _score_vector(scores)
    if labels.shape != scores.shape:
        raise ValueError(
            "labels and scores must have the same flattened shape: "
            f"labels={labels.shape}, scores={scores.shape}"
        )
    return labels, scores


def safe_ratio(numerator, denominator):
    if int(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def safe_f1(precision, recall):
    if precision is None or recall is None:
        return None
    denominator = float(precision) + float(recall)
    if denominator <= 0.0:
        return 0.0
    return 2.0 * float(precision) * float(recall) / denominator


def contiguous_positive_runs(values):
    values = _binary_vector(values).astype(bool, copy=False)
    changes = np.diff(np.pad(values.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def parse_smd_interpretation_text(text, node_count):
    """解析 SMD 官方 interpretation-label 文本格式。

    文本中的传感器编号从 1 开始。时间区间沿用 GFM4MTSAD 已审计的半开约定：
    ``start-end`` 覆盖 ``start, ..., end - 1``。
    """
    node_count = int(node_count)
    if node_count < 1:
        raise ValueError("SMD node_count must be positive")
    labels = {}
    for line_number, raw_line in enumerate(str(text).splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            span, nodes = line.split(":", 1)
            start_text, stop_text = span.split("-", 1)
            start = int(start_text)
            stop = int(stop_text)
            node_set = {int(value) - 1 for value in nodes.split(",") if value}
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid SMD interpretation label at line {line_number}: {line}"
            ) from error
        if start < 0 or stop <= start:
            raise ValueError(
                f"Invalid SMD time span at line {line_number}: {span}"
            )
        invalid = sorted(
            node for node in node_set if node < 0 or node >= node_count
        )
        if invalid:
            raise ValueError(
                "SMD interpretation label contains out-of-range zero-based "
                f"nodes at line {line_number}: {invalid}"
            )
        for time_index in range(start, stop):
            labels.setdefault(time_index, set()).update(node_set)
    return labels


def smd_endpoint_targets(
    interpretation,
    window_count,
    node_count,
    window_size=10,
    step=1,
):
    """把 SMD 传感器标签对齐到 GFM4MTSAD/CAROTS 的窗口末端预测。"""
    window_count = int(window_count)
    node_count = int(node_count)
    window_size = int(window_size)
    step = int(step)
    if min(window_count, node_count, window_size, step) < 1:
        raise ValueError(
            "window_count, node_count, window_size, and step must be positive"
        )
    targets = np.zeros((window_count, node_count), dtype=bool)
    for window_index in range(window_count):
        endpoint = window_index * step + window_size - 1
        nodes = interpretation.get(endpoint, ())
        if nodes:
            indices = np.asarray(sorted(nodes), dtype=np.int64)
            if indices.min() < 0 or indices.max() >= node_count:
                raise ValueError(
                    f"SMD endpoint {endpoint} contains out-of-range nodes"
                )
            targets[window_index, indices] = True
    return targets


def point_adjust(predictions, labels):
    predictions = _binary_vector(predictions, "predictions").copy()
    labels = _binary_vector(labels)
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape")
    for start, stop in contiguous_positive_runs(labels):
        if predictions[start:stop].any():
            predictions[start:stop] = 1
    return predictions


def threshold_free_metrics(labels, scores):
    labels, scores = _validate_pair(labels, scores)
    if np.unique(labels).size < 2:
        return {
            "auroc": None,
            "auprc_trapezoid": None,
            "average_precision": None,
        }
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc_trapezoid": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def point_metrics(labels, predictions):
    labels = _binary_vector(labels)
    predictions = _binary_vector(predictions, "predictions")
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    anomaly = labels == 1
    normal = ~anomaly
    predicted = predictions == 1
    tp = int((predicted & anomaly).sum())
    fp = int((predicted & normal).sum())
    tn = int((~predicted & normal).sum())
    fn = int((~predicted & anomaly).sum())
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    npv = safe_ratio(tn, tn + fn)
    denominator = math.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    )
    mcc = None if denominator == 0.0 else (tp * tn - fp * fn) / denominator
    balanced_accuracy = (
        None
        if recall is None or specificity is None
        else 0.5 * (recall + specificity)
    )
    return {
        "true_positive_points": tp,
        "false_positive_points": fp,
        "true_negative_points": tn,
        "false_negative_points": fn,
        "precision": precision,
        "recall": recall,
        "f1": safe_f1(precision, recall),
        "specificity": specificity,
        "negative_predictive_value": npv,
        "false_positive_rate": safe_ratio(fp, fp + tn),
        "false_negative_rate": safe_ratio(fn, fn + tp),
        "accuracy": safe_ratio(tp + tn, labels.size),
        "balanced_accuracy": balanced_accuracy,
        "matthews_correlation": None if mcc is None else float(mcc),
    }


def _best_f1_curve(labels, scores):
    labels, scores = _validate_pair(labels, scores)
    if np.unique(labels).size < 2:
        raise ValueError("best-F1 threshold selection requires both classes")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError("best-F1 threshold selection produced no thresholds")
    f1 = 2.0 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    best = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "f1": float(f1[best]),
    }


def _best_f1_threshold(
    labels,
    scores,
    source,
    uses_test_labels,
    comparison,
):
    best = _best_f1_curve(labels, scores)
    return ThresholdSelection(
        value=best["threshold"],
        source=source,
        uses_test_labels=bool(uses_test_labels),
        comparison=str(comparison),
    )


def oracle_test_best_f1_threshold(labels, scores):
    return _best_f1_threshold(
        labels,
        scores,
        source="oracle_test_best_f1",
        uses_test_labels=True,
        comparison="greater_than",
    )


def validation_best_f1_threshold(labels, scores):
    return _best_f1_threshold(
        labels,
        scores,
        source="validation_best_f1",
        uses_test_labels=False,
        comparison="greater_equal",
    )


def normal_quantile_threshold(reference_scores, quantile=0.995):
    reference_scores = _score_vector(reference_scores, "reference_scores")
    quantile = float(quantile)
    if not 0.0 < quantile < 1.0:
        raise ValueError("normal reference quantile must be in (0, 1)")
    return ThresholdSelection(
        value=float(np.quantile(reference_scores, quantile)),
        source=f"normal_reference_quantile_{quantile:.6f}",
        uses_test_labels=False,
        comparison="greater_equal",
    )


def fixed_threshold(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("fixed threshold must be finite")
    return ThresholdSelection(
        value=value,
        source="fixed_external_threshold",
        uses_test_labels=False,
        comparison="greater_equal",
    )


def predictions_at_threshold(scores, threshold):
    scores = _score_vector(scores)
    value = threshold.value if isinstance(threshold, ThresholdSelection) else threshold
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("threshold must be finite")
    comparison = (
        threshold.comparison
        if isinstance(threshold, ThresholdSelection)
        else "greater_equal"
    )
    if comparison == "greater_than":
        return (scores > value).astype(np.int64)
    if comparison == "greater_equal":
        return (scores >= value).astype(np.int64)
    raise ValueError(f"Unsupported threshold comparison: {comparison}")


def pgrf_pa_f1_threshold(
    labels,
    scores,
    threshold_count=1000,
    source=None,
    uses_test_labels=True,
):
    """按 PGRF-Net 附录 B 的均匀阈值网格选择最高 PA-F1。

    论文公式使用 ``score >= threshold``。为避免在 SWaT 上重复执行 1000 次
    Python 级 PA 扫描，这里利用“命中事件即补全整段”的等价计数进行向量化。
    """
    labels, scores = _validate_pair(labels, scores)
    if np.unique(labels).size < 2:
        raise ValueError("PGRF threshold selection requires both classes")
    threshold_count = int(threshold_count)
    if threshold_count < 1:
        raise ValueError("PGRF threshold count must be positive")
    if source is None:
        source = f"oracle_test_pa_f1_uniform_{threshold_count}"
    thresholds = np.linspace(
        float(scores.min()),
        float(scores.max()),
        threshold_count,
    )
    true_events = contiguous_positive_runs(labels)
    event_maxima = np.asarray(
        [scores[start:stop].max() for start, stop in true_events],
        dtype=np.float64,
    )
    event_lengths = np.asarray(
        [stop - start for start, stop in true_events],
        dtype=np.float64,
    )
    event_hits = event_maxima[:, None] >= thresholds[None, :]
    true_positive = event_lengths @ event_hits
    positive_count = float(labels.sum())
    false_negative = positive_count - true_positive

    normal_scores = np.sort(scores[labels == 0])
    false_positive = normal_scores.size - np.searchsorted(
        normal_scores,
        thresholds,
        side="left",
    )
    precision_denominator = true_positive + false_positive
    precision = np.divide(
        true_positive,
        precision_denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=precision_denominator > 0,
    )
    recall_denominator = true_positive + false_negative
    recall = np.divide(
        true_positive,
        recall_denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=recall_denominator > 0,
    )
    f1_denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        f1_denominator,
        out=np.zeros_like(precision, dtype=np.float64),
        where=f1_denominator > 0,
    )
    best = int(np.nanargmax(f1))
    selection = ThresholdSelection(
        value=float(thresholds[best]),
        source=str(source),
        uses_test_labels=bool(uses_test_labels),
        comparison="greater_equal",
    )
    return selection, {
        "threshold_count": threshold_count,
        "grid": "uniform_min_to_max_inclusive",
        "comparison": "score_greater_equal_threshold",
        "selection_target": "point_adjusted_f1",
        "best_precision_pa": float(precision[best]),
        "best_recall_pa": float(recall[best]),
        "best_f1_pa": float(f1[best]),
        "tie_break": "first_threshold_in_ascending_grid",
    }


def _pgrf_grid_threshold(
    labels,
    scores,
    objective,
    threshold_count=1000,
    source=None,
    uses_test_labels=True,
):
    """按 PGRF 的均匀阈值网格为 raw 或 range 指标独立选阈值。"""
    labels, scores = _validate_pair(labels, scores)
    threshold_count = int(threshold_count)
    if threshold_count < 1:
        raise ValueError("PGRF threshold count must be positive")
    if objective not in {"raw_f1", "binary_event_range_f1"}:
        raise ValueError(f"Unsupported PGRF threshold objective: {objective}")
    if source is None:
        source = f"oracle_test_{objective}_uniform_{threshold_count}"
    thresholds = np.linspace(
        float(scores.min()),
        float(scores.max()),
        threshold_count,
    )
    best_index = None
    best_value = -math.inf
    best_report = None
    for index, threshold in enumerate(thresholds):
        predictions = (scores >= threshold).astype(np.int64)
        if objective == "raw_f1":
            report = point_metrics(labels, predictions)
        else:
            report = event_metrics(labels, predictions)
        value = report["f1"] if objective == "raw_f1" else report["event_f1"]
        value = 0.0 if value is None else float(value)
        if value > best_value:
            best_index = index
            best_value = value
            best_report = report
    assert best_index is not None and best_report is not None
    selection = ThresholdSelection(
        value=float(thresholds[best_index]),
        source=str(source),
        uses_test_labels=bool(uses_test_labels),
        comparison="greater_equal",
    )
    return selection, {
        "threshold_count": threshold_count,
        "grid": "uniform_min_to_max_inclusive",
        "comparison": "score_greater_equal_threshold",
        "selection_target": objective,
        "best_objective_value": best_value,
        "tie_break": "first_threshold_in_ascending_grid",
    }


def _pgrf_family_report(labels, scores, selection, search, family):
    report = thresholded_protocol(
        labels,
        scores,
        selection,
        protocol_name=f"pgrf_{family}_threshold",
    )
    predictions = report.pop("predictions")
    common = {
        "threshold": report["threshold"],
        "search": search,
        "prediction_positive_count": int(predictions.sum()),
    }
    if family == "point_adjusted":
        common.update(report["point_adjusted"])
        return common
    if family == "raw_point":
        common.update(report["point"])
        return common
    if family == "binary_event_range":
        event = report["event"]
        common.update({
            "r_precision": event["event_precision"],
            "r_recall": event["event_recall"],
            "r_f1": event["event_f1"],
            "predicted_event_count": event["predicted_event_count"],
            "true_event_count": event["true_event_count"],
        })
        return common
    raise ValueError(f"Unsupported PGRF metric family: {family}")


def _pgrf_tracks(
    selection_labels,
    selection_scores,
    evaluation_labels,
    evaluation_scores,
    threshold_count,
    source_prefix,
    uses_test_labels,
):
    pa_selection, pa_search = pgrf_pa_f1_threshold(
        selection_labels,
        selection_scores,
        threshold_count=threshold_count,
        source=f"{source_prefix}_pa_f1_uniform_{int(threshold_count)}",
        uses_test_labels=uses_test_labels,
    )
    raw_selection, raw_search = _pgrf_grid_threshold(
        selection_labels,
        selection_scores,
        objective="raw_f1",
        threshold_count=threshold_count,
        source=f"{source_prefix}_raw_f1_uniform_{int(threshold_count)}",
        uses_test_labels=uses_test_labels,
    )
    range_selection, range_search = _pgrf_grid_threshold(
        selection_labels,
        selection_scores,
        objective="binary_event_range_f1",
        threshold_count=threshold_count,
        source=(
            f"{source_prefix}_binary_event_range_f1_uniform_"
            f"{int(threshold_count)}"
        ),
        uses_test_labels=uses_test_labels,
    )
    return {
        "selection_policy": "independent_best_threshold_per_metric_family",
        "point_adjusted": _pgrf_family_report(
            evaluation_labels,
            evaluation_scores,
            pa_selection,
            pa_search,
            family="point_adjusted",
        ),
        "raw_point": _pgrf_family_report(
            evaluation_labels,
            evaluation_scores,
            raw_selection,
            raw_search,
            family="raw_point",
        ),
        "range_event": _pgrf_family_report(
            evaluation_labels,
            evaluation_scores,
            range_selection,
            range_search,
            family="binary_event_range",
        ),
    }


def pgrf_metrics(
    labels,
    scores,
    validation_labels=None,
    validation_scores=None,
    threshold_count=1000,
):
    """同时保存 PGRF 附录 oracle 轨和可选的正文 validation 轨。"""
    labels, scores = _validate_pair(labels, scores)
    validation = None
    if (validation_labels is None) != (validation_scores is None):
        raise ValueError(
            "PGRF validation labels and scores must be supplied together"
        )
    if validation_labels is not None:
        validation_labels, validation_scores = _validate_pair(
            validation_labels,
            validation_scores,
        )
        validation = _pgrf_tracks(
            validation_labels,
            validation_scores,
            labels,
            scores,
            threshold_count=threshold_count,
            source_prefix="validation",
            uses_test_labels=False,
        )
    return {
        "schema_version": "GFM4MTSAD-PGRF-METRICS-V2",
        "paper_protocol_note": (
            "Section 4.3 states validation threshold selection, while Appendix B.2 "
            "describes best-F1 search over the observed score range. Both tracks "
            "are therefore kept separate."
        ),
        "range_formula_note": (
            "Section 4.3 cites the official Tatbul implementation, but Appendix B.3 "
            "defines binary interval matching by any positive overlap. The PGRF "
            "range track follows the appendix formula; OracleAD/TSB-AD partial-credit "
            "range metrics remain a separate protocol."
        ),
        "threshold_family_note": (
            "The paper does not disclose whether PA, raw and range metrics share one "
            "threshold. Each family is optimized independently to avoid silently "
            "forcing the PA threshold onto the other tables."
        ),
        "appendix_oracle_1000": _pgrf_tracks(
            labels,
            scores,
            labels,
            scores,
            threshold_count=threshold_count,
            source_prefix="oracle_test",
            uses_test_labels=True,
        ),
        "main_validation_1000": validation,
        "auc_pr_disclosure": {
            "paper_formula_unlocated": True,
            "auprc_trapezoid": threshold_free_metrics(labels, scores)[
                "auprc_trapezoid"
            ],
            "average_precision": threshold_free_metrics(labels, scores)[
                "average_precision"
            ],
        },
    }


def event_metrics(labels, predictions):
    labels = _binary_vector(labels)
    predictions = _binary_vector(predictions, "predictions")
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    true_events = contiguous_positive_runs(labels)
    predicted_events = contiguous_positive_runs(predictions)

    true_hits = [
        bool(predictions[start:stop].any())
        for start, stop in true_events
    ]
    predicted_hits = [
        bool(labels[start:stop].any())
        for start, stop in predicted_events
    ]
    event_precision = safe_ratio(sum(predicted_hits), len(predicted_events))
    event_recall = safe_ratio(sum(true_hits), len(true_events))
    delays = []
    coverages = []
    for start, stop in true_events:
        event_predictions = predictions[start:stop]
        coverages.append(float(event_predictions.mean()))
        detected = np.flatnonzero(event_predictions)
        if detected.size:
            delays.append(int(detected[0]))
    return {
        "true_event_count": len(true_events),
        "predicted_event_count": len(predicted_events),
        "hit_event_count": int(sum(true_hits)),
        "missed_event_count": int(len(true_events) - sum(true_hits)),
        "false_alarm_event_count": int(
            len(predicted_events) - sum(predicted_hits)
        ),
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": safe_f1(event_precision, event_recall),
        "event_hit_rate": event_recall,
        "mean_event_coverage": (
            float(np.mean(coverages)) if coverages else None
        ),
        "mean_detection_delay_detected_events": (
            float(np.mean(delays)) if delays else None
        ),
        "median_detection_delay_detected_events": (
            float(np.median(delays)) if delays else None
        ),
        "worst_detection_delay_detected_events": (
            max(delays) if delays else None
        ),
    }


def operational_metrics(labels, predictions):
    labels = _binary_vector(labels)
    predictions = _binary_vector(predictions, "predictions")
    point = point_metrics(labels, predictions)
    false_positive = point["false_positive_points"]
    normal_count = int((labels == 0).sum())
    return {
        "false_positive_windows": false_positive,
        "normal_window_count": normal_count,
        "false_alarms_per_10000_normal_windows": (
            10000.0 * false_positive / normal_count
            if normal_count > 0
            else None
        ),
        "false_alarms_per_10000_all_windows": (
            10000.0 * false_positive / labels.size
        ),
    }


def affiliation_metrics(labels, predictions):
    labels = _binary_vector(labels)
    predictions = _binary_vector(predictions, "predictions")
    try:
        from affiliation.generics import convert_vector_to_events
        from affiliation.metrics import pr_from_events
    except ImportError as error:
        return {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }
    predicted_events = convert_vector_to_events(predictions.tolist())
    true_events = convert_vector_to_events(labels.tolist())
    result = pr_from_events(
        predicted_events,
        true_events,
        (0, len(labels)),
    )
    # The official affiliation package uses ``precision``/``recall`` while
    # the vendored TSB-AD implementation used by PaAno exposes the same
    # probabilities as ``Affiliation_Precision``/``Affiliation_Recall``.
    # Accept both schemas so a baseline-local namespace package cannot break
    # the otherwise optional comprehensive metric replay.
    precision_value = result.get("precision", result.get("Affiliation_Precision"))
    recall_value = result.get("recall", result.get("Affiliation_Recall"))
    if precision_value is None or recall_value is None:
        return {
            "available": False,
            "error": (
                "affiliation result is missing precision/recall fields: "
                f"{sorted(result)}"
            ),
        }
    precision = float(precision_value)
    recall = float(recall_value)
    return {
        "available": True,
        "precision": precision,
        "recall": recall,
        "f1": safe_f1(precision, recall),
    }


def pate_metrics(
    labels,
    scores,
    predictions=None,
    early_buffer=100,
    delayed_buffer=100,
):
    labels, scores = _validate_pair(labels, scores)
    try:
        from pate.PATE_metric import PATE
    except ImportError as error:
        return {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
            "early_buffer": int(early_buffer),
            "delayed_buffer": int(delayed_buffer),
        }
    common = {
        "e_buffer": int(early_buffer),
        "d_buffer": int(delayed_buffer),
        "n_jobs": 1,
    }
    result = {
        "available": True,
        "early_buffer": int(early_buffer),
        "delayed_buffer": int(delayed_buffer),
        "continuous_auprc": float(
            PATE(labels, scores, binary_scores=False, **common)
        ),
    }
    if predictions is not None:
        predictions = _binary_vector(predictions, "predictions")
        if predictions.shape != labels.shape:
            raise ValueError("PATE predictions and labels must have the same shape")
        result["binary_f1"] = float(
            PATE(labels, predictions, binary_scores=True, **common)
        )
    else:
        result["binary_f1"] = None
    return result


def thresholded_protocol(labels, scores, selection, protocol_name):
    labels, scores = _validate_pair(labels, scores)
    if not isinstance(selection, ThresholdSelection):
        raise TypeError("selection must be a ThresholdSelection")
    predictions = predictions_at_threshold(scores, selection)
    adjusted = point_adjust(predictions, labels)
    return {
        "protocol": protocol_name,
        "threshold": asdict(selection),
        "point": point_metrics(labels, predictions),
        "point_adjusted": point_metrics(labels, adjusted),
        "event": event_metrics(labels, predictions),
        "operational": operational_metrics(labels, predictions),
        "affiliation": affiliation_metrics(labels, predictions),
        "predictions": predictions,
    }


def metric_catalog():
    return (
        {
            "group": "threshold_free",
            "metrics": "AUROC/AUPRC-trapezoid/AveragePrecision/PATE-AUPRC",
            "formula_ids": (
                "sklearn.roc_auc_score.raw.v1/"
                "sklearn.precision_recall_curve+auc.trapezoid.raw.v1/"
                "sklearn.average_precision_score.raw.v1/"
                "pate.PATE.continuous.v1"
            ),
            "threshold_source": "none",
            "point_adjustment": "none",
            "test_labels_for_threshold": False,
            "role": "ranking quality",
        },
        {
            "group": "legacy_oracle",
            "metrics": "raw P/R/F1 and PA P/R/F1",
            "formula_ids": (
                "carots.test_pr_curve_best_f1.v1/"
                "gfm4mtsad.binary_event_completion.v1"
            ),
            "threshold_source": "oracle_test_best_f1",
            "point_adjustment": "raw_and_binary_event_completion",
            "test_labels_for_threshold": True,
            "role": "CAROTS-compatible diagnostic only",
        },
        {
            "group": "deployable",
            "metrics": "point P/R/F1, specificity, balanced accuracy, MCC",
            "formula_ids": "gfm4mtsad.binary_confusion.v1",
            "threshold_source": "fixed_or_validation_or_normal_reference",
            "point_adjustment": "none",
            "test_labels_for_threshold": False,
            "role": "strict operating-point evaluation",
        },
        {
            "group": "event",
            "metrics": "Event-F1/hit rate/coverage/delay/false alarms",
            "formula_ids": (
                "gfm4mtsad.event_overlap.v1/gfm4mtsad.false_alarm_rate.v1"
            ),
            "threshold_source": "inherits_named_threshold_track",
            "point_adjustment": "none",
            "test_labels_for_threshold": False,
            "role": "event and operational quality",
        },
        {
            "group": "affiliation",
            "metrics": "affiliation precision/recall/F1",
            "formula_ids": "affiliation.pr_from_events.official.v1",
            "threshold_source": "inherits_named_threshold_track",
            "point_adjustment": "none",
            "test_labels_for_threshold": False,
            "role": "local temporal event quality",
        },
        {
            "group": "oraclead_range",
            "metrics": "R-F1",
            "formula_ids": "tsb_ad.range_f1.alpha_0.2.v1",
            "threshold_source": "oracle_test_uniform_score_grid",
            "point_adjustment": "none",
            "test_labels_for_threshold": True,
            "role": "segment-aware diagnostic; not deployable",
        },
        {
            "group": "oraclead_vus",
            "metrics": "V-ROC/V-PR",
            "formula_ids": (
                "tsb_ad.vus_roc.window_0_to_100.threshold_250.v1/"
                "tsb_ad.vus_pr.window_0_to_100.threshold_250.v1"
            ),
            "threshold_source": "none",
            "point_adjustment": "none",
            "test_labels_for_threshold": False,
            "role": "threshold-free temporal ranking diagnostics",
        },
        {
            "group": "pgrf_compatible",
            "metrics": "PA P/R/F1, raw P/R/F1, R-P/R-R/R-F1",
            "formula_ids": (
                "pgrf.independent_pa_raw_binary_event_uniform_1000.v2"
            ),
            "threshold_source": (
                "oracle_test_or_validation_independent_metric_family_uniform_1000"
            ),
            "point_adjustment": "binary_event_completion_after_threshold",
            "test_labels_for_threshold": True,
            "role": (
                "PGRF-Net Appendix B comparison; paper threshold and range-formula "
                "contradictions are disclosed in the report"
            ),
        },
        {
            "group": "oraclead_paper_200",
            "metrics": "F1/R-F1/Aff-F1/A-ROC/A-PR/V-ROC/V-PR",
            "formula_ids": (
                "oraclead.independent_metric_uniform_200.greater_equal.v1"
            ),
            "threshold_source": "oracle_test_metric_specific_uniform_200",
            "point_adjustment": "none",
            "test_labels_for_threshold": True,
            "role": "OracleAD Appendix D.3 exact paper comparison",
        },
        {
            "group": "localization",
            "metrics": "HR@K/MRR/NDCG/node AUPRC",
            "formula_ids": "gfm4mtsad.localization.ranking.v1",
            "threshold_source": "none_rank_based",
            "point_adjustment": "none",
            "test_labels_for_threshold": False,
            "role": "SMD/Fun-TSG fine-grained reporting",
        },
    )


def evaluate_all_metrics(
    labels,
    scores,
    deployable_threshold=None,
    pgrf_validation_labels=None,
    pgrf_validation_scores=None,
    pate_early_buffer=100,
    pate_delayed_buffer=100,
    aggregation_level="unspecified",
    score_stage="unspecified_cached_score",
    label_stage="window_any",
):
    labels, scores = _validate_pair(labels, scores)
    oracle_selection = oracle_test_best_f1_threshold(labels, scores)
    oracle = thresholded_protocol(
        labels,
        scores,
        oracle_selection,
        protocol_name="legacy_oracle_test_best_f1",
    )
    # CAROTS 直接报告 PR 曲线上的最佳 raw P/R/F1；阈值落地时使用
    # ``scores > threshold``。并列分数会使二者不同，因此必须分别保存。
    oracle["carots_raw_curve"] = _best_f1_curve(labels, scores)
    oracle["threshold_realized_raw"] = oracle.pop("point")
    oracle_predictions = oracle.pop("predictions")
    deployable = None
    deployable_predictions = None
    if deployable_threshold is not None:
        deployable = thresholded_protocol(
            labels,
            scores,
            deployable_threshold,
            protocol_name="strict_no_test_label_threshold",
        )
        deployable_predictions = deployable.pop("predictions")
    pate = pate_metrics(
        labels,
        scores,
        predictions=deployable_predictions,
        early_buffer=pate_early_buffer,
        delayed_buffer=pate_delayed_buffer,
    )
    oraclead = evaluate_oraclead_metrics(labels, scores)
    oraclead_paper = evaluate_oraclead_paper_metrics(labels, scores)
    pgrf = pgrf_metrics(
        labels,
        scores,
        validation_labels=pgrf_validation_labels,
        validation_scores=pgrf_validation_scores,
    )
    protocol_metadata = build_protocol_metadata(
        oracle_selection=oracle_selection,
        deployable_selection=deployable_threshold,
        aggregation_level=aggregation_level,
        score_stage=score_stage,
        label_stage=label_stage,
    )
    return {
        "protocol_version": "GFM4MTSAD-COMPREHENSIVE-METRICS-V2",
        "protocol_metadata": protocol_metadata,
        "sample_count": int(labels.size),
        "anomaly_point_count": int(labels.sum()),
        "threshold_free": threshold_free_metrics(labels, scores),
        "pate": pate,
        "oraclead": oraclead,
        "oraclead_paper_200": oraclead_paper,
        "pgrf": pgrf,
        "physdiff": {
            "status": "MISSING_EXACT_SPOT_PREDICTIONS",
            "required_protocol": (
                "Affiliation precision/recall/F1 from PhysDiff SPOT dynamic "
                "predictions, calibrated on the paper's normal split"
            ),
            "warning": (
                "Oracle best-F1 affiliation metrics are not PhysDiff-compatible"
            ),
        },
        "legacy_oracle": oracle,
        "deployable": deployable,
        "deployable_status": (
            "COMPUTED" if deployable is not None else "MISSING_THRESHOLD_SOURCE"
        ),
        "oracle_prediction_positive_count": int(oracle_predictions.sum()),
        "metric_catalog": list(metric_catalog()),
    }
