"""OracleAD 2026 指标集合的可复现实现。

本模块只接收一维窗口分数和二值窗口标签，不参与模型训练。R-F1、V-ROC
和 V-PR 按 TSB-AD 1.5 的公开公式实现；Aff-F1 通过官方 affiliation
公共 API 计算。这样模型运行时不需要导入实验脚本，缓存重放与在线评测也
能复用同一套公式。

需要特别区分两种 PR 面积：SHARK/CAROTS 的 ``AUPRC`` 是梯形面积，
OracleAD/TSB-AD 的 ``A-PR`` 是 Average Precision。本模块同时返回二者，
调用方不得用一个字段覆盖另一个字段。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ORACLEAD_METRIC_PROTOCOL_VERSION = "SHARK-ORACLEAD-METRICS-V1"
TSB_AD_FORMULA_VERSION = "TSB-AD-1.5"
DEFAULT_ORACLE_THRESHOLD_COUNT = 100
DEFAULT_VUS_TOLERANCE = 100
DEFAULT_VUS_THRESHOLD_COUNT = 250


def _binary_vector(values, name: str = "labels") -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    rounded = np.rint(values).astype(np.int64, copy=False)
    if not np.array_equal(values, rounded):
        raise ValueError(f"{name} must be binary")
    if not set(np.unique(rounded).tolist()).issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0 and 1")
    return rounded


def _score_vector(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("scores must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return values


def _validate_pair(labels, scores) -> Tuple[np.ndarray, np.ndarray]:
    labels = _binary_vector(labels)
    scores = _score_vector(scores)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    if np.unique(labels).size < 2:
        raise ValueError("both normal and anomalous labels are required")
    return labels, scores


def _events(values: np.ndarray) -> List[Tuple[int, int]]:
    """返回包含端点的异常区间，和 TSB-AD ``range_convers_new`` 一致。"""
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    changes = np.diff(np.pad(values, (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _flat_weight(index: int, length: int) -> float:
    del index, length
    return 1.0


def _coverage(anomaly_range: Tuple[int, int], predicted_points: np.ndarray) -> float:
    start, stop = anomaly_range
    if stop < start:
        return 0.0
    covered = predicted_points[start : stop + 1]
    return float(covered.sum()) / float(stop - start + 1)


def _cardinality_factor(
    anomaly_range: Tuple[int, int], predicted_ranges: Sequence[Tuple[int, int]]
) -> float:
    start, stop = anomaly_range
    cardinality = sum(
        int(pred_start <= stop and pred_stop >= start)
        for pred_start, pred_stop in predicted_ranges
    )
    return 0.0 if cardinality == 0 else 1.0 / float(cardinality)


def _range_recall(
    labels: np.ndarray,
    predictions: np.ndarray,
    alpha: float,
) -> float:
    """TSB-AD range recall，含 existence 与 overlap/cardinality 两部分。"""
    true_ranges = _events(labels)
    predicted_ranges = _events(predictions)
    if not true_ranges:
        return 0.0
    existence = sum(
        int(predictions[start : stop + 1].any())
        for start, stop in true_ranges
    )
    overlap = sum(
        _coverage((start, stop), predictions)
        * _cardinality_factor((start, stop), predicted_ranges)
        for start, stop in true_ranges
    )
    return float(alpha * existence + (1.0 - alpha) * overlap) / len(true_ranges)


def range_precision_recall_f1(
    labels,
    predictions,
    alpha: float = 0.2,
) -> Dict[str, float]:
    """计算 TSB-AD 的 Range-Precision、Range-Recall 与 R-F1。"""
    labels = _binary_vector(labels)
    predictions = _binary_vector(predictions, "predictions")
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    recall = _range_recall(labels, predictions, alpha)
    precision = _range_recall(predictions, labels, 0.0)
    f1 = 0.0 if precision + recall <= 0.0 else (
        2.0 * precision * recall / (precision + recall)
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _oracle_thresholds(scores: np.ndarray, count: int) -> np.ndarray:
    count = int(count)
    if count < 1:
        raise ValueError("threshold count must be positive")
    return np.linspace(float(scores.min()), float(scores.max()), count)


def _predictions_at_threshold(
    scores: np.ndarray,
    threshold: float,
    comparison: str,
) -> np.ndarray:
    if comparison == "greater_than":
        return (scores > threshold).astype(np.int64)
    if comparison == "greater_equal":
        return (scores >= threshold).astype(np.int64)
    raise ValueError(f"unsupported threshold comparison: {comparison}")


def point_f1_oracle(
    labels,
    scores,
    threshold_count: int = DEFAULT_ORACLE_THRESHOLD_COUNT,
    comparison: str = "greater_than",
) -> Dict[str, float]:
    """在均匀分数网格上独立搜索标准 point-wise F1。"""
    labels, scores = _validate_pair(labels, scores)
    best = None
    for threshold in _oracle_thresholds(scores, threshold_count):
        predictions = _predictions_at_threshold(scores, threshold, comparison)
        true_positive = int(((predictions == 1) & (labels == 1)).sum())
        false_positive = int(((predictions == 1) & (labels == 0)).sum())
        false_negative = int(((predictions == 0) & (labels == 1)).sum())
        precision = (
            0.0
            if true_positive + false_positive == 0
            else true_positive / float(true_positive + false_positive)
        )
        recall = (
            0.0
            if true_positive + false_negative == 0
            else true_positive / float(true_positive + false_negative)
        )
        f1 = (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
        candidate = (f1, precision, recall, float(threshold))
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    return {
        "available": True,
        "precision": best[1],
        "recall": best[2],
        "f1": best[0],
        "threshold": best[3],
        "threshold_count": int(threshold_count),
        "comparison": comparison,
    }


def range_f1_oracle(
    labels,
    scores,
    threshold_count: int = DEFAULT_ORACLE_THRESHOLD_COUNT,
    comparison: str = "greater_than",
) -> Dict[str, float]:
    """在测试分数上按公开协议搜索最高 R-F1（仅用于 oracle 诊断）。"""
    labels, scores = _validate_pair(labels, scores)
    best = None
    for threshold in _oracle_thresholds(scores, threshold_count):
        predictions = _predictions_at_threshold(scores, threshold, comparison)
        current = range_precision_recall_f1(labels, predictions)
        candidate = (current["f1"], current, float(threshold))
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    result = dict(best[1])
    result["threshold"] = best[2]
    result["threshold_count"] = int(threshold_count)
    result["comparison"] = comparison
    result["available"] = True
    return result


def _affiliation_public_api():
    ensure_tsb_ad_path()
    try:
        from affiliation.generics import convert_vector_to_events
        from affiliation.metrics import pr_from_events
        return convert_vector_to_events, pr_from_events
    except ImportError:
        try:
            from TSB_AD.evaluation.affiliation.generics import (
                convert_vector_to_events,
            )
            from TSB_AD.evaluation.affiliation.metrics import pr_from_events
            return convert_vector_to_events, pr_from_events
        except ImportError as error:
            return None, error


def affiliation_f1_oracle(
    labels,
    scores,
    threshold_count: int = DEFAULT_ORACLE_THRESHOLD_COUNT,
    comparison: str = "greater_than",
) -> Dict[str, object]:
    """按官方 affiliation API 搜索最高 Aff-F1（仅用于 oracle 诊断）。"""
    labels, scores = _validate_pair(labels, scores)
    convert_vector_to_events, api = _affiliation_public_api()
    if convert_vector_to_events is None:
        return {
            "available": False,
            "error": f"{type(api).__name__}: {api}",
            "threshold_count": int(threshold_count),
        }
    true_events = convert_vector_to_events(labels.tolist())
    best = None
    try:
        for threshold in _oracle_thresholds(scores, threshold_count):
            predictions = _predictions_at_threshold(scores, threshold, comparison)
            predicted_events = convert_vector_to_events(predictions.tolist())
            raw = api(predicted_events, true_events, (0, len(labels)))
            precision = raw.get("precision", raw.get("Affiliation_Precision"))
            recall = raw.get("recall", raw.get("Affiliation_Recall"))
            if precision is None or recall is None:
                raise KeyError(
                    "official affiliation result lacks precision/recall"
                )
            precision = float(precision)
            recall = float(recall)
            f1 = 0.0 if precision + recall <= 0.0 else (
                2.0 * precision * recall / (precision + recall)
            )
            candidate = (f1, precision, recall, float(threshold))
            if best is None or candidate[0] > best[0]:
                best = candidate
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        return {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
            "threshold_count": int(threshold_count),
        }
    assert best is not None
    return {
        "available": True,
        "precision": best[1],
        "recall": best[2],
        "f1": best[0],
        "threshold": best[3],
        "threshold_count": int(threshold_count),
        "comparison": comparison,
    }


def evaluate_oraclead_paper_metrics(
    labels,
    scores,
    threshold_count: int = 200,
    vus_max_tolerance: int = DEFAULT_VUS_TOLERANCE,
    vus_threshold_count: int = DEFAULT_VUS_THRESHOLD_COUNT,
) -> Dict[str, object]:
    """严格按 OracleAD 附录 D.3 的 200 点、``score >= threshold`` 协议计算。

    论文令每个依赖阈值的目标指标独立选择最佳阈值，因此 Standard-F1、R-F1
    和 Aff-F1 不共享阈值。A-PR 使用 TSB-AD 的 Average Precision 定义。
    """
    labels, scores = _validate_pair(labels, scores)
    threshold_count = int(threshold_count)
    if threshold_count < 1:
        raise ValueError("OracleAD paper threshold count must be positive")
    return {
        "schema_version": "SHARK-ORACLEAD-PAPER-200-V1",
        "source": "OracleAD-2026-Appendix-D.3",
        "threshold_protocol": {
            "threshold_count": threshold_count,
            "grid": "uniform_min_to_max_inclusive",
            "comparison": "score_greater_equal_threshold",
            "selection": "independent_best_threshold_per_target_metric",
            "uses_test_labels": True,
        },
        "standard_f1": point_f1_oracle(
            labels,
            scores,
            threshold_count=threshold_count,
            comparison="greater_equal",
        ),
        "range_f1": range_f1_oracle(
            labels,
            scores,
            threshold_count=threshold_count,
            comparison="greater_equal",
        ),
        "affiliation_f1": affiliation_f1_oracle(
            labels,
            scores,
            threshold_count=threshold_count,
            comparison="greater_equal",
        ),
        "a_roc": float(roc_auc_score(labels, scores)),
        "a_pr": float(average_precision_score(labels, scores)),
        "vus": vus_metrics(
            labels,
            scores,
            max_tolerance=vus_max_tolerance,
            threshold_count=vus_threshold_count,
        ),
    }


def ensure_tsb_ad_path() -> bool:
    """优先把用户下载的 baseline/TSB-AD 加入当前进程导入路径。"""
    source_file = Path(__file__).resolve()
    for parent in source_file.parents:
        candidate = parent / "baseline" / "TSB-AD"
        if (candidate / "TSB_AD").is_dir():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            return True
    return False


def _tsb_ad_get_metrics():
    ensure_tsb_ad_path()
    try:
        from TSB_AD.evaluation.metrics import get_metrics
        return get_metrics
    except ImportError:
        return None


def _sequencing(
    labels: np.ndarray,
    true_ranges: Sequence[Tuple[int, int]],
    window: int,
) -> np.ndarray:
    """TSB-AD ``sequencing`` 的等价实现。"""
    result = labels.astype(np.float64, copy=True)
    length = result.size
    half = int(window) // 2
    for start, stop in true_ranges:
        right = np.arange(stop + 1, min(stop + half + 1, length))
        if right.size:
            result[right] += np.sqrt(1.0 - (right - stop) / float(window))
        left = np.arange(max(start - half, 0), start)
        if left.size:
            result[left] += np.sqrt(1.0 - (start - left) / float(window))
    return np.minimum(np.ones(length, dtype=np.float64), result)


def _new_sequence(
    true_ranges: Sequence[Tuple[int, int]],
    window: int,
    length: int,
) -> List[Tuple[int, int]]:
    half = int(window) // 2
    start = max(true_ranges[0][0] - half, 0)
    merged = []
    for index in range(len(true_ranges) - 1):
        current_stop = true_ranges[index][1]
        next_start = true_ranges[index + 1][0]
        if current_stop + half < next_start - half:
            merged.append((start, min(current_stop + half, length - 1)))
            start = max(next_start - half, 0)
    merged.append((start, min(true_ranges[-1][1] + half, length - 1)))
    return merged


def _threshold_counts(scores: np.ndarray, count: int):
    """返回 TSB-AD 采样阈值、每个阈值的预测点数和降序索引。"""
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sample_indices = np.linspace(0, len(scores) - 1, int(count)).astype(int)
    thresholds = sorted_scores[sample_indices]
    ascending = np.sort(scores)
    counts = len(scores) - np.searchsorted(
        ascending, thresholds, side="left"
    )
    return thresholds, counts, order


def vus_metrics(
    labels,
    scores,
    max_tolerance: int = DEFAULT_VUS_TOLERANCE,
    threshold_count: int = DEFAULT_VUS_THRESHOLD_COUNT,
) -> Dict[str, object]:
    """计算 TSB-AD 1.5 的 VUS-ROC 与 VUS-PR。

    代码使用分数排序和累计权重向量化，但与公开 ``RangeAUC_volume_opt``
    的阈值采样、范围扩展、存在性奖励和积分端点保持一致。
    """
    labels, scores = _validate_pair(labels, scores)
    max_tolerance = int(max_tolerance)
    threshold_count = int(threshold_count)
    if max_tolerance < 0 or threshold_count < 1:
        raise ValueError("max_tolerance must be non-negative and threshold_count positive")
    true_ranges = _events(labels)
    if not true_ranges:
        return {
            "available": False,
            "error": "ground truth contains no anomaly range",
            "max_tolerance": max_tolerance,
            "threshold_count": threshold_count,
        }

    thresholds, predicted_counts, order = _threshold_counts(
        scores, threshold_count
    )
    sorted_labels = labels[order].astype(np.float64)
    cumulative_labels = np.cumsum(sorted_labels)
    core_count = float(labels.sum())
    tolerance_auroc = []
    tolerance_auprc = []
    for window in range(max_tolerance + 1):
        labels_extended = _sequencing(labels, true_ranges, window)
        merged_ranges = _new_sequence(true_ranges, window, len(labels))
        expansion = labels_extended.copy()
        expansion[labels == 1] = 0.0
        cumulative_extended = np.cumsum(expansion[order])
        counts = np.maximum(predicted_counts - 1, 0)
        tp = cumulative_labels[counts] + (
            cumulative_extended[counts]
        )
        n_labels = core_count + cumulative_extended[counts]

        segment_maxima = np.asarray(
            [scores[start : stop + 1].max() for start, stop in merged_ranges],
            dtype=np.float64,
        )
        existence = (segment_maxima[:, None] >= thresholds[None, :]).sum(axis=0)
        existence_ratio = existence / float(len(merged_ranges))
        p_new = (core_count + n_labels) / 2.0
        recall = np.minimum(tp / p_new, 1.0)
        tpr = recall * existence_ratio
        n_new = len(labels) - p_new
        fpr = (predicted_counts - tp) / n_new
        precision = tp / predicted_counts

        tpr_curve = np.concatenate(([0.0], tpr, [1.0]))
        fpr_curve = np.concatenate(([0.0], fpr, [1.0]))
        precision_curve = np.concatenate(([1.0], precision))
        tolerance_auroc.append(
            float(np.dot(
                np.diff(fpr_curve),
                (tpr_curve[1:] + tpr_curve[:-1]) / 2.0,
            ))
        )
        tolerance_auprc.append(
            float(np.dot(
                tpr_curve[1:-1] - tpr_curve[:-2],
                precision_curve[1:],
            ))
        )

    return {
        "available": True,
        "v_roc": float(np.mean(tolerance_auroc)),
        "v_pr": float(np.mean(tolerance_auprc)),
        "max_tolerance": max_tolerance,
        "threshold_count": threshold_count,
        "formula_version": TSB_AD_FORMULA_VERSION,
    }


def evaluate_oraclead_metrics(
    labels,
    scores,
    oracle_threshold_count: int = DEFAULT_ORACLE_THRESHOLD_COUNT,
    vus_max_tolerance: int = DEFAULT_VUS_TOLERANCE,
    vus_threshold_count: int = DEFAULT_VUS_THRESHOLD_COUNT,
) -> Dict[str, object]:
    """一次性计算 R-F1、Aff-F1、V-ROC、V-PR 与 AP。"""
    labels, scores = _validate_pair(labels, scores)
    official_get_metrics = _tsb_ad_get_metrics()
    official = None
    official_error = None
    if official_get_metrics is not None:
        try:
            official = official_get_metrics(
                scores,
                labels,
                slidingWindow=int(vus_max_tolerance),
                pred=None,
                version="opt",
                thre=int(vus_threshold_count),
            )
            required = (
                "AUC-PR",
                "R-based-F1",
                "Affiliation-F",
                "VUS-ROC",
                "VUS-PR",
            )
            official = {name: float(official[name]) for name in required}
            if not all(math.isfinite(value) for value in official.values()):
                raise ValueError("official TSB-AD metrics contain non-finite values")
        except (ArithmeticError, ImportError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
            # 可选官方实现失败时回退到同协议实现，并保留原因供审计。
            official_error = f"{type(error).__name__}: {error}"
    if official is not None:
        effective_oracle_threshold_count = int(DEFAULT_ORACLE_THRESHOLD_COUNT)
        range_report = {
            "available": True,
            "backend": "TSB_AD.evaluation.metrics.get_metrics@1.5",
            "f1": float(official["R-based-F1"]),
            "threshold_count": effective_oracle_threshold_count,
        }
        affiliation_report = {
            "available": True,
            "backend": "TSB_AD.evaluation.metrics.get_metrics@1.5",
            "f1": float(official["Affiliation-F"]),
            "threshold_count": effective_oracle_threshold_count,
        }
        vus_report = {
            "available": True,
            "backend": "TSB_AD.evaluation.metrics.get_metrics@1.5",
            "v_roc": float(official["VUS-ROC"]),
            "v_pr": float(official["VUS-PR"]),
            "max_tolerance": int(vus_max_tolerance),
            "threshold_count": int(vus_threshold_count),
            "formula_version": TSB_AD_FORMULA_VERSION,
        }
        average_precision = float(official["AUC-PR"])
        backend = "TSB_AD.evaluation.metrics.get_metrics@1.5"
    else:
        range_report = range_f1_oracle(
            labels, scores, threshold_count=oracle_threshold_count
        )
        affiliation_report = affiliation_f1_oracle(
            labels, scores, threshold_count=oracle_threshold_count
        )
        vus_report = vus_metrics(
            labels,
            scores,
            max_tolerance=vus_max_tolerance,
            threshold_count=vus_threshold_count,
        )
        average_precision = float(average_precision_score(labels, scores))
        backend = "shark.vectorized_tsb_ad_formula_fallback.v1"
    return {
        "schema_version": ORACLEAD_METRIC_PROTOCOL_VERSION,
        "source": "OracleAD-2026-Appendix-B-plus-TSB-AD-1.5",
        "backend": backend,
        "official_backend_error": official_error,
        "threshold_protocol": {
            "range_f1": "TSB-AD-1.5-default-100-grid",
            "affiliation_f1": "TSB-AD-1.5-default-100-grid",
            "threshold_count": int(
                DEFAULT_ORACLE_THRESHOLD_COUNT
                if official is not None
                else oracle_threshold_count
            ),
            "comparison": "score_greater_than_threshold",
            "paper_appendix_disclosure": "200-grid; differs from TSB-AD-1.5 default 100-grid",
        },
        "vus_protocol": {
            "max_tolerance": int(vus_max_tolerance),
            "threshold_count": int(vus_threshold_count),
            "tolerance_unit": "evaluation_window_endpoint",
        },
        "average_precision": {
            "available": True,
            "value": average_precision,
            "formula_id": "sklearn.average_precision_score.raw.v1",
        },
        "range_f1": range_report,
        "affiliation_f1": affiliation_report,
        "vus": vus_report,
    }


def display_metrics(report: Dict[str, object]) -> Dict[str, object]:
    """把结构化报告转换为 test.txt 使用的五项扩展字段。"""
    affiliation = report["affiliation_f1"]
    vus = report["vus"]
    def finite_or_na(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "NA"
        return value if math.isfinite(value) else "NA"

    return {
        "AP": finite_or_na(report["average_precision"]["value"]),
        "R-F1": finite_or_na(report["range_f1"].get("f1")),
        "Aff-F1": (
            finite_or_na(affiliation.get("f1"))
            if affiliation.get("available")
            else "NA"
        ),
        "V-ROC": (
            finite_or_na(vus.get("v_roc")) if vus.get("available") else "NA"
        ),
        "V-PR": (
            finite_or_na(vus.get("v_pr")) if vus.get("available") else "NA"
        ),
    }
