from __future__ import annotations
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

def pgrf_test_minmax(scores: np.ndarray) -> np.ndarray:
    """Mirror PGRF-Net inference.py test-score cleaning and MinMax fitting."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1).copy()
    values[np.isinf(values)] = np.nan
    if np.all(np.isnan(values)):
        return np.zeros_like(values)
    values[np.isnan(values)] = np.nanmean(values)
    return MinMaxScaler().fit_transform(values.reshape(-1, 1)).reshape(-1)



def point_adjust(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Mirror PGRF-Net utils.adjust_predicts without changing its semantics."""
    scores = np.asarray(scores)
    actual = np.asarray(labels).reshape(-1) > 0.1
    if scores.shape[0] != actual.shape[0]:
        raise ValueError("score and label lengths differ")
    prediction = np.asarray(scores > threshold, dtype=bool)
    anomaly_state = False
    for index in range(scores.shape[0]):
        if actual[index] and prediction[index] and not anomaly_state:
            anomaly_state = True
            for previous in range(index, 0, -1):
                if not actual[previous]:
                    break
                prediction[previous] = True
        elif not actual[index]:
            anomaly_state = False
        if anomaly_state:
            prediction[index] = True
    return prediction.astype(np.int64)



def point_metrics(prediction: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction).reshape(-1).astype(np.int64)
    labels = (np.asarray(labels).reshape(-1) > 0.1).astype(np.int64)
    if prediction.size != labels.size:
        raise ValueError("prediction and label lengths differ")
    true_positive = int(np.sum(prediction * labels))
    false_positive = int(np.sum(prediction * (1 - labels)))
    false_negative = int(np.sum((1 - prediction) * labels))
    precision = true_positive / (true_positive + false_positive + 1e-5)
    recall = true_positive / (true_positive + false_negative + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }



def pgrf_best_f1(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    step_num: int = 100,
) -> dict[str, Any]:
    """Mirror PGRF-Net's 100-step test-label threshold search."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = (np.asarray(labels).reshape(-1) > 0.1).astype(np.int64)
    if scores.size == 0 or scores.size != labels.size:
        raise ValueError("best-F1 search requires equal non-empty score/label arrays")
    start = float(np.min(scores))
    end = float(np.max(scores))
    search_step = (end - start) / int(step_num)
    thresholds = (
        np.asarray([start], dtype=np.float64)
        if search_step == 0.0
        else np.arange(start, end, search_step, dtype=np.float64)
    )
    best = None
    for threshold in thresholds:
        adjusted = point_adjust(scores, labels, float(threshold))
        metrics = point_metrics(adjusted, labels)
        if best is None or metrics["f1"] > best["adjusted_f1"]:
            raw = point_metrics(scores > threshold, labels)
            best = {
                "threshold": float(threshold),
                "adjusted_f1": metrics["f1"],
                "adjusted_precision": metrics["precision"],
                "adjusted_recall": metrics["recall"],
                "raw_f1_at_oracle_threshold": raw["f1"],
                "raw_precision_at_oracle_threshold": raw["precision"],
                "raw_recall_at_oracle_threshold": raw["recall"],
            }
    return {
        **best,
        "threshold_candidates": int(thresholds.size),
        "test_labels_used_for_threshold": True,
        "point_adjustment_uses_test_event_boundaries": True,
    }



def continuous_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = (np.asarray(labels).reshape(-1) > 0.1).astype(np.int64)
    if scores.size != labels.size or np.unique(labels).size != 2:
        raise ValueError("continuous metrics require binary labels and equal lengths")
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc_trapezoid": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(labels, scores)),
    }
