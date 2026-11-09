import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

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
