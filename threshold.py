from typing import Optional
from functools import cached_property

import numpy as np
import sklearn.metrics as metrics


class Thresholder:
    def __init__(
            self,
            cfg,
            test_scores: np.ndarray,
            test_labels: np.ndarray,
            train_scores: Optional[np.ndarray] = None,
            val_scores: Optional[np.ndarray] = None
    ):
        self.cfg = cfg
        self.test_scores = test_scores
        self.test_labels = test_labels
        if isinstance(train_scores, np.ndarray):
            self.train_scores = train_scores
        if isinstance(val_scores, np.ndarray):
            self.val_scores = val_scores

        self.threshold_type = cfg.TEST.THRESHOLD.TYPE
        assert self.threshold_type in ('ratio', 'best_f1')

    @cached_property
    def threshold_best_f1(self):
        precision, recall, thresholds = metrics.precision_recall_curve(self.test_labels, self.test_scores)
        best_f1_idx = np.argmax(2 * precision * recall / (precision + recall + 1e-12))
        threshold_best_f1 = thresholds[best_f1_idx]

        return threshold_best_f1

    @cached_property
    def threshold_ratio(self):
        # adapted from: https://github.com/thuml/Anomaly-Transformer/solver.py
        assert hasattr(self, 'val_scores')
        self.ratio = self.cfg.TEST.THRESHOLD.ANOMALY_RATIO
        threshold_ratio = np.percentile(self.val_score, 100 - self.ratio)

        return threshold_ratio

    @property
    def threshold(self):
        if self.threshold_type == 'ratio':
            return self.threshold_ratio
        elif self.threshold_type == 'best_f1':
            return self.threshold_best_f1
