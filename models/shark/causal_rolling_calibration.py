import numpy as np


REFERENCE_WINDOW = 1024
MIN_HISTORY = 16
CENTER = "median"
SCALE = "mad"
MAD_SCALE = 1.4826
MIN_SCALE = 1e-6
SOURCE = "past_unlabeled_scores_only"
INCLUDE_CURRENT = False


def causal_rolling_robust_calibration(scores):
    """Calibrate each score against only its bounded causal history."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Causal rolling calibration requires a 1D score sequence")
    if not np.isfinite(values).all():
        raise ValueError("Causal rolling calibration requires finite scores")

    calibrated = np.zeros_like(values, dtype=np.float64)
    for index in range(MIN_HISTORY, values.size):
        start = max(0, index - REFERENCE_WINDOW)
        history = values[start:index]
        center = float(np.median(history))
        mad = float(np.median(np.abs(history - center)))
        scale = max(MAD_SCALE * mad, MIN_SCALE)
        calibrated[index] = (values[index] - center) / scale
    return calibrated
