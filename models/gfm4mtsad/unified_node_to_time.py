import numpy as np


TEMPORAL_CONSENSUS_WIDTH = 10


def causal_moving_mean(scores, width=TEMPORAL_CONSENSUS_WIDTH):
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    width = int(width)
    if width < 1:
        raise ValueError("Causal moving-mean width must be positive")
    if not np.isfinite(values).all():
        raise ValueError("Causal moving-mean scores must be finite")
    if width == 1:
        return values.copy()
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    indices = np.arange(values.size)
    starts = np.maximum(indices - width + 1, 0)
    counts = indices - starts + 1
    return (cumulative[indices + 1] - cumulative[starts]) / counts


def event_preserving_temporal_consensus(
    scores,
    width=TEMPORAL_CONSENSUS_WIDTH,
):
    """Preserve every current peak while adding causal temporal support."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    return np.maximum(values, causal_moving_mean(values, width=width))
