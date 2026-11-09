import numpy as np


VALID_NODE_MAJORITY_THRESHOLD = 0.5
GATE_SIGNAL = "normal_mechanism_valid_node_ratio"
GATE_SOURCE = "normal_training_windows_only"
CAUSAL_SOURCE = "causal_rolling_robust_fixed_1024"
RAW_SOURCE = "anchored_raw"


def normal_validity_reliability_gate(raw_scores, causal_scores, valid_node_ratio):
    """Select one frozen score expert from normal-only mechanism coverage."""
    raw = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    causal = np.asarray(causal_scores, dtype=np.float64).reshape(-1)
    ratio = float(valid_node_ratio)
    if raw.shape != causal.shape:
        raise ValueError("Raw and causal score shapes must match")
    if not np.isfinite(raw).all() or not np.isfinite(causal).all():
        raise ValueError("Reliability gate requires finite score arrays")
    if not np.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise ValueError("Valid-node ratio must be finite and in [0, 1]")

    use_causal = ratio < VALID_NODE_MAJORITY_THRESHOLD
    selected_source = CAUSAL_SOURCE if use_causal else RAW_SOURCE
    selected = causal if use_causal else raw
    return selected.copy(), {
        "gate_signal": GATE_SIGNAL,
        "gate_source": GATE_SOURCE,
        "valid_node_ratio": ratio,
        "majority_threshold": VALID_NODE_MAJORITY_THRESHOLD,
        "selected_source": selected_source,
        "causal_selected": use_causal,
        "test_labels_used": False,
        "test_scores_used_for_selection": False,
        "future_scores_used_for_selection": False,
    }
