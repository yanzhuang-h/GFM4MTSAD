"""Fixed train-normal tail scoring used by the OwlEye main recipe."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

REGISTERED_BLEND = 0.5
REGISTERED_TAIL_PROBABILITY = 0.05
REGISTERED_EVIDENCE_TOPK = 2

def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return values

def upper_tail_surprisal(
    normal_reference: np.ndarray,
    target_scores: np.ndarray,
) -> np.ndarray:
    """Map a scalar score through a train-normal empirical upper-tail CDF."""
    reference = np.sort(_finite_vector(normal_reference, "normal reference"))
    target = _finite_vector(target_scores, "target scores")
    count_less = np.searchsorted(reference, target, side="left")
    count_greater_equal = reference.size - count_less
    probability = (count_greater_equal + 1.0) / (reference.size + 1.0)
    return -np.log(probability)

def evidence_topk_mean(evidence: np.ndarray, topk: int) -> np.ndarray:
    evidence = np.asarray(evidence, dtype=np.float64)
    topk = int(topk)
    if evidence.ndim != 2 or evidence.shape[0] == 0:
        raise ValueError("evidence must be a non-empty [windows, sources] matrix")
    if not np.isfinite(evidence).all():
        raise ValueError("evidence must contain only finite values")
    if topk < 1 or topk > evidence.shape[1]:
        raise ValueError("evidence topk must be in [1, source count]")
    partitioned = np.partition(evidence, kth=evidence.shape[1] - topk, axis=1)
    return partitioned[:, -topk:].mean(axis=1)

def train_tail_gated_consensus(
    scalar_tail: np.ndarray,
    evidence_tail: np.ndarray,
    *,
    blend: float = REGISTERED_BLEND,
    tail_probability: float = REGISTERED_TAIL_PROBABILITY,
) -> np.ndarray:
    """Fuse evidence only inside a fixed train-normal upper-tail region."""
    scalar_tail = _finite_vector(scalar_tail, "scalar tail scores")
    evidence_tail = _finite_vector(evidence_tail, "evidence tail scores")
    if scalar_tail.shape != evidence_tail.shape:
        raise ValueError("scalar and evidence tail scores differ in length")
    blend = float(blend)
    tail_probability = float(tail_probability)
    if not 0.0 <= blend <= 1.0:
        raise ValueError("tail blend must be in [0, 1]")
    if not 0.0 < tail_probability < 1.0:
        raise ValueError("tail probability must be in (0, 1)")

    threshold = -math.log(tail_probability)
    consensus = blend * scalar_tail + (1.0 - blend) * evidence_tail
    gated_consensus = np.maximum(consensus, threshold)
    return np.where(scalar_tail >= threshold, gated_consensus, scalar_tail)

def best_curve_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import precision_recall_curve

    scores = _finite_vector(scores, "tail scores")
    labels = (np.asarray(labels).reshape(-1) > 0.1).astype(np.int64)
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    index = int(np.argmax(f1))
    return {
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
        "threshold_uses_test_labels": True,
    }

def _load_names(path: Path, expected_count: int) -> list[str]:
    values = np.load(path, allow_pickle=True).reshape(-1).tolist()
    if len(values) != expected_count or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError("evidence names do not match evidence matrix")
    return values

def main():
    from evaluation.metrics import continuous_metrics
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--entity', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-arm', default='warm_joint')
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    train = np.load(source/'train_scores.npy', allow_pickle=False)
    test = np.load(source/'test_scores.npy', allow_pickle=False)
    evidence = np.load(source/'test_evidence_protocol_scores.npy', allow_pickle=False)
    names = _load_names(source/'test_evidence_protocol_names.npy', evidence.shape[1])
    if names != ['direct', 'local', 'relation', 'context']:
        raise ValueError('Unexpected evidence order')
    protocol = json.loads((source/'test_metric_protocol.json').read_text(encoding='utf-8'))
    labels_cfg = protocol['label_protocol']
    if labels_cfg['window_size'] != 60 or labels_cfg['window_label'] != 'any anomaly point inside the input window':
        raise ValueError('Expected the registered window-60, window-any protocol')
    scalar = upper_tail_surprisal(train, test)
    consensus = evidence_topk_mean(evidence, REGISTERED_EVIDENCE_TOPK)
    final = train_tail_gated_consensus(scalar, consensus)
    # Labels are read only after score construction, for metric reporting.
    labels = np.load(source/'test_labels.npy', allow_pickle=False).reshape(-1)
    values = continuous_metrics(final, labels)
    metric = {key: values[key] for key in ('auroc', 'auprc_trapezoid')}
    metric['raw_curve'] = best_curve_metrics(final, labels)
    report = {
        'schema_version': 'GFM4MTSAD-TGMC-MAIN-V1',
        'entity': args.entity,
        'source_arm': args.source_arm,
        'tail_probability': REGISTERED_TAIL_PROBABILITY,
        'fusion_weight': REGISTERED_BLEND,
        'reference': 'train_scores',
        'test_labels_used_for_score_construction': False,
        'metrics': {'f1d_train_tail_gated_consensus': metric},
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, scores in [('f1a_train_ecdf_final', scalar), ('f1b_evidence_top2_mean', consensus),
                         ('f1d_train_tail_gated_consensus', final)]:
        np.save(output/f'test_scores_{name}.npy', scores)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(output/'report.json', flush=True)


if __name__ == '__main__':
    main()
