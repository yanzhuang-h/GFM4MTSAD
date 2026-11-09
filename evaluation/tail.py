"""Frozen two-stage normal-tail calibration for completed SHARK-IPRA runs."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np

REGISTERED_BLEND = 0.5
REGISTERED_EVIDENCE_TOPK = 2
REGISTERED_TAIL_PROBABILITY = 0.05


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


def pgrf_self_weighted_evidence(evidence: np.ndarray) -> np.ndarray:
    evidence = np.asarray(evidence, dtype=np.float64)
    if evidence.ndim != 2 or not np.isfinite(evidence).all():
        raise ValueError("PGRF-style evidence must be a finite matrix")
    nonnegative = np.maximum(evidence, 0.0)
    denominator = nonnegative.sum(axis=1)
    numerator = np.square(nonnegative).sum(axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


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


def build_tail_score_variants(
    train_final_scores: np.ndarray,
    test_final_scores: np.ndarray,
    calibrated_test_evidence: np.ndarray,
    *,
    blend: float = REGISTERED_BLEND,
    evidence_topk: int = REGISTERED_EVIDENCE_TOPK,
) -> dict[str, np.ndarray]:
    """Build candidates without accepting labels or dataset-specific settings."""
    blend = float(blend)
    if not 0.0 <= blend <= 1.0:
        raise ValueError("tail blend must be in [0, 1]")
    scalar_tail = upper_tail_surprisal(train_final_scores, test_final_scores)
    topk_tail = evidence_topk_mean(calibrated_test_evidence, evidence_topk)
    if scalar_tail.shape != topk_tail.shape:
        raise ValueError("scalar and evidence tail scores differ in length")
    evidence_mean = np.asarray(calibrated_test_evidence, dtype=np.float64).mean(
        axis=1
    )
    pgrf_weighted = pgrf_self_weighted_evidence(calibrated_test_evidence)
    gated_tail = train_tail_gated_consensus(
        scalar_tail,
        topk_tail,
        blend=blend,
        tail_probability=REGISTERED_TAIL_PROBABILITY,
    )
    return {
        "f1a_train_ecdf_final": scalar_tail,
        "f1b_evidence_top2_mean": topk_tail,
        "f1c_fixed_tail_consensus": (
            blend * scalar_tail + (1.0 - blend) * topk_tail
        ),
        "f1d_train_tail_gated_consensus": gated_tail,
        "sensitivity_evidence_mean": evidence_mean,
        "sensitivity_pgrf_self_weighted": pgrf_weighted,
    }


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


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if abs(denominator) <= 1e-12:
        return None
    return float(numerator / denominator)


def tail_diagnostics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = (np.asarray(labels).reshape(-1) > 0.1).astype(np.int64)
    scores = _finite_vector(scores, "tail diagnostic scores")
    if labels.size != scores.size or np.unique(labels).size != 2:
        raise ValueError("tail diagnostics require aligned binary labels")
    normal = scores[labels == 0]
    anomaly = scores[labels == 1]
    q99 = float(np.quantile(normal, 0.99))
    q995 = float(np.quantile(normal, 0.995))
    anomaly_median = float(np.median(anomaly))
    anomaly_max = float(np.max(anomaly))
    order = np.argsort(-scores, kind="stable")
    return {
        "normal_count": int(normal.size),
        "anomaly_count": int(anomaly.size),
        "normal_q99": q99,
        "normal_q995": q995,
        "normal_max": float(np.max(normal)),
        "anomaly_median": anomaly_median,
        "anomaly_max": anomaly_max,
        "r99": _safe_ratio(anomaly_median, q99),
        "r995": _safe_ratio(anomaly_median, q995),
        "sep": _safe_ratio(anomaly_median - q99, q995 - q99),
        "normal_above_anomaly_max_count": int(np.sum(normal > anomaly_max)),
        "top_30_anomaly_hits": int(labels[order[:30]].sum()),
        "top_60_anomaly_hits": int(labels[order[:60]].sum()),
        "top_100_anomaly_hits": int(labels[order[:100]].sum()),
    }


def train_reference_exceedance(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    train_scores = _finite_vector(train_scores, "train scores")
    test_scores = _finite_vector(test_scores, "test scores")
    labels = (np.asarray(labels).reshape(-1) > 0.1).astype(np.int64)
    if labels.size != test_scores.size:
        raise ValueError("test scores and labels differ in length")
    report = {}
    for quantile in (0.99, 0.995):
        threshold = float(np.quantile(train_scores, quantile))
        name = "q99" if quantile == 0.99 else "q995"
        report[name] = {
            "threshold": threshold,
            "threshold_source": "normal_training_scores_only",
            "normal_test_exceedance_rate": float(
                np.mean(test_scores[labels == 0] >= threshold)
            ),
            "anomaly_test_exceedance_rate": float(
                np.mean(test_scores[labels == 1] >= threshold)
            ),
        }
    return report


def _file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def _load_names(path: Path, expected_count: int) -> list[str]:
    values = np.load(path, allow_pickle=True).reshape(-1).tolist()
    if len(values) != expected_count or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError("evidence names do not match evidence matrix")
    return values


def parse_args() -> argparse.Namespace:
    ENTITY_SPECS = ("MSL_P-15", "SMAP_T-3", "SMD_1-1")

    parser = argparse.ArgumentParser(
        description="Frozen Stage-B tail calibration for a completed B1 run."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--entity", choices=sorted(ENTITY_SPECS), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-arm",
        default="B1_window60",
        help=(
            "immutable training arm that produced --source-dir; this is "
            "provenance only and cannot affect score construction"
        ),
    )
    return parser.parse_args()


def main() -> None:
    # Keep label-free scoring helpers usable for NumPy-only artifact replay.
    from evaluation.metrics import (
        continuous_metrics, pgrf_best_f1,
    )

    args = parse_args()
    source = Path(args.source_dir).resolve()
    output = Path(args.output_dir).resolve()
    source_arm = str(args.source_arm).strip()
    if not source_arm:
        raise ValueError("source arm must be a non-empty provenance label")
    if output.exists():
        raise FileExistsError(f"tail calibration output already exists: {output}")
    paths = {
        "train_raw": source / "train_scores_before_causal_calibration.npy",
        "train_current": source / "train_scores.npy",
        "test_raw": source / "test_scores_before_causal_calibration.npy",
        "test_current": source / "test_scores.npy",
        "test_labels": source / "test_labels.npy",
        "test_evidence": source / "test_evidence_protocol_scores.npy",
        "evidence_names": source / "test_evidence_protocol_names.npy",
        "metric_protocol": source / "test_metric_protocol.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing tail calibration inputs: {missing}")

    train_raw = _finite_vector(
        np.load(paths["train_raw"], allow_pickle=False), "train raw scores"
    )
    train_current = _finite_vector(
        np.load(paths["train_current"], allow_pickle=False),
        "train current scores",
    )
    test_raw = _finite_vector(
        np.load(paths["test_raw"], allow_pickle=False), "test raw scores"
    )
    test_current = _finite_vector(
        np.load(paths["test_current"], allow_pickle=False),
        "current final scores",
    )
    labels = (np.load(paths["test_labels"], allow_pickle=False).reshape(-1) > 0.1).astype(
        np.int64
    )
    evidence = np.asarray(
        np.load(paths["test_evidence"], allow_pickle=False), dtype=np.float64
    )
    if evidence.ndim != 2 or evidence.shape[0] != labels.size:
        raise ValueError("test evidence must be [test windows, evidence sources]")
    if test_raw.size != labels.size or test_current.size != labels.size:
        raise ValueError("SHARK score and label counts differ")
    names = _load_names(paths["evidence_names"], evidence.shape[1])
    if names != ["direct", "local", "relation", "context"]:
        raise ValueError(f"unexpected evidence source order: {names}")

    metric_protocol = json.loads(
        paths["metric_protocol"].read_text(encoding="utf-8")
    )
    label_protocol = metric_protocol.get("label_protocol", {})
    if (
        int(label_protocol.get("window_size", -1)) != 60
        or label_protocol.get("window_label")
        != "any anomaly point inside the input window"
    ):
        raise ValueError("tail calibration source is not T=60 window-any")

    variants = {
        "control_raw": test_raw,
        "control_current": test_current,
        **build_tail_score_variants(train_current, test_current, evidence),
    }
    metrics = {
        name: {
            **continuous_metrics(scores, labels),
            "raw_curve": best_curve_metrics(scores, labels),
            "point_adjusted_oracle_100": pgrf_best_f1(scores, labels),
            "tail": tail_diagnostics(labels, scores),
        }
        for name, scores in variants.items()
    }
    control = metrics["control_current"]
    prior_candidate = metrics["f1c_fixed_tail_consensus"]
    candidate = metrics["f1d_train_tail_gated_consensus"]
    deltas = {
        name: float(candidate[name] - control[name])
        for name in ("auroc", "auprc_trapezoid", "average_precision")
    }
    deltas["raw_f1"] = float(
        candidate["raw_curve"]["f1"] - control["raw_curve"]["f1"]
    )
    deltas_vs_prior = {
        name: float(candidate[name] - prior_candidate[name])
        for name in ("auroc", "auprc_trapezoid", "average_precision")
    }
    deltas_vs_prior["raw_f1"] = float(
        candidate["raw_curve"]["f1"]
        - prior_candidate["raw_curve"]["f1"]
    )
    scalar_delta = {
        name: float(
            metrics["f1a_train_ecdf_final"][name]
            - metrics["control_current"][name]
        )
        for name in ("auroc", "auprc_trapezoid", "average_precision")
    }
    gate_threshold = -math.log(REGISTERED_TAIL_PROBABILITY)
    gate_active = variants["f1a_train_ecdf_final"] >= gate_threshold
    report = {
        "schema_version": "SHARK-IPRA-FROZEN-TAIL-CALIBRATION-V2",
        "entity": args.entity,
        "source_arm": source_arm,
        "stage_a": {
            "status": "completed_checkpoint_and_score_artifacts",
            "parameters_updated_by_this_replay": False,
        },
        "stage_b": {
            "prior_candidate": "f1c_fixed_tail_consensus",
            "registered_candidate": "f1d_train_tail_gated_consensus",
            "normal_scalar_reference": "train_scores",
            "scalar_calibration": "empirical_upper_tail_negative_log_probability",
            "evidence_calibration": (
                "existing_train-normal soft_ecdf_two_sided values saved by SHARK"
            ),
            "evidence_consensus": "mean_of_largest_two_of_four_evidence_scores",
            "fixed_blend": REGISTERED_BLEND,
            "tail_gate_probability": REGISTERED_TAIL_PROBABILITY,
            "tail_gate_surprisal_threshold": gate_threshold,
            "tail_gate_rule": (
                "preserve scalar ECDF below the train-normal tail; inside the "
                "tail use fixed consensus floored at the gate threshold"
            ),
            "train_normal_reference_mass_below_gate": (
                1.0 - REGISTERED_TAIL_PROBABILITY
            ),
            "learned_gate": False,
            "dataset_specific_branch": False,
        },
        "evidence_names": names,
        "metrics": metrics,
        "registered_candidate_deltas_vs_current": deltas,
        "registered_candidate_deltas_vs_prior_f1c": deltas_vs_prior,
        "tail_gate_runtime_diagnostics": {
            "threshold": gate_threshold,
            "active_test_window_count": int(np.sum(gate_active)),
            "active_test_window_rate": float(np.mean(gate_active)),
            "active_normal_window_rate": float(
                np.mean(gate_active[labels == 0])
            ),
            "active_anomaly_window_rate": float(
                np.mean(gate_active[labels == 1])
            ),
            "inactive_scores_unchanged": bool(
                np.array_equal(
                    variants["f1d_train_tail_gated_consensus"][~gate_active],
                    variants["f1a_train_ecdf_final"][~gate_active],
                )
            ),
        },
        "monotonic_scalar_calibration_check": {
            "strict_order_gain_expected": False,
            "finite_reference_ties_can_change_metrics_slightly": True,
            "observed_metric_deltas_vs_current": scalar_delta,
            "purpose": (
                "prove that any gain must come from evidence consensus rather "
                "than scalar rescaling"
            ),
        },
        "train_reference_exceedance_on_raw_score": train_reference_exceedance(
            train_raw, test_raw, labels
        ),
        "train_reference_exceedance_on_current_score": train_reference_exceedance(
            train_current, test_current, labels
        ),
        "information_use": {
            "build_tail_score_variants_signature": str(
                inspect.signature(build_tail_score_variants)
            ),
            "test_labels_accepted_by_score_builder": False,
            "test_labels_used_for_score_construction": False,
            "test_labels_used_for_metrics_and_tail_diagnostics": True,
            "test_distribution_used_to_fit_scalar_calibrator": False,
        },
        "provisional_gate": {
            "auprc_improved": deltas["auprc_trapezoid"] > 0.0,
            "auroc_drop_within_0_01": deltas["auroc"] >= -0.01,
            "raw_f1_nonregression": deltas["raw_f1"] >= 0.0,
            "promotion_authorized": False,
            "reason": (
                "single-seed MSL/SMAP test-label diagnostics cannot select or "
                "promote a calibration rule; SMD/SWaT and >=3 seeds remain required"
            ),
        },
        "source_artifacts": {
            name: _file_identity(path) for name, path in paths.items()
        },
    }

    output.mkdir(parents=True, exist_ok=False)
    for name, values in variants.items():
        np.save(output / f"test_scores_{name}.npy", values)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output / "report.json")


if __name__ == "__main__":
    main()
