import math
import random

import torch

from models.shark.frontend_graph_effective_selector import (
    FrontendGraphEffectiveSelector,
)
from models.shark.frozen_gfm_teacher_selector import FrozenGFMTeacherSelector
from models.shark.otrcl_integrity import (
    normalized_graph_drift,
    normalized_temporal_drift,
)

from models.shark.normal_mechanism import GlobalGraphVARMechanism
from models.shark.funtsg_inspired import function_candidates
from models.shark.mechanism_matched import (
    MechanismMatchedPairBuilder,
    temporal_pair_fairness,
)


class SharkAugmentor:
    SURROGATE_TYPES = ("surrogate_parent_delete", "surrogate_parent_replace")
    DIRECT_FALLBACK_TYPES = (
        "lag_break",
        "neighbor_break",
        "relation_flip",
        "local_function_shift",
    )
    EDIT_SOURCE_INVALID = 0
    EDIT_SOURCE_SURROGATE = 1
    EDIT_SOURCE_FALLBACK = 2
    EDIT_SOURCE_DIRECT = 3
    ACCEPTANCE_INVALID = 0
    ACCEPTANCE_SOFT = 1
    ACCEPTANCE_HARD = 2

    def __init__(self, cfg):
        self.candidate_generator = str(getattr(cfg.SHARK.SYNTHETIC, "CANDIDATE_GENERATOR", "graph_var"))
        if self.candidate_generator not in ("graph_var", "funtsg_inspired", "hybrid"):
            raise ValueError("unknown CANDIDATE_GENERATOR")
        self.generator_counts = {name: dict(proposed=0, generated=0, accepted=0, selected=0)
                                 for name in ("graph_var", "funtsg_inspired", "legacy")}
        self.generator_cells = {}
        self.counterfactual_implementation = str(getattr(
            cfg.SHARK.SYNTHETIC, "COUNTERFACTUAL_IMPLEMENTATION", "scalar"))
        if self.counterfactual_implementation not in ("scalar", "batched_strengths", "batched_precomputed"):
            raise ValueError("unknown COUNTERFACTUAL_IMPLEMENTATION")
        self.noise_std = cfg.SHARK.AUGMENT.NOISE_STD
        self.positive_noise_std = cfg.SHARK.CONTRAST.POSITIVE_NOISE_STD
        self.positive_view_mode = str(
            getattr(cfg.SHARK.CONTRAST, "POSITIVE_VIEW_MODE", "legacy")
        )
        self.positive_time_ratio = float(
            getattr(cfg.SHARK.CONTRAST, "POSITIVE_TIME_RATIO", 1.0)
        )
        self.graph_consistent_positive_enable = bool(
            getattr(
                cfg.SHARK.CONTRAST,
                "GRAPH_CONSISTENT_POSITIVE_ENABLE",
                False,
            )
        )
        self.graph_consistent_positive_candidates = max(
            1,
            int(
                getattr(
                    cfg.SHARK.CONTRAST,
                    "GRAPH_CONSISTENT_POSITIVE_CANDIDATES",
                    3,
                )
            ),
        )
        self.graph_consistent_positive_temporal_quantile = float(
            getattr(
                cfg.SHARK.CONTRAST,
                "GRAPH_CONSISTENT_POSITIVE_TEMPORAL_QUANTILE",
                0.5,
            )
        )
        # H2 uses the original graph-first/temporal-diversity gate. H3-P
        # reverses the priority: keep the graph-consistent half, then choose
        # the most temporally informative positive. The default preserves H2.
        self.graph_consistent_positive_selector = str(
            getattr(
                cfg.SHARK.CONTRAST,
                "GRAPH_CONSISTENT_POSITIVE_SELECTOR",
                "min_graph_after_temporal_gate",
            )
        )
        self.graph_consistent_positive_graph_quantile = float(
            getattr(
                cfg.SHARK.CONTRAST,
                "GRAPH_CONSISTENT_POSITIVE_GRAPH_QUANTILE",
                0.5,
            )
        )
        self.negative_node_ratio = cfg.SHARK.CONTRAST.NEGATIVE_NODE_RATIO
        self.subgraph_topk = cfg.SHARK.CONTRAST.SUBGRAPH_TOPK
        self.synthetic_node_ratio = cfg.SHARK.SYNTHETIC.NODE_RATIO
        self.synthetic_time_ratio = cfg.SHARK.SYNTHETIC.TIME_RATIO
        self.synthetic_time_ratio_min = float(
            getattr(cfg.SHARK.SYNTHETIC, "TIME_RATIO_MIN", self.synthetic_time_ratio)
        )
        self.synthetic_time_ratio_max = float(
            getattr(cfg.SHARK.SYNTHETIC, "TIME_RATIO_MAX", self.synthetic_time_ratio)
        )
        self.segment_length_policy = str(
            getattr(cfg.SHARK.SYNTHETIC, "SEGMENT_LENGTH_POLICY", "uniform_ratio")
        )
        probabilities = tuple(
            float(value)
            for value in getattr(
                cfg.SHARK.SYNTHETIC,
                "EVENT_SCALE_PROBABILITIES",
                (0.25, 0.50, 0.25),
            )
        )
        self.event_scale_probabilities = probabilities
        self.event_scale_micro_min = int(
            getattr(cfg.SHARK.SYNTHETIC, "EVENT_SCALE_MICRO_MIN", 2)
        )
        self.event_scale_micro_max = int(
            getattr(cfg.SHARK.SYNTHETIC, "EVENT_SCALE_MICRO_MAX", 3)
        )
        self.event_scale_persistent_ratio_min = float(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "EVENT_SCALE_PERSISTENT_RATIO_MIN",
                0.75,
            )
        )
        self.event_scale_persistent_ratio_max = float(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "EVENT_SCALE_PERSISTENT_RATIO_MAX",
                1.0,
            )
        )
        self.synthetic_magnitude = cfg.SHARK.SYNTHETIC.MAGNITUDE
        self.synthetic_mode = cfg.SHARK.SYNTHETIC.MODE
        self.synthetic_types = tuple(cfg.SHARK.SYNTHETIC.TYPES)
        self.diagnostic_types = tuple(getattr(cfg.SHARK.SYNTHETIC, "DIAGNOSTIC_TYPES", ()))
        self.edit_mode = getattr(cfg.SHARK.SYNTHETIC, "EDIT_MODE", "legacy")
        self.edit_strength_start = float(getattr(cfg.SHARK.SYNTHETIC, "EDIT_STRENGTH_START", 0.25))
        self.edit_strength_end = float(getattr(cfg.SHARK.SYNTHETIC, "EDIT_STRENGTH_END", 0.75))
        self.max_delta_std = float(getattr(cfg.SHARK.SYNTHETIC, "MAX_DELTA_STD", 3.0))
        self.hybrid_fallback_types = tuple(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "HYBRID_FALLBACK_TYPES",
                self.DIRECT_FALLBACK_TYPES,
            )
        )
        self.hybrid_fallback_min_rms = max(
            0.0,
            float(getattr(cfg.SHARK.SYNTHETIC, "HYBRID_FALLBACK_MIN_RMS", 0.1)),
        )
        self.hybrid_fallback_max_rms = max(
            self.hybrid_fallback_min_rms,
            float(getattr(cfg.SHARK.SYNTHETIC, "HYBRID_FALLBACK_MAX_RMS", 1.0)),
        )
        self.hybrid_fallback_validity_enable = bool(
            getattr(cfg.SHARK.SYNTHETIC, "HYBRID_FALLBACK_VALIDITY_ENABLE", False)
        )
        self.hybrid_fallback_min_mechanism_gain = max(
            0.0,
            float(
                getattr(
                    cfg.SHARK.SYNTHETIC,
                    "HYBRID_FALLBACK_MIN_MECHANISM_GAIN",
                    0.0,
                )
            ),
        )
        self.hybrid_fallback_max_mechanism_gain = max(
            self.hybrid_fallback_min_mechanism_gain,
            float(
                getattr(
                    cfg.SHARK.SYNTHETIC,
                    "HYBRID_FALLBACK_MAX_MECHANISM_GAIN",
                    float("inf"),
                )
            ),
        )
        self.hybrid_fallback_candidates = max(
            1,
            int(getattr(cfg.SHARK.SYNTHETIC, "HYBRID_FALLBACK_CANDIDATES", 5)),
        )
        self.direct_quantile_start = float(
            getattr(cfg.SHARK.SYNTHETIC, "DIRECT_QUANTILE_START", 0.80)
        )
        self.direct_quantile_end = float(
            getattr(cfg.SHARK.SYNTHETIC, "DIRECT_QUANTILE_END", 0.90)
        )
        self.direct_quantile_cap = float(
            getattr(cfg.SHARK.SYNTHETIC, "DIRECT_QUANTILE_CAP", 0.995)
        )
        self.direct_quantile_reference_scale_mode = str(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "DIRECT_QUANTILE_REFERENCE_SCALE_MODE",
                "iqr",
            )
        )
        self.direct_quantile_upper_bound_mode = str(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "DIRECT_QUANTILE_UPPER_BOUND_MODE",
                "mechanism_cap",
            )
        )
        self.direct_quantile_rms_rescue_weight = min(
            max(
                float(
                    getattr(
                        cfg.SHARK.SYNTHETIC,
                        "DIRECT_QUANTILE_RMS_RESCUE_WEIGHT",
                        0.25,
                    )
                ),
                0.0,
            ),
            1.0,
        )
        self.direct_quantile_min_reference_gain = max(
            0.0,
            float(
                getattr(
                    cfg.SHARK.SYNTHETIC,
                    "DIRECT_QUANTILE_MIN_REFERENCE_GAIN",
                    0.10,
                )
            ),
        )
        self.direct_quantile_segment_trials = max(
            1,
            int(
                getattr(
                    cfg.SHARK.SYNTHETIC,
                    "DIRECT_QUANTILE_SEGMENT_TRIALS",
                    3,
                )
            ),
        )
        self.direct_quantile_acceptance_mode = str(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "DIRECT_QUANTILE_ACCEPTANCE_MODE",
                "hard",
            )
        )
        self.direct_quantile_soft_min_weight = float(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "DIRECT_QUANTILE_SOFT_MIN_WEIGHT",
                0.25,
            )
        )
        self.direct_quantile_soft_power = float(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "DIRECT_QUANTILE_SOFT_POWER",
                1.0,
            )
        )
        self.direct_mechanism_score_mode = str(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "DIRECT_MECHANISM_SCORE_MODE",
                "legacy_shared",
            )
        )
        self.direct_mechanism_ridge = max(
            float(
                getattr(
                    cfg.SHARK.SYNTHETIC,
                    "DIRECT_MECHANISM_RIDGE",
                    1e-3,
                )
            ),
            1e-8,
        )
        self.direct_mechanism_min_context = max(
            int(
                getattr(
                    cfg.SHARK.SYNTHETIC,
                    "DIRECT_MECHANISM_MIN_CONTEXT",
                    3,
                )
            ),
            2,
        )
        self.direct_relation_flip_orientation_weight = max(
            float(
                getattr(
                    cfg.SHARK.SYNTHETIC,
                    "DIRECT_RELATION_FLIP_ORIENTATION_WEIGHT",
                    1.0,
                )
            ),
            0.0,
        )
        self.direct_candidate_support_mode = str(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "DIRECT_CANDIDATE_SUPPORT_MODE",
                "none",
            )
        )
        self.direct_candidate_support_low_quantile = float(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "DIRECT_CANDIDATE_SUPPORT_LOW_QUANTILE",
                0.01,
            )
        )
        self.direct_candidate_support_high_quantile = float(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "DIRECT_CANDIDATE_SUPPORT_HIGH_QUANTILE",
                0.99,
            )
        )
        self.negative_screening_mode = str(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "NEGATIVE_SCREENING_MODE",
                "legacy_boundary",
            )
        )
        self.frontend_primary_nonlegacy_weight = float(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "FRONTEND_PRIMARY_NONLEGACY_WEIGHT",
                1.0,
            )
        )
        self.gm_hns_difficulty_low = float(
            getattr(cfg.SHARK.SYNTHETIC, "GM_HNS_DIFFICULTY_LOW", 0.10)
        )
        self.gm_hns_difficulty_high = float(
            getattr(cfg.SHARK.SYNTHETIC, "GM_HNS_DIFFICULTY_HIGH", 0.85)
        )
        self.gm_hns_difficulty_target = float(
            getattr(cfg.SHARK.SYNTHETIC, "GM_HNS_DIFFICULTY_TARGET", 0.45)
        )
        self.gm_hns_rms_weight = max(
            0.0,
            float(getattr(cfg.SHARK.SYNTHETIC, "GM_HNS_RMS_WEIGHT", 0.05)),
        )
        otrcl_settings = getattr(cfg.SHARK, "OTRCL", None)
        self.wf_orthogonality_enable = bool(
            getattr(
                otrcl_settings,
                "FINE_ORTHOGONALITY_ENABLE",
                False,
            )
        )
        self.wf_max_temporal_drift = float(
            getattr(
                otrcl_settings,
                "FINE_MAX_TEMPORAL_DRIFT",
                0.15,
            )
        )
        self.wf_min_relational_margin = float(
            getattr(
                otrcl_settings,
                "FINE_MIN_RELATIONAL_MARGIN_OVER_COARSE",
                0.01,
            )
        )
        self.wf_relation_metric = str(
            getattr(
                otrcl_settings,
                "FINE_RELATION_METRIC",
                "raw_abs_pearson",
            )
        )
        self.surrogate_lag = max(1, int(getattr(cfg.SHARK.SYNTHETIC, "SURROGATE_LAG", 1)))
        self.surrogate_include_contemporaneous = bool(
            getattr(cfg.SHARK.SYNTHETIC, "SURROGATE_INCLUDE_CONTEMPORANEOUS", False)
        )
        self.surrogate_ridge = max(float(getattr(cfg.SHARK.SYNTHETIC, "SURROGATE_RIDGE", 0.01)), 1e-6)
        self.surrogate_min_context = max(
            4,
            int(getattr(cfg.SHARK.SYNTHETIC, "SURROGATE_MIN_CONTEXT", 12)),
        )
        self.surrogate_segment_trials = max(
            1,
            int(getattr(cfg.SHARK.SYNTHETIC, "SURROGATE_SEGMENT_TRIALS", 3)),
        )
        self.surrogate_seed_trials = max(
            1,
            int(getattr(cfg.SHARK.SYNTHETIC, "SURROGATE_SEED_TRIALS", 1)),
        )
        self.surrogate_parent_candidates = max(
            1,
            int(getattr(cfg.SHARK.SYNTHETIC, "SURROGATE_PARENT_CANDIDATES", 5)),
        )
        self.surrogate_min_r2 = float(
            getattr(cfg.SHARK.SYNTHETIC, "SURROGATE_MIN_R2", 0.0)
        )
        self.surrogate_min_delta_r2 = max(
            0.0,
            float(getattr(cfg.SHARK.SYNTHETIC, "SURROGATE_MIN_DELTA_R2", 0.0)),
        )
        self.surrogate_min_residual_ratio = max(
            0.0,
            float(getattr(cfg.SHARK.SYNTHETIC, "SURROGATE_MIN_RESIDUAL_RATIO", 0.05)),
        )
        self.surrogate_min_scale = max(
            1e-4,
            float(getattr(cfg.SHARK.SYNTHETIC, "SURROGATE_MIN_SCALE", 0.01)),
        )
        self.surrogate_max_scale = max(
            self.surrogate_min_scale,
            float(getattr(cfg.SHARK.SYNTHETIC, "SURROGATE_MAX_SCALE", 4.0)),
        )
        self.boundary_candidates = max(
            1,
            int(getattr(cfg.SHARK.SYNTHETIC, "BOUNDARY_CANDIDATES", 3)),
        )
        self.boundary_quantile_start = float(
            getattr(cfg.SHARK.SYNTHETIC, "BOUNDARY_QUANTILE_START", 0.99)
        )
        self.boundary_quantile_end = float(
            getattr(cfg.SHARK.SYNTHETIC, "BOUNDARY_QUANTILE_END", 0.95)
        )
        self.boundary_easy_mix = min(
            max(float(getattr(cfg.SHARK.SYNTHETIC, "BOUNDARY_EASY_MIX", 0.2)), 0.0),
            1.0,
        )
        self.boundary_min_score_increase = max(
            0.0,
            float(getattr(cfg.SHARK.SYNTHETIC, "BOUNDARY_MIN_SCORE_INCREASE", 0.1)),
        )
        self.boundary_mean_shift_cap = max(
            0.0,
            float(getattr(cfg.SHARK.SYNTHETIC, "BOUNDARY_MEAN_SHIFT_CAP", 1.5)),
        )
        self.require_accepted = bool(
            getattr(cfg.SHARK.SYNTHETIC, "REQUIRE_ACCEPTED", False)
        )
        self.seed_selection = cfg.SHARK.SYNTHETIC.SEED_SELECTION
        self.evidence_temperature = cfg.SHARK.SYNTHETIC.EVIDENCE_TEMPERATURE
        self.evidence_random_mix = cfg.SHARK.SYNTHETIC.EVIDENCE_RANDOM_MIX
        self.strength_schedule = cfg.SHARK.SYNTHETIC.STRENGTH_SCHEDULE
        self.magnitude_start = cfg.SHARK.SYNTHETIC.MAGNITUDE_START
        self.magnitude_end = cfg.SHARK.SYNTHETIC.MAGNITUDE_END
        self.random_noise_std = cfg.SHARK.SYNTHETIC.RANDOM_NOISE_STD
        self.scope_aware_negative_enable = bool(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "SCOPE_AWARE_NEGATIVE_ENABLE",
                False,
            )
        )
        self.scope_aware_negative_probability = min(
            max(
                float(
                    getattr(
                        cfg.SHARK.SYNTHETIC,
                        "SCOPE_AWARE_NEGATIVE_PROBABILITY",
                        0.80,
                    )
                ),
                0.0,
            ),
            1.0,
        )
        self.scope_aware_negative_min_nodes = max(
            1,
            int(
                getattr(
                    cfg.SHARK.SYNTHETIC,
                    "SCOPE_AWARE_NEGATIVE_MIN_NODES",
                    1,
                )
            ),
        )
        self.scope_aware_negative_max_nodes = max(
            self.scope_aware_negative_min_nodes,
            int(
                getattr(
                    cfg.SHARK.SYNTHETIC,
                    "SCOPE_AWARE_NEGATIVE_MAX_NODES",
                    5,
                )
            ),
        )
        self.scope_aware_negative_effect_scale = max(
            0.0,
            float(
                getattr(
                    cfg.SHARK.SYNTHETIC,
                    "SCOPE_AWARE_NEGATIVE_EFFECT_SCALE",
                    0.35,
                )
            ),
        )
        self.scope_aware_negative_singleton_weight = min(
            max(
                float(
                    getattr(
                        cfg.SHARK.SYNTHETIC,
                        "SCOPE_AWARE_NEGATIVE_SINGLETON_WEIGHT",
                        0.90,
                    )
                ),
                0.0,
            ),
            1.0,
        )
        self.normal_mechanism = (
            GlobalGraphVARMechanism(
                cfg,
                adaptive=(
                    self.direct_mechanism_score_mode == "adaptive_graph_var"
                ),
            )
            if self.direct_mechanism_score_mode
            in ("global_graph_var", "adaptive_graph_var")
            else None
        )
        self.frontend_graph_effective_selector = (
            FrontendGraphEffectiveSelector(cfg)
            if self.negative_screening_mode
            in ("frontend_graph_effective", "frontend_primary")
            else None
        )
        self.frozen_gfm_teacher_selector = None
        self.coverage_balanced_schedule_enable = bool(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "COVERAGE_BALANCED_SCHEDULE_ENABLE",
                False,
            )
        )
        self.mechanism_availability_routing_enable = bool(
            getattr(
                cfg.SHARK.SYNTHETIC,
                "MECHANISM_AVAILABILITY_ROUTING_ENABLE",
                False,
            )
        )
        self._coverage_schedule_cursor = 0
        self._decision_rng = None
        self.last_diagnostics = {}
        self.last_coverage_diagnostics = {}
        self.last_valid_mask = None
        self.last_sample_weights = None
        self.last_acceptance_tier = None
        self.last_rms_rescue_mask = None
        self.last_selected_delta_rms = None
        self.last_edit_source_ids = None
        self.last_fallback_type_ids = None
        self.last_direct_type_ids = None
        self.last_seed_ids = None
        self.last_consistency_exclusion_mask = None
        self.last_wf_temporal_drift = None
        self.last_wf_relational_drift = None
        self.last_wf_relational_floor = None
        self.last_wf_orthogonality_diagnostics = {}
        self.last_positive_selection_diagnostics = {}
        # Opt-in CPU candidate export for the v2.2 train-normal frontend
        # audit. The frozen v2.1 path leaves this disabled.
        self.collect_candidate_records = False
        self.last_candidate_records = []
        self._last_mechanism_details = {}
        self.mechanism_matched_enabled = bool(
            getattr(
                getattr(cfg.SHARK, "RELATION_RESIDUAL_CONTRAST", None),
                "ENABLE",
                False,
            )
        )
        self.mechanism_matched_lag_offset = max(
            int(
                getattr(
                    cfg.SHARK.SYNTHETIC,
                    "GRAPH_VAR_LAG_OFFSET_STEPS",
                    1,
                )
            ),
            1,
        )

        if self.positive_view_mode not in ("legacy", "shared_affine_segment"):
            raise ValueError(
                "Unsupported SHARK.CONTRAST.POSITIVE_VIEW_MODE: "
                f"{self.positive_view_mode}"
            )
        if self.seed_selection not in (
            "random",
            "evidence_aware",
            "mechanism_aware",
        ):
            raise ValueError(
                "Unsupported SHARK.SYNTHETIC.SEED_SELECTION: "
                f"{self.seed_selection}"
            )
        if self.candidate_generator != "graph_var" and (self.normal_mechanism is None or self.edit_mode != "direct_quantile"):
            raise ValueError("function/hybrid candidates require Graph-VAR scoring and direct_quantile selection")
        if self.seed_selection == "mechanism_aware" and self.normal_mechanism is None:
            raise ValueError(
                "mechanism_aware seed selection requires a fitted "
                "global_graph_var or adaptive_graph_var normal mechanism"
            )
        if self.mechanism_availability_routing_enable and (
            self.normal_mechanism is None
            or self.direct_mechanism_score_mode != "adaptive_graph_var"
            or self.edit_mode != "direct_quantile"
            or self.candidate_generator != "graph_var"
        ):
            raise ValueError(
                "Mechanism-availability routing requires "
                "adaptive_graph_var direct_quantile training"
            )
        if self.negative_screening_mode not in (
            "legacy_boundary",
            "gm_hns",
            "frontend_graph_effective",
            "frontend_primary",
            "frozen_gfm_teacher",
        ):
            raise ValueError(
                "Unsupported SHARK.SYNTHETIC.NEGATIVE_SCREENING_MODE: "
                f"{self.negative_screening_mode}"
            )
        if not 0.0 < self.frontend_primary_nonlegacy_weight <= 1.0:
            raise ValueError(
                "FRONTEND_PRIMARY_NONLEGACY_WEIGHT must satisfy 0 < weight <= 1"
            )
        if (
            self.negative_screening_mode == "frontend_primary"
            and self.wf_orthogonality_enable
        ):
            raise ValueError(
                "frontend_primary screening is not defined for Wf orthogonality"
            )
        if not (
            0.0 <= self.gm_hns_difficulty_low
            <= self.gm_hns_difficulty_target
            <= self.gm_hns_difficulty_high
            <= 1.0
        ):
            raise ValueError(
                "GM-HNS difficulty bounds must satisfy "
                "0 <= low <= target <= high <= 1"
            )
        if (
            self.scope_aware_negative_enable
            and self.normal_mechanism is None
        ):
            raise ValueError(
                "scope-aware negative selection requires a Graph-VAR normal mechanism"
            )
        if self.scope_aware_negative_max_nodes < self.scope_aware_negative_min_nodes:
            raise ValueError(
                "SCOPE_AWARE_NEGATIVE_MAX_NODES must be >= MIN_NODES"
            )
        if not 0.0 < self.positive_time_ratio <= 1.0:
            raise ValueError("SHARK.CONTRAST.POSITIVE_TIME_RATIO must satisfy 0 < ratio <= 1")
        if (
            self.graph_consistent_positive_enable
            and self.graph_consistent_positive_candidates < 2
        ):
            raise ValueError(
                "Graph-consistent positive selection requires at least two candidates"
            )
        if not 0.0 <= self.graph_consistent_positive_temporal_quantile <= 1.0:
            raise ValueError(
                "Graph-consistent positive temporal quantile must be in [0, 1]"
            )
        if not 0.0 <= self.graph_consistent_positive_graph_quantile <= 1.0:
            raise ValueError(
                "Graph-consistent positive graph quantile must be in [0, 1]"
            )
        if self.graph_consistent_positive_selector not in (
            "min_graph_after_temporal_gate",
            "max_temporal_after_graph_gate",
        ):
            raise ValueError(
                "Unsupported graph-consistent positive selector: "
                f"{self.graph_consistent_positive_selector}"
            )
        if self.graph_consistent_positive_enable and self.normal_mechanism is None:
            raise ValueError(
                "Graph-consistent positive selection requires a Graph-VAR normal mechanism"
            )
        if self.edit_mode not in (
            "legacy",
            "interpolated",
            "surrogate_boundary",
            "hybrid_valid",
            "direct_quantile",
        ):
            raise ValueError(f"Unsupported SHARK.SYNTHETIC.EDIT_MODE: {self.edit_mode}")
        unsupported_fallbacks = set(self.hybrid_fallback_types) - set(self.DIRECT_FALLBACK_TYPES)
        if unsupported_fallbacks:
            raise ValueError(
                "Unsupported SHARK.SYNTHETIC.HYBRID_FALLBACK_TYPES: "
                f"{sorted(unsupported_fallbacks)}"
            )
        if not self.hybrid_fallback_types:
            raise ValueError("SHARK.SYNTHETIC.HYBRID_FALLBACK_TYPES must not be empty")
        if not 0.0 < self.synthetic_time_ratio_min <= self.synthetic_time_ratio_max <= 1.0:
            raise ValueError("SHARK.SYNTHETIC.TIME_RATIO_MIN/MAX must satisfy 0 < min <= max <= 1")
        if self.segment_length_policy not in ("uniform_ratio", "event_scale_balanced"):
            raise ValueError(
                "Unsupported SHARK.SYNTHETIC.SEGMENT_LENGTH_POLICY: "
                f"{self.segment_length_policy}"
            )
        if len(self.event_scale_probabilities) != 3 or any(
            value < 0.0 for value in self.event_scale_probabilities
        ) or sum(self.event_scale_probabilities) <= 0.0:
            raise ValueError(
                "SHARK.SYNTHETIC.EVENT_SCALE_PROBABILITIES must contain three "
                "nonnegative values with positive sum"
            )
        if not 1 <= self.event_scale_micro_min <= self.event_scale_micro_max:
            raise ValueError(
                "SHARK.SYNTHETIC.EVENT_SCALE_MICRO_MIN/MAX must be positive and ordered"
            )
        if not (
            0.0 < self.event_scale_persistent_ratio_min
            <= self.event_scale_persistent_ratio_max
            <= 1.0
        ):
            raise ValueError(
                "SHARK.SYNTHETIC.EVENT_SCALE_PERSISTENT_RATIO_MIN/MAX must satisfy "
                "0 < min <= max <= 1"
            )
        for name, value in (
            ("BOUNDARY_QUANTILE_START", self.boundary_quantile_start),
            ("BOUNDARY_QUANTILE_END", self.boundary_quantile_end),
            ("DIRECT_QUANTILE_START", self.direct_quantile_start),
            ("DIRECT_QUANTILE_END", self.direct_quantile_end),
            ("DIRECT_QUANTILE_CAP", self.direct_quantile_cap),
        ):
            if not 0.5 <= value < 1.0:
                raise ValueError(f"SHARK.SYNTHETIC.{name} must satisfy 0.5 <= value < 1.0")
        if self.direct_quantile_cap < max(
            self.direct_quantile_start,
            self.direct_quantile_end,
        ):
            raise ValueError(
                "SHARK.SYNTHETIC.DIRECT_QUANTILE_CAP must be no smaller than "
                "DIRECT_QUANTILE_START/END"
            )
        if self.direct_quantile_acceptance_mode not in (
            "hard",
            "soft_unweighted",
            "soft_weighted",
        ):
            raise ValueError(
                "Unsupported SHARK.SYNTHETIC.DIRECT_QUANTILE_ACCEPTANCE_MODE: "
                f"{self.direct_quantile_acceptance_mode}"
            )
        if self.direct_quantile_reference_scale_mode not in (
            "iqr",
            "iqr_std_fallback",
        ):
            raise ValueError(
                "Unsupported SHARK.SYNTHETIC."
                "DIRECT_QUANTILE_REFERENCE_SCALE_MODE: "
                f"{self.direct_quantile_reference_scale_mode}"
            )
        if self.direct_quantile_upper_bound_mode not in (
            "mechanism_cap",
            "rms_rescue",
        ):
            raise ValueError(
                "Unsupported SHARK.SYNTHETIC."
                "DIRECT_QUANTILE_UPPER_BOUND_MODE: "
                f"{self.direct_quantile_upper_bound_mode}"
            )
        if not 0.0 < self.direct_quantile_soft_min_weight <= 1.0:
            raise ValueError(
                "SHARK.SYNTHETIC.DIRECT_QUANTILE_SOFT_MIN_WEIGHT must satisfy "
                "0 < weight <= 1"
            )
        if self.direct_quantile_soft_power <= 0.0:
            raise ValueError(
                "SHARK.SYNTHETIC.DIRECT_QUANTILE_SOFT_POWER must be positive"
            )
        if self.direct_mechanism_score_mode not in (
            "legacy_shared",
            "type_specific",
            "global_graph_var",
            "adaptive_graph_var",
        ):
            raise ValueError(
                "Unsupported SHARK.SYNTHETIC.DIRECT_MECHANISM_SCORE_MODE: "
                f"{self.direct_mechanism_score_mode}"
            )
        if self.direct_candidate_support_mode not in (
            "none",
            "per_node_quantile",
            "global_per_node_quantile",
            "adaptive_empirical",
        ):
            raise ValueError(
                "Unsupported SHARK.SYNTHETIC.DIRECT_CANDIDATE_SUPPORT_MODE: "
                f"{self.direct_candidate_support_mode}"
            )
        if not (
            0.0 <= self.direct_candidate_support_low_quantile
            < self.direct_candidate_support_high_quantile
            <= 1.0
        ):
            raise ValueError(
                "SHARK.SYNTHETIC direct candidate support quantiles must satisfy "
                "0 <= low < high <= 1"
            )
        if self.wf_max_temporal_drift < 0.0:
            raise ValueError(
                "SHARK.OTRCL.FINE_MAX_TEMPORAL_DRIFT must be nonnegative"
            )
        if self.wf_min_relational_margin < 0.0:
            raise ValueError(
                "SHARK.OTRCL.FINE_MIN_RELATIONAL_MARGIN_OVER_COARSE "
                "must be nonnegative"
            )
        if self.wf_relation_metric != "raw_abs_pearson":
            raise ValueError(
                "Unsupported SHARK.OTRCL.FINE_RELATION_METRIC: "
                f"{self.wf_relation_metric}"
            )

    @property
    def requires_normal_mechanism(self):
        return self.normal_mechanism is not None

    def fit_normal_mechanism(self, source):
        if self.normal_mechanism is None:
            return {}
        return self.normal_mechanism.fit(source)

    @property
    def requires_frontend_graph_selector(self):
        return self.frontend_graph_effective_selector is not None

    def fit_frontend_graph_selector(self, frontend, graph_builder, source):
        """Bind v2.2 to the frozen frontend and fit train-normal scale only."""
        if self.frontend_graph_effective_selector is None:
            return {}
        if self.normal_mechanism is None or not self.normal_mechanism.fitted:
            raise RuntimeError(
                "Fit the normal Graph-VAR before the frontend-effective selector"
            )
        self.frontend_graph_effective_selector.bind(frontend, graph_builder)
        return self.frontend_graph_effective_selector.fit(
            source,
            self.normal_mechanism.valid_nodes,
        )

    @property
    def requires_frozen_gfm_teacher(self):
        return self.negative_screening_mode == "frozen_gfm_teacher"

    def bind_frozen_gfm_teacher(self, teacher):
        if not self.requires_frozen_gfm_teacher:
            return {}
        if self.normal_mechanism is None or not self.normal_mechanism.fitted:
            raise RuntimeError(
                "Fit the normal Graph-VAR before binding the frozen GFM teacher"
            )
        self.frozen_gfm_teacher_selector = FrozenGFMTeacherSelector(
            teacher,
            self.normal_mechanism.valid_nodes,
        )
        self.collect_candidate_records = True
        self.frozen_gfm_teacher_selector.require_ready()
        return {
            "bound": True,
            "valid_node_count": int(
                self.normal_mechanism.valid_nodes.sum().item()
            ),
            "uses_test_labels": False,
            "teacher_trainable": False,
        }

    def next_coverage_balanced_schedule(self, batch_size):
        if not self.coverage_balanced_schedule_enable:
            return {}
        if self.normal_mechanism is None or not self.normal_mechanism.fitted:
            raise RuntimeError(
                "Coverage-balanced scheduling requires fitted Graph-VAR"
            )
        valid_nodes = torch.nonzero(
            self.normal_mechanism.valid_nodes.detach().bool().cpu(),
            as_tuple=False,
        ).flatten().tolist()
        cells = [
            (int(node), type_id, scale)
            for node in valid_nodes
            for type_id in range(len(self.synthetic_types))
            for scale in self._event_scale_names()
        ]
        if not cells:
            raise RuntimeError("Coverage-balanced scheduling has no valid cells")
        assigned = [
            cells[(self._coverage_schedule_cursor + index) % len(cells)]
            for index in range(int(batch_size))
        ]
        self._coverage_schedule_cursor += int(batch_size)
        return {
            "diagnostic_seed_ids": [cell[0] for cell in assigned],
            "diagnostic_type_ids": [cell[1] for cell in assigned],
            "diagnostic_event_scales": [cell[2] for cell in assigned],
        }

    def mechanism_matched_views(
        self,
        windows,
        adjacency,
        node_evidence=None,
        progress=1.0,
        collect_diagnostics=True,
    ):
        """生成共享目标轨迹、错位父节点响应的机制匹配三元组。"""
        if self.normal_mechanism is None or not self.normal_mechanism.fitted:
            raise RuntimeError("机制匹配视图需要已拟合的 Graph-VAR")
        builder = MechanismMatchedPairBuilder(
            self.normal_mechanism,
            lag_offset=self.mechanism_matched_lag_offset,
        )
        batch_size, win_size, n_var = windows.shape
        positive = windows.clone()
        negative = windows.clone()
        direct_mask = torch.zeros(
            batch_size,
            n_var,
            device=windows.device,
            dtype=torch.bool,
        )
        context_mask = torch.zeros_like(direct_mask)
        valid_mask = torch.zeros(
            batch_size,
            device=windows.device,
            dtype=torch.bool,
        )
        anomaly_type_ids = torch.zeros(
            batch_size,
            device=windows.device,
            dtype=torch.long,
        )
        seed_ids = torch.full(
            (batch_size,),
            -1,
            device=windows.device,
            dtype=torch.long,
        )
        segment_length = min(
            win_size,
            max(4, int(round(win_size * self.synthetic_time_ratio))),
        )
        strength = self._scheduled_edit_strength(progress)
        for batch_idx in range(batch_size):
            eligible = self.normal_mechanism.eligible_nodes(
                "neighbor_break",
                windows.device,
            )
            if eligible.numel() == 0:
                continue
            seed, _ = self._sample_seed(
                adjacency[batch_idx],
                n_var,
                node_evidence=node_evidence[batch_idx]
                if node_evidence is not None
                else None,
                eligible_nodes=eligible,
            )
            min_start = self.normal_mechanism.lags + 2
            max_start = win_size - segment_length
            if max_start < min_start:
                continue
            decision_rng = self._decision_rng or random
            start = decision_rng.randint(min_start, max_start)
            result = builder.build(
                windows[batch_idx],
                seed=seed,
                start=start,
                end=start + segment_length,
                strength=strength,
            )
            if result is None:
                continue
            positive[batch_idx] = result["positive"]
            negative[batch_idx] = result["negative"]
            direct_mask[batch_idx] = result["direct_mask"]
            context_mask[batch_idx] = result["context_mask"]
            valid_mask[batch_idx] = True
            seed_ids[batch_idx] = seed

        self.last_valid_mask = valid_mask
        self.last_sample_weights = valid_mask.to(windows.dtype)
        self.last_acceptance_tier = torch.where(
            valid_mask,
            torch.full_like(seed_ids, self.ACCEPTANCE_HARD),
            torch.full_like(seed_ids, self.ACCEPTANCE_INVALID),
        )
        self.last_rms_rescue_mask = torch.zeros_like(valid_mask)
        self.last_selected_delta_rms = torch.zeros_like(self.last_sample_weights)
        self.last_edit_source_ids = torch.where(
            valid_mask,
            torch.full_like(seed_ids, self.EDIT_SOURCE_DIRECT),
            torch.full_like(seed_ids, self.EDIT_SOURCE_INVALID),
        )
        self.last_fallback_type_ids = torch.full_like(seed_ids, -1)
        self.last_direct_type_ids = torch.where(
            valid_mask,
            torch.zeros_like(seed_ids),
            torch.full_like(seed_ids, -1),
        )
        self.last_seed_ids = seed_ids
        self.last_consistency_exclusion_mask = direct_mask | context_mask
        self.last_coverage_diagnostics = {
            "attempted_count": batch_size,
            "valid_count": int(valid_mask.sum().item()),
            "valid_ratio": float(valid_mask.float().mean().item()),
            "matched_lag_offset": self.mechanism_matched_lag_offset,
        }
        if collect_diagnostics and valid_mask.any():
            fairness = temporal_pair_fairness(
                windows[valid_mask],
                positive[valid_mask],
                negative[valid_mask],
            )
            self.last_diagnostics = {
                **self.last_coverage_diagnostics,
                **{
                    name: float(value.mean().item())
                    for name, value in fairness.items()
                },
                "target_shared_max_abs": float(
                    (positive[valid_mask] - negative[valid_mask])
                    .gather(2, seed_ids[valid_mask].view(-1, 1, 1).expand(-1, win_size, 1))
                    .abs()
                    .amax()
                    .item()
                ),
                "edit_mode": "mechanism_matched_lag",
                "edit_strength": float(strength),
            }
        else:
            self.last_diagnostics = dict(self.last_coverage_diagnostics)
        return positive, negative, direct_mask.float(), context_mask.float(), anomaly_type_ids

    def normal_mechanism_diagnostics(self):
        if self.normal_mechanism is None:
            return {}
        return self.normal_mechanism.diagnostics()

    def set_decision_seed(self, seed):
        """Isolate replay decisions from randomness consumed inside candidates."""
        self._decision_rng = None if seed is None else random.Random(int(seed))

    def normal_view(self, windows):
        if self.noise_std <= 0:
            return windows
        return windows + self.noise_std * torch.randn_like(windows)

    def positive_view(self, windows, adjacency):
        augmented = windows.clone()
        batch_size, _, n_var = augmented.shape
        for batch_idx in range(batch_size):
            nodes = self._sample_subgraph_nodes(adjacency[batch_idx], n_var, max(1, self.subgraph_topk))
            noise = self.positive_noise_std * torch.randn_like(augmented[batch_idx, :, nodes])
            augmented[batch_idx, :, nodes] = augmented[batch_idx, :, nodes] + noise
        return augmented

    def synthetic_anomaly(self, windows, adjacency=None):
        augmented = windows.clone()
        batch_size, win_size, n_var = augmented.shape
        node_mask = augmented.new_zeros(batch_size, n_var)
        n_nodes = max(1, int(round(n_var * self.negative_node_ratio)))
        seg_len = max(1, int(round(win_size * self.synthetic_time_ratio)))
        legacy_types = ("spike", "bias", "drift", "noise", "scale")
        available_types = tuple(t for t in self.synthetic_types if t in legacy_types) or legacy_types

        for batch_idx in range(batch_size):
            if adjacency is None:
                nodes = torch.randperm(n_var, device=windows.device)[:n_nodes]
            else:
                nodes = self._sample_subgraph_nodes(adjacency[batch_idx], n_var, n_nodes)
            start = random.randint(0, max(win_size - seg_len, 0))
            end = start + seg_len
            anomaly_type = random.choice(available_types)
            sign = -1.0 if random.random() < 0.5 else 1.0
            magnitude = sign * self.synthetic_magnitude

            node_mask[batch_idx, nodes] = 1.0
            segment = augmented[batch_idx, start:end].index_select(dim=1, index=nodes)

            if anomaly_type == "spike":
                spike_idx = random.randint(start, end - 1)
                row = augmented[batch_idx, spike_idx].clone()
                row[nodes] = row[nodes] + magnitude
                augmented[batch_idx, spike_idx] = row
            elif anomaly_type == "bias":
                augmented[batch_idx, start:end] = self._assign_nodes(
                    augmented[batch_idx, start:end],
                    nodes,
                    segment + magnitude,
                )
            elif anomaly_type == "drift":
                drift = torch.linspace(
                    0.0,
                    magnitude,
                    steps=end - start,
                    device=windows.device,
                    dtype=windows.dtype,
                ).unsqueeze(-1)
                augmented[batch_idx, start:end] = self._assign_nodes(
                    augmented[batch_idx, start:end],
                    nodes,
                    segment + drift,
                )
            elif anomaly_type == "noise":
                augmented[batch_idx, start:end] = self._assign_nodes(
                    augmented[batch_idx, start:end],
                    nodes,
                    segment + abs(self.synthetic_magnitude) * torch.randn_like(segment),
                )
            elif anomaly_type == "scale":
                augmented[batch_idx, start:end] = self._assign_nodes(
                    augmented[batch_idx, start:end],
                    nodes,
                    segment * (1.0 + abs(self.synthetic_magnitude) * 0.25),
                )
            else:
                raise ValueError(f"Unsupported synthetic anomaly type: {anomaly_type}")

        return augmented, node_mask

    def _sample_relation_preserving_view(self, windows, adjacency):
        augmented = windows.clone()
        batch_size, win_size, n_var = augmented.shape
        for batch_idx in range(batch_size):
            nodes = self._sample_subgraph_nodes(adjacency[batch_idx], n_var, max(1, self.subgraph_topk))
            if self.positive_view_mode == "legacy":
                start = 0
                end = win_size
            else:
                seg_len = min(
                    win_size,
                    max(2, int(round(win_size * self.positive_time_ratio))),
                )
                start = random.randint(0, max(win_size - seg_len, 0))
                end = start + seg_len

            segment = augmented[batch_idx, start:end].index_select(dim=1, index=nodes)
            scale = 1.0 + self.positive_noise_std * torch.randn(
                1,
                1,
                device=windows.device,
                dtype=windows.dtype,
            )
            bias = self.positive_noise_std * torch.randn(
                1,
                1,
                device=windows.device,
                dtype=windows.dtype,
            )
            if self.positive_view_mode == "legacy":
                jitter = self.positive_noise_std * torch.randn_like(segment)
            else:
                jitter = segment.new_zeros(segment.shape)
            augmented[batch_idx] = self._assign_nodes(
                augmented[batch_idx],
                nodes,
                torch.cat(
                    [
                        augmented[batch_idx, :start].index_select(dim=1, index=nodes),
                        segment * scale + bias + jitter,
                        augmented[batch_idx, end:].index_select(dim=1, index=nodes),
                    ],
                    dim=0,
                ),
            )
        return augmented

    @staticmethod
    def _normalized_positive_temporal_drift(anchor, candidates, eps=1e-6):
        if candidates.dim() != 4 or tuple(candidates.shape[1:]) != tuple(anchor.shape):
            raise ValueError(
                "Positive candidates must have shape [K, B, T, N] matching the anchor"
            )
        delta = (candidates - anchor.unsqueeze(0)).square().mean(dim=(2, 3)).sqrt()
        centered = anchor - anchor.mean(dim=1, keepdim=True)
        reference = centered.square().mean(dim=(1, 2)).sqrt()
        return delta / reference.clamp_min(float(eps)).unsqueeze(0)

    def _graph_mechanism_positive_drift(self, anchor, candidates):
        if self.normal_mechanism is None or not self.normal_mechanism.fitted:
            raise RuntimeError(
                "Graph-consistent positive selection requires a fitted Graph-VAR mechanism"
            )
        candidate_count, batch_size, time_steps, node_count = candidates.shape
        with torch.no_grad():
            positive_evidence = (
                self.normal_mechanism.positive_selection_evidence_windows
            )
            anchor_evidence = positive_evidence(anchor)
            flat_candidates = candidates.reshape(
                candidate_count * batch_size,
                time_steps,
                node_count,
            )
            candidate_evidence = positive_evidence(flat_candidates)
            per_source = []
            for name in ("conditional_rms", "relation_increment"):
                reference = anchor_evidence[name].unsqueeze(0)
                values = candidate_evidence[name].reshape(
                    candidate_count,
                    batch_size,
                    node_count,
                )
                finite = torch.isfinite(reference) & torch.isfinite(values)
                normalized = (values - reference).abs() / (
                    1.0 + reference.abs()
                )
                numerator = torch.where(
                    finite,
                    normalized,
                    torch.zeros_like(normalized),
                ).sum(dim=2)
                denominator = finite.sum(dim=2)
                drift = numerator / denominator.clamp_min(1)
                drift = torch.where(
                    denominator > 0,
                    drift,
                    torch.full_like(drift, float("inf")),
                )
                per_source.append(drift)
            stacked = torch.stack(per_source, dim=0)
            finite_sources = torch.isfinite(stacked)
            source_sum = torch.where(
                finite_sources,
                stacked,
                torch.zeros_like(stacked),
            ).sum(dim=0)
            source_count = finite_sources.sum(dim=0)
            return torch.where(
                source_count > 0,
                source_sum / source_count.clamp_min(1),
                torch.full_like(source_sum, float("inf")),
            )

    @staticmethod
    def _select_graph_consistent_positive_indices(
        graph_drift,
        temporal_drift,
        temporal_quantile,
    ):
        if graph_drift.shape != temporal_drift.shape or graph_drift.dim() != 2:
            raise ValueError(
                "Positive selection drifts must share shape [K, B]"
            )
        temporal_floor = torch.quantile(
            temporal_drift,
            float(temporal_quantile),
            dim=0,
        )
        eligible = temporal_drift >= temporal_floor.unsqueeze(0)
        finite_graph = torch.isfinite(graph_drift)
        screened = torch.where(
            eligible & finite_graph,
            graph_drift,
            torch.full_like(graph_drift, float("inf")),
        )
        selected = screened.argmin(dim=0)
        has_screened = torch.isfinite(screened).any(dim=0)
        fallback = torch.where(
            finite_graph,
            graph_drift,
            torch.full_like(graph_drift, float("inf")),
        ).argmin(dim=0)
        selected = torch.where(has_screened, selected, fallback)
        return selected, temporal_floor, has_screened

    @staticmethod
    def _select_graph_bounded_temporal_positive_indices(
        graph_drift,
        temporal_drift,
        graph_quantile,
    ):
        """Keep graph-consistent candidates, then maximize temporal diversity.

        The gate is computed independently for each anchor from the candidate
        pool and therefore uses no test information. A fallback retains the
        most temporally informative finite candidate if all graph values are
        non-finite, preventing silent positive-view collapse.
        """
        if graph_drift.shape != temporal_drift.shape or graph_drift.dim() != 2:
            raise ValueError(
                "Positive selection drifts must share shape [K, B]"
            )
        graph_floor = torch.quantile(graph_drift, float(graph_quantile), dim=0)
        finite_graph = torch.isfinite(graph_drift)
        eligible = finite_graph & (graph_drift <= graph_floor.unsqueeze(0))
        finite_temporal = torch.isfinite(temporal_drift)
        screened = torch.where(
            eligible & finite_temporal,
            temporal_drift,
            torch.full_like(temporal_drift, float("-inf")),
        )
        selected = screened.argmax(dim=0)
        has_screened = torch.isfinite(screened).any(dim=0)
        fallback = torch.where(
            finite_temporal,
            temporal_drift,
            torch.full_like(temporal_drift, float("-inf")),
        ).argmax(dim=0)
        selected = torch.where(has_screened, selected, fallback)
        return selected, graph_floor, has_screened

    def relation_preserving_view(self, windows, adjacency):
        self.last_positive_selection_diagnostics = {}
        if not self.graph_consistent_positive_enable:
            return self._sample_relation_preserving_view(windows, adjacency)

        candidates = torch.stack(
            [
                self._sample_relation_preserving_view(windows, adjacency)
                for _ in range(self.graph_consistent_positive_candidates)
            ],
            dim=0,
        )
        temporal_drift = self._normalized_positive_temporal_drift(
            windows,
            candidates,
        )
        graph_drift = self._graph_mechanism_positive_drift(
            windows,
            candidates,
        )
        if self.graph_consistent_positive_selector == (
            "max_temporal_after_graph_gate"
        ):
            selected, graph_floor, has_screened = (
                self._select_graph_bounded_temporal_positive_indices(
                    graph_drift,
                    temporal_drift,
                    self.graph_consistent_positive_graph_quantile,
                )
            )
            temporal_floor = torch.quantile(
                temporal_drift,
                self.graph_consistent_positive_temporal_quantile,
                dim=0,
            )
        else:
            selected, temporal_floor, has_screened = (
                self._select_graph_consistent_positive_indices(
                    graph_drift,
                    temporal_drift,
                    self.graph_consistent_positive_temporal_quantile,
                )
            )
            graph_floor = torch.quantile(
                graph_drift,
                self.graph_consistent_positive_graph_quantile,
                dim=0,
            )
        batch_indices = torch.arange(windows.size(0), device=windows.device)
        selected_view = candidates[selected, batch_indices]
        selected_graph_drift = graph_drift[selected, batch_indices]
        selected_temporal_drift = temporal_drift[selected, batch_indices]
        finite_candidates = torch.isfinite(graph_drift)
        finite_count = finite_candidates.sum().clamp_min(1)
        finite_selected = torch.isfinite(selected_graph_drift)
        selected_graph_total = torch.where(
            finite_selected,
            selected_graph_drift,
            torch.zeros_like(selected_graph_drift),
        ).sum()
        selected_graph_count = finite_selected.sum().clamp_min(1)
        self.last_positive_selection_diagnostics = {
            "candidate_count": int(self.graph_consistent_positive_candidates),
            "batch_size": int(windows.size(0)),
            "screened_selection_count": int(has_screened.sum().item()),
            "candidate_graph_drift_mean": float(
                torch.where(
                    finite_candidates,
                    graph_drift,
                    torch.zeros_like(graph_drift),
                ).sum().div(finite_count).detach().cpu()
            ),
            "selected_graph_drift_mean": float(
                selected_graph_total.div(selected_graph_count).detach().cpu()
            ),
            "candidate_temporal_drift_mean": float(
                temporal_drift.mean().detach().cpu()
            ),
            "selected_temporal_drift_mean": float(
                selected_temporal_drift.mean().detach().cpu()
            ),
            "temporal_floor_mean": float(temporal_floor.mean().detach().cpu()),
            "graph_floor_mean": float(graph_floor.mean().detach().cpu()),
            "selector": self.graph_consistent_positive_selector,
        }
        return selected_view

    def relation_function_breaking_view(
        self,
        windows,
        adjacency,
        node_evidence=None,
        progress=1.0,
        collect_diagnostics=True,
        orthogonality_reference_view=None,
        collect_wf_feasibility_candidates=False,
        frontend_reference=None,
        diagnostic_seed_ids=None,
        diagnostic_type_ids=None,
        diagnostic_event_scales=None,
    ):
        self.last_candidate_records = []
        self.last_valid_mask = None
        self.last_sample_weights = None
        self.last_acceptance_tier = None
        self.last_rms_rescue_mask = None
        self.last_selected_delta_rms = None
        self.last_edit_source_ids = None
        self.last_fallback_type_ids = None
        self.last_direct_type_ids = None
        self.last_seed_ids = None
        self.last_consistency_exclusion_mask = None
        self.last_wf_temporal_drift = None
        self.last_wf_relational_drift = None
        self.last_wf_relational_floor = None
        self.last_wf_orthogonality_diagnostics = {}
        self.last_wf_feasibility_candidates = []
        self.last_coverage_diagnostics = {}
        if collect_wf_feasibility_candidates and not self.wf_orthogonality_enable:
            raise ValueError(
                "Wf feasibility collection requires the orthogonality contract"
            )
        if self.wf_orthogonality_enable and (
            self.synthetic_mode != "relation_function_breaking"
            or self.edit_mode != "direct_quantile"
            or not set(self.synthetic_types).issubset(
                set(self.DIRECT_FALLBACK_TYPES)
            )
        ):
            raise ValueError(
                "Wf orthogonality is only implemented for the "
                "relation_function_breaking/direct_quantile path"
            )
        if self.synthetic_mode == "random_noise":
            return self.random_noise_breaking_view(
                windows,
                progress=progress,
                collect_diagnostics=collect_diagnostics,
            )
        if self.synthetic_mode != "relation_function_breaking":
            raise ValueError(f"Unsupported SHARK.SYNTHETIC.MODE: {self.synthetic_mode}")
        if not self.synthetic_types:
            raise ValueError("SHARK.SYNTHETIC.TYPES must contain at least one counterfactual edit")
        if self.edit_mode == "direct_quantile":
            unsupported_types = set(self.synthetic_types) - set(self.DIRECT_FALLBACK_TYPES)
            if not unsupported_types:
                return self._direct_quantile_view(
                    windows,
                    adjacency,
                    node_evidence=node_evidence,
                    progress=progress,
                    collect_diagnostics=collect_diagnostics,
                    orthogonality_reference_view=(
                        orthogonality_reference_view
                    ),
                    collect_wf_feasibility_candidates=(
                        collect_wf_feasibility_candidates
                    ),
                    frontend_reference=frontend_reference,
                    diagnostic_seed_ids=diagnostic_seed_ids,
                    diagnostic_type_ids=diagnostic_type_ids,
                    diagnostic_event_scales=diagnostic_event_scales,
                )
            if not unsupported_types.issubset(set(self.diagnostic_types)):
                raise ValueError(
                    "direct_quantile only supports direct edit types: "
                    f"{sorted(unsupported_types)}"
                )
        if self.edit_mode in ("surrogate_boundary", "hybrid_valid") and all(
            anomaly_type in self.SURROGATE_TYPES for anomaly_type in self.synthetic_types
        ):
            return self._surrogate_boundary_view(
                windows,
                adjacency,
                node_evidence=node_evidence,
                progress=progress,
                collect_diagnostics=collect_diagnostics,
            )

        augmented = windows.clone()
        batch_size, win_size, n_var = augmented.shape
        direct_node_mask = augmented.new_zeros(batch_size, n_var)
        propagated_node_mask = augmented.new_zeros(batch_size, n_var)
        anomaly_type_ids = torch.empty(batch_size, device=windows.device, dtype=torch.long)
        seg_len = min(win_size, max(2, int(round(win_size * self.synthetic_time_ratio))))
        magnitude = self._scheduled_magnitude(progress)
        edit_strength = self._scheduled_edit_strength(progress)
        selected_evidence = []
        evidence_selected = 0
        random_selected = 0
        mechanism_prior = self._mechanism_seed_prior(
            windows.device,
            windows.dtype,
        )
        type_counts = augmented.new_zeros(len(self.synthetic_types))
        type_delta_rms_sum = (
            augmented.new_zeros(len(self.synthetic_types)) if collect_diagnostics else None
        )

        for batch_idx in range(batch_size):
            evidence_row = (
                mechanism_prior
                if mechanism_prior is not None
                else (None if node_evidence is None else node_evidence[batch_idx])
            )
            seed, neighbors, used_evidence = self._sample_seed_and_neighbors(
                adjacency[batch_idx],
                n_var,
                self.subgraph_topk,
                evidence_row,
            )
            start = random.randint(0, max(win_size - seg_len, 0))
            end = start + seg_len
            anomaly_type_id = random.randrange(len(self.synthetic_types))
            anomaly_type = self.synthetic_types[anomaly_type_id]

            direct_node_mask[batch_idx, seed] = 1.0
            anomaly_type_ids[batch_idx] = anomaly_type_id
            type_counts[anomaly_type_id] += 1.0
            if used_evidence:
                evidence_selected += 1
            else:
                random_selected += 1
            if collect_diagnostics and evidence_row is not None:
                selected_evidence.append(evidence_row[seed].detach())

            original_segment = augmented[batch_idx, start:end, seed].clone()
            interpolation_strength = edit_strength if self.edit_mode == "interpolated" else None

            if anomaly_type == "lag_break":
                self._lag_break(augmented, batch_idx, seed, start, end, interpolation_strength)
            elif anomaly_type == "neighbor_break":
                self._neighbor_break(
                    augmented,
                    windows,
                    adjacency[batch_idx],
                    batch_idx,
                    seed,
                    neighbors,
                    start,
                    end,
                    interpolation_strength,
                )
            elif anomaly_type == "relation_flip":
                self._relation_flip(
                    augmented,
                    adjacency[batch_idx],
                    batch_idx,
                    seed,
                    neighbors,
                    start,
                    end,
                    interpolation_strength,
                )
            elif anomaly_type == "local_function_shift":
                self._local_function_shift(
                    augmented,
                    batch_idx,
                    seed,
                    start,
                    end,
                    magnitude,
                    interpolation_strength,
                )
            elif anomaly_type == "propagation_break":
                propagated = self._propagation_break(
                    augmented,
                    batch_idx,
                    seed,
                    neighbors,
                    start,
                    end,
                    magnitude,
                    interpolation_strength,
                )
                if propagated.numel() > 0:
                    propagated_node_mask[batch_idx, propagated] = 1.0
            else:
                raise ValueError(f"Unsupported v4 synthetic anomaly type: {anomaly_type}")

            if collect_diagnostics:
                edited_segment = augmented[batch_idx, start:end, seed]
                channel_scale = self._channel_scale(windows[batch_idx, :, seed])
                normalized_delta = (edited_segment - original_segment) / channel_scale
                type_delta_rms_sum[anomaly_type_id] += normalized_delta.square().mean().sqrt()

        if not collect_diagnostics:
            self.last_diagnostics = {}
        elif selected_evidence:
            seed_evidence_mean = torch.stack(selected_evidence).mean().item()
        else:
            seed_evidence_mean = float("nan")
        if collect_diagnostics:
            type_delta_rms = type_delta_rms_sum / type_counts.clamp_min(1.0)
            self.last_diagnostics = {
                "seed_evidence_mean": seed_evidence_mean,
                "evidence_sampling_ratio": evidence_selected / max(batch_size, 1),
                "random_sampling_ratio": random_selected / max(batch_size, 1),
                "edit_mode": self.edit_mode,
                "edit_strength": float(edit_strength),
                "magnitude": float(magnitude),
                "type_ratio": (type_counts / max(batch_size, 1)).detach().cpu().tolist(),
                "type_delta_rms": type_delta_rms.detach().cpu().tolist(),
            }
        self.last_valid_mask = torch.ones(
            batch_size,
            device=windows.device,
            dtype=torch.bool,
        )
        self.last_sample_weights = torch.ones(
            batch_size,
            device=windows.device,
            dtype=windows.dtype,
        )
        self.last_acceptance_tier = torch.full(
            (batch_size,),
            self.ACCEPTANCE_HARD,
            device=windows.device,
            dtype=torch.long,
        )
        self.last_edit_source_ids = torch.full(
            (batch_size,),
            self.EDIT_SOURCE_FALLBACK,
            device=windows.device,
            dtype=torch.long,
        )
        self.last_fallback_type_ids = torch.full(
            (batch_size,),
            -1,
            device=windows.device,
            dtype=torch.long,
        )
        self.last_seed_ids = direct_node_mask.argmax(dim=1).long()
        seed_counts = direct_node_mask.sum(dim=0)
        self.last_coverage_diagnostics = {
            "attempted_count": batch_size,
            "valid_count": batch_size,
            "fitted_window_count": 0,
            "surrogate_accepted_count": 0,
            "fallback_count": batch_size,
            "invalid_count": 0,
            "attempted_seed_counts": seed_counts.detach().cpu().tolist(),
            "valid_seed_counts": seed_counts.detach().cpu().tolist(),
            "fallback_type_counts": {},
        }
        return augmented, direct_node_mask, propagated_node_mask, anomaly_type_ids

    @staticmethod
    def _wf_temporal_drift(anchor, candidate, eps=1e-6):
        """整窗时间漂移；与 OTR-CL method-integrity audit 使用同一公式。"""
        squeeze = anchor.dim() == 2
        if squeeze:
            anchor = anchor.unsqueeze(0)
            candidate = candidate.unsqueeze(0)
        result = normalized_temporal_drift(
            anchor,
            candidate,
            eps=eps,
        )
        return result[0] if squeeze else result

    @staticmethod
    def _wf_abs_pearson_graph(windows, eps=1e-6):
        squeeze = windows.dim() == 2
        if squeeze:
            windows = windows.unsqueeze(0)
        if windows.dim() != 3:
            raise ValueError("Wf Pearson graph expects [B, T, N]")
        nodes = windows.transpose(1, 2)
        nodes = nodes - nodes.mean(dim=-1, keepdim=True)
        nodes = nodes / nodes.norm(dim=-1, keepdim=True).clamp_min(
            float(eps)
        )
        graph = torch.bmm(nodes, nodes.transpose(1, 2)).abs()
        return graph[0] if squeeze else graph

    @classmethod
    def _wf_relational_drift(cls, anchor, candidate, eps=1e-6):
        squeeze = anchor.dim() == 2
        if squeeze:
            anchor = anchor.unsqueeze(0)
            candidate = candidate.unsqueeze(0)
        reference = cls._wf_abs_pearson_graph(anchor, eps=eps)
        changed = cls._wf_abs_pearson_graph(candidate, eps=eps)
        result = normalized_graph_drift(
            reference,
            changed,
            eps=eps,
        )
        return result[0] if squeeze else result

    def _wf_candidate_contract(
        self,
        temporal_drift,
        relational_drift,
        relational_floor,
    ):
        temporal_pass = bool(
            torch.isfinite(temporal_drift).item()
            and temporal_drift <= self.wf_max_temporal_drift + 1e-6
        )
        relational_pass = bool(
            torch.isfinite(relational_drift).item()
            and relational_drift >= relational_floor - 1e-6
        )
        return {
            "temporal_pass": temporal_pass,
            "relational_pass": relational_pass,
            "accepted": temporal_pass and relational_pass,
        }

    @staticmethod
    def _wf_candidate_is_better(candidate, incumbent):
        if incumbent is None:
            return True
        temporal_better = bool(
            candidate["wf_temporal_drift"]
            < incumbent["wf_temporal_drift"] - 1e-8
        )
        temporal_tie = bool(
            (
                candidate["wf_temporal_drift"]
                - incumbent["wf_temporal_drift"]
            ).abs()
            <= 1e-8
        )
        relation_better = bool(
            candidate["wf_relational_drift"]
            > incumbent["wf_relational_drift"] + 1e-8
        )
        relation_tie = bool(
            (
                candidate["wf_relational_drift"]
                - incumbent["wf_relational_drift"]
            ).abs()
            <= 1e-8
        )
        return bool(
            temporal_better
            or (
                temporal_tie
                and (
                    relation_better
                    or (
                        relation_tie
                        and candidate["penalty"] < incumbent["penalty"]
                    )
                )
            )
        )

    def _direct_quantile_view(
        self,
        windows,
        adjacency,
        node_evidence=None,
        progress=1.0,
        collect_diagnostics=True,
        orthogonality_reference_view=None,
        collect_wf_feasibility_candidates=False,
        frontend_reference=None,
        diagnostic_seed_ids=None,
        diagnostic_type_ids=None,
        diagnostic_event_scales=None,
    ):
        augmented = windows.clone()
        batch_size, _, n_var = windows.shape
        n_types = len(self.synthetic_types)
        direct_node_mask = windows.new_zeros(batch_size, n_var)
        propagated_node_mask = windows.new_zeros(batch_size, n_var)
        anomaly_type_ids = torch.empty(
            batch_size,
            device=windows.device,
            dtype=torch.long,
        )
        valid_mask = torch.zeros(batch_size, device=windows.device, dtype=torch.bool)
        sample_weights = windows.new_zeros(batch_size)
        acceptance_tier = torch.full(
            (batch_size,),
            self.ACCEPTANCE_INVALID,
            device=windows.device,
            dtype=torch.long,
        )
        rms_rescue_mask = torch.zeros(
            batch_size,
            device=windows.device,
            dtype=torch.bool,
        )
        selected_delta_rms = windows.new_zeros(batch_size)
        selected_wf_temporal_drift = windows.new_full(
            (batch_size,),
            float("nan"),
        )
        selected_wf_relational_drift = windows.new_full(
            (batch_size,),
            float("nan"),
        )
        if self.wf_orthogonality_enable:
            if orthogonality_reference_view is None:
                raise ValueError(
                    "Wf orthogonality requires the Wc reference view"
                )
            if (
                not isinstance(orthogonality_reference_view, torch.Tensor)
                or orthogonality_reference_view.shape != windows.shape
            ):
                raise ValueError(
                    "Wf orthogonality reference must match windows [B, T, N]"
                )
            coarse_relational_drift = self._wf_relational_drift(
                windows,
                orthogonality_reference_view,
            )
            wf_relational_floor = (
                coarse_relational_drift + self.wf_min_relational_margin
            )
        else:
            coarse_relational_drift = windows.new_zeros(batch_size)
            wf_relational_floor = windows.new_zeros(batch_size)
        edit_source_ids = torch.full(
            (batch_size,),
            self.EDIT_SOURCE_INVALID,
            device=windows.device,
            dtype=torch.long,
        )
        direct_type_ids = torch.full(
            (batch_size,),
            -1,
            device=windows.device,
            dtype=torch.long,
        )
        seed_ids = torch.full(
            (batch_size,),
            -1,
            device=windows.device,
            dtype=torch.long,
        )
        consistency_exclusion_mask = torch.zeros(
            batch_size,
            n_var,
            device=windows.device,
            dtype=torch.bool,
        )
        attempted_seed_counts = windows.new_zeros(n_var)
        valid_seed_counts = windows.new_zeros(n_var)
        type_attempted_counts = windows.new_zeros(n_types)
        type_valid_counts = windows.new_zeros(n_types)
        type_delta_rms_sum = windows.new_zeros(n_types)
        type_score_sum = windows.new_zeros(n_types)
        type_target_sum = windows.new_zeros(n_types)
        type_reference_scale_sum = windows.new_zeros(n_types)
        type_normalized_gain_sum = windows.new_zeros(n_types)
        type_sample_weight_sum = windows.new_zeros(n_types)
        type_hard_counts = windows.new_zeros(n_types)
        type_soft_counts = windows.new_zeros(n_types)
        type_candidate_total_counts = windows.new_zeros(n_types)
        type_candidate_finite_counts = windows.new_zeros(n_types)
        type_candidate_plausible_counts = windows.new_zeros(n_types)
        type_candidate_below_floor_counts = windows.new_zeros(n_types)
        type_candidate_above_cap_counts = windows.new_zeros(n_types)
        type_candidate_hard_eligible_counts = windows.new_zeros(n_types)
        type_candidate_soft_eligible_counts = windows.new_zeros(n_types)
        type_candidate_rms_rescue_counts = windows.new_zeros(n_types)
        type_reference_floor_checks = windows.new_zeros(n_types)
        type_reference_floor_hits = windows.new_zeros(n_types)
        type_reference_scale_fallback_counts = windows.new_zeros(n_types)
        type_boundary_attempt_counts = windows.new_zeros(n_types)
        type_boundary_valid_counts = windows.new_zeros(n_types)
        type_boundary_insufficient_counts = windows.new_zeros(n_types)
        type_boundary_context_invalid_counts = windows.new_zeros(n_types)
        type_boundary_nonfinite_counts = windows.new_zeros(n_types)
        type_boundary_collapsed_counts = windows.new_zeros(n_types)
        type_boundary_unknown_counts = windows.new_zeros(n_types)
        type_candidate_normalized_gain_sum = windows.new_zeros(n_types)
        type_candidate_normalized_gain_count = windows.new_zeros(n_types)
        type_candidate_score_gain_sum = windows.new_zeros(n_types)
        type_candidate_score_gain_count = windows.new_zeros(n_types)
        type_above_cap_normalized_gain_sum = windows.new_zeros(n_types)
        type_above_cap_normalized_gain_count = windows.new_zeros(n_types)
        type_candidate_rms_sum = windows.new_zeros(n_types)
        type_support_eligible_counts = windows.new_zeros(n_types)
        type_support_collapsed_counts = windows.new_zeros(n_types)
        type_support_direction_clipped_counts = windows.new_zeros(n_types)
        type_candidate_support_clipped_counts = windows.new_zeros(n_types)
        type_candidate_support_violation_counts = windows.new_zeros(n_types)
        type_candidate_nonfinite_delta_counts = windows.new_zeros(n_types)
        type_candidate_below_min_rms_counts = windows.new_zeros(n_types)
        type_candidate_above_max_delta_counts = windows.new_zeros(n_types)
        type_candidate_context_valid_counts = windows.new_zeros(n_types)
        type_candidate_positive_gain_counts = windows.new_zeros(n_types)
        type_candidate_eligible_counts = windows.new_zeros(n_types)
        type_candidate_wf_contract_attempt_counts = windows.new_zeros(
            n_types
        )
        type_candidate_wf_temporal_rejected_counts = windows.new_zeros(
            n_types
        )
        type_candidate_wf_relational_rejected_counts = windows.new_zeros(
            n_types
        )
        type_candidate_wf_both_rejected_counts = windows.new_zeros(n_types)
        type_candidate_wf_accepted_counts = windows.new_zeros(n_types)
        type_selected_wf_temporal_drift_sum = windows.new_zeros(n_types)
        type_selected_wf_relational_drift_sum = windows.new_zeros(n_types)
        type_selected_orientation_sum = windows.new_zeros(n_types)
        type_selected_orientation_count = windows.new_zeros(n_types)
        gm_hns_candidate_in_band_count = windows.new_zeros(())
        gm_hns_candidate_out_of_band_count = windows.new_zeros(())
        gm_hns_selected_difficulty_sum = windows.new_zeros(())
        gm_hns_selected_count = windows.new_zeros(())
        frontend_effective_candidate_count = 0
        frontend_effective_eligible_count = 0
        frontend_effective_below_count = 0
        frontend_effective_above_count = 0
        frontend_effective_graph_nonlocal_count = 0
        frontend_effective_rescue_used_count = 0
        frontend_primary_nonlegacy_pool_count = 0
        frontend_primary_nonlegacy_eligible_count = 0
        frontend_primary_nonlegacy_selected_count = 0
        frontend_effective_selected_d_fd_sum = 0.0
        frontend_effective_selected_graph_drift_sum = 0.0
        self_temporal_route_attempt_count = 0
        self_temporal_route_selected_count = 0
        frontend_selector = self.frontend_graph_effective_selector
        if frontend_selector is not None:
            if frontend_reference is None:
                frontend_reference = frontend_selector.encode_original(windows)
            else:
                if (
                    frontend_reference.ndim != 3
                    or frontend_reference.size(0) != batch_size
                    or frontend_reference.size(1) != n_var
                ):
                    raise ValueError(
                        "Frontend reference must match [B, N, F] for v2.2 "
                        f"screening, got {tuple(frontend_reference.shape)}"
                    )
                frontend_reference = frontend_reference.detach()
        else:
            frontend_reference = None
        segment_attempt_counts = self._empty_event_scale_counts()
        segment_selected_counts = self._empty_event_scale_counts()
        segment_attempt_length_sums = self._empty_event_scale_sums()
        segment_selected_length_sums = self._empty_event_scale_sums()
        type_segment_attempt_counts = {
            anomaly_type: self._empty_event_scale_counts()
            for anomaly_type in self.synthetic_types
        }
        type_segment_selected_counts = {
            anomaly_type: self._empty_event_scale_counts()
            for anomaly_type in self.synthetic_types
        }
        selected_evidence = []
        evidence_selected = 0
        random_selected = 0
        mechanism_prior = self._mechanism_seed_prior(
            windows.device,
            windows.dtype,
        )
        target_quantile = self._scheduled_direct_quantile(progress)
        decision_rng = self._decision_rng or random
        if diagnostic_type_ids is None:
            type_schedule = (
                list(range(n_types)) * math.ceil(batch_size / max(n_types, 1))
            )[:batch_size]
            decision_rng.shuffle(type_schedule)
        else:
            type_schedule = [int(value) for value in diagnostic_type_ids]
            if len(type_schedule) != batch_size or any(
                value < 0 or value >= n_types for value in type_schedule
            ):
                raise ValueError(
                    "Diagnostic type ids must contain one valid id per window"
                )
        if diagnostic_seed_ids is not None and len(diagnostic_seed_ids) != batch_size:
            raise ValueError(
                "Diagnostic seed ids must contain one node id per window"
            )
        if diagnostic_event_scales is not None:
            diagnostic_event_scales = [
                str(value) for value in diagnostic_event_scales
            ]
            if len(diagnostic_event_scales) != batch_size or any(
                value not in self._event_scale_names()
                for value in diagnostic_event_scales
            ):
                raise ValueError(
                    "Diagnostic event scales must contain one configured "
                    "scale per window"
                )

        for batch_idx in range(batch_size):
            anomaly_type_id = type_schedule[batch_idx]
            anomaly_type = self.synthetic_types[anomaly_type_id]
            evidence_row = (
                mechanism_prior
                if mechanism_prior is not None
                else (None if node_evidence is None else node_evidence[batch_idx])
            )
            eligible_nodes = None
            if diagnostic_seed_ids is not None:
                forced_seed = int(diagnostic_seed_ids[batch_idx])
                if forced_seed < 0 or forced_seed >= n_var:
                    raise ValueError(
                        f"Invalid diagnostic seed id {forced_seed} for {n_var} nodes"
                    )
                eligible_nodes = torch.as_tensor(
                    [forced_seed],
                    device=adjacency.device,
                    dtype=torch.long,
                )
            elif self.direct_mechanism_score_mode == "adaptive_graph_var":
                if self.mechanism_availability_routing_enable:
                    eligible_nodes = self.normal_mechanism.eligible_nodes(
                        anomaly_type,
                        adjacency.device,
                        include_self_temporal=True,
                    )
                else:
                    eligible_nodes = self.normal_mechanism.eligible_nodes(
                        anomaly_type,
                        adjacency.device,
                    )
            seed, neighbors, used_evidence = self._sample_seed_and_neighbors(
                adjacency[batch_idx],
                n_var,
                self.surrogate_parent_candidates,
                evidence_row,
                eligible_nodes=eligible_nodes,
            )
            anomaly_type_ids[batch_idx] = anomaly_type_id
            type_attempted_counts[anomaly_type_id] += 1.0
            attempted_seed_counts[seed] += 1.0
            seed_ids[batch_idx] = seed
            use_self_temporal = bool(
                self.mechanism_availability_routing_enable
                and anomaly_type == "local_function_shift"
                and self.normal_mechanism.uses_self_temporal_fallback(
                    seed,
                    adjacency.device,
                )
            )
            if use_self_temporal:
                self_temporal_route_attempt_count += 1
            if frontend_selector is not None:
                frontend_selector.record_proposal(seed)
            if used_evidence:
                evidence_selected += 1
            else:
                random_selected += 1
            if collect_diagnostics and evidence_row is not None:
                selected_evidence.append(evidence_row[seed].detach())

            support = self._direct_candidate_support(windows, seed)
            if support["collapsed"]:
                type_support_collapsed_counts[anomaly_type_id] += 1.0
                continue
            type_support_eligible_counts[anomaly_type_id] += 1.0

            best_candidate = None
            frontend_candidate_pool = []
            for _ in range(self.direct_quantile_segment_trials):
                forced_event_scale = (
                    None
                    if diagnostic_event_scales is None
                    else diagnostic_event_scales[batch_idx]
                )
                start, end = self._sample_surrogate_segment(
                    windows.size(1),
                    event_scale=forced_event_scale,
                )
                segment_length = end - start
                segment_scale = self._segment_scale_name(
                    segment_length,
                    windows.size(1),
                )
                segment_attempt_counts[segment_scale] += 1
                segment_attempt_length_sums[segment_scale] += segment_length
                type_segment_attempt_counts[anomaly_type][segment_scale] += 1
                type_boundary_attempt_counts[anomaly_type_id] += 1.0
                self._last_direct_boundary_reason = "unknown"
                boundary_args = (
                    windows,
                    adjacency[batch_idx],
                    batch_idx,
                    seed,
                    neighbors,
                    start,
                    end,
                    anomaly_type,
                    target_quantile,
                )
                boundary = (
                    self._direct_quantile_boundary(
                        *boundary_args,
                        self_temporal=True,
                    )
                    if use_self_temporal
                    else self._direct_quantile_boundary(*boundary_args)
                )
                if boundary is None:
                    reason = getattr(
                        self,
                        "_last_direct_boundary_reason",
                        "unknown",
                    )
                    if reason == "insufficient_reference":
                        type_boundary_insufficient_counts[anomaly_type_id] += 1.0
                    elif reason == "context_invalid":
                        type_boundary_context_invalid_counts[anomaly_type_id] += 1.0
                    elif reason == "nonfinite":
                        type_boundary_nonfinite_counts[anomaly_type_id] += 1.0
                    elif reason == "collapsed_tail":
                        type_boundary_collapsed_counts[anomaly_type_id] += 1.0
                    else:
                        type_boundary_unknown_counts[anomaly_type_id] += 1.0
                    continue
                type_boundary_valid_counts[anomaly_type_id] += 1.0
                if boundary.get("reference_scale_fallback", False):
                    type_reference_scale_fallback_counts[anomaly_type_id] += 1.0
                gain_floor = (
                    boundary["original_score"]
                    + self.direct_quantile_min_reference_gain
                    * boundary["reference_scale"]
                )
                type_reference_floor_checks[anomaly_type_id] += 1.0
                if bool((gain_floor >= boundary["target"]).item()):
                    type_reference_floor_hits[anomaly_type_id] += 1.0

                augmented[batch_idx, :, seed] = windows[batch_idx, :, seed]
                original_segment = windows[batch_idx, start:end, seed]
                channel_scale = self._channel_scale(windows[batch_idx, :, seed])
                preserve_mechanism = (
                    self.normal_mechanism is not None
                )
                candidate_segments = []
                if preserve_mechanism:
                    strength_levels = self._direct_intervention_strengths(
                        windows,
                        progress,
                        anomaly_type,
                    )
                    candidate_segments, candidate_sources = self._normal_candidate_segments(
                        windows[batch_idx], seed, start, end, anomaly_type, strength_levels,
                        phase=batch_idx + seed + start,
                        self_temporal=use_self_temporal,
                    )
                else:
                    self._apply_direct_edit_candidate(
                        augmented,
                        windows,
                        adjacency[batch_idx],
                        batch_idx,
                        seed,
                        neighbors,
                        start,
                        end,
                        anomaly_type,
                        progress,
                    )
                    edited_segment, direction_clipped = (
                        self._project_to_direct_support(
                            augmented[batch_idx, start:end, seed],
                            support,
                        )
                    )
                    if direction_clipped:
                        type_support_direction_clipped_counts[anomaly_type_id] += 1.0
                    normalized_delta = (
                        edited_segment - original_segment
                    ) / channel_scale
                    raw_rms = normalized_delta.square().mean().sqrt()
                    if not torch.isfinite(raw_rms) or raw_rms <= 1e-8:
                        continue
                    direction = normalized_delta / raw_rms
                    target_levels = torch.linspace(
                        self.hybrid_fallback_min_rms,
                        self.hybrid_fallback_max_rms,
                        steps=self.hybrid_fallback_candidates,
                        device=windows.device,
                        dtype=windows.dtype,
                    )
                    candidate_segments = [
                        original_segment + direction * target_rms * channel_scale
                        for target_rms in target_levels
                    ]
                    candidate_sources = ["legacy"] * len(candidate_segments)

                for candidate_values, candidate_source in zip(candidate_segments, candidate_sources):
                    type_candidate_total_counts[anomaly_type_id] += 1.0
                    candidate_delta = (candidate_values - original_segment) / channel_scale
                    if self.max_delta_std > 0 and not preserve_mechanism:
                        candidate_delta = candidate_delta.clamp(
                            min=-self.max_delta_std,
                            max=self.max_delta_std,
                        )
                        candidate_values = (
                            original_segment + candidate_delta * channel_scale
                        )
                    candidate_values, support_clipped = self._project_to_direct_support(
                        candidate_values,
                        support,
                    )
                    if support_clipped:
                        type_candidate_support_clipped_counts[anomaly_type_id] += 1.0
                        if preserve_mechanism:
                            continue
                    delta_scale = support.get("delta_scale", channel_scale)
                    candidate_delta = (candidate_values - original_segment) / delta_scale
                    actual_rms = candidate_delta.square().mean().sqrt()
                    if not torch.isfinite(actual_rms):
                        type_candidate_nonfinite_delta_counts[
                            anomaly_type_id
                        ] += 1.0
                        continue
                    if actual_rms + 1e-6 < self.hybrid_fallback_min_rms:
                        type_candidate_below_min_rms_counts[
                            anomaly_type_id
                        ] += 1.0
                        continue
                    if (
                        preserve_mechanism
                        and self.max_delta_std > 0
                        and not support.get("discrete", False)
                        and candidate_delta.abs().max() > self.max_delta_std
                    ):
                        type_candidate_above_max_delta_counts[
                            anomaly_type_id
                        ] += 1.0
                        continue
                    support_violation = self._direct_support_violation(
                        candidate_values,
                        support,
                    )
                    if support_violation:
                        type_candidate_support_violation_counts[anomaly_type_id] += 1.0
                        continue
                    candidate_window = windows[batch_idx:batch_idx + 1].clone()
                    candidate_window[0, start:end, seed] = candidate_values
                    if self.direct_mechanism_score_mode == "adaptive_graph_var":
                        score_args = (
                            windows[batch_idx:batch_idx + 1],
                            candidate_window,
                            seed,
                            start,
                            end,
                        )
                        scores, mechanism_details = (
                            self.normal_mechanism.score_counterfactual(
                                *score_args,
                                self_temporal=True,
                            )
                            if use_self_temporal
                            else self.normal_mechanism.score_counterfactual(
                                *score_args
                            )
                        )
                        self._last_mechanism_details = mechanism_details
                        candidate_score = scores[0]
                    else:
                        candidate_score = self._direct_mechanism_scores(
                            candidate_window,
                            adjacency[batch_idx],
                            seed,
                            neighbors,
                            start,
                            end,
                            anomaly_type,
                            normalization_windows=windows[
                                batch_idx:batch_idx + 1
                            ],
                        )[0]
                        mechanism_details = self._last_mechanism_details
                    context_valid = bool(
                        mechanism_details.get(
                            "context_valid",
                            torch.tensor([False], device=windows.device),
                        )[0]
                    )
                    orientation = mechanism_details.get(
                        "orientation",
                        torch.zeros(1, device=windows.device, dtype=windows.dtype),
                    )[0]
                    if context_valid:
                        type_candidate_context_valid_counts[anomaly_type_id] += 1.0
                    status = self._direct_quantile_candidate_status(
                        candidate_score,
                        actual_rms,
                        boundary,
                    )
                    if status["finite"]:
                        type_candidate_finite_counts[anomaly_type_id] += 1.0
                        score_gain = candidate_score - boundary["original_score"]
                        normalized_gain = score_gain / boundary["reference_scale"]
                        type_candidate_score_gain_sum[
                            anomaly_type_id
                        ] += score_gain.detach()
                        type_candidate_score_gain_count[anomaly_type_id] += 1.0
                        type_candidate_normalized_gain_sum[
                            anomaly_type_id
                        ] += normalized_gain.detach()
                        type_candidate_normalized_gain_count[anomaly_type_id] += 1.0
                        type_candidate_rms_sum[anomaly_type_id] += actual_rms.detach()
                    else:
                        normalized_gain = candidate_score.new_tensor(float("nan"))
                    if status["plausible"]:
                        type_candidate_plausible_counts[anomaly_type_id] += 1.0
                        if context_valid:
                            type_candidate_eligible_counts[anomaly_type_id] += 1.0
                            if bool(
                                (candidate_score > boundary["original_score"]).item()
                            ):
                                type_candidate_positive_gain_counts[
                                    anomaly_type_id
                                ] += 1.0
                    if status["below_gain_floor"]:
                        type_candidate_below_floor_counts[anomaly_type_id] += 1.0
                    if status["above_cap"]:
                        type_candidate_above_cap_counts[anomaly_type_id] += 1.0
                        type_above_cap_normalized_gain_sum[
                            anomaly_type_id
                        ] += normalized_gain.detach()
                        type_above_cap_normalized_gain_count[anomaly_type_id] += 1.0
                    if status["hard_eligible"]:
                        type_candidate_hard_eligible_counts[anomaly_type_id] += 1.0
                    if status["soft_eligible"]:
                        type_candidate_soft_eligible_counts[anomaly_type_id] += 1.0
                    if status["rms_rescue_eligible"]:
                        type_candidate_rms_rescue_counts[anomaly_type_id] += 1.0
                    gm_hns_metadata = None
                    if self.negative_screening_mode == "gm_hns":
                        gm_hns_metadata = self._gm_hns_candidate_metadata(
                            candidate_score,
                            actual_rms,
                            boundary,
                        )
                        if status["accepted"] and gm_hns_metadata["in_band"]:
                            gm_hns_candidate_in_band_count += 1.0
                        elif status["accepted"]:
                            gm_hns_candidate_out_of_band_count += 1.0
                    if self.collect_candidate_records:
                        # Keep only train-normal metadata and the edited seed
                        # segment. The audit reconstructs the candidate window
                        # and runs the actual DLinear frontend on it.
                        self.last_candidate_records.append(
                            {
                                "batch_index": int(batch_idx),
                                "candidate_generator": candidate_source,
                                "seed": int(seed),
                                "anomaly_type": str(anomaly_type),
                                "start": int(start),
                                "end": int(end),
                                "values": candidate_values.detach().cpu(),
                                "candidate_score": float(candidate_score.detach().cpu()),
                                "boundary_target": float(boundary["target"].detach().cpu()),
                                "boundary_cap": float(boundary["cap"].detach().cpu()),
                                "reference_scale": float(boundary["reference_scale"].detach().cpu()),
                                "normalized_gain": float(normalized_gain.detach().cpu()),
                                "perturbation_rms": float(actual_rms.detach().cpu()),
                                "sample_weight": float(status["sample_weight"]),
                                "acceptance_tier": int(
                                    status["acceptance_tier"]
                                ),
                                "rms_rescue_eligible": bool(
                                    status["rms_rescue_eligible"]
                                ),
                                "finite": bool(status["finite"]),
                                "plausible": bool(status["plausible"]),
                                "hard_eligible": bool(status["hard_eligible"]),
                                "soft_eligible": bool(status["soft_eligible"]),
                                "accepted": bool(status["accepted"]),
                                "context_valid": bool(context_valid),
                                "support_clipped": bool(support_clipped),
                                "gm_hns_in_band": (
                                    bool(gm_hns_metadata["in_band"])
                                    if gm_hns_metadata is not None
                                    else None
                                ),
                            }
                        )
                    if (
                        frontend_selector is not None
                        and self.negative_screening_mode == "frontend_primary"
                        and not status["accepted"]
                        and status["plausible"]
                        and context_valid
                    ):
                        # The new selector replaces the Graph-VAR score boundary
                        # as the admission decision.  Candidate construction,
                        # empirical support, finite/RMS and context checks have
                        # already passed; DLinear visibility and graph locality
                        # are evaluated in one batched pool below.
                        penalty = (
                            (candidate_score - boundary["target"]).abs()
                            / boundary["reference_scale"]
                            + 0.01 * actual_rms
                        )
                        frontend_candidate_pool.append(
                            {
                                "candidate_generator": candidate_source,
                                "start": start,
                                "end": end,
                                "values": candidate_values.clone(),
                                "delta_rms": actual_rms.detach(),
                                "score": candidate_score.detach(),
                                "target": boundary["target"].detach(),
                                "reference_scale": boundary[
                                    "reference_scale"
                                ].detach(),
                                "normalized_gain": normalized_gain.detach(),
                                "sample_weight": candidate_score.new_tensor(
                                    self.frontend_primary_nonlegacy_weight
                                ),
                                "acceptance_tier": self.ACCEPTANCE_SOFT,
                                "rms_rescue_eligible": False,
                                "achievement": status["achievement"],
                                "orientation": orientation.detach(),
                                "support_clipped": support_clipped,
                                "penalty": penalty.detach(),
                                "wf_temporal_drift": candidate_score.new_tensor(
                                    float("nan")
                                ),
                                "wf_relational_drift": candidate_score.new_tensor(
                                    float("nan")
                                ),
                                "legacy_accepted": False,
                            }
                        )
                        frontend_primary_nonlegacy_pool_count += 1
                    if not status["accepted"]:
                        continue
                    if frontend_selector is not None and not context_valid:
                        # The v2.2 audit/replay contract is the intersection
                        # of legacy acceptance and candidate-context validity.
                        continue
                    wf_temporal_drift = candidate_score.new_tensor(
                        float("nan")
                    )
                    wf_relational_drift = candidate_score.new_tensor(
                        float("nan")
                    )
                    if self.wf_orthogonality_enable:
                        type_candidate_wf_contract_attempt_counts[
                            anomaly_type_id
                        ] += 1.0
                        wf_temporal_drift = self._wf_temporal_drift(
                            windows[batch_idx],
                            candidate_window[0],
                        )
                        wf_relational_drift = self._wf_relational_drift(
                            windows[batch_idx],
                            candidate_window[0],
                        )
                        wf_contract = self._wf_candidate_contract(
                            wf_temporal_drift,
                            wf_relational_drift,
                            wf_relational_floor[batch_idx],
                        )
                        if (
                            collect_wf_feasibility_candidates
                            and wf_contract["temporal_pass"]
                        ):
                            feasibility_penalty = (
                                (candidate_score - boundary["target"]).abs()
                                / boundary["reference_scale"]
                                + 0.01 * actual_rms
                            )
                            self.last_wf_feasibility_candidates.append({
                                "batch_index": int(batch_idx),
                                "direct_node": int(seed),
                                "anomaly_type": str(anomaly_type),
                                "start": int(start),
                                "end": int(end),
                                "window": candidate_window[0].detach().clone(),
                                "temporal_drift": float(
                                    wf_temporal_drift.detach().cpu()
                                ),
                                "raw_relational_drift": float(
                                    wf_relational_drift.detach().cpu()
                                ),
                                "raw_relational_floor": float(
                                    wf_relational_floor[batch_idx].detach().cpu()
                                ),
                                "mechanism_penalty": float(
                                    feasibility_penalty.detach().cpu()
                                ),
                                "perturbation_rms": float(
                                    actual_rms.detach().cpu()
                                ),
                            })
                        if not wf_contract["temporal_pass"]:
                            type_candidate_wf_temporal_rejected_counts[
                                anomaly_type_id
                            ] += 1.0
                        if not wf_contract["relational_pass"]:
                            type_candidate_wf_relational_rejected_counts[
                                anomaly_type_id
                            ] += 1.0
                        if (
                            not wf_contract["temporal_pass"]
                            and not wf_contract["relational_pass"]
                        ):
                            type_candidate_wf_both_rejected_counts[
                                anomaly_type_id
                            ] += 1.0
                        if not wf_contract["accepted"]:
                            continue
                        type_candidate_wf_accepted_counts[
                            anomaly_type_id
                        ] += 1.0
                    penalty = (
                        (candidate_score - boundary["target"]).abs()
                        / boundary["reference_scale"]
                        + 0.01 * actual_rms
                    )
                    self._record_generator_count(candidate_source, "accepted", 1, seed, anomaly_type, start, end, windows.size(1))
                    candidate_record = {
                        "candidate_generator": candidate_source,
                        "start": start,
                        "end": end,
                        "values": candidate_values.clone(),
                        "delta_rms": actual_rms.detach(),
                        "score": candidate_score.detach(),
                        "target": boundary["target"].detach(),
                        "reference_scale": boundary["reference_scale"].detach(),
                        "normalized_gain": normalized_gain.detach(),
                        "sample_weight": status["sample_weight"],
                        "acceptance_tier": status["acceptance_tier"],
                        "rms_rescue_eligible": status["rms_rescue_eligible"],
                        "achievement": status["achievement"],
                        "orientation": orientation.detach(),
                        "support_clipped": support_clipped,
                        "penalty": penalty.detach(),
                        "wf_temporal_drift": wf_temporal_drift.detach(),
                        "wf_relational_drift": wf_relational_drift.detach(),
                        "legacy_accepted": True,
                    }
                    if gm_hns_metadata is not None:
                        candidate_record.update(
                            {
                                "gm_hns_difficulty": gm_hns_metadata[
                                    "difficulty"
                                ],
                                "gm_hns_in_band": gm_hns_metadata["in_band"],
                                "gm_hns_penalty": gm_hns_metadata["penalty"],
                            }
                        )
                    if frontend_selector is not None:
                        # The candidate already passed the legacy mechanism,
                        # support, context, and optional Wf contracts.  Batch
                        # the remaining DLinear/Pearson checks per window.
                        frontend_candidate_pool.append(candidate_record)
                        select_candidate = False
                    elif self.wf_orthogonality_enable:
                        select_candidate = self._wf_candidate_is_better(
                            candidate_record,
                            best_candidate,
                        )
                    elif self.negative_screening_mode == "gm_hns":
                        select_candidate = self._gm_hns_candidate_is_better(
                            candidate_record,
                            best_candidate,
                        )
                    elif best_candidate is None:
                        select_candidate = True
                    else:
                        select_candidate = bool(
                            penalty < best_candidate["penalty"]
                        )
                    if select_candidate:
                        best_candidate = candidate_record

            if frontend_selector is not None:
                best_candidate, frontend_diagnostics = frontend_selector.select(
                    windows[batch_idx],
                    frontend_reference[batch_idx],
                    adjacency[batch_idx],
                    seed,
                    frontend_candidate_pool,
                )
                frontend_selector.record_pool(
                    seed,
                    frontend_diagnostics["evaluated"],
                    frontend_diagnostics["eligible"],
                    best_candidate is not None,
                )
                frontend_effective_candidate_count += frontend_diagnostics[
                    "evaluated"
                ]
                frontend_effective_eligible_count += frontend_diagnostics[
                    "eligible"
                ]
                frontend_effective_below_count += frontend_diagnostics[
                    "below_frontend"
                ]
                frontend_effective_above_count += frontend_diagnostics[
                    "above_frontend"
                ]
                frontend_effective_graph_nonlocal_count += frontend_diagnostics[
                    "graph_nonlocal"
                ]
                if frontend_diagnostics.get("coverage_rescue_used", False):
                    frontend_effective_rescue_used_count += 1
                frontend_primary_nonlegacy_eligible_count += int(
                    frontend_diagnostics.get("eligible_nonlegacy", 0)
                )

            augmented[batch_idx, :, seed] = windows[batch_idx, :, seed]
            if best_candidate is None:
                continue
            if use_self_temporal:
                self_temporal_route_selected_count += 1
            self._record_generator_count(best_candidate["candidate_generator"], "selected", 1,
                seed, anomaly_type, best_candidate["start"], best_candidate["end"], windows.size(1))

            start = best_candidate["start"]
            end = best_candidate["end"]
            segment_length = end - start
            segment_scale = self._segment_scale_name(
                segment_length,
                windows.size(1),
            )
            segment_selected_counts[segment_scale] += 1
            segment_selected_length_sums[segment_scale] += segment_length
            type_segment_selected_counts[anomaly_type][segment_scale] += 1
            augmented[batch_idx, start:end, seed] = best_candidate["values"]
            direct_node_mask[batch_idx, seed] = 1.0
            if self.normal_mechanism is not None:
                consistency_exclusion_mask[batch_idx] = (
                    self.normal_mechanism.affected_context_mask(
                        seed,
                        windows.device,
                    )
                )
            propagated = self._scope_aware_negative_propagation(
                augmented,
                windows,
                propagated_node_mask,
                batch_idx,
                seed,
                neighbors,
                adjacency[batch_idx, seed],
                start,
                end,
                best_candidate["values"],
            )
            if propagated.numel() > 0:
                consistency_exclusion_mask[batch_idx, propagated] = True
            valid_mask[batch_idx] = True
            edit_source_ids[batch_idx] = self.EDIT_SOURCE_DIRECT
            direct_type_ids[batch_idx] = anomaly_type_id
            sample_weight = best_candidate["sample_weight"]
            if self.scope_aware_negative_enable and propagated.numel() == 0:
                sample_weight *= self.scope_aware_negative_singleton_weight
            sample_weights[batch_idx] = sample_weight
            acceptance_tier[batch_idx] = best_candidate["acceptance_tier"]
            rms_rescue_mask[batch_idx] = best_candidate["rms_rescue_eligible"]
            selected_delta_rms[batch_idx] = best_candidate["delta_rms"]
            if self.negative_screening_mode == "gm_hns":
                difficulty = best_candidate.get("gm_hns_difficulty")
                if difficulty is not None and bool(torch.isfinite(difficulty).item()):
                    gm_hns_selected_difficulty_sum += difficulty
                    gm_hns_selected_count += 1.0
            elif frontend_selector is not None:
                frontend_effective_selected_d_fd_sum += float(
                    best_candidate["frontend_direct_drift"].item()
                )
                frontend_effective_selected_graph_drift_sum += float(
                    best_candidate["graph_nonincident_drift"].item()
                )
                if not bool(best_candidate.get("legacy_accepted", True)):
                    frontend_primary_nonlegacy_selected_count += 1
            if self.wf_orthogonality_enable:
                selected_wf_temporal_drift[batch_idx] = best_candidate[
                    "wf_temporal_drift"
                ]
                selected_wf_relational_drift[batch_idx] = best_candidate[
                    "wf_relational_drift"
                ]
                type_selected_wf_temporal_drift_sum[
                    anomaly_type_id
                ] += best_candidate["wf_temporal_drift"]
                type_selected_wf_relational_drift_sum[
                    anomaly_type_id
                ] += best_candidate["wf_relational_drift"]
            valid_seed_counts[seed] += 1.0
            type_valid_counts[anomaly_type_id] += 1.0
            type_sample_weight_sum[anomaly_type_id] += best_candidate["sample_weight"]
            if best_candidate["acceptance_tier"] == self.ACCEPTANCE_HARD:
                type_hard_counts[anomaly_type_id] += 1.0
            else:
                type_soft_counts[anomaly_type_id] += 1.0
            type_delta_rms_sum[anomaly_type_id] += best_candidate["delta_rms"]
            type_score_sum[anomaly_type_id] += best_candidate["score"]
            type_target_sum[anomaly_type_id] += best_candidate["target"]
            type_reference_scale_sum[anomaly_type_id] += best_candidate[
                "reference_scale"
            ]
            type_normalized_gain_sum[anomaly_type_id] += best_candidate[
                "normalized_gain"
            ]
            if anomaly_type == "relation_flip":
                type_selected_orientation_sum[anomaly_type_id] += best_candidate[
                    "orientation"
                ]
                type_selected_orientation_count[anomaly_type_id] += 1.0

        valid_count = int(valid_mask.sum().item())
        hard_count = int((acceptance_tier == self.ACCEPTANCE_HARD).sum().item())
        soft_count = int((acceptance_tier == self.ACCEPTANCE_SOFT).sum().item())
        sample_weight_sum = float(sample_weights.sum().item())
        rms_rescue_count = int(rms_rescue_mask.sum().item())
        propagated_sample_count = int(
            (propagated_node_mask.sum(dim=1) > 0).sum().item()
        )
        propagated_node_count = int(propagated_node_mask.sum().item())
        wf_selected_mask = valid_mask & torch.isfinite(
            selected_wf_temporal_drift
        ) & torch.isfinite(selected_wf_relational_drift)
        safe_valid_counts = type_valid_counts.clamp_min(1.0)
        type_attempted_dict = {
            name: int(type_attempted_counts[idx].item())
            for idx, name in enumerate(self.synthetic_types)
        }
        type_valid_dict = {
            name: int(type_valid_counts[idx].item())
            for idx, name in enumerate(self.synthetic_types)
        }
        coverage_diagnostics = {
            "attempted_count": batch_size,
            "valid_count": valid_count,
            "fitted_window_count": 0,
            "surrogate_accepted_count": 0,
            "fallback_count": 0,
            "fallback_attempt_count": 0,
            "fallback_rejected_count": 0,
            "mechanism_availability_routing_enabled": bool(
                self.mechanism_availability_routing_enable
            ),
            "self_temporal_route_attempt_count": int(
                self_temporal_route_attempt_count
            ),
            "self_temporal_route_selected_count": int(
                self_temporal_route_selected_count
            ),
            "invalid_count": batch_size - valid_count,
            "hard_count": hard_count,
            "soft_count": soft_count,
            "sample_weight_sum": sample_weight_sum,
            "effective_weighted_count": sample_weight_sum,
            "scope_aware_negative_enable": bool(
                self.scope_aware_negative_enable
            ),
            "scope_propagated_sample_count": propagated_sample_count,
            "scope_propagated_node_count": propagated_node_count,
            "scope_propagation_ratio": propagated_sample_count / max(batch_size, 1),
            "scope_node_mean": propagated_node_count / max(valid_count, 1),
            "selected_delta_rms_sum": float(selected_delta_rms.sum().item()),
            "rms_rescue_sample_count": rms_rescue_count,
            "rms_rescue_delta_rms_sum": float(
                selected_delta_rms[rms_rescue_mask].sum().item()
            ),
            "candidate_total_count": int(type_candidate_total_counts.sum().item()),
            "candidate_finite_count": int(type_candidate_finite_counts.sum().item()),
            "candidate_plausible_count": int(
                type_candidate_plausible_counts.sum().item()
            ),
            "candidate_below_gain_floor_count": int(
                type_candidate_below_floor_counts.sum().item()
            ),
            "candidate_above_cap_count": int(
                type_candidate_above_cap_counts.sum().item()
            ),
            "candidate_hard_eligible_count": int(
                type_candidate_hard_eligible_counts.sum().item()
            ),
            "candidate_soft_eligible_count": int(
                type_candidate_soft_eligible_counts.sum().item()
            ),
            "candidate_rms_rescue_count": int(
                type_candidate_rms_rescue_counts.sum().item()
            ),
            "reference_floor_check_count": int(
                type_reference_floor_checks.sum().item()
            ),
            "reference_floor_hit_count": int(
                type_reference_floor_hits.sum().item()
            ),
            "reference_scale_fallback_count": int(
                type_reference_scale_fallback_counts.sum().item()
            ),
            "boundary_attempt_count": int(type_boundary_attempt_counts.sum().item()),
            "boundary_valid_count": int(type_boundary_valid_counts.sum().item()),
            "boundary_insufficient_count": int(
                type_boundary_insufficient_counts.sum().item()
            ),
            "boundary_context_invalid_count": int(
                type_boundary_context_invalid_counts.sum().item()
            ),
            "boundary_nonfinite_count": int(
                type_boundary_nonfinite_counts.sum().item()
            ),
            "boundary_collapsed_count": int(
                type_boundary_collapsed_counts.sum().item()
            ),
            "boundary_unknown_count": int(type_boundary_unknown_counts.sum().item()),
            "candidate_normalized_gain_sum": float(
                type_candidate_normalized_gain_sum.sum().item()
            ),
            "candidate_normalized_gain_count": int(
                type_candidate_normalized_gain_count.sum().item()
            ),
            "candidate_score_gain_sum": float(
                type_candidate_score_gain_sum.sum().item()
            ),
            "candidate_score_gain_count": int(
                type_candidate_score_gain_count.sum().item()
            ),
            "above_cap_normalized_gain_sum": float(
                type_above_cap_normalized_gain_sum.sum().item()
            ),
            "above_cap_normalized_gain_count": int(
                type_above_cap_normalized_gain_count.sum().item()
            ),
            "candidate_rms_sum": float(type_candidate_rms_sum.sum().item()),
            "support_eligible_count": int(type_support_eligible_counts.sum().item()),
            "support_collapsed_count": int(type_support_collapsed_counts.sum().item()),
            "support_direction_clipped_count": int(
                type_support_direction_clipped_counts.sum().item()
            ),
            "candidate_support_clipped_count": int(
                type_candidate_support_clipped_counts.sum().item()
            ),
            "candidate_support_violation_count": int(
                type_candidate_support_violation_counts.sum().item()
            ),
            "candidate_nonfinite_delta_count": int(
                type_candidate_nonfinite_delta_counts.sum().item()
            ),
            "candidate_below_min_rms_count": int(
                type_candidate_below_min_rms_counts.sum().item()
            ),
            "candidate_above_max_delta_count": int(
                type_candidate_above_max_delta_counts.sum().item()
            ),
            "candidate_context_valid_count": int(
                type_candidate_context_valid_counts.sum().item()
            ),
            "candidate_positive_gain_count": int(
                type_candidate_positive_gain_counts.sum().item()
            ),
            "candidate_eligible_count": int(
                type_candidate_eligible_counts.sum().item()
            ),
            "wf_orthogonality_enabled": bool(
                self.wf_orthogonality_enable
            ),
            "wf_candidate_contract_attempt_count": int(
                type_candidate_wf_contract_attempt_counts.sum().item()
            ),
            "wf_candidate_temporal_rejected_count": int(
                type_candidate_wf_temporal_rejected_counts.sum().item()
            ),
            "wf_candidate_relational_rejected_count": int(
                type_candidate_wf_relational_rejected_counts.sum().item()
            ),
            "wf_candidate_both_rejected_count": int(
                type_candidate_wf_both_rejected_counts.sum().item()
            ),
            "wf_candidate_orthogonal_accepted_count": int(
                type_candidate_wf_accepted_counts.sum().item()
            ),
            "wf_selected_temporal_drift_sum": float(
                selected_wf_temporal_drift[wf_selected_mask].sum().item()
            ),
            "wf_selected_relational_drift_sum": float(
                selected_wf_relational_drift[wf_selected_mask].sum().item()
            ),
            "wf_selected_relational_floor_sum": float(
                wf_relational_floor[wf_selected_mask].sum().item()
            ),
            "wf_selected_count": int(wf_selected_mask.sum().item()),
            "wf_max_temporal_drift": float(self.wf_max_temporal_drift),
            "wf_min_relational_margin": float(
                self.wf_min_relational_margin
            ),
            "selected_orientation_sum": float(
                type_selected_orientation_sum.sum().item()
            ),
            "selected_orientation_count": int(
                type_selected_orientation_count.sum().item()
            ),
            "negative_screening_mode": self.negative_screening_mode,
            "gm_hns_candidate_in_band_count": int(
                gm_hns_candidate_in_band_count.item()
            ),
            "gm_hns_candidate_out_of_band_count": int(
                gm_hns_candidate_out_of_band_count.item()
            ),
            "gm_hns_selected_difficulty_mean": float(
                gm_hns_selected_difficulty_sum.item()
                / max(float(gm_hns_selected_count.item()), 1.0)
            ),
            "gm_hns_selected_count": int(gm_hns_selected_count.item()),
            "gm_hns_difficulty_low": float(self.gm_hns_difficulty_low),
            "gm_hns_difficulty_high": float(self.gm_hns_difficulty_high),
            "gm_hns_difficulty_target": float(self.gm_hns_difficulty_target),
            "frontend_effective_candidate_count": int(
                frontend_effective_candidate_count
            ),
            "frontend_effective_eligible_count": int(
                frontend_effective_eligible_count
            ),
            "frontend_effective_below_count": int(frontend_effective_below_count),
            "frontend_effective_above_count": int(frontend_effective_above_count),
            "frontend_effective_graph_nonlocal_count": int(
                frontend_effective_graph_nonlocal_count
            ),
            "frontend_effective_rescue_used_count": int(
                frontend_effective_rescue_used_count
            ),
            "frontend_primary_nonlegacy_pool_count": int(
                frontend_primary_nonlegacy_pool_count
            ),
            "frontend_primary_nonlegacy_eligible_count": int(
                frontend_primary_nonlegacy_eligible_count
            ),
            "frontend_primary_nonlegacy_selected_count": int(
                frontend_primary_nonlegacy_selected_count
            ),
            "frontend_effective_selected_d_fd_mean": (
                frontend_effective_selected_d_fd_sum / max(valid_count, 1)
            ),
            "frontend_effective_selected_d_fd_sum": (
                frontend_effective_selected_d_fd_sum
            ),
            "frontend_effective_selected_graph_drift_mean": (
                frontend_effective_selected_graph_drift_sum / max(valid_count, 1)
            ),
            "frontend_effective_selected_graph_drift_sum": (
                frontend_effective_selected_graph_drift_sum
            ),
            "direct_count": valid_count,
            "direct_attempt_count": batch_size,
            "attempted_seed_counts": attempted_seed_counts.detach().cpu().tolist(),
            "valid_seed_counts": valid_seed_counts.detach().cpu().tolist(),
            "fallback_type_counts": {},
            "fallback_type_mechanism_gain_sums": {},
            "fallback_mechanism_gain_sum": 0.0,
            "fallback_relation_gain_sum": 0.0,
            "fallback_temporal_gain_sum": 0.0,
            "direct_type_attempted_counts": type_attempted_dict,
            "direct_type_valid_counts": type_valid_dict,
            "direct_type_hard_counts": {
                name: int(type_hard_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_soft_counts": {
                name: int(type_soft_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_invalid_counts": {
                name: int(
                    (type_attempted_counts[idx] - type_valid_counts[idx]).item()
                )
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_sample_weight_sums": {
                name: float(type_sample_weight_sum[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_total_counts": {
                name: int(type_candidate_total_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_finite_counts": {
                name: int(type_candidate_finite_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_plausible_counts": {
                name: int(type_candidate_plausible_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_below_floor_counts": {
                name: int(type_candidate_below_floor_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_above_cap_counts": {
                name: int(type_candidate_above_cap_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_hard_eligible_counts": {
                name: int(type_candidate_hard_eligible_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_soft_eligible_counts": {
                name: int(type_candidate_soft_eligible_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_rms_rescue_counts": {
                name: int(type_candidate_rms_rescue_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_reference_floor_checks": {
                name: int(type_reference_floor_checks[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_reference_floor_hits": {
                name: int(type_reference_floor_hits[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_reference_scale_fallback_counts": {
                name: int(type_reference_scale_fallback_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_boundary_attempt_counts": {
                name: int(type_boundary_attempt_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_boundary_valid_counts": {
                name: int(type_boundary_valid_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_boundary_insufficient_counts": {
                name: int(type_boundary_insufficient_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_boundary_context_invalid_counts": {
                name: int(type_boundary_context_invalid_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_boundary_nonfinite_counts": {
                name: int(type_boundary_nonfinite_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_boundary_collapsed_counts": {
                name: int(type_boundary_collapsed_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_boundary_unknown_counts": {
                name: int(type_boundary_unknown_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_normalized_gain_sums": {
                name: float(type_candidate_normalized_gain_sum[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_normalized_gain_counts": {
                name: int(type_candidate_normalized_gain_count[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_score_gain_sums": {
                name: float(type_candidate_score_gain_sum[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_score_gain_counts": {
                name: int(type_candidate_score_gain_count[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_above_cap_normalized_gain_sums": {
                name: float(type_above_cap_normalized_gain_sum[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_above_cap_normalized_gain_counts": {
                name: int(type_above_cap_normalized_gain_count[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_rms_sums": {
                name: float(type_candidate_rms_sum[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_support_eligible_counts": {
                name: int(type_support_eligible_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_support_collapsed_counts": {
                name: int(type_support_collapsed_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_support_direction_clipped_counts": {
                name: int(type_support_direction_clipped_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_support_clipped_counts": {
                name: int(type_candidate_support_clipped_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_support_violation_counts": {
                name: int(type_candidate_support_violation_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_nonfinite_delta_counts": {
                name: int(type_candidate_nonfinite_delta_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_below_min_rms_counts": {
                name: int(type_candidate_below_min_rms_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_above_max_delta_counts": {
                name: int(type_candidate_above_max_delta_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_context_valid_counts": {
                name: int(type_candidate_context_valid_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_positive_gain_counts": {
                name: int(type_candidate_positive_gain_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_eligible_counts": {
                name: int(type_candidate_eligible_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_wf_temporal_rejected_counts": {
                name: int(
                    type_candidate_wf_temporal_rejected_counts[idx].item()
                )
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_wf_relational_rejected_counts": {
                name: int(
                    type_candidate_wf_relational_rejected_counts[idx].item()
                )
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_wf_contract_attempt_counts": {
                name: int(
                    type_candidate_wf_contract_attempt_counts[idx].item()
                )
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_wf_both_rejected_counts": {
                name: int(
                    type_candidate_wf_both_rejected_counts[idx].item()
                )
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_candidate_wf_accepted_counts": {
                name: int(type_candidate_wf_accepted_counts[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_selected_wf_temporal_drift_sums": {
                name: float(
                    type_selected_wf_temporal_drift_sum[idx].item()
                )
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_selected_wf_relational_drift_sums": {
                name: float(
                    type_selected_wf_relational_drift_sum[idx].item()
                )
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_selected_orientation_sums": {
                name: float(type_selected_orientation_sum[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_selected_orientation_counts": {
                name: int(type_selected_orientation_count[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_score_sums": {
                name: float(type_score_sum[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_target_sums": {
                name: float(type_target_sum[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_reference_scale_sums": {
                name: float(type_reference_scale_sum[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "direct_type_normalized_gain_sums": {
                name: float(type_normalized_gain_sum[idx].item())
                for idx, name in enumerate(self.synthetic_types)
            },
            "segment_length_policy": self.segment_length_policy,
            "direct_segment_attempt_counts": segment_attempt_counts,
            "direct_segment_selected_counts": segment_selected_counts,
            "direct_segment_attempt_length_sums": segment_attempt_length_sums,
            "direct_segment_selected_length_sums": segment_selected_length_sums,
            "direct_type_segment_attempt_counts": type_segment_attempt_counts,
            "direct_type_segment_selected_counts": type_segment_selected_counts,
        }
        self.last_valid_mask = valid_mask
        self.last_sample_weights = sample_weights
        self.last_acceptance_tier = acceptance_tier
        self.last_rms_rescue_mask = rms_rescue_mask
        self.last_selected_delta_rms = selected_delta_rms
        self.last_edit_source_ids = edit_source_ids
        self.last_fallback_type_ids = None
        self.last_direct_type_ids = direct_type_ids
        self.last_seed_ids = seed_ids
        self.last_consistency_exclusion_mask = consistency_exclusion_mask
        self.last_wf_temporal_drift = selected_wf_temporal_drift
        self.last_wf_relational_drift = selected_wf_relational_drift
        self.last_wf_relational_floor = wf_relational_floor
        self.last_wf_orthogonality_diagnostics = {
            key: coverage_diagnostics[key]
            for key in (
                "wf_orthogonality_enabled",
                "wf_candidate_contract_attempt_count",
                "wf_candidate_temporal_rejected_count",
                "wf_candidate_relational_rejected_count",
                "wf_candidate_both_rejected_count",
                "wf_candidate_orthogonal_accepted_count",
                "wf_selected_temporal_drift_sum",
                "wf_selected_relational_drift_sum",
                "wf_selected_relational_floor_sum",
                "wf_selected_count",
                "wf_max_temporal_drift",
                "wf_min_relational_margin",
            )
        }
        if frontend_selector is not None:
            coverage_diagnostics.update(frontend_selector.running_diagnostics())
        self.last_coverage_diagnostics = coverage_diagnostics

        if not collect_diagnostics:
            self.last_diagnostics = {}
            return augmented, direct_node_mask, propagated_node_mask, anomaly_type_ids

        seed_evidence_mean = (
            torch.stack(selected_evidence).mean().item()
            if selected_evidence
            else float("nan")
        )
        self.last_diagnostics = {
            **coverage_diagnostics,
            "seed_evidence_mean": seed_evidence_mean,
            "evidence_sampling_ratio": evidence_selected / max(batch_size, 1),
            "random_sampling_ratio": random_selected / max(batch_size, 1),
            "edit_mode": self.edit_mode,
            "edit_strength": float(self._scheduled_edit_strength(progress)),
            "magnitude": float("nan"),
            "direct_target_quantile": target_quantile,
            "direct_acceptance_mode": self.direct_quantile_acceptance_mode,
            "direct_reference_scale_mode": (
                self.direct_quantile_reference_scale_mode
            ),
            "direct_upper_bound_mode": self.direct_quantile_upper_bound_mode,
            "direct_mechanism_score_mode": self.direct_mechanism_score_mode,
            "direct_candidate_support_mode": self.direct_candidate_support_mode,
            "negative_screening_mode": self.negative_screening_mode,
            "direct_validity_ratio": valid_count / max(batch_size, 1),
            "hard_coverage": hard_count / max(batch_size, 1),
            "soft_coverage": soft_count / max(batch_size, 1),
            "rms_rescue_coverage": rms_rescue_count / max(batch_size, 1),
            "rms_rescue_delta_rms_mean": float(
                selected_delta_rms[rms_rescue_mask].sum().item()
                / max(rms_rescue_count, 1)
            ),
            "effective_weighted_count": sample_weight_sum,
            "sample_weight_mean": sample_weight_sum / max(batch_size, 1),
            "accepted_sample_weight_mean": sample_weight_sum / max(valid_count, 1),
            "scope_aware_negative_enable": bool(
                self.scope_aware_negative_enable
            ),
            "scope_propagated_sample_count": propagated_sample_count,
            "scope_propagated_node_count": propagated_node_count,
            "scope_propagation_ratio": propagated_sample_count / max(batch_size, 1),
            "scope_node_mean": propagated_node_count / max(valid_count, 1),
            "candidate_finite_ratio": float(
                type_candidate_finite_counts.sum().item()
                / max(type_candidate_total_counts.sum().item(), 1.0)
            ),
            "candidate_rms_rescue_ratio": float(
                type_candidate_rms_rescue_counts.sum().item()
                / max(type_candidate_total_counts.sum().item(), 1.0)
            ),
            "candidate_positive_gain_ratio": float(
                type_candidate_positive_gain_counts.sum().item()
                / max(type_candidate_eligible_counts.sum().item(), 1.0)
            ),
            "candidate_score_gain_mean": float(
                type_candidate_score_gain_sum.sum().item()
                / max(type_candidate_score_gain_count.sum().item(), 1.0)
            ),
            "candidate_context_valid_ratio": float(
                type_candidate_context_valid_counts.sum().item()
                / max(type_candidate_total_counts.sum().item(), 1.0)
            ),
            "support_collapsed_ratio": float(
                type_support_collapsed_counts.sum().item()
                / max(type_attempted_counts.sum().item(), 1.0)
            ),
            "candidate_support_clipped_ratio": float(
                type_candidate_support_clipped_counts.sum().item()
                / max(type_candidate_total_counts.sum().item(), 1.0)
            ),
            "candidate_support_violation_ratio": float(
                type_candidate_support_violation_counts.sum().item()
                / max(type_candidate_total_counts.sum().item(), 1.0)
            ),
            "eligible_acceptance_ratio": float(
                type_valid_counts.sum().item()
                / max(type_support_eligible_counts.sum().item(), 1.0)
            ),
            "selected_orientation_mean": float(
                type_selected_orientation_sum.sum().item()
                / max(type_selected_orientation_count.sum().item(), 1.0)
            ),
            "reference_floor_hit_ratio": float(
                type_reference_floor_hits.sum().item()
                / max(type_reference_floor_checks.sum().item(), 1.0)
            ),
            "reference_scale_fallback_ratio": float(
                type_reference_scale_fallback_counts.sum().item()
                / max(type_boundary_valid_counts.sum().item(), 1.0)
            ),
            "invalid_ratio": (batch_size - valid_count) / max(batch_size, 1),
            "seed_coverage": float(
                (valid_seed_counts > 0).float().mean().item()
            ),
            "seed_entropy": self._normalized_count_entropy(valid_seed_counts),
            "attempted_seed_entropy": self._normalized_count_entropy(
                attempted_seed_counts
            ),
            "type_ratio": (
                type_attempted_counts / max(batch_size, 1)
            ).detach().cpu().tolist(),
            "type_valid_ratio": (
                type_valid_counts / type_attempted_counts.clamp_min(1.0)
            ).detach().cpu().tolist(),
            "type_hard_ratio": (
                type_hard_counts / type_attempted_counts.clamp_min(1.0)
            ).detach().cpu().tolist(),
            "type_soft_ratio": (
                type_soft_counts / type_attempted_counts.clamp_min(1.0)
            ).detach().cpu().tolist(),
            "type_invalid_ratio": (
                (type_attempted_counts - type_valid_counts)
                / type_attempted_counts.clamp_min(1.0)
            ).detach().cpu().tolist(),
            "type_sample_weight": (
                type_sample_weight_sum / safe_valid_counts
            ).detach().cpu().tolist(),
            "type_delta_rms": (
                type_delta_rms_sum / safe_valid_counts
            ).detach().cpu().tolist(),
            "type_direct_score": (
                type_score_sum / safe_valid_counts
            ).detach().cpu().tolist(),
            "type_direct_target": (
                type_target_sum / safe_valid_counts
            ).detach().cpu().tolist(),
            "type_reference_scale": (
                type_reference_scale_sum / safe_valid_counts
            ).detach().cpu().tolist(),
            "type_normalized_gain": (
                type_normalized_gain_sum / safe_valid_counts
            ).detach().cpu().tolist(),
            "marginal_delta_rms": float(
                type_delta_rms_sum.sum().item() / max(valid_count, 1)
            ),
        }
        return augmented, direct_node_mask, propagated_node_mask, anomaly_type_ids

    def _direct_quantile_candidate_status(
        self,
        candidate_score,
        perturbation_rms,
        boundary,
    ):
        (
            candidate_value,
            rms_value,
            original_value,
            target_value,
            cap_value,
            reference_scale,
        ) = torch.stack(
            [
                candidate_score.reshape(()),
                perturbation_rms.reshape(()),
                boundary["original_score"].reshape(()),
                boundary["target"].reshape(()),
                boundary["cap"].reshape(()),
                boundary["reference_scale"].reshape(()),
            ]
        ).detach().cpu().tolist()
        finite = all(
            math.isfinite(value)
            for value in (
                candidate_value,
                rms_value,
                original_value,
                target_value,
                cap_value,
                reference_scale,
            )
        )
        plausible = (
            finite
            and reference_scale > 0.0
            and rms_value + 1e-6 >= self.hybrid_fallback_min_rms
            and rms_value <= self.hybrid_fallback_max_rms + 1e-6
        )
        gain_floor = (
            original_value
            + self.direct_quantile_min_reference_gain * reference_scale
        )
        lower = max(target_value, gain_floor)
        below_gain_floor = plausible and candidate_value < gain_floor
        above_cap = plausible and candidate_value > cap_value
        hard_eligible = (
            plausible
            and candidate_value >= lower
            and candidate_value <= cap_value
        )
        soft_eligible = (
            plausible
            and candidate_value >= gain_floor
            and candidate_value < lower
            and candidate_value <= cap_value
        )
        rms_rescue_eligible = (
            self.direct_quantile_upper_bound_mode == "rms_rescue"
            and plausible
            and candidate_value > cap_value
            and candidate_value >= gain_floor
        )

        denominator = max(lower - gain_floor, 1e-12)
        achievement = min(
            max((candidate_value - gain_floor) / denominator, 0.0),
            1.0,
        ) if finite else 0.0
        accepted = hard_eligible or rms_rescue_eligible or (
            self.direct_quantile_acceptance_mode != "hard" and soft_eligible
        )
        if hard_eligible:
            acceptance_tier = self.ACCEPTANCE_HARD
            sample_weight = 1.0
        elif accepted:
            acceptance_tier = self.ACCEPTANCE_SOFT
            if rms_rescue_eligible:
                sample_weight = self.direct_quantile_rms_rescue_weight
            elif self.direct_quantile_acceptance_mode == "soft_unweighted":
                sample_weight = 1.0
            else:
                sample_weight = self.direct_quantile_soft_min_weight + (
                    1.0 - self.direct_quantile_soft_min_weight
                ) * achievement ** self.direct_quantile_soft_power
        else:
            acceptance_tier = self.ACCEPTANCE_INVALID
            sample_weight = 0.0

        return {
            "finite": finite,
            "plausible": plausible,
            "below_gain_floor": below_gain_floor,
            "above_cap": above_cap,
            "hard_eligible": hard_eligible,
            "soft_eligible": soft_eligible,
            "rms_rescue_eligible": rms_rescue_eligible,
            "accepted": accepted,
            "achievement": achievement,
            "sample_weight": sample_weight,
            "acceptance_tier": acceptance_tier,
            "gain_floor": gain_floor,
            "lower": lower,
        }

    def _gm_hns_candidate_metadata(self, candidate_score, perturbation_rms, boundary):
        """Score a valid candidate inside the normal tail boundary.

        ``difficulty`` is the candidate's relative position between the
        scheduled boundary target and the normal-support cap.  Values close
        to zero are just over the acceptance floor, while values close to one
        are saturated/easy negatives.  GM-HNS keeps the middle band and uses
        a small RMS tie-breaker to prefer local edits.
        """
        width = (boundary["cap"] - boundary["target"]).clamp_min(1e-6)
        difficulty = (candidate_score - boundary["target"]) / width
        finite = bool(torch.isfinite(difficulty).item()) and bool(
            torch.isfinite(perturbation_rms).item()
        )
        if not finite:
            return {
                "difficulty": candidate_score.new_tensor(float("nan")),
                "in_band": False,
                "penalty": candidate_score.new_tensor(float("inf")),
            }
        difficulty_value = float(difficulty.detach().item())
        in_band = (
            self.gm_hns_difficulty_low - 1e-8
            <= difficulty_value
            <= self.gm_hns_difficulty_high + 1e-8
        )
        distance = abs(difficulty_value - self.gm_hns_difficulty_target)
        outside = max(self.gm_hns_difficulty_low - difficulty_value, 0.0)
        outside += max(difficulty_value - self.gm_hns_difficulty_high, 0.0)
        penalty = outside + distance + self.gm_hns_rms_weight * max(
            float(perturbation_rms.detach().item()),
            0.0,
        )
        return {
            "difficulty": difficulty.detach(),
            "in_band": bool(in_band),
            "penalty": candidate_score.new_tensor(penalty),
        }

    def _gm_hns_candidate_is_better(self, candidate, incumbent):
        """Prefer medium-hard candidates, with deterministic legacy fallback."""
        if incumbent is None:
            return True
        candidate_in_band = bool(candidate.get("gm_hns_in_band", False))
        incumbent_in_band = bool(incumbent.get("gm_hns_in_band", False))
        if candidate_in_band != incumbent_in_band:
            return candidate_in_band
        candidate_penalty = candidate.get("gm_hns_penalty")
        incumbent_penalty = incumbent.get("gm_hns_penalty")
        if candidate_penalty is not None and incumbent_penalty is not None:
            if bool((candidate_penalty < incumbent_penalty).item()):
                return True
            if bool((candidate_penalty > incumbent_penalty).item()):
                return False
        return bool(candidate["penalty"] < incumbent["penalty"])

    def _direct_quantile_boundary(
        self,
        windows,
        adjacency,
        excluded_batch,
        seed,
        neighbors,
        start,
        end,
        anomaly_type,
        target_quantile,
        self_temporal=False,
    ):
        self._last_direct_boundary_reason = "unknown"
        if self.normal_mechanism is not None:
            if excluded_batch < 0 or excluded_batch >= windows.size(0):
                self._last_direct_boundary_reason = "insufficient_reference"
                return None
            if self.normal_mechanism is None or not self.normal_mechanism.fitted:
                self._last_direct_boundary_reason = "context_invalid"
                return None
            boundary, reason = self.normal_mechanism.boundary(
                windows[excluded_batch:excluded_batch + 1],
                seed,
                start,
                end,
                target_quantile,
                self.direct_quantile_cap,
                self_temporal=self_temporal,
            )
            self._last_direct_boundary_reason = reason
            return boundary
        scores = self._direct_mechanism_scores(
            windows,
            adjacency,
            seed,
            neighbors,
            start,
            end,
            anomaly_type,
        )
        context_valid = self._last_mechanism_details.get("context_valid")
        if excluded_batch < 0 or excluded_batch >= scores.numel():
            self._last_direct_boundary_reason = "insufficient_reference"
            return None
        if (
            context_valid is not None
            and not bool(context_valid[excluded_batch].item())
        ):
            self._last_direct_boundary_reason = "context_invalid"
            return None
        original_score = scores[excluded_batch]
        reference_mask = torch.isfinite(scores)
        reference_mask[excluded_batch] = False
        reference_scores = scores[reference_mask]
        if reference_scores.numel() < 4:
            self._last_direct_boundary_reason = "insufficient_reference"
            return None
        if not torch.isfinite(original_score):
            self._last_direct_boundary_reason = "nonfinite"
            return None
        target = torch.quantile(reference_scores, target_quantile)
        empirical_cap = torch.quantile(reference_scores, self.direct_quantile_cap)
        q25 = torch.quantile(reference_scores, 0.25)
        q75 = torch.quantile(reference_scores, 0.75)
        iqr_scale = (q75 - q25) / 1.349
        std_scale = reference_scores.std(unbiased=False)
        if (
            self.direct_quantile_reference_scale_mode == "iqr_std_fallback"
            and iqr_scale <= 1e-4
        ):
            reference_scale = std_scale.clamp_min(1e-4)
            reference_scale_fallback = True
        else:
            reference_scale = iqr_scale.clamp_min(1e-4)
            reference_scale_fallback = False
        cap = empirical_cap
        if not all(
            bool(torch.isfinite(value))
            for value in (target, empirical_cap, reference_scale, cap)
        ):
            self._last_direct_boundary_reason = "nonfinite"
            return None
        if (
            cap <= target
            and self.direct_quantile_upper_bound_mode == "mechanism_cap"
        ):
            self._last_direct_boundary_reason = "collapsed_tail"
            return None
        self._last_direct_boundary_reason = "valid"
        return {
            "original_score": original_score.detach(),
            "target": target.detach(),
            "cap": cap.detach(),
            "empirical_cap": empirical_cap.detach(),
            "reference_scale": reference_scale.detach(),
            "reference_scale_fallback": reference_scale_fallback,
        }

    def _direct_mechanism_scores(
        self,
        windows,
        adjacency,
        seed,
        neighbors,
        start,
        end,
        anomaly_type,
        normalization_windows=None,
    ):
        if self.normal_mechanism is not None:
            if self.normal_mechanism is None:
                scores = windows.new_full((windows.size(0),), float("nan"))
                details = {
                    "score": scores,
                    "base_score": scores,
                    "context_valid": torch.zeros_like(scores, dtype=torch.bool),
                    "orientation": torch.zeros_like(scores),
                }
            else:
                scores, details = self.normal_mechanism.score_windows(
                    windows,
                    seed,
                    start,
                    end,
                )
            self._last_mechanism_details = details
            return scores
        if self.direct_mechanism_score_mode == "legacy_shared":
            scores = self._legacy_direct_mechanism_scores(
                windows,
                adjacency,
                seed,
                neighbors,
                start,
                end,
                anomaly_type,
                normalization_windows=normalization_windows,
            )
            self._last_mechanism_details = {
                "score": scores,
                "base_score": scores,
                "context_valid": torch.isfinite(scores),
                "orientation": torch.zeros_like(scores),
            }
            return scores

        details = self._type_specific_mechanism_score_components(
            windows,
            adjacency,
            seed,
            neighbors,
            start,
            end,
            anomaly_type,
            normalization_windows=normalization_windows,
        )
        self._last_mechanism_details = details
        return details["score"]

    def _legacy_direct_mechanism_scores(
        self,
        windows,
        adjacency,
        seed,
        neighbors,
        start,
        end,
        anomaly_type,
        normalization_windows=None,
    ):
        normalization_windows = (
            windows if normalization_windows is None else normalization_windows
        )
        seed_values = windows[:, :, seed].float()
        seed_reference = normalization_windows[:, :, seed].float()
        seed_mean = seed_reference.mean(dim=1, keepdim=True)
        seed_scale = seed_reference.std(
            dim=1,
            unbiased=False,
            keepdim=True,
        ).clamp_min(1e-4)
        seed_z = (seed_values - seed_mean) / seed_scale

        if anomaly_type in ("neighbor_break", "relation_flip"):
            valid_neighbors = neighbors.unique()
            valid_neighbors = valid_neighbors[valid_neighbors != seed]
            if valid_neighbors.numel() == 0:
                return seed_values.new_full((seed_values.size(0),), float("nan"))
            neighbor_values = normalization_windows.index_select(
                dim=2,
                index=valid_neighbors,
            ).float()
            neighbor_mean = neighbor_values.mean(dim=1, keepdim=True)
            neighbor_scale = neighbor_values.std(
                dim=1,
                unbiased=False,
                keepdim=True,
            ).clamp_min(1e-4)
            neighbor_z = (neighbor_values - neighbor_mean) / neighbor_scale
            weights = torch.softmax(
                adjacency[seed].index_select(0, valid_neighbors).float(),
                dim=0,
            )
            relation_reference = (
                neighbor_z[:, start:end] * weights.view(1, 1, -1)
            ).sum(dim=2)
            residual = seed_z[:, start:end] - relation_reference
        else:
            target_start = max(start, 1)
            if end <= target_start:
                return seed_values.new_full((seed_values.size(0),), float("nan"))
            residual = (
                seed_z[:, target_start:end]
                - seed_z[:, target_start - 1:end - 1]
            )
        return residual.square().mean(dim=1).sqrt()

    def _type_specific_mechanism_score_components(
        self,
        windows,
        adjacency,
        seed,
        neighbors,
        start,
        end,
        anomaly_type,
        normalization_windows=None,
    ):
        reference = windows if normalization_windows is None else normalization_windows
        values = windows[:, :, seed].float()
        reference_values = reference[:, :, seed].float()
        reference_mean = reference_values.mean(dim=1, keepdim=True)
        reference_scale = reference_values.std(
            dim=1,
            unbiased=False,
            keepdim=True,
        ).clamp_min(1e-4)
        values_z = (values - reference_mean) / reference_scale
        reference_z = (reference_values - reference_mean) / reference_scale

        if anomaly_type == "lag_break":
            return self._lag_mechanism_score(values_z, reference_z, start, end)
        if anomaly_type in ("neighbor_break", "relation_flip"):
            return self._relation_mechanism_score(
                values_z,
                reference_z,
                reference,
                adjacency,
                seed,
                neighbors,
                start,
                end,
                include_orientation=anomaly_type == "relation_flip",
            )
        if anomaly_type == "local_function_shift":
            return self._local_function_mechanism_score(
                values_z,
                reference_z,
                start,
                end,
            )
        raise ValueError(f"Unsupported direct mechanism type: {anomaly_type}")

    def _lag_mechanism_score(self, values_z, reference_z, start, end):
        time_count = reference_z.size(1)
        transition_times = torch.arange(1, time_count, device=reference_z.device)
        context_mask = (transition_times < start) | (transition_times > end)
        eval_mask = (transition_times >= max(start, 1)) & (
            transition_times < min(end + 1, time_count)
        )
        slope, intercept, fit_valid = self._fit_ridge_affine(
            reference_z[:, :-1],
            reference_z[:, 1:],
            context_mask,
            require_x_variance=True,
        )
        residual = values_z[:, 1:] - (
            slope.unsqueeze(1) * values_z[:, :-1] + intercept.unsqueeze(1)
        )
        score = self._masked_rms(residual, eval_mask, fit_valid)
        return {
            "score": score,
            "base_score": score,
            "context_valid": fit_valid,
            "orientation": torch.zeros_like(score),
        }

    def _relation_mechanism_score(
        self,
        values_z,
        reference_z,
        normalization_windows,
        adjacency,
        seed,
        neighbors,
        start,
        end,
        include_orientation,
    ):
        valid_neighbors = neighbors.unique()
        valid_neighbors = valid_neighbors[valid_neighbors != seed]
        batch_size = values_z.size(0)
        invalid_score = values_z.new_full((batch_size,), float("nan"))
        if valid_neighbors.numel() == 0:
            return {
                "score": invalid_score,
                "base_score": invalid_score.clone(),
                "context_valid": torch.zeros(
                    batch_size,
                    device=values_z.device,
                    dtype=torch.bool,
                ),
                "orientation": torch.zeros_like(invalid_score),
            }

        neighbor_values = normalization_windows.index_select(
            dim=2,
            index=valid_neighbors,
        ).float()
        neighbor_mean = neighbor_values.mean(dim=1, keepdim=True)
        neighbor_scale = neighbor_values.std(
            dim=1,
            unbiased=False,
            keepdim=True,
        ).clamp_min(1e-4)
        neighbor_z = (neighbor_values - neighbor_mean) / neighbor_scale
        weights = adjacency[seed].index_select(0, valid_neighbors).float()
        weights = weights.clamp_min(0.0)
        if not torch.isfinite(weights).all() or float(weights.sum()) <= 1e-12:
            weights = torch.ones_like(weights)
        weights = weights / weights.sum().clamp_min(1e-12)
        relation_reference = (
            neighbor_z * weights.view(1, 1, -1)
        ).sum(dim=2)

        time_count = reference_z.size(1)
        time_indices = torch.arange(time_count, device=reference_z.device)
        context_mask = (time_indices < start) | (time_indices >= end)
        eval_mask = (time_indices >= start) & (time_indices < end)
        slope, intercept, fit_valid = self._fit_ridge_affine(
            relation_reference,
            reference_z,
            context_mask,
            require_x_variance=True,
        )
        prediction = (
            slope.unsqueeze(1) * relation_reference + intercept.unsqueeze(1)
        )
        residual = values_z - prediction
        original_residual = reference_z - prediction
        base_score = self._masked_rms(residual, eval_mask, fit_valid)
        orientation = torch.zeros_like(base_score)
        if include_orientation and bool(eval_mask.any()):
            evaluated = residual[:, eval_mask]
            original = original_residual[:, eval_mask]
            dot = (evaluated * original).sum(dim=1)
            evaluated_norm = evaluated.square().sum(dim=1).sqrt()
            original_norm = original.square().sum(dim=1).sqrt()
            orientation_valid = (
                fit_valid
                & (evaluated_norm > 1e-8)
                & (original_norm > 1e-8)
            )
            cosine = dot / (evaluated_norm * original_norm).clamp_min(1e-12)
            orientation = torch.where(
                orientation_valid,
                0.5 * (1.0 - cosine.clamp(min=-1.0, max=1.0)),
                torch.zeros_like(cosine),
            )
        score = base_score * (
            1.0 + self.direct_relation_flip_orientation_weight * orientation
        )
        score = torch.where(fit_valid, score, invalid_score)
        return {
            "score": score,
            "base_score": base_score,
            "context_valid": fit_valid,
            "orientation": orientation,
        }

    def _local_function_mechanism_score(self, values_z, reference_z, start, end):
        time_count = reference_z.size(1)
        time_values = torch.linspace(
            -1.0,
            1.0,
            steps=time_count,
            device=reference_z.device,
            dtype=reference_z.dtype,
        ).unsqueeze(0).expand(reference_z.size(0), -1)
        time_indices = torch.arange(time_count, device=reference_z.device)
        context_mask = (time_indices < start) | (time_indices >= end)
        eval_mask = (time_indices >= start) & (time_indices < end)
        slope, intercept, fit_valid = self._fit_ridge_affine(
            time_values,
            reference_z,
            context_mask,
            require_x_variance=True,
        )
        prediction = slope.unsqueeze(1) * time_values + intercept.unsqueeze(1)
        residual = values_z - prediction
        score = self._masked_rms(residual, eval_mask, fit_valid)
        return {
            "score": score,
            "base_score": score,
            "context_valid": fit_valid,
            "orientation": torch.zeros_like(score),
        }

    def _fit_ridge_affine(self, x, y, mask, require_x_variance):
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(x.size(0), -1)
        mask_float = mask.to(dtype=x.dtype)
        count = mask_float.sum(dim=1)
        safe_count = count.clamp_min(1.0)
        x_mean = (x * mask_float).sum(dim=1) / safe_count
        y_mean = (y * mask_float).sum(dim=1) / safe_count
        centered_x = x - x_mean.unsqueeze(1)
        centered_y = y - y_mean.unsqueeze(1)
        variance = (centered_x.square() * mask_float).sum(dim=1) / safe_count
        covariance = (
            centered_x * centered_y * mask_float
        ).sum(dim=1) / safe_count
        slope = covariance / (variance + self.direct_mechanism_ridge)
        intercept = y_mean - slope * x_mean
        valid = (
            (count >= self.direct_mechanism_min_context)
            & torch.isfinite(slope)
            & torch.isfinite(intercept)
        )
        if require_x_variance:
            valid = valid & (variance > 1e-8)
        return slope, intercept, valid

    @staticmethod
    def _masked_rms(values, mask, valid):
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(values.size(0), -1)
        mask_float = mask.to(dtype=values.dtype)
        count = mask_float.sum(dim=1)
        score = (
            (values.square() * mask_float).sum(dim=1) / count.clamp_min(1.0)
        ).sqrt()
        valid = valid & (count > 0) & torch.isfinite(score)
        return torch.where(
            valid,
            score,
            score.new_full(score.shape, float("nan")),
        )

    def _direct_candidate_support(self, windows, seed):
        if self.direct_candidate_support_mode == "none":
            return {
                "low": windows.new_tensor(float("-inf")),
                "high": windows.new_tensor(float("inf")),
                "collapsed": False,
            }
        if self.direct_candidate_support_mode in (
            "global_per_node_quantile",
            "adaptive_empirical",
        ):
            if self.normal_mechanism is None:
                return {
                    "low": windows.new_tensor(float("nan")),
                    "high": windows.new_tensor(float("nan")),
                    "collapsed": True,
                }
            return self.normal_mechanism.support(
                seed,
                windows.device,
                windows.dtype,
            )
        values = windows[:, :, seed].float().reshape(-1)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return {
                "low": windows.new_tensor(float("nan")),
                "high": windows.new_tensor(float("nan")),
                "collapsed": True,
            }
        low = torch.quantile(
            values,
            self.direct_candidate_support_low_quantile,
        )
        high = torch.quantile(
            values,
            self.direct_candidate_support_high_quantile,
        )
        scale = values.std(unbiased=False).clamp_min(1e-4)
        finite_bounds = bool(torch.isfinite(low).item()) and bool(
            torch.isfinite(high).item()
        )
        collapsed = not finite_bounds or bool(
            (high - low <= 1e-4 * scale).item()
        )
        return {
            "low": low.to(dtype=windows.dtype),
            "high": high.to(dtype=windows.dtype),
            "collapsed": collapsed,
        }

    def _project_to_direct_support(self, values, support):
        if self.direct_candidate_support_mode == "none":
            return values, False
        states = support.get("states")
        if states is not None and states.numel() > 0:
            distances = (values.unsqueeze(-1) - states).abs()
            projected = states[distances.argmin(dim=-1)]
            outside = (values < support["low"]) | (values > support["high"])
            return projected, bool(outside.any().item())
        projected = values.clamp(min=support["low"], max=support["high"])
        clipped = bool(((projected - values).abs() > 1e-8).any().item())
        return projected, clipped

    def _direct_support_violation(self, values, support):
        if self.direct_candidate_support_mode == "none":
            return False
        states = support.get("states")
        if states is not None and states.numel() > 0:
            nearest = (values.unsqueeze(-1) - states).abs().min(dim=-1).values
            if bool((nearest > 1e-6).any().item()):
                return True
        tolerance = 1e-6 * (
            support["high"] - support["low"]
        ).abs().clamp_min(1.0)
        return bool(
            (
                (values < support["low"] - tolerance)
                | (values > support["high"] + tolerance)
                | ~torch.isfinite(values)
            ).any().item()
        )

    def _apply_direct_edit_candidate(
        self,
        augmented,
        windows,
        adjacency,
        batch_idx,
        seed,
        neighbors,
        start,
        end,
        anomaly_type,
        progress,
    ):
        strength = self._scheduled_edit_strength(progress)
        if self.normal_mechanism is not None:
            if self.candidate_generator != "graph_var":
                candidates, _ = self._normal_candidate_segments(windows[batch_idx], seed, start, end,
                    anomaly_type, windows.new_tensor([strength]), phase=batch_idx + seed + start)
                values = candidates[0] if candidates else None
            else:
                values = self.normal_mechanism.counterfactual(
                    windows[batch_idx], seed, start, end, anomaly_type, strength)
            if values is not None:
                augmented[batch_idx, start:end, seed] = values
            return
        if anomaly_type == "lag_break":
            self._lag_break(augmented, batch_idx, seed, start, end, strength)
        elif anomaly_type == "neighbor_break":
            self._neighbor_break(
                augmented,
                windows,
                adjacency,
                batch_idx,
                seed,
                neighbors,
                start,
                end,
                strength,
            )
        elif anomaly_type == "relation_flip":
            self._relation_flip(
                augmented,
                adjacency,
                batch_idx,
                seed,
                neighbors,
                start,
                end,
                strength,
            )
        elif anomaly_type == "local_function_shift":
            self._local_function_shift(
                augmented,
                batch_idx,
                seed,
                start,
                end,
                self._scheduled_magnitude(progress),
                strength,
            )
        else:
            raise ValueError(f"Unsupported direct quantile type: {anomaly_type}")

    def _scheduled_direct_quantile(self, progress):
        if not self.strength_schedule:
            return self.direct_quantile_end
        progress = min(max(float(progress), 0.0), 1.0)
        return float(
            self.direct_quantile_start
            + progress * (self.direct_quantile_end - self.direct_quantile_start)
        )

    def _surrogate_boundary_view(
        self,
        windows,
        adjacency,
        node_evidence=None,
        progress=1.0,
        collect_diagnostics=True,
    ):
        augmented = windows.clone()
        batch_size, win_size, n_var = augmented.shape
        direct_node_mask = augmented.new_zeros(batch_size, n_var)
        propagated_node_mask = augmented.new_zeros(batch_size, n_var)
        anomaly_type_ids = torch.zeros(batch_size, device=windows.device, dtype=torch.long)
        valid_mask = torch.zeros(batch_size, device=windows.device, dtype=torch.bool)
        edit_source_ids = torch.zeros(batch_size, device=windows.device, dtype=torch.long)
        fallback_type_ids = torch.full(
            (batch_size,),
            -1,
            device=windows.device,
            dtype=torch.long,
        )
        seed_ids = torch.full((batch_size,), -1, device=windows.device, dtype=torch.long)
        surrogate_fit_cache = {}
        type_counts = augmented.new_zeros(len(self.synthetic_types))
        type_valid_counts = augmented.new_zeros(len(self.synthetic_types))
        type_delta_rms_sum = augmented.new_zeros(len(self.synthetic_types))
        type_boundary_score_sum = augmented.new_zeros(len(self.synthetic_types))
        type_boundary_valid_count = augmented.new_zeros(len(self.synthetic_types))
        type_boundary_accept_sum = augmented.new_zeros(len(self.synthetic_types))
        selected_evidence = []
        evidence_selected = 0
        random_selected = 0
        boundary_target_sum = 0.0
        boundary_target_count = 0
        boundary_original_sum = 0.0
        boundary_original_count = 0
        marginal_delta_sum = 0.0
        surrogate_r2_sum = 0.0
        surrogate_r2_count = 0
        selected_contribution_sum = 0.0
        selected_contribution_count = 0
        surrogate_delta_r2_sum = 0.0
        surrogate_delta_r2_count = 0
        candidate_fit_count = 0
        fallback_count = 0
        fallback_attempt_count = 0
        fallback_rejected_count = 0
        fallback_mechanism_gain_sum = 0.0
        fallback_relation_gain_sum = 0.0
        fallback_temporal_gain_sum = 0.0
        invalid_count = 0
        fitted_window_count = 0
        surrogate_accepted_count = 0
        fallback_type_counts = {name: 0 for name in self.hybrid_fallback_types}
        fallback_type_mechanism_gain_sums = {name: 0.0 for name in self.hybrid_fallback_types}
        attempted_seed_counts = augmented.new_zeros(n_var)
        valid_seed_counts = augmented.new_zeros(n_var)
        seed_trials = 1 if self.edit_mode == "hybrid_valid" else self.surrogate_seed_trials

        for batch_idx in range(batch_size):
            evidence_row = None if node_evidence is None else node_evidence[batch_idx]
            anomaly_type_id = random.randrange(len(self.synthetic_types))
            anomaly_type = self.synthetic_types[anomaly_type_id]
            candidate = None
            best_unaccepted = None
            start = end = 0
            seed = 0
            neighbors = torch.empty(0, device=windows.device, dtype=torch.long)
            used_evidence = False
            sample_fitted = False

            for seed_trial in range(seed_trials):
                if seed_trial == 0:
                    seed, neighbors, used_evidence = self._sample_seed_and_neighbors(
                        adjacency[batch_idx],
                        n_var,
                        self.surrogate_parent_candidates,
                        evidence_row,
                    )
                else:
                    seed = random.randrange(n_var)
                    neighbors = self._graph_neighbor_candidates(
                        adjacency[batch_idx],
                        seed,
                        n_var,
                    )
                    used_evidence = False
                attempted_seed_counts[seed] += 1.0

                for _ in range(self.surrogate_segment_trials):
                    start, end = self._sample_surrogate_segment(win_size)
                    trial = self._build_surrogate_candidate(
                        windows,
                        batch_idx,
                        adjacency[batch_idx],
                        seed,
                        neighbors,
                        start,
                        end,
                        anomaly_type,
                        progress,
                        fit_cache=surrogate_fit_cache,
                    )
                    if surrogate_fit_cache.get((batch_idx, seed)) is not None:
                        sample_fitted = True
                    if trial is None:
                        continue
                    candidate_fit_count += 1
                    if best_unaccepted is None or trial["penalty"] < best_unaccepted["penalty"]:
                        best_unaccepted = {
                            **trial,
                            "seed": seed,
                            "start": start,
                            "end": end,
                            "used_evidence": used_evidence,
                        }
                    if trial["accepted"]:
                        candidate = {
                            **trial,
                            "seed": seed,
                            "start": start,
                            "end": end,
                            "used_evidence": used_evidence,
                        }
                        break
                if candidate is not None:
                    break

            if sample_fitted:
                fitted_window_count += 1

            if candidate is None and self.edit_mode == "hybrid_valid":
                fallback_attempt_count += 1
                fallback = self._apply_hybrid_fallback(
                    augmented,
                    windows,
                    adjacency[batch_idx],
                    batch_idx,
                    seed,
                    neighbors,
                    start,
                    end,
                    progress,
                )
                if fallback is not None:
                    fallback_type = fallback["type"]
                    fallback_rms = fallback["rms"]
                    fallback_mechanism_gain = fallback["mechanism_gain"]
                    fallback_type_counts[fallback_type] += 1
                    fallback_type_mechanism_gain_sums[fallback_type] += float(
                        fallback_mechanism_gain.item()
                    )
                    fallback_count += 1
                    fallback_mechanism_gain_sum += float(fallback_mechanism_gain.item())
                    fallback_relation_gain_sum += float(fallback["relation_gain"].item())
                    fallback_temporal_gain_sum += float(fallback["temporal_gain"].item())
                    direct_node_mask[batch_idx, seed] = 1.0
                    valid_mask[batch_idx] = True
                    edit_source_ids[batch_idx] = self.EDIT_SOURCE_FALLBACK
                    fallback_type_ids[batch_idx] = self.hybrid_fallback_types.index(
                        fallback_type
                    )
                    seed_ids[batch_idx] = seed
                    valid_seed_counts[seed] += 1.0
                    anomaly_type_ids[batch_idx] = anomaly_type_id
                    type_counts[anomaly_type_id] += 1.0
                    type_valid_counts[anomaly_type_id] += 1.0
                    type_delta_rms_sum[anomaly_type_id] += fallback_rms
                    marginal_delta_sum += fallback_rms.item()
                    if used_evidence:
                        evidence_selected += 1
                    else:
                        random_selected += 1
                    if collect_diagnostics and evidence_row is not None:
                        selected_evidence.append(evidence_row[seed].detach())
                    continue
                fallback_rejected_count += 1

            if candidate is None and not self.require_accepted:
                candidate = best_unaccepted
                if candidate is not None:
                    fallback_count += 1
                else:
                    fallback = self._fallback_surrogate_candidate(
                        windows[batch_idx],
                        adjacency[batch_idx],
                        seed,
                        neighbors,
                        start,
                        end,
                    )
                    candidate = {
                        **fallback,
                        "seed": seed,
                        "start": start,
                        "end": end,
                        "used_evidence": used_evidence,
                    }
                    fallback_count += 1
            if candidate is None:
                invalid_count += 1
                anomaly_type_ids[batch_idx] = anomaly_type_id
                type_counts[anomaly_type_id] += 1.0
                continue

            seed = candidate.pop("seed")
            start = candidate.pop("start")
            end = candidate.pop("end")
            used_evidence = candidate.pop("used_evidence")

            augmented[batch_idx, start:end, seed] = candidate["values"]
            direct_node_mask[batch_idx, seed] = 1.0
            valid_mask[batch_idx] = True
            edit_source_ids[batch_idx] = (
                self.EDIT_SOURCE_SURROGATE
                if bool(candidate["accepted"].item())
                else self.EDIT_SOURCE_FALLBACK
            )
            seed_ids[batch_idx] = seed
            valid_seed_counts[seed] += 1.0
            if bool(candidate["accepted"].item()):
                surrogate_accepted_count += 1
            anomaly_type_ids[batch_idx] = anomaly_type_id
            type_counts[anomaly_type_id] += 1.0
            type_valid_counts[anomaly_type_id] += 1.0

            if used_evidence:
                evidence_selected += 1
            else:
                random_selected += 1
            if collect_diagnostics:
                type_delta_rms_sum[anomaly_type_id] += candidate["delta_rms"]
                if torch.isfinite(candidate["boundary_score"]):
                    type_boundary_score_sum[anomaly_type_id] += candidate["boundary_score"]
                    type_boundary_valid_count[anomaly_type_id] += 1.0
                type_boundary_accept_sum[anomaly_type_id] += candidate["accepted"].float()
                if evidence_row is not None:
                    selected_evidence.append(evidence_row[seed].detach())
                if torch.isfinite(candidate["boundary_target"]):
                    boundary_target_sum += candidate["boundary_target"].item()
                    boundary_target_count += 1
                if torch.isfinite(candidate["original_score"]):
                    boundary_original_sum += candidate["original_score"].item()
                    boundary_original_count += 1
                marginal_delta_sum += candidate["delta_rms"].item()
                if torch.isfinite(candidate["surrogate_r2"]):
                    surrogate_r2_sum += candidate["surrogate_r2"].item()
                    surrogate_r2_count += 1
                if torch.isfinite(candidate["selected_contribution"]):
                    selected_contribution_sum += candidate["selected_contribution"].item()
                    selected_contribution_count += 1
                if torch.isfinite(candidate["surrogate_delta_r2"]):
                    surrogate_delta_r2_sum += candidate["surrogate_delta_r2"].item()
                    surrogate_delta_r2_count += 1

        self.last_valid_mask = valid_mask
        self.last_sample_weights = valid_mask.to(dtype=windows.dtype)
        self.last_acceptance_tier = torch.where(
            valid_mask,
            torch.full_like(valid_mask, self.ACCEPTANCE_HARD, dtype=torch.long),
            torch.full_like(valid_mask, self.ACCEPTANCE_INVALID, dtype=torch.long),
        )
        self.last_edit_source_ids = edit_source_ids
        self.last_fallback_type_ids = fallback_type_ids
        self.last_seed_ids = seed_ids
        coverage_diagnostics = {
            "attempted_count": batch_size,
            "valid_count": int(valid_mask.sum().item()),
            "fitted_window_count": fitted_window_count,
            "surrogate_accepted_count": surrogate_accepted_count,
            "fallback_count": fallback_count,
            "fallback_attempt_count": fallback_attempt_count,
            "fallback_rejected_count": fallback_rejected_count,
            "invalid_count": invalid_count,
            "attempted_seed_counts": attempted_seed_counts.detach().cpu().tolist(),
            "valid_seed_counts": valid_seed_counts.detach().cpu().tolist(),
            "fallback_type_counts": dict(fallback_type_counts),
            "fallback_type_mechanism_gain_sums": dict(fallback_type_mechanism_gain_sums),
            "fallback_mechanism_gain_sum": fallback_mechanism_gain_sum,
            "fallback_relation_gain_sum": fallback_relation_gain_sum,
            "fallback_temporal_gain_sum": fallback_temporal_gain_sum,
        }
        self.last_coverage_diagnostics = coverage_diagnostics

        if not collect_diagnostics:
            self.last_diagnostics = {}
            return augmented, direct_node_mask, propagated_node_mask, anomaly_type_ids

        seed_evidence_mean = (
            torch.stack(selected_evidence).mean().item() if selected_evidence else float("nan")
        )
        safe_counts = type_counts.clamp_min(1.0)
        safe_valid_counts = type_valid_counts.clamp_min(1.0)
        safe_boundary_counts = type_boundary_valid_count.clamp_min(1.0)
        accepted_count = int(valid_mask.sum().item())
        seed_coverage = float((valid_seed_counts > 0).float().mean().item())
        seed_entropy = self._normalized_count_entropy(valid_seed_counts)
        attempted_seed_entropy = self._normalized_count_entropy(attempted_seed_counts)
        type_boundary_score = type_boundary_score_sum / safe_boundary_counts
        type_boundary_score = torch.where(
            type_boundary_valid_count > 0,
            type_boundary_score,
            torch.full_like(type_boundary_score, float("nan")),
        )
        self.last_diagnostics = {
            "seed_evidence_mean": seed_evidence_mean,
            "evidence_sampling_ratio": evidence_selected / max(accepted_count, 1),
            "random_sampling_ratio": random_selected / max(accepted_count, 1),
            "edit_mode": self.edit_mode,
            "edit_strength": (
                float(self._scheduled_edit_strength(progress))
                if self.edit_mode == "hybrid_valid"
                else float("nan")
            ),
            "magnitude": float("nan"),
            "boundary_quantile": self._scheduled_boundary_quantile(progress),
            "boundary_target": (
                boundary_target_sum / boundary_target_count
                if boundary_target_count > 0
                else float("nan")
            ),
            "boundary_original_score": (
                boundary_original_sum / boundary_original_count
                if boundary_original_count > 0
                else float("nan")
            ),
            "boundary_acceptance_ratio": float(
                type_boundary_accept_sum.sum().item()
            ) / max(batch_size, 1),
            **coverage_diagnostics,
            "fitted_window_count": fitted_window_count,
            "surrogate_accepted_count": surrogate_accepted_count,
            "fitted_window_ratio": fitted_window_count / max(batch_size, 1),
            "surrogate_accepted_ratio": surrogate_accepted_count / max(batch_size, 1),
            "hybrid_fallback_ratio": fallback_count / max(batch_size, 1),
            "hybrid_fallback_validity_ratio": fallback_count / max(fallback_attempt_count, 1),
            "hybrid_fallback_mechanism_gain": (
                fallback_mechanism_gain_sum / fallback_count
                if fallback_count > 0
                else float("nan")
            ),
            "hybrid_fallback_relation_gain": (
                fallback_relation_gain_sum / fallback_count
                if fallback_count > 0
                else float("nan")
            ),
            "hybrid_fallback_temporal_gain": (
                fallback_temporal_gain_sum / fallback_count
                if fallback_count > 0
                else float("nan")
            ),
            "invalid_ratio": invalid_count / max(batch_size, 1),
            "seed_coverage": seed_coverage,
            "seed_entropy": seed_entropy,
            "attempted_seed_entropy": attempted_seed_entropy,
            "fallback_type_ratio": {
                name: count / max(batch_size, 1)
                for name, count in fallback_type_counts.items()
            },
            "candidate_fit_rate": candidate_fit_count / max(
                batch_size * seed_trials * self.surrogate_segment_trials,
                1,
            ),
            "marginal_delta_rms": (
                marginal_delta_sum / accepted_count if accepted_count > 0 else float("nan")
            ),
            "surrogate_r2": (
                surrogate_r2_sum / surrogate_r2_count
                if surrogate_r2_count > 0
                else float("nan")
            ),
            "surrogate_delta_r2": (
                surrogate_delta_r2_sum / surrogate_delta_r2_count
                if surrogate_delta_r2_count > 0
                else float("nan")
            ),
            "surrogate_fit_ratio": surrogate_r2_count / max(batch_size, 1),
            "selected_contribution": (
                selected_contribution_sum / selected_contribution_count
                if selected_contribution_count > 0
                else float("nan")
            ),
            "fallback_ratio": fallback_count / max(batch_size, 1),
            "type_ratio": (type_counts / max(batch_size, 1)).detach().cpu().tolist(),
            "type_delta_rms": (
                type_delta_rms_sum / safe_valid_counts
            ).detach().cpu().tolist(),
            "type_boundary_score": type_boundary_score.detach().cpu().tolist(),
            "type_boundary_acceptance": (type_boundary_accept_sum / safe_counts).detach().cpu().tolist(),
        }
        return augmented, direct_node_mask, propagated_node_mask, anomaly_type_ids

    def _apply_hybrid_fallback(
        self,
        augmented,
        windows,
        adjacency,
        batch_idx,
        seed,
        neighbors,
        start,
        end,
        progress,
    ):
        if end <= start:
            start, end = self._sample_surrogate_segment(windows.size(1))
        original = augmented[batch_idx, start:end, seed].clone()
        strength = self._scheduled_edit_strength(progress)
        channel_scale = self._channel_scale(windows[batch_idx, :, seed])
        fallback_types = list(self.hybrid_fallback_types)
        random.shuffle(fallback_types)
        best_candidate = None

        for fallback_type in fallback_types:
            augmented[batch_idx, start:end, seed] = original
            if fallback_type == "lag_break":
                self._lag_break(augmented, batch_idx, seed, start, end, strength)
            elif fallback_type == "neighbor_break":
                self._neighbor_break(
                    augmented,
                    windows,
                    adjacency,
                    batch_idx,
                    seed,
                    neighbors,
                    start,
                    end,
                    strength,
                )
            elif fallback_type == "relation_flip":
                self._relation_flip(
                    augmented,
                    adjacency,
                    batch_idx,
                    seed,
                    neighbors,
                    start,
                    end,
                    strength,
                )
            elif fallback_type == "local_function_shift":
                self._local_function_shift(
                    augmented,
                    batch_idx,
                    seed,
                    start,
                    end,
                    self._scheduled_magnitude(progress),
                    strength,
                )

            edited = augmented[batch_idx, start:end, seed]
            normalized_delta = (edited - original) / channel_scale
            delta_rms = normalized_delta.square().mean().sqrt()
            if not torch.isfinite(delta_rms) or delta_rms <= 1e-8:
                continue

            direction = normalized_delta / delta_rms
            if not self.hybrid_fallback_validity_enable:
                target_rms = delta_rms.clamp(
                    min=self.hybrid_fallback_min_rms,
                    max=self.hybrid_fallback_max_rms,
                )
                candidate_delta = direction * target_rms
                if self.max_delta_std > 0:
                    candidate_delta = candidate_delta.clamp(
                        min=-self.max_delta_std,
                        max=self.max_delta_std,
                    )
                candidate_values = original + candidate_delta * channel_scale
                gains = self._fallback_mechanism_gains(
                    windows,
                    adjacency,
                    batch_idx,
                    seed,
                    neighbors,
                    start,
                    end,
                    candidate_values,
                    fallback_type,
                )
                augmented[batch_idx, start:end, seed] = candidate_values
                return {
                    "type": fallback_type,
                    "rms": candidate_delta.square().mean().sqrt().detach(),
                    **gains,
                }

            target_levels = torch.linspace(
                self.hybrid_fallback_min_rms,
                self.hybrid_fallback_max_rms,
                steps=self.hybrid_fallback_candidates,
                device=windows.device,
                dtype=windows.dtype,
            )
            for target_rms in target_levels:
                candidate_delta = direction * target_rms
                if self.max_delta_std > 0:
                    candidate_delta = candidate_delta.clamp(
                        min=-self.max_delta_std,
                        max=self.max_delta_std,
                    )
                actual_rms = candidate_delta.square().mean().sqrt()
                if (
                    not torch.isfinite(actual_rms)
                    or actual_rms + 1e-8 < self.hybrid_fallback_min_rms
                ):
                    continue
                candidate_values = original + candidate_delta * channel_scale
                gains = self._fallback_mechanism_gains(
                    windows,
                    adjacency,
                    batch_idx,
                    seed,
                    neighbors,
                    start,
                    end,
                    candidate_values,
                    fallback_type,
                )
                mechanism_gain = gains["mechanism_gain"]
                if (
                    not torch.isfinite(mechanism_gain)
                    or mechanism_gain < self.hybrid_fallback_min_mechanism_gain
                    or mechanism_gain > self.hybrid_fallback_max_mechanism_gain
                ):
                    continue
                penalty = (
                    mechanism_gain - self.hybrid_fallback_min_mechanism_gain
                ).abs() + 0.01 * actual_rms
                if best_candidate is None or penalty < best_candidate["penalty"]:
                    best_candidate = {
                        "type": fallback_type,
                        "values": candidate_values.clone(),
                        "rms": actual_rms.detach(),
                        "penalty": penalty.detach(),
                        **gains,
                    }

        if best_candidate is not None:
            augmented[batch_idx, start:end, seed] = best_candidate.pop("values")
            best_candidate.pop("penalty")
            return best_candidate

        augmented[batch_idx, start:end, seed] = original
        return None

    def _fallback_mechanism_gains(
        self,
        windows,
        adjacency,
        batch_idx,
        seed,
        neighbors,
        start,
        end,
        candidate_values,
        fallback_type,
    ):
        original_series = windows[batch_idx, :, seed].float()
        candidate_series = original_series.clone()
        candidate_series[start:end] = candidate_values.float()
        channel_mean = original_series.mean()
        channel_scale = original_series.std(unbiased=False).clamp_min(1e-4)
        original_z = (original_series - channel_mean) / channel_scale
        candidate_z = (candidate_series - channel_mean) / channel_scale

        temporal_gain = self._temporal_mechanism_gain(
            original_z,
            candidate_z,
            start,
            end,
        )
        relation_gain = original_z.new_tensor(float("nan"))
        valid_neighbors = neighbors.unique()
        valid_neighbors = valid_neighbors[valid_neighbors != seed]
        if valid_neighbors.numel() > 0:
            neighbor_values = windows[batch_idx].index_select(
                dim=1,
                index=valid_neighbors,
            ).float()
            neighbor_mean = neighbor_values.mean(dim=0, keepdim=True)
            neighbor_scale = neighbor_values.std(
                dim=0,
                unbiased=False,
                keepdim=True,
            ).clamp_min(1e-4)
            neighbor_z = (neighbor_values - neighbor_mean) / neighbor_scale
            weights = torch.softmax(
                adjacency[seed].index_select(0, valid_neighbors).float(),
                dim=0,
            )
            relation_reference = (neighbor_z[start:end] * weights.unsqueeze(0)).sum(dim=1)
            original_relation = original_z[start:end] - relation_reference
            candidate_relation = candidate_z[start:end] - relation_reference
            relation_gain = (
                candidate_relation.square().mean().sqrt()
                - original_relation.square().mean().sqrt()
            )

        finite_gains = [gain for gain in (relation_gain, temporal_gain) if torch.isfinite(gain)]
        if not finite_gains:
            mechanism_gain = original_z.new_tensor(float("nan"))
        elif fallback_type in ("neighbor_break", "relation_flip") and torch.isfinite(
            relation_gain
        ):
            mechanism_gain = relation_gain
        else:
            mechanism_gain = torch.stack(finite_gains).max()
        return {
            "mechanism_gain": mechanism_gain.detach(),
            "relation_gain": relation_gain.detach(),
            "temporal_gain": temporal_gain.detach(),
        }

    @staticmethod
    def _temporal_mechanism_gain(original, candidate, start, end):
        time_indices = torch.arange(1, original.numel(), device=original.device)
        context_mask = (time_indices < start) | (time_indices >= end)
        context_x = original[:-1][context_mask]
        context_y = original[1:][context_mask]
        if context_x.numel() >= 3 and context_x.var(unbiased=False) > 1e-6:
            x_mean = context_x.mean()
            y_mean = context_y.mean()
            slope = (
                ((context_x - x_mean) * (context_y - y_mean)).mean()
                / context_x.var(unbiased=False).clamp_min(1e-6)
            )
            intercept = y_mean - slope * x_mean
            target_indices = time_indices[(time_indices >= start) & (time_indices < end)]
            original_residual = original[target_indices] - (
                intercept + slope * original[target_indices - 1]
            )
            candidate_residual = candidate[target_indices] - (
                intercept + slope * candidate[target_indices - 1]
            )
        else:
            target_start = max(start, 1)
            original_residual = original[target_start:end] - original[target_start - 1:end - 1]
            candidate_residual = candidate[target_start:end] - candidate[target_start - 1:end - 1]
        if original_residual.numel() == 0:
            return original.new_tensor(float("nan"))
        return (
            candidate_residual.square().mean().sqrt()
            - original_residual.square().mean().sqrt()
        )

    @staticmethod
    def _normalized_count_entropy(counts):
        active = counts[counts > 0].float()
        total_nodes = counts.numel()
        if active.numel() == 0 or total_nodes <= 1:
            return 0.0
        probs = active / active.sum().clamp_min(1e-12)
        entropy = -(probs * probs.log()).sum() / math.log(total_nodes)
        return float(entropy.item())

    def _graph_neighbor_candidates(self, adjacency, seed, n_var):
        row = adjacency[seed].clone()
        row[seed] = -1.0
        k = min(self.surrogate_parent_candidates, max(n_var - 1, 0))
        if k <= 0:
            return torch.empty(0, device=adjacency.device, dtype=torch.long)
        return torch.topk(row, k=k, dim=0).indices

    def _sample_surrogate_segment(self, win_size, event_scale=None):
        decision_rng = self._decision_rng or random
        available = max(win_size - self.surrogate_lag, 1)
        seg_len = self._sample_segment_length(
            win_size,
            decision_rng,
            available,
            event_scale=event_scale,
        )
        seg_len = max(1, seg_len)
        start_min = min(self.surrogate_lag, max(win_size - seg_len, 0))
        start_max = max(start_min, win_size - seg_len)
        start = decision_rng.randint(start_min, start_max)
        return start, start + seg_len

    def _event_scale_names(self):
        if self.segment_length_policy == "event_scale_balanced":
            return ("micro", "medium", "persistent")
        return ("uniform",)

    def _empty_event_scale_counts(self):
        return {name: 0 for name in self._event_scale_names()}

    def _empty_event_scale_sums(self):
        return {name: 0.0 for name in self._event_scale_names()}

    def _segment_scale_name(self, segment_length, win_size):
        if self.segment_length_policy != "event_scale_balanced":
            return "uniform"
        if segment_length <= self.event_scale_micro_max:
            return "micro"
        persistent_min = max(
            1,
            int(round(win_size * self.event_scale_persistent_ratio_min)),
        )
        return "persistent" if segment_length >= persistent_min else "medium"

    def _sample_segment_length(
        self,
        win_size,
        decision_rng,
        available=None,
        event_scale=None,
    ):
        """Sample only the temporal extent; edit mechanism selection is unchanged."""
        available = max(win_size - self.surrogate_lag, 1) if available is None else available
        if self.segment_length_policy == "uniform_ratio":
            if event_scale is not None:
                raise ValueError(
                    "Diagnostic event-scale forcing requires event_scale_balanced"
                )
            ratio = decision_rng.uniform(
                self.synthetic_time_ratio_min,
                self.synthetic_time_ratio_max,
            )
            return min(available, max(2, int(round(win_size * ratio))))

        if event_scale is None:
            weights = self.event_scale_probabilities
        else:
            requested = str(event_scale)
            if requested not in self._event_scale_names():
                raise ValueError(f"Unsupported diagnostic event scale: {requested}")
            weights = tuple(
                1.0 if name == requested else 0.0
                for name in self._event_scale_names()
            )
        draw = decision_rng.random() * sum(weights)
        if draw < weights[0]:
            return min(
                available,
                decision_rng.randint(
                    self.event_scale_micro_min,
                    self.event_scale_micro_max,
                ),
            )
        if draw < weights[0] + weights[1]:
            ratio = decision_rng.uniform(
                self.synthetic_time_ratio_min,
                self.synthetic_time_ratio_max,
            )
        else:
            ratio = decision_rng.uniform(
                self.event_scale_persistent_ratio_min,
                self.event_scale_persistent_ratio_max,
            )
        return min(available, max(1, int(round(win_size * ratio))))

    def _build_surrogate_candidate(
        self,
        windows,
        batch_idx,
        adjacency,
        seed,
        neighbors,
        start,
        end,
        anomaly_type,
        progress,
        fit_cache=None,
    ):
        if anomaly_type not in self.SURROGATE_TYPES or neighbors.numel() == 0:
            return None

        parents = neighbors.unique()
        parents = parents[parents != seed]
        if parents.numel() == 0:
            return None
        cache_key = (batch_idx, seed)
        fit_state = None if fit_cache is None else fit_cache.get(cache_key)
        if fit_state is None:
            fit_state = self._fit_conditional_surrogate(windows, batch_idx, seed, parents)
            if fit_cache is not None:
                fit_cache[cache_key] = fit_state
        if fit_state is None:
            return None

        window = windows[batch_idx]
        time_indices = fit_state["time_indices"]
        segment_mask = (time_indices >= start) & (time_indices < end)
        if not segment_mask.any():
            return None
        lagged_windows = fit_state["lagged_windows"]
        targets = fit_state["targets"]
        context_mask = fit_state["context_mask"]
        feature_specs = fit_state["feature_specs"]
        standardized = fit_state["standardized"]
        standardized_context = fit_state["standardized_context"]
        coefficients = fit_state["coefficients"]
        residuals = fit_state["residuals"]
        context_residuals = fit_state["context_residuals"]
        context_variance = fit_state["context_variance"]
        surrogate_r2 = fit_state["surrogate_r2"]
        delta_r2 = fit_state["delta_r2"]
        channel_scale = fit_state["channel_scale"]
        residual_scale = fit_state["residual_scale"]

        standardized_segment = standardized[batch_idx, segment_mask]
        residual_segment = residuals[batch_idx, segment_mask]
        original_segment = targets[batch_idx, segment_mask]
        parent_terms_context = standardized_context[:, 1:] * coefficients[1:].unsqueeze(0)
        residual_without_parent = context_residuals.unsqueeze(1) + parent_terms_context
        parent_delta_r2 = (
            residual_without_parent.square().mean(dim=0)
            - context_residuals.square().mean()
        ) / context_variance
        parent_contributions = parent_delta_r2.clamp_min(0.0)
        if parent_contributions.numel() == 0:
            return None
        eligible_parent = parent_contributions >= self.surrogate_min_delta_r2
        if not eligible_parent.any():
            return None

        normal_scores = self._normal_residual_scores(
            residuals,
            end - start,
            residual_scale,
            excluded_batch=batch_idx,
            excluded_mask=torch.ones_like(segment_mask),
        )
        if normal_scores.numel() == 0 or not torch.isfinite(normal_scores).all():
            return None
        low_quantile = min(self.boundary_quantile_start, self.boundary_quantile_end)
        high_quantile = max(self.boundary_quantile_start, self.boundary_quantile_end)
        target_quantile = self._scheduled_boundary_quantile(progress)
        if random.random() < self.boundary_easy_mix:
            target_quantile = high_quantile
        boundary_low = torch.quantile(normal_scores, low_quantile)
        boundary_high = torch.quantile(normal_scores, high_quantile)
        boundary_target = torch.quantile(normal_scores, target_quantile)
        original_score = (residual_segment / residual_scale).square().mean().sqrt()
        if not torch.isfinite(original_score) or original_score >= boundary_low:
            return None

        ranked_terms = torch.argsort(
            parent_contributions.masked_fill(~eligible_parent, -1.0),
            descending=True,
        )
        ranked_terms = ranked_terms[eligible_parent[ranked_terms]]

        candidates = []
        for candidate_idx in range(self.boundary_candidates):
            parent_position = int(
                ranked_terms[candidate_idx % ranked_terms.numel()].item()
            )
            feature_position = parent_position + 1
            parent_term = standardized_segment[:, feature_position] * coefficients[feature_position]
            feature_kind, selected_parent, _ = feature_specs[feature_position]
            if anomaly_type == "surrogate_parent_delete":
                base_delta = -parent_term
            else:
                donor = self._select_surrogate_donor(
                    adjacency,
                    seed,
                    parents,
                    selected_parent,
                    candidate_idx,
                )
                if donor is None:
                    continue
                if feature_kind == "parent_current":
                    donor_values = windows.index_select(1, time_indices)[:, :, donor].float()
                else:
                    donor_values = lagged_windows[:, :, donor].float()
                donor_context = donor_values[context_mask]
                donor_mean = donor_context.mean()
                donor_std = donor_context.std(unbiased=False).clamp_min(1e-4)
                donor_segment = (
                    donor_values[batch_idx, segment_mask] - donor_mean
                ) / donor_std
                base_delta = coefficients[feature_position] * (
                    donor_segment - standardized_segment[:, feature_position]
                )

            calibrated = self._calibrate_surrogate_delta(
                original_segment,
                residual_segment,
                base_delta,
                residual_scale,
                channel_scale,
                boundary_low,
                boundary_high,
                boundary_target,
                original_score,
            )
            if calibrated is not None:
                calibrated["surrogate_r2"] = surrogate_r2.detach()
                calibrated["surrogate_delta_r2"] = delta_r2.detach()
                calibrated["selected_contribution"] = parent_contributions[parent_position].detach()
                candidates.append(calibrated)

        if not candidates:
            return None
        penalties = torch.stack([candidate["penalty"] for candidate in candidates])
        accepted = torch.stack([candidate["accepted"] for candidate in candidates])
        accepted_penalty = (~accepted).to(dtype=penalties.dtype) * accepted.any().to(
            dtype=penalties.dtype
        ) * 1e6
        selected_candidate = int((penalties + accepted_penalty).argmin().item())
        return candidates[selected_candidate]

    def _fit_conditional_surrogate(self, windows, batch_idx, seed, parents):
        win_size = windows.size(1)
        time_indices = torch.arange(
            self.surrogate_lag,
            win_size,
            device=windows.device,
            dtype=torch.long,
        )
        if time_indices.numel() == 0:
            return None

        lag_indices = time_indices - self.surrogate_lag
        lagged_windows = windows.index_select(1, lag_indices)
        current_windows = windows.index_select(1, time_indices)
        targets = current_windows[:, :, seed].float()
        context_mask = torch.ones_like(targets, dtype=torch.bool)
        context_mask[batch_idx] = False
        context_count = int(context_mask.sum().item())

        feature_specs = [("self_lag", seed, lagged_windows[:, :, seed])]
        for parent in parents.tolist():
            feature_specs.append(("parent_lag", parent, lagged_windows[:, :, parent]))
            if self.surrogate_include_contemporaneous:
                feature_specs.append(("parent_current", parent, current_windows[:, :, parent]))
        features = torch.stack([spec[2] for spec in feature_specs], dim=2).float()
        min_context = max(self.surrogate_min_context, features.size(2) + 2)
        if context_count < min_context:
            return None

        context_features = features[context_mask]
        context_targets = targets[context_mask]
        feature_mean = context_features.mean(dim=0, keepdim=True)
        feature_std = context_features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
        standardized = (features - feature_mean) / feature_std
        standardized_context = standardized[context_mask]
        target_mean = context_targets.mean()
        centered_targets = context_targets - target_mean
        normalizer = max(context_count, 1)
        gram = standardized_context.transpose(0, 1) @ standardized_context / normalizer
        gram = gram + self.surrogate_ridge * torch.eye(
            gram.size(0),
            device=gram.device,
            dtype=gram.dtype,
        )
        rhs = standardized_context.transpose(0, 1) @ centered_targets / normalizer
        try:
            coefficients = torch.linalg.solve(gram, rhs)
        except RuntimeError:
            return None

        predictions = target_mean + standardized @ coefficients
        residuals = targets - predictions
        context_residuals = residuals[context_mask]
        context_variance = centered_targets.square().mean().clamp_min(1e-8)
        surrogate_r2 = 1.0 - context_residuals.square().mean() / context_variance
        if not torch.isfinite(surrogate_r2) or surrogate_r2 < self.surrogate_min_r2:
            return None

        self_only = standardized_context[:, :1]
        self_gram = self_only.transpose(0, 1) @ self_only / normalizer
        self_gram = self_gram + self.surrogate_ridge * torch.eye(
            1,
            device=self_gram.device,
            dtype=self_gram.dtype,
        )
        self_rhs = self_only.transpose(0, 1) @ centered_targets / normalizer
        try:
            self_coefficient = torch.linalg.solve(self_gram, self_rhs)
        except RuntimeError:
            return None
        self_residual = centered_targets - self_only @ self_coefficient
        self_r2 = 1.0 - self_residual.square().mean() / context_variance
        delta_r2 = surrogate_r2 - self_r2
        if not torch.isfinite(delta_r2) or delta_r2 < self.surrogate_min_delta_r2:
            return None

        channel_scale = self._channel_scale(windows[batch_idx, :, seed]).float()
        residual_scale = context_residuals.std(unbiased=False)
        residual_scale = torch.maximum(
            residual_scale,
            channel_scale * self.surrogate_min_residual_ratio,
        ).clamp_min(1e-4)
        return {
            "time_indices": time_indices,
            "lagged_windows": lagged_windows,
            "targets": targets,
            "context_mask": context_mask,
            "feature_specs": feature_specs,
            "standardized": standardized,
            "standardized_context": standardized_context,
            "coefficients": coefficients,
            "residuals": residuals,
            "context_residuals": context_residuals,
            "context_variance": context_variance,
            "surrogate_r2": surrogate_r2,
            "delta_r2": delta_r2,
            "channel_scale": channel_scale,
            "residual_scale": residual_scale,
        }

    def _calibrate_surrogate_delta(
        self,
        original,
        residual,
        base_delta,
        residual_scale,
        channel_scale,
        boundary_low,
        boundary_high,
        boundary_target,
        original_score,
    ):
        if not torch.isfinite(base_delta).all() or base_delta.square().mean().sqrt() <= 1e-8:
            return None

        lower = torch.maximum(
            boundary_low,
            original_score + self.boundary_min_score_increase,
        )
        upper = torch.maximum(boundary_high, lower + 0.05)
        target = torch.minimum(torch.maximum(boundary_target, lower), upper)
        scales = torch.logspace(
            math.log10(self.surrogate_min_scale),
            math.log10(self.surrogate_max_scale),
            steps=max(16, self.boundary_candidates * 8),
            device=original.device,
            dtype=original.dtype,
        )
        quadratic_a = base_delta.square().mean()
        quadratic_b = 2.0 * (residual * base_delta).mean()
        quadratic_c = residual.square().mean() - (target * residual_scale).square()
        discriminant = quadratic_b.square() - 4.0 * quadratic_a * quadratic_c
        if bool((quadratic_a > 1e-12).item()) and bool((discriminant >= 0).item()):
            root_scale = torch.sqrt(discriminant.clamp_min(0.0))
            roots = torch.stack(
                [
                    (-quadratic_b - root_scale) / (2.0 * quadratic_a),
                    (-quadratic_b + root_scale) / (2.0 * quadratic_a),
                ]
            )
            roots = roots[
                torch.isfinite(roots)
                & (roots >= self.surrogate_min_scale)
                & (roots <= self.surrogate_max_scale)
            ]
            if roots.numel() > 0:
                scales = torch.cat([scales, roots]).unique(sorted=True)
        deltas = scales.unsqueeze(1) * base_delta.unsqueeze(0)
        if self.max_delta_std > 0:
            point_cap = self.max_delta_std * channel_scale
            deltas = deltas.clamp(min=-point_cap, max=point_cap)
        candidate_residuals = residual.unsqueeze(0) + deltas
        boundary_scores = (
            candidate_residuals / residual_scale
        ).square().mean(dim=1).sqrt()
        delta_rms = (deltas / channel_scale).square().mean(dim=1).sqrt()
        mean_shift = deltas.mean(dim=1).abs() / channel_scale
        if not torch.isfinite(boundary_scores).all():
            return None
        valid = (boundary_scores >= lower) & (boundary_scores <= upper)
        if self.max_delta_std > 0:
            valid = valid & (delta_rms <= self.max_delta_std)
        if self.boundary_mean_shift_cap > 0:
            valid = valid & (mean_shift <= self.boundary_mean_shift_cap)
        penalty = (boundary_scores - target).abs()
        penalty = penalty + 5.0 * (lower - boundary_scores).clamp_min(0.0)
        penalty = penalty + 5.0 * (boundary_scores - upper).clamp_min(0.0)
        if self.max_delta_std > 0:
            penalty = penalty + 5.0 * (delta_rms - self.max_delta_std).clamp_min(0.0)
        if self.boundary_mean_shift_cap > 0:
            penalty = penalty + 5.0 * (mean_shift - self.boundary_mean_shift_cap).clamp_min(0.0)
        selection_penalty = penalty + (
            (~valid).to(dtype=penalty.dtype)
            * valid.any().to(dtype=penalty.dtype)
            * 1e6
        )
        selected = selection_penalty.argmin()
        return {
            "values": (original + deltas[selected]).to(dtype=original.dtype),
            "delta_rms": delta_rms[selected].detach(),
            "boundary_score": boundary_scores[selected].detach(),
            "boundary_target": target.detach(),
            "original_score": original_score.detach(),
            "accepted": valid[selected].detach(),
            "penalty": penalty[selected].detach(),
            "surrogate_r2": original.new_tensor(float("nan")),
            "surrogate_delta_r2": original.new_tensor(float("nan")),
            "selected_contribution": original.new_tensor(float("nan")),
        }

    def _fallback_surrogate_candidate(self, window, adjacency, seed, neighbors, start, end):
        original = window[start:end, seed].float()
        if neighbors.numel() > 0:
            donor = int(neighbors[0].item())
        else:
            row = adjacency[seed].clone()
            row[seed] = -1.0
            donor = int(row.argmax().item())
        donor_values = window[start:end, donor].float()
        donor_values = (donor_values - donor_values.mean()) / donor_values.std(unbiased=False).clamp_min(1e-4)
        channel_scale = self._channel_scale(window[:, seed]).float()
        aligned = original.mean() + donor_values * original.std(unbiased=False).clamp_min(0.1 * channel_scale)
        delta = 0.25 * (aligned - original)
        if (delta / channel_scale).square().mean().sqrt() <= 1e-6:
            delta = 0.25 * (torch.roll(original, shifts=1) - original)
        if (delta / channel_scale).square().mean().sqrt() <= 1e-6:
            ramp = torch.linspace(
                -0.05,
                0.05,
                steps=max(original.numel(), 1),
                device=original.device,
                dtype=original.dtype,
            )
            delta = channel_scale * ramp
        if self.max_delta_std > 0:
            point_cap = self.max_delta_std * channel_scale
            delta = delta.clamp(min=-point_cap, max=point_cap)
        return {
            "values": (original + delta).to(dtype=window.dtype),
            "delta_rms": (delta / channel_scale).square().mean().sqrt().detach(),
            "boundary_score": original.new_tensor(float("nan")),
            "boundary_target": original.new_tensor(float("nan")),
            "original_score": original.new_tensor(float("nan")),
            "accepted": original.new_tensor(False, dtype=torch.bool),
            "penalty": original.new_tensor(float("inf")),
            "surrogate_r2": original.new_tensor(float("nan")),
            "surrogate_delta_r2": original.new_tensor(float("nan")),
            "selected_contribution": original.new_tensor(float("nan")),
        }

    def _select_surrogate_donor(self, adjacency, seed, parents, selected_parent, offset):
        n_var = adjacency.size(0)
        excluded = torch.zeros(n_var, device=adjacency.device, dtype=torch.bool)
        excluded[seed] = True
        excluded[parents] = True
        scores = adjacency[seed].clone()
        scores[excluded] = float("inf")
        available = (~excluded).nonzero(as_tuple=False).flatten()
        if available.numel() == 0:
            excluded.zero_()
            excluded[seed] = True
            excluded[selected_parent] = True
            scores = adjacency[seed].clone()
            scores[excluded] = float("inf")
            available = (~excluded).nonzero(as_tuple=False).flatten()
            if available.numel() == 0:
                return None
        ordered = torch.argsort(scores)
        ordered = ordered[~excluded[ordered]]
        if ordered.numel() == 0:
            ordered = available
        return int(ordered[offset % ordered.numel()].item())

    @staticmethod
    def _normal_residual_scores(
        residuals,
        segment_length,
        residual_scale,
        excluded_batch=None,
        excluded_mask=None,
    ):
        normalized = residuals / residual_scale
        score_length = min(max(int(segment_length), 1), normalized.size(1))
        step = max(1, score_length // 4)
        residual_windows = normalized.unfold(1, score_length, step)
        scores = residual_windows.square().mean(dim=2).sqrt()

        if excluded_batch is not None and excluded_mask is not None:
            overlap = excluded_mask.unfold(0, score_length, step).any(dim=1)
            scores = scores.clone()
            scores[excluded_batch, overlap] = float("nan")

        valid_scores = scores[torch.isfinite(scores)]
        if valid_scores.numel() > 0:
            return valid_scores.clamp_min(1e-6)

        point_mask = torch.ones_like(normalized, dtype=torch.bool)
        if excluded_batch is not None and excluded_mask is not None:
            point_mask[excluded_batch, excluded_mask] = False
        return normalized.masked_select(point_mask).abs().clamp_min(1e-6)

    def random_noise_breaking_view(self, windows, progress=1.0, collect_diagnostics=True):
        augmented = windows.clone()
        batch_size, win_size, n_var = augmented.shape
        direct_node_mask = augmented.new_zeros(batch_size, n_var)
        propagated_node_mask = augmented.new_zeros(batch_size, n_var)
        anomaly_type_ids = torch.zeros(batch_size, device=windows.device, dtype=torch.long)
        seg_len = min(win_size, max(2, int(round(win_size * self.synthetic_time_ratio))))
        magnitude = self._scheduled_magnitude(progress)

        for batch_idx in range(batch_size):
            seed = random.randrange(n_var)
            start = random.randint(0, max(win_size - seg_len, 0))
            end = start + seg_len
            direct_node_mask[batch_idx, seed] = 1.0

            segment = augmented[batch_idx, start:end, seed]
            sign = -1.0 if random.random() < 0.5 else 1.0
            bias = sign * magnitude * 0.5
            noise_scale = abs(float(self.random_noise_std)) * max(abs(float(magnitude)), 1e-6)
            noise = noise_scale * torch.randn_like(segment)
            augmented[batch_idx, start:end, seed] = segment + bias + noise

        type_counts = augmented.new_zeros(len(self.synthetic_types))
        if type_counts.numel() > 0:
            type_counts[0] = float(batch_size)
        if collect_diagnostics:
            self.last_diagnostics = {
                "seed_evidence_mean": float("nan"),
                "evidence_sampling_ratio": 0.0,
                "random_sampling_ratio": 1.0,
                "edit_mode": "random_noise",
                "edit_strength": float("nan"),
                "magnitude": float(magnitude),
                "type_ratio": (type_counts / max(batch_size, 1)).detach().cpu().tolist(),
                "mode": "random_noise",
            }
        else:
            self.last_diagnostics = {}
        self.last_valid_mask = torch.ones(
            batch_size,
            device=windows.device,
            dtype=torch.bool,
        )
        self.last_sample_weights = torch.ones(
            batch_size,
            device=windows.device,
            dtype=windows.dtype,
        )
        self.last_acceptance_tier = torch.full(
            (batch_size,),
            self.ACCEPTANCE_HARD,
            device=windows.device,
            dtype=torch.long,
        )
        self.last_edit_source_ids = torch.full(
            (batch_size,),
            self.EDIT_SOURCE_FALLBACK,
            device=windows.device,
            dtype=torch.long,
        )
        self.last_seed_ids = direct_node_mask.argmax(dim=1).long()
        seed_counts = direct_node_mask.sum(dim=0)
        self.last_coverage_diagnostics = {
            "attempted_count": batch_size,
            "valid_count": batch_size,
            "fitted_window_count": 0,
            "surrogate_accepted_count": 0,
            "fallback_count": batch_size,
            "invalid_count": 0,
            "attempted_seed_counts": seed_counts.detach().cpu().tolist(),
            "valid_seed_counts": seed_counts.detach().cpu().tolist(),
            "fallback_type_counts": {"random_noise": batch_size},
        }
        return augmented, direct_node_mask, propagated_node_mask, anomaly_type_ids

    def _sample_seed_and_neighbors(
        self,
        adjacency,
        n_var,
        n_neighbors,
        node_evidence=None,
        eligible_nodes=None,
    ):
        seed, used_evidence = self._sample_seed(
            adjacency,
            n_var,
            node_evidence,
            eligible_nodes=eligible_nodes,
        )
        row = adjacency[seed].clone()
        row[seed] = -1.0
        k = max(0, min(int(n_neighbors), n_var - 1))
        if k == 0:
            return seed, torch.empty(0, device=adjacency.device, dtype=torch.long), used_evidence
        neighbors = torch.topk(row, k=k, dim=0).indices
        return seed, neighbors, used_evidence

    def _sample_seed(
        self,
        adjacency,
        n_var,
        node_evidence=None,
        eligible_nodes=None,
    ):
        if eligible_nodes is None or eligible_nodes.numel() == 0:
            eligible_nodes = torch.arange(n_var, device=adjacency.device)
        decision_rng = self._decision_rng or random

        def sample_random():
            offset = decision_rng.randrange(int(eligible_nodes.numel()))
            return int(eligible_nodes[offset].item())

        if self.seed_selection == "random":
            return sample_random(), False

        if decision_rng.random() < float(self.evidence_random_mix):
            return sample_random(), False

        if self.seed_selection == "mechanism_aware":
            weights = self._mechanism_seed_prior(
                adjacency.device,
                adjacency.dtype,
            )
        else:
            weights = node_evidence

        if weights is None:
            return sample_random(), False
        weights = weights.detach().float().clone()
        if weights.numel() != n_var or not torch.isfinite(weights).all():
            return sample_random(), False
        weights = weights.index_select(0, eligible_nodes)
        weights = weights - weights.min()
        if weights.max() <= 1e-12:
            return sample_random(), False
        temperature = max(float(self.evidence_temperature), 1e-6)
        probs = torch.softmax(weights / temperature, dim=0)
        if not torch.isfinite(probs).all() or probs.sum() <= 0:
            return sample_random(), False
        selected = int(torch.multinomial(probs, num_samples=1).item())
        seed = int(eligible_nodes[selected].item())
        return seed, True

    def _mechanism_seed_prior(self, device, dtype):
        if self.seed_selection != "mechanism_aware":
            return None
        if self.frontend_graph_effective_selector is not None:
            return self.frontend_graph_effective_selector.seed_prior(device, dtype)
        if self.normal_mechanism is None or not self.normal_mechanism.fitted:
            return None
        prior = self.normal_mechanism.seed_relevance(device, dtype)
        if prior.numel() == 0 or not torch.isfinite(prior).all():
            return None
        return prior

    def _scheduled_magnitude(self, progress):
        if not self.strength_schedule:
            return float(self.synthetic_magnitude)
        progress = min(max(float(progress), 0.0), 1.0)
        return float(self.magnitude_start + progress * (self.magnitude_end - self.magnitude_start))

    def _scheduled_edit_strength(self, progress):
        if not self.strength_schedule:
            return float(self.edit_strength_end)
        progress = min(max(float(progress), 0.0), 1.0)
        return float(
            self.edit_strength_start
            + progress * (self.edit_strength_end - self.edit_strength_start)
        )

    def _record_generator_count(self, source, event, count, seed, anomaly_type, start, end, window_size):
        self.generator_counts[source][event] += count
        key = (source, anomaly_type, int(seed), self._segment_scale_name(end - start, window_size), event)
        self.generator_cells[key] = self.generator_cells.get(key, 0) + count

    def _normal_candidate_segments(
        self,
        window,
        seed,
        start,
        end,
        anomaly_type,
        strengths,
        phase=0,
        self_temporal=False,
    ):
        """Share the existing strength budget; hybrid never doubles the pool.

        Alternate ownership of strength slots by sample/segment phase so neither
        generator is permanently assigned the largest intervention strength.
        Counts are cumulative over calls (training AND validation if both call).
        """
        sources = [self.candidate_generator] * len(strengths)
        if self.candidate_generator == "hybrid":
            sources = ["graph_var" if (index + phase) % 2 == 0 else "funtsg_inspired"
                       for index in range(len(strengths))]
        generated = [None] * len(sources)
        for source in ("graph_var", "funtsg_inspired"):
            indices = [i for i, name in enumerate(sources) if name == source]
            if not indices:
                continue
            levels = strengths[indices]
            self._record_generator_count(source, "proposed", len(indices), seed, anomaly_type, start, end, window.size(0))
            if source == "funtsg_inspired":
                values = function_candidates(self.normal_mechanism, window, seed, start, end, anomaly_type, levels)
            elif self.counterfactual_implementation != "scalar":
                candidate_args = (
                    window,
                    seed,
                    start,
                    end,
                    anomaly_type,
                    levels,
                )
                candidate_kwargs = {
                    "precompute_parents": (
                        self.counterfactual_implementation
                        == "batched_precomputed"
                    )
                }
                if self_temporal:
                    candidate_kwargs["self_temporal"] = True
                values = self.normal_mechanism.counterfactual_candidates(
                    *candidate_args,
                    **candidate_kwargs,
                )
            else:
                values = [
                    self.normal_mechanism.counterfactual(
                        window,
                        seed,
                        start,
                        end,
                        anomaly_type,
                        float(level.item()),
                        **({"self_temporal": True} if self_temporal else {}),
                    )
                    for level in levels
                ]
            for i, value in zip(indices, values):
                generated[i] = value
                self._record_generator_count(source, "generated", int(value is not None), seed, anomaly_type, start, end, window.size(0))
        return ([value for value in generated if value is not None],
                [source for source, value in zip(sources, generated) if value is not None])

    def _direct_intervention_strengths(self, reference, progress, anomaly_type):
        max_strength = self._scheduled_edit_strength(progress)
        levels = torch.linspace(
            max_strength / self.hybrid_fallback_candidates,
            max_strength,
            steps=self.hybrid_fallback_candidates,
            device=reference.device,
            dtype=reference.dtype,
        )
        if (
            self.direct_mechanism_score_mode == "adaptive_graph_var"
            and anomaly_type == "lag_break"
            and max_strength < 1.0 - 1e-6
        ):
            levels = torch.cat((levels, levels.new_ones(1)))
        return levels

    def _scheduled_boundary_quantile(self, progress):
        if not self.strength_schedule:
            return float(self.boundary_quantile_end)
        progress = min(max(float(progress), 0.0), 1.0)
        return float(
            self.boundary_quantile_start
            + progress * (self.boundary_quantile_end - self.boundary_quantile_start)
        )

    def _lag_break(self, augmented, batch_idx, seed, start, end, strength=None):
        win_size = augmented.size(1)
        seg_len = end - start
        max_shift = max(1, min(seg_len, win_size - seg_len))
        shift = random.randint(1, max_shift)
        if start + shift + seg_len <= win_size:
            source_start = start + shift
        else:
            source_start = max(0, start - shift)
        source = augmented[batch_idx, source_start:source_start + seg_len, seed].clone()
        if source.size(0) < seg_len:
            source = torch.flip(augmented[batch_idx, start:end, seed], dims=[0])
        self._apply_candidate(augmented, batch_idx, seed, start, end, source, strength)

    def _neighbor_break(
        self,
        augmented,
        windows,
        adjacency,
        batch_idx,
        seed,
        neighbors,
        start,
        end,
        strength=None,
    ):
        row = adjacency[seed].clone()
        row[seed] = float("inf")
        if neighbors.numel() > 0:
            row[neighbors] = float("inf")
        if torch.isinf(row).all():
            row = adjacency[seed].clone()
            row[seed] = float("inf")
        donor = int(torch.argmin(row).item())
        donor = max(0, min(donor, windows.size(2) - 1))
        donor_segment = windows[batch_idx, start:end, donor]
        seed_segment = augmented[batch_idx, start:end, seed]
        legacy_candidate = (
            self.edit_mode != "direct_quantile"
            or (
                self.direct_mechanism_score_mode == "legacy_shared"
                and self.direct_candidate_support_mode == "none"
            )
        )
        if not legacy_candidate:
            donor_window = windows[batch_idx, :, donor]
            seed_window = windows[batch_idx, :, seed]
            donor_mean = donor_window.mean()
            donor_scale = donor_window.std(unbiased=False).clamp_min(1e-4)
            seed_mean = seed_window.mean()
            seed_scale = seed_window.std(unbiased=False).clamp_min(1e-4)
            donor_segment = (
                (donor_segment - donor_mean) / donor_scale * seed_scale + seed_mean
            )
        if strength is None:
            mix = 0.6 + 0.3 * random.random()
            candidate = mix * donor_segment + (1.0 - mix) * seed_segment
        else:
            candidate = donor_segment
        self._apply_candidate(augmented, batch_idx, seed, start, end, candidate, strength)

    def _relation_flip(
        self,
        augmented,
        adjacency,
        batch_idx,
        seed,
        neighbors,
        start,
        end,
        strength=None,
    ):
        seed_segment = augmented[batch_idx, start:end, seed]
        legacy_candidate = (
            self.edit_mode != "direct_quantile"
            or (
                self.direct_mechanism_score_mode == "legacy_shared"
                and self.direct_candidate_support_mode == "none"
            )
        )
        valid_neighbors = neighbors.unique()
        valid_neighbors = valid_neighbors[valid_neighbors != seed]
        if legacy_candidate:
            if valid_neighbors.numel() > 0:
                reference = augmented[batch_idx, start:end].index_select(
                    dim=1,
                    index=valid_neighbors,
                ).mean(dim=1)
            else:
                reference = seed_segment.mean().expand_as(seed_segment)
            candidate = 2.0 * reference - seed_segment
        elif valid_neighbors.numel() > 0:
            neighbor_values = augmented[batch_idx].index_select(
                dim=1,
                index=valid_neighbors,
            )
            neighbor_mean = neighbor_values.mean(dim=0, keepdim=True)
            neighbor_scale = neighbor_values.std(
                dim=0,
                unbiased=False,
                keepdim=True,
            ).clamp_min(1e-4)
            neighbor_z = (neighbor_values - neighbor_mean) / neighbor_scale
            weights = adjacency[seed].index_select(0, valid_neighbors).float()
            weights = weights.clamp_min(0.0)
            if not torch.isfinite(weights).all() or bool(
                (weights.sum() <= 1e-12).item()
            ):
                weights = torch.ones_like(weights)
            weights = weights / weights.sum().clamp_min(1e-12)
            reference_z = (
                neighbor_z[start:end] * weights.view(1, -1)
            ).sum(dim=1)
            seed_window = augmented[batch_idx, :, seed]
            seed_mean = seed_window.mean()
            seed_scale = seed_window.std(unbiased=False).clamp_min(1e-4)
            seed_z = (seed_segment - seed_mean) / seed_scale
            candidate = seed_mean + seed_scale * (2.0 * reference_z - seed_z)
        else:
            candidate = seed_segment.clone()
        self._apply_candidate(augmented, batch_idx, seed, start, end, candidate, strength)

    def _local_function_shift(
        self,
        augmented,
        batch_idx,
        seed,
        start,
        end,
        magnitude_value=None,
        strength=None,
    ):
        segment = augmented[batch_idx, start:end, seed]
        sign = -1.0 if random.random() < 0.5 else 1.0
        if strength is not None:
            channel_scale = self._channel_scale(augmented[batch_idx, :, seed])
            ramp = torch.linspace(
                0.0,
                1.0,
                steps=end - start,
                device=augmented.device,
                dtype=augmented.dtype,
            )
            candidate = segment * (1.0 + 0.2 * sign) + sign * channel_scale * (0.5 + ramp)
            self._apply_candidate(augmented, batch_idx, seed, start, end, candidate, strength)
            return

        base_magnitude = self.synthetic_magnitude if magnitude_value is None else magnitude_value
        magnitude = sign * base_magnitude
        drift = torch.linspace(
            0.0,
            magnitude,
            steps=end - start,
            device=augmented.device,
            dtype=augmented.dtype,
        )
        scale = 1.0 + abs(base_magnitude) * (0.1 + 0.2 * random.random())
        augmented[batch_idx, start:end, seed] = segment * scale + 0.5 * magnitude + drift

    def _propagation_break(
        self,
        augmented,
        batch_idx,
        seed,
        neighbors,
        start,
        end,
        magnitude_value=None,
        strength=None,
    ):
        self._local_function_shift(
            augmented,
            batch_idx,
            seed,
            start,
            end,
            magnitude_value,
            strength,
        )
        if neighbors.numel() == 0 or end - start < 2:
            return torch.empty(0, device=augmented.device, dtype=torch.long)

        n_prop = random.randint(1, max(1, min(neighbors.numel(), self.subgraph_topk)))
        propagated = neighbors[:n_prop]
        delay = random.randint(1, max(1, (end - start) // 3))
        p_start = min(start + delay, end - 1)
        p_end = end
        seed_effect = augmented[batch_idx, p_start - delay:p_end - delay, seed].unsqueeze(-1)
        neighbor_segment = augmented[batch_idx, p_start:p_end].index_select(dim=1, index=propagated)
        values = neighbor_segment + 0.2 * (seed_effect - seed_effect.mean(dim=0, keepdim=True))
        augmented[batch_idx, p_start:p_end] = self._assign_nodes(
            augmented[batch_idx, p_start:p_end],
            propagated,
            values,
        )
        return propagated

    def _scope_aware_negative_propagation(
        self,
        augmented,
        windows,
        propagated_node_mask,
        batch_idx,
        seed,
        neighbors,
        adjacency,
        start,
        end,
        candidate_values,
    ):
        """Expand a selected mechanism break to a graph-supported scope.

        The direct candidate has already been screened against the normal
        Graph-VAR residual reference.  This routine does not invent an
        unrelated anomaly: it transports the selected seed deviation through
        normal parent/child links with a delay, then projects each propagated
        node back into its train-normal marginal support.  Thus a negative view
        can represent both local and cooperative failures while preserving a
        bounded perturbation.  All decisions are made from train-fitted
        mechanism state and the current normal window.
        """
        empty = torch.empty(
            0,
            device=augmented.device,
            dtype=torch.long,
        )
        if not self.scope_aware_negative_enable:
            return empty
        if neighbors is None or neighbors.numel() == 0:
            return empty
        decision_rng = self._decision_rng or random
        if decision_rng.random() >= self.scope_aware_negative_probability:
            return empty
        if end <= start:
            return empty

        candidate_nodes = neighbors.unique()
        candidate_nodes = candidate_nodes[candidate_nodes != seed]
        if candidate_nodes.numel() == 0:
            return empty

        # CALAD-style relevance is represented here by the normal Graph-VAR
        # residual reduction.  Prefer nodes whose local equation consumes the
        # seed, but retain the adjacency fallback for sparse SMD relations.
        affected = self.normal_mechanism.affected_context_mask(
            seed,
            augmented.device,
        )
        relation_nodes = candidate_nodes[affected[candidate_nodes]]
        if relation_nodes.numel() > 0:
            candidate_nodes = relation_nodes

        relevance = self.normal_mechanism.seed_relevance(
            augmented.device,
            augmented.dtype,
        )
        weights = adjacency.index_select(0, candidate_nodes).float()
        weights = weights.clamp_min(0.0)
        if relevance.numel() == augmented.size(2):
            weights = weights * (1.0 + relevance.index_select(0, candidate_nodes).float())
        if not torch.isfinite(weights).all() or float(weights.sum()) <= 1e-12:
            weights = torch.ones_like(weights)

        # Short events stay local; persistent events can exercise a wider
        # cooperative scope, matching the MSL/SMAP/SMD event-scale mixture.
        segment_fraction = min(
            max(float(end - start) / max(windows.size(1), 1), 0.0),
            1.0,
        )
        desired = int(
            round(
                self.scope_aware_negative_min_nodes
                + (
                    self.scope_aware_negative_max_nodes
                    - self.scope_aware_negative_min_nodes
                )
                * segment_fraction ** 0.5
            )
        )
        desired = min(
            candidate_nodes.numel(),
            max(self.scope_aware_negative_min_nodes, desired),
        )
        topk = torch.topk(weights, k=desired, dim=0)
        selected = candidate_nodes[topk.indices]
        # The graph gate currently uses relation strength and Graph-VAR
        # relevance to choose the scope.  Reuse that same train-normal
        # confidence to scale the perturbation, instead of applying an
        # identical effect to weak and strong causal links.
        confidence = topk.values / topk.values.max().clamp_min(1e-6)
        confidence = confidence.clamp_min(0.25)

        original_seed = windows[batch_idx, start:end, seed]
        seed_delta = candidate_values - original_seed
        seed_scale = self._channel_scale(windows[batch_idx, :, seed])
        if (
            not torch.isfinite(seed_delta).all()
            or bool((seed_delta.abs().max() <= 1e-8).item())
            or self.scope_aware_negative_effect_scale <= 0.0
        ):
            return empty

        segment_length = end - start
        delay = 1 if segment_length <= 2 else decision_rng.randint(
            1,
            max(1, segment_length // 3),
        )
        propagation_start = min(start + delay, end - 1)
        propagation_length = end - propagation_start
        # The propagated view starts ``delay`` steps after the direct seed
        # edit.  Align the source perturbation to that delayed interval;
        # taking seed_delta from index zero would replay an earlier part of
        # the event on the neighbours and break the intended causal timing.
        source_delta = seed_delta[delay : delay + propagation_length]
        if source_delta.numel() != propagation_length:
            return empty
        source_delta = source_delta / seed_scale.clamp_min(1e-3)
        propagated_values = []
        valid_nodes = []
        for selected_index, node in enumerate(selected.tolist()):
            node_scale = self._channel_scale(windows[batch_idx, :, node])
            base = windows[batch_idx, propagation_start:end, node]
            values = base + (
                self.scope_aware_negative_effect_scale
                * confidence[selected_index]
                * source_delta
                * node_scale
            )
            support = self._direct_candidate_support(windows, node)
            if not support.get("collapsed", False):
                values, _ = self._project_to_direct_support(values, support)
            if torch.isfinite(values).all():
                valid_nodes.append(node)
                propagated_values.append(values)
        if not valid_nodes:
            return empty

        propagated = torch.as_tensor(
            valid_nodes,
            device=augmented.device,
            dtype=torch.long,
        )
        values = torch.stack(propagated_values, dim=1)
        augmented[batch_idx, propagation_start:end] = self._assign_nodes(
            augmented[batch_idx, propagation_start:end],
            propagated,
            values,
        )
        propagated_node_mask[batch_idx, propagated] = 1.0
        return propagated

    def _apply_candidate(self, augmented, batch_idx, seed, start, end, candidate, strength):
        if strength is None:
            augmented[batch_idx, start:end, seed] = candidate
            return

        original = augmented[batch_idx, start:end, seed]
        alpha = min(max(float(strength), 0.0), 1.0)
        delta = alpha * (candidate - original)
        if self.max_delta_std > 0:
            max_delta = self.max_delta_std * self._channel_scale(augmented[batch_idx, :, seed])
            delta = delta.clamp(min=-max_delta, max=max_delta)
        augmented[batch_idx, start:end, seed] = original + delta

    @staticmethod
    def _channel_scale(values):
        return values.float().std(unbiased=False).to(dtype=values.dtype).clamp_min(1e-3)

    def _sample_subgraph_nodes(self, adjacency, n_var, n_nodes):
        seed = random.randrange(n_var)
        row = adjacency[seed].clone()
        row[seed] = -1.0
        k_neighbors = max(0, min(n_nodes - 1, n_var - 1))
        if k_neighbors == 0:
            return torch.tensor([seed], device=adjacency.device, dtype=torch.long)
        neighbors = torch.topk(row, k=k_neighbors, dim=0).indices
        return torch.cat([torch.tensor([seed], device=adjacency.device, dtype=torch.long), neighbors], dim=0)

    @staticmethod
    def _assign_nodes(window_segment, nodes, values):
        updated = window_segment.clone()
        updated.index_copy_(dim=1, index=nodes, source=values)
        return updated
