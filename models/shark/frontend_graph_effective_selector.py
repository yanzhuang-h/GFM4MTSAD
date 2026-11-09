import math

import torch


class FrontendGraphEffectiveSelector:
    """Train-normal frontend/graph hard-negative selector.

    The selector is deliberately upstream of the GFM and anomaly head.  It
    accepts candidate pools whose direct DLinear drift is sufficient but not
    saturated and whose Pearson-graph change stays local to the intervened
    node.  The caller decides whether the pool is legacy-accepted only or the
    wider support/context-valid pool.  All fitted state comes from normal
    training windows.
    """

    SCALE_FLOOR_QUANTILE = 0.10
    EPS = 1e-8

    def __init__(self, cfg):
        synthetic = cfg.SHARK.SYNTHETIC
        self.d_fd_low = float(
            getattr(synthetic, "FRONTEND_EFFECTIVE_D_FD_LOW", 0.25)
        )
        self.d_fd_high = float(
            getattr(synthetic, "FRONTEND_EFFECTIVE_D_FD_HIGH", 2.0)
        )
        self.d_fd_target = float(
            getattr(synthetic, "FRONTEND_EFFECTIVE_D_FD_TARGET", 1.0)
        )
        self.nonincident_max = float(
            getattr(synthetic, "FRONTEND_EFFECTIVE_NONINCIDENT_MAX", 0.08)
        )
        self.mechanism_weight = max(
            0.0,
            float(
                getattr(
                    synthetic,
                    "FRONTEND_EFFECTIVE_MECHANISM_WEIGHT",
                    0.10,
                )
            ),
        )
        self.graph_weight = max(
            0.0,
            float(
                getattr(
                    synthetic,
                    "FRONTEND_EFFECTIVE_GRAPH_WEIGHT",
                    0.25,
                )
            ),
        )
        self.coverage_debt_min = float(
            getattr(synthetic, "FRONTEND_EFFECTIVE_COVERAGE_DEBT_MIN", 0.5)
        )
        self.coverage_debt_max = float(
            getattr(synthetic, "FRONTEND_EFFECTIVE_COVERAGE_DEBT_MAX", 2.0)
        )
        if not (
            0.0 < self.d_fd_low
            <= self.d_fd_target
            <= self.d_fd_high
        ):
            raise ValueError(
                "Frontend-effective D_fd bounds must satisfy "
                "0 < low <= target <= high"
            )
        if self.nonincident_max < 0.0:
            raise ValueError(
                "FRONTEND_EFFECTIVE_NONINCIDENT_MAX must be nonnegative"
            )
        if not (
            0.0 < self.coverage_debt_min
            <= 1.0
            <= self.coverage_debt_max
        ):
            raise ValueError(
                "Frontend-effective coverage debt bounds must satisfy "
                "0 < min <= 1 <= max"
            )

        self.frontend = None
        self.graph_builder = None
        self.node_scale = None
        self.valid_nodes = None
        self.fit_diagnostics = {}
        self._reset_running_state(0)

    @property
    def fitted(self):
        return self.node_scale is not None

    def bind(self, frontend, graph_builder):
        if not hasattr(frontend, "forward_channel"):
            raise TypeError(
                "Frontend-effective screening requires a channelwise frontend "
                "with forward_channel"
            )
        if not callable(graph_builder):
            raise TypeError("Frontend-effective screening requires a graph builder")
        self.frontend = frontend
        self.graph_builder = graph_builder

    @staticmethod
    def _windows_from_batch(batch):
        windows = batch[0] if isinstance(batch, (tuple, list)) else batch
        if not isinstance(windows, torch.Tensor):
            windows = torch.as_tensor(windows)
        if windows.ndim == 2:
            windows = windows.unsqueeze(0)
        if windows.ndim != 3:
            raise ValueError(
                "Frontend scale fitting expects normal windows [B, T, N], "
                f"got {tuple(windows.shape)}"
            )
        return windows

    def _reset_running_state(self, node_count):
        self.proposal_counts = torch.zeros(node_count, dtype=torch.float64)
        self.evaluated_candidate_counts = torch.zeros(
            node_count, dtype=torch.float64
        )
        self.effective_candidate_counts = torch.zeros(
            node_count, dtype=torch.float64
        )
        self.selected_counts = torch.zeros(node_count, dtype=torch.float64)

    @torch.no_grad()
    def fit(self, normal_source, valid_nodes):
        if self.frontend is None or self.graph_builder is None:
            raise RuntimeError("Bind the frontend and graph builder before fitting")
        if not hasattr(self.graph_builder, "reference_prior_diagnostics"):
            raise RuntimeError(
                "Frontend-effective screening requires the v2.1 reference graph"
            )
        graph_reference = self.graph_builder.reference_prior_diagnostics()
        if not graph_reference.get("reference_prior_fitted", False):
            raise RuntimeError(
                "Frontend-effective screening requires a fitted v2.1 "
                "train-normal graph reference"
            )
        valid_nodes = valid_nodes.detach().bool().cpu()
        if valid_nodes.numel() == 0 or not bool(valid_nodes.any()):
            raise RuntimeError(
                "Frontend-effective screening received no valid Graph-VAR nodes"
            )
        device = next(self.frontend.parameters()).device
        total = None
        total_square = None
        count = 0
        for batch in normal_source:
            windows = self._windows_from_batch(batch).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            features = self.frontend(windows).detach()
            if features.size(1) != valid_nodes.numel():
                raise RuntimeError(
                    "Frontend/Graph-VAR node-count mismatch during scale fit: "
                    f"frontend={features.size(1)}, graph_var={valid_nodes.numel()}"
                )
            if not bool(torch.isfinite(features).all()):
                raise RuntimeError(
                    "Non-finite DLinear feature in train-normal scale pass"
                )
            batch_sum = features.double().sum(dim=0).cpu()
            batch_square = features.double().square().sum(dim=0).cpu()
            total = batch_sum if total is None else total + batch_sum
            total_square = (
                batch_square
                if total_square is None
                else total_square + batch_square
            )
            count += int(features.size(0))
        if count < 2 or total is None or total_square is None:
            raise RuntimeError(
                "Frontend scale pass loaded fewer than two normal windows"
            )
        mean = total / count
        variance = (total_square / count - mean.square()).clamp_min(0.0)
        raw_scale = variance.mean(dim=-1).sqrt().float()
        eligible = raw_scale[
            valid_nodes & torch.isfinite(raw_scale) & (raw_scale > self.EPS)
        ]
        if eligible.numel() == 0:
            raise RuntimeError("No finite positive valid-node frontend scale")
        floor = torch.quantile(
            eligible,
            self.SCALE_FLOOR_QUANTILE,
        ).clamp_min(self.EPS)
        finite = torch.where(
            torch.isfinite(raw_scale),
            raw_scale,
            torch.zeros_like(raw_scale),
        )
        self.node_scale = finite.clamp_min(floor)
        self.valid_nodes = valid_nodes
        self._reset_running_state(int(valid_nodes.numel()))
        self.fit_diagnostics = {
            "fitted": True,
            "fit_windows": count,
            "valid_node_count": int(valid_nodes.sum().item()),
            "scale_floor_quantile": self.SCALE_FLOOR_QUANTILE,
            "scale_floor": float(floor.item()),
            "scale_floor_hit_count": int((finite < floor).sum().item()),
            "uses_test_labels": False,
            "uses_gfm_or_anomaly_head": False,
            "reference_prior_fitted": True,
        }
        return dict(self.fit_diagnostics)

    def require_ready(self):
        if self.frontend is None or self.graph_builder is None or not self.fitted:
            raise RuntimeError(
                "frontend_graph_effective screening must be bound and fitted "
                "on train-normal windows before candidate generation"
            )

    @torch.no_grad()
    def encode_original(self, windows):
        self.require_ready()
        return self.frontend(windows.float()).detach()

    @staticmethod
    def graph_nonincident_drift(original, candidate, seed):
        if original.ndim == 2:
            original = original.unsqueeze(0)
        if candidate.ndim == 2:
            candidate = candidate.unsqueeze(0)
        if original.shape != candidate.shape or original.ndim != 3:
            raise ValueError("Adjacency tensors must have matching [B, N, N] shape")
        delta = candidate - original
        batch_size, node_count, _ = delta.shape
        seeds = torch.as_tensor(seed, device=delta.device, dtype=torch.long)
        if seeds.ndim == 0:
            seeds = seeds.repeat(batch_size)
        nodes = torch.arange(node_count, device=delta.device).view(1, -1)
        nonseed = nodes != seeds.view(-1, 1)
        off_diagonal = ~torch.eye(
            node_count,
            dtype=torch.bool,
            device=delta.device,
        ).unsqueeze(0)
        nonincident = (
            nonseed.unsqueeze(2)
            & nonseed.unsqueeze(1)
            & off_diagonal
        )
        count = nonincident.sum(dim=(1, 2)).clamp_min(1)
        return (
            (delta.square() * nonincident).sum(dim=(1, 2)) / count
        ).sqrt()

    @torch.no_grad()
    def select(self, original_window, original_features, original_adjacency, seed, records):
        """Return the best eligible candidate and per-pool diagnostics."""
        self.require_ready()
        if not records:
            return None, {
                "evaluated": 0,
                "eligible": 0,
                "below_frontend": 0,
                "above_frontend": 0,
                "graph_nonlocal": 0,
            }
        seed = int(seed)
        count = len(records)
        seed_series = original_window[:, seed].unsqueeze(0).repeat(count, 1)
        for row, record in enumerate(records):
            seed_series[row, record["start"]:record["end"]] = record["values"]
        candidate_seed_features = self.frontend.forward_channel(
            seed_series.float(),
            seed,
        ).detach()
        original_features = original_features.detach()
        candidate_features = original_features.unsqueeze(0).repeat(count, 1, 1)
        candidate_features[:, seed, :] = candidate_seed_features
        candidate_adjacency = self.graph_builder(candidate_features.detach())
        original_adjacency = original_adjacency.detach().unsqueeze(0).expand(
            count, -1, -1
        )

        direct_delta = candidate_seed_features - original_features[seed]
        direct_raw = direct_delta.square().mean(dim=-1).sqrt()
        scale = self.node_scale[seed].to(
            device=direct_raw.device,
            dtype=direct_raw.dtype,
        ).clamp_min(self.EPS)
        d_fd = direct_raw / scale
        d_adj_nonincident = self.graph_nonincident_drift(
            original_adjacency,
            candidate_adjacency,
            seed,
        )
        finite = torch.isfinite(d_fd) & torch.isfinite(d_adj_nonincident)
        frontend_low = d_fd >= self.d_fd_low
        frontend_high = d_fd <= self.d_fd_high
        graph_local = d_adj_nonincident <= self.nonincident_max
        gate_eligible = finite & frontend_low & frontend_high & graph_local

        # v2.3 admits mechanism-cap rescues at a low sample weight, but only
        # when the same pool has no ordinary legacy-accepted candidate that
        # passes the frontend/graph gates.  This makes the new path a
        # coverage safeguard for saturated nodes instead of a replacement for
        # the normal hard-negative distribution.
        rescue_flags = torch.as_tensor(
            [bool(record.get("rms_rescue_eligible", False)) for record in records],
            device=d_fd.device,
            dtype=torch.bool,
        )
        ordinary_eligible = gate_eligible & ~rescue_flags
        rescue_eligible = gate_eligible & rescue_flags
        use_rescue = not bool(ordinary_eligible.any().item())
        eligible = ordinary_eligible | (
            rescue_eligible if use_rescue else torch.zeros_like(rescue_eligible)
        )
        legacy_flags = torch.as_tensor(
            [bool(record.get("legacy_accepted", True)) for record in records],
            device=d_fd.device,
            dtype=torch.bool,
        )

        best = None
        best_penalty = None
        for index, record in enumerate(records):
            frontend_penalty = abs(
                math.log(
                    max(float(d_fd[index].item()), self.EPS)
                    / self.d_fd_target
                )
            )
            graph_penalty = float(d_adj_nonincident[index].item()) / max(
                self.nonincident_max,
                self.EPS,
            )
            mechanism_penalty = float(record["penalty"].detach().item())
            selection_penalty = (
                frontend_penalty
                + self.graph_weight * graph_penalty
                + self.mechanism_weight * mechanism_penalty
            )
            metadata = {
                "frontend_direct_drift": d_fd[index].detach(),
                "frontend_direct_drift_raw": direct_raw[index].detach(),
                "graph_nonincident_drift": d_adj_nonincident[index].detach(),
                "frontend_in_band": bool(
                    (frontend_low[index] & frontend_high[index]).item()
                ),
                "graph_local": bool(graph_local[index].item()),
                "eligible": bool(eligible[index].item()),
                "frontend_coverage_rescue": bool(
                    use_rescue and rescue_eligible[index].item()
                ),
                "selection_penalty": d_fd.new_tensor(selection_penalty),
            }
            record.update(metadata)
            if not metadata["eligible"]:
                continue
            if (
                best is None
                or selection_penalty < best_penalty - 1e-12
                or (
                    abs(selection_penalty - best_penalty) <= 1e-12
                    and bool(record["penalty"] < best["penalty"])
                )
            ):
                best = record
                best_penalty = selection_penalty

        diagnostics = {
            "evaluated": count,
            "eligible": int(eligible.sum().item()),
            "evaluated_nonlegacy": int((~legacy_flags).sum().item()),
            "eligible_nonlegacy": int((eligible & ~legacy_flags).sum().item()),
            "selected_nonlegacy": bool(
                best is not None and not bool(best.get("legacy_accepted", True))
            ),
            "ordinary_eligible": int(ordinary_eligible.sum().item()),
            "rescue_eligible": int(rescue_eligible.sum().item())
            if use_rescue
            else 0,
            "coverage_rescue_used": bool(
                use_rescue and bool(rescue_eligible.any().item())
            ),
            "below_frontend": int((finite & ~frontend_low).sum().item()),
            "above_frontend": int((finite & ~frontend_high).sum().item()),
            "graph_nonlocal": int((finite & ~graph_local).sum().item()),
        }
        return best, diagnostics

    def record_proposal(self, seed):
        if self.proposal_counts.numel():
            self.proposal_counts[int(seed)] += 1.0

    def record_pool(self, seed, evaluated, eligible, selected):
        if not self.evaluated_candidate_counts.numel():
            return
        seed = int(seed)
        self.evaluated_candidate_counts[seed] += float(evaluated)
        self.effective_candidate_counts[seed] += float(eligible)
        if selected:
            self.selected_counts[seed] += 1.0

    def seed_prior(self, device, dtype):
        """Effective-yield prior with bounded selected-coverage debt.

        Beta(1, 1) smoothing preserves cold-start exploration.  Mechanism
        relevance is intentionally not multiplied in because the three-dataset
        audit found it was not a stable predictor of frontend effectiveness.
        """
        self.require_ready()
        # A proposal is successful only when its effective pool is nonempty.
        # This window-level yield also penalizes nodes whose support or legacy
        # mechanism checks repeatedly produce no evaluable candidates.
        smoothed_yield = (self.selected_counts + 1.0) / (
            self.proposal_counts + 2.0
        )
        yield_weight = smoothed_yield.clamp_min(self.EPS).sqrt()
        valid_count = max(int(self.valid_nodes.sum().item()), 1)
        target = float(self.selected_counts.sum().item()) / valid_count
        debt = (target + 1.0) / (self.selected_counts + 1.0)
        debt = debt.clamp(self.coverage_debt_min, self.coverage_debt_max)
        weights = yield_weight * debt
        weights = torch.where(self.valid_nodes, weights, torch.zeros_like(weights))
        return weights.to(device=device, dtype=dtype)

    def running_diagnostics(self):
        return {
            "frontend_effective_proposal_counts": self.proposal_counts.tolist(),
            "frontend_effective_evaluated_candidate_counts": (
                self.evaluated_candidate_counts.tolist()
            ),
            "frontend_effective_candidate_counts": (
                self.effective_candidate_counts.tolist()
            ),
            "frontend_effective_selected_counts": self.selected_counts.tolist(),
        }
