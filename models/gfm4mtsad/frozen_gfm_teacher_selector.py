from __future__ import annotations

import torch


class FrozenGFMTeacherSelector:
    """Select legacy-valid counterfactuals with a frozen predecessor GFM.

    The teacher is never optimized and only admits candidates whose native
    node-score gain is positive at the intervened node and larger than every
    competing node gain.  Test arrays and labels are not part of this path.
    """

    def __init__(self, teacher, valid_nodes):
        self.teacher = teacher
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.valid_nodes = valid_nodes.detach().bool().cpu()
        self._reset_running_state(int(self.valid_nodes.numel()))

    def _reset_running_state(self, node_count):
        self.proposal_counts = torch.zeros(node_count, dtype=torch.float64)
        self.evaluated_candidate_counts = torch.zeros(
            node_count, dtype=torch.float64
        )
        self.effective_candidate_counts = torch.zeros(
            node_count, dtype=torch.float64
        )
        self.selected_counts = torch.zeros(node_count, dtype=torch.float64)
        self.window_count = 0
        self.selected_window_count = 0

    def require_ready(self):
        if not getattr(self.teacher, "_bridge_initialized", False):
            raise RuntimeError(
                "Frozen GFM teacher must restore an initialized Token Bridge"
            )
        if not bool(self.valid_nodes.any()):
            raise RuntimeError("Frozen GFM teacher received no valid nodes")

    @torch.no_grad()
    def encode_original(self, windows):
        self.require_ready()
        outputs = self.teacher(windows.float(), return_dict=True)
        scores = outputs["native_node_scores"].detach()
        if scores.ndim != 2:
            raise RuntimeError(
                "Frozen GFM teacher native scores must have shape [B, N]"
            )
        return scores.unsqueeze(-1)

    def record_proposal(self, seed):
        self.proposal_counts[int(seed)] += 1.0

    def record_pool(self, seed, evaluated, eligible, selected):
        seed = int(seed)
        self.evaluated_candidate_counts[seed] += int(evaluated)
        self.effective_candidate_counts[seed] += int(eligible)
        self.selected_counts[seed] += int(bool(selected))

    @torch.no_grad()
    def select_batch(self, windows, records, chunk_size=512):
        """Vectorized selection for all candidate pools in one train batch."""
        self.require_ready()
        self.window_count += int(windows.size(0))
        eligible_records = [
            record
            for record in records
            if bool(record.get("accepted"))
            and bool(record.get("context_valid"))
        ]
        if not eligible_records:
            return {}
        original_scores = self.teacher(
            windows.float(),
            return_dict=True,
        )["native_node_scores"].detach()
        best = {}
        chunk_size = max(int(chunk_size), 1)
        for begin in range(0, len(eligible_records), chunk_size):
            chunk = eligible_records[begin : begin + chunk_size]
            candidate_windows = []
            for record in chunk:
                batch_index = int(record["batch_index"])
                seed = int(record["seed"])
                candidate = windows[batch_index].detach().clone()
                candidate[record["start"] : record["end"], seed] = record[
                    "values"
                ].to(device=candidate.device, dtype=candidate.dtype)
                candidate_windows.append(candidate)
            candidate_scores = self.teacher(
                torch.stack(candidate_windows).float(),
                return_dict=True,
            )["native_node_scores"].detach()
            for row, record in enumerate(chunk):
                batch_index = int(record["batch_index"])
                seed = int(record["seed"])
                gain = candidate_scores[row] - original_scores[batch_index]
                direct_gain = gain[seed]
                competitor = gain.clone()
                competitor[seed] = float("-inf")
                margin = direct_gain - competitor.max()
                if not (
                    bool(torch.isfinite(direct_gain).item())
                    and bool(torch.isfinite(margin).item())
                    and float(direct_gain.item()) > 0.0
                    and float(margin.item()) > 0.0
                ):
                    continue
                rank = (
                    float(margin.item()),
                    float(direct_gain.item()),
                    -float(record["perturbation_rms"]),
                )
                if batch_index not in best or rank > best[batch_index][0]:
                    selected = dict(record)
                    selected["teacher_direct_gain"] = float(direct_gain.item())
                    selected["teacher_direction_margin"] = float(margin.item())
                    best[batch_index] = (rank, selected)
        selected = {index: value[1] for index, value in best.items()}
        self.selected_window_count += len(selected)
        for record in records:
            self.record_proposal(int(record["seed"]))
        for batch_index, record in selected.items():
            seed = int(record["seed"])
            evaluated = sum(
                int(candidate["batch_index"]) == batch_index
                for candidate in eligible_records
            )
            self.record_pool(seed, evaluated, 1, True)
        return selected

    @torch.no_grad()
    def select(
        self,
        original_window,
        original_reference,
        original_adjacency,
        seed,
        records,
    ):
        del original_adjacency
        self.require_ready()
        if not records:
            return None, self._empty_diagnostics()
        seed = int(seed)
        candidate_windows = []
        for record in records:
            candidate = original_window.detach().clone()
            candidate[record["start"] : record["end"], seed] = record[
                "values"
            ].to(device=candidate.device, dtype=candidate.dtype)
            candidate_windows.append(candidate)
        candidate_batch = torch.stack(candidate_windows).float()
        candidate_scores = self.teacher(
            candidate_batch,
            return_dict=True,
        )["native_node_scores"].detach()
        original_scores = original_reference.squeeze(-1).detach()
        gains = candidate_scores - original_scores.unsqueeze(0)
        direct_gain = gains[:, seed]
        competitor = gains.clone()
        competitor[:, seed] = float("-inf")
        max_other_gain = competitor.max(dim=1).values
        direction_margin = direct_gain - max_other_gain
        eligible = (
            torch.isfinite(direct_gain)
            & torch.isfinite(direction_margin)
            & (direct_gain > 0.0)
            & (direction_margin > 0.0)
        )
        eligible_ids = torch.nonzero(eligible, as_tuple=False).flatten()
        if eligible_ids.numel() == 0:
            diagnostics = self._empty_diagnostics()
            diagnostics["evaluated"] = len(records)
            diagnostics["nonpositive_direction"] = len(records)
            return None, diagnostics
        best_local = max(
            eligible_ids.tolist(),
            key=lambda index: (
                float(direction_margin[index].item()),
                float(direct_gain[index].item()),
                -float(records[index]["delta_rms"].item()),
            ),
        )
        selected = records[best_local]
        selected["teacher_direct_gain"] = direct_gain[best_local]
        selected["teacher_direction_margin"] = direction_margin[best_local]
        # Reuse the existing selector accounting fields without changing loss
        # semantics; these values are diagnostic only.
        selected["frontend_direct_drift"] = direct_gain[best_local]
        selected["graph_nonincident_drift"] = direction_margin[best_local]
        diagnostics = self._empty_diagnostics()
        diagnostics.update(
            {
                "evaluated": len(records),
                "eligible": int(eligible.sum().item()),
                "nonpositive_direction": int((~eligible).sum().item()),
            }
        )
        return selected, diagnostics

    @staticmethod
    def _empty_diagnostics():
        return {
            "evaluated": 0,
            "eligible": 0,
            "below_frontend": 0,
            "above_frontend": 0,
            "graph_nonlocal": 0,
            "nonpositive_direction": 0,
            "coverage_rescue_used": False,
            "eligible_nonlegacy": 0,
        }

    def running_diagnostics(self):
        return {
            "frozen_teacher_proposal_counts": self.proposal_counts.tolist(),
            "frozen_teacher_evaluated_counts": (
                self.evaluated_candidate_counts.tolist()
            ),
            "frozen_teacher_eligible_counts": (
                self.effective_candidate_counts.tolist()
            ),
            "frozen_teacher_selected_counts": self.selected_counts.tolist(),
            "window_count": int(self.window_count),
            "selected_window_count": int(self.selected_window_count),
            "selected_window_ratio": (
                self.selected_window_count / max(self.window_count, 1)
            ),
            "uses_test_labels": False,
            "teacher_trainable": False,
        }
