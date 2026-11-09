import torch


class TemporalCounterfactualGenerator:
    """生成 OTR-CL 的正常正视图 ``W+`` 与窗口级时间反事实 ``Wc``。

    ``W+`` 对整个窗口施加共享弱仿射变换。``Wc`` 对全部传感器使用同一个
    时间块置换，从而优先改变时间状态而保持跨传感器关系。候选必须同时
    满足最小时间漂移和最大图漂移，未通过的样本回退为输入并由 valid mask
    排除，绝不生成伪造的 node mask。
    """

    def __init__(
        self,
        positive_scale_max=0.05,
        positive_shift_max=0.05,
        positive_max_temporal_drift=0.15,
        positive_max_graph_drift=1e-4,
        coarse_view_mode="synchronous_block_swap",
        coarse_block_ratio=0.3,
        coarse_candidate_trials=4,
        coarse_min_temporal_drift=0.25,
        coarse_max_graph_drift=0.05,
        eps=1e-6,
    ):
        self.positive_scale_max = float(positive_scale_max)
        self.positive_shift_max = float(positive_shift_max)
        self.positive_max_temporal_drift = float(
            positive_max_temporal_drift
        )
        self.positive_max_graph_drift = float(positive_max_graph_drift)
        self.coarse_view_mode = str(coarse_view_mode)
        self.coarse_block_ratio = float(coarse_block_ratio)
        self.coarse_candidate_trials = int(coarse_candidate_trials)
        self.coarse_min_temporal_drift = float(coarse_min_temporal_drift)
        self.coarse_max_graph_drift = float(coarse_max_graph_drift)
        self.eps = float(eps)
        self._validate_settings()
        self.last_diagnostics = {}

    @classmethod
    def from_config(cls, cfg):
        settings = cfg.SHARK.OTRCL
        return cls(
            positive_scale_max=settings.POSITIVE_SCALE_MAX,
            positive_shift_max=settings.POSITIVE_SHIFT_MAX,
            positive_max_temporal_drift=(
                settings.POSITIVE_MAX_TEMPORAL_DRIFT
            ),
            positive_max_graph_drift=settings.POSITIVE_MAX_GRAPH_DRIFT,
            coarse_view_mode=settings.COARSE_VIEW_MODE,
            coarse_block_ratio=settings.COARSE_BLOCK_RATIO,
            coarse_candidate_trials=settings.COARSE_CANDIDATE_TRIALS,
            coarse_min_temporal_drift=settings.COARSE_MIN_TEMPORAL_DRIFT,
            coarse_max_graph_drift=settings.COARSE_MAX_GRAPH_DRIFT,
            eps=settings.ORTHOGONALITY_EPS,
        )

    def _validate_settings(self):
        if self.coarse_view_mode != "synchronous_block_swap":
            raise ValueError(
                "Unsupported OTR-CL coarse view mode: "
                f"{self.coarse_view_mode}"
            )
        if not 0.0 <= self.positive_scale_max < 1.0:
            raise ValueError("positive_scale_max must be in [0, 1)")
        if self.positive_shift_max < 0.0:
            raise ValueError("positive_shift_max must be nonnegative")
        if not 0.0 < self.coarse_block_ratio <= 0.5:
            raise ValueError("coarse_block_ratio must be in (0, 0.5]")
        if self.coarse_candidate_trials <= 0:
            raise ValueError("coarse_candidate_trials must be positive")
        if self.coarse_min_temporal_drift < 0.0:
            raise ValueError("coarse_min_temporal_drift must be nonnegative")
        if self.coarse_max_graph_drift < 0.0:
            raise ValueError("coarse_max_graph_drift must be nonnegative")
        if self.eps <= 0.0:
            raise ValueError("OTR-CL orthogonality eps must be positive")

    @staticmethod
    def _validate_windows(windows):
        if not isinstance(windows, torch.Tensor):
            raise TypeError("windows must be a torch.Tensor")
        if windows.dim() != 3:
            raise ValueError("windows must have shape [B, T, N]")
        if windows.size(0) <= 0 or windows.size(1) < 4 or windows.size(2) <= 0:
            raise ValueError(
                "temporal counterfactuals require B>0, T>=4 and N>0"
            )
        if not torch.isfinite(windows).all():
            raise ValueError("windows must be finite")

    def _dense_pearson_graph(self, windows):
        nodes = windows.transpose(1, 2)
        nodes = nodes - nodes.mean(dim=-1, keepdim=True)
        nodes = nodes / nodes.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return torch.bmm(nodes, nodes.transpose(1, 2)).abs()

    def _graph(self, windows, graph_evaluator):
        if graph_evaluator is None:
            graph = self._dense_pearson_graph(windows)
        else:
            graph = graph_evaluator(windows)
        expected = (windows.size(0), windows.size(2), windows.size(2))
        if not isinstance(graph, torch.Tensor) or tuple(graph.shape) != expected:
            raise ValueError(
                "graph_evaluator must return [B, N, N], got "
                f"{getattr(graph, 'shape', None)}"
            )
        if not torch.isfinite(graph).all():
            raise ValueError("graph_evaluator returned non-finite values")
        return graph

    def _temporal_drift(self, anchor, candidate):
        delta = (candidate - anchor).square().mean(dim=(1, 2)).sqrt()
        centered = anchor - anchor.mean(dim=1, keepdim=True)
        reference = centered.square().mean(dim=(1, 2)).sqrt()
        return delta / reference.clamp_min(self.eps)

    def _graph_drift(self, reference, candidate):
        delta = (candidate - reference).abs().mean(dim=(1, 2))
        scale = reference.abs().mean(dim=(1, 2))
        return delta / scale.clamp_min(self.eps)

    def _positive_view(self, windows):
        batch_size = windows.size(0)
        shape = (batch_size, 1, 1)
        scale_delta = (
            2.0 * torch.rand(shape, device=windows.device) - 1.0
        ) * self.positive_scale_max
        scale = 1.0 + scale_delta.to(dtype=windows.dtype)
        window_scale = (
            windows - windows.mean(dim=1, keepdim=True)
        ).square().mean(dim=(1, 2), keepdim=True).sqrt()
        shift = (
            2.0 * torch.rand(shape, device=windows.device) - 1.0
        ).to(dtype=windows.dtype)
        shift = shift * self.positive_shift_max * window_scale
        return windows * scale + shift

    def _synchronous_block_swap(self, windows):
        batch_size, time_steps, _ = windows.shape
        block = max(1, int(round(time_steps * self.coarse_block_ratio)))
        block = min(block, time_steps // 2)
        permutations = torch.arange(
            time_steps,
            device=windows.device,
        ).repeat(batch_size, 1)
        for row in range(batch_size):
            first_max = time_steps - 2 * block
            first = int(
                torch.randint(
                    0,
                    first_max + 1,
                    (),
                    device=windows.device,
                ).item()
            )
            second = int(
                torch.randint(
                    first + block,
                    time_steps - block + 1,
                    (),
                    device=windows.device,
                ).item()
            )
            first_indices = permutations[row, first:first + block].clone()
            second_indices = permutations[row, second:second + block].clone()
            permutations[row, first:first + block] = second_indices
            permutations[row, second:second + block] = first_indices
        gather_index = permutations.unsqueeze(-1).expand_as(windows)
        return windows.gather(dim=1, index=gather_index)

    @torch.no_grad()
    def __call__(self, windows, graph_evaluator=None):
        self._validate_windows(windows)
        reference_graph = self._graph(windows, graph_evaluator)

        positive_view = self._positive_view(windows)
        positive_graph = self._graph(positive_view, graph_evaluator)
        positive_temporal_drift = self._temporal_drift(
            windows,
            positive_view,
        )
        positive_graph_drift = self._graph_drift(
            reference_graph,
            positive_graph,
        )
        positive_valid_mask = (
            positive_temporal_drift <= self.positive_max_temporal_drift
        ) & (
            positive_graph_drift <= self.positive_max_graph_drift
        )

        best_view = windows.clone()
        best_temporal_drift = windows.new_full(
            (windows.size(0),),
            -1.0,
        )
        best_graph_drift = windows.new_full(
            (windows.size(0),),
            float("inf"),
        )
        for _ in range(self.coarse_candidate_trials):
            candidate = self._synchronous_block_swap(windows)
            candidate_graph = self._graph(candidate, graph_evaluator)
            temporal_drift = self._temporal_drift(windows, candidate)
            graph_drift = self._graph_drift(reference_graph, candidate_graph)
            admissible = graph_drift <= self.coarse_max_graph_drift
            replace = admissible & (temporal_drift > best_temporal_drift)
            best_view = torch.where(
                replace[:, None, None],
                candidate,
                best_view,
            )
            best_temporal_drift = torch.where(
                replace,
                temporal_drift,
                best_temporal_drift,
            )
            best_graph_drift = torch.where(
                replace,
                graph_drift,
                best_graph_drift,
            )

        coarse_valid_mask = (
            best_temporal_drift >= self.coarse_min_temporal_drift
        ) & (
            best_graph_drift <= self.coarse_max_graph_drift
        )
        coarse_view = torch.where(
            coarse_valid_mask[:, None, None],
            best_view,
            windows,
        )
        self.last_diagnostics = {
            "positive_valid_count": int(positive_valid_mask.sum().item()),
            "coarse_valid_count": int(coarse_valid_mask.sum().item()),
            "batch_size": int(windows.size(0)),
            "positive_temporal_drift": positive_temporal_drift.detach(),
            "positive_graph_drift": positive_graph_drift.detach(),
            "coarse_temporal_drift": best_temporal_drift.detach(),
            "coarse_graph_drift": best_graph_drift.detach(),
        }
        return {
            "positive_view": positive_view,
            "positive_valid_mask": positive_valid_mask,
            "coarse_view": coarse_view,
            "coarse_valid_mask": coarse_valid_mask,
            "diagnostics": self.last_diagnostics,
        }
