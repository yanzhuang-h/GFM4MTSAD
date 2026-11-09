"""机制匹配的时间—图反事实工具。

本模块只负责生成和审计成对反事实，不改变 GFM4MTSAD 默认训练路径。
保持视图与破坏视图共享目标节点轨迹和扰动能量，二者的主要差异是
Graph-VAR 父节点响应的时间滞后是否符合正常机制。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch


def moving_average_decomposition(
    windows: torch.Tensor,
    kernel_size: int = 25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 ``seasonal, trend``，输入形状为 ``[B, T, N]``。"""
    if windows.dim() != 3:
        raise ValueError(
            "时间序列窗口必须是 [B, T, N]，"
            f"实际为 {tuple(windows.shape)}"
        )
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("移动平均窗口必须为正奇数")
    pad = (kernel_size - 1) // 2
    # 使用与 DLinear MovingAvg 相同的时间维边界复制，支持窗口短于核大小。
    front = windows[:, 0:1, :].repeat(1, pad, 1)
    end = windows[:, -1:, :].repeat(1, pad, 1)
    padded = torch.cat((front, windows, end), dim=1)
    trend = torch.nn.functional.avg_pool1d(
        padded.transpose(1, 2),
        kernel_size=kernel_size,
        stride=1,
    ).transpose(1, 2)
    seasonal = windows - trend
    return seasonal, trend


def _safe_rms(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return values.square().mean(dim=1).sqrt().clamp_min(eps)


def temporal_match_diagnostics(
    positive: torch.Tensor,
    negative: torch.Tensor,
    kernel_size: int = 25,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """计算保持/破坏视图的时间内容匹配量。

    返回每个样本的 trend、seasonal、RMS 和边界差异。该函数不读取标签，
    可用于训练前的反事实有效性门控。
    """
    if positive.shape != negative.shape or positive.dim() != 3:
        raise ValueError(
            "保持视图和破坏视图必须共享 [B, T, N] 形状："
            f"positive={tuple(positive.shape)}, negative={tuple(negative.shape)}"
        )
    positive_seasonal, positive_trend = moving_average_decomposition(
        positive,
        kernel_size=kernel_size,
    )
    negative_seasonal, negative_trend = moving_average_decomposition(
        negative,
        kernel_size=kernel_size,
    )
    delta = positive - negative
    trend_delta = positive_trend - negative_trend
    seasonal_delta = positive_seasonal - negative_seasonal
    boundary = torch.stack(
        (
            delta[:, 0].abs().mean(dim=-1),
            delta[:, -1].abs().mean(dim=-1),
        ),
        dim=1,
    ).amax(dim=1)
    return {
        "trend_rms": _safe_rms(trend_delta, eps=eps).mean(dim=-1),
        "seasonal_rms": _safe_rms(seasonal_delta, eps=eps).mean(dim=-1),
        "value_rms": _safe_rms(delta, eps=eps).mean(dim=-1),
        "boundary_abs": boundary,
        "positive_trend_rms": _safe_rms(positive_trend, eps=eps).mean(dim=-1),
        "negative_trend_rms": _safe_rms(negative_trend, eps=eps).mean(dim=-1),
        "positive_seasonal_rms": _safe_rms(
            positive_seasonal,
            eps=eps,
        ).mean(dim=-1),
        "negative_seasonal_rms": _safe_rms(
            negative_seasonal,
            eps=eps,
        ).mean(dim=-1),
    }


def temporal_pair_fairness(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    kernel_size: int = 25,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """比较两个反事实相对事实窗口的时间内容能量。"""
    if anchor.shape != positive.shape or anchor.shape != negative.shape:
        raise ValueError("事实与两个反事实视图必须共享 [B, T, N] 形状")
    positive_delta = positive - anchor
    negative_delta = negative - anchor
    positive_seasonal, positive_trend = moving_average_decomposition(
        positive_delta,
        kernel_size=kernel_size,
    )
    negative_seasonal, negative_trend = moving_average_decomposition(
        negative_delta,
        kernel_size=kernel_size,
    )

    def energy(values):
        return values.square().mean(dim=(1, 2)).sqrt()

    def relative_gap(left, right):
        return (left - right).abs() / torch.maximum(left, right).clamp_min(eps)

    pos_value = energy(positive_delta)
    neg_value = energy(negative_delta)
    pos_trend = energy(positive_trend)
    neg_trend = energy(negative_trend)
    pos_seasonal = energy(positive_seasonal)
    neg_seasonal = energy(negative_seasonal)
    return {
        "value_energy_positive": pos_value,
        "value_energy_negative": neg_value,
        "value_relative_gap": relative_gap(pos_value, neg_value),
        "trend_energy_positive": pos_trend,
        "trend_energy_negative": neg_trend,
        "trend_relative_gap": relative_gap(pos_trend, neg_trend),
        "seasonal_energy_positive": pos_seasonal,
        "seasonal_energy_negative": neg_seasonal,
        "seasonal_relative_gap": relative_gap(pos_seasonal, neg_seasonal),
    }


class MechanismMatchedPairBuilder:
    """构造 Graph-VAR 机制保持/破坏的滞后匹配反事实对。

    对每个样本选择一个目标节点和一个最强父节点。保持视图在正确滞后
    位置注入父节点扰动，破坏视图在错误滞后位置注入同一扰动；两者随后
    使用保持视图的目标节点轨迹，因此目标局部时间内容完全一致。
    """

    def __init__(
        self,
        mechanism,
        lag_offset: int = 1,
        waveform_cycles: float = 2.0,
    ):
        self.mechanism = mechanism
        self.lag_offset = max(int(lag_offset), 1)
        self.waveform_cycles = max(float(waveform_cycles), 1.0)

    @staticmethod
    def _strongest_parent_lag(state, seed: int):
        parent_index = state["parent_index"][seed]
        parent_coef = state["parent_coef"][seed]
        strengths = parent_coef.abs()
        strengths = strengths.masked_fill(parent_index.unsqueeze(-1) < 0, -1.0)
        flat = strengths.reshape(-1)
        index = int(flat.argmax().item())
        slot, lag_index = divmod(index, strengths.size(-1))
        parent = int(parent_index[slot].item())
        if parent < 0 or float(flat[index].item()) <= 0.0:
            return None
        return parent, slot, lag_index + 1

    def _waveform(self, length: int, device, dtype, strength: float):
        phase = torch.linspace(
            0.0,
            1.0,
            steps=length,
            device=device,
            dtype=dtype,
        )
        trend = phase - 0.5
        seasonal = torch.sin(
            2.0 * math.pi * self.waveform_cycles * phase
        )
        waveform = 0.5 * trend + 0.5 * seasonal
        waveform = waveform / waveform.square().mean().sqrt().clamp_min(1e-6)
        return waveform * float(strength)

    def build(
        self,
        window: torch.Tensor,
        seed: int,
        start: int,
        end: int,
        strength: float = 0.5,
        wrong_lag: Optional[int] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """返回单窗口的 ``positive``、``negative`` 及掩码，失败返回 ``None``。"""
        if window.dim() != 2:
            raise ValueError("单窗口必须是 [T, N]")
        state = self.mechanism._state(window.device, window.dtype)
        if state is None or not self.mechanism.fitted:
            return None
        if seed < 0 or seed >= window.size(1):
            return None
        selected = self._strongest_parent_lag(state, seed)
        if selected is None:
            return None
        parent, slot, good_lag = selected
        if wrong_lag is None:
            wrong_lag = good_lag + self.lag_offset
            if wrong_lag > self.mechanism.lags:
                wrong_lag = max(1, good_lag - self.lag_offset)
        wrong_lag = int(wrong_lag)
        if wrong_lag < 1 or wrong_lag > self.mechanism.lags or wrong_lag == good_lag:
            return None

        start = max(int(start), self.mechanism.lags + 1)
        end = min(int(end), window.size(0))
        if end - start < 4:
            return None
        edit_target_positions = torch.arange(start, end, device=window.device)
        rollout_positions = torch.arange(start, window.size(0), device=window.device)
        good_parent_positions = edit_target_positions - good_lag
        wrong_parent_positions = edit_target_positions - wrong_lag
        if good_parent_positions.min() < 0 or wrong_parent_positions.min() < 0:
            return None

        standardized = (
            window - state["mean"].view(1, -1)
        ) / state["scale"].view(1, -1).clamp_min(self.mechanism.min_node_scale)
        original_prediction = self.mechanism._predict_seed(
            standardized.unsqueeze(0),
            seed,
            rollout_positions,
            state,
        )[0]
        innovations = standardized[rollout_positions, seed] - original_prediction
        delta = self._waveform(
            edit_target_positions.numel(),
            window.device,
            window.dtype,
            strength,
        )

        positive_z = standardized.clone()
        negative_z = standardized.clone()
        positive_z[good_parent_positions, parent] += delta
        negative_z[wrong_parent_positions, parent] += delta

        # 顺序重算目标节点，保留同一创新项，使保持视图遵守正常 Graph-VAR 方程。
        positive_target = []
        for offset, time_index in enumerate(rollout_positions.tolist()):
            position = rollout_positions.new_tensor([time_index])
            prediction = self.mechanism._predict_seed(
                positive_z.unsqueeze(0),
                seed,
                position,
                state,
            )[0, 0]
            value = prediction + innovations[offset]
            positive_z[time_index, seed] = value
            positive_target.append(value)
        positive_target = torch.stack(positive_target)
        negative_z[rollout_positions, seed] = positive_target

        positive = positive_z * state["scale"].view(1, -1) + state["mean"].view(1, -1)
        negative = negative_z * state["scale"].view(1, -1) + state["mean"].view(1, -1)
        direct_mask = torch.zeros(window.size(1), device=window.device, dtype=torch.bool)
        direct_mask[seed] = True
        context_mask = torch.zeros_like(direct_mask)
        context_mask[parent] = True
        return {
            "positive": positive,
            "negative": negative,
            "direct_mask": direct_mask,
            "context_mask": context_mask,
            "seed": torch.tensor(seed, device=window.device),
            "parent": torch.tensor(parent, device=window.device),
            "good_lag": torch.tensor(good_lag, device=window.device),
            "wrong_lag": torch.tensor(wrong_lag, device=window.device),
            "target_shared_max_abs": (positive[:, seed] - negative[:, seed]).abs().max(),
            "parent_delta_rms": delta.square().mean().sqrt(),
            "positive_parent_delta_rms": (
                positive_z[:, parent] - standardized[:, parent]
            ).square().mean().sqrt(),
            "negative_parent_delta_rms": (
                negative_z[:, parent] - standardized[:, parent]
            ).square().mean().sqrt(),
        }
