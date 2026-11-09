"""GFM 相对局部时间内容的关系残差。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


RELATION_PATH_PARAMETER_PREFIXES = (
    "gfm_adapter.",
    "information_residual.",
)


@torch.no_grad()
def snapshot_relation_path_parameters(model):
    """记录关系对比应直接更新的共享适配参数。"""
    snapshot = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name.startswith(RELATION_PATH_PARAMETER_PREFIXES)
    }
    if not snapshot:
        raise RuntimeError("关系路径没有可训练的共享适配参数")
    return snapshot


@torch.no_grad()
def relation_path_parameter_drift(model, initial_snapshot):
    """计算当前模型相对训练前快照的关系路径参数漂移。"""
    current = {
        name: parameter.detach().float().cpu()
        for name, parameter in model.named_parameters()
        if name in initial_snapshot
    }
    if set(current) != set(initial_snapshot):
        missing = sorted(set(initial_snapshot).difference(current))
        extra = sorted(set(current).difference(initial_snapshot))
        raise RuntimeError(
            f"关系路径参数集合发生变化: missing={missing}, extra={extra}"
        )

    total_squared = 0.0
    reference_squared = 0.0
    max_absolute = 0.0
    changed_count = 0
    per_parameter = {}
    for name in sorted(initial_snapshot):
        reference = initial_snapshot[name]
        value = current[name]
        delta = value - reference
        squared = float(delta.square().sum().item())
        reference_norm_squared = float(reference.square().sum().item())
        l2 = squared ** 0.5
        reference_l2 = reference_norm_squared ** 0.5
        absolute = float(delta.abs().max().item()) if delta.numel() else 0.0
        if l2 > 0.0:
            changed_count += 1
        total_squared += squared
        reference_squared += reference_norm_squared
        max_absolute = max(max_absolute, absolute)
        per_parameter[name] = {
            "l2": l2,
            "relative_l2": l2 / max(reference_l2, 1e-12),
            "max_absolute": absolute,
        }

    total_l2 = total_squared ** 0.5
    return {
        "parameter_count": len(initial_snapshot),
        "changed_parameter_count": changed_count,
        "total_l2": total_l2,
        "relative_l2": total_l2 / max(reference_squared ** 0.5, 1e-12),
        "max_absolute": max_absolute,
        "per_parameter": per_parameter,
    }


class TimeRelationResidual(nn.Module):
    """用正常数据拟合时间内容基线，再提取 GFM 关系增量。"""

    def __init__(self, source_dim, embedding_dim, ridge=1e-3):
        super().__init__()
        self.source_dim = int(source_dim)
        self.embedding_dim = int(embedding_dim)
        self.ridge = max(float(ridge), 1e-8)
        self.proj = nn.Linear(self.source_dim * 2, self.embedding_dim)
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def fit_from_moments(
        self,
        count,
        feature_sum,
        feature_cross,
        target_sum,
        feature_target,
    ):
        count = int(count)
        if count < 2:
            raise ValueError("时间关系投影至少需要两个正常节点样本")
        feature_sum = feature_sum.detach().double().cpu()
        feature_cross = feature_cross.detach().double().cpu()
        target_sum = target_sum.detach().double().cpu()
        feature_target = feature_target.detach().double().cpu()
        dim = self.source_dim * 2
        if tuple(feature_sum.shape) != (dim,):
            raise ValueError("时间关系投影 feature_sum 形状错误")
        if tuple(feature_cross.shape) != (dim, dim):
            raise ValueError("时间关系投影 feature_cross 形状错误")
        if tuple(target_sum.shape) != (self.embedding_dim,):
            raise ValueError("时间关系投影 target_sum 形状错误")
        if tuple(feature_target.shape) != (dim, self.embedding_dim):
            raise ValueError("时间关系投影 feature_target 形状错误")
        mean_x = feature_sum / float(count)
        mean_y = target_sum / float(count)
        covariance = feature_cross / float(count) - torch.outer(mean_x, mean_x)
        cross = feature_target / float(count) - torch.outer(mean_x, mean_y)
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        system = covariance + self.ridge * torch.eye(dim, dtype=torch.float64)
        weight = torch.linalg.solve(system, cross)
        bias = mean_y - mean_x @ weight
        self.proj.weight.copy_(weight.transpose(0, 1).to(self.proj.weight))
        self.proj.bias.copy_(bias.to(self.proj.bias))
        self.fitted.fill_(True)

    def forward(self, gfm_node_embeddings, trend_features, seasonal_features):
        if not bool(self.fitted.item()):
            raise RuntimeError("时间关系投影必须先用正常数据拟合")
        content = torch.cat((trend_features, seasonal_features), dim=-1).float()
        baseline = self.proj(content)
        gfm = F.normalize(gfm_node_embeddings.float(), p=2, dim=-1)
        baseline = F.normalize(baseline, p=2, dim=-1)
        residual = gfm - baseline.detach()
        return {
            "relation_residual": residual,
            "time_content_baseline": baseline,
            "relation_residual_rms": residual.square().mean(dim=-1).sqrt(),
        }
