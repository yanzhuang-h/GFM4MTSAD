import math

import torch
import torch.nn.functional as F


def _weighted_sample_mean(per_sample_loss, sample_weights, zero_reference):
    if per_sample_loss.numel() == 0:
        return zero_reference.sum() * 0.0
    if sample_weights is None:
        return per_sample_loss.mean()

    weights = sample_weights.detach().to(
        device=per_sample_loss.device,
        dtype=per_sample_loss.dtype,
    ).reshape(-1)
    if weights.numel() != per_sample_loss.numel():
        raise ValueError(
            "sample_weights must match the per-sample loss count: "
            f"weights={weights.numel()}, losses={per_sample_loss.numel()}"
        )
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("sample_weights must be finite and non-negative")
    weight_sum = weights.sum()
    if float(weight_sum.detach()) <= 0.0:
        return zero_reference.sum() * 0.0
    return (per_sample_loss * weights).sum() / weight_sum


def carots_style_contrastive_loss(
    anchor,
    positive,
    negative,
    temperature,
    sample_weights=None,
):
    if anchor.size(0) < 1:
        return anchor.sum() * 0.0

    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)
    pos_sim = torch.sum(anchor * positive, dim=1, keepdim=True)
    neg_sim = torch.sum(anchor * negative, dim=1, keepdim=True)
    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    return _weighted_sample_mean(per_sample, sample_weights, anchor)


def temporal_regime_contrastive_loss(
    anchor,
    positive,
    coarse,
    temperature,
    sample_weights=None,
):
    """窗口级分量感知时间状态对比。

    正样本必须保持正常时间状态，coarse 负样本只表达全局时间漂移；该函数
    不读取任何节点标签或节点 mask。
    """
    views = (anchor, positive, coarse)
    if any(view.dim() != 2 for view in views):
        raise ValueError(
            "Temporal regime contrast expects three [B, D] embeddings"
        )
    if anchor.shape != positive.shape or anchor.shape != coarse.shape:
        raise ValueError(
            "Temporal regime views must have identical shape: "
            f"anchor={tuple(anchor.shape)}, positive={tuple(positive.shape)}, "
            f"coarse={tuple(coarse.shape)}"
        )
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("Temporal regime temperature must be positive")
    total = carots_style_contrastive_loss(
        anchor,
        positive,
        coarse,
        temperature,
        sample_weights=sample_weights,
    )
    normalized = [F.normalize(view, p=2, dim=1) for view in views]
    positive_similarity = (normalized[0] * normalized[1]).sum(dim=1)
    coarse_similarity = (normalized[0] * normalized[2]).sum(dim=1)
    return {
        "total": total,
        "positive_similarity": positive_similarity.detach().mean(),
        "coarse_similarity": coarse_similarity.detach().mean(),
        "similarity_gap": (
            positive_similarity - coarse_similarity
        ).detach().mean(),
        "sample_count": torch.as_tensor(
            anchor.size(0),
            device=anchor.device,
            dtype=torch.long,
        ),
    }


def decoupled_local_global_contrastive_loss(
    anchor_graph,
    positive_graph,
    negative_graph,
    anchor_nodes,
    positive_nodes,
    negative_nodes,
    direct_node_mask,
    global_temperature=0.2,
    local_temperature=0.2,
    global_margin=0.0,
    local_margin=0.0,
    local_loss_weight=0.0,
    sample_weights=None,
):
    """解耦图级窗口目标与直接干预节点目标。

    全局分量只读取图嵌入，局部分量只读取 ``direct_node_mask`` 标记的
    节点行。局部权重为零时，函数沿用现有二元图级 InfoNCE 的逐样本数值
    路径，保证损失值和图嵌入梯度与
    :func:`carots_style_contrastive_loss` 严格等价。
    """
    graph_views = (anchor_graph, positive_graph, negative_graph)
    if any(view.ndim != 2 for view in graph_views):
        raise ValueError("解耦对比的图级三视图必须具有 [B, D] 形状")
    if (
        anchor_graph.shape != positive_graph.shape
        or anchor_graph.shape != negative_graph.shape
    ):
        raise ValueError(
            "解耦对比的图级三视图形状不一致: "
            f"anchor={tuple(anchor_graph.shape)}, "
            f"positive={tuple(positive_graph.shape)}, "
            f"negative={tuple(negative_graph.shape)}"
        )

    node_views = (anchor_nodes, positive_nodes, negative_nodes)
    if any(view.ndim != 3 for view in node_views):
        raise ValueError("解耦对比的节点级三视图必须具有 [B, N, D] 形状")
    if (
        anchor_nodes.shape != positive_nodes.shape
        or anchor_nodes.shape != negative_nodes.shape
    ):
        raise ValueError(
            "解耦对比的节点级三视图形状不一致: "
            f"anchor={tuple(anchor_nodes.shape)}, "
            f"positive={tuple(positive_nodes.shape)}, "
            f"negative={tuple(negative_nodes.shape)}"
        )
    if anchor_nodes.size(0) != anchor_graph.size(0):
        raise ValueError(
            "解耦对比的图级与节点级批大小不一致: "
            f"graph={anchor_graph.size(0)}, nodes={anchor_nodes.size(0)}"
        )
    if direct_node_mask.shape != anchor_nodes.shape[:2]:
        raise ValueError(
            "direct_node_mask 必须具有 [B, N] 形状: "
            f"mask={tuple(direct_node_mask.shape)}, "
            f"nodes={tuple(anchor_nodes.shape[:2])}"
        )

    global_temperature = float(global_temperature)
    local_temperature = float(local_temperature)
    global_margin = float(global_margin)
    local_margin = float(local_margin)
    local_loss_weight = float(local_loss_weight)
    if not math.isfinite(global_temperature) or global_temperature <= 0.0:
        raise ValueError("GLOBAL_TEMPERATURE 必须是有限正数")
    if not math.isfinite(local_temperature) or local_temperature <= 0.0:
        raise ValueError("LOCAL_TEMPERATURE 必须是有限正数")
    if not math.isfinite(global_margin):
        raise ValueError("GLOBAL_MARGIN 必须是有限数")
    if not math.isfinite(local_margin):
        raise ValueError("LOCAL_MARGIN 必须是有限数")
    if not math.isfinite(local_loss_weight) or local_loss_weight < 0.0:
        raise ValueError("LOCAL_LOSS_WEIGHT 必须是有限非负数")

    if sample_weights is not None:
        weights = sample_weights.detach().to(
            device=anchor_graph.device,
            dtype=anchor_graph.dtype,
        ).reshape(-1)
        if weights.numel() != anchor_graph.size(0):
            raise ValueError(
                "sample_weights 必须与批大小一致: "
                f"weights={weights.numel()}, batch={anchor_graph.size(0)}"
            )
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("sample_weights 必须为有限非负数")
    else:
        weights = None

    anchor_graph_normalized = F.normalize(anchor_graph, p=2, dim=1)
    positive_graph_normalized = F.normalize(positive_graph, p=2, dim=1)
    negative_graph_normalized = F.normalize(negative_graph, p=2, dim=1)
    positive_similarity = torch.sum(
        anchor_graph_normalized * positive_graph_normalized,
        dim=1,
        keepdim=True,
    )
    negative_similarity = torch.sum(
        anchor_graph_normalized * negative_graph_normalized,
        dim=1,
        keepdim=True,
    )
    global_gap_per_sample = (
        positive_similarity - negative_similarity
    ).squeeze(1)

    if global_margin == 0.0:
        # 与现有二元图级 InfoNCE 使用完全相同的算子顺序，避免仅因
        # softplus/log-softmax 的浮点实现差异破坏等价性契约。
        logits = torch.cat(
            [positive_similarity, negative_similarity],
            dim=1,
        ) / global_temperature
        labels = torch.zeros(
            anchor_graph.size(0),
            dtype=torch.long,
            device=anchor_graph.device,
        )
        per_sample_global = F.cross_entropy(
            logits,
            labels,
            reduction="none",
        )
    else:
        per_sample_global = F.softplus(
            (global_margin - global_gap_per_sample) / global_temperature
        )
    if not torch.isfinite(per_sample_global).all():
        raise FloatingPointError("解耦对比的全局分量产生了非有限损失")
    global_loss = _weighted_sample_mean(
        per_sample_global,
        weights,
        anchor_graph_normalized,
    )
    global_weight_sum = (
        anchor_graph.new_tensor(float(anchor_graph.size(0)))
        if weights is None
        else weights.sum().detach()
    )

    direct_mask = direct_node_mask.bool().detach()
    active_local_samples = direct_mask.any(dim=1)
    if weights is not None:
        active_local_samples = active_local_samples & (weights > 0)
    has_local_samples = bool(active_local_samples.any())
    local_measured = bool(local_loss_weight > 0.0 and has_local_samples)

    if local_loss_weight == 0.0 or not has_local_samples:
        # 使用空节点切片保持 autograd 连接，但不读取任何真实节点值；
        # 即使非 direct 行含有非有限值，也不能污染零局部分量。
        local_loss = (
            anchor_nodes[..., :0].sum()
            + positive_nodes[..., :0].sum()
            + negative_nodes[..., :0].sum()
        ) * 0.0
        direct_gap = local_loss.detach().clone()
        weighted_local_loss = local_loss
        local_weight_sum = local_loss.detach().clone()
    else:
        active_direct_mask = direct_mask & active_local_samples.unsqueeze(1)
        active_direct_counts = active_direct_mask.sum(dim=1)
        # 先索引再归一化，局部分量的前向图中不读取任何非 direct 行。
        anchor_direct = F.normalize(
            anchor_nodes[active_direct_mask],
            p=2,
            dim=-1,
        )
        positive_direct = F.normalize(
            positive_nodes[active_direct_mask],
            p=2,
            dim=-1,
        )
        negative_direct = F.normalize(
            negative_nodes[active_direct_mask],
            p=2,
            dim=-1,
        )
        direct_node_gap = (anchor_direct * positive_direct).sum(dim=-1) - (
            anchor_direct * negative_direct
        ).sum(dim=-1)
        direct_sample_indices = torch.arange(
            anchor_nodes.size(0),
            device=anchor_nodes.device,
        ).unsqueeze(1).expand_as(active_direct_mask)[active_direct_mask]
        direct_gap_sums = direct_node_gap.new_zeros(anchor_nodes.size(0))
        direct_gap_sums = direct_gap_sums.scatter_add(
            0,
            direct_sample_indices,
            direct_node_gap,
        )
        per_sample_direct_gap = direct_gap_sums / active_direct_counts.clamp_min(
            1
        ).to(direct_node_gap.dtype)
        active_direct_gap = per_sample_direct_gap[active_local_samples]
        per_sample_local = F.softplus(
            (local_margin - active_direct_gap) / local_temperature
        )
        if not torch.isfinite(per_sample_local).all():
            raise FloatingPointError("解耦对比的局部分量产生了非有限损失")
        active_weights = (
            weights[active_local_samples] if weights is not None else None
        )
        local_weight_sum = (
            anchor_direct.new_tensor(float(active_direct_gap.numel()))
            if active_weights is None
            else active_weights.sum().detach()
        )
        local_loss = _weighted_sample_mean(
            per_sample_local,
            active_weights,
            anchor_direct,
        )
        direct_gap = _weighted_sample_mean(
            active_direct_gap,
            active_weights,
            anchor_direct,
        )
        weighted_local_loss = local_loss_weight * local_loss

    # lambda=0 时直接返回 global_loss，不通过“加零”构造新的 autograd 图。
    total = (
        global_loss
        if local_loss_weight == 0.0
        else global_loss + weighted_local_loss
    )
    return {
        "total": total,
        "global": global_loss,
        "local": local_loss,
        "weighted_local": weighted_local_loss,
        "global_gap": _weighted_sample_mean(
            global_gap_per_sample,
            weights,
            anchor_graph_normalized,
        ),
        "direct_gap": direct_gap,
        "global_weight_sum": global_weight_sum,
        "local_weight_sum": local_weight_sum,
        "local_measured": local_measured,
        "sample_count": torch.tensor(
            anchor_graph.size(0),
            device=anchor_graph.device,
            dtype=torch.long,
        ),
        "local_sample_count": active_local_samples.sum().detach(),
        "direct_node_count": (
            direct_mask & active_local_samples.unsqueeze(1)
        ).sum().detach(),
    }


def native_evidence_mechanism_contrastive_loss(
    anchor_graph,
    positive_graph,
    negative_graph,
    anchor_node_evidence,
    positive_node_evidence,
    negative_node_evidence,
    direct_node_mask,
    global_temperature=0.2,
    local_temperature=0.2,
    pair_margin=0.0,
    local_loss_weight=0.0,
    sample_weights=None,
    eps=1e-6,
    min_reference_scale=0.1,
):
    """在后端原生节点异常证据空间对比 Graph-VAR 机制反事实。

    支持节点异常证据的 GFM 后端必须显式遵守统一的 ``node_scores`` 契约：
    与传感器一一对齐、有限、非负且数值越大表示越异常。损失先按锚点/正视图
    的停止梯度正常水平无量纲化并取 log1p，再做跨节点尺度归一化；因此
    AnomalyGFM 的 prompt 组合分数与 OWLEYE 的 pattern 距离可使用同一公式。
    """
    anchor_evidence = anchor_node_evidence
    positive_evidence = positive_node_evidence
    negative_evidence_input = negative_node_evidence
    graph_views = (anchor_graph, positive_graph, negative_graph)
    if any(view.ndim != 2 for view in graph_views):
        raise ValueError("图级三视图必须共享 [B, D] 形状")
    if any(view.shape != anchor_graph.shape for view in graph_views[1:]):
        raise ValueError("图级三视图形状必须完全一致")
    evidence_views = (
        anchor_evidence,
        positive_evidence,
        negative_evidence_input,
    )
    if any(view.ndim != 2 for view in evidence_views):
        raise ValueError("原生节点证据三视图必须共享 [B, N] 形状")
    if any(view.shape != anchor_evidence.shape for view in evidence_views[1:]):
        raise ValueError("原生节点证据三视图形状必须完全一致")
    if anchor_evidence.size(0) != anchor_graph.size(0):
        raise ValueError("原生节点证据批大小必须与图级表示一致")
    if direct_node_mask.shape != anchor_evidence.shape:
        raise ValueError("direct_node_mask 必须与原生节点证据共享 [B, N] 形状")

    scalars = {
        "GLOBAL_TEMPERATURE": global_temperature,
        "LOCAL_TEMPERATURE": local_temperature,
        "PAIR_MARGIN": pair_margin,
        "LOCAL_LOSS_WEIGHT": local_loss_weight,
        "EPS": eps,
        "MIN_REFERENCE_SCALE": min_reference_scale,
    }
    for name, value in scalars.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} 必须为有限数")
    if float(global_temperature) <= 0.0:
        raise ValueError("GLOBAL_TEMPERATURE 必须为正数")
    if float(local_temperature) <= 0.0:
        raise ValueError("LOCAL_TEMPERATURE 必须为正数")
    if float(local_loss_weight) < 0.0:
        raise ValueError("LOCAL_LOSS_WEIGHT 必须为非负数")
    if float(eps) <= 0.0:
        raise ValueError("EPS 必须为正数")
    if float(min_reference_scale) <= 0.0:
        raise ValueError("MIN_REFERENCE_SCALE 必须为正数")

    if any(not torch.isfinite(view).all() for view in evidence_views):
        raise FloatingPointError("原生节点证据出现非有限值")
    if any((view < 0).any() for view in evidence_views):
        raise ValueError("原生节点证据必须非负且数值越大越异常")
    # 先用 anchor/positive 的停止梯度正常证据水平消除后端分数单位，再做
    # log1p。只在正常参考恰为零时使用 eps，因此近零但非零的 OWLEYE 距离
    # 仍保持共同倍率不变；零距离本身也保留非零梯度，不形成 clamp 死区。
    reference_level = torch.cat(
        [anchor_evidence, positive_evidence],
        dim=1,
    ).mean(dim=1, keepdim=True).detach()
    safe_reference_level = torch.where(
        reference_level > 0,
        reference_level,
        torch.full_like(reference_level, float(eps)),
    )
    log_anchor = torch.log1p(anchor_evidence / safe_reference_level)
    log_positive = torch.log1p(positive_evidence / safe_reference_level)
    log_negative = torch.log1p(
        negative_evidence_input / safe_reference_level
    )
    reference_log = 0.5 * (log_anchor + log_positive)
    reference_center = reference_log.mean(dim=1, keepdim=True).detach()
    reference_scale = torch.cat(
        [log_anchor, log_positive],
        dim=1,
    ).std(dim=1, keepdim=True, unbiased=False).detach().clamp_min(
        float(min_reference_scale)
    )
    anchor_evidence = (log_anchor - reference_center) / reference_scale
    positive_evidence = (
        log_positive - reference_center
    ) / reference_scale
    negative_evidence_input = (
        log_negative - reference_center
    ) / reference_scale

    if sample_weights is None:
        weights = None
    else:
        weights = sample_weights.detach().to(
            device=anchor_graph.device,
            dtype=anchor_graph.dtype,
        ).reshape(-1)
        if weights.numel() != anchor_graph.size(0):
            raise ValueError("sample_weights 必须与批大小一致")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("sample_weights 必须为有限非负数")

    global_loss = carots_style_contrastive_loss(
        anchor_graph,
        positive_graph,
        negative_graph,
        temperature=float(global_temperature),
        sample_weights=weights,
    )
    anchor_graph_normalized = F.normalize(anchor_graph, p=2, dim=1)
    positive_graph_normalized = F.normalize(positive_graph, p=2, dim=1)
    negative_graph_normalized = F.normalize(negative_graph, p=2, dim=1)
    global_gap_per_sample = (
        (anchor_graph_normalized * positive_graph_normalized).sum(dim=1)
        - (anchor_graph_normalized * negative_graph_normalized).sum(dim=1)
    )
    global_gap = _weighted_sample_mean(
        global_gap_per_sample,
        weights,
        anchor_graph_normalized,
    )
    global_weight_sum = (
        anchor_graph.new_tensor(float(anchor_graph.size(0)))
        if weights is None
        else weights.sum().detach()
    )

    direct_mask = direct_node_mask.bool().detach()
    active_samples = direct_mask.any(dim=1)
    if weights is not None:
        active_samples = active_samples & (weights > 0)
    has_local_samples = bool(active_samples.any())
    local_measured = bool(float(local_loss_weight) > 0.0 and has_local_samples)
    if float(local_loss_weight) == 0.0 or not has_local_samples:
        zero = (
            anchor_evidence[..., :0].sum()
            + positive_evidence[..., :0].sum()
            + negative_evidence_input[..., :0].sum()
        ) * 0.0
        local_pair_loss = zero
        local_loss = zero
        weighted_local_loss = zero
        pair_gap = zero.detach().clone()
        reference_evidence = zero.detach().clone()
        negative_evidence = zero.detach().clone()
        pair_satisfaction = zero.detach().clone()
        local_weight_sum = zero.detach().clone()
    else:
        active_direct_mask = direct_mask & active_samples.unsqueeze(1)
        direct_counts = active_direct_mask.sum(dim=1)
        sample_indices = torch.arange(
            anchor_graph.size(0),
            device=anchor_graph.device,
        ).unsqueeze(1).expand_as(active_direct_mask)[active_direct_mask]
        reference_direct = 0.5 * (
            anchor_evidence[active_direct_mask]
            + positive_evidence[active_direct_mask]
        )
        negative_direct = negative_evidence_input[active_direct_mask]
        direct_pair_gap = negative_direct - reference_direct
        if not (
            torch.isfinite(reference_direct).all()
            and torch.isfinite(negative_direct).all()
        ):
            raise FloatingPointError("direct 原生节点证据出现非有限值")

        def per_sample_mean(values):
            sums = values.new_zeros(anchor_graph.size(0)).scatter_add(
                0,
                sample_indices,
                values,
            )
            return sums / direct_counts.clamp_min(1).to(values.dtype)

        per_node_pair_loss = F.softplus(
            (float(pair_margin) - direct_pair_gap)
            / float(local_temperature)
        )
        active_weights = weights[active_samples] if weights is not None else None
        reference_tensor = reference_direct
        local_pair_loss = _weighted_sample_mean(
            per_sample_mean(per_node_pair_loss)[active_samples],
            active_weights,
            reference_tensor,
        )
        local_loss = local_pair_loss
        weighted_local_loss = float(local_loss_weight) * local_loss
        pair_gap = _weighted_sample_mean(
            per_sample_mean(direct_pair_gap)[active_samples],
            active_weights,
            reference_tensor,
        )
        reference_evidence = _weighted_sample_mean(
            per_sample_mean(reference_direct)[active_samples],
            active_weights,
            reference_tensor,
        )
        negative_evidence = _weighted_sample_mean(
            per_sample_mean(negative_direct)[active_samples],
            active_weights,
            reference_tensor,
        )
        pair_satisfaction = _weighted_sample_mean(
            per_sample_mean(
                (direct_pair_gap >= float(pair_margin)).to(
                    direct_pair_gap.dtype
                )
            )[active_samples],
            active_weights,
            reference_tensor,
        )
        local_weight_sum = (
            reference_direct.new_tensor(float(active_samples.sum()))
            if active_weights is None
            else active_weights.sum().detach()
        )

    total = (
        global_loss
        if float(local_loss_weight) == 0.0
        else global_loss + weighted_local_loss
    )
    return {
        "total": total,
        "global": global_loss,
        "local": local_loss,
        "weighted_local": weighted_local_loss,
        "local_pair": local_pair_loss,
        "global_gap": global_gap,
        "pair_gap": pair_gap,
        "reference_evidence": reference_evidence,
        "negative_evidence": negative_evidence,
        "pair_satisfaction": pair_satisfaction,
        "global_weight_sum": global_weight_sum,
        "local_weight_sum": local_weight_sum,
        "local_measured": local_measured,
        "sample_count": torch.tensor(
            anchor_graph.size(0),
            device=anchor_graph.device,
            dtype=torch.long,
        ),
        "local_sample_count": active_samples.sum().detach(),
        "direct_node_count": (
            direct_mask & active_samples.unsqueeze(1)
        ).sum().detach(),
    }


def tail_preserving_mechanism_window_contrastive_loss(
    anchor_graph,
    positive_graph,
    negative_graph,
    anchor_window_risk,
    positive_window_risk,
    negative_window_risk,
    global_temperature=0.2,
    window_temperature=0.5,
    mechanism_margin=0.0,
    normal_variation_weight=1.0,
    window_loss_weight=0.0,
    tail_loss_weight=0.0,
    tail_fraction=0.1,
    tail_temperature=0.5,
    tail_budget=0.0,
    sample_weights=None,
):
    """在最终窗口风险空间分离机制破坏，并直接压缩高分正常尾部。

    锚点和关系保持正视图共同定义“正常内容变化”。只有 Graph-VAR 机制
    破坏负视图相对这两个正常视图产生了更大的窗口风险增量，机制间隔才算
    满足。尾部项与旧 ``hard_normal_tail_ranking_loss`` 的关键区别是：旧项
    只把合成异常推到困难正常窗口之上，本项直接降低批内最高分正常窗口
    相对正常中位风险的超额，因此不会把高分正常尾部原样保留下来。

    该目标只消费所有 GFM 后端共享的 ``graph_embedding`` 和
    ``window_score``，不读取 AnomalyGFM prompt、OWLEYE pattern 或专有节点
    语义，所以保持后端可替换边界。
    """
    graph_views = (anchor_graph, positive_graph, negative_graph)
    if any(view.ndim != 2 for view in graph_views):
        raise ValueError("图级三视图必须共享 [B, D] 形状")
    if any(view.shape != anchor_graph.shape for view in graph_views[1:]):
        raise ValueError("图级三视图形状必须完全一致")

    risk_views = tuple(
        view.reshape(-1)
        for view in (
            anchor_window_risk,
            positive_window_risk,
            negative_window_risk,
        )
    )
    if any(view.numel() != anchor_graph.size(0) for view in risk_views):
        raise ValueError("窗口风险三视图必须与图级批大小一致")
    if any(not torch.isfinite(view).all() for view in graph_views + risk_views):
        raise FloatingPointError("图级表示或窗口风险出现非有限值")

    scalars = {
        "GLOBAL_TEMPERATURE": global_temperature,
        "WINDOW_TEMPERATURE": window_temperature,
        "MECHANISM_MARGIN": mechanism_margin,
        "NORMAL_VARIATION_WEIGHT": normal_variation_weight,
        "WINDOW_LOSS_WEIGHT": window_loss_weight,
        "TAIL_LOSS_WEIGHT": tail_loss_weight,
        "TAIL_FRACTION": tail_fraction,
        "TAIL_TEMPERATURE": tail_temperature,
        "TAIL_BUDGET": tail_budget,
    }
    for name, value in scalars.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} 必须为有限数")
    if float(global_temperature) <= 0.0:
        raise ValueError("GLOBAL_TEMPERATURE 必须为正数")
    if float(window_temperature) <= 0.0:
        raise ValueError("WINDOW_TEMPERATURE 必须为正数")
    if float(tail_temperature) <= 0.0:
        raise ValueError("TAIL_TEMPERATURE 必须为正数")
    if float(normal_variation_weight) < 0.0:
        raise ValueError("NORMAL_VARIATION_WEIGHT 必须为非负数")
    if float(window_loss_weight) < 0.0:
        raise ValueError("WINDOW_LOSS_WEIGHT 必须为非负数")
    if float(tail_loss_weight) < 0.0:
        raise ValueError("TAIL_LOSS_WEIGHT 必须为非负数")
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("TAIL_FRACTION 必须位于 (0, 1]")
    if float(tail_budget) < 0.0:
        raise ValueError("TAIL_BUDGET 必须为非负数")

    if sample_weights is None:
        weights = None
        active_mask = torch.ones(
            anchor_graph.size(0),
            device=anchor_graph.device,
            dtype=torch.bool,
        )
    else:
        weights = sample_weights.detach().to(
            device=anchor_graph.device,
            dtype=anchor_graph.dtype,
        ).reshape(-1)
        if weights.numel() != anchor_graph.size(0):
            raise ValueError("sample_weights 必须与批大小一致")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("sample_weights 必须为有限非负数")
        active_mask = weights > 0

    global_loss = carots_style_contrastive_loss(
        anchor_graph,
        positive_graph,
        negative_graph,
        temperature=float(global_temperature),
        sample_weights=weights,
    )
    anchor_graph_normalized = F.normalize(anchor_graph, p=2, dim=1)
    positive_graph_normalized = F.normalize(positive_graph, p=2, dim=1)
    negative_graph_normalized = F.normalize(negative_graph, p=2, dim=1)
    global_gap_per_sample = (
        (anchor_graph_normalized * positive_graph_normalized).sum(dim=1)
        - (anchor_graph_normalized * negative_graph_normalized).sum(dim=1)
    )
    global_gap = _weighted_sample_mean(
        global_gap_per_sample,
        weights,
        anchor_graph,
    )

    anchor_risk, positive_risk, negative_risk = risk_views
    normal_reference = torch.maximum(anchor_risk, positive_risk)
    normal_variation = (anchor_risk - positive_risk).abs()
    mechanism_gain = negative_risk - normal_reference
    required_gain = (
        float(mechanism_margin)
        + float(normal_variation_weight) * normal_variation
    )
    window_violation = required_gain - mechanism_gain
    per_sample_window_loss = F.softplus(
        window_violation / float(window_temperature)
    )
    window_loss = _weighted_sample_mean(
        per_sample_window_loss,
        weights,
        negative_risk,
    )
    weighted_window_loss = float(window_loss_weight) * window_loss

    active_normal_reference = normal_reference[active_mask]
    active_anchor = anchor_risk[active_mask]
    active_positive = positive_risk[active_mask]
    active_weights = weights[active_mask] if weights is not None else None
    if active_normal_reference.numel() == 0:
        tail_loss = normal_reference.sum() * 0.0
        weighted_tail_loss = tail_loss
        normal_center_reference = tail_loss.detach().clone()
        hard_normal_tail_reference = tail_loss.detach().clone()
        hard_normal_tail_mean = tail_loss.detach().clone()
        tail_excess_mean = tail_loss.detach().clone()
        tail_violation_rate = tail_loss.detach().clone()
        hard_tail_count = torch.zeros(
            (), device=anchor_graph.device, dtype=torch.long
        )
    else:
        # 中位数只作为当前批的无标签正常基准，不允许通过移动基准逃避惩罚。
        normal_center_reference = torch.cat(
            [active_anchor, active_positive], dim=0
        ).median().detach()
        hard_count = max(
            1,
            int(
                math.ceil(
                    float(tail_fraction) * active_normal_reference.numel()
                )
            ),
        )
        hard_values, hard_indices = torch.topk(
            active_normal_reference,
            k=hard_count,
            largest=True,
            sorted=False,
        )
        tail_excess = hard_values - normal_center_reference
        per_tail_loss = F.softplus(
            (tail_excess - float(tail_budget))
            / float(tail_temperature)
        )
        hard_weights = (
            active_weights[hard_indices]
            if active_weights is not None
            else None
        )
        tail_loss = _weighted_sample_mean(
            per_tail_loss,
            hard_weights,
            hard_values,
        )
        weighted_tail_loss = float(tail_loss_weight) * tail_loss
        hard_normal_tail_reference = hard_values.min().detach()
        hard_normal_tail_mean = _weighted_sample_mean(
            hard_values,
            hard_weights,
            hard_values,
        ).detach()
        tail_excess_mean = _weighted_sample_mean(
            tail_excess,
            hard_weights,
            hard_values,
        ).detach()
        tail_violation_rate = _weighted_sample_mean(
            (tail_excess > float(tail_budget)).to(hard_values.dtype),
            hard_weights,
            hard_values,
        ).detach()
        hard_tail_count = torch.tensor(
            hard_count,
            device=anchor_graph.device,
            dtype=torch.long,
        )

    total = global_loss + weighted_window_loss + weighted_tail_loss
    weight_sum = (
        anchor_graph.new_tensor(float(anchor_graph.size(0)))
        if weights is None
        else weights.sum().detach()
    )
    return {
        "total": total,
        "global": global_loss,
        "window": window_loss,
        "weighted_window": weighted_window_loss,
        "tail": tail_loss,
        "weighted_tail": weighted_tail_loss,
        "global_gap": global_gap,
        "normal_variation": _weighted_sample_mean(
            normal_variation, weights, anchor_risk
        ).detach(),
        "normal_reference": _weighted_sample_mean(
            normal_reference, weights, anchor_risk
        ).detach(),
        "negative_risk": _weighted_sample_mean(
            negative_risk, weights, negative_risk
        ).detach(),
        "mechanism_gain": _weighted_sample_mean(
            mechanism_gain, weights, negative_risk
        ).detach(),
        "required_gain": _weighted_sample_mean(
            required_gain, weights, negative_risk
        ).detach(),
        "window_pair_gap": _weighted_sample_mean(
            mechanism_gain - required_gain,
            weights,
            negative_risk,
        ).detach(),
        "pair_satisfaction": _weighted_sample_mean(
            (mechanism_gain >= required_gain).to(negative_risk.dtype),
            weights,
            negative_risk,
        ).detach(),
        "normal_center_reference": normal_center_reference,
        "hard_normal_tail_reference": hard_normal_tail_reference,
        "hard_normal_tail_mean": hard_normal_tail_mean,
        "tail_excess_mean": tail_excess_mean,
        "tail_violation_rate": tail_violation_rate,
        "weight_sum": weight_sum,
        "sample_count": active_mask.sum().detach(),
        "hard_tail_count": hard_tail_count.detach(),
        "window_measured": bool(
            float(window_loss_weight) > 0.0 and active_mask.any()
        ),
        "tail_measured": bool(
            float(tail_loss_weight) > 0.0 and active_mask.any()
        ),
    }


def mechanism_conditioned_local_global_contrastive_loss(
    anchor_graph,
    positive_graph,
    negative_graph,
    anchor_nodes,
    positive_nodes,
    negative_nodes,
    direct_node_mask,
    unchanged_node_mask=None,
    temperature=0.2,
    margin=0.0,
    direct_weight=0.0,
    unchanged_weight=0.0,
    sample_weights=None,
):
    """计算机制条件的局部—全局反事实增量对比。

    原图级二元 InfoNCE 可写成 ``softplus((sim(a,n)-sim(a,p))/T)``。
    本函数在这个全局间隔上增加两项：Graph-VAR 直接干预传感器的局部
    间隔，以及未被直接/传播影响传感器的反事实漂移惩罚。它只作用于
    GFM 表征，不直接修改最终 MIL 风险，因此细粒度证据服务于窗口检测。

    ``margin=direct_weight=unchanged_weight=0`` 时与
    :func:`carots_style_contrastive_loss` 等价，作为第一阶段的回归边界。
    """
    graph_views = (anchor_graph, positive_graph, negative_graph)
    if any(view.ndim != 2 for view in graph_views):
        raise ValueError("图级三视图必须共享 [B, D] 形状")
    if (
        anchor_graph.shape != positive_graph.shape
        or anchor_graph.shape != negative_graph.shape
    ):
        raise ValueError(
            "图级三视图形状不一致: "
            f"anchor={tuple(anchor_graph.shape)}, "
            f"positive={tuple(positive_graph.shape)}, "
            f"negative={tuple(negative_graph.shape)}"
        )
    node_views = (anchor_nodes, positive_nodes, negative_nodes)
    if any(view.ndim != 3 for view in node_views):
        raise ValueError("节点级三视图必须共享 [B, N, D] 形状")
    if (
        anchor_nodes.shape != positive_nodes.shape
        or anchor_nodes.shape != negative_nodes.shape
    ):
        raise ValueError(
            "节点级三视图形状不一致: "
            f"anchor={tuple(anchor_nodes.shape)}, "
            f"positive={tuple(positive_nodes.shape)}, "
            f"negative={tuple(negative_nodes.shape)}"
        )
    if anchor_nodes.shape[0] != anchor_graph.shape[0]:
        raise ValueError(
            "图级和节点级批大小必须一致: "
            f"graph={anchor_graph.size(0)}, nodes={anchor_nodes.size(0)}"
        )
    if direct_node_mask.shape != anchor_nodes.shape[:2]:
        raise ValueError(
            "direct_node_mask 必须具有 [B, N] 形状: "
            f"mask={tuple(direct_node_mask.shape)}, "
            f"nodes={tuple(anchor_nodes.shape[:2])}"
        )

    temperature = float(temperature)
    margin = float(margin)
    direct_weight = float(direct_weight)
    unchanged_weight = float(unchanged_weight)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("局部—全局对比温度必须为有限正数")
    if not math.isfinite(margin):
        raise ValueError("局部—全局对比间隔必须为有限数")
    if not math.isfinite(direct_weight) or direct_weight < 0.0:
        raise ValueError("DIRECT_WEIGHT 必须为有限非负数")
    if not math.isfinite(unchanged_weight) or unchanged_weight < 0.0:
        raise ValueError("UNCHANGED_WEIGHT 必须为有限非负数")

    anchor_graph = F.normalize(anchor_graph, p=2, dim=-1)
    positive_graph = F.normalize(positive_graph, p=2, dim=-1)
    negative_graph = F.normalize(negative_graph, p=2, dim=-1)
    global_gap = (anchor_graph * positive_graph).sum(dim=-1) - (
        anchor_graph * negative_graph
    ).sum(dim=-1)

    direct_mask = direct_node_mask.bool().detach()
    direct_counts = direct_mask.sum(dim=1)
    # 全局等价臂不读取节点表征，保证它是原图级目标的严格回归对照。
    anchor_nodes_normalized = None
    negative_nodes_normalized = None
    if direct_weight != 0.0 or unchanged_weight != 0.0:
        anchor_nodes_normalized = F.normalize(anchor_nodes, p=2, dim=-1)
        negative_nodes_normalized = F.normalize(negative_nodes, p=2, dim=-1)

    if direct_weight == 0.0:
        direct_gap = global_gap.new_zeros(global_gap.shape)
    else:
        positive_nodes_normalized = F.normalize(positive_nodes, p=2, dim=-1)
        node_gap = (anchor_nodes_normalized * positive_nodes_normalized).sum(
            dim=-1
        ) - (anchor_nodes_normalized * negative_nodes_normalized).sum(dim=-1)
        direct_gap = (node_gap * direct_mask.to(node_gap.dtype)).sum(dim=1)
        direct_gap = direct_gap / direct_counts.clamp_min(1).to(node_gap.dtype)
        direct_gap = direct_gap.masked_fill(~direct_mask.any(dim=1), 0.0)

    if unchanged_node_mask is None:
        unchanged_mask = ~direct_mask
    else:
        if unchanged_node_mask.shape != anchor_nodes.shape[:2]:
            raise ValueError(
                "unchanged_node_mask 必须具有 [B, N] 形状: "
                f"mask={tuple(unchanged_node_mask.shape)}, "
                f"nodes={tuple(anchor_nodes.shape[:2])}"
            )
        unchanged_mask = unchanged_node_mask.bool().detach()
    unchanged_mask = unchanged_mask & ~direct_mask
    unchanged_counts = unchanged_mask.sum(dim=1)
    if unchanged_weight == 0.0:
        unchanged_drift = global_gap.new_zeros(global_gap.shape)
    else:
        node_drift = (
            1.0
            - (anchor_nodes_normalized * negative_nodes_normalized).sum(dim=-1)
        ).clamp_min(0.0)
        unchanged_drift = (
            node_drift * unchanged_mask.to(node_drift.dtype)
        ).sum(dim=1)
        unchanged_drift = unchanged_drift / unchanged_counts.clamp_min(1).to(
            node_drift.dtype
        )
        unchanged_drift = unchanged_drift.masked_fill(
            ~unchanged_mask.any(dim=1),
            0.0,
        )

    counterfactual_gap = (
        global_gap
        + direct_weight * direct_gap
        - unchanged_weight * unchanged_drift
    )
    per_sample = F.softplus((margin - counterfactual_gap) / temperature)
    if not torch.isfinite(per_sample).all():
        raise FloatingPointError("局部—全局反事实对比产生了非有限样本损失")

    total = _weighted_sample_mean(per_sample, sample_weights, anchor_graph)
    return {
        "total": total,
        "global_gap": _weighted_sample_mean(global_gap, sample_weights, anchor_graph),
        "direct_gap": _weighted_sample_mean(direct_gap, sample_weights, anchor_graph),
        "unchanged_drift": _weighted_sample_mean(
            unchanged_drift,
            sample_weights,
            anchor_graph,
        ),
        "counterfactual_gap": _weighted_sample_mean(
            counterfactual_gap,
            sample_weights,
            anchor_graph,
        ),
        "sample_count": torch.tensor(anchor_graph.size(0), device=anchor_graph.device),
        "direct_node_count": direct_counts.sum().detach(),
        "unchanged_node_count": unchanged_counts.sum().detach(),
    }


def mechanism_matched_relation_contrastive_loss(
    anchor_nodes,
    positive_nodes,
    negative_nodes,
    direct_node_mask,
    unchanged_node_mask=None,
    temperature=0.2,
    unchanged_weight=0.25,
    sample_weights=None,
):
    """只在 GFM 相对时间内容的关系残差上执行成对对比。

    ``direct_node_mask`` 决定主要监督位置；未干预节点只承担较弱的一致性
    约束，避免细粒度目标压过窗口 MIL。样本选择和节点掩码均不参与梯度。
    """
    if (
        anchor_nodes.shape != positive_nodes.shape
        or anchor_nodes.shape != negative_nodes.shape
        or anchor_nodes.ndim != 3
    ):
        raise ValueError("关系残差视图必须共享 [B, N, D] 形状")
    if direct_node_mask.shape != anchor_nodes.shape[:2]:
        raise ValueError("direct_node_mask 必须是 [B, N]")
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("关系残差对比温度必须为正")
    direct_mask = direct_node_mask.bool().detach()
    anchor = F.normalize(anchor_nodes, p=2, dim=-1)
    positive = F.normalize(positive_nodes, p=2, dim=-1)
    negative = F.normalize(negative_nodes, p=2, dim=-1)
    pos_sim = (anchor * positive).sum(dim=-1)
    neg_sim = (anchor * negative).sum(dim=-1)
    logits = torch.stack((pos_sim, neg_sim), dim=-1) / temperature
    labels = torch.zeros_like(logits[..., 0], dtype=torch.long)
    direct_losses = F.cross_entropy(
        logits.reshape(-1, 2),
        labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)
    direct_losses = direct_losses.masked_fill(~direct_mask, 0.0)
    direct_count = direct_mask.sum(dim=1).clamp_min(1)
    per_sample_direct = direct_losses.sum(dim=1) / direct_count

    if unchanged_node_mask is None:
        unchanged_mask = ~direct_mask
    else:
        unchanged_mask = unchanged_node_mask.bool().detach()
    unchanged_mask = unchanged_mask & ~direct_mask
    consistency = (1.0 - (anchor * negative).sum(dim=-1)).clamp_min(0.0)
    consistency = consistency.masked_fill(~unchanged_mask, 0.0)
    unchanged_count = unchanged_mask.sum(dim=1).clamp_min(1)
    per_sample_unchanged = consistency.sum(dim=1) / unchanged_count
    per_sample = per_sample_direct + float(unchanged_weight) * per_sample_unchanged
    active = direct_mask.any(dim=1)
    per_sample = per_sample.masked_fill(~active, 0.0)
    if sample_weights is not None:
        sample_weights = sample_weights.detach() * active.to(sample_weights.dtype)
    else:
        sample_weights = active.to(per_sample.dtype)
    total = _weighted_sample_mean(per_sample, sample_weights, anchor_nodes)
    return {
        "total": total,
        "direct": _weighted_sample_mean(
            per_sample_direct, sample_weights, anchor_nodes
        ),
        "unchanged": _weighted_sample_mean(
            per_sample_unchanged, sample_weights, anchor_nodes
        ),
        "active_count": active.sum().detach(),
        "direct_count": direct_mask.sum().detach(),
    }


def batch_hard_normal_contrastive_loss(
    anchor,
    positive,
    negative,
    normal_scores,
    temperature,
    hard_normal_fraction,
    hard_negative_count,
):
    """Contrast high-scoring normal anchors against batch-hard anomalies.

    High-scoring normal windows remain normal: their relation-preserving view
    is the positive, while the most similar accepted synthetic anomalies in
    the batch form the negative set.  Score selection is detached so this
    term changes representation geometry rather than directly suppressing
    the selected window scores.
    """
    if anchor.shape != positive.shape or anchor.ndim != 2:
        raise ValueError(
            "anchor and positive must have the same [B, D] shape: "
            f"anchor={tuple(anchor.shape)}, positive={tuple(positive.shape)}"
        )
    if negative.ndim != 2 or negative.size(1) != anchor.size(1):
        raise ValueError(
            "negative must have shape [V, D] with the shared embedding dim: "
            f"anchor={tuple(anchor.shape)}, negative={tuple(negative.shape)}"
        )
    scores = normal_scores.reshape(-1)
    if scores.numel() != anchor.size(0):
        raise ValueError(
            "normal_scores must contain one score per normal anchor: "
            f"scores={scores.numel()}, anchors={anchor.size(0)}"
        )
    hard_normal_fraction = float(hard_normal_fraction)
    if not 0.0 < hard_normal_fraction <= 1.0:
        raise ValueError("hard_normal_fraction must be in (0, 1]")
    hard_negative_count = int(hard_negative_count)
    if hard_negative_count < 1:
        raise ValueError("hard_negative_count must be positive")
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if anchor.size(0) == 0 or negative.size(0) == 0:
        return anchor.sum() * 0.0

    hard_normal_count = max(
        1,
        int(math.ceil(hard_normal_fraction * anchor.size(0))),
    )
    hard_indices = torch.topk(
        scores.detach(),
        k=hard_normal_count,
        largest=True,
        sorted=False,
    ).indices
    hard_anchor = F.normalize(anchor[hard_indices], p=2, dim=1)
    hard_positive = F.normalize(positive[hard_indices], p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)

    positive_logits = torch.sum(
        hard_anchor * hard_positive,
        dim=1,
        keepdim=True,
    )
    all_negative_logits = hard_anchor @ negative.transpose(0, 1)
    negative_count = min(hard_negative_count, negative.size(0))
    hard_negative_logits = torch.topk(
        all_negative_logits,
        k=negative_count,
        dim=1,
        largest=True,
        sorted=False,
    ).values
    logits = torch.cat([positive_logits, hard_negative_logits], dim=1)
    logits = logits / temperature
    labels = torch.zeros(
        hard_anchor.size(0),
        dtype=torch.long,
        device=hard_anchor.device,
    )
    loss = F.cross_entropy(logits, labels)
    chance_scale = math.log(2.0) / math.log(float(negative_count + 1))
    return loss * chance_scale


def node_aware_hard_contrastive_loss(
    anchor_nodes,
    positive_nodes,
    negative_nodes,
    direct_node_mask,
    normal_node_scores,
    temperature,
    hard_normal_fraction,
    hard_negative_count,
    sample_weights=None,
):
    """Contrast hard normal nodes against same-sensor direct interventions.

    Candidate anchors are the normal counterparts of accepted directly
    intervened nodes.  Their relation-preserving counterparts are positives.
    Negatives are mined only from directly intervened nodes with the same
    sensor identity, so cross-sensor semantics cannot become a shortcut.
    Score-based hard-anchor selection is detached; gradients act on the
    shared node representation rather than on the selecting node score.
    """
    if (
        anchor_nodes.shape != positive_nodes.shape
        or anchor_nodes.shape != negative_nodes.shape
        or anchor_nodes.ndim != 3
    ):
        raise ValueError(
            "node views must share shape [B, N, D]: "
            f"anchor={tuple(anchor_nodes.shape)}, "
            f"positive={tuple(positive_nodes.shape)}, "
            f"negative={tuple(negative_nodes.shape)}"
        )
    expected_node_shape = anchor_nodes.shape[:2]
    if direct_node_mask.shape != expected_node_shape:
        raise ValueError(
            "direct_node_mask must have shape [B, N]: "
            f"mask={tuple(direct_node_mask.shape)}, "
            f"nodes={tuple(expected_node_shape)}"
        )
    if normal_node_scores.shape != expected_node_shape:
        raise ValueError(
            "normal_node_scores must have shape [B, N]: "
            f" scores={tuple(normal_node_scores.shape)}, "
            f"nodes={tuple(expected_node_shape)}"
        )
    hard_normal_fraction = float(hard_normal_fraction)
    if not 0.0 < hard_normal_fraction <= 1.0:
        raise ValueError("hard_normal_fraction must be in (0, 1]")
    hard_negative_count = int(hard_negative_count)
    if hard_negative_count < 1:
        raise ValueError("hard_negative_count must be positive")
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    direct_mask = direct_node_mask.bool()
    direct_indices = torch.nonzero(direct_mask, as_tuple=False)
    if direct_indices.numel() == 0:
        return anchor_nodes.sum() * 0.0

    candidate_scores = normal_node_scores[direct_mask]
    hard_normal_count = max(
        1,
        int(math.ceil(hard_normal_fraction * candidate_scores.numel())),
    )
    selected = torch.topk(
        candidate_scores.detach(),
        k=hard_normal_count,
        largest=True,
        sorted=False,
    ).indices
    selected_indices = direct_indices[selected]

    anchor_nodes = F.normalize(anchor_nodes, p=2, dim=-1)
    positive_nodes = F.normalize(positive_nodes, p=2, dim=-1)
    negative_nodes = F.normalize(negative_nodes, p=2, dim=-1)
    per_anchor_losses = []
    selected_weights = []
    for batch_idx, sensor_idx in selected_indices.tolist():
        anchor = anchor_nodes[batch_idx, sensor_idx]
        positive = positive_nodes[batch_idx, sensor_idx]
        same_sensor_mask = direct_mask[:, sensor_idx]
        negative_pool = negative_nodes[same_sensor_mask, sensor_idx]
        negative_count = min(hard_negative_count, negative_pool.size(0))
        negative_logits = torch.topk(
            negative_pool @ anchor,
            k=negative_count,
            largest=True,
            sorted=False,
        ).values
        positive_logit = torch.sum(anchor * positive).reshape(1)
        logits = torch.cat([positive_logit, negative_logits]) / temperature
        per_anchor = torch.logsumexp(logits, dim=0) - logits[0]
        chance_scale = math.log(2.0) / math.log(float(negative_count + 1))
        per_anchor_losses.append(per_anchor * chance_scale)
        if sample_weights is not None:
            selected_weights.append(sample_weights[batch_idx])

    per_anchor_losses = torch.stack(per_anchor_losses)
    weights = (
        torch.stack(selected_weights)
        if sample_weights is not None
        else None
    )
    return _weighted_sample_mean(per_anchor_losses, weights, anchor_nodes)


def intervention_gain_node_loss(
    anchor_scores,
    anomaly_scores,
    node_mask,
    margin,
    detach_anchor=True,
    sample_weights=None,
):
    if anchor_scores.shape != anomaly_scores.shape or anomaly_scores.shape != node_mask.shape:
        raise ValueError(
            "intervention_gain_node_loss got inconsistent shapes: "
            f"anchor={tuple(anchor_scores.shape)}, anomaly={tuple(anomaly_scores.shape)}, "
            f"mask={tuple(node_mask.shape)}"
        )

    baseline = anchor_scores.detach() if detach_anchor else anchor_scores
    gains = anomaly_scores - baseline
    node_mask = node_mask.bool()
    losses = []
    selected_weights = []

    for sample_idx, (sample_gains, mask) in enumerate(zip(gains, node_mask)):
        direct_gains = sample_gains[mask]
        unmodified_gains = sample_gains[~mask]
        if direct_gains.numel() == 0 or unmodified_gains.numel() == 0:
            continue
        pairwise_margin = margin + unmodified_gains.unsqueeze(0) - direct_gains.unsqueeze(1)
        losses.append(F.relu(pairwise_margin).mean())
        if sample_weights is not None:
            selected_weights.append(sample_weights[sample_idx])

    if not losses:
        return anomaly_scores.sum() * 0.0
    selected_weights = (
        torch.stack(selected_weights) if sample_weights is not None else None
    )
    return _weighted_sample_mean(
        torch.stack(losses),
        selected_weights,
        anomaly_scores,
    )


def masked_node_consistency_loss(
    reference_scores,
    view_scores,
    mask=None,
    detach_reference=True,
    sample_weights=None,
):
    if reference_scores.shape != view_scores.shape:
        raise ValueError(
            "masked_node_consistency_loss got inconsistent score shapes: "
            f"reference={tuple(reference_scores.shape)}, view={tuple(view_scores.shape)}"
        )
    if mask is None:
        mask = torch.ones_like(view_scores, dtype=torch.bool)
    if mask.shape != view_scores.shape:
        raise ValueError(
            "masked_node_consistency_loss got inconsistent mask shape: "
            f"scores={tuple(view_scores.shape)}, mask={tuple(mask.shape)}"
        )

    target = reference_scores.detach() if detach_reference else reference_scores
    elementwise = F.smooth_l1_loss(view_scores, target, reduction="none")
    mask = mask.bool()
    if sample_weights is None:
        selected = elementwise.masked_select(mask)
        if selected.numel() == 0:
            return view_scores.sum() * 0.0
        return selected.mean()

    per_sample = []
    selected_weights = []
    for sample_idx, (sample_loss, sample_mask) in enumerate(zip(elementwise, mask)):
        selected = sample_loss.masked_select(sample_mask)
        if selected.numel() == 0:
            continue
        per_sample.append(selected.mean())
        selected_weights.append(sample_weights[sample_idx])
    if not per_sample:
        return view_scores.sum() * 0.0
    return _weighted_sample_mean(
        torch.stack(per_sample),
        torch.stack(selected_weights),
        view_scores,
    )


def ranking_loss(normal_scores, anomaly_scores, margin, sample_weights=None):
    per_sample = F.relu(margin + normal_scores - anomaly_scores)
    return _weighted_sample_mean(per_sample, sample_weights, anomaly_scores)


def hard_normal_tail_ranking_loss(
    normal_scores,
    anomaly_scores,
    margin,
    hard_fraction,
    positive_normal_scores=None,
    sample_weights=None,
):
    """Rank synthetic anomalies above the hardest normal-score tail.

    The normal pool contains the original normal windows and, when supplied,
    their relation-preserving positive views.  The mean of the highest fixed
    fraction is used instead of a label-selected threshold.
    """
    hard_fraction = float(hard_fraction)
    if not 0.0 < hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be in (0, 1]")

    normal_parts = [normal_scores.reshape(-1)]
    if positive_normal_scores is not None:
        normal_parts.append(positive_normal_scores.reshape(-1))
    normal_pool = torch.cat(normal_parts, dim=0)
    anomalies = anomaly_scores.reshape(-1)
    if normal_pool.numel() == 0 or anomalies.numel() == 0:
        return anomaly_scores.sum() * 0.0

    hard_count = max(1, int(math.ceil(hard_fraction * normal_pool.numel())))
    hard_reference = torch.topk(
        normal_pool,
        k=hard_count,
        largest=True,
        sorted=False,
    ).values.mean()
    per_sample = F.relu(margin + hard_reference - anomalies)
    return _weighted_sample_mean(per_sample, sample_weights, anomaly_scores)


def window_bce_loss(
    normal_scores,
    anomaly_scores,
    positive_scores=None,
    sample_weights=None,
):
    losses = [
        F.binary_cross_entropy_with_logits(
            normal_scores,
            torch.zeros_like(normal_scores),
            reduction="none",
        ),
        F.binary_cross_entropy_with_logits(
            anomaly_scores,
            torch.ones_like(anomaly_scores),
            reduction="none",
        ),
    ]
    if positive_scores is not None:
        losses.append(
            F.binary_cross_entropy_with_logits(
                positive_scores,
                torch.zeros_like(positive_scores),
                reduction="none",
            )
        )
    per_sample = torch.stack(losses, dim=0).mean(dim=0)
    return _weighted_sample_mean(per_sample, sample_weights, anomaly_scores)


def head_fidelity_loss(node_scores_calibrated, raw_node_scores, eps=1e-6):
    raw_log = torch.log(raw_node_scores.clamp_min(eps))
    return F.smooth_l1_loss(node_scores_calibrated, raw_log)
