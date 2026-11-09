# reference: https://github.com/facebookresearch/SlowFast
import torch
import utils.lr_policy as lr_policy


def get_optimizer(optim_params, cfg):
    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == 'adam':
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == 'Radam':
        return torch.optim.RAdam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == 'adamw':
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def construct_optimizer(model, cfg):
    optim_params = get_param_groups(model)
    optimizer = get_optimizer(optim_params, cfg)

    return optimizer


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decay.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for idx, param_group in enumerate(optimizer.param_groups):
        lr_scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = new_lr * lr_scale


def get_param_groups(model):
    cfg = getattr(model, "cfg", None)
    if cfg is None or not hasattr(cfg, "GFM4MTSAD") or not hasattr(cfg.GFM4MTSAD, "GFM_TUNING"):
        return [p for n, p in model.named_parameters() if p.requires_grad]

    base_params = []
    peft_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_peft_parameter(name):
            peft_params.append(param)
        else:
            base_params.append(param)

    param_groups = []
    if base_params:
        param_groups.append({
            "params": base_params,
            "lr": cfg.SOLVER.BASE_LR,
            "lr_scale": 1.0,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            "name": "base",
        })

    if peft_params:
        peft_lr = float(getattr(cfg.GFM4MTSAD.GFM_TUNING, "PEFT_LR", cfg.SOLVER.BASE_LR))
        peft_lr_scale = float(getattr(cfg.GFM4MTSAD.GFM_TUNING, "PEFT_LR_SCALE", 1.0))
        if peft_lr > 0:
            lr_scale = peft_lr / max(float(cfg.SOLVER.BASE_LR), 1e-12)
        else:
            lr_scale = peft_lr_scale
        peft_weight_decay = float(getattr(cfg.GFM4MTSAD.GFM_TUNING, "PEFT_WEIGHT_DECAY", -1.0))
        if peft_weight_decay < 0:
            peft_weight_decay = cfg.SOLVER.WEIGHT_DECAY
        param_groups.append({
            "params": peft_params,
            "lr": cfg.SOLVER.BASE_LR * lr_scale,
            "lr_scale": lr_scale,
            "weight_decay": peft_weight_decay,
            "name": "peft",
        })

    if param_groups:
        return param_groups

    return [p for n, p in model.named_parameters() if p.requires_grad]


def _is_peft_parameter(name):
    return (
        name.startswith("sensor_graph_prompt.")
        or name.startswith("gfm.")
    )


def get_trainable_params(model):
    return [p for _, p in model.named_parameters() if p.requires_grad]
