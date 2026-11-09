from __future__ import annotations
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import torch
from yacs.config import CfgNode
from config import get_cfg_defaults
from datasets.loader import get_val_dataloader
from models.shark.modeling_information_preserving_shark import InformationPreservingUnifiedSHARK
from training.joint import JointTrainer, assert_joint_trainables
from training import data as profile
from training.assets import ROOT, SPECS, CURRENT_OWLEYE_SHA256, sha256 as _sha256, asset_path
from training.config_merge import _merge_verified_source_cfg
from utils.misc import set_devices, set_seeds


def _known_tree(source: CfgNode, template: CfgNode, prefix: str = ""):
    kept = {}
    dropped = []
    for key, value in source.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in template:
            dropped.append(path)
            continue
        template_value = template[key]
        if isinstance(value, CfgNode) and isinstance(template_value, CfgNode):
            child, child_dropped = _known_tree(value, template_value, path)
            kept[key] = child
            dropped.extend(child_dropped)
        else:
            kept[key] = value
    return kept, dropped



def build_current_config(
    source_path: Path,
    *,
    entity: str,
    backend: str,
    seed: int,
    frontend_root: Path,
    eval_root: Path,
    data_dir: Path | None = None,
    owleye_checkpoint: Path | None = None,
    assets: dict | None = None,
):
    if backend != "owleye" and owleye_checkpoint is not None:
        raise ValueError("OWLEYE checkpoint supplied to AnomalyGFM run")
    spec = SPECS[(backend, entity)]
    expected_source = (ROOT / spec["snapshot"]).resolve()
    source_path = source_path.resolve()
    if source_path != expected_source:
        raise ValueError(
            f"source config path mismatch: expected={expected_source}, got={source_path}"
        )
    actual_sha256 = _sha256(source_path)
    if actual_sha256 != spec["config_sha256"]:
        raise ValueError("registered source config SHA256 changed")

    source = CfgNode.load_cfg(source_path.read_text(encoding="utf-8"))
    cfg = get_cfg_defaults()
    _, dropped = _known_tree(source, cfg)
    _merge_verified_source_cfg(cfg, source)
    if cfg.DATA.NAME != entity or str(cfg.SHARK.GFM).lower() != backend:
        raise ValueError("source config entity/backend mismatch")

    cfg.SEED = int(seed)
    cfg.VISIBLE_DEVICES = 0
    cfg.DATA.NAME = entity
    cfg.DATA.WIN_SIZE = 60
    cfg.DATA.TRAIN_STEP = 1
    cfg.DATA.TEST_STEP = 1
    cfg.DATA.SCALE = "standard"
    cfg.DATA.PREPROCESS.PROTOCOL = "train_only"
    cfg.TRAIN.ENABLE = True
    cfg.TRAIN.RESUME = False
    cfg.TRAIN.BATCH_SIZE = 256
    cfg.TRAIN.EVAL_PERIOD = 5
    cfg.SOLVER.MAX_EPOCH = 30
    cfg.DLINEAR.FREEZE = False
    cfg.TEST.ENABLE = False
    cfg.SHARK.UNIFIED.FRONTEND_CHECKPOINT = ""
    cfg.SHARK.UNIFIED.FRONTEND_SHA256 = ""
    experiment_id = f"SHARK-WARM-FRONTEND-{entity}-SEED{seed}-{backend}"
    cfg.SHARK.UNIFIED.EXPERIMENT_ID = experiment_id
    cfg.SHARK.GFM_BACKEND.EXPERIMENT_ID = experiment_id
    cfg.RESULT_DIR = str(frontend_root.resolve())
    cfg.TRAIN.CHECKPOINT_DIR = cfg.RESULT_DIR
    cfg.SHARK.EVAL.RESULT_DIR = str(eval_root.resolve())
    cfg.DATA.BASE_DIR = (
        str(data_dir.resolve()) if data_dir is not None
        else asset_path(assets, "data_dir")
    )
    cfg.ANOMALYGFM.CHECKPOINT_PATH = asset_path(assets, "anomalygfm_checkpoint", required=backend == "anomalygfm")
    cfg.ANOMALYGFM.SOURCE_MODEL_PATH = asset_path(assets, "anomalygfm_source", required=backend == "anomalygfm")
    prefixes = list(cfg.SHARK.GFM_BACKEND.TRAINABLE_PREFIXES)
    if "imlp." not in prefixes:
        prefixes.append("imlp.")
    cfg.SHARK.GFM_BACKEND.TRAINABLE_PREFIXES = prefixes
    cfg.SHARK.SYNTHETIC.COUNTERFACTUAL_IMPLEMENTATION = "batched_precomputed"
    cfg.SHARK.ROOT_PROP.CONTROLLED_EVAL_ENABLE = False
    cfg.SHARK.UNIFIED.RUN_MODE = "primary"
    if backend == "owleye":
        checkpoint = Path(owleye_checkpoint or asset_path(assets, "owleye_checkpoint")).resolve()
        if _sha256(checkpoint) != CURRENT_OWLEYE_SHA256:
            raise ValueError("OWLEYE pretrained checkpoint is not the verified artifact")
        cfg.SHARK.GFM_BACKEND.CHECKPOINT_PATH = str(checkpoint)
        cfg.SHARK.GFM_BACKEND.CHECKPOINT_SHA256 = CURRENT_OWLEYE_SHA256
        cfg.SHARK.GFM_BACKEND.SOURCE_MODEL_PATH = asset_path(assets, "owleye_source")

    n_vars = {"SMD_1-1": 38, "SMAP_T-3": 25, "MSL_P-15": 55}
    cfg.DATA.N_VAR = n_vars[entity]
    cfg.TIMESNET.enc_in = cfg.DATA.N_VAR
    cfg.TIMESNET.c_out = cfg.DATA.N_VAR
    cfg.ITRANSFORMER.enc_in = cfg.DATA.N_VAR
    cfg.ITRANSFORMER.c_out = cfg.DATA.N_VAR
    cfg.CUTS_PLUS.N_NODES = cfg.DATA.N_VAR
    cfg.CUTS_PLUS.N_GROUPS = cfg.DATA.N_VAR
    cfg.TRAIN.CHECKPOINT_DIR = str(
        (Path(cfg.TRAIN.CHECKPOINT_DIR) / entity).resolve()
    )
    cfg.RESULT_DIR = str((Path(cfg.RESULT_DIR) / entity).resolve())

    report = {
        "schema_version": "SHARK-WARM-FRONTEND-UNIFIED-TRAINING-V2",
        "entity": entity,
        "backend": backend,
        "seed": int(seed),
        "epochs": int(cfg.SOLVER.MAX_EPOCH),
        "batch_size": int(cfg.TRAIN.BATCH_SIZE),
        "validation_period": int(cfg.TRAIN.EVAL_PERIOD),
        "source_config": str(source_path),
        "source_config_sha256": actual_sha256,
        "dropped_legacy_key_count": len(dropped),
        "dropped_legacy_keys": sorted(dropped),
        "frontend_checkpoint_loaded": False,
        "model_class": "InformationPreservingUnifiedSHARK",
        "trainer_class": "JointTrainer",
        "test_enabled": False,
        "test_loader_substituted_with_train_normal_validation": True,
        "test_distribution_used_by_preprocessing": False,
        "test_labels_used_for_training_or_selection": False,
    }
    return cfg, report



def _frontend_digest(module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()



def train_frontend(cfg, report: dict, *, config_output: Path, report_output: Path, threads: int, device_override=None):
    config_output.write_text(cfg.dump(), encoding="utf-8")
    report.update(status="TRAINING", effective_config=str(config_output.resolve()))
    report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.set_num_threads(threads)
    set_devices(cfg.VISIBLE_DEVICES)
    set_seeds(cfg.SEED)
    device = torch.device(device_override or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = InformationPreservingUnifiedSHARK(cfg).to(device)
    if not hasattr(model, "imlp"):
        raise RuntimeError("SHARK model has no DLinear frontend")
    if not any(parameter.requires_grad for parameter in model.imlp.parameters()):
        raise RuntimeError("DLinear frontend is frozen in frontend training")
    initial_digest = _frontend_digest(model.imlp)
    with patch("trainer.get_test_dataloader", side_effect=get_val_dataloader):
        trainer = JointTrainer(cfg, model)
    assert_joint_trainables(model, trainer.optimizer)
    trainer.train()
    checkpoint = Path(cfg.TRAIN.CHECKPOINT_DIR) / "checkpoint_best.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frontend training checkpoint missing: {checkpoint}")
    best = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state"], strict=True)
    final_digest = _frontend_digest(model.imlp)
    if final_digest == initial_digest:
        raise RuntimeError("best frontend checkpoint did not update DLinear")
    report.update(
        status="COMPLETE",
        completed_epochs=int(trainer.cur_epoch) + 1,
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=_sha256(checkpoint),
        selected_epoch_1based=int(best["epoch"]) + 1,
        frontend_initial_sha256=initial_digest,
        frontend_final_sha256=final_digest,
        frontend_updated=True,
    )
    report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[seeded warm frontend] COMPLETE entity={report['entity']} "
        f"backend={report['backend']} seed={report['seed']} checkpoint={checkpoint}",
        flush=True,
    )
