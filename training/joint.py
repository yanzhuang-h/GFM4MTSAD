from __future__ import annotations
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode
import models.optimizer as optimizer_utils
from config import get_cfg_defaults
from models.shark.modeling_information_preserving_shark import InformationPreservingUnifiedSHARK
from models.shark.trainer_shark import SharkTrainer
from evaluation.predictor import InformationPreservingPredictor
from training import data as profile
from training.assets import ROOT, SPECS, CURRENT_OWLEYE_SHA256, sha256 as _sha256, asset_path
from training.references import save_evaluation_reference
from training.config_merge import _merge_verified_source_cfg
from utils.misc import set_seeds
BACKENDS = ("anomalygfm", "owleye")
ENTITIES = ("SMD_1-1", "SMAP_T-3", "MSL_P-15")
FRONTEND_INITS = ("pretrained",)


def _load_seeded_frontend_registration(args):
    registration_path = getattr(args, "frontend_registration", None)
    seed = int(getattr(args, "seed", 0))
    if registration_path is None:
        raise ValueError("warm-start requires an explicit frontend registration")
    registration_path = Path(registration_path).resolve()
    if not registration_path.is_file():
        raise FileNotFoundError(
            f"frontend registration not found: {registration_path}"
        )
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("entity") != args.entity:
        raise ValueError("frontend registration entity mismatch")
    if registration.get("backend") != args.backend:
        raise ValueError("frontend registration backend mismatch")
    if int(registration.get("window", -1)) != 60:
        raise ValueError("frontend registration window must be 60")
    if registration.get("preprocessing_protocol") != "train_only":
        raise ValueError("warm-start frontend must use train-only preprocessing")
    if registration.get("test_distribution_used_by_preprocessing", True):
        raise ValueError("warm-start frontend registration used test distribution")
    if registration.get("test_labels_used_by_preprocessing", True):
        raise ValueError("warm-start frontend registration used test labels")
    if int(registration.get("seed", -1)) != seed:
        raise ValueError("frontend registration seed mismatch")
    if registration.get("observed_input_dimensions") != [60]:
        raise ValueError("frontend registration has the wrong DLinear input shape")
    checkpoint = Path(registration["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = registration_path.parent / checkpoint
    checkpoint = checkpoint.resolve()
    registration["checkpoint"] = str(checkpoint)
    expected_sha256 = str(registration.get("checkpoint_sha256", ""))
    if not expected_sha256 or _sha256(checkpoint) != expected_sha256:
        raise ValueError("seeded frontend checkpoint SHA256 mismatch")
    return registration_path, registration



def configure(args):
    if args.backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if args.entity not in ENTITIES:
        raise ValueError(f"entity must be one of {ENTITIES}")
    if args.run_mode not in ("smoke", "primary"):
        raise ValueError("run_mode must be smoke or primary")
    if args.frontend_init not in FRONTEND_INITS:
        raise ValueError(f"frontend_init must be one of {FRONTEND_INITS}")
    seed = int(getattr(args, "seed", 0))
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    frontend_registration_path = None
    frontend_registration = None
    if args.frontend_init == "fresh" and getattr(args, "frontend_registration", None):
        raise ValueError("fresh initialization cannot load a frontend registration")
    spec = SPECS[(args.backend, args.entity)]
    source_path = ROOT / spec["snapshot"]
    if _sha256(source_path) != spec["config_sha256"]:
        raise ValueError("registered source config changed; refuse joint initialization run")
    source = CfgNode.load_cfg(source_path.read_text(encoding="utf-8"))
    cfg = get_cfg_defaults()
    _merge_verified_source_cfg(cfg, source)
    if cfg.DATA.NAME != args.entity or str(cfg.SHARK.GFM).lower() != args.backend:
        raise ValueError("source config entity/backend mismatch")
    if cfg.DATA.WIN_SIZE != 60 or not cfg.DLINEAR.INDIVIDUAL:
        raise ValueError("joint initialization control requires individual DLinear at window 60")
    if (cfg.TRAIN.BATCH_SIZE != 256 or cfg.SOLVER.MAX_EPOCH != 30
            or cfg.TRAIN.EVAL_PERIOD != 5):
        raise ValueError("unexpected registered training schedule")
    if cfg.DATA.PREPROCESS.PROTOCOL != "train_only" or cfg.TRAIN.RESUME:
        raise ValueError("joint training requires train-only preprocessing and no resume")
    if cfg.SHARK.OTRCL.ENABLE:
        raise ValueError("the joint initialization control does not enable OTRCL")
    if cfg.SHARK.GFM_BACKEND.TRAINING_MODE != "frozen_backend_shared_outer":
        raise ValueError("unexpected backend tuning mode")

    cfg.DLINEAR.FREEZE = False
    if args.frontend_init == "pretrained":
        frontend_registration_path, frontend_registration = (
            _load_seeded_frontend_registration(args)
        )
        frontend_checkpoint = Path(frontend_registration["checkpoint"])
        expected_frontend_sha256 = str(frontend_registration["checkpoint_sha256"])
        if not expected_frontend_sha256:
            raise ValueError("registered source config has no frontend SHA256")
        if _sha256(frontend_checkpoint) != expected_frontend_sha256:
            raise ValueError("registered frontend checkpoint changed; refuse warm-start run")
        cfg.SHARK.UNIFIED.FRONTEND_CHECKPOINT = str(frontend_checkpoint)
        cfg.SHARK.UNIFIED.FRONTEND_SHA256 = expected_frontend_sha256
    else:
        cfg.SHARK.UNIFIED.FRONTEND_CHECKPOINT = ""
    cfg.SEED = seed
    args._frontend_registration_path = frontend_registration_path
    args._frontend_registration = frontend_registration
    cfg.SHARK.GFM_BACKEND.TRAINABLE_PREFIXES = list(
        cfg.SHARK.GFM_BACKEND.TRAINABLE_PREFIXES
    ) + ["imlp."]
    experiment_id = (
        f"SHARK-{args.entity}-{args.frontend_init.upper()}-JOINT-SEED{seed}-{args.backend}"
    )
    cfg.SHARK.UNIFIED.EXPERIMENT_ID = experiment_id
    cfg.SHARK.GFM_BACKEND.EXPERIMENT_ID = experiment_id
    cfg.SHARK.UNIFIED.RUN_MODE = args.run_mode
    cfg.SHARK.SYNTHETIC.COUNTERFACTUAL_IMPLEMENTATION = "batched_precomputed"
    cfg.RESULT_DIR = str(args.output_dir.resolve())
    cfg.TRAIN.CHECKPOINT_DIR = cfg.RESULT_DIR
    cfg.SHARK.EVAL.RESULT_DIR = str((args.output_dir / "scores").resolve())
    cfg.DATA.BASE_DIR = (
        str(args.data_dir.resolve()) if args.data_dir
        else asset_path(args.assets, "data_dir")
    )
    cfg.ANOMALYGFM.CHECKPOINT_PATH = asset_path(args.assets, "anomalygfm_checkpoint", required=args.backend == "anomalygfm")
    cfg.ANOMALYGFM.SOURCE_MODEL_PATH = asset_path(args.assets, "anomalygfm_source", required=args.backend == "anomalygfm")
    cfg.SHARK.ROOT_PROP.CONTROLLED_EVAL_ENABLE = False
    if args.backend == "owleye":
        if args.owleye_checkpoint is None:
            raise ValueError("OWLEYE requires an explicit pretrained checkpoint")
        checkpoint = args.owleye_checkpoint.resolve()
        if _sha256(checkpoint) != CURRENT_OWLEYE_SHA256:
            raise ValueError("OWLEYE pretrained checkpoint SHA256 is not the verified current artifact")
        cfg.SHARK.GFM_BACKEND.CHECKPOINT_PATH = str(checkpoint)
        cfg.SHARK.GFM_BACKEND.CHECKPOINT_SHA256 = CURRENT_OWLEYE_SHA256
        cfg.SHARK.GFM_BACKEND.SOURCE_MODEL_PATH = asset_path(args.assets, "owleye_source")
    elif args.owleye_checkpoint is not None:
        raise ValueError("OWLEYE checkpoint supplied to AnomalyGFM run")
    if args.run_mode == "smoke":
        cfg.SOLVER.MAX_EPOCH = 1
        cfg.TRAIN.EVAL_PERIOD = 1
    if args.device == "cpu":
        cfg.DATA_LOADER.NUM_WORKERS = 0
        cfg.DATA_LOADER.PIN_MEMORY = False
    return cfg, source_path, source



def parameter_digest(module):
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()



def assert_joint_trainables(model, optimizer):
    if not any(p.requires_grad for p in model.imlp.parameters()):
        raise RuntimeError("DLinear is frozen in a joint run")
    if not any(p.requires_grad for p in model.gfm_adapter.parameters()):
        raise RuntimeError("Token Bridge is frozen in a joint run")
    if not any(p.requires_grad for p in model.information_residual.parameters()):
        raise RuntimeError("IPRA is frozen in a joint run")
    if not any(p.requires_grad for p in model.gfm.parameters()):
        raise RuntimeError("backend PEFT is frozen in a joint run")
    active = {id(p) for group in optimizer.param_groups for p in group["params"]}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and id(parameter) not in active:
            raise RuntimeError(f"trainable parameter absent from optimizer: {name}")
    native = set(model.gfm.adaptation_parameter_names())
    for name, parameter in model.gfm.named_parameters():
        if name not in native and parameter.requires_grad:
            raise RuntimeError(f"pretrained GFM backbone was unfrozen: {name}")



class JointTrainer(SharkTrainer):
    def train_step(self, inputs):
        from models.shark import trainer_shark as trainer_module

        windows, _ = trainer_module.prepare_inputs(inputs)
        loss_outputs = self._forward_loss(windows)
        loss = loss_outputs["total"]
        if not torch.isfinite(loss).all():
            raise FloatingPointError("nonfinite joint training loss")
        self.optimizer.zero_grad()
        loss.backward()
        invalid = [
            name for name, parameter in self.model.named_parameters()
            if parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
        ]
        if invalid:
            raise FloatingPointError(
                "nonfinite gradients before clipping/optimizer step: "
                + ", ".join(invalid[:20])
            )
        if self.cfg.SOLVER.GRADIENT_CLIP:
            norm = torch.nn.utils.clip_grad_norm_(
                optimizer_utils.get_trainable_params(self.model),
                self.cfg.SOLVER.GRADIENT_CLIP_NORM,
                error_if_nonfinite=True,
            )
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite joint gradient norm")
        self.optimizer.step()
        return {
            "metrics": (loss.detach(),),
            "losses": loss_outputs["losses"],
        }

    def train_epoch(self):
        super().train_epoch()
        if self.cfg.GRAPHBUILDER.REFERENCE_PRIOR_ENABLE:
            report = self.model.fit_graph_reference(
                self.train_loader,
                max_batches=self.cfg.GRAPHBUILDER.REFERENCE_PRIOR_MAX_BATCHES,
            )
            if not report.get("reference_prior_fitted", False):
                raise RuntimeError("updated DLinear graph reference was not fitted")
            # The normal score reference must describe the newly fitted graph.
            self.model.fit_evidence_calibrator(
                self.train_loader,
                max_batches=self.cfg.SHARK.HEAD.NORMAL_CALIBRATION_MAX_BATCHES,
            )
            print(
                "[joint reference] epoch="
                f"{self.cur_epoch + 1} windows={report['reference_prior_fit_windows']}",
                flush=True,
            )



def run(args):
    cfg, source_path, source = configure(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.txt").write_text(cfg.dump(), encoding="utf-8")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seeds(cfg.SEED)
    with ExitStack() as stack:
        train = profile.NormalTrainingDataset(cfg)
        val = profile.NormalTrainingDataset(cfg, "val")
        train_loader = profile.make_loader(cfg, train, cfg.TRAIN.BATCH_SIZE)
        val_loader = DataLoader(
            val, batch_size=cfg.VAL.BATCH_SIZE, shuffle=cfg.VAL.SHUFFLE,
            drop_last=cfg.VAL.DROP_LAST,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        )
        for split, loader in (("train", train_loader), ("val", val_loader), ("test", val_loader)):
            stack.enter_context(patch(f"trainer.get_{split}_dataloader", return_value=loader))
        stack.enter_context(patch(
            "models.shark.trainer_shark.prepare_inputs",
            lambda value: profile.prepare_on_device(value, device),
        ))
        model = InformationPreservingUnifiedSHARK(cfg).to(device)
        frontend_random_sha256 = parameter_digest(model.imlp)
        frontend_load_report = None
        if args.frontend_init == "pretrained":
            frontend_load_report = model.load_shared_frontend_checkpoint(
                cfg.SHARK.UNIFIED.FRONTEND_CHECKPOINT
            )
        frontend_before = parameter_digest(model.imlp)
        frontend_checkpoint_changed_initialization = (
            frontend_before != frontend_random_sha256
        )
        if (
            args.frontend_init == "pretrained"
            and not frontend_checkpoint_changed_initialization
        ):
            raise RuntimeError(
                "registered frontend checkpoint did not change DLinear initialization"
            )
        trainer = JointTrainer(cfg, model)
        assert_joint_trainables(model, trainer.optimizer)
        report = {
            "schema_version": "SHARK-MTS-JOINT-INIT-ABLATION-V1",
            "backend": args.backend,
            "entity": args.entity,
            "status": "TRAINING",
            "run_mode": args.run_mode,
            "epochs": cfg.SOLVER.MAX_EPOCH,
            "batch_size": cfg.TRAIN.BATCH_SIZE,
            "validation_period": int(cfg.TRAIN.EVAL_PERIOD),
            "seed": cfg.SEED,
            "source_config": str(source_path),
            "source_config_sha256": _sha256(source_path),
            "source_shark_checkpoint_loaded": False,
            "frontend_init": args.frontend_init,
            "source_frontend_checkpoint_loaded": frontend_load_report is not None,
            "source_frontend_checkpoint": (
                None if frontend_load_report is None
                else frontend_load_report["checkpoint"]
            ),
            "source_frontend_checkpoint_sha256": (
                None if frontend_load_report is None
                else _sha256(Path(frontend_load_report["checkpoint"]))
            ),
            "source_frontend_parameter_count": (
                0 if frontend_load_report is None
                else int(frontend_load_report["parameter_count"])
            ),
            "source_frontend_registration": (
                None if args._frontend_registration_path is None
                else str(args._frontend_registration_path)
            ),
            "source_frontend_registration_sha256": (
                None if args._frontend_registration_path is None
                else _sha256(args._frontend_registration_path)
            ),
            "source_frontend_registration_seed": (
                None if args._frontend_registration is None
                else int(args._frontend_registration["seed"])
            ),
            "pretrained_gfm_checkpoint": (
                cfg.SHARK.GFM_BACKEND.CHECKPOINT_PATH if args.backend == "owleye"
                else cfg.ANOMALYGFM.CHECKPOINT_PATH
            ),
            "pretrained_gfm_sha256": (
                CURRENT_OWLEYE_SHA256 if args.backend == "owleye"
                else _sha256(Path(cfg.ANOMALYGFM.CHECKPOINT_PATH))
            ),
            "owleye_historical_gfm_weight_reproduced": (
                None if args.backend != "owleye" else
                str(source.SHARK.GFM_BACKEND.CHECKPOINT_SHA256) == CURRENT_OWLEYE_SHA256
            ),
            "ipra_implementation_sha256": _sha256(ROOT / "models/shark/information_preserving_residual.py"),
            "frontend_random_sha256_before_load": frontend_random_sha256,
            "frontend_sha256_before": frontend_before,
            "frontend_checkpoint_changed_initialization": (
                frontend_checkpoint_changed_initialization
            ),
            "trainable_names": sorted(n for n, p in model.named_parameters() if p.requires_grad),
            "graph_reference_refit_each_epoch": bool(cfg.GRAPHBUILDER.REFERENCE_PRIOR_ENABLE),
            "checkpoint_selection": "train-normal synthetic validation loss",
            "counterfactual_implementation": cfg.SHARK.SYNTHETIC.COUNTERFACTUAL_IMPLEMENTATION,
            "test_labels_used_for_training_or_selection": False,
        }
        (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.gradient_audit_only:
            from models.shark import trainer_shark as trainer_module

            trainer.cur_epoch = 0
            trainer.cur_iter = 1
            model.train()
            batch = next(iter(trainer.train_loader))
            windows, _ = trainer_module.prepare_inputs(batch)
            trainer.optimizer.zero_grad()
            try:
                with torch.autograd.detect_anomaly():
                    loss_outputs = trainer._forward_loss(windows)
                    loss_outputs["total"].backward()
            except RuntimeError as error:
                report.update(status="GRADIENT_AUDIT_FAILED", gradient_error=str(error))
                (args.output_dir / "audit.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                raise
            invalid = [
                name for name, parameter in model.named_parameters()
                if parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ]
            report.update(
                status="GRADIENT_AUDIT_COMPLETE",
                loss=float(loss_outputs["total"].detach().cpu()),
                nonfinite_gradients=invalid,
                frontend_gradient_count=sum(
                    p.grad is not None for p in model.imlp.parameters()
                ),
                optimizer_steps=0,
            )
            (args.output_dir / "audit.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"[joint init gradient audit] init={args.frontend_init} "
                f"{report['status']} invalid={invalid[:20]}",
                flush=True,
            )
            return report
        trainer.train()
        best = torch.load(args.output_dir / "checkpoint_best.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(best["model_state"], strict=True)
        frontend_after = parameter_digest(model.imlp)
        if frontend_after == frontend_before:
            raise RuntimeError("best joint checkpoint did not update DLinear")
        if cfg.SHARK.HEAD.NORMAL_CALIBRATION_REFIT_AT_TEST:
            model.fit_evidence_calibrator(
                train_loader, max_batches=cfg.SHARK.HEAD.NORMAL_CALIBRATION_MAX_BATCHES,
            )
        reference = save_evaluation_reference(model, args.output_dir / "evaluation_reference.pt")
        model._shark_normal_mechanism = trainer.augmentor.normal_mechanism
        torch.save(trainer.augmentor.normal_mechanism, args.output_dir / "normal_mechanism.pt")
        InformationPreservingPredictor(cfg, model).predict()
        metric_path = Path(cfg.SHARK.EVAL.RESULT_DIR) / args.entity / "test_metric_protocol.json"
        native_metrics = json.loads(metric_path.read_text(encoding="utf-8"))["metrics"]
        if args.backend == "owleye":
            tail_dir = args.output_dir / "tail_protocol"
            subprocess.run([
                sys.executable, "-u", "-m",
                "evaluation.tail",
                "--source-dir", str(metric_path.parent),
                "--entity", args.entity,
                "--output-dir", str(tail_dir),
                "--source-arm", f"{args.frontend_init}_joint_backend_v3",
            ], cwd=ROOT, check=True)
            tail = json.loads((tail_dir / "report.json").read_text(encoding="utf-8"))
            values = tail["metrics"]["f1d_train_tail_gated_consensus"]
            metrics = {
                "AUROC": float(values["auroc"]),
                "AUPRC": float(values["auprc_trapezoid"]),
                "F1": float(values["raw_curve"]["f1"]),
            }
            score_protocol = "registered_f1d_train_tail_gated_consensus"
        else:
            metrics = native_metrics
            score_protocol = "native_unified_node_to_time"
        report.update({
            "status": "COMPLETE",
            "completed_epochs": int(trainer.cur_epoch) + 1,
            "selected_epoch_1based": int(best["epoch"]) + 1,
            "frontend_sha256_after": frontend_after,
            "frontend_updated": True,
            "evaluation_reference_artifact": reference,
            "native_metrics": native_metrics,
            "score_protocol": score_protocol,
            "metrics": metrics,
        })
        (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"[joint init ablation] init={args.frontend_init} entity={args.entity} "
            f"backend={args.backend} mode={args.run_mode} "
            f"metrics={report['metrics']} output={args.output_dir}",
            flush=True,
        )
        return report
