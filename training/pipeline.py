"""Stage orchestration. Mathematical model/loss implementations remain unchanged."""
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
from training.assets import ROOT, SPECS, sha256
from training import data, frontend, joint
from training.registration import build_registration


def write_json(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2) + '\n').encode('utf-8'))


def joint_args(args, output, registration):
    return SimpleNamespace(entity=args.entity, backend=args.backend, seed=args.seed,
        frontend_init='pretrained', frontend_registration=Path(registration),
        run_mode='primary', device=args.device, threads=args.threads,
        output_dir=Path(output), assets=args.assets,
        data_dir=Path(args.assets['data_dir']),
        owleye_checkpoint=(Path(args.assets['owleye_checkpoint']) if args.backend == 'owleye' else None),
        gradient_audit_only=False)


def frontend_config(args, output):
    spec = SPECS[(args.backend, args.entity)]
    return frontend.build_current_config(ROOT / spec['snapshot'], entity=args.entity,
        backend=args.backend, seed=args.seed, frontend_root=Path(output) / 'checkpoints',
        eval_root=Path(output) / 'unused_evaluation', data_dir=Path(args.assets['data_dir']),
        owleye_checkpoint=(Path(args.assets['owleye_checkpoint']) if args.backend == 'owleye' else None),
        assets=args.assets)


def pretrain(args, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg, report = frontend_config(args, output)
    # A normal-only loader reads precisely the same chronological train/val
    # splits, without opening test arrays during frontend learning. Parity is
    # tested against the inherited loaders before promotion of this extraction.
    if args.device == 'cpu':
        cfg.DATA_LOADER.NUM_WORKERS = 0
        cfg.DATA_LOADER.PIN_MEMORY = False
    train = data.NormalTrainingDataset(cfg)
    val = data.NormalTrainingDataset(cfg, 'val')
    train_loader = data.make_loader(cfg, train, cfg.TRAIN.BATCH_SIZE)
    val_loader = DataLoader(val, batch_size=cfg.VAL.BATCH_SIZE, shuffle=cfg.VAL.SHUFFLE,
        drop_last=cfg.VAL.DROP_LAST, num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY)
    device = torch.device(args.device)
    with ExitStack() as stack:
        for split, loader in (('train', train_loader), ('val', val_loader), ('test', val_loader)):
            stack.enter_context(patch(f'trainer.get_{split}_dataloader', return_value=loader))
        # The extracted frontend's inner compatibility patch resolves this name.
        stack.enter_context(patch('training.frontend.get_val_dataloader', return_value=val_loader))
        stack.enter_context(patch('models.gfm4mtsad.trainer_gfm4mtsad.prepare_inputs',
                                 lambda values: data.prepare_on_device(values, device)))
        frontend.train_frontend(cfg, report, config_output=output/'config.txt',
            report_output=output/'report.json', threads=args.threads, device_override=args.device)
    reg = build_registration(report['checkpoint'], entity=args.entity, backend=args.backend,
        window=60, source_config=output/'config.txt', preprocessing_protocol='train_only', seed=args.seed)
    reg['checkpoint'] = str(Path(report['checkpoint']).relative_to(output)).replace('\\', '/')
    write_json(output/'frontend_registration.json', reg)
    return output/'frontend_registration.json'


def seal_bundle(path):
    path = Path(path)
    names = ('config.txt', 'audit.json', 'checkpoint_best.pth',
             'evaluation_reference.pt', 'normal_mechanism.pt')
    write_json(path/'bundle.json', {'schema_version': 1,
        'files': {name: sha256(path/name) for name in names}})


def execute(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    if args.command == 'pretrain':
        pretrain(args, output)
    elif args.command == 'train':
        joint.run(joint_args(args, output, args.frontend_registration))
        seal_bundle(output)
    elif args.command == 'run':
        output.mkdir(parents=True, exist_ok=False)
        registration = pretrain(args, output/'frontend')
        joint.run(joint_args(args, output/'joint', registration))
        seal_bundle(output/'joint')
    elif args.command == 'evaluate':
        from evaluation.evaluate import evaluate
        evaluate(args)
