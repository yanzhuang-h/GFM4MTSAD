"""Evaluate a sealed joint run with its saved normal references, without fitting."""
import json
from pathlib import Path
import subprocess
import sys
import torch
from yacs.config import CfgNode
from training.assets import ROOT, sha256, asset_path
from training.compatibility import normalize_legacy_config, load_normal_mechanism
from training.data import nested_digest
from training.pipeline import write_json
from training.joint import InformationPreservingUnifiedGFM4MTSAD
from evaluation.predictor import InformationPreservingPredictor
from utils.misc import set_seeds


def evaluate(args):
    run = args.run_dir.resolve()
    bundle = json.loads((run/'bundle.json').read_text())
    required = {'config.txt', 'audit.json', 'checkpoint_best.pth',
                'evaluation_reference.pt', 'normal_mechanism.pt'}
    if set(bundle['files']) != required:
        raise ValueError('Incomplete inference bundle')
    for name, expected in bundle['files'].items():
        if sha256(run/name) != expected:
            raise ValueError('Changed inference artifact: ' + name)
    audit = json.loads((run/'audit.json').read_text())
    if (audit['status'], audit['entity'], audit['backend']) != ('COMPLETE', args.entity, args.backend):
        raise ValueError('Run identity mismatch')
    cfg = CfgNode.load_cfg((run/'config.txt').read_text())
    checkpoint = torch.load(run/'checkpoint_best.pth', map_location='cpu', weights_only=False)
    if CfgNode.load_cfg(checkpoint['cfg']).dump() != cfg.dump():
        raise ValueError('Checkpoint/config mismatch')
    if int(checkpoint['epoch']) + 1 != audit['selected_epoch_1based']:
        raise ValueError('Checkpoint epoch mismatch')
    cfg = normalize_legacy_config(cfg)
    cfg.DATA.BASE_DIR = asset_path(args.assets, 'data_dir')
    cfg.ANOMALYGFM.CHECKPOINT_PATH = asset_path(args.assets, 'anomalygfm_checkpoint', required=args.backend == 'anomalygfm')
    cfg.ANOMALYGFM.SOURCE_MODEL_PATH = asset_path(args.assets, 'anomalygfm_source', required=args.backend == 'anomalygfm')
    if args.backend == 'owleye':
        cfg.GFM4MTSAD.GFM_BACKEND.CHECKPOINT_PATH = asset_path(args.assets, 'owleye_checkpoint')
        cfg.GFM4MTSAD.GFM_BACKEND.SOURCE_MODEL_PATH = asset_path(args.assets, 'owleye_source')
    cfg.GFM4MTSAD.EVAL.RESULT_DIR = str(args.output.resolve()/'scores')
    cfg.DATA_LOADER.NUM_WORKERS = 0
    cfg.DATA_LOADER.PIN_MEMORY = False
    torch.set_num_threads(args.threads)
    set_seeds(cfg.SEED)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model = InformationPreservingUnifiedGFM4MTSAD(cfg).to(device)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    reference = torch.load(run/'evaluation_reference.pt', map_location='cpu', weights_only=False)
    for name, value in reference.items():
        if not hasattr(model.head, name):
            raise ValueError('Unknown saved reference: ' + name)
        setattr(model.head, name, value.to(device))
    model._bridge_initialized = True
    model._gfm4mtsad_normal_mechanism = load_normal_mechanism(run/'normal_mechanism.pt')
    model.eval().requires_grad_(False)
    before = nested_digest((model.state_dict(), {n:getattr(model.head,n) for n in reference}))
    args.output.mkdir(parents=True, exist_ok=False)
    with torch.no_grad():
        InformationPreservingPredictor(cfg, model).predict()
    after = nested_digest((model.state_dict(), {n:getattr(model.head,n) for n in reference}))
    if before != after:
        raise RuntimeError('Inference changed weights or saved reference tensors')
    scores = args.output/'scores'/args.entity
    native = json.loads((scores/'test_metric_protocol.json').read_text())['metrics']
    if args.backend == 'owleye':
        subprocess.run([sys.executable, '-m', 'evaluation.tail', '--source-dir', str(scores),
            '--entity', args.entity, '--output-dir', str(args.output/'tail_protocol'),
            '--source-arm', 'sealed_warm_joint'], cwd=ROOT, check=True)
        values = json.loads((args.output/'tail_protocol/report.json').read_text())['metrics']['f1d_train_tail_gated_consensus']
        metrics = dict(AUROC=values['auroc'], AUPRC=values['auprc_trapezoid'], F1=values['raw_curve']['f1'])
    else:
        metrics = native
    write_json(args.output/'evaluation.json', dict(status='COMPLETE', entity=args.entity,
        backend=args.backend, seed=cfg.SEED, metrics=metrics, reference_refitted=False,
        source_bundle_sha256=sha256(run/'bundle.json'), state_unchanged=before == after))
