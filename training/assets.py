"""Explicit external assets and immutable configuration identities."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = {(s['backend'], s['entity']): dict(s, config_sha256=s['sha256'])
         for s in json.loads((ROOT / 'configs/sources.json').read_text())['sources']}
CURRENT_OWLEYE_SHA256 = 'fd35544a57e8e6ffa271ad0e68719fdec8648fd2a2d08c7db5de7f62ebc57fa2'
ANOMALYGFM_SHA256 = 'b6783375a0c340acd2b5ee42bae40280b7a2adb5dc27a82b8ca0504ac8671413'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_assets(path):
    path = Path(path).resolve()
    values = json.loads(path.read_text(encoding='utf-8'))
    resolved = {}
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Asset {name} needs a nonempty path')
        p = Path(value).expanduser()
        resolved[name] = str((p if p.is_absolute() else path.parent / p).resolve())
    return resolved


def asset_path(assets, key, required=True):
    value = (assets or {}).get(key)
    if value is None:
        if required:
            raise ValueError(f'Missing asset: {key}')
        return ''
    p = Path(value)
    if required and not p.exists():
        raise FileNotFoundError(f'{key}: {p}')
    return str(p.resolve())


def verify_assets(assets, backend, entity):
    spec = SPECS[(backend, entity)]
    snapshot = ROOT / spec['snapshot']
    if sha256(snapshot) != spec['config_sha256']:
        raise ValueError('Registered configuration changed: ' + str(snapshot))
    data = Path(asset_path(assets, 'data_dir'))
    if entity.startswith('SMD_'):
        paths = [data / 'ServerMachineDataset/preprocessed' / f'machine-1-1_{suffix}.pkl'
                 for suffix in ('train', 'test', 'test_label')]
    else:
        name = entity.split('_', 1)[1]
        paths = [data / 'SMAP_MSL' / split / f'{name}.npy' for split in ('train', 'test')]
        paths += [data / 'SMAP_MSL' / 'test' / 'labeled_anomalies.csv']
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    key = 'anomalygfm_checkpoint' if backend == 'anomalygfm' else 'owleye_checkpoint'
    weight = Path(asset_path(assets, key))
    expected = ANOMALYGFM_SHA256 if backend == 'anomalygfm' else CURRENT_OWLEYE_SHA256
    observed = sha256(weight)
    if observed != expected:
        raise ValueError(f'{key} SHA256 mismatch: {observed}')
    if backend == 'owleye':
        asset_path(assets, 'owleye_source')  # model wrapper verifies its pinned SHA256
    else:
        from models.anomalygfm.source import verified_source
        verified_source(asset_path(assets, 'anomalygfm_source'))
    return {'entity': entity, 'backend': backend, 'config_sha256': sha256(snapshot),
            'checkpoint_sha256': observed, 'data_files_present': len(paths)}
