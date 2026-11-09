"""Read legacy SHARK artifacts without changing their files or tensor keys."""
import io
import pickle
import types


def normalize_legacy_config(source):
    """Return a renamed copy; paths and numerical settings remain untouched."""
    cfg = source.clone()
    frozen = cfg.is_frozen()
    cfg.defrost()
    if 'SHARK' in cfg:
        if 'GFM4MTSAD' in cfg:
            raise ValueError('Ambiguous config: both SHARK and GFM4MTSAD are defined')
        cfg.GFM4MTSAD = cfg.pop('SHARK')
    if 'MODEL' in cfg and cfg.MODEL.get('NAME') == 'SHARK':
        cfg.MODEL.NAME = 'GFM4MTSAD'
    framework = cfg.get('GFM4MTSAD', {})
    head = framework.get('HEAD', {})
    if head.get('NAME') == 'SharkHead':
        head.NAME = 'GFM4MTSADHead'
    evaluation = framework.get('EVAL', {})
    if evaluation.get('SCORE_SOURCE') == 'shark_head':
        evaluation.SCORE_SOURCE = 'gfm4mtsad_head'
    if frozen:
        cfg.freeze()
    return cfg


class _ArtifactUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == (
            'models.shark.normal_mechanism', 'GlobalGraphVARMechanism'
        ):
            from models.gfm4mtsad.normal_mechanism import GlobalGraphVARMechanism
            return GlobalGraphVARMechanism
        return super().find_class(module, name)


def _load(handle, **kwargs):
    return _ArtifactUnpickler(handle, **kwargs).load()


def _loads(data, **kwargs):
    return _load(io.BytesIO(data), **kwargs)


_artifact_pickle = types.ModuleType('gfm4mtsad_artifact_pickle')
_artifact_pickle.Unpickler = _ArtifactUnpickler
_artifact_pickle.Pickler = pickle.Pickler
_artifact_pickle.load = _load
_artifact_pickle.loads = _loads
_artifact_pickle.dump = pickle.dump
_artifact_pickle.dumps = pickle.dumps


def load_normal_mechanism(path, map_location='cpu'):
    """Load a trusted, bundle-verified normal reference from either namespace."""
    import torch
    return torch.load(path, map_location=map_location,
                      pickle_module=_artifact_pickle, weights_only=False)
