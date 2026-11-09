"""The two registered GFM backends; wrappers retain their research semantics."""
from models.gfm4mtsad.anomalygfm_backend import AnomalyGFMHighDimBackend
from models.gfm4mtsad.gfm_backend import GFMBackendContract
from models.gfm4mtsad.owleye_backend import OWLEYEBackend


def build_gfm_backend(cfg, source_feature_dim):
    builders = {
        'AnomalyGFM': lambda: AnomalyGFMHighDimBackend(cfg, high_dim_input=int(source_feature_dim)),
        'OWLEYE': lambda: OWLEYEBackend.from_config(cfg),
    }
    name = str(cfg.GFM4MTSAD.GFM)
    if name not in builders:
        raise ValueError(f'Unsupported backend: {name}; choose AnomalyGFM or OWLEYE')
    backend = builders[name]()
    if not isinstance(backend, GFMBackendContract):
        raise TypeError(f'Backend {name} violates GFMBackendContract')
    return backend
