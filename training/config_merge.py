"""Merge registered namespaced configuration without dropping fields."""


def require_main_recipe(cfg):
    """Reject archived research branches instead of silently changing behavior."""
    settings = cfg.GFM4MTSAD
    for group in ('OTRCL', 'RELATION_RESIDUAL_CONTRAST', 'LOCAL_GLOBAL_CONTRAST',
                  'DECOUPLED_LOCAL_GLOBAL_CONTRAST', 'NATIVE_EVIDENCE_MECHANISM_CONTRAST',
                  'TAIL_PRESERVING_WINDOW_CONTRAST'):
        if getattr(getattr(settings, group, None), 'ENABLE', False):
            raise ValueError(f'{group} is not part of the released main recipe')
    for name, expected in (('CANDIDATE_GENERATOR', 'graph_var'),
                           ('NEGATIVE_SCREENING_MODE', 'legacy_boundary')):
        if getattr(settings.SYNTHETIC, name, expected) != expected:
            raise ValueError(f'The released main recipe requires {name}={expected}')

def _merge_verified_source_cfg(cfg, source_cfg):
    """Merge a hash-pinned legacy dump without leaving YACS unlocked."""
    cfg.set_new_allowed(True)
    try:
        cfg.merge_from_other_cfg(source_cfg)
    finally:
        cfg.set_new_allowed(False)
