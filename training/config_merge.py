"""Merge registered namespaced configuration without dropping fields."""

def _merge_verified_source_cfg(cfg, source_cfg):
    """Merge a hash-pinned legacy dump without leaving YACS unlocked."""
    cfg.set_new_allowed(True)
    try:
        cfg.merge_from_other_cfg(source_cfg)
    finally:
        cfg.set_new_allowed(False)
