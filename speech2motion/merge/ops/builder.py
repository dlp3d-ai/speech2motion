from .linear_interpolate_ops import LinearInterpolateOps

_MERGE_OPS = dict(
    LinearInterpolateOps=LinearInterpolateOps)


def build_merge_ops(cfg: dict) -> LinearInterpolateOps:
    """Build a merge ops instance from a configuration dictionary."""
    cfg = cfg.copy()
    cls_name = cfg.pop('type')
    if cls_name not in _MERGE_OPS:
        msg = f'Unknown merge ops type: {cls_name}'
        raise TypeError(msg)
    ret_inst = _MERGE_OPS[cls_name](**cfg)
    return ret_inst
