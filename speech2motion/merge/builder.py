from .blending import Blending
from .concatenation import Concatenation
from .interpolation import Interpolation

_MOTION_CLIP_MERGE = dict(
    Concatenation=Concatenation,
    Interpolation=Interpolation,
    Blending=Blending,
)


def build_motion_clip_merge(cfg: dict) -> Concatenation | Interpolation | Blending:
    """Build motion clip merge instance based on configuration dictionary.

    Args:
        cfg (dict):
            Configuration dictionary containing merge type and parameters.
            Must include 'type' field specifying the merge type.

    Returns:
        Concatenation | Interpolation | Blending:
            Motion clip merge instance built according to configuration.

    Raises:
        TypeError:
            Raised when the specified merge type is not in the supported
            type list.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop('type')
    if cls_name not in _MOTION_CLIP_MERGE:
        msg = f'Unknown motion clip merge type: {cls_name}'
        raise TypeError(msg)
    ret_inst = _MOTION_CLIP_MERGE[cls_name](**cfg)
    return ret_inst
