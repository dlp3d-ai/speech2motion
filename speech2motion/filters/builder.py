from .avatar_filter import AvatarFilter
from .base_filter import BaseFilter
from .cutoff_duration_filter import CutoffDurationFilter
from .duration_filter import DurationFilter
from .keyword_align_filter import KeywordAlignFilter
from .keyword_filter import KeywordFilter
from .label_filter import LabelFilter
from .memory_filter import MemoryFilter
from .random_filter import RandomFilter
from .type_filter import TypeFilter

_FILTERS = dict(
    AvatarFilter=AvatarFilter,
    CutoffDurationFilter=CutoffDurationFilter,
    DurationFilter=DurationFilter,
    KeywordFilter=KeywordFilter,
    MemoryFilter=MemoryFilter,
    RandomFilter=RandomFilter,
    TypeFilter=TypeFilter,
    KeywordAlignFilter=KeywordAlignFilter,
    LabelFilter=LabelFilter,
)


def build_filter(cfg: dict) -> BaseFilter:
    """Build a filter instance from configuration dictionary.

    Args:
        cfg (dict):
            Filter configuration dictionary, must contain 'type' key to specify
            filter type. Other key-value pairs will be passed as parameters
            to the filter class constructor.

    Returns:
        BaseFilter:
            Built filter instance.

    Raises:
        TypeError:
            Raised when the filter type corresponding to the 'type' key
            in configuration does not exist.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop('type')
    if cls_name not in _FILTERS:
        msg = f'Unknown filter type: {cls_name}'
        raise TypeError(msg)
    ret_inst = _FILTERS[cls_name](**cfg)
    return ret_inst

