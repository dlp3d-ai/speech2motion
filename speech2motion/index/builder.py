from .dict_index_mapping import DictIndexMapping

_INDEX_MAPPINGS = dict(
    DictIndexMapping=DictIndexMapping)


def build_index_mapping(cfg: dict) -> DictIndexMapping:
    """Build an index mapping instance from configuration dictionary.

    Args:
        cfg (dict):
            Configuration dictionary, must contain 'type' key to specify
            index mapping type. Other key-value pairs will be passed as
            parameters to the index mapping class constructor.

    Returns:
        DictIndexMapping:
            Built index mapping instance.

    Raises:
        TypeError:
            Raised when the index mapping type corresponding to the 'type' key
            in configuration does not exist.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop('type')
    if cls_name not in _INDEX_MAPPINGS:
        msg = f'Unknown index mapping type: {cls_name}'
        raise TypeError(msg)
    ret_inst = _INDEX_MAPPINGS[cls_name](**cfg)
    return ret_inst

