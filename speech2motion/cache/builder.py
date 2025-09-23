import json

from ..utils.hash import str_to_md5
from .local_cache import LocalCache

_CACHE = dict(
    LocalCache=LocalCache)

_cache_built = dict()

def build_cache(cfg: dict) -> LocalCache:
    """Build cache instance based on configuration dictionary.

    This function uses configuration hash to cache built instances, avoiding
    duplicate creation of cache objects with the same configuration.
    If a cache instance with the same configuration already exists,
    the cached instance is returned directly.

    Args:
        cfg (dict):
            Cache configuration dictionary, must contain 'type' key to specify
            cache type. Other key-value pairs will be passed as parameters
            to the cache class constructor.

    Returns:
        LocalCache:
            Built cache instance.

    Raises:
        TypeError:
            Raised when the cache type corresponding to the 'type' key
            in configuration does not exist.
    """
    serializable_cfg = dict()
    for key, value in cfg.items():
        if key == 'logger_cfg' and isinstance(value, dict):
            value_wo_logger_name = value.copy()
            value_wo_logger_name.pop('logger_name', None)
            serializable_cfg[key] = value_wo_logger_name
        else:
            serializable_cfg[key] = str(value)
    cfg_hash = str_to_md5(json.dumps(serializable_cfg))
    if cfg_hash in _cache_built:
        return _cache_built[cfg_hash]['inst']
    cfg = cfg.copy()
    cls_name = cfg.pop('type')
    if cls_name not in _CACHE:
        msg = f'Unknown cache type: {cls_name}'
        raise TypeError(msg)
    ret_inst = _CACHE[cls_name](**cfg)
    _cache_built[cfg_hash] = dict(
        cfg=cfg,
        inst=ret_inst
    )
    return ret_inst
