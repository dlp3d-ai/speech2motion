import json

from ...utils.hash import str_to_md5
from .mysql_meta_reader import MySQLMetaReader
from .sqlite_meta_reader import SQLiteMetaReader

_META_READER = dict(
    MySQLMetaReader=MySQLMetaReader,
    SQLiteMetaReader=SQLiteMetaReader)

_meta_readers_built = dict()

def build_meta_reader(cfg: dict) -> MySQLMetaReader | SQLiteMetaReader:
    """Build meta reader instance based on configuration dictionary.

    This function uses configuration hash values to cache built instances,
    avoiding duplicate creation of meta reader objects with the same
    configuration. If a meta reader instance with the same configuration
    already exists, it returns the cached instance directly.

    Args:
        cfg (dict):
            Meta reader configuration dictionary that must contain a 'type'
            key specifying the reader type. Other key-value pairs will be
            passed as parameters to the reader class constructor.

    Returns:
        MySQLMetaReader | SQLiteMetaReader:
            Built meta reader instance.

    Raises:
        TypeError:
            Raised when the 'type' key in configuration corresponds to a
            non-existent reader type.
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
    if cfg_hash in _meta_readers_built:
        return _meta_readers_built[cfg_hash]
    cfg = cfg.copy()
    cls_name = cfg.pop('type')
    if cls_name not in _META_READER:
        msg = f'Unknown meta reader type: {cls_name}'
        raise TypeError(msg)
    ret_inst = _META_READER[cls_name](**cfg)
    _meta_readers_built[cfg_hash] = ret_inst
    return ret_inst
