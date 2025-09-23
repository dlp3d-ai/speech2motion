import json

from ...utils.hash import str_to_md5
from .filesystem_restpose_reader import FilesystemRestposeReader
from .minio_restpose_reader import MinioRestposeReader

_restpose_readers_built = dict()

_RESTPOSE_READER = dict(
    MinioRestposeReader=MinioRestposeReader,
    FilesystemRestposeReader=FilesystemRestposeReader)


def build_restpose_reader(cfg: dict) -> MinioRestposeReader | FilesystemRestposeReader:
    """Build a restpose reader instance based on configuration dictionary.

    This function uses the configuration hash value to cache built instances,
    avoiding duplicate creation of restpose reader objects with the same
    configuration. If a restpose reader instance with the same configuration
    already exists, the cached instance is returned directly.

    Args:
        cfg (dict):
            Restpose reader configuration dictionary, must contain 'type' key
            to specify the reader type. Other key-value pairs will be passed
            as parameters to the reader class constructor.

    Returns:
        MinioRestposeReader | FilesystemRestposeReader:
            Built restpose reader instance.

    Raises:
        TypeError:
            Raised when the reader type specified by the 'type' key in the
            configuration does not exist.
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
    if cfg_hash in _restpose_readers_built:
        return _restpose_readers_built[cfg_hash]
    cfg = cfg.copy()
    cls_name = cfg.pop('type')
    if cls_name not in _RESTPOSE_READER:
        msg = f'Unknown restpose reader type: {cls_name}'
        raise TypeError(msg)
    ret_inst = _RESTPOSE_READER[cls_name](**cfg)
    _restpose_readers_built[cfg_hash] = ret_inst
    return ret_inst
