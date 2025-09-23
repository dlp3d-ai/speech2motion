import json

from ...utils.hash import str_to_md5
from .filesystem_motion_reader import FilesystemMotionReader
from .minio_motion_reader import MinioMotionReader
from .minio_mysql_motion_reader import MinioMySQLMotionReader
from .sqlite_filesystem_motion_reader import SQLiteFilesystemMotionReader

_motion_readers_built = dict()

_MOTION_READER = dict(
    MinioMySQLMotionReader=MinioMySQLMotionReader,
    MinioMotionReader=MinioMotionReader,
    FilesystemMotionReader=FilesystemMotionReader,
    SQLiteFilesystemMotionReader=SQLiteFilesystemMotionReader)


def build_motion_reader(
        cfg: dict
        ) -> MinioMotionReader | MinioMySQLMotionReader |\
             FilesystemMotionReader | SQLiteFilesystemMotionReader:
    """Build a motion reader instance from configuration dictionary.

    This function uses the configuration hash value to cache built instances,
    avoiding duplicate creation of motion reader objects with the same
    configuration. If a motion reader instance with the same configuration
    already exists, the cached instance is returned directly.

    Args:
        cfg (dict):
            Motion reader configuration dictionary, must contain 'type' key
            to specify the reader type. Other key-value pairs will be passed
            as parameters to the reader class constructor.

    Returns:
        MinioMotionReader | MinioMySQLMotionReader |
        FilesystemMotionReader | SQLiteFilesystemMotionReader:
            Built motion reader instance.

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
    if cfg_hash in _motion_readers_built:
        return _motion_readers_built[cfg_hash]
    cfg = cfg.copy()
    cls_name = cfg.pop('type')
    if cls_name not in _MOTION_READER:
        msg = f'Unknown motion reader type: {cls_name}'
        raise TypeError(msg)
    ret_inst = _MOTION_READER[cls_name](**cfg)
    _motion_readers_built[cfg_hash] = ret_inst
    return ret_inst
