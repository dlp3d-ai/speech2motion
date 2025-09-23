import json
from abc import ABC, abstractmethod
from typing import Any

from ...data_structures.motion_record import MotionRecord
from ...utils.super import Super


class BaseMetaReader(Super, ABC):
    """Base class for meta data readers of motion records.

    This abstract base class defines the interface for reading motion record
    metadata from various data sources. It provides common functionality and
    defines abstract methods that must be implemented by concrete subclasses.

    Subclasses should implement the abstract methods to provide specific
    database access functionality (e.g., MySQL, SQLite).
    """

    def __init__(self,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the base meta reader.

        Args:
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        ABC.__init__(self)
        Super.__init__(self, logger_cfg)

    @abstractmethod
    async def get_version(self) -> str:
        """Get the version of motion database metadata.

        Returns:
            str: Version string representing the database metadata version.
        """
        pass

    @abstractmethod
    async def get_ids(self) -> list[int]:
        """Get all motion_record_id from metadata.

        Returns:
            list[int]: List of motion record IDs.
        """
        pass

    @abstractmethod
    async def get_meta_by_id(self, id: int) -> dict[str, Any]:
        """Get metadata by ID.

        Args:
            id (int): MotionRecord ID to retrieve metadata for.

        Returns:
            dict[str, Any]: Type-converted metadata dictionary.
        """
        raise NotImplementedError(
            'get_meta_by_id() is not implemented in BaseMetaSync.')

    async def get_motion_record_by_id(self, id: int) -> MotionRecord:
        """Get MotionRecord object by ID.

        Args:
            id (int): MotionRecord ID to retrieve.

        Returns:
            MotionRecord: MotionRecord object created from metadata.
        """
        meta = await self.get_meta_by_id(id)
        return MotionRecord.from_dict(meta, self.logger_cfg)

def _convert_sql_set_to_list(sql_set: str) -> list[str]:
    """Convert MySQL SET type string to list.

    Args:
        sql_set (str): Comma-separated string from MySQL SET type.

    Returns:
        list[str]: List of values from the SET string.
    """
    return sql_set.split(',')

MDB_COLUMNS_BY_TYPE = dict(
    int=dict(
        type_cast=int,
        columns=[
            'motion_record_id',
            'n_frames',
            'startup_frame',
            'recovery_frame',
            'motion_keyword_frame',
            'speech_keyword_frame'
        ]),
    str=dict(
        type_cast=str,
        columns=[
            'avatar_name',
            'record_type',
            'label',
            'label_cn',
            'keywords',
            'keywords_cn',
            'speech_keywords',
            'speech_keywords_cn',
            'input_path',
        ]),
    bool=dict(
        type_cast=bool,
        columns=['is_idle_long']),
    float=dict(type_cast=float, columns=[
        'fps',
    ]),
    json=dict(
        type_cast=json.loads,
        columns=[
            'cutoff_frames',
            'cutoff_ranges']),
    sql_set=dict(
        type_cast=_convert_sql_set_to_list,
        columns=[
            'motion_keywords_ch',
            'speech_keywords_ch',
            'states',
            'labels',
        ]),
)
MDB_COLUMNS_BY_COLOMN = dict()
for type_name, type_dict in MDB_COLUMNS_BY_TYPE.items():
    for column_name in type_dict['columns']:
        MDB_COLUMNS_BY_COLOMN[column_name] = dict(
            type_name=type_name,
            type_cast=type_dict['type_cast'],
        )
