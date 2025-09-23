
import numpy as np

from ..data_structures.motion_record import MotionRecord, MotionRecordType
from ..index.base_index_mapping import BaseIndexMapping
from .base_filter import BaseFilter


class TypeFilter(BaseFilter):
    """Filter for filtering motion_records based on motion record type.
    """

    def __init__(self,
                 mapping: BaseIndexMapping,
                 logger_cfg: None | dict = None) -> None:
        """Initialize type filter.

        Args:
            mapping (BaseIndexMapping):
                Index mapping instance that maps motion record types to a set of IDs.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name. Defaults to None.
        """
        super().__init__(logger_cfg=logger_cfg)
        self.mapping = mapping

    async def reset_mapping(self, mapping: BaseIndexMapping) -> None:
        """Reset index mapping instance.

        Args:
            mapping (BaseIndexMapping):
                New index mapping instance.
        """
        self.mapping = mapping

    async def filter(self, motion_records: dict[int, MotionRecord],
                     motion_record_type: MotionRecordType,
               **kwargs) -> dict[int, MotionRecord]:
        """Filter motion_records based on motion record type.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be filtered, key is motion_record id,
                value is motion_record instance.
            motion_record_type (MotionRecordType):
                Motion record type to filter by.

        Returns:
            dict[int, MotionRecord]:
                Remaining motion_records after filtering, same format as input
                motion_records but possibly reduced in number. Returns empty
                dictionary if no matching motion_record is found.

        Raises:
            TypeError:
                Raised when motion_record_type is not of type MotionRecordType.
        """
        if len(motion_records) == 0:
            msg = 'Input motion_records is empty.'
            self.logger.warning(msg)
            return dict()
        if not isinstance(motion_record_type, MotionRecordType):
            msg = ('motion_record_type must be of type MotionRecordType, ' +
                   f'current type is {type(motion_record_type)}.')
            self.logger.error(msg)
            raise TypeError(msg)
        try:
            available_ids = await self.mapping.get_items(
                motion_record_type.value)
        except KeyError:
            available_ids = set()
        input_ids = set(motion_records.keys())
        filtered_ids = available_ids.intersection(input_ids)
        ret_dict = dict()
        for k in filtered_ids:
            ret_dict[k] = motion_records[k]
        self.logger.debug(
            f'Out of {len(motion_records)} motion records, ' +
            f'after filtering by motion_record_type="{motion_record_type}", ' +
            f'returned {len(ret_dict)} motion records.')
        return ret_dict

    async def select_one(self, motion_records: dict[int, MotionRecord],
                         motion_record_type: MotionRecordType,
                   **kwargs) -> MotionRecord | None:
        """Randomly select one motion_record from motion_records.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be selected from, key is motion_record id,
                value is motion_record instance.
            motion_record_type (MotionRecordType):
                Motion record type to filter by.

        Returns:
            MotionRecord | None:
                Selected motion_record. Returns None if no matching
                motion_record is found.

        Raises:
            TypeError:
                Raised when motion_record_type is not of type MotionRecordType.
        """
        if len(motion_records) == 0:
            msg = 'Input motion_records is empty.'
            self.logger.warning(msg)
            return None
        if not isinstance(motion_record_type, MotionRecordType):
            msg = ('motion_record_type must be of type MotionRecordType, ' +
                   f'current type is {type(motion_record_type)}.')
            self.logger.error(msg)
            raise TypeError(msg)
        try:
            available_ids = await self.mapping.get_items(
                motion_record_type.value)
        except KeyError:
            available_ids = set()
        input_ids = set(motion_records.keys())
        filtered_ids = available_ids.intersection(input_ids)
        # randomly select one motion record
        if len(filtered_ids) == 0:
            msg = (f'Out of {len(motion_records)} motion records, ' +
                   f'after filtering by motion_record_type="{motion_record_type}", ' +
                   'no motion record returned.')
            self.logger.warning(msg)
            return None
        selected_id = np.random.choice(list(filtered_ids))
        self.logger.debug(
            f'Out of {len(motion_records)} motion records, ' +
            f'after filtering by motion_record_type="{motion_record_type}", ' +
            f'randomly selected one motion record, id={selected_id}.')
        return motion_records[selected_id]
