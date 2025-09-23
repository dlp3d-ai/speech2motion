import numpy as np

from ..data_structures.motion_record import MotionRecord
from ..index.base_index_mapping import BaseIndexMapping
from .base_filter import BaseFilter


class AvatarFilter(BaseFilter):
    """Filter for filtering motion_records based on avatar.
    """

    def __init__(self,
                 mapping: BaseIndexMapping,
                 logger_cfg: None | dict = None) -> None:
        """Initialize avatar filter.

        Args:
            mapping (BaseIndexMapping):
                Index mapping instance that maps avatar names to a set of IDs.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name. Defaults to None.
        """
        BaseFilter.__init__(self, logger_cfg=logger_cfg)
        self.mapping = mapping

    async def reset_mapping(self, mapping: BaseIndexMapping) -> None:
        """Reset index mapping instance.

        Args:
            mapping (BaseIndexMapping):
                New index mapping instance.
        """
        self.mapping = mapping

    async def filter(self, motion_records: dict[int, MotionRecord],
                     avatar: str,
               **kwargs) -> dict[int, MotionRecord]:
        """Filter motion_records based on avatar name.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be filtered, key is motion_record id,
                value is motion_record instance.
            avatar (str):
                Avatar name.

        Returns:
            dict[int, MotionRecord]:
                Remaining motion_records after filtering, same format as input
                motion_records but possibly reduced in number. Returns empty
                dictionary if no matching motion_record is found.
        """
        if len(motion_records) == 0:
            msg = 'Input motion_records is empty.'
            self.logger.warning(msg)
            return dict()
        try:
            available_ids = await self.mapping.get_items(avatar)
        except KeyError:
            available_ids = set()
        input_ids = set(motion_records.keys())
        filtered_ids = available_ids.intersection(input_ids)
        ret_dict = dict()
        for k in filtered_ids:
            ret_dict[k] = motion_records[k]
        self.logger.debug(
            f'Out of {len(motion_records)} motion records, ' +
            f'after filtering by avatar="{avatar}", ' +
            f'returned {len(ret_dict)} motion records.')
        return ret_dict

    async def select_one(self, motion_records: dict[int, MotionRecord], avatar: str,
                   **kwargs) -> MotionRecord | None:
        """Randomly select one matching motion record.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be selected from, key is motion_record id,
                value is motion_record instance.
            avatar (str):
                Avatar name.

        Returns:
            MotionRecord | None:
                Selected motion_record. Returns None if no matching
                motion_record is found.
        """
        if len(motion_records) == 0:
            msg = 'Input motion_records is empty.'
            self.logger.warning(msg)
            return None
        try:
            available_ids = await self.mapping.get_items(avatar)
        except KeyError:
            available_ids = set()
        input_ids = set(motion_records.keys())
        filtered_ids = available_ids.intersection(input_ids)
        # randomly select one motion record
        if len(filtered_ids) == 0:
            msg = f'Out of {len(motion_records)} motion records, ' + \
                f'after filtering by avatar="{avatar}", ' + \
                'no motion record returned.'
            self.logger.warning(msg)
            return None
        selected_id = np.random.choice(list(filtered_ids))
        self.logger.debug(
            f'Out of {len(motion_records)} motion records, ' +
            f'after filtering by avatar="{avatar}", ' +
            f'randomly selected one motion record, id={selected_id}.')
        return motion_records[selected_id]
