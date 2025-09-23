import numpy as np

from speech2motion.utils.log import setup_logger

from ..data_structures.motion_record import MotionRecord
from ..variety.base_memory import BaseMemory
from .base_filter import BaseFilter


class MemoryFilter(BaseFilter):
    """Filter for filtering motion_records based on memory.
    It is used to filter out motion_records that users have already seen.
    """

    def __init__(self,
                 name: str,
                 memory: BaseMemory,
                 logger_cfg: None | dict = None) -> None:
        """Initialize memory filter.

        Args:
            name (str):
                Filter name, used as logger name.
            memory (BaseMemory):
                Memory instance for determining if actions are in memory.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name. Defaults to None.
        """
        BaseFilter.__init__(self, logger_cfg=logger_cfg)
        self.name = name
        self.logger_cfg['logger_name'] = self.name
        self.logger = setup_logger(**self.logger_cfg)
        self.memory = memory

    async def reset_memory(self, memory: BaseMemory) -> None:
        """Reset memory instance.

        Args:
            memory (BaseMemory):
                New memory instance.
        """
        self.memory = memory

    async def filter(
            self,
            motion_records: dict[int, MotionRecord],
            user_id: str,
            **kwargs) -> dict[int, MotionRecord]:
        """Filter motion_records based on memory.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be filtered, key is motion_record id,
                value is motion_record instance.
            user_id (str):
                User ID.
            memory_duration_override (float | None, optional):
                Memory duration in seconds.
                Defaults to None, uses `memory_duration` parameter from memory
                instance construction.

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
        filtered_ids = await self.memory.recall_filter(
            user_id=user_id,
            event_ids=list(motion_records.keys()))
        ret_dict = dict()
        for k in filtered_ids:
            ret_dict[k] = motion_records[k]
        self.logger.debug(
            f'Out of {len(motion_records)} motion records, ' +
            f'after filtering by user_id="{user_id}", ' +
            f'returned {len(ret_dict)} motion records.')
        return ret_dict

    async def select_one(self, motion_records: dict[int, MotionRecord],
                         user_id: str,
                         **kwargs) -> MotionRecord | None:
        """Randomly select one motion record that meets the criteria.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be selected from, key is motion_record id,
                value is motion_record instance.
            user_id (str):
                User ID.

        Returns:
            MotionRecord | None:
                Selected motion_record. Returns None if no matching
                motion_record is found.
        """
        if len(motion_records) == 0:
            msg = 'Input motion_records is empty.'
            self.logger.warning(msg)
            return None
        selected_id = None
        shuffle_keys = list(motion_records.keys())
        np.random.shuffle(shuffle_keys)
        for k in shuffle_keys:
            seen = await self.memory.recall(
                user_id=user_id,
                event_id=k)
            if not seen:
                selected_id = k
                break
        if selected_id is None:
            msg = (f'Out of {len(motion_records)} motion records, ' +
                   f'after filtering by user_id="{user_id}", ' +
                   'no motion record returned.')
            self.logger.warning(msg)
            return None
        self.logger.debug(
            f'Out of {len(motion_records)} motion records, ' +
            f'after filtering by user_id="{user_id}", ' +
            f'randomly selected one motion record, id={selected_id}.')
        return motion_records[selected_id]
