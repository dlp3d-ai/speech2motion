from abc import ABC, abstractmethod

from ..data_structures.motion_record import MotionRecord
from ..utils.super import Super


class BaseFilter(Super, ABC):
    """Base class for all filters.
    """

    def __init__(self,
                 logger_cfg: None | dict = None) -> None:
        """Initialize base filter.

        Args:
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name. Defaults to None.
        """
        super().__init__(logger_cfg=logger_cfg)

    @abstractmethod
    async def filter(self, motion_records: dict[int, MotionRecord],
               **kwargs) -> dict[int, MotionRecord]:
        """Filter motion_records.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be filtered, key is motion_record id,
                value is motion_record instance.

        Returns:
            dict[int, MotionRecord]:
                Remaining motion_records after filtering, same format as input
                motion_records but possibly reduced in number. Returns empty
                dictionary if no matching motion_record is found.
        """
        pass

    @abstractmethod
    async def select_one(self, motion_records: dict[int, MotionRecord],
                   **kwargs) -> MotionRecord | None:
        """Select one motion_record from motion_records.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be selected from, key is motion_record id,
                value is motion_record instance.

        Returns:
            MotionRecord | None:
                Selected motion_record. Returns None if no matching
                motion_record is found.
        """
        pass
