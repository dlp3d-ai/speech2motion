import numpy as np

from ..data_structures.motion_record import MotionRecord
from .base_filter import BaseFilter


class RandomFilter(BaseFilter):
    """Filter for randomly selecting motion_records.
    """

    def __init__(self,
                 logger_cfg: None | dict = None) -> None:
        """Initialize random filter.

        Args:
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name. Defaults to None.
        """
        BaseFilter.__init__(self, logger_cfg=logger_cfg)

    async def filter(self, motion_records: dict[int, MotionRecord],
                     return_length: int| None = None,
               **kwargs) -> dict[int, MotionRecord]:
        """Randomly select and return motion_records.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be filtered, key is motion_record id,
                value is motion_record instance.
            return_length (int | None, optional):
                Number of motion_records to return.
                Defaults to None, meaning return all qualifying motion_records.

        Returns:
            dict[int, MotionRecord]:
                Randomly selected motion_records, same format as input
                motion_records but possibly reduced in number. Returns empty
                dictionary if no motion_record is available.
        """
        if len(motion_records) == 0:
            msg = 'Input motion_records is empty.'
            self.logger.warning(msg)
            return dict()
        if return_length is None:
            ret_dict = motion_records
        elif return_length >= len(motion_records):
            msg = (f'Requested return_length {return_length} ' +
                   f'is greater than or equal to motion_records count '
                   f'{len(motion_records)}, ' +
                   'returning all motion_records.')
            self.logger.warning(msg)
            ret_dict = motion_records
        else:
            ret_dict = dict()
            shuffle_keys = list(motion_records.keys())
            np.random.shuffle(shuffle_keys)
            for k in shuffle_keys[:return_length]:
                ret_dict[k] = motion_records[k]
            self.logger.debug(
                f'Out of {len(motion_records)} motion records, ' +
                f'randomly selected {len(ret_dict)} motion records.')
        return ret_dict

    async def select_one(self, motion_records: dict[int, MotionRecord],
                   **kwargs) -> MotionRecord | None:
        """Randomly select one motion record.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be selected from, key is motion_record id,
                value is motion_record instance.

        Returns:
            MotionRecord | None:
                Randomly selected motion_record. Returns None if no
                motion_record is available.
        """
        if len(motion_records) == 0:
            return None
        input_ids = list(motion_records.keys())
        # randomly select one motion record
        selected_id = np.random.choice(list(input_ids))
        self.logger.debug(
            f'Out of {len(motion_records)} motion records, ' +
            f'randomly selected one motion record, id={selected_id}.')
        return motion_records[selected_id]
