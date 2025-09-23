
import numpy as np

from ..data_structures.motion_record import MotionRecord
from .base_filter import BaseFilter


class DurationFilter(BaseFilter):
    """Filter for filtering motion_records based on total motion duration.
    """

    def __init__(self,
                 logger_cfg: None | dict = None) -> None:
        """Initialize duration filter.

        Args:
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name. Defaults to None.
        """
        super().__init__(logger_cfg=logger_cfg)

    async def filter(
            self,
            motion_records: dict[int, MotionRecord],
            duration_lowerbound: float | None = None,
            duration_upperbound: float | None = None,
            **kwargs) -> dict[int, MotionRecord]:
        """Filter motion_records based on total duration.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be filtered, key is motion_record id,
                value is motion_record instance.
            duration_lowerbound (float | None, optional):
                Lower bound of total motion duration. All motion_records in the result
                must have total duration greater than or equal to this value.
                Defaults to None.
            duration_upperbound (float | None, optional):
                Upper bound of total motion duration. All motion_records in the result
                must have total duration less than or equal to this value.
                Defaults to None.

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
        duration_lowerbound_value = 0 \
            if duration_lowerbound is None \
            else duration_lowerbound
        duration_upperbound_value = float('inf') \
            if duration_upperbound is None \
            else duration_upperbound
        ret_dict = dict()
        for k, v in motion_records.items():
            cur_duration = v.n_frames / v.fps
            if duration_lowerbound_value <= cur_duration and\
                    cur_duration <= duration_upperbound_value:
                ret_dict[k] = v
        self.logger.debug(
            f'Out of {len(motion_records)} motion records, ' +
            f'after filtering by lowerbound={duration_lowerbound}, ' +
            f'upperbound={duration_upperbound}, ' +
            f'returned {len(ret_dict)} motion records.')
        return ret_dict

    async def select_one(self,
            motion_records: dict[int, MotionRecord],
            duration_lowerbound: float | None = None,
            duration_upperbound: float | None = None,
                   **kwargs) -> MotionRecord | None:
        """Select one motion_record from motion_records.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be selected from, key is motion_record id,
                value is motion_record instance.
            duration_lowerbound (float | None, optional):
                Lower bound of total motion duration. All motion_records in the result
                must have total duration greater than or equal to this value.
                Defaults to None.
            duration_upperbound (float | None, optional):
                Upper bound of total motion duration. All motion_records in the result
                must have total duration less than or equal to this value.
                Defaults to None.

        Returns:
            MotionRecord | None:
                Selected motion_record. Returns None if no matching
                motion_record is found.
        """
        if len(motion_records) == 0:
            msg = 'Input motion_records is empty.'
            self.logger.warning(msg)
            return None
        duration_lowerbound_value = 0 \
            if duration_lowerbound is None \
            else duration_lowerbound
        duration_upperbound_value = float('inf') \
            if duration_upperbound is None \
            else duration_upperbound
        selected_id = None
        shuffle_keys = list(motion_records.keys())
        np.random.shuffle(shuffle_keys)
        for k in shuffle_keys:
            motion_record = motion_records[k]
            cur_duration = motion_record.n_frames / motion_record.fps
            if duration_lowerbound_value <= cur_duration and\
                    cur_duration <= duration_upperbound_value:
                selected_id = k
                break
        if selected_id is None:
            self.logger.warning(
                f'Out of {len(motion_records)} motion records, ' +
                f'after filtering by lowerbound={duration_lowerbound}, ' +
                f'upperbound={duration_upperbound}, ' +
                'no motion record returned.')
            return None
        self.logger.debug(
            f'Out of {len(motion_records)} motion records, ' +
            f'after filtering by lowerbound={duration_lowerbound}, ' +
            f'upperbound={duration_upperbound}, ' +
            f'randomly selected one motion record, id={selected_id}.')
        return motion_records[selected_id]
