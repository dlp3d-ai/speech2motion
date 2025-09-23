import math

from ..data_structures.motion_record import MotionRecord
from .base_filter import BaseFilter


class CutoffDurationFilter(BaseFilter):
    """Filter for filtering motion_records based on cutoff frame
    and interval early termination duration.
    """

    def __init__(self,
                 logger_cfg: None | dict = None) -> None:
        """Initialize cutoff duration filter.

        Args:
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name. Defaults to None.
        """
        super().__init__(logger_cfg=logger_cfg)

    async def filter(
            self,
            motion_records: dict[int, MotionRecord],
            cutoff_duration_lowerbound: float | None = None,
            cutoff_duration_upperbound: float | None = None,
            **kwargs) -> dict[int, MotionRecord]:
        """Filter motion_records based on cutoff duration.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be filtered, key is motion_record id,
                value is motion_record instance.
            cutoff_duration_lowerbound (float | None, optional):
                Lower bound of motion duration. All motion_records in the result
                must have at least one cutoff position where the truncated duration
                is greater than or equal to this value. Defaults to None.
            cutoff_duration_upperbound (float | None, optional):
                Upper bound of motion duration. All motion_records in the result
                must have at least one cutoff position where the truncated duration
                is less than or equal to this value. Defaults to None.

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
        cutoff_duration_lowerbound_value = 0 \
            if cutoff_duration_lowerbound is None \
            else cutoff_duration_lowerbound
        cutoff_duration_upperbound_value = float('inf') \
            if cutoff_duration_upperbound is None \
            else cutoff_duration_upperbound
        ret_dict = dict()
        for k, v in motion_records.items():
            # every frame in idle_long motion is considered as a cutoff position
            if v.is_idle_long:
                duration = v.n_frames / v.fps
                if cutoff_duration_lowerbound_value <= duration and\
                        duration <= cutoff_duration_upperbound_value:
                    ret_dict[k] = v
                continue
            # non-idle_long motion, need to check cutoff positions
            elif v.is_random():
                # check cutoff frames
                cutoff_frames = v.random.cutoff_frames
                if cutoff_frames is not None:
                    for frame in cutoff_frames:
                        duration = frame / v.fps
                        if cutoff_duration_lowerbound_value <= duration and\
                                duration <= cutoff_duration_upperbound_value:
                            ret_dict[k] = v
                            break
                # already found qualifying cutoff_frame, skip remaining cutoff ranges
                if k in ret_dict:
                    continue
                # check cutoff ranges
                cutoff_ranges = v.random.cutoff_ranges
                if cutoff_ranges is not None:
                    for start, end in cutoff_ranges:
                        start_time = start / v.fps
                        end_time = end / v.fps
                        # if [start_time, end_time) overlaps with constraint interval,
                        # add to ret_dict
                        if start_time <= cutoff_duration_upperbound_value and\
                                end_time >= cutoff_duration_lowerbound_value:
                            ret_dict[k] = v
                            break
            else:
                pass
        self.logger.debug(
            f'Out of {len(motion_records)} motion records, ' +
            f'after filtering by lowerbound={cutoff_duration_lowerbound}, ' +
            f'upperbound={cutoff_duration_upperbound}, ' +
            f'returned {len(ret_dict)} motion records.')
        return ret_dict

    async def select_one(self,
            motion_records: dict[int, MotionRecord],
            cutoff_duration_lowerbound: float | None = None,
            cutoff_duration_upperbound: float | None = None,
            **kwargs) -> MotionRecord | None:
        """Select one motion_record with the earliest cutoff position from all
        qualifying motion_records.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be selected from, key is motion_record id,
                value is motion_record instance.
            cutoff_duration_lowerbound (float | None, optional):
                Lower bound of motion duration. All motion_records in the result
                must have at least one cutoff position where the truncated duration
                is greater than or equal to this value. Defaults to None.
            cutoff_duration_upperbound (float | None, optional):
                Upper bound of motion duration. All motion_records in the result
                must have at least one cutoff position where the truncated duration
                is less than or equal to this value. Defaults to None.

        Returns:
            MotionRecord | None:
                Selected motion_record. Returns None if no matching
                motion_record is found.
        """
        if len(motion_records) == 0:
            msg = 'Input motion_records is empty.'
            self.logger.warning(msg)
            return None
        cutoff_duration_lowerbound_value = 0 \
            if cutoff_duration_lowerbound is None \
            else cutoff_duration_lowerbound
        cutoff_duration_upperbound_value = float('inf') \
            if cutoff_duration_upperbound is None \
            else cutoff_duration_upperbound
        selected_id = None
        selected_frame_idx = float('inf')
        for k, v in motion_records.items():
            frame_idx = await self.get_earliest_cutoff_frame(
                v,
                cutoff_duration_lowerbound_value,
                cutoff_duration_upperbound_value
            )
            if frame_idx is not None and frame_idx < selected_frame_idx:
                selected_id = k
                selected_frame_idx = frame_idx
        if selected_id is None:
            self.logger.warning(
                f'Out of {len(motion_records)} motion records, ' +
                f'after filtering by lowerbound={cutoff_duration_lowerbound}, ' +
                f'upperbound={cutoff_duration_upperbound}, ' +
                'no motion record returned.')
            return None
        self.logger.debug(
            f'Out of {len(motion_records)} motion records, ' +
            f'after filtering by lowerbound={cutoff_duration_lowerbound}, ' +
            f'upperbound={cutoff_duration_upperbound}, ' +
            f'selected earliest cutoff motion record, id={selected_id}, ' +
            f'cutoff frame={selected_frame_idx}.')
        return motion_records[selected_id]

    async def get_earliest_cutoff_frame(
            self,
            motion_record: MotionRecord,
            cutoff_duration_lowerbound: float,
            cutoff_duration_upperbound: float,
            ) -> int | None:
        """Get the earliest cutoff frame index in motion_record that meets the criteria.

        Args:
            motion_record (MotionRecord):
                Motion record to get cutoff frame from.
            cutoff_duration_lowerbound (float):
                Lower bound of motion duration.
            cutoff_duration_upperbound (float):
                Upper bound of motion duration.

        Returns:
            int | None:
                Earliest cutoff frame index. Returns None if no qualifying
                cutoff frame is found.
        """
        frame_idx = None
        # idle_long motion has all segments as cutoff positions
        if motion_record.is_idle_long:
            next_frame_idx = math.ceil(cutoff_duration_lowerbound * motion_record.fps)
            if next_frame_idx < motion_record.n_frames and \
                    cutoff_duration_upperbound * motion_record.fps >= next_frame_idx:
                frame_idx = next_frame_idx
        # random motion, need to check cutoff positions
        elif motion_record.is_random():
            # check cutoff frames
            cutoff_frames = motion_record.random.cutoff_frames
            if cutoff_frames is not None:
                for frame in cutoff_frames:
                    duration = frame / motion_record.fps
                    if cutoff_duration_lowerbound <= duration and\
                            duration <= cutoff_duration_upperbound:
                        frame_idx = frame
                        break
            # check if cutoff ranges have earlier cutoff positions
            cutoff_ranges = motion_record.random.cutoff_ranges
            if cutoff_ranges is not None:
                for start, end in cutoff_ranges:
                    start_time = start / motion_record.fps
                    end_time = end / motion_record.fps
                    # if [start_time, end_time) overlaps with constraint interval,
                    # add to ret_dict
                    if start_time <= cutoff_duration_upperbound and\
                            end_time >= cutoff_duration_lowerbound:
                        frame_idx = math.ceil(start * motion_record.fps)
                        break
        return frame_idx
