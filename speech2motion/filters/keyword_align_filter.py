
from typing import Literal

import numpy as np

from ..data_structures.motion_record import MotionRecord
from .base_filter import BaseFilter


class KeywordAlignFilter(BaseFilter):
    """Filter for filtering motion_records based on time range
    after keyword motion keyframe alignment.
    """

    def __init__(self,
                 logger_cfg: None | dict = None) -> None:
        """Initialize keyword align filter.

        Args:
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name. Defaults to None.
        """
        super().__init__(logger_cfg=logger_cfg)

    async def filter(
            self,
            motion_records: dict[int, MotionRecord],
            align_frame: int,
            start_frame_lowerbound: int,
            end_frame_upperbound: int,
            keyword_attr_name: Literal['motion_keyword', 'speech_keyword'],
            **kwargs) -> dict[int, MotionRecord]:
        """Filter motion_records based on keyword alignment.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be filtered, key is motion_record id,
                value is motion_record instance.
            align_frame (int):
                Frame index to align with keyword motion keyframe.
            start_frame_lowerbound (int):
                Lower bound of start frame.
            end_frame_upperbound (int):
                Upper bound of end frame.
            keyword_attr_name (Literal['motion_keyword', 'speech_keyword']):
                Keyword attribute name to use for filtering.

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
        ret_dict = dict()
        for k, v in motion_records.items():
            if not hasattr(v, keyword_attr_name):
                continue
            keyword_attr = getattr(v, keyword_attr_name)
            if keyword_attr is None:
                continue
            frame_attr_name = f'{keyword_attr_name}_frame'
            keyword_frame_value = getattr(keyword_attr, frame_attr_name)
            n_frames = v.n_frames
            aligned_start_frame = align_frame - keyword_frame_value
            if aligned_start_frame < start_frame_lowerbound:
                continue
            aligned_end_frame = aligned_start_frame + n_frames
            if aligned_end_frame > end_frame_upperbound:
                continue
            ret_dict[k] = v
        self.logger.debug(
            f'Out of {len(motion_records)} motion records, ' +
            f'after filtering by {keyword_attr_name} keyframe alignment with ' +
            f'align_frame={align_frame}, ' +
            f'start_frame_lowerbound={start_frame_lowerbound}, ' +
            f'end_frame_upperbound={end_frame_upperbound}, ' +
            f'returned {len(ret_dict)} motion records.')
        return ret_dict

    async def select_one(self,
            motion_records: dict[int, MotionRecord],
            align_frame: int,
            start_frame_lowerbound: int,
            end_frame_upperbound: int,
            keyword_attr_name: Literal['motion_keyword', 'speech_keyword'],
            **kwargs) -> MotionRecord | None:
        """Select one motion_record from motion_records.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be selected from, key is motion_record id,
                value is motion_record instance.
            align_frame (int):
                Frame index to align with keyword motion keyframe.
            start_frame_lowerbound (int):
                Lower bound of start frame.
            end_frame_upperbound (int):
                Upper bound of end frame.
            keyword_attr_name (Literal['motion_keyword', 'speech_keyword']):
                Keyword attribute name to use for filtering.

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
            motion_record = motion_records[k]
            if not hasattr(motion_record, keyword_attr_name):
                continue
            keyword_attr = getattr(motion_record, keyword_attr_name)
            if keyword_attr is None:
                continue
            frame_attr_name = f'{keyword_attr_name}_frame'
            keyword_frame_value = getattr(keyword_attr, frame_attr_name)
            n_frames = motion_record.n_frames
            aligned_start_frame = align_frame - keyword_frame_value
            if aligned_start_frame < start_frame_lowerbound:
                continue
            aligned_end_frame = aligned_start_frame + n_frames
            if aligned_end_frame > end_frame_upperbound:
                continue
            selected_id = k
            break
        if selected_id is None:
            self.logger.warning(
                f'Out of {len(motion_records)} motion records, ' +
                f'after filtering by {keyword_attr_name} keyframe alignment with ' +
                f'align_frame={align_frame}, ' +
                f'start_frame_lowerbound={start_frame_lowerbound}, ' +
                f'end_frame_upperbound={end_frame_upperbound}, ' +
                'no motion record returned.')
            return None
        self.logger.debug(
            f'Out of {len(motion_records)} motion records, ' +
            f'after filtering by {keyword_attr_name} keyframe alignment with ' +
            f'align_frame={align_frame}, ' +
            f'start_frame_lowerbound={start_frame_lowerbound}, ' +
            f'end_frame_upperbound={end_frame_upperbound}, ' +
            f'randomly selected one motion record, id={selected_id}.')
        return motion_records[selected_id]
