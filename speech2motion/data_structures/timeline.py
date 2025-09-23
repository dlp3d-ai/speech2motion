import json
from asyncio import Task

import prettytable

from ..utils.super import Super
from .interval_tree import IntervalTreeNode
from .motion_record import MotionRecord


class Timeline(Super):
    """Frame-based timeline for motion scheduling and management.

    Manages the arrangement and scheduling of motion records on a timeline,
    supporting motion insertion, querying, and export functionality.
    """

    def __init__(self,
                 start_frame: int,
                 end_frame: int,
                 enable_extension: bool = False,
                 logger_cfg: None | dict = None) -> None:
        """Initialize timeline object.

        Args:
            start_frame (int):
                Start frame of timeline, timeline range is [start_frame, end_frame).
            end_frame (int):
                End frame of timeline, timeline range is [start_frame, end_frame).
            enable_extension (bool, optional):
                Whether to allow timeline extension.
                If True, timeline can be extended before frame 0 and after end_frame.
                Defaults to False.
            logger_cfg (None | dict, optional):
                Logger configuration.
                Defaults to None.
        """
        Super.__init__(self, logger_cfg)

        self.enable_extension = enable_extension
        self.interval_tree = IntervalTreeNode(
            start=start_frame, end=end_frame,
            logger_cfg=self.logger_cfg)

    @property
    def start(self) -> int:
        """Get start frame of timeline.

        Returns:
            int: Start frame index of timeline.
        """
        return self.interval_tree.start

    @property
    def end(self) -> int:
        """Get end frame of timeline.

        Returns:
            int: End frame index of timeline.
        """
        return self.interval_tree.end

    @property
    def n_frames(self) -> int:
        """Get number of frames in timeline.

        Returns:
            int: Total number of frames in timeline.
        """
        return self.interval_tree.end - self.interval_tree.start

    def get_next_blank(self,
            intervals_to_skip: list[tuple[float, float]] | None = None
            ) -> tuple[int, int] | None:
        """Get next blank interval in timeline.

        Args:
            intervals_to_skip (list[tuple[float, float]], optional):
                List of intervals to skip. Returned interval should not overlap
                with intervals in this list. Defaults to None, no restriction
                on return value.

        Returns:
            tuple[int, int] | None:
                Next blank interval.
                Returns None if no blank interval is found.
        """
        blank_node = self.interval_tree.get_next_blank_recursively(intervals_to_skip)
        if blank_node is None:
            return None
        else:
            return (blank_node.start, blank_node.end)

    def insert(
        self,
        start_idx: int,
        end_idx: int,
        motion_record: MotionRecord,
        in_point: int,
        out_point: int,
        trigger: str | None = None,
        preload_task: Task | None = None
    ) -> None:
        """Insert a motion record at [start_idx, end_idx) position in timeline.

        Args:
            start_idx (int):
                Start frame of motion record on timeline.
            end_idx (int):
                End frame of motion record on timeline.
            motion_record (MotionRecord):
                Motion record to be inserted.
            in_point (int):
                Start frame of motion record.
            out_point (int):
                End frame of motion record.
            trigger (str | None, optional):
                Trigger word for motion record.
                Defaults to None, indicating no trigger word.
            preload_task (Task | None, optional):
                Preload task.
                Defaults to None, indicating no preload task.
        """
        payload_dict = dict(
            motion_record=motion_record,
            in_point=in_point,
            out_point=out_point,
            trigger=trigger,
            preload_task=preload_task
        )
        new_node = IntervalTreeNode(
            start=start_idx,
            end=end_idx,
            payload=payload_dict,
            logger_cfg=self.logger_cfg)
        if start_idx < self.interval_tree.start or \
                end_idx > self.interval_tree.end:
            if not self.enable_extension:
                msg = f'Interval to insert [{start_idx}, {end_idx}) ' +\
                    f'is out of timeline range [{self.start}, {self.end}), ' +\
                    'please set enable_extension=True to allow extension.'
                self.logger.error(msg)
                raise ValueError(msg)
            else:
                if start_idx < self.interval_tree.start:
                    self.logger.debug(
                        f'Start frame extended from {self.interval_tree.start} ' +
                        f'to {new_node.start}')
                    if self.interval_tree.is_leaf():
                        new_root = IntervalTreeNode(
                            start=start_idx,
                            end=self.interval_tree.end,
                            logger_cfg=self.logger_cfg)
                        previous_nodes = self.interval_tree.to_list()
                        for node_dict in previous_nodes:
                            new_root.insert_node_recursively(
                                IntervalTreeNode(
                                    start=node_dict['start'],
                                    end=node_dict['end'],
                                    payload=node_dict['payload'],
                                    logger_cfg=self.logger_cfg))
                        self.interval_tree = new_root
                    else:
                        self.interval_tree.extend_start(new_node.start)
                if end_idx > self.interval_tree.end:
                    self.logger.debug(
                        f'End frame extended from {self.interval_tree.end} ' +
                        f'to {new_node.end}')
                    if self.interval_tree.is_leaf():
                        new_root = IntervalTreeNode(
                            start=self.interval_tree.start,
                            end=end_idx,
                            logger_cfg=self.logger_cfg)
                        previous_nodes = self.interval_tree.to_list()
                        for node_dict in previous_nodes:
                            new_root.insert_node_recursively(
                                IntervalTreeNode(
                                    start=node_dict['start'],
                                    end=node_dict['end'],
                                    payload=node_dict['payload'],
                                    logger_cfg=self.logger_cfg))
                        self.interval_tree = new_root
                    else:
                        self.interval_tree.extend_end(new_node.end)
        self.interval_tree.insert_node_recursively(new_node)

    def to_list(
            self,
            slice_start_idx: int | None = None,
            slice_end_idx: int | None = None) -> list[dict]:
        """Convert timeline to list of dictionaries.

        Args:
            slice_start_idx (int | None, optional):
                Slice start frame. If None, use timeline start frame.
                Defaults to None.
            slice_end_idx (int | None, optional):
                Slice end frame. If None, use timeline end frame.
                Defaults to None.

        Returns:
            list[dict]:
                List of dictionaries representing timeline.
        """
        node_list = self.interval_tree.to_list()
        if slice_start_idx is not None:
            if slice_start_idx < self.start or slice_start_idx > self.end:
                msg = f'Slice start frame {slice_start_idx} ' +\
                    f'is out of timeline range [{self.start}, {self.end})'
                self.logger.error(msg)
                raise ValueError(msg)
        else:
            slice_start_idx = self.start
        if slice_end_idx is not None:
            if slice_end_idx < self.start or slice_end_idx > self.end:
                msg = f'Slice end frame {slice_end_idx} ' +\
                    f'is out of timeline range [{self.start}, {self.end})'
                self.logger.error(msg)
                raise ValueError(msg)
        else:
            slice_end_idx = self.end
        if slice_start_idx >= slice_end_idx:
            msg = (f'Slice start frame {slice_start_idx} is greater than or equal to '
                   f'slice end frame {slice_end_idx}')
            self.logger.error(msg)
            raise ValueError(msg)
        if slice_start_idx == self.start and slice_end_idx == self.end:
            return node_list
        else:
            ret_list = list()
            for node_dict in node_list:
                node_start = node_dict['start']
                node_end = node_dict['end']
                sliced_node_dict = dict()
                if node_end <= slice_start_idx:
                    continue
                elif node_start >= slice_end_idx:
                    break
                else:
                    if slice_start_idx > node_start:
                        start_shift = slice_start_idx - node_start
                        sliced_node_dict['start'] = slice_start_idx
                        sliced_node_dict['end'] = node_dict['end']
                        sliced_node_dict['payload'] = node_dict['payload'].copy()
                        sliced_node_dict['payload']['in_point'] += start_shift
                    elif slice_end_idx < node_end:
                        sliced_node_dict['start'] = node_dict['start']
                        end_shift = node_end - slice_end_idx
                        sliced_node_dict['end'] = slice_end_idx
                        sliced_node_dict['payload'] = node_dict['payload'].copy()
                        sliced_node_dict['payload']['out_point'] -= end_shift
                    else:
                        sliced_node_dict['start'] = node_dict['start']
                        sliced_node_dict['end'] = node_dict['end']
                        sliced_node_dict['payload'] = node_dict['payload'].copy()
                    ret_list.append(sliced_node_dict)
            return ret_list

    def to_table(self,
                 fps: float = 30,
                 payload_keys_to_ignore: list[str] | None = None,
                 slice_start_idx: int | None = None,
                 slice_end_idx: int | None = None) -> str:
        """Convert timeline to prettytable string.

        Args:
            fps (float, optional):
                Frame rate of timeline.
                Defaults to 30.
            payload_keys_to_ignore (list[str] | None, optional):
                Keys to ignore. If a key is in the list, it will not be
                displayed as a column in the table.
                Defaults to None.
            slice_start_idx (int | None, optional):
                Slice start frame. If None, use timeline start frame.
                Defaults to None.
            slice_end_idx (int | None, optional):
                Slice end frame. If None, use timeline end frame.
                Defaults to None.

        Returns:
            str:
                Prettytable string representing timeline.
        """
        rows = self.to_rows(fps, payload_keys_to_ignore=payload_keys_to_ignore,
                            slice_start_idx=slice_start_idx,
                            slice_end_idx=slice_end_idx)
        if len(rows) == 0:
            return ''
        table = prettytable.PrettyTable()
        field_names = list(rows[0].keys())
        table.field_names = field_names
        for row_dict in rows:
            pretty_row = list()
            for key in field_names:
                data = row_dict[key]
                if data is None:
                    pretty_row.append('')
                elif isinstance(data, float):
                    pretty_row.append(f'{data:.2f}')
                else:
                    pretty_row.append(str(data))
            table.add_row(pretty_row)
        table_str = table.get_string()
        return table_str

    def to_json(self,
                fps: float = 30,
                payload_keys_to_ignore: list[str] | None = None,
                slice_start_idx: int | None = None,
                slice_end_idx: int | None = None) -> str:
        """Convert timeline to JSON string.

        Args:
            fps (float, optional):
                Frame rate of timeline.
                Defaults to 30.
            payload_keys_to_ignore (list[str] | None, optional):
                Keys to ignore. If a key is in the list, it will not be
                displayed as a field in JSON.
                Defaults to None.
            slice_start_idx (int | None, optional):
                Slice start frame. If None, use timeline start frame.
                Defaults to None.
            slice_end_idx (int | None, optional):
                Slice end frame. If None, use timeline end frame.
                Defaults to None.

        Returns:
            str:
                JSON string representing timeline.
        """
        rows = self.to_rows(fps, payload_keys_to_ignore=payload_keys_to_ignore,
                            slice_start_idx=slice_start_idx,
                            slice_end_idx=slice_end_idx)
        json_str = json.dumps(rows, indent=4, ensure_ascii=False)
        return json_str

    def to_rows(
        self,
        fps: float = 30,
        payload_keys_to_ignore: list[str] | None = None,
        slice_start_idx: int | None = None,
        slice_end_idx: int | None = None
    ) -> list[dict[str, float | int | str | None]]:
        """Convert timeline to list of row dictionaries.

        Args:
            fps (float, optional):
                Frame rate of timeline.
                Defaults to 30.
            payload_keys_to_ignore (list[str] | None, optional):
                Keys to ignore. If a key is in the list, it will not be
                displayed as a column in the table.
                Defaults to None.
            slice_start_idx (int | None, optional):
                Slice start frame. If None, use timeline start frame.
                Defaults to None.
            slice_end_idx (int | None, optional):
                Slice end frame. If None, use timeline end frame.
                Defaults to None.

        Returns:
            list[dict[str, float | int | str | None]]:
                List of dictionaries corresponding to table row data.
        """
        frame_timeline_list = self.to_list(
            slice_start_idx=slice_start_idx,
            slice_end_idx=slice_end_idx)
        base_field_names = [
            'start',
            'end',
            'duration',
            'motion id',
        ]
        payload_field_names = list()
        default_keys_to_ignore = ['motion_record', 'preload_task', 'motion_clip']
        if payload_keys_to_ignore is not None:
            payload_keys_to_ignore = payload_keys_to_ignore + default_keys_to_ignore
        else:
            payload_keys_to_ignore = default_keys_to_ignore
        for timeline_dict in frame_timeline_list:
            payload_keys = timeline_dict['payload'].keys()
            for key in payload_keys:
                if key not in payload_keys_to_ignore and\
                        key not in base_field_names and\
                        key not in payload_field_names:
                    payload_field_names.append(key)
        final_field_names = base_field_names + payload_field_names
        rows = list()
        for timeline_dict in frame_timeline_list:
            row_dict = dict()
            motion_record = timeline_dict['payload']['motion_record']
            start = float(timeline_dict['start']) \
                if fps is None \
                else timeline_dict['start'] / fps
            end = float(timeline_dict['end']) \
                if fps is None \
                else timeline_dict['end'] / fps
            base_row = [
                start,
                end,
                end - start,
                motion_record.motion_record_id,
            ]
            payload_row = list()
            payload_row.extend(
                timeline_dict['payload'].get(key, None) for key in payload_field_names)
            row = base_row + payload_row
            for idx, data in enumerate(row):
                row_dict[final_field_names[idx]] = data
            rows.append(row_dict)
        return rows

    def shallow_copy(self) -> 'Timeline':
        """Create shallow copy of timeline.

        Returns:
            Timeline: Shallow copy of timeline.
        """
        ret_timeline = Timeline(
            start_frame=self.start,
            end_frame=self.end,
            enable_extension=self.enable_extension,
            logger_cfg=self.logger_cfg)
        ret_timeline.interval_tree = self.interval_tree.shallow_copy()
        return ret_timeline

    def get_first_item(self) -> dict | None:
        """Get first motion record in timeline.

        Returns:
            dict | None:
                First motion record.
                Returns None if no motion record is found.
        """
        return self.interval_tree.get_first_leaf()

    def get_last_item(self) -> dict | None:
        """Get last motion record in timeline.

        Returns:
            dict | None:
                Last motion record.
                Returns None if no motion record is found.
        """
        return self.interval_tree.get_last_leaf()
