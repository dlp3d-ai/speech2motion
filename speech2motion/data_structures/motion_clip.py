import io
from typing import Any, Literal

import numpy as np

from ..utils.io import load_npz
from ..utils.log import setup_logger
from ..utils.super import Super


class MotionClip(Super):
    """Class for storing and passing matrix basis rotation data for each joint
    in animation and root bone translation data in world coordinate system.
    """

    def __init__(self,
                 n_frames: int,
                 joint_names: list[str],
                 joint_rotmat: np.ndarray,
                 root_world_position: np.ndarray,
                 restpose_name: str,
                 app_name: Literal['python_backend', 'babylon'] | None = None,
                 cutoff_frames: list[tuple[int, int, int]] | None = None,
                 cutoff_ranges: list[tuple[int, int, int, int]] | None = None,
                 blendshape_names: list[str] | None = None,
                 blendshape_values: np.ndarray | None = None,
                 motion_record_id: int | None = None,
                 timeline_start_idx: int | None = None,
                 logger_cfg: None | dict = None) -> None:
        """Initialize motion clip.

        Args:
            n_frames (int):
                Total number of frames in the motion clip.
            joint_names (list[str]):
                List of joint names.
            joint_rotmat (np.ndarray):
                Joint rotation matrices with shape (n_frames, n_joints, 3, 3).
            root_world_position (np.ndarray):
                Root bone position in world coordinate system with shape (n_frames, 3).
            restpose_name (str):
                Restpose name of the motion clip.
            app_name (Literal['python_backend', 'babylon'] | None, optional):
                App name using the motion clip. Defaults to None, indicating
                motion data format is consistent with database.
            cutoff_frames (list[tuple[int, int, int]], optional):
                Cutoff frame positions of the motion clip. Each tuple contains
                (frame_idx, left_priority, right_priority). Defaults to None.
            cutoff_ranges (list[tuple[int, int, int, int]], optional):
                Cutoff ranges of the motion clip. Each tuple contains
                (start_frame, end_frame, left_priority, right_priority).
                Defaults to None.
            blendshape_names (list[str], optional):
                List of high-priority blendshape animation names
                bound to the motion clip.
                Defaults to None.
            blendshape_values (np.ndarray, optional):
                High-priority blendshape animation values bound to the motion clip
                with shape (n_frames, n_blendshapes). Defaults to None.
            motion_record_id (int | None, optional):
                Motion record ID in database. Defaults to None.
            timeline_start_idx (int | None, optional):
                Start index of motion clip in timeline. Only used when alignment
                with other channels on timeline is needed. Most of the time pass None.
                Defaults to None.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Defaults to None.
        """
        Super.__init__(self, logger_cfg)
        logger_name = f'{self.__class__.__name__}_{motion_record_id}'
        self.logger_cfg['logger_name'] = logger_name
        self.logger = setup_logger(**self.logger_cfg)
        self.n_frames = n_frames
        # cannot be None after set_joint_rotmat
        self.joint_names: np.ndarray | None = None
        self.joint_rotmat: np.ndarray | None = None
        # cannot be None after set_root_world_position
        self.root_world_position: np.ndarray | None = None
        # may be None
        self.blendshape_names: list[str] | None = None
        self.blendshape_values: np.ndarray | None = None
        self.set_joint_rotmat(
            joint_rotmat=joint_rotmat,
            joint_names=joint_names
        )
        self.set_root_world_position(
            root_world_position=root_world_position
        )
        self.set_blendshape(
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values
        )
        if cutoff_frames is None:
            cutoff_frames = [(0, 0, 0), (n_frames - 1, 0, 0)]
        self.set_cutoff_frames(cutoff_frames)
        if cutoff_ranges is None:
            self.cutoff_ranges: list[tuple[int, int, int, int]] | None = None
        else:
            self.set_cutoff_ranges(cutoff_ranges)
        self.restpose_name = restpose_name
        self.app_name = app_name
        self.motion_record_id = motion_record_id
        self.timeline_start_idx = timeline_start_idx

    def _check_n_joints(self) -> None:
        n_joints_np = self.joint_rotmat.shape[1]
        n_joints_list = len(self.joint_names)
        if n_joints_np != n_joints_list:
            msg = f'Length of joint_names {n_joints_list} != ' +\
                f'number of bones in joint_rotmat {n_joints_np}'
            self.logger.error(msg)
            raise ValueError(msg)

    def _check_n_frames(self) -> None:
        array_names = ('joint_rotmat', 'root_world_position', 'blendshape_values')
        for array_name in array_names:
            ndarray = getattr(self, array_name)
            if ndarray is None:
                continue
            n_frames_np = ndarray.shape[0]
            if n_frames_np != self.n_frames:
                msg = f'n_frames of {array_name} {n_frames_np} != ' +\
                    f'MotionClip n_frames {self.n_frames}'
                self.logger.error(msg)
                raise ValueError(msg)

    def _check_n_blendshapes(self) -> None:
        none_count = 0
        if self.blendshape_values is None:
            none_count += 1
        if self.blendshape_names is None:
            none_count += 1
        if none_count == 1:
            msg = 'Only one of blendshape_values and blendshape_names is None'
            self.logger.error(msg)
            raise ValueError(msg)
        if self.blendshape_values is not None:
            if self.blendshape_names is None:
                msg = 'blendshape_values is not None, but blendshape_names is None'
                self.logger.error(msg)
                raise ValueError(msg)
            n_blendshapes_np = self.blendshape_values.shape[1]
            n_blendshapes_list = len(self.blendshape_names)
            if n_blendshapes_np != n_blendshapes_list:
                msg = f'Length of blendshape_names {n_blendshapes_list} != ' +\
                    f'number of blendshapes in blendshape_values {n_blendshapes_np}'
                self.logger.error(msg)
                raise ValueError(msg)

    def __len__(self) -> int:
        return self.n_frames

    def set_timeline_start_idx(self, timeline_start_idx: int | None) -> None:
        self.timeline_start_idx = timeline_start_idx

    def set_joint_rotmat(self, joint_rotmat: np.ndarray,
                         joint_names: list[str]) -> None:
        """Set joint rotation matrices.

        Args:
            joint_rotmat (np.ndarray):
                Joint rotation matrices with shape (n_frames, n_joints, 3, 3).
            joint_names (list[str]):
                List of joint names.
        """
        self.joint_names = joint_names
        self.joint_rotmat = joint_rotmat
        self._check_n_joints()
        self._check_n_frames()

    def set_blendshape(
            self,
            blendshape_names: list[str] | None,
            blendshape_values: np.ndarray | None) -> None:
        """Set high-priority blendshape animation names and values.

        Args:
            blendshape_names (list[str] | None):
                List of high-priority blendshape animation names.
                Can be None, indicating no high-priority blendshape animation is bound.
            blendshape_values (np.ndarray | None):
                High-priority blendshape animation values
                with shape (n_frames, n_blendshapes).
                Can be None, indicating no high-priority blendshape
                animation is bound.
        """
        self.blendshape_names = blendshape_names
        self.blendshape_values = blendshape_values
        self._check_n_frames()
        self._check_n_blendshapes()

    def set_root_world_position(self, root_world_position: np.ndarray) -> None:
        """Set root bone position in world coordinate system.

        Args:
            root_world_position (np.ndarray):
                Root bone position in world coordinate system.
        """
        self.root_world_position = root_world_position
        self._check_n_frames()

    def set_cutoff_frames(self, cutoff_frames: list[tuple[int, int, int]]) -> None:
        """Set cutoff frame properties.

        Args:
            cutoff_frames (list[tuple[int, int, int]]):
                List of cutoff frames. Each tuple contains
                (frame_idx, left_priority, right_priority).
                left_priority indicates priority of left motion_clip
                when splicing rightward after cutoff.
                right_priority indicates priority of right motion_clip
                when splicing leftward after cutoff.
        """
        new_cutoff_frames = []
        for frame_idx, left_priority, right_priority in cutoff_frames:
            if frame_idx < 0 or frame_idx >= self.n_frames:
                msg = f'When setting priority, frame_idx {frame_idx} ' +\
                    f'is out of range [0, {self.n_frames})'
                self.logger.error(msg)
                raise ValueError(msg)
            new_cutoff_frames.append((frame_idx, left_priority, right_priority))
        self.cutoff_frames = new_cutoff_frames

    def set_cutoff_ranges(self, cutoff_ranges: list[tuple[int, int, int, int]]) -> None:
        """Set cutoff range properties.

        Args:
            cutoff_ranges (list[tuple[int, int, int, int]]):
                List of cutoff ranges. Each tuple contains
                (start_frame, end_frame, left_priority, right_priority).
                Any frame in [start_frame, end_frame) can be cut off.
                left_priority indicates priority of left motion_clip
                when splicing rightward after cutoff.
                right_priority indicates priority of right motion_clip
                when splicing leftward after cutoff.
        """
        new_cutoff_ranges = []
        for start_frame, end_frame, left_priority, right_priority in cutoff_ranges:
            if start_frame < 0 or end_frame > self.n_frames:
                msg = f'When setting priority, start_frame {start_frame} ' +\
                    f'or end_frame {end_frame} is out of range [0, {self.n_frames})'
                self.logger.error(msg)
                raise ValueError(msg)
            new_cutoff_ranges.append(
                (start_frame, end_frame, left_priority, right_priority))
        self.cutoff_ranges = new_cutoff_ranges

    def clone(self) -> 'MotionClip':
        """Clone current motion clip.

        Returns:
            MotionClip: Clone of current motion clip.
        """
        return MotionClip(
            n_frames=self.n_frames,
            joint_names=self.joint_names,
            joint_rotmat=self.joint_rotmat,
            root_world_position=self.root_world_position,
            cutoff_frames=self.cutoff_frames,
            cutoff_ranges=self.cutoff_ranges,
            restpose_name=self.restpose_name,
            app_name=self.app_name,
            blendshape_names=self.blendshape_names,
            blendshape_values=self.blendshape_values,
            motion_record_id=self.motion_record_id,
            logger_cfg=self.logger_cfg)

    def slice(self, start_frame: int, end_frame: int) -> 'MotionClip':
        """Slice motion clip.

        Args:
            start_frame (int): Start frame of slice.
            end_frame (int): End frame of slice.

        Returns:
            MotionClip:
                Slice of current motion clip with range [start_frame, end_frame).
        """
        new_cutoff_frames = []
        for frame_idx, left_priority, right_priority in self.cutoff_frames:
            if frame_idx >= start_frame and frame_idx < end_frame:
                # no cutoff property for slice start frame,
                # add 0 priority cutoff property record
                if len(new_cutoff_frames) == 0 and frame_idx != start_frame:
                    new_cutoff_frames.append(
                        (0, 0, 0))
                new_cutoff_frames.append(
                    (frame_idx - start_frame, left_priority, right_priority))
        # no cutoff property for slice start frame,
        # add 0 priority cutoff property record
        if len(new_cutoff_frames) == 0:
            new_cutoff_frames.append((0, 0, 0))
        # no cutoff property for slice end frame,
        # add 0 priority cutoff property record
        if new_cutoff_frames[-1][0] != end_frame - start_frame - 1:
            new_cutoff_frames.append((end_frame - start_frame - 1, 0, 0))
        if self.cutoff_ranges is not None:
            new_cutoff_ranges = []
            for cutoff_range in self.cutoff_ranges:
                range_start, range_end, left_priority, right_priority = cutoff_range
                if range_end > start_frame or range_start < end_frame - 1:
                    new_range_start = max(range_start, start_frame)
                    new_range_end = min(range_end, end_frame)
                    new_cutoff_ranges.append(
                        (
                            new_range_start - start_frame,
                            new_range_end - start_frame,
                            left_priority,
                            right_priority))
        else:
            new_cutoff_ranges = None
        if self.blendshape_values is not None:
            new_blendshape_values = self.blendshape_values[start_frame:end_frame].copy()
            new_blendshape_names = self.blendshape_names.copy()
        else:
            new_blendshape_names = None
            new_blendshape_values = None
        if self.timeline_start_idx is not None:
            timeline_start_idx = self.timeline_start_idx + start_frame
        else:
            timeline_start_idx = None
        return MotionClip(
            n_frames=end_frame - start_frame,
            joint_names=self.joint_names.copy(),
            joint_rotmat=self.joint_rotmat[start_frame:end_frame].copy(),
            root_world_position=self.root_world_position[start_frame:end_frame].copy(),
            cutoff_frames=new_cutoff_frames,
            cutoff_ranges=new_cutoff_ranges,
            blendshape_names=new_blendshape_names,
            blendshape_values=new_blendshape_values,
            restpose_name=self.restpose_name,
            app_name=self.app_name,
            timeline_start_idx=timeline_start_idx,
            logger_cfg=self.logger_cfg)

    def to_dict(self) -> dict[str, Any]:
        """Convert current motion clip to dictionary.

        Returns:
            dict: Dictionary representation of current motion clip.
        """
        ret_dict = dict(
            joint_names=self.joint_names,
            rotmat=self.joint_rotmat,
            transl=self.root_world_position,
            len=self.n_frames,
            cutoff_frames=self.cutoff_frames,
            cutoff_ranges=self.cutoff_ranges,
            restpose_name=self.restpose_name,
        )
        if self.blendshape_names is not None and self.blendshape_values is not None:
            ret_dict['blendshape_names'] = self.blendshape_names
            ret_dict['blendshape_values'] = self.blendshape_values
        if self.motion_record_id is not None:
            ret_dict['motion_record_id'] = self.motion_record_id
        if self.timeline_start_idx is not None:
            ret_dict['timeline_start_idx'] = self.timeline_start_idx
        if self.app_name is not None:
            ret_dict['app_name'] = self.app_name
        return ret_dict

    def to_npz(self) -> io.BytesIO:
        """Convert current motion clip to compressed npz file.

        Returns:
            io.BytesIO: Compressed npz file.
        """
        npz_dict = self.to_dict()
        ret_io = io.BytesIO()
        np.savez_compressed(ret_io, **npz_dict)
        ret_io.seek(0)
        return ret_io

    @classmethod
    def from_dict(cls,
                  motion_dict: dict[str, Any],
                  logger_cfg: None | dict[str, Any] = None) -> 'MotionClip':
        """Construct motion clip from dictionary.

        Args:
            motion_dict (dict): Dictionary representation of motion clip.
            logger_cfg (Union[None, dict], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Defaults to None.

        Returns:
            MotionClip: Motion clip constructed from dictionary.
        """
        joint_names = motion_dict['joint_names']
        joint_rotmat = motion_dict['rotmat']
        root_world_position = motion_dict['transl']
        n_frames = motion_dict['len']
        restpose_name = motion_dict['restpose_name']
        cutoff_frames = motion_dict.get('cutoff_frames', None)
        cutoff_ranges = motion_dict.get('cutoff_ranges', None)
        blendshape_names = motion_dict.get('blendshape_names', None)
        blendshape_values = motion_dict.get('blendshape_values', None)
        motion_record_id = motion_dict.get('motion_record_id', None)
        timeline_start_idx = motion_dict.get('timeline_start_idx', None)
        app_name = motion_dict.get('app_name', None)
        return MotionClip(
            n_frames=n_frames,
            joint_names=joint_names,
            joint_rotmat=joint_rotmat,
            root_world_position=root_world_position,
            cutoff_frames=cutoff_frames,
            cutoff_ranges=cutoff_ranges,
            restpose_name=restpose_name,
            app_name=app_name,
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            motion_record_id=motion_record_id,
            timeline_start_idx=timeline_start_idx,
            logger_cfg=logger_cfg)

    @classmethod
    def from_npz(cls,
                 npz_io: io.BytesIO,
                 logger_cfg: None | dict[str, Any] = None) -> 'MotionClip':
        """Construct motion clip from compressed npz file.

        Args:
            npz_io (io.BytesIO): Compressed npz file.
            logger_cfg (Union[None, dict], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Defaults to None.

        Returns:
            MotionClip: Motion clip constructed from compressed npz file.
        """
        npz_dict = load_npz(npz_io)
        return cls.from_dict(npz_dict, logger_cfg=logger_cfg)

    @classmethod
    def concat(cls,
               motion_clips: list['MotionClip']) -> 'MotionClip':
        """Concatenate a list of motion clips.

        Args:
            motion_clips (list[MotionClip]): List of motion clips.
            logger_cfg (Union[None, dict], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Defaults to None.

        Returns:
            MotionClip: Concatenated motion clip.
        """
        logger_cfg = dict()
        logger_cfg['logger_name'] = cls.__name__
        logger = setup_logger(**logger_cfg)
        if len(motion_clips) <= 0:
            msg = 'Parameter motion_clips of MotionClip.concat() is empty list.'
            logger.error(msg)
            raise ValueError(msg)
        logger = motion_clips[0].logger
        n_frames = sum([mc.n_frames for mc in motion_clips])
        restpose_name = motion_clips[0].restpose_name
        app_name = motion_clips[0].app_name
        joint_names = motion_clips[0].joint_names
        blendshape_names = motion_clips[0].blendshape_names
        timeline_start_idx = motion_clips[0].timeline_start_idx
        # check restpose name, app name and joint names
        for idx in range(1, len(motion_clips)):
            cur_restpose_name = motion_clips[idx].restpose_name
            cur_app_name = motion_clips[idx].app_name
            if restpose_name is not None:
                if cur_restpose_name != restpose_name:
                    msg = 'Parameter motion_clips of MotionClip.concat() contains ' +\
                        'different restpose_name.\n' +\
                        f'Clip0: {restpose_name}\n' +\
                        f'Clip{idx}: {cur_restpose_name}'
                    logger.error(msg)
                    raise ValueError(msg)
            if app_name is not None:
                if cur_app_name != app_name:
                    msg = 'Parameter motion_clips of MotionClip.concat() contains ' +\
                        'different app_name.\n' +\
                        f'Clip0: {app_name}\n' +\
                        f'Clip{idx}: {cur_app_name}'
                    logger.error(msg)
                    raise ValueError(msg)
            cur_joint_names = motion_clips[idx].joint_names
            if len(cur_joint_names) != len(joint_names):
                msg = 'Parameter motion_clips of MotionClip.concat() contains ' +\
                    'different joint_names.\n' +\
                    f'Clip0: {joint_names}, ' +\
                    f'Clip{idx}: {cur_joint_names}'
                logger.error(msg)
                raise ValueError(msg)
            cur_blendshape_names = motion_clips[idx].blendshape_names
            if blendshape_names != cur_blendshape_names:
                msg = 'Parameter motion_clips of MotionClip.concat() contains ' +\
                    'different blendshape_names.\n' +\
                    f'Clip0: {blendshape_names}, ' +\
                    f'Clip{idx}: {cur_blendshape_names}'
                logger.error(msg)
                raise ValueError(msg)
        joint_rotmat = np.zeros(
            (n_frames, len(joint_names), 3, 3),
            dtype=motion_clips[0].joint_rotmat.dtype)
        if blendshape_names is not None:
            blendshape_values = np.zeros(
                (n_frames, len(blendshape_names)),
                dtype=motion_clips[0].blendshape_values.dtype)
        else:
            blendshape_values = None
        cutoff_frames = []
        cutoff_ranges = []
        clip_start_frame = 0
        for idx in range(0, len(motion_clips)):
            cur_rotmat = motion_clips[idx].joint_rotmat
            cur_joint_names = motion_clips[idx].joint_names
            cur_n_frames = motion_clips[idx].n_frames
            joint_rotmat[clip_start_frame:clip_start_frame +
                         cur_n_frames, :] = cur_rotmat
            for cutoff_frame in motion_clips[idx].cutoff_frames:
                rela_frame_idx = cutoff_frame[0]
                abs_frame_idx = clip_start_frame + rela_frame_idx
                if len(cutoff_frames) > 0 and abs_frame_idx == cutoff_frames[-1][0]:
                    left_priority = cutoff_frames[-1][1]
                    right_priority = cutoff_frame[2]
                    cutoff_frames.append((abs_frame_idx, left_priority, right_priority))
                else:
                    cutoff_frames.append(
                        (abs_frame_idx, cutoff_frame[1], cutoff_frame[2]))
            if motion_clips[idx].cutoff_ranges is not None:
                for cutoff_range in motion_clips[idx].cutoff_ranges:
                    range_start = cutoff_range[0]
                    range_end = cutoff_range[1]
                    left_priority = cutoff_range[2]
                    right_priority = cutoff_range[3]
                    cutoff_ranges.append((
                        clip_start_frame + range_start,
                        clip_start_frame + range_end,
                        left_priority,
                        right_priority))
            if blendshape_values is not None:
                cur_blendshape_values = motion_clips[idx].blendshape_values
                blendshape_values[clip_start_frame:clip_start_frame +
                                  cur_n_frames, :] = cur_blendshape_values
            clip_start_frame += cur_n_frames
        if len(cutoff_ranges) == 0:
            cutoff_ranges = None
        root_world_position = np.concatenate(
            [mc.root_world_position for mc in motion_clips],
            axis=0,
            dtype=motion_clips[0].root_world_position.dtype)
        return MotionClip(
            n_frames=n_frames,
            joint_names=joint_names,
            joint_rotmat=joint_rotmat,
            root_world_position=root_world_position,
            cutoff_frames=cutoff_frames,
            cutoff_ranges=cutoff_ranges,
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values,
            restpose_name=restpose_name,
            app_name=app_name,
            timeline_start_idx=timeline_start_idx,
            logger_cfg=logger_cfg)
