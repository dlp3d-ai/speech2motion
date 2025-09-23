import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from ..data_structures.motion_clip import MotionClip
from ..data_structures.motion_record import MotionRecord
from ..merge.base_merge import BaseMerge
from .ops.builder import build_merge_ops

if TYPE_CHECKING:
    from .ops.base_interpolate_ops import BaseInterpolateOps


class Interpolation(BaseMerge):
    """Motion clip interpolation merge class.

    This class merges multiple MotionClip objects using interpolation
    transitions between clips, creating smooth transitions between
    different motion sequences.
    """
    INTERPOLATE_OPS_NAMES = ('LinearInterpolateOps',)

    def __init__(self,
                 transit_frames: int,
                 max_workers: int = 1,
                 thread_pool_executor: ThreadPoolExecutor | None = None,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the motion clip interpolation merge class.

        Args:
            transit_frames (int):
                Number of transition frames for interpolation between clips.
            max_workers (int, optional):
                Maximum number of worker threads for concurrent processing.
                Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                External thread pool executor to use. If None, creates a new
                one with max_workers. Defaults to None.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        super().__init__(max_workers, thread_pool_executor, logger_cfg)
        self.transit_frames = transit_frames
        self.ops_list: list[BaseInterpolateOps] | None = None
        self._build_ops()


    def _build_ops(self) -> None:
        """Build interpolation transition operations.

        Creates a list of interpolation operations based on the configured
        operation names and transition frames.
        """
        ops_list = []
        for op_name in self.INTERPOLATE_OPS_NAMES:
            cfg = dict(
                type=op_name,
                transit_frames=self.transit_frames,
                logger_cfg=self.logger_cfg
            )
            ops = build_merge_ops(cfg)
            ops_list.append(ops)
        self.ops_list = ops_list

    async def merge(self,
              motion_clips: list[MotionClip]) -> MotionClip:
        """Merge MotionClip objects using interpolation.

        Args:
            motion_clips (list[MotionClip]):
                List of MotionClip objects to be merged.

        Returns:
            MotionClip:
                The merged MotionClip object with interpolation transitions.
        """
        motion_clips_to_merge = motion_clips.copy()
        loop = asyncio.get_running_loop()
        while len(motion_clips_to_merge) > 1:
            coroutines = []
            for i in range(0, len(motion_clips_to_merge), 2):
                if i + 1 < len(motion_clips_to_merge):
                    coroutine = loop.run_in_executor(
                        self.executor,
                        self._merge_task,
                        motion_clips_to_merge[i],
                        motion_clips_to_merge[i + 1]
                    )
                    coroutines.append(coroutine)
            merge_results = await asyncio.gather(*coroutines)
            result_list = merge_results.copy()
            if len(motion_clips_to_merge) % 2 == 1:
                result_list.append(motion_clips_to_merge[-1])
            motion_clips_to_merge = result_list
        return motion_clips_to_merge[0]

    def _merge_task(self,
                    motion_clip_a: MotionClip,
                    motion_clip_b: MotionClip) -> MotionClip:
        """Merge two MotionClip objects using interpolation.

        Args:
            motion_clip_a (MotionClip):
                First MotionClip to be merged.
            motion_clip_b (MotionClip):
                Second MotionClip to be merged.

        Returns:
            MotionClip:
                The merged MotionClip object with interpolation transition.
        """
        src_dtype = motion_clip_a.joint_rotmat.dtype
        id_a = motion_clip_a.motion_record_id
        id_b = motion_clip_b.motion_record_id
        restpose_name_a = motion_clip_a.restpose_name
        restpose_name_b = motion_clip_b.restpose_name
        if restpose_name_a != restpose_name_b:
            msg = 'Restpose names of motion a and motion b do not match, ' +\
                f'motion a restpose name is {restpose_name_a}, ID is {id_a}, ' +\
                f'motion b restpose name is {restpose_name_b}, ID is {id_b}.'
            self.logger.error(msg)
            raise ValueError(msg)
        joint_names_a = motion_clip_a.joint_names
        joint_names_b = motion_clip_b.joint_names
        if len(joint_names_a) != len(joint_names_b):
            msg = 'Joint counts of motion a and motion b do not match, ' +\
                f'motion a joint list is {joint_names_a}, ID is {id_a}, ' +\
                f'motion b joint list is {joint_names_b}, ID is {id_b}.'
            self.logger.error(msg)
            raise ValueError(msg)
        blendshape_values_a = motion_clip_a.blendshape_values
        blendshape_values_b = motion_clip_b.blendshape_values
        if blendshape_values_a is not None and blendshape_values_b is not None:
            if blendshape_values_a.shape[1] != blendshape_values_b.shape[1]:
                msg = 'Blendshape counts of motion a and motion b do not match, ' +\
                    f'motion a blendshape count is {blendshape_values_a.shape[1]}, ' +\
                    f'ID is {id_a}, motion b blendshape count is ' +\
                    f'{blendshape_values_b.shape[1]}, ID is {id_b}.'
                self.logger.error(msg)
                raise ValueError(msg)
        if blendshape_values_a is None and blendshape_values_b is not None:
            msg = 'Motion a does not have blendshape_values attribute, ' +\
                f'ID is {id_a}, motion b has blendshape_values attribute, ' +\
                f'ID is {id_b}.'
            self.logger.error(msg)
            raise ValueError(msg)
        if blendshape_values_a is not None and blendshape_values_b is None:
            msg = 'Motion a has blendshape_values attribute, ' +\
                f'ID is {id_a}, motion b does not have blendshape_values attribute, ' +\
                f'ID is {id_b}.'
            self.logger.error(msg)
            raise ValueError(msg)
        n_frames_a = motion_clip_a.n_frames
        n_frames_b = motion_clip_b.n_frames
        priority_a = self._get_cutoff_priority(
            motion_clip_a,
            n_frames_a - 1,
            1
        )
        priority_b = self._get_cutoff_priority(
            motion_clip_b,
            n_frames_b - 1,
            2
        )
        if priority_a < priority_b:
            cut_frames_a = self.transit_frames
            cut_frames_b = 0
        elif priority_a > priority_b:
            cut_frames_a = 0
            cut_frames_b = self.transit_frames
        else:
            cut_frames_a = int(self.transit_frames / 2)
            cut_frames_b = self.transit_frames - cut_frames_a
        if cut_frames_a >= n_frames_a - 1:
            cut_frames_a = n_frames_a - 1
        if cut_frames_b >= n_frames_b - 1:
            cut_frames_b = n_frames_b - 1
        actual_transit_frames = cut_frames_a + cut_frames_b
        if actual_transit_frames < self.transit_frames:
            msg = 'Due to insufficient motion frames, transition frames ' +\
                f'corrected from {self.transit_frames} to {actual_transit_frames}, ' +\
                f'where motion a contributes {cut_frames_a} frames, ' +\
                f'motion b contributes {cut_frames_b} frames.'
            self.logger.warning(msg)
        n_frames_a_interpolate = n_frames_a - cut_frames_a
        n_frames_b_interpolate = n_frames_b - cut_frames_b
        safe_blend_ops = None
        for blend_ops in self.ops_list:
            a_ok = blend_ops.__class__.NFRAMES_LOWERBOUND <= \
                n_frames_a_interpolate
            b_ok = blend_ops.__class__.NFRAMES_LOWERBOUND <= \
                n_frames_b_interpolate
            if a_ok and b_ok:
                safe_blend_ops = blend_ops
                break
        if safe_blend_ops is None:
            msg = 'All interpolation transition operations cannot satisfy ' +\
                'the minimum length requirements for concatenation input. ' +\
                'Please add more basic interpolation transition operations in config.'
            self.logger.error(msg)
            raise ValueError(msg)
        a_end_idx = n_frames_a - cut_frames_a
        a_start_idx = max(0, a_end_idx - 15)
        b_start_idx = cut_frames_b
        b_end_idx = min(n_frames_b, b_start_idx + 15)
        rotmat_merge = safe_blend_ops.interpolate_rotmat(
            motion_clip_a.joint_rotmat[a_start_idx:a_end_idx],
            motion_clip_b.joint_rotmat[b_start_idx:b_end_idx],
            transit_frames_overwrite=actual_transit_frames
        )
        position_merge = safe_blend_ops.interpolate_vector(
            motion_clip_a.root_world_position[a_start_idx:a_end_idx],
            motion_clip_b.root_world_position[b_start_idx:b_end_idx],
            transit_frames_overwrite=actual_transit_frames
        )
        rotmat_for_concat = [rotmat_merge,]
        position_for_concat = [position_merge,]
        if blendshape_values_a is not None and blendshape_values_b is not None:
            blendshape_values_merge = safe_blend_ops.interpolate_vector(
                blendshape_values_a[a_start_idx:a_end_idx],
                blendshape_values_b[b_start_idx:b_end_idx],
                transit_frames_overwrite=actual_transit_frames
            )
            blendshape_values_for_concat = [blendshape_values_merge,]
        if a_start_idx > 0:
            position_for_concat.insert(
                0, motion_clip_a.root_world_position[0:a_start_idx])
            rotmat_for_concat.insert(0, motion_clip_a.joint_rotmat[0:a_start_idx])
            if blendshape_values_a is not None:
                blendshape_values_for_concat.insert(
                    0, blendshape_values_a[0:a_start_idx])
        if b_end_idx < n_frames_b:
            position_for_concat.append(motion_clip_b.root_world_position[b_end_idx:])
            rotmat_for_concat.append(motion_clip_b.joint_rotmat[b_end_idx:])
            if blendshape_values_b is not None:
                blendshape_values_for_concat.append(
                    blendshape_values_b[b_end_idx:])
        rotmat_merge = np.concatenate(rotmat_for_concat, axis=0, dtype=src_dtype)
        position_merge = np.concatenate(position_for_concat, axis=0, dtype=src_dtype)
        if blendshape_values_a is not None and blendshape_values_b is not None:
            blendshape_names = motion_clip_a.blendshape_names
            blendshape_values_merge = np.concatenate(
                blendshape_values_for_concat, axis=0, dtype=src_dtype)
        else:
            blendshape_names = None
            blendshape_values_merge = None
        n_frames_merge = n_frames_a_interpolate + \
            actual_transit_frames + n_frames_b_interpolate
        motion_clip_concat = MotionClip.concat([
            motion_clip_a,
            motion_clip_b
        ])
        timeline_start_idx = motion_clip_a.timeline_start_idx
        motion_clip_merge = MotionClip(
            n_frames=n_frames_merge,
            joint_names=joint_names_a,
            joint_rotmat=rotmat_merge,
            root_world_position=position_merge,
            restpose_name=restpose_name_a,
            cutoff_frames=motion_clip_concat.cutoff_frames,
            cutoff_ranges=motion_clip_concat.cutoff_ranges,
            blendshape_names=blendshape_names,
            blendshape_values=blendshape_values_merge,
            motion_record_id=None,
            timeline_start_idx=timeline_start_idx,
            logger_cfg=self.logger_cfg
        )
        return motion_clip_merge

    def _get_cutoff_priority(self,
                          motion_clip: MotionClip,
                          frame_idx: int,
                          side_idx: int) -> int:
        """Get the priority of the specified cutoff position in merging.

        Args:
            motion_clip (MotionClip):
                MotionClip to get cutoff_frames from.
            frame_idx (int):
                Frame index to get cutoff_frames for.
            side_idx (int):
                Side index for cutoff_frames, 1 for left side, 2 for right side.

        Returns:
            int:
                Priority of the specified cutoff position in merging.
        """
        id = motion_clip.motion_record_id
        if motion_clip.cutoff_frames is None\
                or len(motion_clip.cutoff_frames) == 0:
            msg = f'motion_clip(ID={id}) cutoff_frames is empty, ' +\
                'using 0 as default priority. ' +\
                'Please check the source of motion_clip.'
            self.logger.warning(msg)
            priority = 0
        frame_found = False
        for cutoff_params in motion_clip.cutoff_frames:
            if cutoff_params[0] == frame_idx:
                priority = cutoff_params[side_idx]
                frame_found = True
                break
        if not frame_found:
            msg = f'motion_clip(ID={id}) cutoff_frames does not contain ' +\
                f'cutoff position for frame index {frame_idx}. ' +\
                'Using 0 as default priority.'
            self.logger.warning(msg)
            priority = 0
        return priority

    async def predict_merge_n_frames(self,
                                     motion_records: list[MotionRecord]) -> int:
        """Predict the total number of frames after merging MotionRecord objects.

        Args:
            motion_records (list[MotionRecord]):
                List of MotionRecord objects to be merged.

        Returns:
            int:
                Total number of frames in the merged motion sequence.
        """
        n_frames = 0
        for motion_record in motion_records:
            n_frames += motion_record.n_frames
        return n_frames
