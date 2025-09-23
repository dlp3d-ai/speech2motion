import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..data_structures.motion_clip import MotionClip
from ..data_structures.motion_record import MotionRecord
from .base_merge import BaseMerge


class WrongInputNumber(Exception):
    """Exception raised when the number of MotionClip or MotionRecord objects is
    incorrect.

    This exception is raised when the input MotionClip or MotionRecord count
    does not match the expected number for the operation.
    """
    pass

class Blending(BaseMerge):
    """Motion blending merge class.

    This class is used to blend and merge two MotionClip objects with support
    for smooth transitions including startup and recovery phases.
    """

    def __init__(self,
                 max_workers: int = 1,
                 thread_pool_executor: ThreadPoolExecutor | None = None,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the motion blending merge class.

        Args:
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
        BaseMerge.__init__(
            self,
            max_workers=max_workers,
            thread_pool_executor=thread_pool_executor,
            logger_cfg=logger_cfg)

    async def merge(
            self,
            motion_clips: list[MotionClip],
            align_frame: int,
            startup_frame: int,
            recovery_frame: int,
            ) -> MotionClip:
        """Merge MotionClip objects with blending.

        Args:
            motion_clips (list[MotionClip]):
                List of MotionClip objects to be merged.
            align_frame (int):
                Alignment frame specifying the position in the second motion
                where alignment should occur.
            startup_frame (int):
                Number of startup frames specifying the startup portion of
                the first motion.
            recovery_frame (int):
                Number of recovery frames specifying the recovery portion of
                the first motion.

        Returns:
            MotionClip:
                The merged MotionClip object.

        Raises:
            WrongInputNumber:
                Raised when the number of MotionClip objects is not 2.
            ValueError:
                Raised when frame indices are out of range or motion
                attributes do not match.
        """
        if len(motion_clips) != 2:
            msg = 'Incorrect number of MotionClips, ' +\
                f'expected 2 but got {len(motion_clips)}'
            self.logger.error(msg)
            raise WrongInputNumber(msg)
        motion_clip_a = motion_clips[0]
        motion_clip_b = motion_clips[1]
        frame_idxs_to_check = (
            align_frame,
            align_frame + startup_frame,
            align_frame + recovery_frame,
            align_frame + motion_clip_a.n_frames,
        )
        for frame_idx in frame_idxs_to_check:
            if frame_idx < 0 or frame_idx > motion_clip_b.n_frames:
                msg = f'Frame index {frame_idx} out of range, ' +\
                    f'motion_clip_a has {motion_clip_a.n_frames} frames, ' +\
                    f'motion_clip_b has {motion_clip_b.n_frames} frames, ' +\
                    f'align_frame is {align_frame}, ' +\
                    f'startup_frame is {startup_frame}, ' +\
                    f'recovery_frame is {recovery_frame}.'
                self.logger.error(msg)
                raise ValueError(msg)
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
                    f'ID is {id_a}, ' +\
                    f'motion b blendshape count is {blendshape_values_b.shape[1]}, ' +\
                    f'ID is {id_b}.'
                self.logger.error(msg)
                raise ValueError(msg)
        if blendshape_values_a is None and blendshape_values_b is not None:
            msg = 'Motion a does not have blendshape_values attribute, ' +\
                f'ID is {id_a}, ' +\
                f'motion b has blendshape_values attribute, ID is {id_b}.'
            self.logger.error(msg)
            raise ValueError(msg)
        if blendshape_values_a is not None and blendshape_values_b is None:
            msg = f'Motion a has blendshape_values attribute, ID is {id_a}, ' +\
                f'motion b does not have blendshape_values attribute, ID is {id_b}.'
            self.logger.error(msg)
            raise ValueError(msg)
        loop = asyncio.get_event_loop()
        ret_motion_clip = await loop.run_in_executor(
            self.executor,
            motion_clip_b.clone
        )
        joint_rotmat = ret_motion_clip.joint_rotmat
        root_world_position = ret_motion_clip.root_world_position
        blendshape_values = ret_motion_clip.blendshape_values
        # Allow thread pool parallelization
        # For replacement
        joint_rotmat[align_frame+startup_frame: align_frame+recovery_frame] = \
            motion_clip_a.joint_rotmat[startup_frame:recovery_frame]
        root_world_position[align_frame+startup_frame: align_frame+recovery_frame] = \
            motion_clip_a.root_world_position[startup_frame:recovery_frame]
        if blendshape_values is not None:
            blendshape_values[align_frame+startup_frame: align_frame+recovery_frame] = \
                motion_clip_a.blendshape_values[startup_frame:recovery_frame]
        # For startup
        blend_tasks: list[dict] = list()
        if startup_frame > 0:
            startup_clip_a_future = loop.run_in_executor(
                self.executor,
                motion_clip_a.slice,
                0,
                startup_frame
            )
            startup_clip_b_future = loop.run_in_executor(
                self.executor,
                motion_clip_b.slice,
                align_frame,
                align_frame + startup_frame
            )
            startup_clip_a, startup_clip_b = await asyncio.gather(
                startup_clip_a_future,
                startup_clip_b_future
            )
            blend_startup_clip_future = self._blend_motion_clips(
                startup_clip_a,
                startup_clip_b
            )
            blend_tasks.append(
                dict(
                    future=blend_startup_clip_future,
                    start=align_frame,
                    end=align_frame + startup_frame,
                )
            )
        # For recovery
        if recovery_frame < motion_clip_a.n_frames:
            recovery_clip_a_future = loop.run_in_executor(
                self.executor,
                motion_clip_a.slice,
                recovery_frame,
                motion_clip_a.n_frames
            )
            recovery_clip_b_future = loop.run_in_executor(
                self.executor,
                motion_clip_b.slice,
                align_frame + recovery_frame,
                align_frame + motion_clip_a.n_frames
            )
            recovery_clip_a, recovery_clip_b = await asyncio.gather(
                recovery_clip_a_future,
                recovery_clip_b_future
            )
            blend_recovery_clip_future = self._blend_motion_clips(
                recovery_clip_b,
                recovery_clip_a,
            )
            blend_tasks.append(
                dict(
                    future=blend_recovery_clip_future,
                    start=align_frame + recovery_frame,
                    end=align_frame + motion_clip_a.n_frames,
                )
            )
        blend_results = await asyncio.gather(
            *[task['future'] for task in blend_tasks]
        )
        for idx, motion_clip in enumerate(blend_results):
            start_idx = blend_tasks[idx]['start']
            end_idx = blend_tasks[idx]['end']
            joint_rotmat[start_idx:end_idx] = motion_clip.joint_rotmat
            root_world_position[start_idx:end_idx] = motion_clip.root_world_position
            if blendshape_values is not None:
                blendshape_values[start_idx:end_idx] = motion_clip.blendshape_values
        ret_motion_clip.set_joint_rotmat(
            joint_rotmat=joint_rotmat,
            joint_names=ret_motion_clip.joint_names
        )
        ret_motion_clip.set_root_world_position(
            root_world_position=root_world_position
        )
        ret_motion_clip.set_blendshape(
            blendshape_names=ret_motion_clip.blendshape_names,
            blendshape_values=blendshape_values
        )
        return ret_motion_clip

    async def _blend_motion_clips(
            self,
            motion_clip_a: MotionClip,
            motion_clip_b: MotionClip) -> MotionClip:
        """Blend two MotionClip objects.

        Args:
            motion_clip_a (MotionClip):
                First motion clip to be blended.
            motion_clip_b (MotionClip):
                Second motion clip to be blended.

        Returns:
            MotionClip:
                The blended motion clip.
        """
        loop = asyncio.get_event_loop()
        joint_rotmat_a = motion_clip_a.joint_rotmat
        joint_rotmat_b = motion_clip_b.joint_rotmat
        joint_rot6d_a_future = loop.run_in_executor(
            self.executor,
            self.flatten_rotmat,
            joint_rotmat_a
        )
        joint_rot6d_b_future = loop.run_in_executor(
            self.executor,
            self.flatten_rotmat,
            joint_rotmat_b
        )
        joint_rot6d_a, joint_rot6d_b = await asyncio.gather(
            joint_rot6d_a_future,
            joint_rot6d_b_future
        )
        blend_joint_rot6d_future = loop.run_in_executor(
            self.executor,
            self._blend_ndarray,
            joint_rot6d_a,
            joint_rot6d_b
        )
        root_world_position_a = motion_clip_a.root_world_position
        root_world_position_b = motion_clip_b.root_world_position
        blend_root_world_position_future = loop.run_in_executor(
            self.executor,
            self._blend_ndarray,
            root_world_position_a,
            root_world_position_b
        )
        blend_joint_rot6d, blend_root_world_position = \
            await asyncio.gather(
                blend_joint_rot6d_future,
                blend_root_world_position_future
            )
        blend_joint_rotmat = await loop.run_in_executor(
            self.executor,
            self.restore_rotmat,
            blend_joint_rot6d,
            motion_clip_a.n_frames,
            len(motion_clip_a.joint_names)
        )
        ret_motion_clip = motion_clip_b.clone()
        ret_motion_clip.set_joint_rotmat(
            joint_rotmat=blend_joint_rotmat,
            joint_names=ret_motion_clip.joint_names
        )
        ret_motion_clip.set_root_world_position(
            blend_root_world_position
        )
        blendshape_values_a = motion_clip_a.blendshape_values
        blendshape_values_b = motion_clip_b.blendshape_values
        if blendshape_values_a is not None and blendshape_values_b is not None:
            blend_blendshape_values = await loop.run_in_executor(
                self.executor,
                self._blend_ndarray,
                blendshape_values_a,
                blendshape_values_b
            )
            ret_motion_clip.set_blendshape(
                blendshape_names=ret_motion_clip.blendshape_names,
                blendshape_values=blend_blendshape_values
            )
        return ret_motion_clip

    def _blend_ndarray(
            self,
            ndarray_a: np.ndarray,
            ndarray_b: np.ndarray) -> np.ndarray:
        """Blend two numpy arrays using linear interpolation.

        Uses linear interpolation to blend two arrays, gradually transitioning
        from the first array to the second array.

        Args:
            ndarray_a (np.ndarray):
                First array with shape [n_frames, ...].
            ndarray_b (np.ndarray):
                Second array with shape [n_frames, ...].

        Returns:
            np.ndarray:
                Blended array with the same shape as input arrays.
        """
        shape_backup = ndarray_a.shape
        n_frames = shape_backup[0]
        vec_a = ndarray_a.reshape(n_frames, -1)
        vec_b = ndarray_b.reshape(n_frames, -1)
        progress_a = np.linspace(0, 1, n_frames)
        progress_b = 1 - progress_a
        progress_a = progress_a.reshape(-1, 1)
        progress_b = progress_b.reshape(-1, 1)
        # Linear interpolation blending
        blend_vec = vec_a * progress_a + vec_b * progress_b
        blend_ndarray = blend_vec.reshape(*shape_backup)
        return blend_ndarray

    async def predict_merge_n_frames(self,
                                     motion_records: list[MotionRecord]) -> int:
        """Predict the total number of frames after merging MotionRecord objects.

        Args:
            motion_records (list[MotionRecord]):
                List of MotionRecord objects to be merged.

        Returns:
            int:
                Total number of frames in the merged motion sequence.

        Raises:
            WrongInputNumber:
                Raised when the number of MotionRecord objects is not 2.
            ValueError:
                Raised when the first motion has more frames than the second motion.
        """
        if len(motion_records) != 2:
            msg = 'Incorrect number of MotionRecords, ' +\
                f'expected 2 but got {len(motion_records)}'
            self.logger.error(msg)
            raise WrongInputNumber(msg)
        if motion_records[0].n_frames > motion_records[1].n_frames:
            msg = f'motion_records[0] has {motion_records[0].n_frames} frames, ' +\
                'which is greater than motion_records[1] with ' +\
                f'{motion_records[1].n_frames} frames'
            self.logger.error(msg)
            raise ValueError(msg)
        return motion_records[1].n_frames

    @staticmethod
    def flatten_rotmat(rotmat: np.ndarray) -> np.ndarray:
        """Convert rotation matrices to rot6d format and flatten for vector processing.

        Args:
            rotmat (np.ndarray):
                Rotation matrices with shape [n_frames, n_joints, 3, 3].

        Returns:
            np.ndarray:
                Flattened array with shape [n_frames, n_joints * 6].
        """
        n_frames, n_joints, _, _ = rotmat.shape
        rot6d = rotmat[:, :, :2, :].reshape(
            n_frames, n_joints * 6)
        return rot6d

    @staticmethod
    def restore_rotmat(
            rot6d: np.ndarray,
            n_frames: int,
            n_joints: int) -> np.ndarray:
        """Restore rot6d format back to rotation matrices.

        Args:
            rot6d (np.ndarray):
                Flattened array with shape [n_frames, n_joints * 6].
            n_frames (int):
                Number of frames.
            n_joints (int):
                Number of joints.

        Returns:
            np.ndarray:
                Rotation matrices with shape [n_frames, n_joints, 3, 3].
        """
        rot6d = rot6d.reshape(n_frames, n_joints, 6)
        first_row = rot6d[:, :, :3]
        second_row = rot6d[:, :, 3:]
        first_row = first_row.reshape(n_frames * n_joints, 3)
        second_row = second_row.reshape(n_frames * n_joints, 3)
        third_row = np.cross(first_row, second_row)
        rotmat = np.concatenate([
                first_row[:, None, :],
                second_row[:, None, :],
                third_row[:, None, :]],
            axis=1).reshape(n_frames, n_joints, 3, 3)
        # re-orthogonalizing rotation matrix
        u, _, v = np.linalg.svd(rotmat)
        rotmat = u @ v
        return rotmat
