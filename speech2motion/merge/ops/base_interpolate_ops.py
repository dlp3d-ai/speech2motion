from abc import ABC, abstractmethod

import numpy as np

from ...utils.super import Super
from .base_ops import BaseOps


class BaseInterpolateOps(BaseOps, ABC):
    """Base class for interpolation transition operations.

    Any input motion animation cannot be shorter than __class__.NFRAMES_LOWERBOUND.
    """
    NFRAMES_LOWERBOUND = 0

    def __init__(self,
                 transit_frames: int,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the interpolation transition operations.

        Args:
            transit_frames (int):
                Number of transition frames to generate.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        ABC.__init__(self)
        Super.__init__(self, logger_cfg)
        self.transit_frames = transit_frames

    def interpolate_rotmat(
        self,
        motion_rotmat_a: np.ndarray,
        motion_rotmat_b: np.ndarray,
        transit_frames_overwrite: int | None = None,
    ) -> np.ndarray:
        """Create smooth transition between two rotation matrix sequences.

        The base class uses flatten_rotmat and restore_rotmat to convert rotation
        matrices to vector form by default, then processes using interpolate_vector.
        If subclasses need independent processing methods, they should override
        the interpolate_rotmat method.

        Args:
            motion_rotmat_a (np.ndarray):
                First rotation matrix sequence with shape [n_frames_a, n_joints, 3, 3].
            motion_rotmat_b (np.ndarray):
                Second rotation matrix sequence with shape [n_frames_b, n_joints, 3, 3].
            transit_frames_overwrite (int | None, optional):
                Override the transition frames set during construction.
                Defaults to None.

        Returns:
            np.ndarray:
                Interpolated rotation matrix sequence with shape
                [n_frames_a + transit_frames + n_frames_b, n_joints, 3, 3].
        """
        n_frames_a, n_joints_a = motion_rotmat_a.shape[:2]
        n_frames_b, n_joints_b = motion_rotmat_b.shape[:2]
        if n_joints_a != n_joints_b:
            self.logger.error('Numbers of joints are different: ' +
                              f'n_joints_a={n_joints_a}, ' +
                              f'n_joints_b={n_joints_b}')
            raise ValueError
        motion_rot6d_a = self.__class__.flatten_rotmat(motion_rotmat_a)
        motion_rot6d_b = self.__class__.flatten_rotmat(motion_rotmat_b)
        interpolate_rot6d = self.interpolate_vector(
            motion_vec_a=motion_rot6d_a,
            motion_vec_b=motion_rot6d_b,
            transit_frames_overwrite=transit_frames_overwrite)
        transit_frames = self.transit_frames \
            if transit_frames_overwrite is None \
            else transit_frames_overwrite
        interpolate_rotmat = self.__class__.restore_rotmat(
            rot6d=interpolate_rot6d,
            n_frames=n_frames_a + transit_frames + n_frames_b,
            n_joints=n_joints_a)
        return interpolate_rotmat

    @abstractmethod
    def interpolate_vector(
        self,
        motion_vec_a: np.ndarray,
        motion_vec_b: np.ndarray,
        transit_frames_overwrite: int | None = None,
    ) -> np.ndarray:
        """Create smooth transition between two vector sequences.

        Args:
            motion_vec_a (np.ndarray):
                First vector sequence with shape [n_frames_a, n_dim].
            motion_vec_b (np.ndarray):
                Second vector sequence with shape [n_frames_b, n_dim].
            transit_frames_overwrite (int | None, optional):
                Override the transition frames set during construction.
                Defaults to None.

        Returns:
            np.ndarray:
                Interpolated vector sequence with shape
                [n_frames_a + transit_frames + n_frames_b, n_dim].
        """
        pass
