import numpy as np

from ...utils.super import Super


class BaseOps(Super):
    """Base class for transition operations.

    This abstract base class provides common functionality for motion
    transition operations, including rotation matrix conversion utilities.
    """

    def __init__(self,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the base transition operations class.

        Args:
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        Super.__init__(self, logger_cfg)


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
