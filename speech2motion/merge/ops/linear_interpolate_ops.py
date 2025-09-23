import numpy as np

from .base_interpolate_ops import BaseInterpolateOps


class LinearInterpolateOps(BaseInterpolateOps):
    """Linear interpolation transition operations for rotation matrices or position
    vectors.

    Input motion animations cannot be shorter than
    LinearInterpolateOps.NFRAMES_LOWERBOUND=0.
    """
    NFRAMES_LOWERBOUND = 0

    def __init__(self,
                 transit_frames: int,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the linear interpolation transition operations.

        Args:
            transit_frames (int):
                Number of transition frames to generate.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        super().__init__(transit_frames, logger_cfg)

    def interpolate_vector(
        self,
        motion_vec_a: np.ndarray,
        motion_vec_b: np.ndarray,
        transit_frames_overwrite: int | None = None,
    ) -> np.ndarray:
        """Create smooth transition between two vector sequences using linear
        interpolation.

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
        n_frames_a, n_dim_a = motion_vec_a.shape
        n_frames_b, n_dim_b = motion_vec_b.shape
        assert n_frames_a > 0 and n_frames_b > 0
        if n_dim_a != n_dim_b:
            msg = 'Multi-frame vector dimensions do not match: ' + \
                f'vector_a dimension is {n_dim_a}, vector_b dimension is {n_dim_b}'
            self.logger.error(msg)
            raise ValueError(msg)

        transit_frames = self.transit_frames \
            if transit_frames_overwrite is None \
            else transit_frames_overwrite
        n_frames_transit = transit_frames

        # Calculate transition vectors
        ref_frame_a = motion_vec_a[n_frames_a - 1]
        ref_frame_b = motion_vec_b[0]
        # Calculate transition vectors
        merge_transit_seq = np.zeros((n_frames_transit, n_dim_a))
        # Calculate transition vectors
        for i in range(n_frames_transit):
            alpha = (i + 1) / (n_frames_transit + 1)
            merge_transit_seq[i] = (1 -
                                    alpha) * ref_frame_a + alpha * ref_frame_b

        merge_seq = np.concatenate(
            [motion_vec_a, merge_transit_seq, motion_vec_b])
        return merge_seq
