from typing import Any

from ...utils.super import Super


class RangeOverlapError(Exception):
    """Exception raised when ranges overlap.
    """
    pass

class RangeInvalidError(Exception):
    """Exception raised when range is invalid.
    """
    pass

class Random(Super):
    """Object for storing random motion related data from Random table.
    """

    def __init__(
            self,
            cutoff_frames: list[int] | None = None,
            cutoff_ranges: list[tuple[int, int]] | None = None,
            logger_cfg: None | dict = None):
        """Initialize random motion object.

        Args:
            cutoff_frames (list[int] | None, optional):
                Safe cutoff frames for random motion.
            cutoff_ranges (list[tuple[int, int]] | None, optional):
                Safe cutoff ranges for random motion.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name.
                Defaults to None.
        """
        super().__init__(logger_cfg)
        self.cutoff_frames = sorted(cutoff_frames) if cutoff_frames else None
        self.cutoff_ranges = self._sort_ranges(cutoff_ranges)

    def _sort_ranges(self, cutoff_ranges: list[tuple[int, int]] | None) -> None:
        """Sort cutoff_ranges and validate them.

        Args:
            cutoff_ranges (list[tuple[int, int]] | None):
                List of cutoff ranges to sort and validate.

        Returns:
            list[tuple[int, int]] | None: Sorted and validated cutoff ranges.

        Raises:
            RangeInvalidError: If any range has invalid length.
            RangeOverlapError: If any ranges overlap.
        """
        if cutoff_ranges is None:
            return None
        # sort by start
        cutoff_ranges.sort(key=lambda x: x[0])
        # check if range length is valid
        for start, end in cutoff_ranges:
            if start >= end:
                msg = f'Range [{start}, {end}) has invalid length.'
                self.logger.error(msg)
                raise RangeInvalidError(msg)
        # check if ranges overlap
        for i in range(len(cutoff_ranges) - 1):
            if cutoff_ranges[i][1] > cutoff_ranges[i + 1][0]:
                msg = (f'Range [{cutoff_ranges[i][0]}, {cutoff_ranges[i][1]}) and '
                       f'range [{cutoff_ranges[i + 1][0]}, {cutoff_ranges[i + 1][1]}) '
                       f'overlap.')
                self.logger.error(msg)
                raise RangeOverlapError(msg)
        return cutoff_ranges

    def to_dict(self) -> dict[str, Any]:
        """Convert Random object to dictionary.

        Returns:
            dict[str, Any]: Dictionary containing Random object attributes.
        """
        return {
            'cutoff_frames': self.cutoff_frames,
            'cutoff_ranges': self.cutoff_ranges
        }

    @classmethod
    def from_dict(cls, data: dict, logger_cfg: None | dict = None) -> 'Random':
        """Create Random object from dictionary.

        Args:
            data (dict): Dictionary containing Random object attributes.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Defaults to None.

        Returns:
            Random: Random object with attributes from dictionary.
        """
        return cls(
            cutoff_frames=data.get('cutoff_frames', None),
            cutoff_ranges=data.get('cutoff_ranges', None),
            logger_cfg=logger_cfg
        )

    def clone(self) -> 'Random':
        """Clone Random object.

        Returns:
            Random: Cloned Random object.
        """
        return Random(
            cutoff_frames=self.cutoff_frames,
            cutoff_ranges=self.cutoff_ranges)
