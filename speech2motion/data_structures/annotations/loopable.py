from typing import Any

from ...utils.super import Super


class Loopable(Super):
    """Object for storing loopable motion related data from Loopable table.
    """
    def __init__(
            self,
            loop_start_frame: int,
            loop_end_frame: int,
            logger_cfg: None | dict = None):
        """Initialize loopable object.

        Args:
            loop_start_frame (int):
                Start frame of loopable motion.
            loop_end_frame (int):
                End frame of loopable motion.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Defaults to None.
        """
        super().__init__(logger_cfg)
        self.loop_start_frame = loop_start_frame
        self.loop_end_frame = loop_end_frame

    def to_dict(self) -> dict[str, Any]:
        """Convert Loopable object to dictionary.

        Returns:
            dict[str, Any]: Dictionary containing Loopable object attributes.
        """
        return {
            'loop_start_frame': self.loop_start_frame,
            'loop_end_frame': self.loop_end_frame
        }

    @classmethod
    def from_dict(cls, data: dict, logger_cfg: None | dict = None) -> 'Loopable':
        """Create Loopable object from dictionary.

        Args:
            data (dict): Dictionary containing Loopable object attributes.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Defaults to None.

        Returns:
            Loopable: Loopable object with attributes from dictionary.
        """
        return cls(
            loop_start_frame=data['loop_start_frame'],
            loop_end_frame=data['loop_end_frame'],
            logger_cfg=logger_cfg
        )

    def clone(self) -> 'Loopable':
        """Clone Loopable object.

        Returns:
            Loopable: Cloned Loopable object.
        """
        return Loopable(
            loop_start_frame=self.loop_start_frame,
            loop_end_frame=self.loop_end_frame
        )
