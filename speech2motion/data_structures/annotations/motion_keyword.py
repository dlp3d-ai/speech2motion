from typing import Any

from ...utils.super import Super


class MotionKeyword(Super):
    """Object for storing data from a row in MotionKeyword table.
    """
    def __init__(
            self,
            motion_keywords_ch: list[str],
            motion_keyword_frame: int,
            logger_cfg: None | dict = None):
        """Initialize motion keyword object.

        Args:
            motion_keywords_ch (list[str]):
                List of keywords for matchable motions.
            motion_keyword_frame (int):
                Frame index used for motion keyword alignment.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name.
                Defaults to None.
        """
        Super.__init__(self, logger_cfg)
        self.motion_keywords_ch = motion_keywords_ch
        if len(motion_keywords_ch) == 0:
            msg = 'motion_keywords_ch is an empty list.'
            self.logger.error(msg)
            raise ValueError(msg)
        self.motion_keyword_frame = motion_keyword_frame

    def to_dict(self) -> dict[str, Any]:
        """Convert MotionKeyword object to dictionary.

        Returns:
            dict: Dictionary containing MotionKeyword object attributes.
        """
        ret_dict = dict(
            motion_keywords_ch=self.motion_keywords_ch,
            motion_keyword_frame=self.motion_keyword_frame
        )
        return ret_dict

    @classmethod
    def from_dict(
            cls,
            data: dict[str, Any],
            logger_cfg: None | dict = None) -> 'MotionKeyword':
        """Create MotionKeyword object from dictionary.

        Args:
            data (dict[str, Any]):
                Dictionary containing MotionKeyword object attributes.
            logger_cfg (None | dict, optional): Logger configuration.
                Defaults to None.

        Returns:
            MotionKeyword: MotionKeyword object.
        """
        return cls(
            motion_keywords_ch=data['motion_keywords_ch'],
            motion_keyword_frame=data['motion_keyword_frame'],
            logger_cfg=logger_cfg)

    def clone(self) -> 'MotionKeyword':
        """Clone MotionKeyword object.

        Returns:
            MotionKeyword: Cloned MotionKeyword object.
        """
        return MotionKeyword(
            motion_keywords_ch=self.motion_keywords_ch,
            motion_keyword_frame=self.motion_keyword_frame)
