from typing import Any

from ...utils.super import Super


class SpeechKeyword(Super):
    """Object for storing data from a row in SpeechKeyword table.
    """
    def __init__(
            self,
            speech_keywords_ch: list[str],
            speech_keyword_frame: int,
            logger_cfg: None | dict = None):
        """Initialize speech keyword object.

        Args:
            speech_keywords_ch (list[str]):
                List of keywords for matchable speech.
            speech_keyword_frame (int):
                Frame index used for speech keyword alignment.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name.
                Defaults to None.
        """
        Super.__init__(self, logger_cfg)
        self.speech_keywords_ch = speech_keywords_ch
        if len(speech_keywords_ch) == 0:
            msg = 'speech_keywords_ch is an empty list.'
            self.logger.error(msg)
            raise ValueError(msg)
        self.speech_keyword_frame = speech_keyword_frame

    def to_dict(self) -> dict[str, Any]:
        """Convert SpeechKeyword object to dictionary.

        Returns:
            dict: Dictionary containing SpeechKeyword object attributes.
        """
        ret_dict = dict(
            speech_keywords_ch=self.speech_keywords_ch,
            speech_keyword_frame=self.speech_keyword_frame
        )
        return ret_dict

    @classmethod
    def from_dict(
            cls,
            data: dict[str, Any],
            logger_cfg: None | dict = None) -> 'SpeechKeyword':
        """Create SpeechKeyword object from dictionary.

        Args:
            data (dict[str, Any]):
                Dictionary containing SpeechKeyword object attributes.
            logger_cfg (None | dict, optional): Logger configuration.
                Defaults to None.

        Returns:
            SpeechKeyword: SpeechKeyword object.
        """
        return cls(
            speech_keywords_ch=data['speech_keywords_ch'],
            speech_keyword_frame=data['speech_keyword_frame'],
            logger_cfg=logger_cfg)

    def clone(self) -> 'SpeechKeyword':
        """Clone SpeechKeyword object.

        Returns:
            SpeechKeyword: Cloned SpeechKeyword object.
        """
        return SpeechKeyword(
            speech_keywords_ch=self.speech_keywords_ch,
            speech_keyword_frame=self.speech_keyword_frame)
