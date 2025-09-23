from abc import ABC, abstractmethod

from ..utils.super import Super


class BaseTextSegmentation(ABC, Super):
    """Base class for text segmentation.

    This abstract base class provides the interface for text segmentation
    operations, including white list management and text cutting functionality.
    """

    def __init__(self,
                 init_white_list: list[str] | None = None,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the text segmentation base class.

        Args:
            init_white_list (list[str] | None, optional):
                Initial white list for text segmentation. If None, no custom
                white list will be added. Defaults to None.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        ABC.__init__(self)
        Super.__init__(self, logger_cfg)
        if init_white_list is not None:
            self._register_white_list(init_white_list)

    @abstractmethod
    def _register_white_list(self, white_list: list[str]) -> None:
        """Register a white list for text segmentation.

        Args:
            white_list (list[str]):
                List of words to be added to the white list for segmentation.
        """
        pass

    @abstractmethod
    def cut_text(self, text: str) -> list[dict]:
        """Perform text segmentation on a Chinese text string.

        Args:
            text (str):
                A Chinese text string that may contain punctuation marks.

        Returns:
            list[dict]:
                Segmentation results with relative order of words preserved.
                Each element in the list is a dictionary where 'text_idx'
                represents the character index where the word starts in the
                original text, and 'text_content' is the word content.
        """
        pass
