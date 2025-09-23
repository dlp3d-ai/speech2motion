import logging

import jieba

from ..utils.log import setup_logger
from .base_text_segmentation import BaseTextSegmentation

# Reduce jieba built-in logger output
setup_logger(
    logger_name='jieba',
    file_level=logging.INFO,
    console_level=logging.INFO
)

class JiebaTextSegmentation(BaseTextSegmentation):
    """Text segmentation implementation based on jieba.

    This class provides text segmentation functionality using the jieba library,
    with support for custom split marks and white list management.
    """
    _DEFAULT_SPLIT_CHAR = (',', '.', '，', '。', '!', '！', '?', '？')

    def __init__(self,
                 split_marks: list[str] | None = None,
                 init_white_list: list[str] | None = None,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the jieba text segmentation instance.

        Args:
            split_marks (list[str] | None, optional):
                List of punctuation marks used for text splitting. If None,
                uses default split marks. Defaults to None.
            init_white_list (list[str] | None, optional):
                Initial white list for text segmentation. If None, no custom
                white list will be added. Defaults to None.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        BaseTextSegmentation.__init__(
            self,
            init_white_list=init_white_list,
            logger_cfg=logger_cfg
        )
        if split_marks is not None:
            self.split_marks = split_marks
        else:
            self.split_marks = self._DEFAULT_SPLIT_CHAR

    def _register_white_list(self, white_list: list[str]) -> None:
        """Register a white list for text segmentation.

        Args:
            white_list (list[str]):
                List of words to be added to the white list for segmentation.

        Raises:
            TypeError:
                Raised when white_list is not a list.
        """
        if not isinstance(white_list, list):
            msg = f'Provided white_list is not a list: {type(white_list)}'
            self.logger.error(msg)
            raise TypeError(msg)
        for word in white_list:
            jieba.add_word(word)
        self.logger.debug(f'Added words from white list to dictionary: {white_list}')


    def cut_text(self, text: str) -> list[dict]:
        """Perform text segmentation on a Chinese text string.

        First splits the text according to predefined punctuation marks in
        self.split_marks, then performs jieba segmentation on each split segment.

        Args:
            text (str):
                A Chinese text string that may contain punctuation marks.

        Returns:
            list[dict]:
                Segmentation results with relative order of words preserved.
                Each element in the list is a dictionary where 'idx' represents
                the character index where the word starts in the original text,
                and 'str' is the word content.
        """
        ret_list = []
        start_idx = 0
        last_char = text[-1]
        # If the last character is not a punctuation mark,
        # add a punctuation mark at the end
        if last_char not in self.split_marks:
            text += self.split_marks[0]
        for i in range(1, len(text)):
            char = text[i]
            # First split by punctuation marks
            if char in self.split_marks:
                valid_words = text[start_idx:i]
                word_list = jieba.cut(valid_words)
                relative_idx = 0
                for word in word_list:
                    words_dict = dict(idx=start_idx + relative_idx, str=word)
                    ret_list.append(words_dict)
                    relative_idx += len(word)
                start_idx = i + 1
        return ret_list
