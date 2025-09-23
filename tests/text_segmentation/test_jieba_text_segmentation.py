import logging

from speech2motion.text_segmentation.jieba_text_segmentation import (
    JiebaTextSegmentation,
)
from speech2motion.utils.log import setup_logger

LOGGER_CFG = dict(
    logger_name='test_jieba_text_segmentation',
    file_level=logging.DEBUG,
    console_level=logging.INFO,
    logger_path='logs/pytest.log',
)

def test_cut_text():
    """Test text segmentation functionality.

    Tests the JiebaTextSegmentation cut_text method to ensure it correctly
    segments Chinese text and returns the expected format with word indices
    and content.
    """
    logger = setup_logger(**LOGGER_CFG)
    jieba_segmenter = JiebaTextSegmentation()
    text = "你好，世界！这是一个测试。"
    expected_output = [
        {'idx': 0, 'str': '你好'},
        {'idx': 3, 'str': '世界'},
        {'idx': 6, 'str': '这是'},
        {'idx': 8, 'str': '一个'},
        {'idx': 10, 'str': '测试'}
    ]

    result = jieba_segmenter.cut_text(text)
    logger.debug(f'Segmentation result: {result}')

    # Check segmentation results
    assert len(result) == len(expected_output)
    for res, exp in zip(result, expected_output, strict=False):
        assert res['str'] == exp['str']
        assert res['idx'] == exp['idx']

def test_register_white_list():
    """Test white list registration functionality.

    Tests the JiebaTextSegmentation white list feature to ensure that
    custom words can be registered and will be properly segmented
    as single units.
    """
    logger = setup_logger(**LOGGER_CFG)
    sentence = '这不是一个开源算法。'
    default_segmenter = JiebaTextSegmentation()
    default_result = default_segmenter.cut_text(sentence)
    logger.debug(f'Default segmentation result: {default_result}')
    word_found = False
    for res in default_result:
        if res['str'] == '开源算法':
            word_found = True
            break
    assert not word_found
    white_list_segmenter = JiebaTextSegmentation(
        init_white_list=["开源算法"]
    )
    white_list_result = white_list_segmenter.cut_text(sentence)
    logger.debug(f'Segmentation result after white list registration: '
                 f'{white_list_result}')
    word_found = False
    for res in white_list_result:
        if res['str'] == '开源算法':
            word_found = True
            break
    assert word_found
