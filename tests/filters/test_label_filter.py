import logging
import os

import pytest

from speech2motion.filters.builder import build_filter
from speech2motion.filters.label_filter import (
    EmptyExpressionError,
    InvalidCharacterError,
    InvalidTokenError,
    LabelFilter,
    LabelNotFoundError,
    MismatchedParenthesesError,
)
from speech2motion.index.dict_index_mapping import DictIndexMapping

LOGGER_CFG = dict(
    logger_name='test_label_filter',
    file_level=logging.DEBUG,
    console_level=logging.INFO,
    logger_path='logs/pytest.log',
)

os.makedirs('logs', exist_ok=True)

def test_build():
    """Test filter construction.

    Tests the build_filter function to ensure it correctly creates
    a LabelFilter instance from configuration.
    """
    cfg = dict(
        type='LabelFilter',
        mapping=DictIndexMapping(),
        name='label_filter',
    )
    filter = build_filter(cfg)
    assert isinstance(filter, LabelFilter)

@pytest.mark.asyncio
async def test_filter():
    """Test batch filtering functionality.

    Tests the LabelFilter with various expressions including:
    - Simple label matching
    - Negation operations
    - Logical AND/OR operations
    - Parentheses precedence
    - Error handling for invalid expressions
    """
    label_id_mapping = DictIndexMapping(logger_cfg=LOGGER_CFG)
    await label_id_mapping.add_item('label1', 1)
    await label_id_mapping.add_item('label2', 2)
    await label_id_mapping.add_item('label3', 3)
    await label_id_mapping.add_item('label1_label2', 1)
    await label_id_mapping.add_item('label1_label2', 2)
    filter = LabelFilter(
        name='label_filter',
        mapping=label_id_mapping,
        logger_cfg=LOGGER_CFG
    )
    motion_records = {
        1: 'motion_record_1',
        2: 'motion_record_2',
        3: 'motion_record_3',
    }
    expression = 'label1'
    result = await filter.filter(motion_records, expression)
    assert 1 in result and len(result) == 1
    expression = '! label1'
    result = await filter.filter(motion_records, expression)
    assert 2 in result and 3 in result and len(result) == 2
    expression = 'label1 & label2'
    result = await filter.filter(motion_records, expression)
    assert len(result) == 0
    expression = 'label1 | label2'
    result = await filter.filter(motion_records, expression)
    assert 1 in result and 2 in result and len(result) == 2
    expression = 'label1 & label2 | label3'
    result = await filter.filter(motion_records, expression)
    assert 3 in result and len(result) == 1
    expression = 'label1_label2 & label1'
    result = await filter.filter(motion_records, expression)
    assert 1 in result and len(result) == 1
    # Test parentheses precedence
    expression = '! (label1 & label2)'
    result = await filter.filter(motion_records, expression)
    assert len(result) == 3
    expression = 'label1 & (label1 | label2)'
    result = await filter.filter(motion_records, expression)
    assert 1 in result and len(result) == 1
    # Test exceptions
    with pytest.raises(LabelNotFoundError):
        expression = 'label100'
        result = await filter.filter(motion_records, expression)
    with pytest.raises(InvalidCharacterError):
        expression = '/ label1'
        result = await filter.filter(motion_records, expression)
    with pytest.raises(MismatchedParenthesesError):
        expression = 'label1 & (label1 | label2'
        result = await filter.filter(motion_records, expression)
    with pytest.raises(MismatchedParenthesesError):
        expression = 'label1 & label1 | label2)'
        result = await filter.filter(motion_records, expression)
    with pytest.raises(EmptyExpressionError):
        expression = ''
        result = await filter.filter(motion_records, expression)
    with pytest.raises(InvalidTokenError):
        expression = 'label1 & label2 &'
        result = await filter.filter(motion_records, expression)
