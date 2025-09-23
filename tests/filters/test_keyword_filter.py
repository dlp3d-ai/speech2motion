import logging
import os
import uuid

import pytest

from speech2motion.data_structures.motion_record import MotionRecord
from speech2motion.filters.builder import build_filter
from speech2motion.filters.keyword_filter import KeywordFilter
from speech2motion.index.dict_index_mapping import DictIndexMapping
from speech2motion.io.meta.sqlite_meta_reader import SQLiteMetaReader
from speech2motion.utils.log import setup_logger

LOGGER_CFG = dict(
    logger_name='test_keyword_filter',
    file_level=logging.DEBUG,
    console_level=logging.INFO,
    logger_path='logs/pytest.log',
)
SQLITE_PATH = 'data/motion_database.db'
SQL_JOIN_CMD_PATH = 'configs/3dac_sql_join.sql'

os.makedirs('logs', exist_ok=True)


def is_sqlite_available() -> bool:
    """Check if SQLite database file is available for testing.

    Returns:
        bool: True if SQLite database file exists, False otherwise.
    """
    if os.path.exists(SQLITE_PATH):
        return True
    return False

# Skip all tests if SQLite is not available
pytestmark = pytest.mark.skipif(
    not is_sqlite_available(),
    reason="SQLite server is not available for testing"
)


def test_build():
    """Test the construction of KeywordFilter.

    This test verifies that the build_filter function can correctly
    create a KeywordFilter instance from configuration.
    """
    cfg = dict(
        type='KeywordFilter',
        mapping=DictIndexMapping(),
        name='keyword_filter',
    )
    filter = build_filter(cfg)
    assert isinstance(filter, KeywordFilter)

@pytest.mark.asyncio
async def test_filter():
    """Test batch filtering of motion records by keywords.

    This test verifies the KeywordFilter's ability to:
    1. Filter motion records by motion keywords
    2. Filter motion records by speech keywords
    3. Handle non-existent keywords (return empty results)
    4. Handle empty input dictionaries

    The test loads motion records from SQLite database, collects
    both motion and speech keywords, and tests filtering with
    existing and non-existing keywords.
    """
    logger = setup_logger(**LOGGER_CFG)
    meta_reader = SQLiteMetaReader(
        sqlite_path=SQLITE_PATH,
        sqlite_join_cmd_path=SQL_JOIN_CMD_PATH,
        logger_cfg=LOGGER_CFG
    )
    ids = await meta_reader.get_ids()
    motion_records: dict[int, MotionRecord] = dict()
    motion_keyword_mapping = DictIndexMapping(
        logger_cfg=LOGGER_CFG
    )
    motion_keyword_collected = False
    speech_keyword_mapping = DictIndexMapping(
        logger_cfg=LOGGER_CFG
    )
    speech_keyword_collected = False
    for id in ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records[id] = motion_record
        if motion_record.is_motion_keyword():
            keywords = motion_record.motion_keyword.motion_keywords_ch
            motion_keyword_collected = True
            for keyword in keywords:
                await motion_keyword_mapping.add_item(keyword, id)
        if motion_record.is_speech_keyword():
            keywords = motion_record.speech_keyword.speech_keywords_ch
            speech_keyword_collected = True
            for keyword in keywords:
                await speech_keyword_mapping.add_item(keyword, id)
        if motion_keyword_collected and speech_keyword_collected:
            break
    motion_keywords = await motion_keyword_mapping.keys()
    speech_keywords = await speech_keyword_mapping.keys()
    logger.info(
        f'Loaded {len(motion_records)} motion records, ' +
        f'collected {len(motion_keywords)} different motion keywords: '
        f'{motion_keywords}, collected {len(speech_keywords)} different '
        f'speech keywords: {speech_keywords}.'
    )
    # Test motion keyword filtering
    motion_keyword_filter = KeywordFilter(
        name='motion_keyword_filter',
        mapping=motion_keyword_mapping,
        logger_cfg=LOGGER_CFG
    )
    motion_keyword = next(iter(motion_keywords))
    filtered_motion_records = await motion_keyword_filter.filter(
        motion_records, motion_keyword)
    assert len(filtered_motion_records) > 0
    # Generate a random string
    random_keyword = str(uuid.uuid4())
    filtered_motion_records = await motion_keyword_filter.filter(
        motion_records, random_keyword)
    assert len(filtered_motion_records) == 0
    # Test speech keyword filtering
    speech_keyword = next(iter(speech_keywords))
    speech_keyword_filter = KeywordFilter(
        name='speech_keyword_filter',
        mapping=speech_keyword_mapping,
        logger_cfg=LOGGER_CFG
    )
    filtered_motion_records = await speech_keyword_filter.filter(
        motion_records, speech_keyword)
    assert len(filtered_motion_records) > 0
    filtered_motion_records = await speech_keyword_filter.filter(
        motion_records, random_keyword)
    assert len(filtered_motion_records) == 0
    # Test with empty dictionary input
    filtered_motion_records = await speech_keyword_filter.filter(
        dict(), speech_keyword)
    assert len(filtered_motion_records) == 0


@pytest.mark.asyncio
async def test_filter_one():
    """Test single motion record selection by keywords.

    This test verifies the KeywordFilter's select_one method ability to:
    1. Select a single motion record by motion keyword
    2. Select a single motion record by speech keyword
    3. Return None for non-existent keywords
    4. Handle empty input dictionaries

    The test loads motion records from SQLite database, collects
    both motion and speech keywords, and tests single selection
    with existing and non-existing keywords.
    """
    logger = setup_logger(**LOGGER_CFG)
    meta_reader = SQLiteMetaReader(
        sqlite_path=SQLITE_PATH,
        sqlite_join_cmd_path=SQL_JOIN_CMD_PATH,
        logger_cfg=LOGGER_CFG
    )
    ids = await meta_reader.get_ids()
    motion_records: dict[int, MotionRecord] = dict()
    motion_keyword_mapping = DictIndexMapping(
        logger_cfg=LOGGER_CFG
    )
    motion_keyword_collected = False
    speech_keyword_mapping = DictIndexMapping(
        logger_cfg=LOGGER_CFG
    )
    speech_keyword_collected = False
    for id in ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records[id] = motion_record
        if motion_record.is_motion_keyword():
            keywords = motion_record.motion_keyword.motion_keywords_ch
            motion_keyword_collected = True
            for keyword in keywords:
                await motion_keyword_mapping.add_item(keyword, id)
        if motion_record.is_speech_keyword():
            keywords = motion_record.speech_keyword.speech_keywords_ch
            speech_keyword_collected = True
            for keyword in keywords:
                await speech_keyword_mapping.add_item(keyword, id)
        if motion_keyword_collected and speech_keyword_collected:
            break
    motion_keywords = await motion_keyword_mapping.keys()
    speech_keywords = await speech_keyword_mapping.keys()
    logger.info(
        f'Loaded {len(motion_records)} motion records, ' +
        f'collected {len(motion_keywords)} different motion keywords: '
        f'{motion_keywords}, collected {len(speech_keywords)} different '
        f'speech keywords: {speech_keywords}.'
    )
    motion_keyword = next(iter(motion_keywords))
    # Test motion keyword selection
    motion_keyword_filter = KeywordFilter(
        name='motion_keyword_filter',
        mapping=motion_keyword_mapping,
        logger_cfg=LOGGER_CFG
    )
    filtered_motion_record = await motion_keyword_filter.select_one(
        motion_records, motion_keyword)
    assert filtered_motion_record is not None
    assert isinstance(filtered_motion_record, MotionRecord)
    # Generate a random string
    random_motion_keyword = str(uuid.uuid4())
    filtered_motion_record = await motion_keyword_filter.select_one(
        motion_records, random_motion_keyword)
    assert filtered_motion_record is None
    # Test speech keyword selection
    speech_keyword = next(iter(speech_keywords))
    speech_keyword_filter = KeywordFilter(
        name='speech_keyword_filter',
        mapping=speech_keyword_mapping,
        logger_cfg=LOGGER_CFG
    )
    filtered_motion_record = await speech_keyword_filter.select_one(
        motion_records, speech_keyword)
    assert filtered_motion_record is not None
    assert isinstance(filtered_motion_record, MotionRecord)
    random_speech_keyword = str(uuid.uuid4())
    filtered_motion_record = await speech_keyword_filter.select_one(
        motion_records, random_speech_keyword)
    assert filtered_motion_record is None
    # Test with empty dictionary input
    filtered_motion_record = await speech_keyword_filter.select_one(
        dict(), speech_keyword)
    assert filtered_motion_record is None
