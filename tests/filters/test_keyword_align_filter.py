import logging
import os

import pytest

from speech2motion.data_structures.motion_record import MotionRecord
from speech2motion.filters.builder import build_filter
from speech2motion.filters.keyword_align_filter import KeywordAlignFilter
from speech2motion.io.meta.sqlite_meta_reader import SQLiteMetaReader
from speech2motion.utils.log import setup_logger

LOGGER_CFG = dict(
    logger_name='test_keyword_align_filter',
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
    """Test the construction of KeywordAlignFilter.

    This test verifies that the build_filter function can correctly
    create a KeywordAlignFilter instance from configuration.
    """
    cfg = dict(
        type='KeywordAlignFilter',
        logger_cfg=LOGGER_CFG
    )
    filter = build_filter(cfg)
    assert isinstance(filter, KeywordAlignFilter)

@pytest.mark.asyncio
async def test_filter():
    """Test batch filtering of motion records by keyword alignment.

    This test verifies the KeywordAlignFilter's ability to:
    1. Filter motion records based on keyword frame alignment
    2. Handle different frame offset scenarios (0, -10, +10)
    3. Support both motion and speech keyword attributes
    4. Handle cases where no matching records are found
    5. Handle empty input dictionaries

    The test loads motion records from SQLite database, collects
    both motion and speech keywords, and tests alignment filtering
    with various frame offset scenarios.
    """
    logger = setup_logger(**LOGGER_CFG)
    meta_reader = SQLiteMetaReader(
        sqlite_path=SQLITE_PATH,
        sqlite_join_cmd_path=SQL_JOIN_CMD_PATH,
        logger_cfg=LOGGER_CFG
    )
    ids = await meta_reader.get_ids()
    motion_records: dict[int, MotionRecord] = dict()
    motion_keywords = list()
    speech_keywords = list()
    for id in ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records[id] = motion_record
        if motion_record.is_motion_keyword():
            motion_keywords.append(motion_record.motion_record_id)
        if motion_record.is_speech_keyword():
            speech_keywords.append(motion_record.motion_record_id)
        if len(motion_keywords) > 0 and len(speech_keywords) > 0:
            break
    logger.info(
        f'Loaded {len(motion_records)} motion records, ' +
        f'collected {len(motion_keywords)} different motion keywords: '
        f'{motion_keywords}, collected {len(speech_keywords)} different '
        f'speech keywords: {speech_keywords}.'
    )
    # Test motion keyword alignment
    motion_keyword_filter = KeywordAlignFilter(
        logger_cfg=LOGGER_CFG
    )
    motion_keyword_id = next(iter(motion_keywords))
    motion_record = motion_records[motion_keyword_id]
    n_frames = motion_record.n_frames
    keyword_frame = motion_record.motion_keyword.motion_keyword_frame
    # Test alignment starting from frame 0
    filtered_motion_records = await motion_keyword_filter.filter(
        motion_records,
        align_frame=keyword_frame,
        start_frame_lowerbound=0,
        end_frame_upperbound=n_frames,
        keyword_attr_name='motion_keyword'
    )
    assert len(filtered_motion_records) > 0
    # Test alignment starting from frame -10
    filtered_motion_records = await motion_keyword_filter.filter(
        motion_records,
        align_frame=keyword_frame-10,
        start_frame_lowerbound=-10,
        end_frame_upperbound=n_frames-10,
        keyword_attr_name='motion_keyword'
    )
    assert len(filtered_motion_records) > 0
    # Test alignment starting from frame 10
    filtered_motion_records = await motion_keyword_filter.filter(
        motion_records,
        align_frame=keyword_frame+10,
        start_frame_lowerbound=10,
        end_frame_upperbound=n_frames+10,
        keyword_attr_name='motion_keyword'
    )
    assert len(filtered_motion_records) > 0
    one_motion_record_list = motion_keywords \
        if len(motion_keywords) == 1 \
        else speech_keywords
    attr_name = 'motion_keyword' \
        if len(motion_keywords) == 1 \
        else 'speech_keyword'
    motion_record_id = next(iter(one_motion_record_list))
    motion_record = motion_records[motion_record_id]
    n_frames = motion_record.n_frames
    keyword_attr = getattr(motion_record, attr_name)
    keyword_frame = getattr(keyword_attr, f'{attr_name}_frame')
    filtered_motion_records = await motion_keyword_filter.filter(
        motion_records,
        align_frame=keyword_frame,
        start_frame_lowerbound=0,
        end_frame_upperbound=n_frames,
        keyword_attr_name=attr_name
    )
    assert len(filtered_motion_records) > 0
    # Test failed alignment matching
    filtered_motion_records = await motion_keyword_filter.filter(
        motion_records,
        align_frame=keyword_frame,
        start_frame_lowerbound=0,
        end_frame_upperbound=n_frames - 1,
        keyword_attr_name=attr_name
    )
    assert len(filtered_motion_records) == 0
    filtered_motion_records = await motion_keyword_filter.filter(
        motion_records,
        align_frame=keyword_frame,
        start_frame_lowerbound=1,
        end_frame_upperbound=n_frames,
        keyword_attr_name=attr_name
    )
    assert len(filtered_motion_records) == 0
    # Test with empty dictionary input
    filtered_motion_records = await motion_keyword_filter.filter(
        dict(),
        align_frame=keyword_frame,
        start_frame_lowerbound=0,
        end_frame_upperbound=n_frames,
        keyword_attr_name=attr_name
    )
    assert len(filtered_motion_records) == 0

@pytest.mark.asyncio
async def test_filter_one():
    """Test single motion record selection by keyword alignment.

    This test verifies the KeywordAlignFilter's select_one method ability to:
    1. Select a single motion record based on keyword frame alignment
    2. Handle different frame offset scenarios (0, -10, +10)
    3. Support both motion and speech keyword attributes
    4. Return None when no matching records are found

    The test loads motion records from SQLite database, collects
    both motion and speech keywords, and tests single selection
    with various frame offset scenarios.
    """
    logger = setup_logger(**LOGGER_CFG)
    meta_reader = SQLiteMetaReader(
        sqlite_path=SQLITE_PATH,
        sqlite_join_cmd_path=SQL_JOIN_CMD_PATH,
        logger_cfg=LOGGER_CFG
    )
    ids = await meta_reader.get_ids()
    motion_records: dict[int, MotionRecord] = dict()
    motion_keywords = list()
    speech_keywords = list()
    for id in ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records[id] = motion_record
        if motion_record.is_motion_keyword():
            motion_keywords.append(motion_record.motion_record_id)
        if motion_record.is_speech_keyword():
            speech_keywords.append(motion_record.motion_record_id)
        if len(motion_keywords) > 0 and len(speech_keywords) > 0:
            break
    logger.info(
        f'Loaded {len(motion_records)} motion records, ' +
        f'collected {len(motion_keywords)} different motion keywords: '
        f'{motion_keywords}, collected {len(speech_keywords)} different '
        f'speech keywords: {speech_keywords}.'
    )
    # Test motion keyword selection
    motion_keyword_filter = KeywordAlignFilter(
        logger_cfg=LOGGER_CFG
    )
    motion_keyword_id = next(iter(motion_keywords))
    motion_record = motion_records[motion_keyword_id]
    n_frames = motion_record.n_frames
    keyword_frame = motion_record.motion_keyword.motion_keyword_frame
    # Test selection starting from frame 0
    filtered_motion_record = await motion_keyword_filter.select_one(
        motion_records,
        align_frame=keyword_frame,
        start_frame_lowerbound=0,
        end_frame_upperbound=n_frames,
        keyword_attr_name='motion_keyword'
    )
    assert isinstance(filtered_motion_record, MotionRecord)
    # Test selection starting from frame -10
    filtered_motion_record = await motion_keyword_filter.select_one(
        motion_records,
        align_frame=keyword_frame-10,
        start_frame_lowerbound=-10,
        end_frame_upperbound=n_frames-10,
        keyword_attr_name='motion_keyword'
    )
    assert isinstance(filtered_motion_record, MotionRecord)
    # Test selection starting from frame 10
    filtered_motion_record = await motion_keyword_filter.select_one(
        motion_records,
        align_frame=keyword_frame+10,
        start_frame_lowerbound=10,
        end_frame_upperbound=n_frames+10,
        keyword_attr_name='motion_keyword'
    )
    assert isinstance(filtered_motion_record, MotionRecord)
    one_motion_record_list = motion_keywords \
        if len(motion_keywords) == 1 \
        else speech_keywords
    attr_name = 'motion_keyword' \
        if len(motion_keywords) == 1 \
        else 'speech_keyword'
    motion_record_id = next(iter(one_motion_record_list))
    motion_record = motion_records[motion_record_id]
    n_frames = motion_record.n_frames
    keyword_attr = getattr(motion_record, attr_name)
    keyword_frame = getattr(keyword_attr, f'{attr_name}_frame')
    filtered_motion_record = await motion_keyword_filter.select_one(
        motion_records,
        align_frame=keyword_frame,
        start_frame_lowerbound=0,
        end_frame_upperbound=n_frames,
        keyword_attr_name=attr_name
    )
    assert isinstance(filtered_motion_record, MotionRecord)
    # Test failed alignment matching
    filtered_motion_record = await motion_keyword_filter.select_one(
        motion_records,
        align_frame=keyword_frame,
        start_frame_lowerbound=0,
        end_frame_upperbound=n_frames - 1,
        keyword_attr_name=attr_name
    )
    assert filtered_motion_record is None
    filtered_motion_record = await motion_keyword_filter.select_one(
        motion_records,
        align_frame=keyword_frame,
        start_frame_lowerbound=1,
        end_frame_upperbound=n_frames,
        keyword_attr_name=attr_name
    )
    assert filtered_motion_record is None
