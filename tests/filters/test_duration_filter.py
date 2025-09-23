import logging
import os
from typing import TYPE_CHECKING

import pytest

from speech2motion.filters.builder import build_filter
from speech2motion.filters.duration_filter import DurationFilter
from speech2motion.io.meta.sqlite_meta_reader import SQLiteMetaReader
from speech2motion.utils.log import setup_logger

if TYPE_CHECKING:
    from speech2motion.data_structures.motion_record import MotionRecord


LOGGER_CFG = dict(
    logger_name='test_duration_filter',
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
    """Test the construction of DurationFilter.

    This test verifies that the build_filter function can correctly
    create a DurationFilter instance from configuration.
    """
    cfg = dict(
        type='DurationFilter',
    )
    filter = build_filter(cfg)
    assert isinstance(filter, DurationFilter)

@pytest.mark.asyncio
async def test_filter_duration():
    """Test filtering motion records by duration.

    This test verifies the DurationFilter's ability to:
    1. Filter motion records by duration range (lowerbound, upperbound)
    2. Handle various duration range scenarios
    3. Support single duration filtering (exact match)
    4. Handle edge cases with no matching records
    5. Handle empty input dictionaries

    The test loads motion records from SQLite database, calculates
    durations from frame count and FPS, and tests filtering with
    various duration range combinations.
    """
    logger = setup_logger(**LOGGER_CFG)
    meta_reader = SQLiteMetaReader(
        sqlite_path=SQLITE_PATH,
        sqlite_join_cmd_path=SQL_JOIN_CMD_PATH,
        logger_cfg=LOGGER_CFG
    )
    ids = await meta_reader.get_ids()
    motion_records: dict[int, MotionRecord] = dict()
    count = 0
    first_duration = None
    max_duration = 0
    min_duration = float('inf')
    duration_set = set()
    for id in ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records[id] = motion_record
        duration = motion_record.n_frames / motion_record.fps
        if duration in duration_set:
            continue
        duration_set.add(duration)
        if first_duration is None:
            first_duration = duration
        max_duration = max(max_duration, duration)
        min_duration = min(min_duration, duration)
        count += 1
    logger.info(
        f'duration_set={duration_set}, ' +
        f'min_duration={min_duration}, ' +
        f'max_duration={max_duration}'
    )
    duration_filter = DurationFilter(
        logger_cfg=LOGGER_CFG
    )
    # Test range [first_duration, +∞)
    filtered_motion_records = await duration_filter.filter(
        motion_records, duration_lowerbound=first_duration
    )
    assert len(filtered_motion_records) > 0
    # Test range [first_duration, first_duration + 0.1]
    filtered_motion_records = await duration_filter.filter(
        motion_records, duration_lowerbound=first_duration,
        duration_upperbound=first_duration + 0.1
    )
    assert len(filtered_motion_records) > 0
    # Test range [min_duration, max_duration]
    filtered_motion_records = await duration_filter.filter(
        motion_records, duration_lowerbound=min_duration,
        duration_upperbound=max_duration
    )
    assert len(filtered_motion_records) == len(motion_records)
    # Test range [max_duration+0.1, +∞)
    filtered_motion_records = await duration_filter.filter(
        motion_records, duration_lowerbound=max_duration + 0.1
    )
    assert len(filtered_motion_records) == 0
    # Test range [0, min_duration]
    filtered_motion_records = await duration_filter.filter(
        motion_records, duration_lowerbound=0,
        duration_upperbound=min_duration
    )
    assert len(filtered_motion_records) == 1
    # Test range [first_duration, first_duration]
    filtered_motion_records = await duration_filter.filter(
        motion_records, duration_lowerbound=first_duration,
        duration_upperbound=first_duration
    )
    assert len(filtered_motion_records) == 1
    # Test with empty dictionary input
    filtered_motion_records = await duration_filter.filter(
        dict(), duration_lowerbound=first_duration,
        duration_upperbound=first_duration
    )
    assert len(filtered_motion_records) == 0

@pytest.mark.asyncio
async def test_select_one_duration():
    """Test selecting one motion record by duration.

    This test verifies the DurationFilter's select_one method ability to:
    1. Select a single motion record within duration range
    2. Handle various duration range scenarios
    3. Return None when no matching records are found
    4. Handle empty input dictionaries

    The test loads motion records from SQLite database, calculates
    durations from frame count and FPS, and tests single selection
    with various duration range combinations.
    """
    logger = setup_logger(**LOGGER_CFG)
    meta_reader = SQLiteMetaReader(
        sqlite_path=SQLITE_PATH,
        sqlite_join_cmd_path=SQL_JOIN_CMD_PATH,
        logger_cfg=LOGGER_CFG
    )
    ids = await meta_reader.get_ids()
    motion_records: dict[int, MotionRecord] = dict()
    count = 0
    first_duration = None
    max_duration = 0
    min_duration = float('inf')
    duration_set = set()
    for id in ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records[id] = motion_record
        duration = motion_record.n_frames / motion_record.fps
        if duration in duration_set:
            continue
        duration_set.add(duration)
        if first_duration is None:
            first_duration = duration
        max_duration = max(max_duration, duration)
        min_duration = min(min_duration, duration)
        count += 1
    logger.info(
        f'duration_set={duration_set}, ' +
        f'min_duration={min_duration}, ' +
        f'max_duration={max_duration}'
    )
    duration_filter = DurationFilter(
        logger_cfg=LOGGER_CFG
    )
    # Test range [first_duration, +∞)
    selected_motion_record = await duration_filter.select_one(
        motion_records, duration_lowerbound=first_duration
    )
    assert selected_motion_record is not None
    # Test range [first_duration, first_duration + 0.1]
    selected_motion_record = await duration_filter.select_one(
        motion_records, duration_lowerbound=first_duration,
        duration_upperbound=first_duration + 0.1
    )
    assert selected_motion_record is not None
    # Test range [min_duration, max_duration]
    selected_motion_record = await duration_filter.select_one(
        motion_records, duration_lowerbound=min_duration,
        duration_upperbound=max_duration
    )
    assert selected_motion_record is not None
    # Test range [max_duration+0.1, +∞)
    selected_motion_record = await duration_filter.select_one(
        motion_records, duration_lowerbound=max_duration + 0.1
    )
    assert selected_motion_record is None
    # Test range [0, min_duration]
    selected_motion_record = await duration_filter.select_one(
        motion_records, duration_lowerbound=0,
        duration_upperbound=min_duration
    )
    assert selected_motion_record is not None
    # Test range [0, min_duration - 0.1]
    selected_motion_record = await duration_filter.select_one(
        motion_records, duration_lowerbound=0,
        duration_upperbound=min_duration - 0.1
    )
    assert selected_motion_record is None
    # Test with empty dictionary input
    selected_motion_record = await duration_filter.select_one(
        dict(), duration_lowerbound=first_duration,
        duration_upperbound=first_duration
    )
    assert selected_motion_record is None
