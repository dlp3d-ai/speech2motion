import logging
import os
from typing import TYPE_CHECKING

import pytest

from speech2motion.filters.builder import build_filter
from speech2motion.filters.random_filter import RandomFilter
from speech2motion.io.meta.sqlite_meta_reader import SQLiteMetaReader

if TYPE_CHECKING:
    from speech2motion.data_structures.motion_record import MotionRecord

LOGGER_CFG = dict(
    logger_name='test_random_filter',
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
    """Test the construction of RandomFilter.

    This test verifies that the build_filter function can correctly
    create a RandomFilter instance from configuration.
    """
    cfg = dict(
        type='RandomFilter',
    )
    filter = build_filter(cfg)
    assert isinstance(filter, RandomFilter)

@pytest.mark.asyncio
async def test_filter_random():
    """Test random filtering of motion records.

    This test verifies the RandomFilter's ability to:
    1. Return all motion records when no return_length is specified
    2. Return a specified number of randomly selected records
    3. Handle cases where return_length exceeds available records
    4. Handle empty input dictionaries

    The test loads motion records from SQLite database and tests
    various filtering scenarios with different return_length parameters.
    """
    meta_reader = SQLiteMetaReader(
        sqlite_path=SQLITE_PATH,
        sqlite_join_cmd_path=SQL_JOIN_CMD_PATH,
        logger_cfg=LOGGER_CFG
    )
    ids = await meta_reader.get_ids()
    motion_records: dict[int, MotionRecord] = dict()

    for id in ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records[id] = motion_record
    random_filter = RandomFilter(logger_cfg=LOGGER_CFG)

    # Test returning all records
    filtered_motion_records = await random_filter.filter(motion_records)
    assert len(filtered_motion_records) == len(motion_records)

    # Test returning specified number of records
    return_length = 2
    filtered_motion_records = await random_filter.filter(
        motion_records, return_length=return_length)
    assert len(filtered_motion_records) == return_length

    # Test returning more records than available
    filtered_motion_records = await random_filter.filter(
        motion_records, return_length=len(motion_records) + 1)
    assert len(filtered_motion_records) == len(motion_records)

    # Test with empty dictionary input
    filtered_motion_records = await random_filter.filter({}, return_length=1)
    assert len(filtered_motion_records) == 0


@pytest.mark.asyncio
async def test_select_one_random():
    """Test random selection of one motion record.

    This test verifies the RandomFilter's select_one method ability to:
    1. Randomly select a single motion record from available records
    2. Return None when input dictionary is empty

    The test loads motion records from SQLite database and tests
    random selection functionality.
    """
    meta_reader = SQLiteMetaReader(
        sqlite_path=SQLITE_PATH,
        sqlite_join_cmd_path=SQL_JOIN_CMD_PATH,
        logger_cfg=LOGGER_CFG
    )
    ids = await meta_reader.get_ids()
    motion_records: dict[int, MotionRecord] = dict()

    for id in ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records[id] = motion_record
    random_filter = RandomFilter(logger_cfg=LOGGER_CFG)

    # Test selecting one record
    selected_motion_record = await random_filter.select_one(motion_records)
    assert selected_motion_record is not None

    # Test with empty dictionary input
    selected_motion_record = await random_filter.select_one({})
    assert selected_motion_record is None

