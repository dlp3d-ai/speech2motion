import logging
import os
import uuid

import pytest

from speech2motion.data_structures.motion_record import MotionRecord
from speech2motion.filters.avatar_filter import AvatarFilter
from speech2motion.filters.builder import build_filter
from speech2motion.index.dict_index_mapping import DictIndexMapping
from speech2motion.io.meta.sqlite_meta_reader import SQLiteMetaReader
from speech2motion.utils.log import setup_logger

LOGGER_CFG = dict(
    logger_name='test_avatar_filter',
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
    """Test the construction of AvatarFilter.

    This test verifies that the build_filter function can correctly
    create an AvatarFilter instance from configuration.
    """
    cfg = dict(
        type='AvatarFilter',
        mapping=DictIndexMapping(),
    )
    filter = build_filter(cfg)
    assert isinstance(filter, AvatarFilter)

@pytest.mark.asyncio
async def test_filter():
    """Test batch filtering of motion records by avatar name.

    This test verifies the AvatarFilter's ability to:
    1. Filter motion records by avatar name
    2. Handle non-existent avatar names (return empty results)
    3. Handle empty input dictionaries

    The test loads motion records from SQLite database, creates
    an avatar name mapping, and tests filtering with existing
    and non-existing avatar names.
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
    mapping = DictIndexMapping(
        logger_cfg=LOGGER_CFG
    )
    for id in ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records[id] = motion_record
        await mapping.add_item(motion_record.avatar_name, id)
        count += 1
    logger.info(f'Loaded {count} motion records.')
    avatar_names = await mapping.keys()
    avatar_name = next(iter(avatar_names))
    avatar_filter = AvatarFilter(
        mapping=mapping,
        logger_cfg=LOGGER_CFG
    )
    filtered_motion_records = await avatar_filter.filter(
        motion_records, avatar_name)
    assert len(filtered_motion_records) > 0
    # Generate a random string
    random_avatar_name = str(uuid.uuid4())
    filtered_motion_records = await avatar_filter.filter(
        motion_records, random_avatar_name)
    assert len(filtered_motion_records) == 0
    # Test with empty dictionary input
    filtered_motion_records = await avatar_filter.filter(
        dict(), avatar_name)
    assert len(filtered_motion_records) == 0


@pytest.mark.asyncio
async def test_filter_one():
    """Test single motion record selection by avatar name.

    This test verifies the AvatarFilter's select_one method ability to:
    1. Select a single motion record by avatar name
    2. Return None for non-existent avatar names
    3. Handle empty input dictionaries

    The test loads motion records from SQLite database, creates
    an avatar name mapping, and tests single selection with
    existing and non-existing avatar names.
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
    mapping = DictIndexMapping(
        logger_cfg=LOGGER_CFG
    )
    for id in ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records[id] = motion_record
        await mapping.add_item(motion_record.avatar_name, id)
        count += 1
    logger.info(f'Loaded {count} motion records.')
    avatar_names = await mapping.keys()
    avatar_name = next(iter(avatar_names))
    avatar_filter = AvatarFilter(
        mapping=mapping,
        logger_cfg=LOGGER_CFG
    )
    filtered_motion_record = await avatar_filter.select_one(
        motion_records, avatar_name)
    assert filtered_motion_record is not None
    assert isinstance(filtered_motion_record, MotionRecord)
    # Generate a random string
    random_avatar_name = str(uuid.uuid4())
    filtered_motion_record = await avatar_filter.select_one(
        motion_records, random_avatar_name)
    assert filtered_motion_record is None
    # Test with empty dictionary input
    filtered_motion_record = await avatar_filter.select_one(
        dict(), avatar_name)
    assert filtered_motion_record is None
