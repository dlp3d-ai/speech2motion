import logging
import os
from typing import TYPE_CHECKING

import pytest

from speech2motion.filters.builder import build_filter
from speech2motion.filters.memory_filter import MemoryFilter
from speech2motion.io.meta.sqlite_meta_reader import SQLiteMetaReader
from speech2motion.utils.log import setup_logger
from speech2motion.variety.local_memory import LocalMemory

if TYPE_CHECKING:
    from speech2motion.data_structures.motion_record import MotionRecord

LOGGER_CFG = dict(
    logger_name='test_memory_filter',
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
    """Test the construction of MemoryFilter.

    This test verifies that the build_filter function can correctly
    create a MemoryFilter instance from configuration with LocalMemory.
    """
    cfg = dict(
        type='MemoryFilter',
        name='memory_filter',
        memory=LocalMemory(
            memory_duration=60*60*24,
            logger_cfg=LOGGER_CFG,
        ),
        logger_cfg=LOGGER_CFG,
    )
    filter = build_filter(cfg)
    assert isinstance(filter, MemoryFilter)

@pytest.mark.asyncio
async def test_filter():
    """Test filtering motion records based on user memory.

    This test verifies the MemoryFilter's ability to:
    1. Filter out motion records that users have already seen
    2. Handle different memory scenarios for different users
    3. Return appropriate filtered results based on memory state

    The test creates three user scenarios:
    - User 1: Remembers all motion records (gradually filters them out)
    - User 2: Only remembers one motion record (with short memory duration)
    - User 3: Remembers no motion records (returns all records)
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

    for id in ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records[id] = motion_record
        count += 1
    logger.info(f'Loaded {count} motion records.')
    memory = LocalMemory(
        memory_duration=60*60*24,
        logger_cfg=LOGGER_CFG,
    )
    memory_filter = MemoryFilter(
        name='memory_filter',
        memory=memory,
        logger_cfg=LOGGER_CFG,
    )
    # User 1 remembers all motion records
    user_id = 'user_1'
    for k, v in motion_records.items():
        await memory.remember(user_id, k, event_duration=v.n_frames / v.fps)
        filter_results = await memory_filter.filter(motion_records, user_id)
        assert len(filter_results) == count - 1
        count -= 1
    # User 2 only remembers one motion record
    user_id = 'user_2'
    for k, v in motion_records.items():
        await memory.remember(
            user_id,
            k,
            event_duration=v.n_frames / v.fps,
            memory_duration_override=0.1,
            )
        filter_results = await memory_filter.filter(motion_records, user_id)
        assert len(filter_results) == len(motion_records) - 1
    # User 3 remembers no motion records
    user_id = 'user_3'
    filter_results = await memory_filter.filter(motion_records, user_id)
    assert len(filter_results) == len(motion_records)

@pytest.mark.asyncio
async def test_select_one():
    """Test selecting one motion record based on user memory.

    This test verifies the MemoryFilter's select_one method ability to:
    1. Select motion records that users have not seen before
    2. Return None when all records have been seen
    3. Handle different memory scenarios for different users

    The test creates two user scenarios:
    - User 1: Gradually remembers all motion records, ensuring selection
      of unseen records until all are remembered
    - User 2: Only remembers one motion record, ensuring selection
      of different records
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

    for id in ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records[id] = motion_record
        count += 1
    logger.info(f'Loaded {count} motion records.')
    memory = LocalMemory(
        memory_duration=60*60*24,
        logger_cfg=LOGGER_CFG,
    )
    memory_filter = MemoryFilter(
        name='memory_filter',
        memory=memory,
        logger_cfg=LOGGER_CFG,
    )
    # User 1 remembers all motion records
    user_id = 'user_1'
    seen_set = set()
    for k, v in motion_records.items():
        await memory.remember(user_id, k, event_duration=v.n_frames / v.fps)
        seen_set.add(k)
        ret_motion_record = await memory_filter.select_one(motion_records, user_id)
        if len(seen_set) < count:
            assert ret_motion_record is not None
            assert ret_motion_record.motion_record_id not in seen_set
        else:
            assert ret_motion_record is None
    # User 2 only remembers one motion record
    user_id = 'user_2'
    for k, v in motion_records.items():
        await memory.remember(
            user_id,
            k,
            event_duration=v.n_frames / v.fps,
            memory_duration_override=0.1,
            )
        ret_motion_record = await memory_filter.select_one(motion_records, user_id)
        assert ret_motion_record is not None
        assert ret_motion_record.motion_record_id != k
