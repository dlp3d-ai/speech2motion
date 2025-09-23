import asyncio
import logging
import os

import pytest

from speech2motion.data_structures.timeline import Timeline
from speech2motion.io.meta.builder import build_meta_reader
from speech2motion.utils.log import setup_logger

LOGGER_CFG = dict(
    logger_name='test_timeline',
    file_level=logging.DEBUG,
    console_level=logging.INFO,
    logger_path='logs/pytest.log',
)
SQLITE_PATH = 'data/motion_database.db'
SQL_JOIN_CMD_PATH = 'configs/3dac_sql_join.sql'


os.makedirs('logs', exist_ok=True)

def is_sqlite_available() -> bool:
    """Check if SQLite server is available for testing.

    Returns:
        bool: True if SQLite is available, False otherwise.
    """
    if os.path.exists(SQLITE_PATH):
        return True
    return False


# Skip all tests if SQLite is not available
pytestmark = pytest.mark.skipif(
    not is_sqlite_available(),
    reason="SQLite server is not available for testing"
)

@pytest.mark.asyncio
async def test_sync():
    """Test Timeline synchronization and various operations.

    This test verifies the Timeline's ability to:
    1. Insert motion records at different frame positions
    2. Handle timelines that don't start from frame 0
    3. Support timeline extension when enable_extension=True
    4. Find next blank intervals in the timeline
    5. Handle trigger events for motion records
    6. Support preload tasks for motion records
    7. Convert timeline to table format
    8. Extend timeline from zero length

    The test loads motion records from SQLite database and tests
    various timeline operations including insertion, extension,
    blank detection, and formatting.
    """
    logger = setup_logger(**LOGGER_CFG)
    meta_cfg = {
        'type': 'SQLiteMetaReader',
        'sqlite_path': SQLITE_PATH,
        'sqlite_join_cmd_path': SQL_JOIN_CMD_PATH,
        'logger_cfg': LOGGER_CFG
    }
    meta_reader = build_meta_reader(meta_cfg)
    ids = await meta_reader.get_ids()
    motion_records = list()
    for id in ids[:2]:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records.append(motion_record)
    n_frames_0 = motion_records[0].n_frames
    n_frames_1 = motion_records[1].n_frames
    # Test normal insert
    timeline = Timeline(
        start_frame=0,
        end_frame=n_frames_0 + n_frames_1,
        logger_cfg=LOGGER_CFG,
    )
    timeline.insert(
        start_idx=0,
        end_idx=n_frames_0,
        motion_record=motion_records[0],
        in_point=0,
        out_point=n_frames_0,
    )
    timeline.insert(
        start_idx=n_frames_0,
        end_idx=n_frames_0 + n_frames_1,
        motion_record=motion_records[1],
        in_point=0,
        out_point=n_frames_1,
    )
    # Test not starting from 0
    timeline = Timeline(
        start_frame=10,
        end_frame=n_frames_0 + n_frames_1 + 10,
        logger_cfg=LOGGER_CFG,
    )
    timeline.insert(
        start_idx=10,
        end_idx=n_frames_0 + 10,
        motion_record=motion_records[0],
        in_point=0,
        out_point=n_frames_0,
    )
    timeline.insert(
        start_idx=n_frames_0 + 10,
        end_idx=n_frames_0 + n_frames_1 + 10,
        motion_record=motion_records[0],
        in_point=0,
        out_point=n_frames_0,
    )
    # Test extension
    timeline = Timeline(
        start_frame=0,
        end_frame=n_frames_0 + n_frames_1,
        enable_extension=True,
        logger_cfg=LOGGER_CFG,
    )
    timeline.insert(
        start_idx=-2,
        end_idx=n_frames_0-2,
        motion_record=motion_records[0],
        in_point=0,
        out_point=n_frames_0,
    )
    assert timeline.start == -2
    assert timeline.n_frames == n_frames_0 + n_frames_1 + 2
    timeline.insert(
        start_idx=n_frames_0 + n_frames_1 + 1,
        end_idx=n_frames_0 + n_frames_1 + 1 + n_frames_1,
        motion_record=motion_records[1],
        in_point=0,
        out_point=n_frames_1,
    )
    assert timeline.end == n_frames_0 + n_frames_1 + 1 + n_frames_1
    assert timeline.n_frames == n_frames_0 + n_frames_1 + 1 + n_frames_1 + 2
    # Test get next blank
    blank = timeline.get_next_blank()
    logger.info(f'blank: {blank}')
    assert blank is not None
    assert blank[0] == n_frames_0 - 2
    assert blank[1] == n_frames_0 + n_frames_1 + 1
    # Test trigger
    timeline = Timeline(
        start_frame=0,
        end_frame=n_frames_0 + n_frames_1,
        logger_cfg=LOGGER_CFG,
    )
    timeline.insert(
        start_idx=0,
        end_idx=n_frames_0,
        motion_record=motion_records[0],
        in_point=0,
        out_point=n_frames_0,
        trigger='trigger_0',
    )
    timeline.insert(
        start_idx=n_frames_0,
        end_idx=n_frames_0 + n_frames_1,
        motion_record=motion_records[1],
        in_point=0,
        out_point=n_frames_1,
        trigger='trigger_1',
    )
    # Test preload task
    timeline = Timeline(
        start_frame=0,
        end_frame=n_frames_0 + n_frames_1,
        logger_cfg=LOGGER_CFG,
    )
    timeline.insert(
        start_idx=0,
        end_idx=n_frames_0,
        motion_record=motion_records[0],
        in_point=0,
        out_point=n_frames_0,
        preload_task=asyncio.create_task(asyncio.sleep(1)),
    )
    timeline.insert(
        start_idx=n_frames_0,
        end_idx=n_frames_0 + n_frames_1,
        motion_record=motion_records[1],
        in_point=0,
        out_point=n_frames_1,
        trigger='trigger_1',
    )
    # Test to_table
    table = timeline.to_table()
    logger.info('\n' + table)
    # Test extend a timeline from zero length
    timeline = Timeline(
        start_frame=0,
        end_frame=0,
        enable_extension=True,
        logger_cfg=LOGGER_CFG,
    )
    timeline.insert(
        start_idx=0,
        end_idx=n_frames_0,
        motion_record=motion_records[0],
        in_point=0,
        out_point=n_frames_0,
    )
    assert timeline.start == 0
    assert timeline.end == n_frames_0
    assert timeline.n_frames == n_frames_0
