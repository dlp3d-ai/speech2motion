import logging
import os
import time

import pytest

from speech2motion.data_structures.motion_clip import MotionClip
from speech2motion.io.meta.builder import build_meta_reader
from speech2motion.io.motion.builder import build_motion_reader
from speech2motion.merge.builder import build_motion_clip_merge
from speech2motion.merge.concatenation import Concatenation
from speech2motion.utils.log import setup_logger

LOGGER_CFG = dict(
    logger_name='test_concatenation',
    file_level=logging.DEBUG,
    console_level=logging.INFO,
    logger_path='logs/pytest.log',
)
SQLITE_PATH = 'data/motion_database.db'
SQL_JOIN_CMD_PATH = 'configs/3dac_sql_join.sql'
ROOT_DIR = 'data/motion_files'

os.makedirs('logs', exist_ok=True)


def is_sqlite_available() -> bool:
    """Check if SQLite database file is available for testing.

    This function verifies that the SQLite database file exists at the
    specified path and is accessible for testing purposes.

    Returns:
        bool: True if SQLite database file exists and is accessible,
            False otherwise.
    """
    if os.path.exists(SQLITE_PATH):
        return True
    return False


# Skip all tests if SQLite is not available
pytestmark = pytest.mark.skipif(
    not is_sqlite_available(),
    reason="SQLite server is not available for testing"
)

def test_build_motion_clip_merge():
    """Test building a motion clip merge instance.

    This test verifies that the build_motion_clip_merge function can
    correctly create a Concatenation instance from configuration.
    """
    cfg = dict(
        type='Concatenation',
        logger_cfg=LOGGER_CFG
    )
    merge = build_motion_clip_merge(cfg)
    assert isinstance(merge, Concatenation)

@pytest.mark.asyncio
async def test_predict_merge_n_frames():
    """Test predicting the number of frames after concatenating motion records.

    This test verifies that the predict_merge_n_frames method correctly
    calculates the total number of frames when concatenating multiple motion
    records. It loads motion records from the database and compares the
    predicted frame count with the sum of individual frame counts.
    """
    cfg = dict(
        type='Concatenation',
        logger_cfg=LOGGER_CFG
    )
    merge = build_motion_clip_merge(cfg)
    assert isinstance(merge, Concatenation)
    meta_cfg = {
        'type': 'SQLiteMetaReader',
        'sqlite_path': SQLITE_PATH,
        'sqlite_join_cmd_path': SQL_JOIN_CMD_PATH,
        'logger_cfg': LOGGER_CFG
    }
    meta_reader = build_meta_reader(meta_cfg)
    meta_ids = await meta_reader.get_ids()
    motion_records = []
    for id in meta_ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records.append(motion_record)
    n_frames_sum = 0
    for motion_record in motion_records:
        n_frames_sum += motion_record.n_frames
    n_frames = await merge.predict_merge_n_frames(motion_records)
    assert n_frames == n_frames_sum

@pytest.mark.asyncio
async def test_concatenation():
    """Test concatenation of motion clips with different thread counts.

    This test verifies that motion clips can be successfully concatenated
    using the Concatenation merge strategy with different numbers of worker
    threads. It loads motion data from the database, filters by avatar name,
    and tests concatenation performance with 1, 2, and 3 worker threads.
    """
    logger = setup_logger(**LOGGER_CFG)
    meta_cfg = {
        'type': 'SQLiteMetaReader',
        'sqlite_path': SQLITE_PATH,
        'sqlite_join_cmd_path': SQL_JOIN_CMD_PATH,
        'logger_cfg': LOGGER_CFG
    }
    meta_reader = build_meta_reader(meta_cfg)
    motion_cfg = {
        'type': 'SQLiteFilesystemMotionReader',
        'sqlite_path': SQLITE_PATH,
        'sqlite_join_cmd_path': SQL_JOIN_CMD_PATH,
        'root_dir': ROOT_DIR,
        'logger_cfg': LOGGER_CFG
    }
    motion_reader = build_motion_reader(motion_cfg)
    meta_ids = await meta_reader.get_ids()
    avatar_name = None
    motion_records = []
    motion_clips = []
    for id in meta_ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        if avatar_name is None:
            avatar_name = motion_record.avatar_name
        elif avatar_name != motion_record.avatar_name:
            continue
        motion_clip = await motion_reader.get_motion_clip_by_id(id)
        motion_clips.append(motion_clip)
        motion_records.append(motion_record)
    logger.info(f'Loaded {len(motion_clips)} motion data for {avatar_name}.')
    # Test concatenation with different thread counts
    for max_workers in range(1, 4):
        cfg = dict(
            type='Concatenation',
            max_workers=max_workers,
            logger_cfg=LOGGER_CFG
        )
        merge = build_motion_clip_merge(cfg)
        start_time = time.time()
        motion_clip = await merge.merge(motion_clips)
        assert isinstance(motion_clip, MotionClip)
        n_frames_predict = await merge.predict_merge_n_frames(motion_records)
        assert motion_clip.n_frames == n_frames_predict
        end_time = time.time()
        logger.info(
            f'Concatenated {len(motion_clips)} motion data using '
            f'{max_workers} threads, took {end_time - start_time:.5f} seconds.')
