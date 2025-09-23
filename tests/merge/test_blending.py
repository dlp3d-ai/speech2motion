import logging
import os
import time

import pytest

from speech2motion.data_structures.motion_clip import MotionClip
from speech2motion.io.meta.builder import build_meta_reader
from speech2motion.io.motion.builder import build_motion_reader
from speech2motion.merge.blending import Blending, WrongInputNumber
from speech2motion.merge.builder import build_motion_clip_merge
from speech2motion.utils.log import setup_logger

LOGGER_CFG = dict(
    logger_name='test_blending',
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
    correctly create a Blending instance from configuration.
    """
    cfg = dict(
        type='Blending',
        logger_cfg=LOGGER_CFG
    )
    merge = build_motion_clip_merge(cfg)
    assert isinstance(merge, Blending)

@pytest.mark.asyncio
async def test_predict_merge_n_frames():
    """Test predicting the number of frames after blending motion records.

    This test verifies that the predict_merge_n_frames method correctly
    calculates the number of frames when blending two motion records.
    It also tests error handling when more than two motion records are
    provided, which should raise a WrongInputNumber exception.
    """
    cfg = dict(
        type='Blending',
        logger_cfg=LOGGER_CFG
    )
    merge = build_motion_clip_merge(cfg)
    assert isinstance(merge, Blending)
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
        if len(motion_records) >= 2:
            break
    motion_records.sort(key=lambda x: x.n_frames)
    n_frames = await merge.predict_merge_n_frames(motion_records)
    assert n_frames == motion_records[1].n_frames
    with pytest.raises(WrongInputNumber):
        await merge.predict_merge_n_frames(motion_records + motion_records)


@pytest.mark.asyncio
async def test_blending():
    """Test blending of motion clips with different thread counts.

    This test verifies that motion clips can be successfully blended
    using the Blending merge strategy with different numbers of worker
    threads. It loads motion data from the database, filters by avatar name,
    and tests blending performance with 1, 2, and 3 worker threads.
    The test uses specific alignment parameters for blending.
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
        if len(motion_clips) >= 2:
            break
    logger.info(f'Loaded {len(motion_clips)} motion data for {avatar_name}.')
    motion_records.sort(key=lambda x: x.n_frames)
    motion_clips.sort(key=lambda x: x.n_frames)
    # Test blending with different thread counts
    for max_workers in range(1, 4):
        cfg = dict(
            type='Blending',
            max_workers=max_workers,
            logger_cfg=LOGGER_CFG
        )
        merge = build_motion_clip_merge(cfg)
        start_time = time.time()
        motion_clip = await merge.merge(
            motion_clips,
            align_frame=0,
            startup_frame=motion_records[0].startup_frame,
            recovery_frame=motion_records[0].recovery_frame,
            )
        assert isinstance(motion_clip, MotionClip)
        n_frames_predict = await merge.predict_merge_n_frames(motion_records)
        assert motion_clip.n_frames == n_frames_predict
        end_time = time.time()
        logger.info(
            f'Blended {len(motion_clips)} motion data using '
            f'{max_workers} threads, took {end_time - start_time:.5f} seconds.')
