import logging
import os
import time

import numpy as np
import pytest

from speech2motion.data_structures.motion_clip import MotionClip
from speech2motion.io.meta.builder import build_meta_reader
from speech2motion.io.motion.builder import build_motion_reader
from speech2motion.merge.builder import build_motion_clip_merge
from speech2motion.merge.interpolation import Interpolation
from speech2motion.utils.log import setup_logger

LOGGER_CFG = dict(
    logger_name='test_interpolation',
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
    correctly create an Interpolation instance from configuration.
    """
    cfg = dict(
        type='Interpolation',
        transit_frames=15,
        logger_cfg=LOGGER_CFG
    )
    merge = build_motion_clip_merge(cfg)
    assert isinstance(merge, Interpolation)

@pytest.mark.asyncio
async def test_predict_merge_n_frames():
    """Test predicting the number of frames after merging motion records.

    This test verifies that the predict_merge_n_frames method correctly
    calculates the total number of frames when merging multiple motion
    records. It loads motion records from the database and compares the
    predicted frame count with the sum of individual frame counts.
    """
    cfg = dict(
        type='Interpolation',
        transit_frames=15,
        logger_cfg=LOGGER_CFG
    )
    merge = build_motion_clip_merge(cfg)
    assert isinstance(merge, Interpolation)
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
async def test_merge_with_different_thread_counts():
    """Test concatenation of motion clips with different thread counts.

    This test verifies that motion clips can be successfully concatenated
    using interpolation with different numbers of worker threads. It loads
    motion data from the database, filters by avatar name, and tests
    concatenation performance with 1, 2, and 3 worker threads.
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
            type='Interpolation',
            transit_frames=15,
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
            f'Merged {len(motion_clips)} motion data using '
            f'{max_workers} threads, took {end_time - start_time:.2f} seconds.')

def create_test_motion_clip(n_frames: int,
                           joint_names: list[str],
                           has_blendshape: bool = False,
                           blendshape_names: list[str] | None = None) -> MotionClip:
    """Create a test MotionClip for testing purposes.

    This function creates a MotionClip with random data for testing
    interpolation and merging functionality. It can optionally include
    blendshape data.

    Args:
        n_frames (int): Number of frames in the motion clip.
        joint_names (list[str]): List of joint names.
        has_blendshape (bool, optional): Whether to include blendshape data.
            Defaults to False.
        blendshape_names (list[str] | None, optional): List of blendshape names.
            Required if has_blendshape is True. Defaults to None.

    Returns:
        MotionClip: Test MotionClip with random motion data.
    """
    joint_rotmat = np.random.rand(n_frames, len(joint_names), 3, 3)
    root_world_position = np.random.rand(n_frames, 3)

    if has_blendshape and blendshape_names is not None:
        blendshape_values = np.random.rand(n_frames, len(blendshape_names))
    else:
        blendshape_names = None
        blendshape_values = None

    return MotionClip(
        n_frames=n_frames,
        joint_names=joint_names,
        joint_rotmat=joint_rotmat,
        root_world_position=root_world_position,
        restpose_name='test_restpose',
        blendshape_names=blendshape_names,
        blendshape_values=blendshape_values,
        motion_record_id=1,
        logger_cfg=LOGGER_CFG
    )

@pytest.mark.asyncio
async def test_merge_with_both_blendshapes():
    """Test merging two MotionClips that both have blendshape data.

    This test verifies that two MotionClips with blendshape data can be
    successfully merged using interpolation. It ensures that the merged
    result preserves the blendshape information correctly.
    """
    logger = setup_logger(**LOGGER_CFG)

    # Create two MotionClips that both have blendshape data
    joint_names = ['joint1', 'joint2', 'joint3']
    blendshape_names = ['blendshape1', 'blendshape2']

    motion_clip_a = create_test_motion_clip(
        n_frames=20,
        joint_names=joint_names,
        has_blendshape=True,
        blendshape_names=blendshape_names
    )
    motion_clip_b = create_test_motion_clip(
        n_frames=25,
        joint_names=joint_names,
        has_blendshape=True,
        blendshape_names=blendshape_names
    )

    cfg = dict(
        type='Interpolation',
        transit_frames=10,
        logger_cfg=LOGGER_CFG
    )
    merge = build_motion_clip_merge(cfg)

    # Execute merging
    merged_clip = await merge.merge([motion_clip_a, motion_clip_b])

    # Verify results
    assert isinstance(merged_clip, MotionClip)
    assert merged_clip.blendshape_names == blendshape_names
    assert merged_clip.blendshape_values is not None
    assert merged_clip.blendshape_values.shape[1] == len(blendshape_names)
    assert merged_clip.joint_names == joint_names

    # Verify frame count calculation (subtract transition frames)
    expected_frames = await merge.predict_merge_n_frames([motion_clip_a, motion_clip_b])
    assert merged_clip.n_frames == expected_frames

    logger.info('Successfully merged two MotionClips with blendshapes, ' +
               f'merged frame count: {merged_clip.n_frames}')

@pytest.mark.asyncio
async def test_merge_without_blendshapes():
    """Test merging two MotionClips that both lack blendshape data.

    This test verifies that two MotionClips without blendshape data can be
    successfully merged using interpolation. It ensures that the merged
    result correctly handles the absence of blendshape information.
    """
    logger = setup_logger(**LOGGER_CFG)

    # Create two MotionClips that both lack blendshape data
    joint_names = ['joint1', 'joint2', 'joint3']

    motion_clip_a = create_test_motion_clip(
        n_frames=15,
        joint_names=joint_names,
        has_blendshape=False
    )
    motion_clip_b = create_test_motion_clip(
        n_frames=18,
        joint_names=joint_names,
        has_blendshape=False
    )

    cfg = dict(
        type='Interpolation',
        transit_frames=8,
        logger_cfg=LOGGER_CFG
    )
    merge = build_motion_clip_merge(cfg)

    # Execute merging
    merged_clip = await merge.merge([motion_clip_a, motion_clip_b])

    # Verify results
    assert isinstance(merged_clip, MotionClip)
    assert merged_clip.blendshape_names is None
    assert merged_clip.blendshape_values is None
    assert merged_clip.joint_names == joint_names

    # Verify frame count calculation (subtract transition frames)
    expected_frames = await merge.predict_merge_n_frames([motion_clip_a, motion_clip_b])
    assert merged_clip.n_frames == expected_frames

    logger.info('Successfully merged two MotionClips without blendshapes, ' +
               f'merged frame count: {merged_clip.n_frames}')

@pytest.mark.asyncio
async def test_merge_mixed_blendshapes():
    """Test merging MotionClips with mixed blendshape availability.

    This test verifies that attempting to merge a MotionClip with blendshape
    data and another without blendshape data raises an appropriate ValueError.
    It ensures proper error handling for incompatible blendshape configurations.
    """
    logger = setup_logger(**LOGGER_CFG)

    # Create MotionClips with mixed blendshape availability
    joint_names = ['joint1', 'joint2', 'joint3']
    blendshape_names = ['blendshape1', 'blendshape2', 'blendshape3']

    motion_clip_a = create_test_motion_clip(
        n_frames=22,
        joint_names=joint_names,
        has_blendshape=True,
        blendshape_names=blendshape_names
    )
    motion_clip_b = create_test_motion_clip(
        n_frames=19,
        joint_names=joint_names,
        has_blendshape=False
    )

    cfg = dict(
        type='Interpolation',
        transit_frames=12,
        logger_cfg=LOGGER_CFG
    )
    merge = build_motion_clip_merge(cfg)

    # Execute merging, should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        await merge.merge([motion_clip_a, motion_clip_b])

    # Verify error message contains blendshape-related content
    error_msg = str(exc_info.value)
    assert 'blendshape' in error_msg.lower() or 'blendshape_values' in error_msg

    logger.info('Mixed blendshape MotionClip merging test passed, ' +
               f'correctly raised ValueError: {error_msg}')

@pytest.mark.asyncio
async def test_merge_multiple_clips_with_blendshapes():
    """Test merging multiple MotionClips that all have blendshape data.

    This test verifies that multiple MotionClips with blendshape data can be
    successfully merged using interpolation. It ensures that the merged
    result preserves blendshape information across multiple clips.
    """
    logger = setup_logger(**LOGGER_CFG)

    # Create multiple MotionClips that all have blendshape data
    joint_names = ['joint1', 'joint2']
    blendshape_names = ['blendshape1', 'blendshape2', 'blendshape3']

    motion_clips = []
    for i in range(3):
        motion_clip = create_test_motion_clip(
            n_frames=15 + i * 5,  # 15, 20, 25 frames
            joint_names=joint_names,
            has_blendshape=True,
            blendshape_names=blendshape_names
        )
        motion_clips.append(motion_clip)

    cfg = dict(
        type='Interpolation',
        transit_frames=6,
        logger_cfg=LOGGER_CFG
    )
    merge = build_motion_clip_merge(cfg)

    # Execute merging
    merged_clip = await merge.merge(motion_clips)

    # Verify results
    assert isinstance(merged_clip, MotionClip)
    assert merged_clip.blendshape_names == blendshape_names
    assert merged_clip.blendshape_values is not None
    assert merged_clip.blendshape_values.shape[1] == len(blendshape_names)
    assert merged_clip.joint_names == joint_names

    # Verify frame count calculation
    expected_frames = await merge.predict_merge_n_frames(motion_clips)
    assert merged_clip.n_frames == expected_frames

    logger.info('Successfully merged multiple MotionClips with blendshapes, ' +
               f'merged frame count: {merged_clip.n_frames}')

@pytest.mark.asyncio
async def test_merge_blendshape_values_consistency():
    """Test blendshape values consistency during merging process.

    This test verifies that blendshape values remain consistent and within
    valid ranges during the merging process. It uses fixed blendshape values
    to ensure predictable behavior and validates the merged result.
    """
    logger = setup_logger(**LOGGER_CFG)

    # Create two MotionClips with fixed blendshape values for verification
    joint_names = ['joint1', 'joint2']
    blendshape_names = ['blendshape1', 'blendshape2']

    # Create fixed blendshape values
    n_frames_a = 10
    n_frames_b = 12
    blendshape_values_a = np.ones((n_frames_a, len(blendshape_names))) * 0.5
    blendshape_values_b = np.ones((n_frames_b, len(blendshape_names))) * 0.8

    motion_clip_a = MotionClip(
        n_frames=n_frames_a,
        joint_names=joint_names,
        joint_rotmat=np.random.rand(n_frames_a, len(joint_names), 3, 3),
        root_world_position=np.random.rand(n_frames_a, 3),
        restpose_name='test_restpose',
        blendshape_names=blendshape_names,
        blendshape_values=blendshape_values_a,
        motion_record_id=1,
        logger_cfg=LOGGER_CFG
    )

    motion_clip_b = MotionClip(
        n_frames=n_frames_b,
        joint_names=joint_names,
        joint_rotmat=np.random.rand(n_frames_b, len(joint_names), 3, 3),
        root_world_position=np.random.rand(n_frames_b, 3),
        restpose_name='test_restpose',
        blendshape_names=blendshape_names,
        blendshape_values=blendshape_values_b,
        motion_record_id=2,
        logger_cfg=LOGGER_CFG
    )

    cfg = dict(
        type='Interpolation',
        transit_frames=4,
        logger_cfg=LOGGER_CFG
    )
    merge = build_motion_clip_merge(cfg)

    # Execute merging
    merged_clip = await merge.merge([motion_clip_a, motion_clip_b])

    # Verify blendshape values consistency
    assert merged_clip.blendshape_values is not None
    assert merged_clip.blendshape_values.shape[1] == len(blendshape_names)

    # Verify merged blendshape values are within reasonable range
    assert np.all(merged_clip.blendshape_values >= 0)
    assert np.all(merged_clip.blendshape_values <= 1)

    logger.info('Blendshape values consistency test passed, ' +
               f'merged blendshape shape: {merged_clip.blendshape_values.shape}')
