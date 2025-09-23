import logging
import os
import random
import uuid

import pytest

from speech2motion.apis.builder import build_api
from speech2motion.apis.streaming_speech2motion_v3 import (
    StreamingSpeech2MotionV3ChunkBody,
    StreamingSpeech2MotionV3ChunkEnd,
    StreamingSpeech2MotionV3ChunkStart,
)
from speech2motion.data_structures.motion_clip import MotionClip

LOGGER_CFG = dict(
    logger_name='test_streaming_speech2motion_v3',
    file_level=logging.DEBUG,
    console_level=logging.ERROR,
    logger_path='logs/pytest.log',
)
SQLITE_PATH = 'data/motion_database.db'
SQL_JOIN_CMD_PATH = 'configs/3dac_sql_join.sql'
MOTION_FILE_ROOT_DIR = 'data/motion_files'
RESTPOSE_FILE_PATHS = {'KQ-default': 'KQ_default_0326_skeleton.npz'}
RESTPOSE_ROOT_DIR = 'data/restpose_npz'

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

@pytest.mark.asyncio
async def test_generate():
    meta_cfg = {
        'type': 'SQLiteMetaReader',
        'sqlite_path': SQLITE_PATH,
        'sqlite_join_cmd_path': SQL_JOIN_CMD_PATH,
        'logger_cfg': LOGGER_CFG
    }
    motion_cfg = {
        'type': 'SQLiteFilesystemMotionReader',
        'sqlite_path': SQLITE_PATH,
        'sqlite_join_cmd_path': SQL_JOIN_CMD_PATH,
        'root_dir': MOTION_FILE_ROOT_DIR,
        'blendshape_names': 'configs/mmd_blendshapes.json',
        'logger_cfg': LOGGER_CFG,
    }
    restpose_cfg = {
        'type': 'FilesystemRestposeReader',
        'file_paths': RESTPOSE_FILE_PATHS,
        'root_dir': RESTPOSE_ROOT_DIR,
        'logger_cfg': LOGGER_CFG
    }
    memory_cfg = {
        'type': 'LocalMemory',
        'memory_duration': 60,
    }
    text_segmentation_cfg = {
        'type': 'JiebaTextSegmentation'
    }
    cache_cfg = {
        'type': 'LocalCache',
        'index_mapping_cfg_template': {
            'type': 'DictIndexMapping',
            'logger_cfg': LOGGER_CFG
        }
    }
    interpolation_cfg = {
        'type': 'Interpolation',
        'transit_frames': 15,
        'logger_cfg': LOGGER_CFG,
    }
    blending_cfg = {
        'type': 'Blending',
        'logger_cfg': LOGGER_CFG,
    }
    api_cfg = {
        'type': 'StreamingSpeech2MotionV3',
        'meta_reader_cfg': meta_cfg,
        'motion_reader_cfg': motion_cfg,
        'restpose_reader_cfg': restpose_cfg,
        'memory_cfg': memory_cfg,
        'text_segmentation_cfg': text_segmentation_cfg,
        'cache_cfg': cache_cfg,
        'interpolation_cfg': interpolation_cfg,
        'blending_cfg': blending_cfg,
        'logger_cfg': LOGGER_CFG,
    }
    api = await build_api(api_cfg)
    # Test using only one BodyChunk with as many default parameters as possible
    request_id = str(uuid.uuid4())
    start_chunk = StreamingSpeech2MotionV3ChunkStart(
        request_id=request_id,
        user_id='pytest_user_0',
        avatar='KQ-default',
        max_front_extension_duration=1.0,
        max_rear_extension_duration=10.0,
    )
    body_chunk = StreamingSpeech2MotionV3ChunkBody(
        request_id=request_id,
        duration=4.64,
        speech_text='已经是冬天了，真是好冷呀，好想吃火锅暖和暖和。',
        sequence_number=0,
    )
    end_chunk = StreamingSpeech2MotionV3ChunkEnd(
        request_id=request_id,
    )
    await api.handle_chunk_start(start_chunk)
    body_result = await api.handle_chunk_body(body_chunk)
    assert isinstance(body_result, MotionClip)
    # Default app_name is python_backend
    assert body_result.app_name == 'python_backend'
    end_result = await api.handle_chunk_end(end_chunk)
    assert end_result is None or isinstance(
        end_result, MotionClip)
    # Test app_name as babylon
    request_id = str(uuid.uuid4())
    start_chunk = StreamingSpeech2MotionV3ChunkStart(
        request_id=request_id,
        user_id='pytest_user_0',
        avatar='KQ-default',
        max_front_extension_duration=1.0,
        max_rear_extension_duration=10.0,
        app_name='babylon',
    )
    body_chunk = StreamingSpeech2MotionV3ChunkBody(
        request_id=request_id,
        duration=4.64,
        speech_text='已经是冬天了，真是好冷呀，好想吃火锅暖和暖和。',
        sequence_number=0,
    )
    end_chunk = StreamingSpeech2MotionV3ChunkEnd(
        request_id=request_id,
    )
    await api.handle_chunk_start(start_chunk)
    body_result = await api.handle_chunk_body(body_chunk)
    assert isinstance(body_result, MotionClip)
    assert body_result.app_name == 'babylon'
    end_result = await api.handle_chunk_end(end_chunk)
    assert end_result is None or isinstance(
        end_result, MotionClip)
    # Test covering first_body_fast_response
    request_id = str(uuid.uuid4())
    start_chunk = StreamingSpeech2MotionV3ChunkStart(
        request_id=request_id,
        user_id='pytest_user_0',
        avatar='KQ-default',
        max_front_extension_duration=1.0,
        max_rear_extension_duration=10.0,
    )
    body_chunk = StreamingSpeech2MotionV3ChunkBody(
        request_id=request_id,
        duration=4.64,
        speech_text='已经是冬天了，真是好冷呀，好想吃火锅暖和暖和。',
        sequence_number=0,
    )
    end_chunk = StreamingSpeech2MotionV3ChunkEnd(
        request_id=request_id,
    )
    await api.handle_chunk_start(start_chunk)
    body_result = await api.handle_chunk_body(body_chunk)
    assert isinstance(body_result, MotionClip)
    end_result = await api.handle_chunk_end(end_chunk)
    assert end_result is None or isinstance(end_result, MotionClip)
    # Test label_expression
    request_id = str(uuid.uuid4())
    start_chunk = StreamingSpeech2MotionV3ChunkStart(
        request_id=request_id,
        user_id='pytest_user_0',
        avatar='KQ-default',
        max_front_extension_duration=1.0,
        max_rear_extension_duration=10.0,
    )
    body_chunk = StreamingSpeech2MotionV3ChunkBody(
        request_id=request_id,
        duration=4.64,
        speech_text='已经是冬天了，真是好冷呀，好想吃火锅暖和暖和。',
        sequence_number=0,
        label_expression='Happiness | Neutral',
    )
    end_chunk = StreamingSpeech2MotionV3ChunkEnd(
        request_id=request_id,
    )
    await api.handle_chunk_start(start_chunk)
    body_result = await api.handle_chunk_body(body_chunk)
    assert isinstance(body_result, MotionClip)
    end_result = await api.handle_chunk_end(end_chunk)
    assert end_result is None or isinstance(end_result, MotionClip)
    # Test input/gpt_4o_responses.txt
    with open('input/gpt_4o_responses.txt', encoding='utf-8') as f:
        test_text_list = f.readlines()
    logging.info(f'Read {len(test_text_list)} lines of test text.')
    max_front_extension_duration = 1.0
    max_rear_extension_duration = 10.0
    for line in test_text_list:
        request_id = str(uuid.uuid4())
        # Randomly select app_name
        app_name = random.choice(['babylon', 'python_backend'])

        start_chunk = StreamingSpeech2MotionV3ChunkStart(
            request_id=request_id,
            user_id='pytest_user_0',
            avatar='KQ-default',
            max_front_extension_duration=max_front_extension_duration,
            max_rear_extension_duration=max_rear_extension_duration,
            app_name=app_name,
        )
        await api.handle_chunk_start(start_chunk)
        splits = line.split('，')
        sequence_number = 0
        for split in splits:
            clean_text = split.strip()
            if len(clean_text) == 0:
                continue
            duration = len(clean_text)/5
            body_chunk = StreamingSpeech2MotionV3ChunkBody(
                request_id=request_id,
                duration=duration,
                speech_text=clean_text,
                sequence_number=sequence_number,
            )
            body_result = await api.handle_chunk_body(body_chunk)
            n_frames_lowerbound = int(duration * 30) - 1
            n_frames_upperbound = int(duration * 30) + 1
            if sequence_number == 0:
                n_frames_upperbound += int(max_front_extension_duration * 30)
            assert isinstance(body_result, MotionClip)
            assert body_result.n_frames >= n_frames_lowerbound
            assert body_result.n_frames <= n_frames_upperbound
            sequence_number += 1
        end_chunk = StreamingSpeech2MotionV3ChunkEnd(
            request_id=request_id,
        )
        end_result = await api.handle_chunk_end(end_chunk)
        if end_result is not None:
            assert isinstance(end_result, MotionClip)
            n_frames_lowerbound = 1
            n_frames_upperbound = int(max_rear_extension_duration * 30) + 1
            assert end_result.n_frames >= n_frames_lowerbound
            assert end_result.n_frames <= n_frames_upperbound

