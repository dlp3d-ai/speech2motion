import logging
import os
import uuid

import pytest

from speech2motion.apis.builder import build_api
from speech2motion.apis.streaming_speech2motion_v1 import (
    StreamingSpeech2MotionV1ChunkBody,
    StreamingSpeech2MotionV1ChunkEnd,
    StreamingSpeech2MotionV1ChunkStart,
)
from speech2motion.data_structures.motion_clip import MotionClip

LOGGER_CFG = dict(
    logger_name='test_streaming_speech2motion_v1',
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
        'max_workers': 8,
        'index_mapping_cfg_template': {
            'type': 'DictIndexMapping',
            'logger_cfg': LOGGER_CFG
        }
    }
    merge_cfg = {
        'type': 'Interpolation',
        'transit_frames': 15,
        'logger_cfg': LOGGER_CFG,
    }
    api_cfg = {
        'type': 'StreamingSpeech2MotionV1',
        'meta_reader_cfg': meta_cfg,
        'motion_reader_cfg': motion_cfg,
        'memory_cfg': memory_cfg,
        'text_segmentation_cfg': text_segmentation_cfg,
        'cache_cfg': cache_cfg,
        'merge_cfg': merge_cfg,
        'max_workers': 8,
        'first_body_fast_response': False,
        'logger_cfg': LOGGER_CFG,
    }
    api = await build_api(api_cfg)
    # Test using only one BodyChunk with as many default parameters as possible
    request_id = str(uuid.uuid4())
    start_chunk = StreamingSpeech2MotionV1ChunkStart(
        request_id=request_id,
        user_id='pytest_user_0',
        avatar='KQ-default',
        max_front_extension_duration=1.0,
        max_rear_extension_duration=10.0,
    )
    body_chunk = StreamingSpeech2MotionV1ChunkBody(
        request_id=request_id,
        duration=4.64,
        speech_text='已经是冬天了，真是好冷呀，好想吃火锅暖和暖和。',
        sequence_number=0,
    )
    end_chunk = StreamingSpeech2MotionV1ChunkEnd(
        request_id=request_id,
    )
    await api.handle_chunk_start(start_chunk)
    body_result = await api.handle_chunk_body(body_chunk)
    assert isinstance(body_result, MotionClip)
    end_result = await api.handle_chunk_end(end_chunk)
    assert end_result is None or isinstance(
        end_result, MotionClip)
    # Test covering first_body_fast_response
    request_id = str(uuid.uuid4())
    start_chunk = StreamingSpeech2MotionV1ChunkStart(
        request_id=request_id,
        user_id='pytest_user_0',
        avatar='KQ-default',
        max_front_extension_duration=1.0,
        max_rear_extension_duration=10.0,
        first_body_fast_response_override=True,
    )
    body_chunk = StreamingSpeech2MotionV1ChunkBody(
        request_id=request_id,
        duration=4.64,
        speech_text='已经是冬天了，真是好冷呀，好想吃火锅暖和暖和。',
        sequence_number=0,
    )
    end_chunk = StreamingSpeech2MotionV1ChunkEnd(
        request_id=request_id,
    )
    await api.handle_chunk_start(start_chunk)
    body_result = await api.handle_chunk_body(body_chunk)
    assert isinstance(body_result, MotionClip)
    end_result = await api.handle_chunk_end(end_chunk)
    assert end_result is None or isinstance(end_result, MotionClip)
    # Test returning logs
    request_id = str(uuid.uuid4())
    start_chunk = StreamingSpeech2MotionV1ChunkStart(
        request_id=request_id,
        user_id='pytest_user_0',
        avatar='KQ-default',
        max_front_extension_duration=1.0,
        max_rear_extension_duration=10.0,
        return_content='log',
    )
    body_chunk = StreamingSpeech2MotionV1ChunkBody(
        request_id=request_id,
        duration=4.64,
        speech_text='已经是冬天了，真是好冷呀，好想吃火锅暖和暖和。',
        sequence_number=0,
    )
    end_chunk = StreamingSpeech2MotionV1ChunkEnd(
        request_id=request_id,
    )
    await api.handle_chunk_start(start_chunk)
    body_result = await api.handle_chunk_body(body_chunk)
    assert isinstance(body_result, str)
    logging.info(f'body_log={body_result}')
    end_result = await api.handle_chunk_end(end_chunk)
    assert end_result is None or isinstance(end_result, str)
    logging.info(f'end_log={end_result}')

