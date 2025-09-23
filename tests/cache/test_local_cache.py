import logging
import os
import time

import pytest

from speech2motion.cache.local_cache import CacheNotReadyError, LocalCache
from speech2motion.data_structures.motion_clip import MotionClip
from speech2motion.data_structures.restpose import Restpose
from speech2motion.io.meta.builder import build_meta_reader
from speech2motion.io.motion.builder import build_motion_reader
from speech2motion.io.restpose.builder import build_restpose_reader
from speech2motion.utils.log import setup_logger

LOGGER_CFG = dict(
    logger_name='test_local_cache',
    file_level=logging.DEBUG,
    console_level=logging.INFO,
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
async def test_sync():
    """Test local cache synchronization and data retrieval.

    This test verifies that the LocalCache can be properly initialized,
    synchronized with data sources, and used to retrieve motion clips and
    restpose data. It tests error handling, cache preparation, and data
    access functionality.
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
        'root_dir': MOTION_FILE_ROOT_DIR,
        'logger_cfg': LOGGER_CFG
    }
    motion_reader = build_motion_reader(motion_cfg)
    restpose_cfg = {
        'type': 'FilesystemRestposeReader',
        'file_paths': RESTPOSE_FILE_PATHS,
        'root_dir': RESTPOSE_ROOT_DIR,
        'logger_cfg': LOGGER_CFG
    }
    restpose_reader = build_restpose_reader(restpose_cfg)
    index_mapping_cfg_template = {
        'type': 'DictIndexMapping',
        'logger_cfg': LOGGER_CFG
    }
    local_cache = LocalCache(
        meta_reader=meta_reader,
        motion_reader=motion_reader,
        restpose_reader=restpose_reader,
        index_mapping_cfg_template=index_mapping_cfg_template,
        logger_cfg=LOGGER_CFG,
    )
    # Test CacheNotReadyError when cache is not ready
    with pytest.raises(CacheNotReadyError):
        await local_cache.get_motion_clip_by_id(1)
    await local_cache.prepare_next()
    await local_cache.switch_to_next()
    assert local_cache.cache_ready
    motion_records = local_cache.motion_records
    motion_record_id = next(iter(motion_records.keys()))
    motion_clip = await local_cache.get_motion_clip_by_id(motion_record_id)
    assert isinstance(motion_clip, MotionClip)
    # Test KeyError for non-existent motion clip ID
    with pytest.raises(KeyError):
        await local_cache.get_motion_clip_by_id(-1)
    avatar_mapping_keys = await local_cache.avatar_mapping.keys()
    assert len(avatar_mapping_keys) > 0
    type_mapping_keys = await local_cache.type_mapping.keys()
    assert len(type_mapping_keys) > 0
    motion_keyword_mapping_keys = await local_cache.motion_keyword_mapping.keys()
    assert len(motion_keyword_mapping_keys) > 0
    speech_keyword_mapping_keys = await local_cache.speech_keyword_mapping.keys()
    assert len(speech_keyword_mapping_keys) > 0
    label_mapping_keys = await local_cache.label_mapping.keys()
    assert len(label_mapping_keys) > 0
    assert len(local_cache.motion_clips_cache) > 0
    ids = await meta_reader.get_ids()
    start_time = time.time()
    for id in ids:
        motion_clip = await local_cache.get_motion_clip_by_id(id)
        assert isinstance(motion_clip, MotionClip)
    logger.info(f'Loaded {len(ids)} motion data, took: {time.time() - start_time:.2f}s')
    restpose_names = await restpose_reader.get_restpose_names()
    for restpose_name in restpose_names:
        restpose = await local_cache.get_restpose_by_name(restpose_name)
        assert isinstance(restpose, Restpose)

