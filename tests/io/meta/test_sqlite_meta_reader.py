import logging
import os

import pytest
import pytest_asyncio

from speech2motion.data_structures.motion_record import MotionRecord
from speech2motion.io.meta.builder import build_meta_reader
from speech2motion.io.meta.sqlite_meta_reader import SQLiteMetaReader

LOGGER_CFG = dict(
    logger_name='test_sqlite_meta_reader',
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

def test_build_meta_reader():
    """Test building a meta reader instance.

    Verifies that the build_meta_reader function correctly creates a
    SQLiteMetaReader instance with the proper configuration.
    """
    cfg = {
        'type': 'SQLiteMetaReader',
        'sqlite_path': SQLITE_PATH,
        'sqlite_join_cmd_path': SQL_JOIN_CMD_PATH,
        'logger_cfg': LOGGER_CFG
    }
    meta_reader = build_meta_reader(cfg)
    assert isinstance(meta_reader, SQLiteMetaReader)

@pytest_asyncio.fixture
async def sqlite_meta_reader():
    """Fixture to create a SQLiteMetaReader instance for testing.

    Creates a SQLiteMetaReader instance with test configuration and provides
    it to test functions. The reader creates new database connections for
    each operation to ensure thread safety and proper resource management.

    Yields:
        SQLiteMetaReader: SQLiteMetaReader instance for testing.
    """
    reader = SQLiteMetaReader(
        sqlite_path=SQLITE_PATH,
        sqlite_join_cmd_path=SQL_JOIN_CMD_PATH,
        logger_cfg=LOGGER_CFG
    )
    yield reader

@pytest.mark.asyncio
async def test_get_version(sqlite_meta_reader):
    """Test getting the version of the motion database metadata.

    Verifies that the get_version method returns a valid version string
    representing the database metadata version.
    """
    version = await sqlite_meta_reader.get_version()
    assert isinstance(version, str)  # Ensure return value is a string

@pytest.mark.asyncio
async def test_get_ids(sqlite_meta_reader):
    """Test getting all motion_record_ids.

    Verifies that the get_ids method returns a non-empty list of motion
    record IDs from the database.
    """
    ids = await sqlite_meta_reader.get_ids()
    assert isinstance(ids, list)
    assert len(ids) > 0

@pytest.mark.asyncio
async def test_get_meta_by_id(sqlite_meta_reader):
    """Test getting metadata by ID.

    Verifies that the get_meta_by_id method returns valid metadata dictionaries
    for all available motion record IDs, and that the returned metadata
    contains the correct motion_record_id.
    """
    ids = await sqlite_meta_reader.get_ids()
    for id in ids:
        meta = await sqlite_meta_reader.get_meta_by_id(id)
        assert isinstance(meta, dict)
        assert meta['motion_record_id'] == id

@pytest.mark.asyncio
async def test_get_motion_record_by_id(sqlite_meta_reader):
    """Test getting MotionRecord objects by ID.

    Verifies that the get_motion_record_by_id method returns valid MotionRecord
    objects for all available motion record IDs, and that the objects contain
    the expected properties and methods.
    """
    ids = await sqlite_meta_reader.get_ids()
    idle_long_exist = False
    loopable_exist = False
    random_exist = False
    motion_keyword_exist = False
    speech_keyword_exist = False
    labels_exist = False
    for id in ids:
        motion_record = await sqlite_meta_reader.get_motion_record_by_id(id)
        assert isinstance(motion_record, MotionRecord)
        assert motion_record.motion_record_id == id
        idle_long_exist = idle_long_exist or motion_record.is_idle_long
        loopable_exist = loopable_exist or motion_record.is_loopable()
        random_exist = random_exist or motion_record.is_random()
        motion_keyword_exist = motion_keyword_exist or motion_record.is_motion_keyword()
        speech_keyword_exist = speech_keyword_exist or motion_record.is_speech_keyword()
        labels_exist = labels_exist or len(motion_record.labels) > 0
    assert idle_long_exist
    assert loopable_exist
    assert random_exist
    assert motion_keyword_exist
    assert speech_keyword_exist

