import logging
import os

import pytest
import pytest_asyncio

from speech2motion.data_structures.motion_clip import MotionClip
from speech2motion.io.motion.builder import build_motion_reader
from speech2motion.io.motion.sqlite_filesystem_motion_reader import (
    SQLiteFilesystemMotionReader,
)

LOGGER_CFG = dict(
    logger_name='test_sqlite_filesystem_motion_reader',
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


def test_build_motion_reader():
    """Test building a SQLiteFilesystemMotionReader instance.

    This test verifies that the build_motion_reader function can correctly
    create a SQLiteFilesystemMotionReader instance from configuration with
    the proper SQLite and filesystem parameters.
    """
    cfg = {
        'type': 'SQLiteFilesystemMotionReader',
        'sqlite_path': SQLITE_PATH,
        'sqlite_join_cmd_path': SQL_JOIN_CMD_PATH,
        'root_dir': ROOT_DIR,
        'logger_cfg': LOGGER_CFG
    }
    motion_reader = build_motion_reader(cfg)
    assert isinstance(motion_reader, SQLiteFilesystemMotionReader)

@pytest_asyncio.fixture
async def motion_reader():
    """Fixture to create a SQLiteFilesystemMotionReader instance for testing.

    This fixture creates a SQLiteFilesystemMotionReader instance with the
    configured SQLite and filesystem parameters and provides the reader for
    testing. The reader creates new database connections for each operation
    to ensure thread safety and proper resource management. It automatically
    shuts down all resources (including thread pool executor) after the
    test completes to prevent resource leaks.

    Yields:
        SQLiteFilesystemMotionReader: Motion reader instance ready for testing.
    """
    reader = SQLiteFilesystemMotionReader(
        sqlite_path=SQLITE_PATH,
        sqlite_join_cmd_path=SQL_JOIN_CMD_PATH,
        root_dir=ROOT_DIR,
        logger_cfg=LOGGER_CFG
    )
    yield reader
    # Ensure proper cleanup
    # Shutdown thread pool executor
    if hasattr(reader, 'permanent_executor') and reader.permanent_executor:
        reader.permanent_executor.shutdown(wait=True)

@pytest.mark.asyncio
async def test_get_version(motion_reader: SQLiteFilesystemMotionReader):
    """Test getting the motion database version.

    This test verifies that the get_version method returns a valid
    version string and that the version is properly initialized. It
    ensures the version is not the default 'not_initialized' value.
    """
    version = await motion_reader.get_version()
    assert isinstance(version, str)  # Ensure return value is string
    assert version != 'not_initialized'  # Ensure version is initialized

@pytest.mark.asyncio
async def test_get_ids(motion_reader: SQLiteFilesystemMotionReader):
    """Test getting all motion IDs from the database.

    This test verifies that the get_ids method returns a list of
    motion IDs and that there is at least one motion record available
    in the database. It ensures the returned data structure is correct
    and contains valid motion data.
    """
    ids = await motion_reader.get_ids()
    assert isinstance(ids, list)
    assert len(ids) > 0  # Ensure there is motion data

@pytest.mark.asyncio
async def test_get_version_from_sqlite(motion_reader: SQLiteFilesystemMotionReader):
    """Test getting version information from SQLite database.

    This test verifies that the _get_version_from_sqlite method
    returns a valid version string from the SQLite database. It
    ensures the version is properly retrieved and is not the default
    'not_initialized' value.
    """
    version = await motion_reader._get_version_from_sqlite()
    assert isinstance(version, str)
    assert version != 'not_initialized'

@pytest.mark.asyncio
async def test_get_motion_clip_by_id(motion_reader: SQLiteFilesystemMotionReader):
    """Test getting motion data by ID from SQLite database.

    This test verifies that the get_motion_clip_by_id method returns
    valid MotionClip instances and that the motion data contains
    expected cutoff ranges and cutoff frames. It iterates through
    available motion IDs to find records with both cutoff ranges
    and cutoff frames to ensure data completeness.
    """
    ids = await motion_reader.get_ids()
    cutoff_ranges_exist = False
    cutoff_frames_exist = False
    for id in ids:
        motion_clip = await motion_reader.get_motion_clip_by_id(id)
        assert isinstance(motion_clip, MotionClip)
        if motion_clip.cutoff_ranges is not None and \
                len(motion_clip.cutoff_ranges) > 0:
            cutoff_ranges_exist = True
        if len(motion_clip.cutoff_frames) > 2:
            cutoff_frames_exist = True
        if cutoff_frames_exist and cutoff_ranges_exist:
            break
    assert cutoff_ranges_exist
    assert cutoff_frames_exist

