import asyncio
import logging
import os
import socket

import pytest
import pytest_asyncio

from speech2motion.data_structures.motion_record import MotionRecord
from speech2motion.io.meta.builder import build_meta_reader
from speech2motion.io.meta.mysql_meta_reader import MySQLMetaReader

LOGGER_CFG = dict(
    logger_name='test_mysql_meta_reader',
    file_level=logging.DEBUG,
    console_level=logging.INFO,
    logger_path='logs/pytest.log',
)
# Load database connection information from environment variables
MYSQL_HOST = os.getenv('MYSQL_HOST', None)
MYSQL_PORT = int(os.getenv('MYSQL_PORT', 0))
MYSQL_USER = os.getenv('MYSQL_USER', '')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
MYSQL_DATABASE = 'motion_db'
SQL_JOIN_CMD_PATH = 'configs/3dac_sql_join.sql'

os.makedirs('logs', exist_ok=True)


def is_mysql_available() -> bool:
    """Check if MySQL server is available for testing.

    Returns:
        bool: True if MySQL is available, False otherwise.
    """
    if MYSQL_HOST is None:
        return False

    try:
        # Test TCP connection to MySQL port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)  # 2 second timeout
        result = sock.connect_ex((MYSQL_HOST, MYSQL_PORT))
        sock.close()
        return result == 0
    except Exception:
        return False


# Skip all tests if MySQL is not available
pytestmark = pytest.mark.skipif(
    not is_mysql_available(),
    reason="MySQL server is not available for testing"
)

def test_build_meta_reader():
    """Test building a meta reader instance.

    Verifies that the build_meta_reader function correctly creates a
    MySQLMetaReader instance and that it can establish a database connection.
    """
    cfg = {
        'type': 'MySQLMetaReader',
        'mysql_host': MYSQL_HOST,
        'mysql_port': MYSQL_PORT,
        'mysql_username': MYSQL_USER,
        'mysql_password': MYSQL_PASSWORD,
        'mysql_database': MYSQL_DATABASE,
        'mysql_join_cmd_path': SQL_JOIN_CMD_PATH,
        'logger_cfg': LOGGER_CFG
    }
    meta_reader = build_meta_reader(cfg)
    assert isinstance(meta_reader, MySQLMetaReader)
    async def _connect_and_disconnect():
        await meta_reader.connect()
        await meta_reader.disconnect()
    asyncio.run(_connect_and_disconnect())

@pytest_asyncio.fixture
async def mysql_meta_reader():
    """Fixture to create a MySQLMetaReader instance for testing.

    Creates a MySQLMetaReader instance with test configuration, establishes
    a database connection, and provides it to test functions. Automatically
    disconnects after each test.

    Yields:
        MySQLMetaReader: Connected MySQLMetaReader instance for testing.
    """
    reader = MySQLMetaReader(
        mysql_host=MYSQL_HOST,
        mysql_port=MYSQL_PORT,
        mysql_username=MYSQL_USER,
        mysql_password=MYSQL_PASSWORD,
        mysql_database=MYSQL_DATABASE,
        mysql_join_cmd_path=SQL_JOIN_CMD_PATH,
        logger_cfg=LOGGER_CFG
    )
    await reader.connect()
    yield reader
    await reader.disconnect()

@pytest.mark.asyncio
async def test_get_version(mysql_meta_reader):
    """Test getting the version of the motion database metadata.

    Verifies that the get_version method returns a valid version string
    representing the database metadata version.
    """
    version = await mysql_meta_reader.get_version()
    assert isinstance(version, str)  # Ensure return value is a string

@pytest.mark.asyncio
async def test_get_ids(mysql_meta_reader):
    """Test getting all motion_record_ids.

    Verifies that the get_ids method returns a non-empty list of motion
    record IDs from the database.
    """
    ids = await mysql_meta_reader.get_ids()
    assert isinstance(ids, list)
    assert len(ids) > 0

@pytest.mark.asyncio
async def test_get_meta_by_id(mysql_meta_reader):
    """Test getting metadata by ID.

    Verifies that the get_meta_by_id method returns valid metadata dictionaries
    for all available motion record IDs, and that the returned metadata
    contains the correct motion_record_id.
    """
    ids = await mysql_meta_reader.get_ids()
    for id in ids:
        meta = await mysql_meta_reader.get_meta_by_id(id)
        assert isinstance(meta, dict)
        assert meta['motion_record_id'] == id

@pytest.mark.asyncio
async def test_get_motion_record_by_id(mysql_meta_reader):
    """Test getting MotionRecord objects by ID.

    Verifies that the get_motion_record_by_id method returns valid MotionRecord
    objects for all available motion record IDs, and that the objects contain
    the expected properties and methods.
    """
    ids = await mysql_meta_reader.get_ids()
    idle_long_exist = False
    loopable_exist = False
    random_exist = False
    motion_keyword_exist = False
    speech_keyword_exist = False
    labels_exist = False
    for id in ids:
        motion_record = await mysql_meta_reader.get_motion_record_by_id(id)
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

@pytest.mark.asyncio
async def test_reconnect(mysql_meta_reader):
    """Test reconnecting to the database.

    Verifies that the connect method can establish a database connection
    and that the connection object is properly set.
    """
    await mysql_meta_reader.reconnect()
    # Ensure connection is not None
    assert mysql_meta_reader.mysql_connection is not None
