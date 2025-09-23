import asyncio
import logging
import os
import socket

import pytest
import pytest_asyncio

from speech2motion.data_structures.motion_clip import MotionClip
from speech2motion.io.motion.builder import build_motion_reader
from speech2motion.io.motion.minio_mysql_motion_reader import MinioMySQLMotionReader

LOGGER_CFG = dict(
    logger_name='test_minio_mysql_motion_reader',
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
MINIO_ENDPOINT = os.getenv('OSS_ENDPOINT', None)
MINIO_ACCESS_KEY = os.getenv('OSS_ACCESS_KEY', '')
MINIO_SECRET_KEY = os.getenv('OSS_SECRET_KEY', '')
MINIO_BUCKET = '3dac'


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

def is_minio_available() -> bool:
    """Check if Minio server is available for testing.

    Returns:
        bool: True if Minio is available, False otherwise.
    """
    if MINIO_ENDPOINT is None:
        return False
    try:
        # Test TCP connection to MinIO port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)  # 2 second timeout
        if ':' in MINIO_ENDPOINT:
            splits = MINIO_ENDPOINT.split(':')
            host = splits[0]
            port = int(splits[1])
        else:
            host = MINIO_ENDPOINT
            port = 9000
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

pytestmark = pytest.mark.skipif(
    not is_mysql_available() or not is_minio_available(),
    reason="MySQL server or Minio server is not available for testing"
)

os.makedirs('logs', exist_ok=True)

def test_build_motion_reader():
    """Test building a MinioMySQLMotionReader instance.

    This test verifies that the build_motion_reader function can correctly
    create a MinioMySQLMotionReader instance from configuration and
    establish connections to both MySQL and MinIO services.
    """
    cfg = {
        'type': 'MinioMySQLMotionReader',
        'mysql_host': MYSQL_HOST,
        'mysql_port': MYSQL_PORT,
        'mysql_username': MYSQL_USER,
        'mysql_password': MYSQL_PASSWORD,
        'mysql_database': MYSQL_DATABASE,
        'mysql_join_cmd_path': SQL_JOIN_CMD_PATH,
        'endpoint': MINIO_ENDPOINT,
        'access_key': MINIO_ACCESS_KEY,
        'secret_key': MINIO_SECRET_KEY,
        'bucket_name': MINIO_BUCKET,
        'logger_cfg': LOGGER_CFG
    }
    motion_reader = build_motion_reader(cfg)
    assert isinstance(motion_reader, MinioMySQLMotionReader)
    async def _connect_and_disconnect():
        await motion_reader.connect()
        await motion_reader.disconnect()
    asyncio.run(_connect_and_disconnect())

@pytest_asyncio.fixture
async def motion_reader():
    """Fixture to create a MinioMySQLMotionReader instance for testing.

    This fixture creates a MinioMySQLMotionReader instance with the
    configured MySQL and MinIO connection parameters, establishes
    connections, and provides the reader for testing. It automatically
    disconnects after the test completes.

    Yields:
        MinioMySQLMotionReader: Connected motion reader instance.
    """
    reader = MinioMySQLMotionReader(
        mysql_host=MYSQL_HOST,
        mysql_port=MYSQL_PORT,
        mysql_username=MYSQL_USER,
        mysql_password=MYSQL_PASSWORD,
        mysql_database=MYSQL_DATABASE,
        mysql_join_cmd_path=SQL_JOIN_CMD_PATH,
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        bucket_name=MINIO_BUCKET,
        logger_cfg=LOGGER_CFG
    )
    await reader.connect()
    yield reader
    await reader.disconnect()

@pytest.mark.asyncio
async def test_get_version(motion_reader: MinioMySQLMotionReader):
    """Test getting the motion database version.

    This test verifies that the get_version method returns a valid
    version string and that the version is properly initialized.
    """
    version = await motion_reader.get_version()
    assert isinstance(version, str)  # Ensure return value is string
    assert version != 'not_initialized'  # Ensure version is initialized

@pytest.mark.asyncio
async def test_get_ids(motion_reader: MinioMySQLMotionReader):
    """Test getting all motion IDs.

    This test verifies that the get_ids method returns a list of
    motion IDs and that there is at least one motion record available.
    """
    ids = await motion_reader.get_ids()
    assert isinstance(ids, list)
    assert len(ids) > 0  # Ensure there is motion data

@pytest.mark.asyncio
async def test_reconnect(motion_reader: MinioMySQLMotionReader):
    """Test reconnecting to the database.

    This test verifies that the reconnect method properly re-establishes
    the MySQL connection after an initial connection.
    """
    await motion_reader.connect()
    assert motion_reader.mysql_connection is not None
    await motion_reader.reconnect()
    assert motion_reader.mysql_connection is not None

@pytest.mark.asyncio
async def test_disconnect(motion_reader: MinioMySQLMotionReader):
    """Test disconnecting from the database.

    This test verifies that the disconnect method properly closes
    the MySQL connection and sets the connection to None.
    """
    await motion_reader.connect()
    assert motion_reader.mysql_connection is not None
    await motion_reader.disconnect()
    assert motion_reader.mysql_connection is None

@pytest.mark.asyncio
async def test_ensure_connected(motion_reader: MinioMySQLMotionReader):
    """Test ensuring database connection.

    This test verifies that the ensure_connected method properly
    establishes a MySQL connection when none exists.
    """
    await motion_reader.disconnect()
    assert motion_reader.mysql_connection is None
    await motion_reader.ensure_connected()
    assert motion_reader.mysql_connection is not None

@pytest.mark.asyncio
async def test_get_version_from_mysql(motion_reader: MinioMySQLMotionReader):
    """Test getting version information from MySQL.

    This test verifies that the _get_version_from_mysql method
    returns a valid version string from the MySQL database.
    """
    await motion_reader.connect()
    version = await motion_reader._get_version_from_mysql()
    assert isinstance(version, str)
    assert version != 'not_initialized'

@pytest.mark.asyncio
async def test_get_motion_clip_by_id(motion_reader: MinioMySQLMotionReader):
    """Test getting motion data by ID.

    This test verifies that the get_motion_clip_by_id method returns
    valid MotionClip instances and that the motion data contains
    expected cutoff ranges and cutoff frames.
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

