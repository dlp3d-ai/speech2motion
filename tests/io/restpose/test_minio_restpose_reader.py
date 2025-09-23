import logging
import os
import socket

import pytest
import pytest_asyncio

from speech2motion.data_structures.restpose import Restpose
from speech2motion.io.restpose.builder import build_restpose_reader
from speech2motion.io.restpose.minio_restpose_reader import MinioRestposeReader

LOGGER_CFG = dict(
    logger_name='test_minio_restpose_reader',
    file_level=logging.DEBUG,
    console_level=logging.INFO,
    logger_path='logs/pytest.log',
)
# Load database connection information from environment variables
MINIO_ENDPOINT = os.getenv('OSS_ENDPOINT', None)
MINIO_ACCESS_KEY = os.getenv('OSS_ACCESS_KEY', '')
MINIO_SECRET_KEY = os.getenv('OSS_SECRET_KEY', '')
MINIO_BUCKET = '3dac'
FILE_PATHS = {'KQ-default': 'restpose_npz/KQ_default_0326_skeleton.npz'}

os.makedirs('logs', exist_ok=True)

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
    not is_minio_available(),
    reason="Minio server is not available for testing"
)

def test_build_motion_reader():
    """Test building a restpose reader instance.

    Tests the build_restpose_reader function to ensure it correctly creates
    a MinioRestposeReader instance from configuration.
    """
    cfg = {
        'type': 'MinioRestposeReader',
        'endpoint': MINIO_ENDPOINT,
        'access_key': MINIO_ACCESS_KEY,
        'secret_key': MINIO_SECRET_KEY,
        'bucket_name': MINIO_BUCKET,
        'file_paths': FILE_PATHS,
        'logger_cfg': LOGGER_CFG
    }
    restpose_reader = build_restpose_reader(cfg)
    assert isinstance(restpose_reader, MinioRestposeReader)

@pytest_asyncio.fixture
async def restpose_reader():
    """Fixture to create a MinioRestposeReader instance for testing.

    Returns:
        MinioRestposeReader:
            Configured MinioRestposeReader instance for testing.
    """
    restpose_reader = MinioRestposeReader(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        bucket_name=MINIO_BUCKET,
        file_paths=FILE_PATHS,
        logger_cfg=LOGGER_CFG
    )
    return restpose_reader

@pytest.mark.asyncio
async def test_get_version(restpose_reader: MinioRestposeReader):
    """Test getting motion database version.

    Args:
        restpose_reader (MinioRestposeReader):
            MinioRestposeReader instance for testing.
    """
    version = await restpose_reader.get_version()
    assert isinstance(version, str)  # Ensure return value is string
    assert version != 'not_initialized'  # Ensure version is initialized

@pytest.mark.asyncio
async def test_get_ids(restpose_reader: MinioRestposeReader):
    """Test getting all Restpose names.

    Args:
        restpose_reader (MinioRestposeReader):
            MinioRestposeReader instance for testing.
    """
    ids = await restpose_reader.get_restpose_names()
    assert isinstance(ids, list)
    assert len(ids) > 0  # Ensure there is Restpose data

@pytest.mark.asyncio
async def test_get_restpose_by_name(restpose_reader: MinioRestposeReader):
    """Test getting Restpose data by name.

    Args:
        restpose_reader (MinioRestposeReader):
            MinioRestposeReader instance for testing.
    """
    names = await restpose_reader.get_restpose_names()
    for name in names:
        restpose = await restpose_reader.get_restpose_by_name(name)
        assert isinstance(restpose, Restpose)

