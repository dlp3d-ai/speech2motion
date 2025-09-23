import logging
import os

import pytest
import pytest_asyncio

from speech2motion.data_structures.restpose import Restpose
from speech2motion.io.restpose.builder import build_restpose_reader
from speech2motion.io.restpose.filesystem_restpose_reader import (
    FilesystemRestposeReader,
)

LOGGER_CFG = dict(
    logger_name='test_filesystem_restpose_reader',
    file_level=logging.DEBUG,
    console_level=logging.INFO,
    logger_path='logs/pytest.log',
)
ROOT_DIR = 'data/restpose_npz'
FILE_PATHS = {'KQ-default': 'KQ_default_0326_skeleton.npz'}

os.makedirs('logs', exist_ok=True)

def test_build_restpose_reader():
    """Test building a restpose reader instance.

    This test verifies that the build_restpose_reader function can
    correctly create a FilesystemRestposeReader instance from configuration.
    """
    cfg = {
        'type': 'FilesystemRestposeReader',
        'file_paths': FILE_PATHS,
        'root_dir': ROOT_DIR,
        'logger_cfg': LOGGER_CFG
    }
    restpose_reader = build_restpose_reader(cfg)
    assert isinstance(restpose_reader, FilesystemRestposeReader)

@pytest_asyncio.fixture
async def restpose_reader():
    """Fixture to create a FilesystemRestposeReader instance for testing.

    This fixture creates a FilesystemRestposeReader instance with the
    configured filesystem parameters and provides the reader for testing.

    Yields:
        FilesystemRestposeReader: Restpose reader instance ready for testing.
    """
    restpose_reader = FilesystemRestposeReader(
        root_dir=ROOT_DIR,
        file_paths=FILE_PATHS,
        logger_cfg=LOGGER_CFG
    )
    return restpose_reader

@pytest.mark.asyncio
async def test_get_version(restpose_reader: FilesystemRestposeReader):
    """Test getting the restpose database version.

    This test verifies that the get_version method returns a valid
    version string and that the version is properly initialized.
    """
    version = await restpose_reader.get_version()
    assert isinstance(version, str)  # Ensure return value is string
    assert version != 'not_initialized'  # Ensure version is initialized

@pytest.mark.asyncio
async def test_get_restpose_names(restpose_reader: FilesystemRestposeReader):
    """Test getting all restpose names.

    This test verifies that the get_restpose_names method returns a list of
    restpose names and that there is at least one restpose available.
    """
    ids = await restpose_reader.get_restpose_names()
    assert isinstance(ids, list)
    assert len(ids) > 0  # Ensure there is restpose data

@pytest.mark.asyncio
async def test_get_restpose_by_name(restpose_reader: FilesystemRestposeReader):
    """Test getting restpose data by name.

    This test verifies that the get_restpose_by_name method returns
    valid Restpose instances for all available restpose names.
    """
    names = await restpose_reader.get_restpose_names()
    for name in names:
        restpose = await restpose_reader.get_restpose_by_name(name)
        assert isinstance(restpose, Restpose)

