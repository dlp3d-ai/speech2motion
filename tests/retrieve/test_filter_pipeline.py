import logging
import os
from typing import TYPE_CHECKING

import pytest

from speech2motion.filters.builder import build_filter
from speech2motion.index.dict_index_mapping import DictIndexMapping
from speech2motion.io.meta.sqlite_meta_reader import SQLiteMetaReader
from speech2motion.retrieve.filter_pipeline import filter_pipeline_retrieve
from speech2motion.utils.log import setup_logger

if TYPE_CHECKING:
    from speech2motion.data_structures.motion_record import MotionRecord

LOGGER_CFG = dict(
    logger_name='test_filter_pipeline',
    file_level=logging.DEBUG,
    console_level=logging.INFO,
    logger_path='logs/pytest.log',
)
SQLITE_PATH = 'data/motion_database.db'
SQL_JOIN_CMD_PATH = 'configs/3dac_sql_join.sql'

os.makedirs('logs', exist_ok=True)


def is_sqlite_available() -> bool:
    """Check if SQLite database file is available for testing.

    Returns:
        bool: True if SQLite database file exists, False otherwise.
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
async def test_retrieve():
    """Test the filter pipeline retrieve functionality with various scenarios.

    This test verifies the filter pipeline's ability to:
    1. Retrieve all motion record candidates
    2. Return a limited number of candidates
    3. Return a single unique candidate
    4. Handle empty motion records input

    The test uses SQLiteMetaReader to load motion records and applies
    AvatarFilter and RandomFilter in sequence to filter the results.
    """
    logger = setup_logger(**LOGGER_CFG)
    meta_reader = SQLiteMetaReader(
        sqlite_path=SQLITE_PATH,
        sqlite_join_cmd_path=SQL_JOIN_CMD_PATH,
        logger_cfg=LOGGER_CFG
    )
    ids = await meta_reader.get_ids()
    motion_records: dict[int, MotionRecord] = dict()
    mapping = DictIndexMapping(
        logger_cfg=LOGGER_CFG
    )
    for id in ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records[id] = motion_record
        await mapping.add_item(motion_record.avatar_name, id)
    logger.info(f'Loaded {len(motion_records)} motion records.')
    avatar_names = await mapping.keys()
    avatar_filter = build_filter(dict(
        type='AvatarFilter',
        mapping=mapping,
        logger_cfg=LOGGER_CFG,
    ))
    random_filter = build_filter(dict(
        type='RandomFilter',
        logger_cfg=LOGGER_CFG,
    ))
    filter_pipeline = [avatar_filter, random_filter]
    # Return all candidates
    pipeline_input = dict(
        motion_records=motion_records,
        avatar=next(iter(avatar_names)),
        return_length=None,
    )
    filtered_motion_records = await filter_pipeline_retrieve(
        filter_pipeline, pipeline_input, return_candidates=True)
    assert len(filtered_motion_records) > 0
    # Return partial candidates
    pipeline_input['return_length'] = 2
    filtered_motion_records = await filter_pipeline_retrieve(
        filter_pipeline, pipeline_input, return_candidates=True)
    assert len(filtered_motion_records) == 2
    # Return unique candidate
    pipeline_input.pop('return_length')
    filtered_motion_records = await filter_pipeline_retrieve(
        filter_pipeline, pipeline_input, return_candidates=False)
    assert len(filtered_motion_records) == 1
    # Test with empty dictionary input
    pipeline_input['motion_records'] = dict()
    filtered_motion_records = await filter_pipeline_retrieve(
        filter_pipeline, pipeline_input, return_candidates=True)
    assert len(filtered_motion_records) == 0
