import logging
import os

import pytest

from speech2motion.data_structures.motion_record import MotionRecord, MotionRecordType
from speech2motion.filters.builder import build_filter
from speech2motion.filters.type_filter import TypeFilter
from speech2motion.index.dict_index_mapping import DictIndexMapping
from speech2motion.io.meta.sqlite_meta_reader import SQLiteMetaReader
from speech2motion.utils.log import setup_logger

LOGGER_CFG = dict(
    logger_name='test_type_filter',
    file_level=logging.DEBUG,
    console_level=logging.INFO,
    logger_path='logs/pytest.log',
)
SQLITE_PATH = 'data/motion_database.db'
SQL_JOIN_CMD_PATH = 'configs/3dac_sql_join.sql'
N_TYPES = 3

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

def test_build():
    """Test the construction of TypeFilter.

    This test verifies that the build_filter function can correctly
    create a TypeFilter instance from configuration.
    """
    cfg = dict(
        type='TypeFilter',
        mapping=DictIndexMapping(),
    )
    filter = build_filter(cfg)
    assert isinstance(filter, TypeFilter)

@pytest.mark.asyncio
async def test_filter_type():
    """Test filtering motion records by type.

    This test verifies the TypeFilter's ability to:
    1. Filter motion records by different MotionRecordType values
    2. Handle invalid type inputs with proper error handling
    3. Return empty results for non-existent types
    4. Handle empty input dictionaries

    The test loads motion records from SQLite database and tests
    filtering with various motion record types including IDLE_LONG,
    MOTION_KEYWORD, SPEECH_KEYWORD, LOOPABLE, and RANDOM.
    """
    logger = setup_logger(**LOGGER_CFG)
    meta_reader = SQLiteMetaReader(
        sqlite_path=SQLITE_PATH,
        sqlite_join_cmd_path=SQL_JOIN_CMD_PATH,
        logger_cfg=LOGGER_CFG
    )
    ids = await meta_reader.get_ids()
    motion_records: dict[int, MotionRecord] = dict()
    type_set = set()
    mapping = DictIndexMapping(
        logger_cfg=LOGGER_CFG)

    for id in ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records[id] = motion_record
        if motion_record.is_idle_long:
            type_set.add(MotionRecordType.IDLE_LONG)
            await mapping.add_item(MotionRecordType.IDLE_LONG.value, id)
        if motion_record.is_motion_keyword():
            type_set.add(MotionRecordType.MOTION_KEYWORD)
            await mapping.add_item(MotionRecordType.MOTION_KEYWORD.value, id)
        if motion_record.is_speech_keyword():
            type_set.add(MotionRecordType.SPEECH_KEYWORD)
            await mapping.add_item(MotionRecordType.SPEECH_KEYWORD.value, id)
        if motion_record.is_loopable():
            type_set.add(MotionRecordType.LOOPABLE)
            await mapping.add_item(MotionRecordType.LOOPABLE.value, id)
        if motion_record.is_random():
            type_set.add(MotionRecordType.RANDOM)
            await mapping.add_item(MotionRecordType.RANDOM.value, id)
        if len(type_set) >= N_TYPES:
            break

    logger.info(f'type_set={type_set}')
    logger.info(f'mapping.keys()={await mapping.keys()}')

    if len(type_set) < N_TYPES:
        msg = f'In {len(ids)} motion records, ' + \
            f'found less than {N_TYPES} different motion record types.'
        raise ValueError(msg)

    type_filter = TypeFilter(mapping=mapping,
        logger_cfg=LOGGER_CFG)

    for motion_record_type in type_set:
        filtered_motion_records = await type_filter.filter(
            motion_records, motion_record_type
        )
        assert len(filtered_motion_records) > 0
        assert len(filtered_motion_records) <= len(motion_records)

    with pytest.raises(TypeError):
        filtered_motion_records = await type_filter.filter(
            motion_records, 'idle_long'
        )
    # find an enum of MotionRecordType that is not in type_set
    if len(set(MotionRecordType) - type_set) > 0:
        non_exist_type = next(iter(set(MotionRecordType) - type_set))
        filtered_motion_records = await type_filter.filter(
            motion_records, non_exist_type
        )
        assert len(filtered_motion_records) == 0
    # Test with empty dictionary input
    filtered_motion_records = await type_filter.filter(
        dict(), motion_record_type)
    assert len(filtered_motion_records) == 0

@pytest.mark.asyncio
async def test_select_one_type():
    """Test selecting one motion record from type filter.

    This test verifies the TypeFilter's select_one method ability to:
    1. Select a single motion record of a specific type
    2. Validate the selected record matches the expected type
    3. Handle invalid type inputs with proper error handling
    4. Return None for non-existent types
    5. Handle empty input dictionaries

    The test loads motion records from SQLite database and tests
    selection with various motion record types, ensuring the selected
    record has the correct type properties.
    """
    logger = setup_logger(**LOGGER_CFG)
    meta_reader = SQLiteMetaReader(
        sqlite_path=SQLITE_PATH,
        sqlite_join_cmd_path=SQL_JOIN_CMD_PATH,
        logger_cfg=LOGGER_CFG
    )
    ids = await meta_reader.get_ids()
    motion_records: dict[int, MotionRecord] = dict()
    type_set = set()
    mapping = DictIndexMapping(
        logger_cfg=LOGGER_CFG)

    for id in ids:
        motion_record = await meta_reader.get_motion_record_by_id(id)
        motion_records[id] = motion_record
        if motion_record.is_idle_long:
            type_set.add(MotionRecordType.IDLE_LONG)
            await mapping.add_item(MotionRecordType.IDLE_LONG.value, id)
        if motion_record.is_motion_keyword():
            type_set.add(MotionRecordType.MOTION_KEYWORD)
            await mapping.add_item(MotionRecordType.MOTION_KEYWORD.value, id)
        if motion_record.is_speech_keyword():
            type_set.add(MotionRecordType.SPEECH_KEYWORD)
            await mapping.add_item(MotionRecordType.SPEECH_KEYWORD.value, id)
        if motion_record.is_loopable():
            type_set.add(MotionRecordType.LOOPABLE)
            await mapping.add_item(MotionRecordType.LOOPABLE.value, id)
        if motion_record.is_random():
            type_set.add(MotionRecordType.RANDOM)
            await mapping.add_item(MotionRecordType.RANDOM.value, id)
        if len(type_set) >= N_TYPES:
            break

    logger.info(f'type_set={type_set}')
    logger.info(f'mapping.keys()={await mapping.keys()}')

    if len(type_set) < N_TYPES:
        msg = f'In {len(ids)} motion records, ' + \
            f'found less than {N_TYPES} different motion record types.'
        raise ValueError(msg)

    type_filter = TypeFilter(mapping=mapping,
        logger_cfg=LOGGER_CFG)

    for motion_record_type in type_set:
        selected_motion_record = await type_filter.select_one(
            motion_records, motion_record_type
        )
        assert selected_motion_record is not None
        assert isinstance(selected_motion_record, MotionRecord)
        if motion_record_type == MotionRecordType.IDLE_LONG:
            assert selected_motion_record.is_idle_long
        elif motion_record_type == MotionRecordType.MOTION_KEYWORD:
            assert selected_motion_record.is_motion_keyword()
        elif motion_record_type == MotionRecordType.SPEECH_KEYWORD:
            assert selected_motion_record.is_speech_keyword()
        elif motion_record_type == MotionRecordType.LOOPABLE:
            assert selected_motion_record.is_loopable()
        elif motion_record_type == MotionRecordType.RANDOM:
            assert selected_motion_record.is_random()

    with pytest.raises(TypeError):
        selected_motion_record = await type_filter.select_one(
            motion_records, 'idle_long'
        )
    # find an enum of MotionRecordType that is not in type_set
    if len(set(MotionRecordType) - type_set) > 0:
        non_exist_type = next(iter(set(MotionRecordType) - type_set))
        logger.info(f'non_exist_type={non_exist_type}')
        selected_motion_record = await type_filter.select_one(
            motion_records, non_exist_type
        )
        assert selected_motion_record is None
    # Test with empty dictionary input
    selected_motion_record = await type_filter.select_one(
        dict(), motion_record_type)
    assert selected_motion_record is None
