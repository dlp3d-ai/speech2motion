import pytest

from speech2motion.index.builder import build_index_mapping
from speech2motion.index.dict_index_mapping import DictIndexMapping


def test_build_dict_index_mapping():
    mapping = {
        'type': 'DictIndexMapping',
    }
    index_mapping = build_index_mapping(mapping)
    assert isinstance(index_mapping, DictIndexMapping)


@pytest.fixture
def dict_index_mapping():
    """Fixture for creating a DictIndexMapping instance."""
    return DictIndexMapping()

@pytest.mark.asyncio
async def test_set_item(dict_index_mapping: DictIndexMapping):
    await dict_index_mapping.set_item("key1", {1, 2})
    assert await dict_index_mapping.get_items("key1") == {1, 2}

@pytest.mark.asyncio
async def test_get_items(dict_index_mapping: DictIndexMapping):
    await dict_index_mapping.set_item("key1", {1, 2})
    result = await dict_index_mapping.get_items("key1")
    assert result == {1, 2}

@pytest.mark.asyncio
async def test_add_item(dict_index_mapping: DictIndexMapping):
    await dict_index_mapping.add_item("key1", 1)
    await dict_index_mapping.add_item("key1", 2)
    result = await dict_index_mapping.get_items("key1")
    assert result == {1, 2}

@pytest.mark.asyncio
async def test_keys(dict_index_mapping: DictIndexMapping):
    await dict_index_mapping.set_item("key1", {1})
    await dict_index_mapping.set_item("key2", {2})
    keys = await dict_index_mapping.keys()
    assert keys == {'key1', 'key2'}

@pytest.mark.asyncio
async def test_to_dict(dict_index_mapping: DictIndexMapping):
    await dict_index_mapping.set_item("key1", {1})
    await dict_index_mapping.set_item("key2", {2})
    result = await dict_index_mapping.to_dict()
    assert result == {
        "key1": {1},
        "key2": {2},
    }

@pytest.mark.asyncio
async def test_get_items_key_error(dict_index_mapping: DictIndexMapping):
    with pytest.raises(KeyError):
        await dict_index_mapping.get_items("non_existent_key")
