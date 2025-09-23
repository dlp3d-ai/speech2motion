import logging
import os

import pytest

from speech2motion.data_structures.interval_tree import (
    IntervalExtendError,
    IntervalOutOfRangeError,
    IntervalTreeNode,
    IntervalUnavailableError,
)

LOGGER_CFG = dict(
    logger_name='test_interval_tree',
    file_level=logging.DEBUG,
    console_level=logging.INFO,
    logger_path='logs/pytest.log',
)

os.makedirs('logs', exist_ok=True)

@pytest.fixture
def interval_tree_node():
    """Fixture to create an IntervalTreeNode instance for testing."""
    return IntervalTreeNode(start=0, end=10, logger_cfg=LOGGER_CFG)

def test_init(interval_tree_node):
    """Test the initialization of IntervalTreeNode."""
    assert interval_tree_node.start == 0
    assert interval_tree_node.end == 10
    assert interval_tree_node.payload is None
    assert interval_tree_node.lchild is None
    assert interval_tree_node.rchild is None

def test_is_leaf(interval_tree_node):
    """Test if the node is a leaf."""
    assert interval_tree_node.is_leaf() is False
    interval_tree_node.payload = "test"
    assert interval_tree_node.is_leaf() is True

def test_set_child(interval_tree_node):
    """Test setting a child node."""
    child_node = IntervalTreeNode(start=0, end=5)
    interval_tree_node.set_child(child_node, 'l')
    assert interval_tree_node.lchild == child_node
    assert child_node.parent == interval_tree_node
    assert child_node.side == 'l'

    with pytest.raises(ValueError):
        interval_tree_node.set_child(child_node, 'invalid_side')

def test_get_next_blank_recursively(interval_tree_node):
    """Test getting the next blank node recursively."""
    top_blank_node = interval_tree_node.get_next_blank_recursively()
    assert top_blank_node is not None
    assert top_blank_node.start == interval_tree_node.start
    assert top_blank_node.end == interval_tree_node.end

    # insert a child and test again
    child_node = IntervalTreeNode(start=1, end=5)
    interval_tree_node.insert_node_recursively(child_node)
    blank = interval_tree_node.get_next_blank_recursively()
    assert blank is not None
    assert blank.start == 0
    assert blank.end == 1

    # test with intervals to skip
    blank = interval_tree_node.get_next_blank_recursively(
        [(0, 1)]
    )
    assert blank is not None
    assert blank.start == 1
    assert blank.end == 5
    blank = interval_tree_node.get_next_blank_recursively(
        [(0, 1), (1, 5)]
    )
    assert blank is not None
    assert blank.start == 5
    assert blank.end == interval_tree_node.end

    # test with leaf
    child_node = IntervalTreeNode(start=0, end=1, payload="test")
    interval_tree_node.insert_node_recursively(child_node)
    blank = interval_tree_node.get_next_blank_recursively()
    assert blank is not None
    assert blank.start == 1
    assert blank.end == 5


def test_insert_node_recursively(interval_tree_node):
    """Test inserting a node recursively."""
    new_node = IntervalTreeNode(start=0, end=2, payload="test")
    interval_tree_node.insert_node_recursively(new_node)

    assert interval_tree_node.lchild is not None
    assert interval_tree_node.lchild.start == 0
    assert interval_tree_node.lchild.end == 2

    # Test inserting out of range
    out_of_range_node = IntervalTreeNode(start=11, end=15)
    with pytest.raises(IntervalOutOfRangeError):
        interval_tree_node.insert_node_recursively(out_of_range_node)

    # Test IntervalUnavailableError
    with pytest.raises(IntervalUnavailableError):
        new_node = IntervalTreeNode(start=0, end=1, payload="test")
        interval_tree_node.insert_node_recursively(new_node)

def test_extend_start(interval_tree_node):
    """Test extending the start of the interval."""
    interval_tree_node.extend_start(-5)
    assert interval_tree_node.start == -5

    with pytest.raises(IntervalExtendError):
        interval_tree_node.extend_start(5)

def test_extend_end(interval_tree_node):
    """Test extending the end of the interval."""
    interval_tree_node.extend_end(15)
    assert interval_tree_node.end == 15

    with pytest.raises(IntervalExtendError):
        interval_tree_node.extend_end(5)

def test_shallow_copy(interval_tree_node):
    """Test shallow copying the node."""
    copy_node = interval_tree_node.shallow_copy()
    assert copy_node.start == interval_tree_node.start
    assert copy_node.end == interval_tree_node.end
    assert copy_node.payload == interval_tree_node.payload
    assert copy_node.lchild is None  # Shallow copy, should reference the same children
