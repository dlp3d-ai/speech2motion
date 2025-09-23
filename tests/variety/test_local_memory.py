import asyncio
import logging

import pytest

from speech2motion.variety.local_memory import LocalMemory

LOGGER_CFG = dict(
    logger_name='test_local_memory',
    file_level=logging.DEBUG,
    console_level=logging.INFO,
    logger_path='logs/pytest.log',
)

@pytest.mark.asyncio
async def test_forget_user():
    """Test clearing user memory.

    Tests the forget_user method to ensure it correctly removes
    all memories for a specific user from the LocalMemory instance.
    """
    memory = LocalMemory(
        memory_duration=60,
        logger_cfg=LOGGER_CFG,
    )
    user_id = "user_1"
    event_id = 1
    await memory.remember(user_id, event_id, event_duration=5)

    assert len(memory) == 1  # Ensure user memory exists

    await memory.forget_user(user_id)
    assert len(memory) == 0  # Ensure user memory has been cleared

@pytest.mark.asyncio
async def test_recall():
    """Test checking if user has seen a specific event.

    Tests the recall method to ensure it correctly determines whether
    a user has seen a specific event, including memory expiration
    behavior when new events exceed memory duration.
    """
    memory = LocalMemory(
        memory_duration=60,
        logger_cfg=LOGGER_CFG,
    )
    user_id = "user_1"
    event_id = 1
    await memory.remember(user_id, event_id, event_duration=5)

    # User should remember this event
    assert await memory.recall(user_id, event_id) is True
    # User should not remember other events
    assert await memory.recall(user_id, 2) is False

    await memory.remember(user_id, 2, event_duration=120)
    # User should not remember this event
    assert await memory.recall(user_id, 1) is False
    assert await memory.recall(user_id, 2) is True  # User should remember this event

@pytest.mark.asyncio
async def test_recall_filter():
    """Test filtering events that user has not seen.

    Tests the recall_filter method to ensure it correctly returns
    a list of event IDs that the user has not seen, and updates
    the filtered list when new events are remembered.
    """
    memory = LocalMemory(
        memory_duration=60,
        logger_cfg=LOGGER_CFG,
    )
    user_id = "user_1"
    event_ids = [1, 2, 3]
    await memory.remember(user_id, 1, event_duration=5)

    unseen_events = await memory.recall_filter(user_id, event_ids)
    assert unseen_events == [2, 3]  # Events user has not seen

    await memory.remember(user_id, 2, event_duration=120)
    unseen_events = await memory.recall_filter(user_id, event_ids)
    assert unseen_events == [1, 3]  # Events user has not seen

@pytest.mark.asyncio
async def test_maintain_task():
    """Test memory maintenance task.

    Tests the automatic maintenance task that runs periodically to
    clean up expired user memories and recount memory cache to
    eliminate cumulative errors.
    """
    memory = LocalMemory(
        memory_duration=60,
        maintain_interval=1,
        expire_time=2,
        logger_cfg=LOGGER_CFG,
    )
    user_id = "user_1"
    event_id = 1
    await memory.remember(user_id, event_id, event_duration=5)
    await asyncio.sleep(1.01)  # Wait for maintenance task interval
    # Trigger maintenance task through remember
    await memory.remember('user_2', 2, event_duration=5)
    assert await memory.recall(user_id, event_id) is True  # User memory still exists
    await asyncio.sleep(1.01)  # Wait for maintenance task interval
    # Trigger maintenance task through remember
    await memory.remember('user_2', 2, event_duration=5)
    await asyncio.sleep(0.01)  # Wait for maintenance task completion
    # User memory should be expired and cleared
    assert await memory.recall(user_id, event_id) is False
