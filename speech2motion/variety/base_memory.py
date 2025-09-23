import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

from ..utils.super import Super


class UserNotFoundError(Exception):
    """Exception raised when a user is not found in memory.

    This exception is raised when attempting to access or manipulate
    memory data for a user that does not exist in the memory system.
    """
    pass


class BaseMemory(Super, ABC):
    """Base class for maintaining user memory.

    This abstract base class provides the interface for user memory management,
    including remembering events, recalling past events, and maintaining
    memory state with automatic cleanup of expired memories.
    """

    def __init__(self,
                 memory_duration: float,
                 maintain_interval: float = 10 * 60,
                 expire_time: float = 60 * 60,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the base memory instance.

        Args:
            memory_duration (float):
                Duration of memory in seconds.
            maintain_interval (float, optional):
                Interval for memory maintenance in seconds, including
                clearing expired memories and recounting memory cache
                to eliminate cumulative errors. Defaults to 10 minutes.
            expire_time (float, optional):
                Expiration time for memories in seconds.
                Defaults to 60 * 60, memories not updated for one hour
                will be deleted.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed
                description. Logger name will use the class name.
                Defaults to None.
        """
        ABC.__init__(self)
        Super.__init__(self, logger_cfg=logger_cfg)
        self.memory_duration = memory_duration
        self.maintain_interval = maintain_interval
        self.expire_time = expire_time
        self.maintain_tasks: set[asyncio.Task] = set()
        self.last_maintain_time = 0.0

    @abstractmethod
    def __len__(self) -> int:
        """Get the number of users in memory.

        Returns:
            int:
                Number of users currently stored in memory.
        """
        pass

    @abstractmethod
    async def forget_user(self, user_id: str) -> None:
        """Clear all memories for a specific user.

        Args:
            user_id (str):
                User ID.
        """
        pass

    @abstractmethod
    async def recall(self, user_id: str, event_id: int) -> bool:
        """Check if user has seen a specific event.

        Args:
            user_id (str):
                User ID.
            event_id (int):
                Event ID.

        Returns:
            bool:
                Whether the user has seen the event.
        """
        pass

    @abstractmethod
    async def recall_filter(self, user_id: str, event_ids: list[int]) -> list[int]:
        """Use memory instance as filter to return events user has not seen.

        Args:
            user_id (str):
                User ID.
            event_ids (list[int]):
                List of event IDs.

        Returns:
            list[int]:
                List of event IDs that the user has not seen.
        """
        pass

    async def remember(
            self,
            user_id: str,
            event_id: int,
            event_duration: float,
            memory_duration_override: float | None = None) -> None:
        """Make user remember an event with known duration.

        Args:
            user_id (str):
                User ID.
            event_id (int):
                Event ID.
            event_duration (float):
                Event duration in seconds.
            memory_duration_override (float | None, optional):
                Memory duration in seconds. If None, uses
                `memory_duration` parameter. Defaults to None.
        """
        memory_duration = self.memory_duration \
            if memory_duration_override is None \
            else memory_duration_override
        await self._remember(
            user_id,
            event_id,
            event_duration,
            memory_duration)
        cur_time = time.time()
        if cur_time - self.last_maintain_time > self.maintain_interval:
            self.last_maintain_time = cur_time
            task = asyncio.create_task(self._maintain_task(cur_time))
            if len(self.maintain_tasks) != 0:
                msg = 'Unfinished maintenance tasks exist, please check logs.'
                self.logger.warning(msg)
            self.maintain_tasks.add(task)
            task.add_done_callback(lambda _: self.maintain_tasks.remove(task))

    @abstractmethod
    async def _remember(self, user_id: str, event_id: int,
                        event_duration: float,
                        memory_duration: float) -> None:
        """Make user remember an event with known duration.

        Args:
            user_id (str):
                User ID.
            event_id (int):
                Event ID.
            event_duration (float):
                Event duration in seconds.
            memory_duration (float):
                Memory duration in seconds.
        """
        pass

    @abstractmethod
    async def _maintain_task(self, current_time: float) -> None:
        """Asynchronous task for memory maintenance.

        Maintenance tasks include:
        1. Recounting to eliminate cumulative errors
        2. Deleting expired users

        Args:
            current_time (float):
                Current time.
        """
        pass

    @abstractmethod
    async def _get_single_user_lock(self, user_id: str) -> Any:
        """Get lock for a single user.

        Args:
            user_id (str):
                User ID.

        Returns:
            Any:
                Lock object.
        """
        pass
