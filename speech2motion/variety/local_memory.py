import time
from asyncio import Lock
from typing import Any

from .base_memory import BaseMemory, UserNotFoundError


class LocalMemory(BaseMemory):
    """Local implementation for maintaining user memory.

    This class provides a local in-memory implementation of user memory
    management, storing user memories in a dictionary structure with
    thread-safe operations using asyncio locks.
    """

    def __init__(self,
                 memory_duration: float,
                 maintain_interval: float = 10 * 60,
                 expire_time: float = 60 * 60,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the local memory instance.

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
        BaseMemory.__init__(self,
                            memory_duration=memory_duration,
                            maintain_interval=maintain_interval,
                            expire_time=expire_time,
                            logger_cfg=logger_cfg)
        self.memory_dict: dict[str, dict[str, Any]] = dict()

    def __len__(self) -> int:
        """Get the number of users in memory.

        Returns:
            int:
                Number of users currently stored in memory.
        """
        return len(self.memory_dict)

    async def forget_user(self, user_id: str) -> None:
        """Clear all memories for a specific user.

        Args:
            user_id (str):
                User ID.

        Raises:
            UserNotFoundError:
                Raised when the user does not exist in memory.
        """
        try:
            lock = self.memory_dict[user_id]['lock']
        except KeyError as e:
            msg = f'User {user_id} does not exist, cannot clear memory.'
            self.logger.error(msg)
            raise UserNotFoundError(msg) from e
        async with lock:
            self.memory_dict.pop(user_id)

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
        if user_id not in self.memory_dict:
            return False
        lock = self.memory_dict[user_id]['lock']
        async with lock:
            return event_id in self.memory_dict[user_id]['event_id_queue']

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
        if user_id not in self.memory_dict:
            return event_ids
        lock = self.memory_dict[user_id]['lock']
        async with lock:
            ret_list = []
            existing_events = self.memory_dict[user_id]['event_id_queue']
            ret_list.extend(
                event_id for event_id in event_ids if event_id not in existing_events)
            return ret_list

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
        cur_time = time.time()
        if user_id not in self.memory_dict:
            self.memory_dict[user_id] = dict(
                lock=Lock(),
                event_id_queue=list(),
                duration_queue=list(),
                duration_sum=0.0,
                last_update_time=cur_time,
                memory_duration=memory_duration,
            )
        lock = self.memory_dict[user_id]['lock']
        async with lock:
            self.memory_dict[user_id]['event_id_queue'].append(event_id)
            self.memory_dict[user_id]['duration_queue'].append(event_duration)
            self.memory_dict[user_id]['duration_sum'] += event_duration
            self.memory_dict[user_id]['last_update_time'] = cur_time
            self.memory_dict[user_id]['memory_duration'] = memory_duration
            # When removing event memories that exceed memory_duration,
            # keep at least one event memory
            while len(self.memory_dict[user_id]['event_id_queue']) > 1 and\
                    self.memory_dict[user_id]['duration_sum'] > \
                        self.memory_dict[user_id]['memory_duration']:
                first_duration = self.memory_dict[user_id]['duration_queue'].pop(0)
                self.memory_dict[user_id]['event_id_queue'].pop(0)
                self.memory_dict[user_id]['duration_sum'] -= first_duration

    async def _maintain_task(self, current_time: float) -> None:
        """Asynchronous task for memory maintenance.

        Maintenance tasks include:
        1. Recounting to eliminate cumulative errors
        2. Deleting expired users

        Args:
            current_time (float):
                Current time.
        """
        users = list(self.memory_dict.keys())
        for user_id in users:
            try:
                lock = self.memory_dict[user_id]['lock']
            except KeyError:
                msg = f'User {user_id} does not exist, ' +\
                    'may have been manually deleted during maintenance task, ' +\
                    'skipping memory maintenance.'
                self.logger.warning(msg)
                continue
            async with lock:
                last_update_time = self.memory_dict[user_id]['last_update_time']
                if current_time - last_update_time > self.expire_time:
                    self.memory_dict.pop(user_id)
                    self.logger.info(f'User {user_id} memory expired, deleted.')
                    continue
                else:
                    new_duration_sum = sum(self.memory_dict[user_id]['duration_queue'])
                    self.memory_dict[user_id]['duration_sum'] = new_duration_sum
                    # Do not update last_update_time here, otherwise it may cause
                    # external non-updating tasks to continuously change and
                    # cannot be properly deleted

    async def _get_single_user_lock(self, user_id: str) -> Lock:
        """Get lock for a single user.

        Args:
            user_id (str):
                User ID.

        Returns:
            Lock:
                Lock object for the user.

        Raises:
            UserNotFoundError:
                Raised when the user does not exist in memory.
        """
        try:
            lock = self.memory_dict[user_id]['lock']
        except KeyError as e:
            msg = f'User {user_id} does not exist, cannot return lock object.'
            self.logger.error(msg)
            raise UserNotFoundError(msg) from e
        return lock
