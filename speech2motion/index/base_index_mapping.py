from abc import ABC, abstractmethod

from ..utils.super import Super


class BaseIndexMapping(Super, ABC):
    """Base class for index mapping types used for fast lookup of one or more
    primary keys corresponding to a key value.
    """

    def __init__(self,
                 logger_cfg: None | dict = None) -> None:
        """Initialize base index mapping.

        Args:
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name. Defaults to None.
        """
        Super.__init__(self, logger_cfg=logger_cfg)

    @abstractmethod
    async def keys(self) -> set[str]:
        """Get all keys.

        Returns:
            set[str]:
                Set of all keys.
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, set[int]]:
        """Copy mapping to a dictionary and return.

        Returns:
            dict[str, set[int]]:
                Dictionary representation of the mapping.
        """
        pass

    @abstractmethod
    async def get_items(self, key: str) -> set[int]:
        """Get the set corresponding to the key.

        Args:
            key (str):
                Key to look up.

        Returns:
            set[int]:
                Set corresponding to the key.
        """
        pass

    @abstractmethod
    async def set_item(self, key: str, value: set[int]) -> None:
        """Set the set of all primary keys corresponding to a key.

        Args:
            key (str):
                Key to set.
            value (set[int]):
                Set of primary keys to associate with the key.
        """
        pass

    @abstractmethod
    async def add_item(self, key: str, value: int) -> None:
        """Add a mapped object to the set corresponding to the key.

        Args:
            key (str):
                Key to add to.
            value (int):
                Primary key value to add to the set.
        """
        pass

