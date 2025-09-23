from .base_index_mapping import BaseIndexMapping


class DictIndexMapping(BaseIndexMapping):
    """Index mapping type implemented using Python dictionary for fast lookup
    of one or more primary keys corresponding to a key value.
    """

    def __init__(self,
                 logger_cfg: None | dict = None) -> None:
        """Initialize dictionary index mapping.

        Args:
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name. Defaults to None.
        """
        super().__init__(logger_cfg=logger_cfg)
        self._mapping = dict()

    async def keys(self) -> set[str]:
        """Get all keys.

        Returns:
            set[str]:
                Set of all keys.
        """
        keys = self._mapping.keys()
        keys_set = set(keys)
        return keys_set

    async def to_dict(self) -> dict[str, set[int]]:
        """Copy mapping to a dictionary and return.

        Returns:
            dict[str, set[int]]:
                Dictionary representation of the mapping.
        """
        ret_mapping = dict()
        for key in await self.keys():
            value_set = await self.get_items(key)
            ret_mapping[key] = value_set
        return ret_mapping

    async def get_items(self, key: str) -> set[int]:
        """Get the set corresponding to the key.

        Args:
            key (str):
                Key to look up.

        Returns:
            set[int]:
                Set corresponding to the key.

        Raises:
            KeyError:
                Raised when the key is not found in the mapping.
        """
        return self._mapping[key]

    async def set_item(self, key: str, value: set[int]) -> None:
        """Set the set of all primary keys corresponding to a key.

        Args:
            key (str):
                Key to set.
            value (set[int]):
                Set of primary keys to associate with the key.
        """
        value_int_set = set(map(int, value))
        self._mapping[key] = value_int_set

    async def add_item(self, key: str, value: int) -> None:
        """Add a mapped object to the set corresponding to the key.

        Args:
            key (str):
                Key to add to.
            value (int):
                Primary key value to add to the set.
        """
        if key not in self._mapping:
            self._mapping[key] = set()
        self._mapping[key].add(int(value))
