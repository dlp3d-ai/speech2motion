from abc import ABC, abstractmethod

from ...data_structures.restpose import Restpose
from ...utils.super import Super


class BaseRestposeReader(Super, ABC):
    """Base class for restpose data readers.

    This abstract base class defines the interface for reading restpose data
    from various sources. It provides common functionality for version
    management, restpose name retrieval, and restpose data access.
    """

    def __init__(self,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the BaseRestposeReader.

        Args:
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        ABC.__init__(self)
        Super.__init__(self, logger_cfg)

    @abstractmethod
    async def get_version(self) -> str:
        """Get the version of all restposes in the library.

        Returns:
            str: Version string.
        """
        pass

    @abstractmethod
    async def get_restpose_names(self) -> list[str]:
        """Get all restpose names in the library.

        Returns:
            list[str]: List of restpose names.
        """
        pass

    @abstractmethod
    async def get_restpose_by_name(self,
                                   name: str) -> Restpose:
        """Get restpose data by name.

        Args:
            name (str):
                Restpose name.

        Returns:
            Restpose: Restpose data.
        """
        pass
