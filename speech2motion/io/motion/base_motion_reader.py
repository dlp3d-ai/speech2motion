import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ...data_structures.motion_clip import MotionClip
from ...utils.super import Super


class BaseMotionReader(Super, ABC):
    """Abstract base class for motion data readers.

    This class defines the interface for reading motion data from various
    sources. It provides common functionality for handling blendshape
    animations and data type conversion.
    """

    def __init__(self,
                 float_dtype: np.floating = np.float16,
                 blendshape_names: list[str] | str | None = None,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the base motion reader.

        Args:
            float_dtype (np.floating, optional):
                Data type of numpy floating point arrays in returned
                motion data. Defaults to np.float16.
            blendshape_names (list[str] | str | None, optional):
                List of names of high-priority blendshape animations
                bound to motion clips. The MotionClip's blendshape_names
                attribute will use this parameter's value, and 0 values
                will be used as placeholders even if there is no
                corresponding animation in the motion data.
                If a string is provided, it will be treated as a file path
                to load the names from JSON. If None, blendshape animation
                will not be used. Defaults to None.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed
                description. Logger name will use the class name.
                Defaults to None.

        Raises:
            ValueError: If blendshape_names has an invalid type.
        """
        self.float_dtype = float_dtype
        ABC.__init__(self)
        Super.__init__(self, logger_cfg)
        if isinstance(blendshape_names, str):
            # open with utf-8 encoding to avoid UnicodeDecodeError
            with open(blendshape_names, encoding='utf-8') as f:
                self.blendshape_names = json.load(f)
        elif blendshape_names is None:
            self.blendshape_names = None
        elif isinstance(blendshape_names, list):
            self.blendshape_names = blendshape_names
        else:
            msg = f'Invalid type for blendshape_names: {type(blendshape_names)}'
            self.logger.error(msg)
            raise ValueError(msg)

    @abstractmethod
    async def get_version(self) -> str:
        """Get the version of all motions in the motion database.

        Returns:
            str: Version string.
        """
        pass

    @abstractmethod
    async def get_ids(self) -> list[int]:
        """Get all motion IDs in the motion database.

        Returns:
            list[int]: List of motion IDs.
        """
        pass

    @abstractmethod
    async def get_motion_dict_by_id(self,
                              id: int,
                              thread_pool_executor: ThreadPoolExecutor | None = None
                              ) -> dict:
        """Get motion data by ID.

        Args:
            id (int):
                Motion record ID.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor.
                If None, the permanent thread pool executor from
                constructor will be used. Defaults to None.

        Returns:
            dict: Motion data dictionary.
        """
        pass

    async def get_motion_clip_by_id(
            self,
            id: int,
            thread_pool_executor: ThreadPoolExecutor | None = None) -> MotionClip:
        """Get MotionClip instance by ID.

        Args:
            id (int):
                Motion record ID.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor.
                If None, the permanent thread pool executor from
                constructor will be used. Defaults to None.

        Returns:
            MotionClip: MotionClip instance.
        """
        motion_dict = await self.get_motion_dict_by_id(id, thread_pool_executor)
        motion_clip = MotionClip.from_dict(motion_dict, logger_cfg=self.logger_cfg)
        return motion_clip
