import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ...data_structures.restpose import Restpose
from .base_restpose_reader import BaseRestposeReader


class FilesystemRestposeReader(BaseRestposeReader):
    """Filesystem-based restpose data reader.

    This class implements restpose data reading from the local filesystem.
    It provides functionality to retrieve restpose data from local files
    using asynchronous operations with thread pool execution.
    """

    def __init__(self,
                 file_paths: dict[str, str],
                 root_dir: str | None = None,
                 max_workers: int = 2,
                 thread_pool_executor: ThreadPoolExecutor | None = None,
                 float_dtype: np.floating = np.float32,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the FilesystemRestposeReader.

        Args:
            file_paths (dict[str, str]):
                Dictionary mapping restpose names to file paths.
                Files will be read from the values.
            root_dir (str | None, optional):
                Root directory for file paths. If provided, file paths will be
                joined with this root directory. Defaults to None.
            max_workers (int, optional):
                Maximum number of worker threads. Defaults to 2.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor.
                If None, a new thread pool executor will be created based on
                max_workers. Defaults to None.
            float_dtype (np.floating, optional):
                Data type for numpy floating point arrays in returned Restpose.
                Defaults to np.float32. Avoid using np.float16 as np.linalg
                does not support np.float16.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        super().__init__(logger_cfg)
        self.float_dtype = float_dtype
        self.version = 'one_version'
        self.file_paths = file_paths
        self.root_dir = root_dir
        self.executor = thread_pool_executor \
            if thread_pool_executor is not None \
            else ThreadPoolExecutor(max_workers=max_workers)
        self.executor_external = True \
            if thread_pool_executor is not None \
            else False

    def __del__(self) -> None:
        """Destructor, cleanup thread pool executor."""
        if not self.executor_external:
            self.executor.shutdown(wait=True)

    async def get_version(self) -> str:
        """Get the version of all restposes in the library.

        Returns:
            str: Version string.
        """
        return self.version

    async def get_restpose_names(self) -> list[str]:
        """Get all restpose names in the library.

        Returns:
            list[str]: List of restpose names.
        """
        return list(self.file_paths.keys())

    async def get_restpose_by_name(self,
                                   name: str) -> Restpose:
        """Get restpose data by name.

        Args:
            name (str):
                Restpose name.

        Returns:
            Restpose: Restpose data.
        """
        file_path = self.file_paths.get(name, None)
        if file_path is None:
            msg = f'No restpose data record found for name={name}.'
            self.logger.error(msg)
            raise KeyError(msg)
        if self.root_dir is not None:
            file_path = os.path.join(self.root_dir, file_path)
        loop = asyncio.get_running_loop()
        restpose = await loop.run_in_executor(
            self.executor,
            Restpose.from_npz,
            file_path,
            name,
            self.float_dtype,
            self.logger_cfg
        )
        return restpose
