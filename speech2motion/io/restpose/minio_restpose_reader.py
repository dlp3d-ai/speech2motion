import asyncio
import io
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import overload

import numpy as np

from ...data_structures.restpose import Restpose
from .base_restpose_reader import BaseRestposeReader

try:
    from minio import Minio
    minio_installed = True
except (ImportError, ModuleNotFoundError):
    minio_installed = False
    import_traceback_str = traceback.format_exc()


class MinioRestposeReader(BaseRestposeReader):
    """MinIO-based restpose data reader.

    This class implements restpose data reading from MinIO object storage.
    It provides functionality to retrieve restpose data from MinIO buckets
    using asynchronous operations with thread pool execution.
    """

    def __init__(self,
                 endpoint: str,
                 access_key: str,
                 secret_key: str,
                 file_paths: dict[str, str],
                 bucket_name: str = '3dac',
                 max_workers: int = 2,
                 thread_pool_executor: ThreadPoolExecutor | None = None,
                 float_dtype: np.floating = np.float32,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the MinioRestposeReader.

        Args:
            endpoint (str):
                MinIO server endpoint.
            access_key (str):
                MinIO access key.
            secret_key (str):
                MinIO secret key.
            file_paths (dict[str, str]):
                Dictionary mapping restpose names to MinIO file paths.
                Binary files will be read from the values.
            bucket_name (str, optional):
                MinIO bucket name. Defaults to '3dac'.
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
        if not minio_installed:
            msg = 'MinioRestposeReader requires minio to be installed.' +\
                f'Traceback:\n{import_traceback_str}'
            self.logger.error(msg)
            raise ImportError(msg)
        self.float_dtype = float_dtype
        self.version = 'one_version'
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.file_paths = file_paths
        self.minio_client = Minio(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=False
        )
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
        oss_path = self.file_paths.get(name, None)
        if oss_path is None:
            msg = f'No restpose data record found for name={name}.'
            self.logger.error(msg)
            raise KeyError(msg)
        loop = asyncio.get_running_loop()
        npz_io = await loop.run_in_executor(
            self.executor,
            self._get_bytes_io,
            self.bucket_name,
            oss_path
        )
        restpose = await loop.run_in_executor(
            self.executor,
            Restpose.from_npz,
            npz_io,
            name,
            self.float_dtype,
            self.logger_cfg
        )
        return restpose

    @overload
    def _get_bytes_io(self, full_path: str) -> io.BytesIO:
        """Get object from MinIO server as io.BytesIO.

        Args:
            full_path (str):
                Object path in format 'bucket_name/object_name'.

        Returns:
            io.BytesIO: Object as io.BytesIO.
        """
        ...

    @overload
    def _get_bytes_io(self, bucket_name: str, object_name: str) -> io.BytesIO:
        """Get object from MinIO server as io.BytesIO.

        Args:
            bucket_name (str): Bucket name.
            object_name (str): Object name.

        Returns:
            io.BytesIO: Object as io.BytesIO.
        """
        ...

    def _get_bytes_io(self,
                     path_0: str,
                     path_1: str | None = None) -> io.BytesIO:
        """Get object from MinIO server as io.BytesIO.

        Args:
            path_0 (str):
                Object path in format 'bucket_name/object_name'.
            path_1 (str | None, optional):
                Object name.
                If not provided, path_0 will be split into bucket name and
                object name.

        Returns:
            io.BytesIO: Object as io.BytesIO.
        """
        bucket_name, object_name = self.__class__._convert_paths(path_0, path_1)
        minio_resp = self.minio_client.get_object(
            bucket_name=bucket_name, object_name=object_name)
        oss_io = io.BytesIO()
        for data in minio_resp.stream(amt=1024 * 1024):
            oss_io.write(data)
        oss_io.seek(0)
        return oss_io

    @staticmethod
    def _convert_paths(
            path_0: str,
            path_1: str | None = None) -> tuple[str, str]:
        """Convert paths to bucket name and object name.

        Args:
            path_0 (str):
                Path.
            path_1 (str | None, optional):
                Object name.

        Returns:
            tuple[str, str]: Tuple of (bucket_name, object_name).
        """
        if path_1 is None:
            bucket_name, object_name = path_0.split('/', 1)
        else:
            bucket_name = path_0
            object_name = path_1
        return bucket_name, object_name
