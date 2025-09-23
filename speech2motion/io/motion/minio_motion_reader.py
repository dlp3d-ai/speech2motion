import asyncio
import io
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import overload

import numpy as np

from ...utils.io import load_npz
from .base_motion_reader import BaseMotionReader

try:
    from minio import Minio
    minio_installed = True
except (ImportError, ModuleNotFoundError):
    minio_installed = False
    import_traceback_str = traceback.format_exc()


class MinioMotionReader(BaseMotionReader):
    """Motion reader that loads motion data from MinIO object storage.

    This class provides functionality for reading motion data from MinIO
    object storage, supporting asynchronous operations and thread pool
    execution for efficient data loading.
    """

    def __init__(self,
                 endpoint: str,
                 access_key: str,
                 secret_key: str,
                 file_paths: dict[int, str],
                 bucket_name: str = '3dac',
                 version: str | None = None,
                 max_workers: int = 2,
                 thread_pool_executor: ThreadPoolExecutor | None = None,
                 float_dtype: np.floating = np.float16,
                 blendshape_names: list[str] | str | None = None,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the MinIO motion reader.

        Args:
            endpoint (str):
                MinIO endpoint URL.
            access_key (str):
                MinIO access key.
            secret_key (str):
                MinIO secret key.
            file_paths (dict[int, str]):
                Dictionary mapping IDs to MinIO object paths.
                Binary files will be read from the values.
            bucket_name (str, optional):
                MinIO bucket name. Defaults to '3dac'.
            version (str | None, optional):
                Version of the motion database binary files.
                If None, a random version will be generated.
                Defaults to None.
            max_workers (int, optional):
                Maximum number of worker threads. Defaults to 2.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor.
                If None, a new thread pool executor will be created
                based on max_workers. Defaults to None.
            float_dtype (np.floating, optional):
                Data type of numpy floating point arrays in returned
                motion data. Defaults to np.float16.
            blendshape_names (list[str] | str | None, optional):
                List of names of high-priority blendshape animations
                bound to motion clips. The MotionClip's blendshape_names
                attribute will use this parameter's value, and 0 values
                will be used as placeholders even if there is no
                corresponding animation in the motion data.
                If None, blendshape animation will not be used.
                Defaults to None.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed
                description. Logger name will use the class name.
                Defaults to None.

        Raises:
            ImportError: If minio is not installed.
        """
        super().__init__(float_dtype=float_dtype,
                         blendshape_names=blendshape_names,
                         logger_cfg=logger_cfg)
        if not minio_installed:
            msg = 'MinioMotionReader requires minio to be installed.' +\
                f'Traceback:\n{import_traceback_str}'
            self.logger.error(msg)
            raise ImportError(msg)
        if version is None:
            self.version = str(uuid.uuid4())
        else:
            self.version = version
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
        self.permanent_executor = thread_pool_executor \
            if thread_pool_executor is not None \
            else ThreadPoolExecutor(max_workers=max_workers)
        self.permanent_executor_external = True \
            if thread_pool_executor is not None \
            else False

    def __del__(self) -> None:
        """Destructor, cleanup thread pool executor."""
        if not self.permanent_executor_external:
            self.permanent_executor.shutdown(wait=True)

    async def get_version(self) -> str:
        """Get the version of all motions in the motion database.

        Returns:
            str: Version string.
        """
        return self.version

    async def get_ids(self) -> list[int]:
        """Get all motion IDs in the motion database.

        Returns:
            list[int]: List of motion IDs.
        """
        return list(map(int, self.file_paths.keys()))

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

        Raises:
            KeyError: If no motion data record exists for the given ID.
        """
        file_path = self.file_paths.get(id, None)
        if file_path is None:
            msg = f'No motion data record found for ID={id}.'
            self.logger.error(msg)
            raise KeyError(msg)
        loop = asyncio.get_running_loop()
        if thread_pool_executor is None:
            thread_pool_executor = self.permanent_executor
        npz_io = await loop.run_in_executor(
            thread_pool_executor,
            self._get_bytes_io, self.bucket_name, file_path)
        ret_dict = await loop.run_in_executor(
            thread_pool_executor,
            load_npz, npz_io, self.float_dtype)
        if self.blendshape_names is None:
            if 'blendshape_names' in ret_dict:
                ret_dict.pop('blendshape_names')
            if 'blendshape_values' in ret_dict:
                ret_dict.pop('blendshape_values')
        else:
            blendshape_values = np.zeros(
                (ret_dict['len'], len(self.blendshape_names)),
                dtype=self.float_dtype)
            blendshape_names = self.blendshape_names.copy()
            src_bs_names = ret_dict.pop('blendshape_names', None)
            src_bs_values = ret_dict.pop('blendshape_values', None)
            if src_bs_names is not None and src_bs_values is not None:
                if src_bs_values.shape[0] < blendshape_values.shape[0]:
                    msg = ('Blendshape animation frame count ' +
                           f'{src_bs_values.shape[0]} is less than skeleton ' +
                           f'animation frame count {blendshape_values.shape[0]}, ' +
                           f'ID={id}, will fill with 0 values.')
                    self.logger.warning(msg)
                    n_frames_actual = src_bs_values.shape[0]
                elif src_bs_values.shape[0] > blendshape_values.shape[0]:
                    msg = ('Blendshape animation frame count ' +
                           f'{src_bs_values.shape[0]} is greater than skeleton ' +
                           f'animation frame count {blendshape_values.shape[0]}, ' +
                           f'ID={id}, will only use first ' +
                           f'{blendshape_values.shape[0]} frames of values.')
                    self.logger.warning(msg)
                    n_frames_actual = blendshape_values.shape[0]
                else:
                    n_frames_actual = src_bs_values.shape[0]
                src_name_idx_mapping = {
                    name: idx for idx, name in enumerate(src_bs_names)}
                for idx, name in enumerate(blendshape_names):
                    if name in src_name_idx_mapping:
                        src_idx = src_name_idx_mapping[name]
                        blendshape_values[:n_frames_actual, idx] = \
                            src_bs_values[:n_frames_actual, src_idx]
            ret_dict['blendshape_names'] = blendshape_names
            ret_dict['blendshape_values'] = blendshape_values
        ret_dict['motion_record_id'] = id
        return ret_dict

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
                If not provided, path_0 will be split into bucket name
                and object name. Defaults to None.

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
                Path string.
            path_1 (str | None, optional):
                Object name. Defaults to None.

        Returns:
            tuple[str, str]: Tuple of (bucket_name, object_name).
        """
        if path_1 is None:
            bucket_name, object_name = path_0.split('/', 1)
        else:
            bucket_name = path_0
            object_name = path_1
        return bucket_name, object_name
