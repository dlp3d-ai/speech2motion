import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ...utils.io import load_npz
from .base_motion_reader import BaseMotionReader


class FilesystemMotionReader(BaseMotionReader):
    """Motion reader that loads motion data from filesystem files.

    This class provides functionality to read motion data from NPZ files
    stored on the local filesystem. It supports concurrent loading using
    thread pool executors and handles blendshape animation data.
    """

    def __init__(self,
                 file_paths: dict[int, str],
                 root_dir: str | None = None,
                 version: str | None = None,
                 max_workers: int = 2,
                 thread_pool_executor: ThreadPoolExecutor | None = None,
                 float_dtype: np.floating = np.float16,
                 blendshape_names: list[str] | str | None = None,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the filesystem motion reader.

        Args:
            file_paths (dict[int, str]):
                A dictionary mapping IDs to filesystem file paths.
                Binary files will be read from the values.
            root_dir (str | None, optional):
                Root directory path for relative file paths.
                If provided, all file paths will be resolved relative
                to this directory. If None, file paths are used as-is.
                Defaults to None.
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
        """
        super().__init__(float_dtype=float_dtype,
                         blendshape_names=blendshape_names,
                         logger_cfg=logger_cfg)
        if version is None:
            self.version = str(uuid.uuid4())
        else:
            self.version = version
        self.file_paths = file_paths
        self.root_dir = root_dir
        self.permanent_executor = thread_pool_executor \
            if thread_pool_executor is not None \
            else ThreadPoolExecutor(max_workers=max_workers)
        self.permanent_executor_external = True \
            if thread_pool_executor is not None \
            else False

    def __del__(self) -> None:
        """"""
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
        if self.root_dir is not None:
            file_path = os.path.join(self.root_dir, file_path)
        ret_dict = await loop.run_in_executor(
            thread_pool_executor,
            load_npz, file_path, self.float_dtype)
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
                    msg = 'Blendshape animation frame count ' +\
                        f'{src_bs_values.shape[0]} is less than bone animation ' +\
                        f'frame count {blendshape_values.shape[0]} for ID {id}, ' +\
                        'will fill with 0 values.'
                    self.logger.warning(msg)
                    n_frames_actual = src_bs_values.shape[0]
                elif src_bs_values.shape[0] > blendshape_values.shape[0]:
                    msg = 'Blendshape animation frame count ' +\
                        f'{src_bs_values.shape[0]} is greater than bone animation ' +\
                        f'frame count {blendshape_values.shape[0]} for ID {id}, ' +\
                        f'will only use the first {src_bs_values.shape[0]} frames.'
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
