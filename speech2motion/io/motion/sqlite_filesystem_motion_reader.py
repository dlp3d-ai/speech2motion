import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .filesystem_motion_reader import FilesystemMotionReader

try:
    import aiosqlite
    sqlite_installed = True
except (ImportError, ModuleNotFoundError):
    sqlite_installed = False
    import_traceback_str = traceback.format_exc()


class SQLiteFilesystemMotionReader(FilesystemMotionReader):
    """Motion reader that loads motion data from SQLite database and filesystem files.

    This class extends FilesystemMotionReader to provide functionality for reading
    motion data from SQLite database metadata and corresponding NPZ files stored
    on the local filesystem. It creates a new database connection for each operation
    to ensure thread safety and proper resource management.
    """

    def __init__(self,
                 sqlite_path: str,
                 sqlite_join_cmd_path: str,
                 root_dir: str | None = None,
                 max_workers: int = 4,
                 thread_pool_executor: ThreadPoolExecutor | None = None,
                 motion_record_table_name: str = 'motion_record',
                 enabled_column_name: str = 'enabled',
                 float_dtype: np.floating = np.float16,
                 blendshape_names: list[str] | str | None = None,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the SQLite filesystem motion reader.

        Args:
            sqlite_path (str):
                Path to the SQLite database file containing motion metadata.
            sqlite_join_cmd_path (str):
                Path to the SQL join command file used for querying
                motion record metadata.
            root_dir (str | None, optional):
                Root directory path for relative file paths.
                If provided, all file paths will be resolved relative
                to this directory. If None, file paths are used as-is.
                Defaults to None.
            max_workers (int, optional):
                Maximum number of worker threads. Defaults to 4.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor.
                If None, a new thread pool executor will be created
                based on max_workers. Defaults to None.
            motion_record_table_name (str, optional):
                Name of the motion record table in the database.
                Defaults to 'motion_record'.
            enabled_column_name (str, optional):
                Name of the enabled column in the motion record table.
                Defaults to 'enabled'.
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
            ImportError: If aiosqlite is not installed.
        """
        super().__init__(
            file_paths=dict(),
            root_dir=root_dir,
            version='not_initialized',
            max_workers=max_workers,
            thread_pool_executor=thread_pool_executor,
            float_dtype=float_dtype,
            blendshape_names=blendshape_names,
            logger_cfg=logger_cfg
        )
        if not sqlite_installed:
            msg = 'SQLiteFilesystemMotionReader requires aiosqlite to be installed.' +\
                f'Traceback:\n{import_traceback_str}'
            self.logger.error(msg)
            raise ImportError(msg)
        self.sqlite_path = sqlite_path
        self.sqlite_join_cmd_path = sqlite_join_cmd_path
        with open(self.sqlite_join_cmd_path) as f:
            self.sqlite_join_cmd = f.read().replace('\n', ' ').strip()
        self.motion_record_table_name = motion_record_table_name
        self.enabled_column_name = enabled_column_name


    async def _get_version_from_sqlite(self) -> str:
        """Get the version of motion database metadata.

        Creates a new database connection for this operation to ensure thread
        safety and proper resource management.

        Returns:
            str: Version string.
        """
        async with aiosqlite.connect(self.sqlite_path) as sqlite_connection:
            async with sqlite_connection.cursor() as cursor:
                # Get database file modification time using PRAGMA
                command = "PRAGMA user_version;"
                await cursor.execute(command)
                result = await cursor.fetchall()

        # If user_version is not set (0), use a fallback approach
        if result[0][0] == 0:
            # Use file modification time as version
            file_mtime = os.path.getmtime(self.sqlite_path)
            return str(int(file_mtime))
        else:
            return str(result[0][0])

    async def get_version(self) -> str:
        """Get the version of all motions in the motion database.

        Returns:
            str: Version string.
        """
        version = await self._get_version_from_sqlite()
        self.version = version
        return await super().get_version()

    async def get_ids(self) -> list[int]:
        """Get all motion IDs in the motion database.

        Creates a new database connection for this operation to ensure thread
        safety and proper resource management.

        Returns:
            list[int]: List of motion IDs.
        """
        async with aiosqlite.connect(self.sqlite_path) as sqlite_connection:
            async with sqlite_connection.cursor() as cursor:
                command = f'''
                    SELECT motion_record_id FROM
                    {self.motion_record_table_name}
                    WHERE {self.enabled_column_name} = 1
                '''
                await cursor.execute(command)
                result = await cursor.fetchall()
        return [int(x[0]) for x in result]

    async def get_motion_dict_by_id(
            self,
            id: int,
            thread_pool_executor: ThreadPoolExecutor | None = None) -> dict:
        """Get motion data by ID.

        Creates a new database connection for this operation to ensure thread
        safety and proper resource management.

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
        command = self.sqlite_join_cmd.replace(';', '')
        command += f' WHERE motion_record_id = {id};'
        class_priority = 0
        cutoff_frames = None
        cutoff_ranges = None
        npz_oss_path = None
        # Though connect and disconnect for every call is inefficient,
        # it is safer to ensure the connection is closed and lock is released.
        async with aiosqlite.connect(self.sqlite_path) as sqlite_connection:
            async with sqlite_connection.cursor() as cursor:
                await cursor.execute(command)
                result = await cursor.fetchall()
                if len(result) == 0:
                    msg = f'No motion data record found for ID={id}.'
                    self.logger.error(msg)
                    raise KeyError(msg)
        column_names = [x[0] for x in cursor.description]
        name_idx_mapping = {name: idx for idx, name in enumerate(column_names)}
        enabled_value = bool(result[0][name_idx_mapping[self.enabled_column_name]])
        if not enabled_value:
            msg = f'motion_record_id={id} data is disabled.'
            self.logger.warning(msg)
        npz_oss_path = result[0][name_idx_mapping['npz_oss_path']]
        motion_keyword_id = result[0][name_idx_mapping['motion_keyword_id']]
        speech_keyword_id = result[0][name_idx_mapping['speech_keyword_id']]
        is_random_audio = result[0][name_idx_mapping['is_random_audio']]
        is_idle_long = result[0][name_idx_mapping['is_idle_long']]
        n_frames = result[0][name_idx_mapping['n_frames']]
        if motion_keyword_id is not None or speech_keyword_id is not None:
            class_priority = 2
            cutoff_frames = [
                (0, 0, class_priority), (n_frames - 1, class_priority, 0)]
            cutoff_ranges = None
        elif is_random_audio:
            class_priority = 1
            cutoff_frames_str = result[0][name_idx_mapping['cutoff_frames']]
            if cutoff_frames_str is not None:
                cutoff_frames_list = json.loads(cutoff_frames_str)
                cutoff_frames = [(0, 0, class_priority)]
                for frame_idx in cutoff_frames_list:
                    cutoff_frames.append(
                        (frame_idx, class_priority, class_priority))
                cutoff_frames.append((n_frames - 1, class_priority, 0))
            else:
                cutoff_frames = [
                    (0, 0, class_priority), (n_frames - 1, class_priority, 0)]
            cutoff_ranges_str = result[0][name_idx_mapping['cutoff_ranges']]
            if cutoff_ranges_str is not None:
                cutoff_ranges = list()
                cutoff_ranges_list = json.loads(cutoff_ranges_str)
                for start_idx, end_idx in cutoff_ranges_list:
                    cutoff_ranges.append(
                        (start_idx, end_idx, class_priority, class_priority))
            else:
                cutoff_ranges = None
        elif is_idle_long:
            class_priority = 0
            cutoff_frames = [
                (0, 0, class_priority), (n_frames - 1, class_priority, 0)]
            cutoff_ranges = [(0, n_frames, class_priority, class_priority)]
        else:
            class_priority = 0
            cutoff_frames = None
            cutoff_ranges = None
        self.file_paths[id] = npz_oss_path
        motion_dict = await super().get_motion_dict_by_id(id, thread_pool_executor)
        motion_dict['cutoff_frames'] = cutoff_frames
        motion_dict['cutoff_ranges'] = cutoff_ranges
        return motion_dict

