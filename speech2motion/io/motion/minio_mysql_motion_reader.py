import json
import traceback
from asyncio import Lock
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .minio_motion_reader import MinioMotionReader

try:
    import aiomysql
    mysql_installed = True
except (ImportError, ModuleNotFoundError):
    mysql_installed = False
    import_traceback_str = traceback.format_exc()


class MinioMySQLMotionReader(MinioMotionReader):
    """Motion reader that loads motion data from MinIO and MySQL database.

    This class extends MinioMotionReader to provide functionality for reading
    motion data from MinIO object storage with metadata from MySQL database.
    It supports database connection management and handles motion record
    metadata queries.
    """

    def __init__(self,
                 mysql_host: str,
                 mysql_port: int,
                 mysql_username: str,
                 mysql_password: str,
                 mysql_database: str,
                 mysql_join_cmd_path: str,
                 endpoint: str,
                 access_key: str,
                 secret_key: str,
                 bucket_name: str = '3dac',
                 max_workers: int = 4,
                 thread_pool_executor: ThreadPoolExecutor | None = None,
                 motion_record_table_name: str = 'motion_record',
                 enabled_column_name: str = 'enabled',
                 float_dtype: np.floating = np.float16,
                 blendshape_names: list[str] | str | None = None,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the MinIO MySQL motion reader.

        Args:
            mysql_host (str):
                MySQL server host address.
            mysql_port (int):
                MySQL server port number.
            mysql_username (str):
                MySQL username for authentication.
            mysql_password (str):
                MySQL password for authentication.
            mysql_database (str):
                MySQL database name.
            mysql_join_cmd_path (str):
                Path to the SQL file containing the JOIN query.
            endpoint (str):
                MinIO endpoint URL.
            access_key (str):
                MinIO access key.
            secret_key (str):
                MinIO secret key.
            bucket_name (str, optional):
                MinIO bucket name. Defaults to '3dac'.
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
            ImportError: If aiomysql is not installed.
        """
        super().__init__(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            file_paths=dict(),
            bucket_name=bucket_name,
            version='not_initialized',
            max_workers=max_workers,
            thread_pool_executor=thread_pool_executor,
            float_dtype=float_dtype,
            blendshape_names=blendshape_names,
            logger_cfg=logger_cfg
        )
        if not mysql_installed:
            msg = 'MinioMySQLMotionReader requires aiomysql to be installed.' +\
                f'Traceback:\n{import_traceback_str}'
            self.logger.error(msg)
            raise ImportError(msg)
        self.mysql_host = mysql_host
        self.mysql_port = int(mysql_port)
        self.mysql_username = mysql_username
        self.mysql_password = mysql_password
        self.mysql_database = mysql_database
        self.mysql_join_cmd_path = mysql_join_cmd_path
        with open(self.mysql_join_cmd_path) as f:
            self.mysql_join_cmd = f.read().replace('\n', ' ').strip()
        self.mysql_connection: aiomysql.Connection | None = None
        self.motion_record_table_name = motion_record_table_name
        self.enabled_column_name = enabled_column_name
        self.mysql_connection_lock = Lock()

    async def disconnect(self) -> None:
        """Disconnect from MySQL database.

        Safely closes the MySQL connection and handles any errors that may
        occur during the disconnection process.
        """
        if self.mysql_connection is not None:
            try:
                self.mysql_connection.close()
            except Exception as e:
                traceback_str = traceback.format_exc()
                msg = 'Failed to close MySQL connection.' +\
                    f'Error: {e}\nTraceback: {traceback_str}'
                self.logger.warning(msg)
            self.mysql_connection = None

    async def connect(self) -> None:
        """Connect to MySQL database.

        Establishes a new connection to the MySQL database using the configured
        connection parameters.
        """
        if self.mysql_connection is None:
            self.mysql_connection = await aiomysql.connect(
                host=self.mysql_host, port=self.mysql_port,
                user=self.mysql_username, password=self.mysql_password,
                db=self.mysql_database)

    async def reconnect(self) -> None:
        """Reconnect to MySQL database.

        Closes the current connection and establishes a new one. This method
        is useful for recovering from connection errors.
        """
        await self.disconnect()
        await self.connect()

    async def ensure_connected(self) -> None:
        """Ensure MySQL connection is active.

        Checks if the current connection is valid by executing a test query.
        If the connection is invalid or None, automatically reconnects to the
        database.
        """
        if self.mysql_connection is not None:
            try:
                async with self.mysql_connection_lock:
                    async with self.mysql_connection.cursor() as cur:
                        command = 'SELECT 1;'
                        await cur.execute(command)
            except Exception as e:
                traceback_str = traceback.format_exc()
                msg = 'MySQL connection failed, attempting to reconnect.' +\
                    f'Error: {e}\nTraceback: {traceback_str}'
                self.logger.error(msg)
                self.mysql_connection = None
        if self.mysql_connection is None:
            await self.connect()

    async def _get_version_from_mysql(self) -> str:
        """Get the version of motion database metadata.

        Returns:
            str: Version string.
        """
        await self.disconnect()
        async with aiomysql.connect(
                host=self.mysql_host, port=self.mysql_port,
                user=self.mysql_username, password=self.mysql_password,
                db=self.mysql_database) as mysql_connection:
            async with mysql_connection.cursor() as cur:
                command = '''SELECT MAX(last_update_time) AS latest_update_time
                    FROM (
                    SELECT MAX(update_time) AS last_update_time
                    FROM information_schema.tables
                    WHERE table_schema = 'motion_db'
                    GROUP BY TABLE_NAME
                ) AS subquery;'''
                await cur.execute(command)
                result = await cur.fetchall()
        return str(result[0][0])

    async def get_version(self) -> str:
        """Get the version of all motions in the motion database.

        Returns:
            str: Version string.
        """
        await self.ensure_connected()
        version = await self._get_version_from_mysql()
        self.version = version
        return await super().get_version()

    async def get_ids(self) -> list[int]:
        """Get all motion IDs in the motion database.

        Returns:
            list[int]: List of motion IDs.
        """
        await self.ensure_connected()
        async with self.mysql_connection_lock:
            async with self.mysql_connection.cursor() as cur:
                command = f'''
                    SELECT motion_record_id FROM
                    {self.motion_record_table_name}
                    WHERE {self.enabled_column_name} = 1
                '''
                await cur.execute(command)
                result = await cur.fetchall()
        return [int(x[0]) for x in result]

    async def get_motion_dict_by_id(
            self,
            id: int,
            thread_pool_executor: ThreadPoolExecutor | None = None) -> dict:
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
        await self.ensure_connected()
        command = self.mysql_join_cmd.replace(';', '')
        command += f' WHERE motion_record_id = {id};'
        class_priority = 0
        cutoff_frames = None
        cutoff_ranges = None
        npz_oss_path = None
        async with self.mysql_connection_lock:
            async with self.mysql_connection.cursor() as cur:
                await cur.execute(command)
                result = await cur.fetchall()
                if len(result) == 0:
                    msg = f'No motion data record found for ID={id}.'
                    self.logger.error(msg)
                    raise KeyError(msg)
                column_names = [x[0] for x in cur.description]
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

