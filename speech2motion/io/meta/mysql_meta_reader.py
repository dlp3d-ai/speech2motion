import json
import traceback
from asyncio import Lock
from typing import Any

from .base_meta_reader import MDB_COLUMNS_BY_COLOMN, BaseMetaReader

try:
    import aiomysql
    mysql_installed = True
except (ImportError, ModuleNotFoundError):
    mysql_installed = False
    import_traceback_str = traceback.format_exc()

class MySQLMetaReader(BaseMetaReader):
    """MySQL implementation of meta data reader for motion records.

    This class provides methods to read motion record metadata from a MySQL
    database, including version information, record IDs, and detailed metadata
    for specific records.
    """

    def __init__(self,
                 mysql_host: str,
                 mysql_port: int,
                 mysql_username: str,
                 mysql_password: str,
                 mysql_database: str,
                 mysql_join_cmd_path: str,
                 motion_record_table_name: str = 'motion_record',
                 enabled_column_name: str = 'enabled',
                 logger_cfg: None | dict = None) -> None:
        """Initialize the MySQL meta reader.

        Args:
            mysql_host (str):
                MySQL database host address.
            mysql_port (int):
                MySQL database port number.
            mysql_username (str):
                MySQL database username.
            mysql_password (str):
                MySQL database password.
            mysql_database (str):
                MySQL database name.
            mysql_join_cmd_path (str):
                Path to the SQL file containing the JOIN query.
            motion_record_table_name (str, optional):
                Name of the motion record table. Defaults to 'motion_record'.
            enabled_column_name (str, optional):
                Name of the enabled column. Defaults to 'enabled'.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        BaseMetaReader.__init__(self, logger_cfg)
        if not mysql_installed:
            msg = 'MySQLMetaReader requires aiomysql to be installed.' +\
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

    async def get_version(self) -> str:
        """Get the version of motion database metadata.

        Queries the MySQL information_schema to get the latest update time
        across all tables in the motion database.

        Returns:
            str: Version string representing the latest update time.
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

    async def get_ids(self) -> list[int]:
        """Get all motion_record_id from metadata.

        Queries the database for all motion record IDs that are currently
        enabled and returns them as a list of integers.

        Returns:
            list[int]: List of enabled motion record IDs.
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

    async def get_meta_by_id(self, id: int) -> dict[str, Any]:
        """Get metadata by ID.

        Retrieves complete metadata for a specific motion record by its ID.
        The metadata includes all related information from joined tables and
        is converted to appropriate Python data types.

        Args:
            id (int): MotionRecord ID to retrieve metadata for.

        Returns:
            dict[str, Any]: Type-converted metadata dictionary containing all
                motion record information and related data.

        Raises:
            KeyError: If no motion record is found with the specified ID.
            ValueError: If JSON decoding fails for any JSON fields.
        """
        await self.ensure_connected()
        command = self.mysql_join_cmd.replace(';', '')
        command += f' WHERE motion_record_id = {id};'
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
        # convert mysql result to python built-in types
        mysql_dict = dict()
        for column_name in name_idx_mapping.keys():
            idx = name_idx_mapping[column_name]
            value = result[0][idx]
            if value is not None:
                if column_name in MDB_COLUMNS_BY_COLOMN:
                    type_cast = MDB_COLUMNS_BY_COLOMN[column_name]['type_cast']
                    try:
                        value = type_cast(value)
                    except json.JSONDecodeError as e:
                        msg = f'motion_record_id={id} data {column_name} ' +\
                            f'json decode failed, data before decode: {value}'
                        self.logger.error(msg)
                        raise ValueError(msg) from e
            else:
                value = None
            mysql_dict[column_name] = value
        loop_id = mysql_dict.pop('loop_id', None)
        if loop_id is not None:
            mysql_dict['loopable'] = dict(
                loop_start_frame=mysql_dict.pop('loop_start_frame'),
                loop_end_frame=mysql_dict.pop('loop_end_frame')
            )
        is_random_audio = bool(mysql_dict.pop('is_random_audio'))
        if is_random_audio:
            mysql_dict['random'] = dict(
                cutoff_frames=mysql_dict.pop('cutoff_frames', None),
                cutoff_ranges=mysql_dict.pop('cutoff_ranges', None)
            )
        motion_keyword_id = mysql_dict.pop('motion_keyword_id', None)
        if motion_keyword_id is not None:
            mysql_dict['motion_keyword'] = dict(
                motion_keywords_ch=mysql_dict.pop('motion_keywords_ch'),
                motion_keyword_frame=mysql_dict.pop('motion_keyword_frame')
            )
        speech_keyword_id = mysql_dict.pop('speech_keyword_id', None)
        if speech_keyword_id is not None:
            mysql_dict['speech_keyword'] = dict(
                speech_keywords_ch=mysql_dict.pop('speech_keywords_ch'),
                speech_keyword_frame=mysql_dict.pop('speech_keyword_frame')
            )
        n_frames = mysql_dict['n_frames']
        startup_frame = mysql_dict['startup_frame']
        recovery_frame = mysql_dict['recovery_frame']
        startup_frame_overwrite = None
        recovery_frame_overwrite = None
        if startup_frame < 15:
            startup_frame_overwrite = 15
        if n_frames - recovery_frame < 15:
            recovery_frame_overwrite = n_frames - 15
        if startup_frame_overwrite is not None and \
                recovery_frame_overwrite is not None and \
                startup_frame_overwrite + recovery_frame_overwrite > n_frames:
            startup_frame_overwrite = int(n_frames / 2)
            recovery_frame_overwrite = startup_frame_overwrite + 1
        if startup_frame_overwrite is not None or recovery_frame_overwrite is not None:
            motion_record_id = mysql_dict['motion_record_id']
            msg = f'Found motion with short transition interval, ID={motion_record_id},'
            if startup_frame_overwrite is not None:
                msg += f' startup frame count changed from {startup_frame} ' +\
                    f'to {startup_frame_overwrite},'
                mysql_dict['startup_frame'] = startup_frame_overwrite
            if recovery_frame_overwrite is not None:
                msg += f' recovery frame count changed from {recovery_frame} ' +\
                    f'to {recovery_frame_overwrite}.'
                mysql_dict['recovery_frame'] = recovery_frame_overwrite
            msg = msg[:-1] + '.'
            # TODO: When database is re-annotated, upgrade log level below to warning
            self.logger.debug(msg)
        return mysql_dict
