import json
import os
import traceback
from typing import Any

from .base_meta_reader import MDB_COLUMNS_BY_COLOMN, BaseMetaReader

try:
    import aiosqlite
    sqlite_installed = True
except (ImportError, ModuleNotFoundError):
    sqlite_installed = False
    import_traceback_str = traceback.format_exc()


class SQLiteMetaReader(BaseMetaReader):
    """SQLite implementation of meta data reader for motion records.

    This class provides methods to read motion record metadata from a SQLite
    database, including version information, record IDs, and detailed metadata
    for specific records. It creates a new database connection for each operation
    to ensure thread safety and proper resource management.
    """

    def __init__(self,
                 sqlite_path: str,
                 sqlite_join_cmd_path: str,
                 motion_record_table_name: str = 'motion_record',
                 enabled_column_name: str = 'enabled',
                 logger_cfg: None | dict = None) -> None:
        """Initialize the SQLite meta reader.

        Args:
            sqlite_path (str):
                Path to the SQLite database file.
            sqlite_join_cmd_path (str):
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
        if not sqlite_installed:
            msg = 'SQLiteMetaReader requires aiosqlite to be installed.' +\
                f'Traceback:\n{import_traceback_str}'
            self.logger.error(msg)
            raise ImportError(msg)
        self.sqlite_path = sqlite_path
        self.sqlite_join_cmd_path = sqlite_join_cmd_path
        with open(self.sqlite_join_cmd_path) as f:
            self.sqlite_join_cmd = f.read().replace('\n', ' ').strip()
        self.motion_record_table_name = motion_record_table_name
        self.enabled_column_name = enabled_column_name

    async def get_version(self) -> str:
        """Get the version of motion database metadata.

        Creates a new database connection for this operation to ensure thread
        safety and proper resource management. Since SQLite doesn't track table
        modification times, this method returns the database file's modification
        time as a version indicator.

        Returns:
            str: Version string (database file modification time).
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

    async def get_ids(self) -> list[int]:
        """Get all motion_record_id from metadata.

        Creates a new database connection for this operation to ensure thread
        safety and proper resource management. Queries the database for all
        motion record IDs that are currently enabled and returns them as a
        list of integers.

        Returns:
            list[int]: List of enabled motion record IDs.
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

    async def get_meta_by_id(self, id: int) -> dict[str, Any]:
        """Get metadata by ID.

        Creates a new database connection for this operation to ensure thread
        safety and proper resource management. Retrieves complete metadata for
        a specific motion record by its ID. The metadata includes all related
        information from joined tables and is converted to appropriate Python
        data types.

        Args:
            id (int): MotionRecord ID to retrieve metadata for.

        Returns:
            dict[str, Any]: Type-converted metadata dictionary containing all
                motion record information and related data.

        Raises:
            KeyError: If no motion record is found with the specified ID.
            ValueError: If JSON decoding fails for any JSON fields.
        """
        command = self.sqlite_join_cmd.replace(';', '')
        command += f' WHERE motion_record_id = {id};'
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
        # convert sqlite result to python built-in types
        sqlite_dict = dict()
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
            sqlite_dict[column_name] = value
        loop_id = sqlite_dict.pop('loop_id', None)
        if loop_id is not None:
            sqlite_dict['loopable'] = dict(
                loop_start_frame=sqlite_dict.pop('loop_start_frame'),
                loop_end_frame=sqlite_dict.pop('loop_end_frame')
            )
        is_random_audio = bool(sqlite_dict.pop('is_random_audio'))
        if is_random_audio:
            sqlite_dict['random'] = dict(
                cutoff_frames=sqlite_dict.pop('cutoff_frames', None),
                cutoff_ranges=sqlite_dict.pop('cutoff_ranges', None)
            )
        motion_keyword_id = sqlite_dict.pop('motion_keyword_id', None)
        if motion_keyword_id is not None:
            sqlite_dict['motion_keyword'] = dict(
                motion_keywords_ch=sqlite_dict.pop('motion_keywords_ch'),
                motion_keyword_frame=sqlite_dict.pop('motion_keyword_frame')
            )
        speech_keyword_id = sqlite_dict.pop('speech_keyword_id', None)
        if speech_keyword_id is not None:
            sqlite_dict['speech_keyword'] = dict(
                speech_keywords_ch=sqlite_dict.pop('speech_keywords_ch'),
                speech_keyword_frame=sqlite_dict.pop('speech_keyword_frame')
            )
        n_frames = sqlite_dict['n_frames']
        startup_frame = sqlite_dict['startup_frame']
        recovery_frame = sqlite_dict['recovery_frame']
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
            motion_record_id = sqlite_dict['motion_record_id']
            msg = f'Found motion with short transition interval, ID={motion_record_id},'
            if startup_frame_overwrite is not None:
                msg += f' startup frame count changed from {startup_frame} ' +\
                    f'to {startup_frame_overwrite},'
                sqlite_dict['startup_frame'] = startup_frame_overwrite
            if recovery_frame_overwrite is not None:
                msg += f' recovery frame count changed from {recovery_frame} ' +\
                    f'to {recovery_frame_overwrite}.'
                sqlite_dict['recovery_frame'] = recovery_frame_overwrite
            msg = msg[:-1] + '.'
            # TODO: When database is re-annotated, upgrade log level below to warning
            self.logger.debug(msg)
        return sqlite_dict
