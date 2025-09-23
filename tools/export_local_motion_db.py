import argparse
import asyncio
import json
import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import aiomysql
from minio import Minio
from tqdm import tqdm

from speech2motion.utils.log import setup_logger

_MYSQL_TO_SQLITE_TYPES = {
    'int': 'INTEGER',
    'bigint': 'INTEGER',
    'smallint': 'INTEGER',
    'tinyint': 'INTEGER',
    'varchar': 'TEXT',
    'char': 'TEXT',
    'text': 'TEXT',
    'longtext': 'TEXT',
    'mediumtext': 'TEXT',
    'datetime': 'TEXT',
    'timestamp': 'TEXT',
    'date': 'TEXT',
    'time': 'TEXT',
    'float': 'REAL',
    'double': 'REAL',
    'decimal': 'REAL',
    'json': 'TEXT',
    'blob': 'BLOB',
    'longblob': 'BLOB',
    'mediumblob': 'BLOB',
    'tinyblob': 'BLOB'
}


async def determine_export_scope(
        mysql_conn: aiomysql.Connection,
        motion_record_id_json_path: str | None,
        logger: logging.Logger
        ) -> list[int]:
    """Determine the scope of motion records to export.

    This function either reads motion record IDs from a JSON file or queries
    the MySQL database to get motion record IDs that match specific criteria.

    Args:
        mysql_conn (aiomysql.Connection):
            MySQL database connection object.
        motion_record_id_json_path (str | None):
            Path to JSON file containing motion record IDs. If None, queries
            database for records where states is not null or enabled = 1.
        logger (logging.Logger):
            Logger instance for logging messages.

    Returns:
        list[int]:
            List of motion record IDs to export.
    """
    if motion_record_id_json_path is not None:
        with open(motion_record_id_json_path) as f:
            motion_record_ids = json.load(f)
        logger.info(
            f'Exporting {len(motion_record_ids)} motion records '
            f'according to the motion_record_ids file.'
        )
        return motion_record_ids
    else:
        cmd = '''
            SELECT
                motion_record_id
            FROM
                motion_record
            WHERE states is not null or enabled = 1;
        '''
        async with mysql_conn.cursor() as cursor:
            await cursor.execute(cmd)
            results = await cursor.fetchall()
            motion_record_ids = [row[0] for row in results]
        logger.info(
            f'Exporting {len(motion_record_ids)} motion records '
            f'matching the criteria.'
        )
        return motion_record_ids

async def get_foreign_key_constraints(
        mysql_conn: aiomysql.Connection, database_name: str,
        ) -> dict[str, list[dict]]:
    """Retrieve foreign key constraint information from MySQL database.

    Args:
        mysql_conn:
            MySQL database connection object.
        database_name (str):
            Name of the database to query.

    Returns:
        dict[str, list[dict]]:
            Dictionary mapping table names to lists of foreign key
            constraint dictionaries. Each constraint dictionary contains
            'column', 'referenced_table', 'referenced_column', and
            'constraint_name' keys.
    """
    foreign_keys_query = """
        SELECT
            TABLE_NAME,
            COLUMN_NAME,
            REFERENCED_TABLE_NAME,
            REFERENCED_COLUMN_NAME,
            CONSTRAINT_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE REFERENCED_TABLE_NAME IS NOT NULL
            AND TABLE_SCHEMA = %s
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """

    async with mysql_conn.cursor() as cursor:
        await cursor.execute(foreign_keys_query, (database_name,))
        foreign_keys = await cursor.fetchall()

    # Group foreign key constraints by table name
    constraints_by_table = {}
    for fk in foreign_keys:
        table_name = fk[0]
        if table_name not in constraints_by_table:
            constraints_by_table[table_name] = []
        constraints_by_table[table_name].append({
            'column': fk[1],
            'referenced_table': fk[2],
            'referenced_column': fk[3],
            'constraint_name': fk[4]
        })

    return constraints_by_table

async def transfer_mysql_to_sqlite(
        mysql_conn: aiomysql.Connection,
        sqlite_conn: sqlite3.Connection,
        database_name: str,
        motion_record_ids: list[int],
        logger: logging.Logger
    ) -> None:
    """Transfer motion data from MySQL to SQLite database.

    This function exports motion-related data from a MySQL database to a SQLite
    database, preserving table structures, data types, and foreign key
    relationships. It processes tables in a specific order to maintain
    referential integrity.

    Args:
        mysql_conn (aiomysql.Connection):
            MySQL database connection object.
        sqlite_conn (sqlite3.Connection):
            SQLite database connection object.
        database_name (str):
            Name of the MySQL database to export from.
        motion_record_ids (list[int]):
            List of motion record IDs to export.
        logger (logging.Logger):
            Logger instance for logging messages.
    """

    # Select primary key of every table
    primary_keys: dict[str, dict[str, str | set[int]]] = dict(
        motion_record=dict(
            primary_key_column='motion_record_id',
            column_with_table_name='mr.motion_record_id',
            primary_key_values=set(),
            rank=3
        ),
        motion_file=dict(
            primary_key_column='motion_file_id',
            column_with_table_name='mf.motion_file_id',
            primary_key_values=set(),
            rank=2
        ),
        origin=dict(
            primary_key_column='origin_id',
            column_with_table_name='ori.origin_id',
            primary_key_values=set(),
            rank=0
        ),
        armature=dict(
            primary_key_column='armature_id',
            column_with_table_name='armature.armature_id',
            primary_key_values=set(),
            rank=1
        ),
        avatar=dict(
            primary_key_column='avatar_id',
            column_with_table_name='avatar.avatar_id',
            primary_key_values=set(),
            rank=0
        ),
        motion_keyword=dict(
            primary_key_column='motion_keyword_id',
            column_with_table_name='mk.motion_keyword_id',
            primary_key_values=set(),
            rank=0
        ),
        speech_keyword=dict(
            primary_key_column='speech_keyword_id',
            column_with_table_name='sk.speech_keyword_id',
            primary_key_values=set(),
            rank=0
        ),
        random=dict(
            primary_key_column='random_id',
            column_with_table_name='ra.random_id',
            primary_key_values=set(),
            rank=0
        ),
        loopable=dict(
            primary_key_column='loop_id',
            column_with_table_name='lo.loop_id',
            primary_key_values=set(),
            rank=0
        ),
    )
    cmd = f'''
        SELECT
            {', '.join(
                [primary_keys[table]['column_with_table_name']
                for table in primary_keys])}
        FROM
            motion_record mr
            LEFT JOIN motion_file mf ON mr.motion_file_id = mf.motion_file_id
            LEFT JOIN origin ori ON mf.origin_id = ori.origin_id
            LEFT JOIN armature ON mf.armature_id = armature.armature_id
            LEFT JOIN avatar ON armature.avatar_id = avatar.avatar_id
            LEFT JOIN motion_keyword mk ON mk.motion_keyword_id =
                mr.motion_keyword_id
            LEFT JOIN speech_keyword sk ON sk.speech_keyword_id =
                mr.speech_keyword_id
            LEFT JOIN random ra ON ra.random_id = mr.random_id
            LEFT JOIN loopable lo ON lo.loop_id = mr.loop_id
        WHERE
            motion_record_id in ({', '.join(map(str, motion_record_ids))});
    '''
    async with mysql_conn.cursor() as cursor:
        await cursor.execute(cmd)
        results = await cursor.fetchall()
        # Iterate through results and
        # add non-null primary key values to corresponding sets
        for row in results:
            for i, table_name in enumerate(primary_keys.keys()):
                if row[i] is not None:
                    primary_keys[table_name]['primary_key_values'].add(row[i])
    # Get foreign key constraint information
    foreign_key_constraints = await get_foreign_key_constraints(
        mysql_conn, database_name
    )
    # Enable SQLite foreign key constraints
    sqlite_conn.execute("PRAGMA foreign_keys = ON;")

    table_names_in_order = sorted(
        primary_keys.keys(), key=lambda x: primary_keys[x]['rank'])
    for table_name in table_names_in_order:
        if not primary_keys[table_name]['primary_key_values']:
            msg = f'No data found for table {table_name}.'
            logger.error(msg)
            raise ValueError(msg)
        logger.info(
            f'Processing table {table_name} with '
            f'{len(primary_keys[table_name]["primary_key_values"])} records.'
        )
        # Get table structure information
        table_structure_cmd = f'''
            DESCRIBE {table_name}
        '''
        async with mysql_conn.cursor() as cursor:
            await cursor.execute(table_structure_cmd)
            columns_info = await cursor.fetchall()
        # Build column definitions and data type mapping
        column_definitions = []
        foreign_key_definitions = []

        for col_info in columns_info:
            col_name = col_info[0]
            col_type = col_info[1].lower()
            is_nullable = col_info[2] == 'YES'
            is_key = col_info[3] == 'PRI'
            # Convert MySQL types to SQLite types
            sqlite_type = 'TEXT'  # Default type
            for mysql_type, sqlite_type_mapped in _MYSQL_TO_SQLITE_TYPES.items():
                if mysql_type in col_type:
                    sqlite_type = sqlite_type_mapped
                    break
            # Build column definition
            col_def = f'`{col_name}` {sqlite_type}'
            if is_key:
                col_def += ' PRIMARY KEY'
            if not is_nullable and not is_key:
                col_def += ' NOT NULL'
            column_definitions.append(col_def)

        # Add foreign key constraints
        if table_name in foreign_key_constraints:
            logger.info(
                f'Adding {len(foreign_key_constraints[table_name])} '
                f'foreign key constraints for table {table_name}'
            )
            for fk in foreign_key_constraints[table_name]:
                fk_def = (
                    f'FOREIGN KEY (`{fk["column"]}`) '
                    f'REFERENCES `{fk["referenced_table"]}` '
                    f'(`{fk["referenced_column"]}`)'
                )
                foreign_key_definitions.append(fk_def)
                logger.debug(
                    f'  - {fk["column"]} -> '
                    f'{fk["referenced_table"]}.{fk["referenced_column"]}'
                )
        else:
            logger.info(f'No foreign key constraints found for table {table_name}')

        # Create SQLite table
        all_definitions = column_definitions + foreign_key_definitions
        create_table_cmd = f'''
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                {', '.join(all_definitions)}
            )
        '''
        sqlite_conn.execute(create_table_cmd)
        logger.info(f'Created table {table_name} in SQLite.')

        # Get all column names
        column_names = [col_info[0] for col_info in columns_info]

        # Query data from MySQL
        primary_key_col = primary_keys[table_name]['primary_key_column']
        primary_key_values = list(primary_keys[table_name]['primary_key_values'])

        # Process in batches to avoid overly long SQL queries
        batch_size = 1000
        for i in range(0, len(primary_key_values), batch_size):
            batch_values = primary_key_values[i:i + batch_size]
            placeholders = ', '.join(['%s'] * len(batch_values))
            select_cmd = f'''
                SELECT {', '.join([f'`{col}`' for col in column_names])}
                FROM `{table_name}`
                WHERE `{primary_key_col}` IN ({placeholders})
            '''

            async with mysql_conn.cursor() as cursor:
                await cursor.execute(select_cmd, batch_values)
                rows = await cursor.fetchall()

            # Insert data into SQLite
            if rows:
                placeholders_sqlite = ', '.join(['?' for _ in column_names])
                insert_cmd = f'''
                    INSERT OR REPLACE INTO `{table_name}`
                    ({', '.join([f'`{col}`' for col in column_names])})
                    VALUES ({placeholders_sqlite})
                '''
                sqlite_conn.executemany(insert_cmd, rows)
                logger.info(
                    f'Inserted {len(rows)} records into {table_name} '
                    f'(batch {i//batch_size + 1}).'
                )

        logger.info(f'Completed processing table {table_name}.')

    # Commit all changes
    sqlite_conn.commit()
    logger.info('All data has been successfully exported to SQLite database.')

async def download_motion_files(
        sqlite_conn: sqlite3.Connection,
        minio_client: Minio,
        motion_file_root: str,
        logger: logging.Logger,
        max_workers: int = 4,
        ) -> None:
    """Download motion files from MinIO to local storage.

    This function queries the SQLite database for NPZ file paths, then downloads
    them from MinIO to the local file system using a thread pool executor for
    parallel processing.

    Args:
        sqlite_conn (sqlite3.Connection):
            SQLite database connection to query file paths.
        minio_client (Minio):
            MinIO client for downloading files.
        motion_file_root (str):
            Root directory for storing downloaded motion files.
        logger (logging.Logger):
            Logger instance for logging messages.
        max_workers (int, optional):
            Maximum number of worker threads for parallel downloads.
            Defaults to 4.
    """
    cmd = 'SELECT oss_bucket, npz_oss_path FROM motion_file;'

    # Query all NPZ file paths from SQLite
    cursor = sqlite_conn.cursor()
    cursor.execute(cmd)
    file_paths = cursor.fetchall()
    cursor.close()

    if not file_paths:
        logger.info('No motion files found in database.')
        return

    logger.info(f'Found {len(file_paths)} motion files to download.')

    # Create local directories if they don't exist
    os.makedirs(motion_file_root, exist_ok=True)

    def download_single_file(
        bucket_path_tuple: tuple[str, str]
    ) -> tuple[bool, str, str, str | None]:
        """Download a single file from MinIO to local storage.

        Args:
            bucket_path_tuple (tuple):
                Tuple containing (bucket_name, oss_path).

        Returns:
            tuple: (success, bucket_name, oss_path, error_message)
        """
        bucket_name, oss_path = bucket_path_tuple

        try:
            # Create local file path
            local_file_path = os.path.join(motion_file_root, oss_path)

            # Create local directory if it doesn't exist
            local_dir = os.path.dirname(local_file_path)
            os.makedirs(local_dir, exist_ok=True)

            # Download file from MinIO
            minio_client.fget_object(bucket_name, oss_path, local_file_path)

            return (True, bucket_name, oss_path, None)

        except Exception as e:
            error_msg = f'Failed to download {oss_path} from {bucket_name}: {e!s}'
            return (False, bucket_name, oss_path, error_msg)

    # Use ThreadPoolExecutor for parallel downloads
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        tasks = [
            loop.run_in_executor(executor, download_single_file, bucket_path_tuple)
            for bucket_path_tuple in file_paths
        ]

        # Wait for all downloads to complete with progress bar
        results = []
        with tqdm(total=len(tasks), desc="Downloading files", unit="file") as pbar:
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                pbar.update(1)

    # Process results and log statistics
    successful_downloads = 0
    failed_downloads = 0

    for result in results:
        if isinstance(result, Exception):
            logger.error(f'Unexpected error during download: {result!s}')
            failed_downloads += 1
        else:
            success, _, oss_path, error_msg = result
            if success:
                successful_downloads += 1
                logger.debug(f'Successfully downloaded: {oss_path}')
            else:
                failed_downloads += 1
                logger.error(error_msg)

    logger.info(
        f'Download completed: {successful_downloads} successful, '
        f'{failed_downloads} failed out of {len(file_paths)} total files.'
    )

async def main(args) -> None:
    """Export motion database from MySQL to SQLite with foreign key constraints.

    This function exports motion-related data from a MySQL database to a SQLite
    database, preserving table structures, data types, and foreign key
    relationships. It processes tables in a specific order to maintain
    referential integrity.

    Args:
        args:
            Command line arguments containing MySQL connection details,
            MinIO configuration, export conditions, and output paths.
    """
    logger = setup_logger(
        logger_name="export_local_motion_db", logger_path=args.log_path)
    if os.path.exists(args.sqlite_path):
        msg = f'SQLite database file already exists at {args.sqlite_path}. ' +\
            'Please remove it or choose a different path.'
        logger.error(msg)
        raise FileExistsError(msg)
    sqlite_conn = sqlite3.connect(args.sqlite_path)
    async with aiomysql.connect(
            host=args.mysql_host, port=args.mysql_port,
            user=args.mysql_username, password=args.mysql_password,
            db=args.mysql_database) as mysql_conn:
        minio_client = Minio(
            endpoint=args.minio_endpoint,
            access_key=args.minio_access_key,
            secret_key=args.minio_secret_key,
            secure=False
        )
        motion_record_ids = await determine_export_scope(
            mysql_conn=mysql_conn,
            motion_record_id_json_path=args.motion_record_ids,
            logger=logger
        )
        await transfer_mysql_to_sqlite(
            mysql_conn=mysql_conn,
            sqlite_conn=sqlite_conn,
            database_name=args.mysql_database,
            motion_record_ids=motion_record_ids,
            logger=logger
        )

        # Download motion files from MinIO
        await download_motion_files(
            sqlite_conn=sqlite_conn,
            minio_client=minio_client,
            motion_file_root=args.motion_file_root,
            logger=logger
        )



def parse_args():
    """Parse command line arguments for the motion database export tool.

    This function defines and parses command line arguments required for
    exporting motion data from MySQL to SQLite and downloading motion files
    from MinIO storage.

    Returns:
        argparse.Namespace:
            Parsed command line arguments containing MySQL connection
            parameters, MinIO configuration, export conditions, and
            output file paths.
    """
    parser = argparse.ArgumentParser(
        description="Export motion database from MySQL to SQLite and download "
        "motion files from MinIO storage."
    )
    # MySQL connection parameters
    parser.add_argument(
        "--mysql_host",
        type=str,
        required=True,
        help="MySQL database host address"
    )
    parser.add_argument(
        "--mysql_port",
        type=int,
        default=3306,
        help="MySQL database port number (default: 3306)"
    )
    parser.add_argument(
        "--mysql_username",
        type=str,
        required=True,
        help="MySQL database username"
    )
    parser.add_argument(
        "--mysql_password",
        type=str,
        required=True,
        help="MySQL database password"
    )
    parser.add_argument(
        "--mysql_database",
        type=str,
        default='motion_db',
        help="MySQL database name (default: motion_db)"
    )
    # MinIO connection parameters
    parser.add_argument(
        "--minio_endpoint",
        type=str,
        required=True,
        help="MinIO server endpoint URL. e.g. 127.0.0.1:9000"
    )
    parser.add_argument(
        "--minio_access_key",
        type=str,
        required=True,
        help="MinIO access key"
    )
    parser.add_argument(
        "--minio_secret_key",
        type=str,
        required=True,
        help="MinIO secret key"
    )
    # Export condition parameters
    parser.add_argument(
        "--motion_record_ids",
        type=str,
        required=False,
        default=None,
        help="Path to JSON file containing specific motion record IDs to export "
        "(optional)"
    )
    # Output configuration parameters
    parser.add_argument(
        "--motion_file_root",
        type=str,
        default='data/motion_files',
        help="Root directory for storing downloaded motion files "
        "(default: data/motion_files)"
    )
    parser.add_argument(
        "--sqlite_path",
        type=str,
        default='data/motion_database.db',
        help="Path for the output SQLite database file "
        "(default: data/motion_database.db)"
    )
    # Log configuration parameters
    parser.add_argument(
        '--log_path',
        type=str,
        help='Path for the log file (optional)',
        required=False,
        default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    asyncio.run(main(args))

