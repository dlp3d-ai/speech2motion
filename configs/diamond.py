import os

import numpy as np

# CRITICAL = 50
# ERROR = 40
# WARNING = 30
# INFO = 20
# DEBUG = 10
# NOTSET = 0

__logger_cfg__ = dict(
    logger_name="root",
    file_level=10,
    console_level=20,
    logger_path='logs/server.log',
    aws_use_queues=True
)
type: str = 'FastAPIServer'
max_workers: int = 8
enable_cors: bool = True
host: str = '0.0.0.0'
port: int = 18084
logger_cfg: dict = __logger_cfg__

__meta_reader_cfg__ = dict(
    type="MySQLMetaReader",
    mysql_host=os.getenv('MYSQL_HOST'),
    mysql_port=os.getenv('MYSQL_PORT'),
    mysql_username=os.getenv('MYSQL_USER'),
    mysql_password=os.getenv('MYSQL_PASSWORD'),
    mysql_database='motion_db',
    mysql_join_cmd_path='configs/3dac_sql_join.sql',
    logger_cfg=__logger_cfg__
)
__motion_reader_cfg__ = dict(
    type="MinioMySQLMotionReader",
    mysql_host=os.getenv('MYSQL_HOST'),
    mysql_port=os.getenv('MYSQL_PORT'),
    mysql_username=os.getenv('MYSQL_USER'),
    mysql_password=os.getenv('MYSQL_PASSWORD'),
    mysql_database='motion_db',
    mysql_join_cmd_path='configs/3dac_sql_join.sql',
    endpoint=os.getenv('OSS_ENDPOINT'),
    access_key=os.getenv('OSS_ACCESS_KEY'),
    secret_key=os.getenv('OSS_SECRET_KEY'),
    blendshape_names='configs/mmd_blendshapes.json',
    bucket_name='3dac',
    float_dtype=np.float32,
    logger_cfg=__logger_cfg__
)
__restpose_reader_cfg__ = dict(
    type="MinioRestposeReader",
    file_paths={
        'KQ-default': 'restpose_npz/KQ_default_0326_skeleton.npz',
        'FNN-default': 'restpose_npz/FNN_default_0819_skeleton.npz',
        'HT-default': 'restpose_npz/HT_default_0819_skeleton.npz',
        'NXD-default': 'restpose_npz/NXD_default_1202_skeleton.npz',
        'KL-default': 'restpose_npz/KL_default_1202_skeleton.npz',
        'Ani-default': 'restpose_npz/Ani_default_0827_skeleton.npz'
    },
    endpoint=os.getenv('OSS_ENDPOINT'),
    access_key=os.getenv('OSS_ACCESS_KEY'),
    secret_key=os.getenv('OSS_SECRET_KEY'),
    bucket_name='3dac',
    float_dtype=np.float32,
    logger_cfg=__logger_cfg__
)
__memory_cfg__ = dict(
    type="LocalMemory",
    memory_duration=60,
    logger_cfg=__logger_cfg__
)
__text_segmentation_cfg__ = dict(
    type="JiebaTextSegmentation",
    logger_cfg=__logger_cfg__
)
__cache_cfg__ = dict(
    type="LocalCache",
    max_workers=8,
    index_mapping_cfg_template=dict(
        type="DictIndexMapping",
    ),
    logger_cfg=__logger_cfg__
)

python_api_v2_cfg = dict(
    type="StreamingSpeech2MotionV2",
    meta_reader_cfg=__meta_reader_cfg__,
    motion_reader_cfg=__motion_reader_cfg__,
    restpose_reader_cfg=__restpose_reader_cfg__,
    memory_cfg=__memory_cfg__,
    text_segmentation_cfg=__text_segmentation_cfg__,
    cache_cfg=__cache_cfg__,
    merge_cfg=dict(
        type="Interpolation",
        transit_frames=15,
        logger_cfg=__logger_cfg__,
    ),
    request_expire_time=60,
    maintain_check_interval=60,
    max_workers=8,
    first_body_fast_response=False,
    sleep_time=0.01,
    logger_cfg=__logger_cfg__,
)

python_api_v3_cfg = dict(
    type="StreamingSpeech2MotionV3",
    meta_reader_cfg=__meta_reader_cfg__,
    motion_reader_cfg=__motion_reader_cfg__,
    restpose_reader_cfg=__restpose_reader_cfg__,
    memory_cfg=__memory_cfg__,
    text_segmentation_cfg=__text_segmentation_cfg__,
    cache_cfg=__cache_cfg__,
    interpolation_cfg=dict(
        type="Interpolation",
        transit_frames=15,
        logger_cfg=__logger_cfg__,
    ),
    blending_cfg=dict(
        type="Blending",
        logger_cfg=__logger_cfg__,
    ),
    request_expire_time=60,
    maintain_check_interval=60,
    max_workers=8,
    sleep_time=0.01,
    logger_cfg=__logger_cfg__,
)
