import asyncio
import io
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from ..data_structures.motion_clip import MotionClip
from ..data_structures.motion_record import MotionRecord, MotionRecordType
from ..data_structures.restpose import Restpose
from ..index.builder import build_index_mapping
from ..io.meta.base_meta_reader import BaseMetaReader
from ..io.motion.base_motion_reader import BaseMotionReader
from ..io.restpose.base_restpose_reader import BaseRestposeReader
from ..utils.log import setup_logger
from ..utils.super import Super

if TYPE_CHECKING:
    from ..index.base_index_mapping import BaseIndexMapping


class VersionMismatchError(Exception):
    """Version mismatch error."""
    pass

class CacheNotReadyError(Exception):
    """Cache not ready error."""
    pass

class LocalCache(Super):
    """Local cache using memory and tempfile.

    This class provides local caching functionality for motion data, supporting
    asynchronous operations and version management. The cache includes motion
    records, motion clips, restpose data, and various index mappings.
    """

    def __init__(
            self,
            meta_reader: BaseMetaReader,
            motion_reader: BaseMotionReader,
            index_mapping_cfg_template: dict,
            max_workers: int = 1,
            thread_pool_executor: ThreadPoolExecutor | None = None,
            restpose_reader: BaseRestposeReader | None = None,
            motion_clip_ttl: int = 10,
            logger_cfg: None | dict = None) -> None:
        """Initialize the local cache instance.

        Args:
            meta_reader (BaseMetaReader):
                Meta reader instance for reading metadata.
            motion_reader (BaseMotionReader):
                Motion reader instance for reading motion data.
            index_mapping_cfg_template (dict):
                Template configuration file for creating multiple index mappings.
            max_workers (int, optional):
                Maximum number of worker threads. Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor. If None, a new thread pool executor will be
                created based on max_workers. Defaults to None.
            restpose_reader (BaseRestposeReader | None, optional):
                Restpose reader instance. If None, restpose will not be cached.
                Defaults to None.
            motion_clip_ttl (int, optional):
                Time-to-live for motion clip cache. Defaults to 10.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Defaults to None.
        """
        Super.__init__(self, logger_cfg)
        self.meta_reader = meta_reader
        self.motion_reader = motion_reader
        self.restpose_reader = restpose_reader
        self.index_mapping_cfg_template = index_mapping_cfg_template
        self.max_workers = max_workers
        self.permanent_executor = thread_pool_executor \
            if thread_pool_executor is not None \
            else ThreadPoolExecutor(max_workers=max_workers)
        self.permanent_executor_external = True \
            if thread_pool_executor is not None \
            else False
        self.motion_clip_ttl = motion_clip_ttl

        self.version_lock = asyncio.Lock()
        self.prepare_lock = asyncio.Lock()
        self.cache_ready = False
        self.next_cache_ready = False
        self.version: str | None = None
        self.next_version: str | None = None

        self.motion_records: dict[int, MotionRecord] | None = None
        self.next_motion_records: dict[int, MotionRecord] | None = None
        self.motion_file_dir: tempfile.TemporaryDirectory | None = None
        self.next_motion_file_dir: tempfile.TemporaryDirectory | None = None
        self.motion_paths: dict[int, str] | None = None
        self.next_motion_paths: dict[int, str] | None = None
        self.motion_clips_cache: dict[int, dict[str, MotionClip | int]] | None = None
        self.next_motion_clips_cache: dict[
            int, dict[str, MotionClip | int]] | None = None
        self.restpose_file_dir: tempfile.TemporaryDirectory | None = None
        self.next_restpose_file_dir: tempfile.TemporaryDirectory | None = None
        self.restpose_paths: dict[str, str] | None = None
        self.next_restpose_paths: dict[str, str] | None = None
        self.restpose_cache: dict[str, Restpose] | None = None
        self.next_restpose_cache: dict[str, Restpose] | None = None

        self.avatar_mapping: BaseIndexMapping | None = None
        self.next_avatar_mapping: BaseIndexMapping | None = None
        self.type_mapping: BaseIndexMapping | None = None
        self.next_type_mapping: BaseIndexMapping | None = None
        self.motion_keyword_mapping: BaseIndexMapping | None = None
        self.next_motion_keyword_mapping: BaseIndexMapping | None = None
        self.speech_keyword_mapping: BaseIndexMapping | None = None
        self.next_speech_keyword_mapping: BaseIndexMapping | None = None
        self.label_mapping: BaseIndexMapping | None = None
        self.next_label_mapping: BaseIndexMapping | None = None

    def __del__(self) -> None:
        """Destructor, cleanup temporary directories."""
        temp_dirs = []
        if self.motion_file_dir is not None:
            temp_dirs.append(self.motion_file_dir)
        if self.restpose_file_dir is not None:
            temp_dirs.append(self.restpose_file_dir)
        if self.next_motion_file_dir is not None:
            temp_dirs.append(self.next_motion_file_dir)
        if self.next_restpose_file_dir is not None:
            temp_dirs.append(self.next_restpose_file_dir)
        for temp_dir in temp_dirs:
            try:
                temp_dir.cleanup()
            except Exception as e:  # noqa: PERF203
                self.logger.error(f'Failed to cleanup temporary directory: {e}')
        if not self.permanent_executor_external:
            self.permanent_executor.shutdown(wait=True)

    async def get_version(self) -> str | None:
        """Get the current cache version.

        Returns:
            str | None:
                Current cache version string. Returns None if cache is not ready.
        """
        async with self.version_lock:
            return self.version

    async def need_update(self) -> bool:
        """Check if cache needs to be updated.

        Returns:
            bool:
                True if cache needs to be updated, False otherwise.
        """
        async with self.version_lock:
            if self.version is None:
                return True
            meta_version = await self.meta_reader.get_version()
            if self.version != meta_version:
                return True
            return False

    async def prepare_next(self) -> None:
        """Prepare the next version of cache data.

        Synchronizes data from various data sources to local cache, including
        meta data, motion data, and restpose data. If the next version cache
        is already ready, a warning will be issued.

        Raises:
            VersionMismatchError:
                Raised when meta version and motion version do not match.
        """
        async with self.prepare_lock:
            if self.next_cache_ready:
                msg = 'Next version cache is ready, please call switch method.'
                self.logger.warning(msg)
                return
            self.next_cache_ready = False
            start_time = time.time()
            meta_version = await self.meta_reader.get_version()
            motion_version = await self.motion_reader.get_version()
            # Ignore restpose version comparison, as restpose reader version is constant
            if meta_version != motion_version:
                msg = f'Version mismatch, meta_version: {meta_version}, ' +\
                    f'motion_version: {motion_version}'
                self.logger.error(msg)
                raise VersionMismatchError(msg)
            # Sync meta data
            await self._sync_from_meta_reader()
            # Sync motion data, need to avoid parallel with meta sync,
            # MySQL Client cannot handle properly
            await self._sync_from_motion_reader()
            # Sync restpose data
            if self.restpose_reader is not None:
                await self._sync_from_restpose_reader()
            else:
                msg = ('restpose_reader not configured during construction, '
                       'restpose caching not available.')
                self.logger.warning(msg)
            # Set motion_clips_cache
            self.next_motion_clips_cache = dict()
            # Set restpose_cache
            self.next_restpose_cache = dict()
            # set attrs
            self.next_version = meta_version
            self.next_cache_ready = True
            self.logger.info(
                'Next version cache prepared in '
                f'{time.time() - start_time:.2f}s')

    async def switch_to_next(self) -> None:
        """Switch to the next version of cache data.

        Switches the current cache to the prepared next version. Cleans up
        outdated files and updates all cache attributes. If the next version
        cache is not ready, a warning will be issued.
        """
        async with self.version_lock:
            if not self.next_cache_ready:
                msg = 'Next version cache not found, please call prepare method first.'
                self.logger.warning(msg)
                return
            # set ready attr
            self.cache_ready = False
            # clean out-dated files
            if self.motion_file_dir is not None:
                try:
                    self.motion_file_dir.cleanup()
                except Exception as e:
                    self.logger.error(
                        f'Failed to cleanup motion file temporary directory: {e}')
            if self.restpose_file_dir is not None:
                try:
                    self.restpose_file_dir.cleanup()
                except Exception as e:
                    self.logger.error(
                        f'Failed to cleanup restpose file temporary directory: {e}')
            # set attrs
            self.version = self.next_version
            self.motion_records = self.next_motion_records
            self.motion_file_dir = self.next_motion_file_dir
            self.motion_paths = self.next_motion_paths
            self.motion_clips_cache = self.next_motion_clips_cache
            self.restpose_file_dir = self.next_restpose_file_dir
            self.restpose_paths = self.next_restpose_paths
            self.restpose_cache = self.next_restpose_cache
            self.avatar_mapping = self.next_avatar_mapping
            self.type_mapping = self.next_type_mapping
            self.motion_keyword_mapping = self.next_motion_keyword_mapping
            self.speech_keyword_mapping = self.next_speech_keyword_mapping
            self.label_mapping = self.next_label_mapping

            # reset next attrs
            self.next_version = None
            self.next_motion_records = None
            self.next_motion_file_dir = None
            self.next_motion_paths = None
            self.next_motion_clips_cache = None
            self.next_restpose_file_dir = None
            self.next_restpose_paths = None
            self.next_restpose_cache = None
            self.next_avatar_mapping = None
            self.next_type_mapping = None
            self.next_motion_keyword_mapping = None
            self.next_speech_keyword_mapping = None
            self.next_label_mapping = None
            # show version in logger
            self.logger_cfg['logger_name'] = f'{self.__class__.__name__}-{self.version}'
            self.logger = setup_logger(**self.logger_cfg)
            # set ready attr
            self.next_cache_ready = False
            self.cache_ready = True

    async def get_motion_clip_by_id(self, motion_record_id: int) -> MotionClip:
        """Get motion clip by motion_record_id.

        First checks memory cache, if not hit then loads from disk.
        Uses TTL mechanism to manage cache lifecycle.

        Args:
            motion_record_id (int):
                Motion record ID.

        Returns:
            MotionClip:
                Corresponding motion clip object.

        Raises:
            CacheNotReadyError:
                Raised when cache is not ready.
            KeyError:
                Raised when no motion data record is found for the specified ID.
        """
        if not self.cache_ready:
            msg = 'Cache not ready, cannot get motion_clip.'
            self.logger.error(msg)
            raise CacheNotReadyError(msg)
        # check if cache hits
        ret_motion_clip = None
        if motion_record_id in self.motion_clips_cache:
            ret_motion_clip = self.motion_clips_cache[motion_record_id]['motion_clip']
        # not hit, load from disk
        elif motion_record_id in self.motion_paths:
            npz_path = self.motion_paths[motion_record_id]
            loop = asyncio.get_running_loop()
            ret_motion_clip = await loop.run_in_executor(
                self.permanent_executor,
                _load_motion_clip, npz_path)
            self.motion_clips_cache[motion_record_id] = dict(
                motion_clip=ret_motion_clip,
                ttl=self.motion_clip_ttl
            )
        if ret_motion_clip is None:
            msg = f'No motion data record found for ID={motion_record_id}.'
            self.logger.error(msg)
            raise KeyError(msg)
        # update ttl, and pop expired items
        keys_to_pop = list()
        self.motion_clips_cache[motion_record_id]['ttl'] = self.motion_clip_ttl
        for key, value in self.motion_clips_cache.items():
            if key != motion_record_id:
                value['ttl'] -= 1
                if value['ttl'] <= 0:
                    keys_to_pop.append(key)
        if len(keys_to_pop) > 0:
            self.logger.debug(
                f'Removing expired motion_clip cache: {keys_to_pop}'
            )
            for key in keys_to_pop:
                self.motion_clips_cache.pop(key)
        return ret_motion_clip

    async def get_restpose_by_name(self, restpose_name: str) -> Restpose:
        """Get restpose by restpose_name.

        First checks memory cache, if not hit then loads from disk.
        Since the total number of Restpose is small and each Restpose data
        size is small, it is cached directly.

        Args:
            restpose_name (str):
                Restpose name.

        Returns:
            Restpose:
                Corresponding restpose object.

        Raises:
            CacheNotReadyError:
                Raised when cache is not ready or restpose_reader is not configured.
            KeyError:
                Raised when no restpose data record is found for the specified name.
        """
        if not self.cache_ready:
            msg = 'Cache not ready, cannot get restpose.'
            self.logger.error(msg)
            raise CacheNotReadyError(msg)
        if self.restpose_reader is None:
            msg = ('restpose_reader not configured during construction, '
                   'cannot get restpose.')
            self.logger.error(msg)
            raise CacheNotReadyError(msg)
        if restpose_name in self.restpose_cache:
            return self.restpose_cache[restpose_name]
        if restpose_name in self.restpose_paths:
            npz_path = self.restpose_paths[restpose_name]
            loop = asyncio.get_running_loop()
            restpose = await loop.run_in_executor(
                self.permanent_executor,
                _load_restpose,
                npz_path,)
            # Since the total number of Restpose is small and each Restpose data
            # size is small, cache directly
            self.restpose_cache[restpose_name] = restpose
            return restpose
        msg = f'No restpose data record found for name={restpose_name}.'
        self.logger.error(msg)
        raise KeyError(msg)

    async def _sync_from_meta_reader(self) -> None:
        """Sync data from meta_reader to local cache.

        Synchronizes motion record metadata and builds various index mappings,
        including avatar mapping, type mapping, motion keyword mapping,
        speech keyword mapping, and label mapping.
        """
        # sync meta
        start_time = time.time()
        motion_records = dict()
        avatar_mapping_cfg = self.index_mapping_cfg_template.copy()
        avatar_mapping_cfg['logger_cfg'] = self.logger_cfg
        avatar_mapping = build_index_mapping(avatar_mapping_cfg)
        type_mapping_cfg = self.index_mapping_cfg_template.copy()
        type_mapping_cfg['logger_cfg'] = self.logger_cfg
        type_mapping = build_index_mapping(type_mapping_cfg)
        motion_keyword_mapping_cfg = self.index_mapping_cfg_template.copy()
        motion_keyword_mapping_cfg['logger_cfg'] = self.logger_cfg
        motion_keyword_mapping = build_index_mapping(motion_keyword_mapping_cfg)
        speech_keyword_mapping_cfg = self.index_mapping_cfg_template.copy()
        speech_keyword_mapping_cfg['logger_cfg'] = self.logger_cfg
        speech_keyword_mapping = build_index_mapping(speech_keyword_mapping_cfg)
        label_mapping_cfg = self.index_mapping_cfg_template.copy()
        label_mapping_cfg['logger_cfg'] = self.logger_cfg
        label_mapping = build_index_mapping(label_mapping_cfg)
        motion_record_ids = await self.meta_reader.get_ids()
        for motion_record_id in motion_record_ids:
            motion_record = await self.meta_reader.get_motion_record_by_id(
                motion_record_id)
            motion_records[motion_record_id] = motion_record
            coroutines = list()
            # update avatar mapping
            coroutines.append(avatar_mapping.add_item(
                motion_record.avatar_name, motion_record_id))
            # update type mapping
            if motion_record.is_idle_long:
                coroutines.append(type_mapping.add_item(
                    MotionRecordType.IDLE_LONG.value, motion_record_id))
            if motion_record.is_motion_keyword():
                coroutines.append(type_mapping.add_item(
                    MotionRecordType.MOTION_KEYWORD.value, motion_record_id))
            if motion_record.is_speech_keyword():
                coroutines.append(type_mapping.add_item(
                    MotionRecordType.SPEECH_KEYWORD.value, motion_record_id))
            if motion_record.is_loopable():
                coroutines.append(type_mapping.add_item(
                    MotionRecordType.LOOPABLE.value, motion_record_id))
            if motion_record.is_random():
                coroutines.append(type_mapping.add_item(
                    MotionRecordType.RANDOM.value, motion_record_id))
            # update motion keyword mapping
            if motion_record.is_motion_keyword():
                keywords = motion_record.motion_keyword.motion_keywords_ch
                coroutines.extend(motion_keyword_mapping.add_item(
                    keyword, motion_record_id) for keyword in keywords)
            # update speech keyword mapping
            if motion_record.is_speech_keyword():
                keywords = motion_record.speech_keyword.speech_keywords_ch
                coroutines.extend(speech_keyword_mapping.add_item(
                    keyword, motion_record_id) for keyword in keywords)
            # update label mapping
            labels = motion_record.get_labels()
            if len(labels) > 0:
                coroutines.extend(label_mapping.add_item(
                    label, motion_record_id) for label in labels)
            await asyncio.gather(*coroutines)
        # set attrs
        self.next_motion_records = motion_records
        self.next_avatar_mapping = avatar_mapping
        self.next_type_mapping = type_mapping
        self.next_motion_keyword_mapping = motion_keyword_mapping
        self.next_speech_keyword_mapping = speech_keyword_mapping
        self.next_label_mapping = label_mapping
        self.logger.info(
            f'Meta data sync completed, {len(motion_records)} records, ' +\
            f'took: {time.time() - start_time:.2f}s')

    async def _sync_from_motion_reader(self) -> None:
        """Sync data from motion_reader to local cache.

        Downloads all motion data and saves to temporary directory, using
        multi-threading parallel processing to improve efficiency.
        """
        start_time = time.time()
        motion_paths = dict()
        motion_record_ids = await self.meta_reader.get_ids()
        loop = asyncio.get_running_loop()
        self.next_motion_file_dir = tempfile.TemporaryDirectory(
            prefix='speech2motion_local_cache_')
        with ThreadPoolExecutor(max_workers=self.max_workers) as motion_executor:
            async def _download_and_write(motion_record_id: int,
                                          motion_paths: dict[int, str],
                                          motion_executor: ThreadPoolExecutor) -> None:
                motion_clip = await self.motion_reader.get_motion_clip_by_id(
                    motion_record_id, motion_executor)
                npz_path = os.path.join(self.next_motion_file_dir.name,
                                        f'{motion_record_id:08d}.npz')
                npz_io = await loop.run_in_executor(
                    motion_executor,
                    motion_clip.to_npz)
                with open(npz_path, 'wb') as f:
                    await loop.run_in_executor(
                        motion_executor,
                        f.write,
                        npz_io.getvalue())
                motion_paths[motion_record_id] = npz_path
            await asyncio.gather(*[
                _download_and_write(motion_record_id, motion_paths, motion_executor)
                for motion_record_id in motion_record_ids])
        # set attrs
        self.next_motion_paths = motion_paths
        self.logger.info(
            f'Motion data sync completed, {len(motion_paths)} records, ' +\
            f'took: {time.time() - start_time:.2f}s')

    async def _sync_from_restpose_reader(self) -> None:
        """Sync data from restpose_reader to local cache.

        Downloads all restpose data and saves to temporary directory.
        """
        start_time = time.time()
        restpose_paths = dict()
        restpose_names = await self.restpose_reader.get_restpose_names()
        loop = asyncio.get_running_loop()
        self.next_restpose_file_dir = tempfile.TemporaryDirectory(
            prefix='speech2motion_local_cache_')
        with ThreadPoolExecutor(max_workers=1) as restpose_executor:
            async def _download_and_write(restpose_name: str,
                                          restpose_paths: dict[str, str],
                                          restpose_executor: ThreadPoolExecutor
                                          ) -> None:
                restpose = await self.restpose_reader.get_restpose_by_name(
                    restpose_name)
                npz_path = os.path.join(self.next_restpose_file_dir.name,
                                        f'{restpose_name}.npz')
                npz_io = await loop.run_in_executor(
                    restpose_executor,
                    restpose.to_npz)
                with open(npz_path, 'wb') as f:
                    await loop.run_in_executor(
                        restpose_executor,
                        f.write,
                        npz_io.getvalue())
                restpose_paths[restpose_name] = npz_path
            await asyncio.gather(*[
                _download_and_write(restpose_name, restpose_paths, restpose_executor)
                for restpose_name in restpose_names])
        # set attrs
        self.next_restpose_paths = restpose_paths
        self.logger.info(
            f'Restpose data sync completed, {len(restpose_paths)} records, ' +\
            f'took: {time.time() - start_time:.2f}s')


def _load_motion_clip(npz_path: str) -> MotionClip:
    """Load MotionClip object from NPZ file.

    Args:
        npz_path (str):
            NPZ file path.

    Returns:
        MotionClip:
            Loaded MotionClip object.
    """
    with open(npz_path, 'rb') as f:
        npz_io = io.BytesIO(f.read())
        npz_io.seek(0)
    return MotionClip.from_npz(npz_io)

def _load_restpose(npz_path: str) -> Restpose:
    """Load Restpose object from NPZ file.

    Args:
        npz_path (str):
            NPZ file path.

    Returns:
        Restpose:
            Loaded Restpose object.
    """
    with open(npz_path, 'rb') as f:
        npz_io = io.BytesIO(f.read())
        npz_io.seek(0)
    return Restpose.from_npz(npz_io)
