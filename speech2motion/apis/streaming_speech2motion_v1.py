import asyncio
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from ..cache.builder import build_cache
from ..cache.local_cache import LocalCache
from ..data_structures.motion_clip import MotionClip
from ..data_structures.motion_record import MotionRecord, MotionRecordType
from ..data_structures.timeline import Timeline
from ..filters.builder import (
    AvatarFilter,
    CutoffDurationFilter,
    KeywordAlignFilter,
    KeywordFilter,
    MemoryFilter,
    RandomFilter,
    TypeFilter,
    build_filter,
)
from ..io.meta.builder import build_meta_reader
from ..io.motion.builder import build_motion_reader
from ..merge.builder import build_motion_clip_merge
from ..retrieve import filter_pipeline_retrieve
from ..text_segmentation.builder import build_text_segmentation
from ..utils.super import Super
from ..variety.builder import build_memory

if TYPE_CHECKING:
    from ..io.meta.base_meta_reader import BaseMetaReader
    from ..io.motion.base_motion_reader import BaseMotionReader
    from ..merge.base_merge import BaseMerge
    from ..text_segmentation.jieba_text_segmentation import JiebaTextSegmentation

class StreamingSpeech2MotionV1ChunkStart(BaseModel):
    """V1 streaming motion generation start request.

    Args:
        request_id (str):
            Request ID for identifying an independent streaming motion
            generation request.
        user_id (str):
            User ID for distinguishing the requesting user and calling
            related memory.
        avatar (str):
            Avatar selected by user.
        max_front_extension_duration (float, optional):
            Maximum additional duration that generated motion can extend
            before frame 0. Defaults to 0.0.
        max_rear_extension_duration (float, optional):
            Maximum additional duration that generated motion can extend
            after the last frame. Defaults to 0.0.
        memory_duration_override (float | None, optional):
            Memory duration in seconds. If None, uses the default
            `memory_duration` parameter. Defaults to None.
        first_body_fast_response_override (bool | None, optional):
            Whether to override the fast response configuration of
            for the request. If None, uses the default
            `first_body_fast_response` parameter from api instance.
            Defaults to None.
        idle_long_extendable (bool, optional):
            Whether to allow long idle motions to extend across Chunks
            with higher priority than random motions. Defaults to False.
        return_content (Literal['motion_clip', 'log']):
            Return content type. Defaults to 'motion_clip'.
    """
    request_id: str
    user_id: str
    avatar: str
    max_front_extension_duration: float = 0.0
    max_rear_extension_duration: float = 0.0
    memory_duration_override: float | None = None
    first_body_fast_response_override: bool | None = None
    idle_long_extendable: bool = False
    return_content: Literal['motion_clip', 'log'] = 'motion_clip'

class StreamingSpeech2MotionV1ChunkBody(BaseModel):
    """V1 streaming motion generation body request.

    Args:
        request_id (str):
            Request ID for identifying an independent streaming motion
            generation request.
        duration (float):
            Duration of the body chunk.
        speech_text (str):
            Speech text of the body chunk.
        sequence_number (int):
            Sequence number of current request among all Body requests,
            starting from 0.
        speech_time (list[tuple[int, float]] | None, optional):
            List of tuples containing character index and start time
            in seconds. If None, each character takes equal time.
            Defaults to None.
        motion_keywords (list[tuple[int, str]] | None, optional):
            List of tuples containing start character index and motion
            description keywords. Defaults to None.
    """
    request_id: str
    duration: float
    speech_text: str
    sequence_number: int
    speech_time: list[tuple[int, float]] | None = None
    motion_keywords: list[tuple[int, str]] | None = None

class StreamingSpeech2MotionV1ChunkEnd(BaseModel):
    """V1 streaming motion generation end request.

    Args:
        request_id (str):
            Request ID for identifying an independent streaming motion
            generation request.
    """
    request_id: str

class StreamingSpeech2MotionV1(Super):
    """V1 streaming motion generator.

    This class provides streaming motion generation functionality for V1 API,
    supporting real-time motion generation based on speech input with
    keyword matching, memory management, and smooth transitions.
    """
    FPS: int = 30
    CHUNK_END_SUFFIXES = (
        # Motion record ID selected at the end of previous BodyChunk
        '_chunk_end_motion_record_id',
        # Whether motion clip can be extended from previous BodyChunk end
        # to current BodyChunk start
        '_chunk_end_extendable',
        # Start time of motion clip in-point on timeline at previous BodyChunk end
        '_chunk_end_interval_start',
        # In-point of motion clip selected at previous BodyChunk end
        '_chunk_end_in_point',
        # Out-point of motion clip selected at previous BodyChunk end
        '_chunk_end_out_point',
        # Motion clip selected at previous BodyChunk end, used for transition assistance
        '_chunk_end_payload',
    )

    def __init__(
            self,
            meta_reader_cfg: dict[str, Any],
            motion_reader_cfg: dict[str, Any],
            memory_cfg: dict,
            text_segmentation_cfg: dict[str, Any],
            cache_cfg: dict[str, Any],
            merge_cfg: dict[str, Any],
            request_expire_time: int = 120,
            maintain_check_interval: int = 60,
            max_workers: int = 4,
            thread_pool_executor: ThreadPoolExecutor | None = None,
            first_body_fast_response: bool = False,
            sleep_time: float = 0.01,
            logger_cfg: None | dict[str, Any] = None):
        """Initialize the V1 streaming motion generator.

        Args:
            meta_reader_cfg (dict):
                Metadata reader configuration.
            motion_reader_cfg (dict):
                Motion data reader configuration.
            memory_cfg (dict):
                Memory configuration.
            text_segmentation_cfg (dict):
                Text segmentation configuration.
            cache_cfg (dict):
                Cache configuration.
            merge_cfg (dict):
                Motion merge configuration.
            request_expire_time (int):
                Request expiration time in seconds. Defaults to 120.
                When a request has no new ChunkBody or ChunkEnd requests
                within request_expire_time, it will automatically expire.
            maintain_check_interval (int):
                Maintenance task check interval, including motion library
                cache version check and expired request check.
            max_workers (int):
                Number of worker threads for finding blank intervals
                and data reading during matching.
            thread_pool_executor (ThreadPoolExecutor, optional):
                Thread pool executor for finding blank intervals and
                data reading. If None, creates new thread pool executor
                based on max_workers. Defaults to None.
            first_body_fast_response (bool):
                Whether to fast response on first Body request.
                Defaults to False.
            sleep_time (float):
                Sleep time when waiting for data reading operations
                that are not completed. Defaults to 0.01 seconds.
            logger_cfg (dict | None):
                Logger configuration.
        """
        super().__init__(logger_cfg)
        self.meta_reader_cfg = meta_reader_cfg
        self.motion_reader_cfg = motion_reader_cfg
        self.cache_cfg = cache_cfg
        self.memory_cfg = memory_cfg
        self.text_segmentation_cfg = text_segmentation_cfg
        self.merge_cfg = merge_cfg
        self.request_expire_time = request_expire_time
        self.first_body_fast_response = first_body_fast_response
        self.maintain_check_interval = maintain_check_interval
        if thread_pool_executor is None:
            self.thread_pool_executor = ThreadPoolExecutor(
                max_workers=max_workers)
        else:
            self.thread_pool_executor = thread_pool_executor
        self.executor_external = True \
            if thread_pool_executor is not None \
            else False
        self.sleep_time = sleep_time
        # Space for recording generation requests
        self.request_space: dict[str, Any] = dict()

        # Cache layer and IO instances that cache updates depend on
        self.meta_reader: BaseMetaReader | None = None
        self.motion_reader: BaseMotionReader | None = None
        self.cache: LocalCache | None = None
        self.cache_prepare_task: asyncio.Task | None = None
        self.last_maintain_check_time: float = 0

        # Filters used for retrieval
        self.filters_version: str | None = None
        self.avatar_filter: AvatarFilter | None = None
        self.type_filter: TypeFilter | None = None
        self.motion_keyword_filter: KeywordFilter | None = None
        self.speech_keyword_filter: KeywordFilter | None = None
        self.keyword_align_filter: KeywordAlignFilter | None = None
        self.cutoff_duration_filter: CutoffDurationFilter | None = None
        self.random_filter: RandomFilter | None = None
        self.motion_keyword_memory_filter: MemoryFilter | None = None
        self.speech_keyword_memory_filter: MemoryFilter | None = None
        self.chunk_end_memory_filter: MemoryFilter | None = None
        self.random_memory_filter: MemoryFilter | None = None

        # Text segmentation
        self.text_segmentation: JiebaTextSegmentation | None = None

        # Motion merge
        self.motion_clip_merge: BaseMerge | None = None

        self.startup_called: bool = False

        # Log file path
        log_path = None
        for logger_handler in self.logger.handlers:
            if hasattr(logger_handler, "baseFilename"):
                log_path = logger_handler.baseFilename
                break
        self.log_path = log_path

    def __del__(self) -> None:
        """Destructor, cleanup thread pool executor."""
        if not self.executor_external:
            self.thread_pool_executor.shutdown(wait=True)


    async def startup(self) -> None:
        """Start up the API.

        Initialize all necessary components including readers, cache,
        filters, and merge components.

        Raises:
            RuntimeError:
                Raised when the API has already been started.
        """
        if self.startup_called:
            msg = 'API has already been started, please do not start again.'
            self.logger.error(msg)
            raise RuntimeError(msg)
        self._build_readers()
        self.cache = self._build_cache()
        cache_need_update = await self.cache.need_update()
        if cache_need_update:
            self.logger.info(
                'Performing initial cache synchronization during startup...'
            )
            await self.cache.prepare_next()
            await self.cache.switch_to_next()
        # Filters must be built after self.cache synchronization is complete
        await self._build_filters()
        await self._build_text_segmentation()
        await self._build_motion_clip_merge()
        self.logger.info('Startup completed.')
        self.startup_called = True

    async def handle_chunk_start(
            self, chunk_start: StreamingSpeech2MotionV1ChunkStart) -> None:
        """Handle streaming motion generation start request.

        Args:
            chunk_start (StreamingSpeech2MotionV1ChunkStart):
                Streaming motion generation start request.
        """
        self.logger.info(f'Received start signal for request {chunk_start.request_id}.')
        await self._maintain_check()
        request_dict = chunk_start.model_dump()
        request_dict['last_chunk_end_frame_idx'] = 0
        request_dict['received_sequence_number'] = -1
        request_dict['generated_sequence_number'] = -1
        request_dict['return_content'] = chunk_start.return_content
        request_dict['first_body_fast_response'] = \
            chunk_start.first_body_fast_response_override\
            if chunk_start.first_body_fast_response_override is not None\
            else self.first_body_fast_response
        request_dict['idle_long_extendable'] = chunk_start.idle_long_extendable
        request_dict['last_request_time'] = time.time()
        for suffix in self.__class__.CHUNK_END_SUFFIXES:
            request_dict[f'last{suffix}'] = None
            request_dict[f'current{suffix}'] = None
        self.request_space[chunk_start.request_id] = request_dict

    async def handle_chunk_body(
            self,
            chunk_body: StreamingSpeech2MotionV1ChunkBody
            ) -> MotionClip | str:
        """Handle streaming motion generation body request.

        Args:
            chunk_body (StreamingSpeech2MotionV1ChunkBody):
                Streaming motion generation body request.

        Returns:
            MotionClip | str:
                Return result. If return_content is 'motion_clip', returns
                MotionClip instance, otherwise returns log string.
        """
        chunk_body_msg = f'Received BodyChunk request {chunk_body.sequence_number} ' +\
            f'for request {chunk_body.request_id}.'
        self.logger.info(chunk_body_msg)
        msg = f'Speech duration: {chunk_body.duration}s, ' +\
            f'speech text: {chunk_body.speech_text}\n' +\
            f'speech time list: {chunk_body.speech_time}\n' +\
            f'motion keywords list: {chunk_body.motion_keywords}'
        self.logger.debug(msg)
        body_duration = chunk_body.duration
        body_n_frames = int(body_duration * self.__class__.FPS)
        if body_n_frames < 1:
            msg = f'Request {chunk_body.request_id} ' +\
                f'BodyChunk {chunk_body.sequence_number} ' + \
                'duration is too short, less than one frame time, only ' +\
                f'{body_duration:.5f} seconds, cannot match.'
            self.logger.error(msg)
            raise ValueError(msg)
        request_dict = self.request_space[chunk_body.request_id]
        request_dict['received_sequence_number'] = max(
            request_dict['received_sequence_number'],
            chunk_body.sequence_number
        )
        request_dict['last_request_time'] = time.time()
        await self._maintain_check()
        while request_dict['generated_sequence_number'] < \
                chunk_body.sequence_number - 1:
            await asyncio.sleep(self.sleep_time)
        start_time = time.time()
        body_start_frame = request_dict['last_chunk_end_frame_idx']
        body_start_time = body_start_frame / self.__class__.FPS
        max_front_extension_duration = request_dict['max_front_extension_duration']
        max_front_extension_n_frames = int(
            max_front_extension_duration * self.__class__.FPS)
        max_rear_extension_duration = request_dict['max_rear_extension_duration']
        max_rear_extension_n_frames = int(
            max_rear_extension_duration * self.__class__.FPS)
        if max_front_extension_n_frames > 0 or max_rear_extension_n_frames > 0:
            extension_enabled = True
        else:
            extension_enabled = False
        if chunk_body.speech_time is None:
            speech_time = list()
            msg = f'Request {chunk_body.request_id} ' +\
                f'BodyChunk {chunk_body.sequence_number} ' +\
                'has no speech time information, predicting speech time ' +\
                'based on uniform distribution.'
            self.logger.warning(msg)
        else:
            speech_time = chunk_body.speech_time
        loop = asyncio.get_running_loop()
        full_time_list = await loop.run_in_executor(
            self.thread_pool_executor,
            self._convert_speech_time,
            speech_time,
            chunk_body.speech_text,
            body_duration,
            body_start_time)
        timeline = Timeline(
            start_frame=body_start_frame,
            end_frame=body_start_frame + body_n_frames,
            enable_extension=extension_enabled,
            logger_cfg=self.logger_cfg,
        )
        first_body_fast_response = request_dict['first_body_fast_response']
        if not (body_start_frame == 0 and first_body_fast_response):
            # Motion keyword matching
            motion_keywords = chunk_body.motion_keywords\
                if chunk_body.motion_keywords is not None\
                else list()
            timeline = await self._match_motion_keywords(
                request_id=chunk_body.request_id,
                timeline=timeline,
                full_time_list=full_time_list,
                motion_keywords=motion_keywords
            )
            # Speech keyword matching
            timeline = await self._match_speech_keywords(
                request_id=chunk_body.request_id,
                timeline=timeline,
                full_time_list=full_time_list,
                speech_text=chunk_body.speech_text
            )
            # Extend previous chunk end
            smooth_start, timeline = await self._extend_previous_chunk_end(
                request_id=chunk_body.request_id,
                timeline=timeline
            )
            if not smooth_start:
                last_chunk_end_payload = \
                    request_dict['last_chunk_end_payload']
            # Handle current chunk end
            timeline = await self._handle_current_chunk_end(
                request_id=chunk_body.request_id,
                timeline=timeline
            )
        else:
            msg = f'Request {chunk_body.request_id} ' +\
                f'BodyChunk {chunk_body.sequence_number} ' + \
                'is first Body request, enabling fast response, ' +\
                'skipping motion keyword and speech keyword matching.'
            self.logger.debug(msg)
            smooth_start = False
            last_chunk_end_payload = None
        # Fill up with random motion
        timeline = await self._fill_up_with_random_motion(
            request_id=chunk_body.request_id,
            timeline=timeline
        )
        # Fill up with long IDLE motion
        timeline = await self._fill_up_with_idle_long_motion(
            request_id=chunk_body.request_id,
            timeline=timeline
        )
        request_dict['last_chunk_end_frame_idx'] = timeline.end
        await self._update_last_chunk_end(chunk_body.request_id)
        table_task = asyncio.wrap_future(
            loop.run_in_executor(
                self.thread_pool_executor,
                timeline.to_table
            )
        )
        self.logger.info(
            f'Request {chunk_body.request_id} BodyChunk {chunk_body.sequence_number} ' +
            f'matching completed, took {time.time() - start_time:.2f} seconds.'
        )
        if request_dict['return_content'] == 'motion_clip':
            waiting_for_read_start = time.time()
            motion_clips = await self._wait_for_motion_clips(timeline)
            end_time = time.time()
            self.logger.debug(
                f'Request {chunk_body.request_id} ' +\
                f'BodyChunk {chunk_body.sequence_number} ' + \
                f'finishes loading {len(motion_clips)} motion clips, ' + \
                f'took {end_time - waiting_for_read_start:.2f} seconds.'
            )
            merge_start_time = time.time()
            # First BodyChunk request has last_chunk_end_payload as None
            if not smooth_start and last_chunk_end_payload is not None:
                msg = f'Request {chunk_body.request_id} ' +\
                    f'BodyChunk {chunk_body.sequence_number} ' + \
                    'chunk start cannot smoothly connect with previous chunk end, ' + \
                    'transition animation required.'
                self.logger.debug(msg)
                motion_clips = await self._smoothen_start(
                    last_chunk_end_payload=last_chunk_end_payload,
                    motion_clips=motion_clips)
        # After updating smoothen_start with last_chunk_end_payload,
        # set generated_sequence_number to current BodyChunk's sequence_number
        request_dict['generated_sequence_number'] = chunk_body.sequence_number
        table_str = await table_task
        self.logger.info(
            f'Request {chunk_body.request_id} BodyChunk {chunk_body.sequence_number} ' +
            'matching result table conversion completed, ' +
            f'matching result:\n{table_str}'
        )
        if request_dict['return_content'] == 'motion_clip':
            motion_clip = await self.motion_clip_merge.merge(motion_clips)
            merge_end_time = time.time()
            self.logger.debug(
                f'Request {chunk_body.request_id} ' +\
                f'BodyChunk {chunk_body.sequence_number} ' + \
                'merge completed, ' + \
                f'took {merge_end_time - merge_start_time:.2f} seconds.'
            )
            motion_clip.set_timeline_start_idx(timeline.start)
            ret_value = motion_clip
        else:
            ret_value = await self._read_log_file(
                chunk_body_msg)
        return ret_value

    async def handle_chunk_end(
            self, chunk_end: StreamingSpeech2MotionV1ChunkEnd
            ) -> MotionClip | str | None:
        """Handle streaming motion generation end request.

        Args:
            chunk_end (StreamingSpeech2MotionV1ChunkEnd):
                Streaming motion generation end request.

        Returns:
            MotionClip | str | None:
                Returns None if there is no extension data needed based on
                previous requests. If extension is needed, returns MotionClip
                instance if return_content is 'motion_clip', otherwise returns
                log string.
        """
        start_time = time.time()
        chunk_end_msg = f'Received ChunkEnd request for request {chunk_end.request_id}.'
        self.logger.info(chunk_end_msg)
        request_dict = self.request_space[chunk_end.request_id]
        request_dict['last_request_time'] = start_time
        await self._maintain_check()
        while request_dict['generated_sequence_number'] < \
                request_dict['received_sequence_number']:
            await asyncio.sleep(self.sleep_time)
        last_chunk_end_extendable = request_dict['last_chunk_end_extendable']
        if not last_chunk_end_extendable or \
                request_dict['last_chunk_end_frame_idx'] == 0:
            msg = f'Request {chunk_end.request_id} has no unplayed motion ' + \
                'at stream end, ChunkEnd request returns empty.'
            self.logger.info(msg)
            ret_value = None
        elif request_dict['last_chunk_end_frame_idx'] >= \
                request_dict['last_chunk_end_interval_start'] + \
                (request_dict['last_chunk_end_out_point'] -
                 request_dict['last_chunk_end_in_point']):
            msg = f'Request {chunk_end.request_id} has no unplayed motion ' + \
                'at stream end, ChunkEnd request returns empty.'
            self.logger.info(msg)
            ret_value = None
        else:
            last_mc_extend_end = request_dict['last_chunk_end_interval_start'] + \
                (request_dict['last_chunk_end_out_point'] -
                 request_dict['last_chunk_end_in_point'])
            timeline = Timeline(
                start_frame=request_dict['last_chunk_end_frame_idx'],
                end_frame=last_mc_extend_end,
                enable_extension=True,
                logger_cfg=self.logger_cfg,
            )
            # Extend previous chunk end
            smooth_start, timeline = await self._extend_previous_chunk_end(
                request_id=chunk_end.request_id,
                timeline=timeline
            )
            if not smooth_start:
                last_chunk_end_payload = \
                    request_dict['last_chunk_end_payload']
            else:
                last_chunk_end_payload = None
            loop = asyncio.get_running_loop()
            # ChunkEnd is not first request, blocking time for to_table is not sensitive
            table_str = await loop.run_in_executor(
                self.thread_pool_executor,
                timeline.to_table
            )
            msg = f'Request {chunk_end.request_id} at stream end ' + \
                'matched unplayed motion, took ' +\
                f'{time.time() - start_time:.2f} seconds, ' + \
                'matching result:\n' + \
                f'\n{table_str}'
            self.logger.info(msg)
            if request_dict['return_content'] == 'motion_clip':
                waiting_for_read_start = time.time()
                motion_clips = await self._wait_for_motion_clips(timeline)
                end_time = time.time()
                self.logger.debug(
                    f'Request {chunk_end.request_id} stream end request ' + \
                    f'finishes loading {len(motion_clips)} motion clips, ' + \
                    f'took {end_time - waiting_for_read_start:.2f} seconds.'
                )
                merge_start_time = time.time()
                if not smooth_start and last_chunk_end_payload is not None:
                    msg = f'Request {chunk_end.request_id} stream end request ' + \
                        'chunk start cannot smoothly connect ' +\
                        'with previous chunk end, ' + \
                        'transition animation required.'
                    self.logger.debug(msg)
                    motion_clips = await self._smoothen_start(
                        last_chunk_end_payload=last_chunk_end_payload,
                        motion_clips=motion_clips
                    )
                motion_clip = await self.motion_clip_merge.merge(motion_clips)
                merge_end_time = time.time()
                self.logger.debug(
                    f'Request {chunk_end.request_id} stream end request ' + \
                    'merge completed, took ' +\
                    f'{merge_end_time - merge_start_time:.2f} seconds.'
                )
                motion_clip.set_timeline_start_idx(timeline.start)
                ret_value = motion_clip
            else:
                ret_value = await self._read_log_file(
                    chunk_end_msg)
        self.logger.info(
            f'Request {chunk_end.request_id} stream end request processing completed.')
        self.request_space.pop(chunk_end.request_id)
        return ret_value

    def _convert_speech_time(
            self,
            speech_time: list[tuple[int, float]],
            speech_text: str,
            duration: float,
            base_time: float = 0.0) -> list[tuple[int, float]]:
        """Predict speech time for each character based on incomplete speech time.

        Args:
            speech_time (list[tuple[int, float]]):
                Speech time list containing tuples of character index and
                start time in seconds. If the first character is spoken,
                speech time likely starts from 0.
            speech_text (str):
                Speech text.
            duration (float):
                Total duration corresponding to speech_text.
            base_time (float, optional):
                Audio duration that already exists before this speech starts,
                will be added to predicted speech time. Defaults to 0.0.

        Returns:
            list[tuple[int, float]]:
                Predicted speech time list containing tuples of character index
                and start time in seconds. Character indices are continuous,
                starting from 0, with total list length equal to len(speech_text).
        """
        # full_time_list is a list of start time,
        # for every character in speech_text
        full_time_list = []
        last_idx = -1
        last_time = 0.0
        for char_idx, char_time in speech_time:
            if char_idx <= last_idx:
                continue
            elif last_idx + 1 < char_idx:
                # use average time to fill the gap
                # for the missing characters
                avg_time = (char_time - last_time) / (char_idx - last_idx)
                for i in range(last_idx + 1, char_idx):
                    times = i - last_idx
                    full_time_list.append(last_time + avg_time * times)
            # use input time for the current character
            full_time_list.append(char_time)
            last_idx = char_idx
            last_time = char_time
        # fill time for the remaining characters
        if last_idx + 1 < len(speech_text):
            avg_time = (duration - last_time) / (len(speech_text) - last_idx)
            for i in range(last_idx + 1, len(speech_text)):
                times = i - last_idx
                full_time_list.append(last_time + avg_time * times)
        if len(full_time_list) != len(speech_text):
            msg = 'Predicted speech time list length ' +\
                'does not match speech text length, ' + \
                f'len(full_time_list): {len(full_time_list)}, ' + \
                f'len(speech_text): {len(speech_text)}\n' + \
                f'speech_text: {speech_text}\n' + \
                f'speech_time: {speech_time}\n' + \
                f'full_time_list: {full_time_list}'
            self.logger.error(msg)
            raise ValueError(msg)
        if base_time > 0.0:
            full_time_list = [t + base_time for t in full_time_list]
        return full_time_list

    async def _match_motion_keywords(
            self,
            request_id: str,
            timeline: Timeline,
            full_time_list: list[tuple[int, float]],
            motion_keywords: list[tuple[int, str]]) -> Timeline:
        """Match motion keywords.

        Args:
            request_id (str):
                Request ID.
            timeline (Timeline):
                Motion timeline.
            full_time_list (list[tuple[int, float]]):
                Predicted speech time list.
            motion_keywords (list[tuple[int, str]]):
                Motion keywords list.

        Returns:
            Timeline:
                Processed motion timeline.
        """
        start_time = time.time()
        if len(motion_keywords) == 0:
            return timeline
        request_dict = self.request_space[request_id]
        sequence_number = request_dict['generated_sequence_number'] + 1
        last_chunk_end_motion_record_id = \
            request_dict['last_chunk_end_motion_record_id']
        max_front_extension_duration = \
            request_dict['max_front_extension_duration']
        max_front_extension_n_frames = int(
            max_front_extension_duration * self.__class__.FPS)
        max_rear_extension_duration = \
            request_dict['max_rear_extension_duration']
        max_rear_extension_n_frames = int(
            max_rear_extension_duration * self.__class__.FPS)
        filter_pipeline = [
            self.avatar_filter,
            self.type_filter,
            self.motion_keyword_filter,
            self.keyword_align_filter,
            self.motion_keyword_memory_filter,
            self.speech_keyword_memory_filter
        ]
        pipeline_input_template = dict(
            # All candidate motion records
            motion_records=self.cache.motion_records,
            # For avatar_filter
            avatar=request_dict['avatar'],
            # For type_filter
            motion_record_type=MotionRecordType.MOTION_KEYWORD,
            # For motion_keyword_filter
            keyword=None,
            # For keyword_align_filter
            align_frame=None,
            start_frame_lowerbound=None,
            end_frame_upperbound=None,
            keyword_attr_name='motion_keyword',
            # For motion_keyword_memory_filter and speech_keyword_memory_filter
            user_id=request_dict['user_id'],
        )
        intervals_to_skip = list()
        if last_chunk_end_motion_record_id is not None:
            last_interval_start = \
                request_dict['last_chunk_end_interval_start']
            last_in_point = \
                request_dict['last_chunk_end_in_point']
            last_out_point = \
                request_dict['last_chunk_end_out_point']
            last_interval_end = last_interval_start + \
                (last_out_point - last_in_point)
            timeline_start = timeline.start
            intervals_to_skip.append((timeline_start, last_interval_end))
            # Record placeholder in memory
            n_frames_after_start = last_interval_end - timeline_start
            await self.motion_keyword_memory_filter.memory.remember(
                user_id=request_dict['user_id'],
                event_id=-1,
                event_duration=n_frames_after_start / self.__class__.FPS,
            )
            last_memory_end_frame = last_interval_end
        else:
            last_memory_end_frame = timeline.start
        loop = asyncio.get_running_loop()
        blank_interval = await loop.run_in_executor(
            self.thread_pool_executor,
            timeline.get_next_blank, intervals_to_skip)
        while blank_interval is not None:
            interval_start = blank_interval[0]
            pipeline_input = pipeline_input_template.copy()
            if interval_start == 0:
                pipeline_input['start_frame_lowerbound'] = \
                    0 - max_front_extension_n_frames
            else:
                pipeline_input['start_frame_lowerbound'] = interval_start
            interval_end = blank_interval[1]
            if interval_end == timeline.end:
                pipeline_input['end_frame_upperbound'] = \
                    timeline.end + max_rear_extension_n_frames
            else:
                pipeline_input['end_frame_upperbound'] = interval_end
            interval_matched = False
            for motion_keyword in motion_keywords:
                keyword_idx = motion_keyword[0]
                keyword_str = motion_keyword[1]
                align_time = full_time_list[keyword_idx]
                align_frame = int(align_time * self.__class__.FPS)
                if align_frame < interval_start:
                    continue
                elif align_frame > interval_end:
                    break
                pipeline_input['align_frame'] = align_frame
                pipeline_input['keyword'] = keyword_str
                match_results = await filter_pipeline_retrieve(
                    filter_pipeline=filter_pipeline,
                    pipeline_input=pipeline_input,
                    return_candidates=False
                )
                if len(match_results) > 0:
                    motion_record_id = next(iter(match_results.keys()))
                    motion_record = match_results[motion_record_id]
                    n_frames = motion_record.n_frames
                    keyword_frame = motion_record.motion_keyword.motion_keyword_frame
                    return_content = request_dict['return_content']
                    preload_task = asyncio.create_task(
                        self.cache.get_motion_clip_by_id(motion_record_id)
                    ) if return_content == 'motion_clip' else None
                    start_idx = align_frame - keyword_frame
                    # To ensure timeline end position matches request body requirements
                    # this_chunk_outpoint may end earlier than actual motion length
                    # remaining animation will be supplemented in next BodyChunk
                    cross_chunk_outpoint = n_frames
                    this_chunk_outpoint = n_frames \
                        if start_idx + n_frames <= timeline.end \
                        else timeline.end - start_idx
                    end_idx = start_idx + this_chunk_outpoint
                    insert_success, timeline = await self._try_to_insert(
                        timeline=timeline,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        motion_record=motion_record,
                        in_point=0,
                        out_point=this_chunk_outpoint,
                        trigger=f'MotionKeyword-{keyword_str}',
                        preload_task=preload_task
                    )
                    if not insert_success:
                        msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                            'motion keyword insertion failed, ' +\
                            f'motion ID: {motion_record_id}, ' +\
                            'in-point: 0, ' +\
                            f'out-point: {this_chunk_outpoint}'
                        self.logger.error(msg)
                        interval_matched = False
                        break
                    else:
                        # Record placeholder in memory, considering forward extension
                        if start_idx > last_memory_end_frame:
                            place_holder_duration = (
                                start_idx - last_memory_end_frame
                            ) / self.__class__.FPS
                            await self.motion_keyword_memory_filter.memory.remember(
                                user_id=request_dict['user_id'],
                                event_id=-1,
                                event_duration=place_holder_duration,
                            )
                        # Add motion_record_id to memory
                        await self.motion_keyword_memory_filter.memory.remember(
                            user_id=request_dict['user_id'],
                            event_id=motion_record_id,
                            event_duration=n_frames / self.__class__.FPS,
                        )
                        last_memory_end_frame = end_idx
                        if end_idx == timeline.end:
                            request_dict['current_chunk_end_motion_record_id'] = \
                                motion_record_id
                            request_dict['current_chunk_end_extendable'] = False \
                                if cross_chunk_outpoint == this_chunk_outpoint \
                                else True
                            request_dict['current_chunk_end_interval_start'] = start_idx
                            request_dict['current_chunk_end_in_point'] = 0
                            request_dict['current_chunk_end_out_point'] = \
                                cross_chunk_outpoint
                            request_dict['current_chunk_end_payload'] = dict(
                                preload_task=preload_task,
                                in_point=0,
                                out_point=this_chunk_outpoint,
                            )
                        interval_matched = True
                        break
            if not interval_matched:
                intervals_to_skip.append(blank_interval)
            blank_interval = await loop.run_in_executor(
                self.thread_pool_executor,
                timeline.get_next_blank,
                intervals_to_skip
            )
        if last_memory_end_frame < timeline.end:
            place_holder_duration = (
                timeline.end - last_memory_end_frame
            ) / self.__class__.FPS
            await self.motion_keyword_memory_filter.memory.remember(
                user_id=request_dict['user_id'],
                event_id=-1,
                event_duration=place_holder_duration,
            )
        end_time = time.time()
        sequence_number = request_dict['generated_sequence_number'] + 1
        self.logger.info(
            f'Request {request_id} BodyChunk {sequence_number} ' +\
            'motion keyword matching completed, ' +
            f'took {end_time - start_time:.2f} seconds.'
        )
        return timeline

    async def _match_speech_keywords(
            self,
            request_id: str,
            timeline: Timeline,
            full_time_list: list[tuple[int, float]],
            speech_text: str) -> Timeline:
        """Match speech keywords.

        Args:
            request_id (str):
                Request ID.
            timeline (Timeline):
                Motion timeline.
            full_time_list (list[tuple[int, float]]):
                Predicted speech time list.
            speech_text (str):
                Speech text.

        Returns:
            Timeline:
                Processed motion timeline.
        """
        start_time = time.time()
        request_dict = self.request_space[request_id]
        sequence_number = request_dict['generated_sequence_number'] + 1
        last_chunk_end_motion_record_id = \
            request_dict['last_chunk_end_motion_record_id']
        max_front_extension_duration = \
            request_dict['max_front_extension_duration']
        max_front_extension_n_frames = int(
            max_front_extension_duration * self.__class__.FPS)
        max_rear_extension_duration = \
            request_dict['max_rear_extension_duration']
        max_rear_extension_n_frames = int(
            max_rear_extension_duration * self.__class__.FPS)
        filter_pipeline = [
            self.avatar_filter,
            self.type_filter,
            self.speech_keyword_filter,
            self.keyword_align_filter,
            self.speech_keyword_memory_filter,
            self.speech_keyword_memory_filter
        ]
        keywords_set = await self.speech_keyword_filter.mapping.keys()
        pipeline_input_template = dict(
            # All candidate motion records
            motion_records=self.cache.motion_records,
            # For avatar_filter
            avatar=request_dict['avatar'],
            # For type_filter
            motion_record_type=MotionRecordType.SPEECH_KEYWORD,
            # For motion_keyword_filter
            keyword=None,
            # For keyword_align_filter
            align_frame=None,
            start_frame_lowerbound=None,
            end_frame_upperbound=None,
            keyword_attr_name='speech_keyword',
            # For motion_keyword_memory_filter and speech_keyword_memory_filter
            user_id=request_dict['user_id'],
        )
        intervals_to_skip = list()
        if last_chunk_end_motion_record_id is not None:
            last_interval_start = \
                request_dict['last_chunk_end_interval_start']
            last_in_point = \
                request_dict['last_chunk_end_in_point']
            last_out_point = \
                request_dict['last_chunk_end_out_point']
            last_interval_end = last_interval_start + \
                (last_out_point - last_in_point)
            timeline_start = timeline.start
            intervals_to_skip.append((timeline_start, last_interval_end))
            # Record placeholder in memory
            n_frames_after_start = last_interval_end - timeline_start
            await self.speech_keyword_memory_filter.memory.remember(
                user_id=request_dict['user_id'],
                event_id=-1,
                event_duration=n_frames_after_start / self.__class__.FPS,
            )
            last_memory_end_frame = last_interval_end
        else:
            last_memory_end_frame = timeline.start
        loop = asyncio.get_running_loop()
        text_segments = await loop.run_in_executor(
            self.thread_pool_executor,
            self.text_segmentation.cut_text,
            speech_text
        )
        blank_interval = await loop.run_in_executor(
            self.thread_pool_executor,
            timeline.get_next_blank, intervals_to_skip)
        while blank_interval is not None:
            interval_start = blank_interval[0]
            pipeline_input = pipeline_input_template.copy()
            if interval_start == 0:
                pipeline_input['start_frame_lowerbound'] = \
                    0 - max_front_extension_n_frames
            else:
                pipeline_input['start_frame_lowerbound'] = interval_start
            interval_end = blank_interval[1]
            if interval_end == timeline.end:
                pipeline_input['end_frame_upperbound'] = \
                    timeline.end + max_rear_extension_n_frames
            else:
                pipeline_input['end_frame_upperbound'] = interval_end
            # Select text within time range
            interval_start_time = blank_interval[0] * self.__class__.FPS
            interval_end_time = blank_interval[1] * self.__class__.FPS
            start_char_idx = 0
            end_char_idx = len(speech_text)
            for char_idx, char_time in enumerate(full_time_list):
                if char_time < interval_start_time:
                    continue
                if start_char_idx == 0:
                    start_char_idx = char_idx
                if char_time > interval_end_time:
                    end_char_idx = char_idx
                    break
            interval_matched = False
            for text_segment in text_segments:
                char_idx = text_segment['idx']
                if char_idx < start_char_idx:
                    continue
                elif char_idx > end_char_idx:
                    break
                keyword_str = text_segment['str']
                if len(keyword_str.strip()) == 0 or keyword_str not in keywords_set:
                    continue
                align_time = full_time_list[char_idx]
                align_frame = int(align_time * self.__class__.FPS)
                pipeline_input['align_frame'] = align_frame
                pipeline_input['keyword'] = keyword_str
                match_results = await filter_pipeline_retrieve(
                    filter_pipeline=filter_pipeline,
                    pipeline_input=pipeline_input,
                    return_candidates=False
                )
                if len(match_results) > 0:
                    motion_record_id = next(iter(match_results.keys()))
                    motion_record = match_results[motion_record_id]
                    n_frames = motion_record.n_frames
                    keyword_frame = motion_record.motion_keyword.motion_keyword_frame
                    return_content = request_dict['return_content']
                    preload_task = asyncio.create_task(
                        self.cache.get_motion_clip_by_id(motion_record_id)
                    ) if return_content == 'motion_clip' else None
                    start_idx = align_frame - keyword_frame
                    # To ensure timeline end position matches request body requirements
                    # this_chunk_outpoint may end earlier than actual motion length
                    # remaining animation will be supplemented in next BodyChunk
                    cross_chunk_outpoint = n_frames
                    this_chunk_outpoint = n_frames \
                        if start_idx + n_frames <= timeline.end \
                        else timeline.end - start_idx
                    end_idx = start_idx + this_chunk_outpoint
                    insert_success, timeline = await self._try_to_insert(
                        timeline=timeline,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        motion_record=motion_record,
                        in_point=0,
                        out_point=this_chunk_outpoint,
                        trigger=f'SpeechKeyword-{text_segment["str"]}',
                        preload_task=preload_task
                    )
                    if not insert_success:
                        msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                            'speech keyword insertion failed, ' +\
                            f'motion ID: {motion_record_id}, ' +\
                            'in-point: 0, ' +\
                            f'out-point: {this_chunk_outpoint}'
                        self.logger.error(msg)
                        interval_matched = False
                        break
                    else:
                        # Record placeholder in memory, considering forward extension
                        if start_idx > last_memory_end_frame:
                            place_holder_duration = (
                                start_idx - last_memory_end_frame
                            ) / self.__class__.FPS
                            await self.speech_keyword_memory_filter.memory.remember(
                                user_id=request_dict['user_id'],
                                event_id=-1,
                                event_duration=place_holder_duration,
                            )
                        # Add motion_record_id to memory
                        await self.speech_keyword_memory_filter.memory.remember(
                            user_id=request_dict['user_id'],
                            event_id=motion_record_id,
                            event_duration=n_frames / self.__class__.FPS,
                        )
                        last_memory_end_frame = end_idx
                        if end_idx == timeline.end:
                            request_dict['current_chunk_end_motion_record_id'] = \
                                motion_record_id
                            request_dict['current_chunk_end_extendable'] = False \
                                if cross_chunk_outpoint == this_chunk_outpoint \
                                else True
                            request_dict['current_chunk_end_interval_start'] = start_idx
                            request_dict['current_chunk_end_in_point'] = 0
                            request_dict['current_chunk_end_out_point'] = \
                                cross_chunk_outpoint
                            request_dict['current_chunk_end_payload'] = dict(
                                preload_task=preload_task,
                                in_point=0,
                                out_point=this_chunk_outpoint,
                            )
                        interval_matched = True
                        break
            if not interval_matched:
                intervals_to_skip.append(blank_interval)
            blank_interval = await loop.run_in_executor(
                self.thread_pool_executor,
                timeline.get_next_blank,
                intervals_to_skip
            )
        if last_memory_end_frame < timeline.end:
            place_holder_duration = (
                timeline.end - last_memory_end_frame
            ) / self.__class__.FPS
            await self.speech_keyword_memory_filter.memory.remember(
                user_id=request_dict['user_id'],
                event_id=-1,
                event_duration=place_holder_duration,
            )
        end_time = time.time()
        sequence_number = request_dict['generated_sequence_number'] + 1
        self.logger.debug(
            f'Request {request_id} BodyChunk {sequence_number} ' +\
            'speech keyword matching completed, ' +
            f'took {end_time - start_time:.2f} seconds.'
        )
        return timeline

    async def _extend_previous_chunk_end(
            self,
            request_id: str,
            timeline: Timeline) -> tuple[bool, Timeline]:
        """Extend the last motion of previous Chunk end.

        Args:
            request_id (str):
                Request ID.
            timeline (Timeline):
                Motion timeline.

        Returns:
            tuple[bool, Timeline]:
                Whether current Chunk Timeline start smoothly connects with
                previous Chunk end. If successfully extended, returns True
                and processed Timeline; if failed, returns False and original
                Timeline.
        """
        start_time = time.time()
        request_dict = self.request_space[request_id]
        sequence_number = request_dict['generated_sequence_number'] + 1
        last_motion_record_id = \
            request_dict['last_chunk_end_motion_record_id']
        last_extendable = \
            request_dict['last_chunk_end_extendable']
        if last_motion_record_id is None or not last_extendable:
            return False, timeline
        last_interval_start = \
            request_dict['last_chunk_end_interval_start']
        last_in_point = \
            request_dict['last_chunk_end_in_point']
        last_out_point = \
            request_dict['last_chunk_end_out_point']
        this_in_point = \
            timeline.start - last_interval_start + last_in_point
        loop = asyncio.get_running_loop()
        first_blank_interval = await loop.run_in_executor(
            self.thread_pool_executor,
            timeline.get_next_blank
        )
        if first_blank_interval is None or first_blank_interval[0] > timeline.start:
            msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                'previous chunk end extension failed, ' +\
                'no blank interval at timeline start.'
            self.logger.warning(msg)
            return False, timeline
        interval_start = first_blank_interval[0]
        interval_end = first_blank_interval[1]
        cutoff_duration_upperbound = (
            interval_end - last_interval_start + last_in_point) / self.__class__.FPS
        cutoff_duration_lowerbound = (last_out_point + 1) \
            / self.__class__.FPS
        last_motion_record = self.cache.motion_records[last_motion_record_id]
        latest_cutoff_frame = last_out_point
        while True:
            earliest_cutoff_frame = \
                await self.cutoff_duration_filter.get_earliest_cutoff_frame(
                    motion_record=last_motion_record,
                    cutoff_duration_lowerbound=cutoff_duration_lowerbound,
                    cutoff_duration_upperbound=cutoff_duration_upperbound
                )
            if earliest_cutoff_frame is None:
                break
            latest_cutoff_frame = earliest_cutoff_frame
            cutoff_duration_lowerbound = (latest_cutoff_frame + 1) \
                / self.__class__.FPS
        if latest_cutoff_frame == last_out_point:
            if latest_cutoff_frame == this_in_point:
                msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                    'previous chunk end extension attempt failed, ' + \
                    'previous motion just ended at last BodyChunk end, '
                ret_value = False
            else:
                msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                    'previous chunk end extension attempt failed, ' + \
                    'previous motion still ends ' +\
                    f'at last match out_point={last_out_point}, '
                ret_value = True
        # latest_cutoff_frame > last_out_point:
        else:
            msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                'previous chunk end ' +\
                'extension attempt successful, ' + \
                f'out_point extended from {last_out_point} to {latest_cutoff_frame}, '
            ret_value = True
        if ret_value:
            return_content = request_dict['return_content']
            preload_task = asyncio.create_task(
                self.cache.get_motion_clip_by_id(last_motion_record_id)
            ) if return_content == 'motion_clip' else None
            cross_chunk_outpoint = latest_cutoff_frame
            this_chunk_outpoint = min(
                latest_cutoff_frame,
                interval_end - last_interval_start + last_in_point)
            n_frames_insert = this_chunk_outpoint - this_in_point
            insert_success, timeline = await self._try_to_insert(
                timeline=timeline,
                start_idx=interval_start,
                end_idx=interval_start + n_frames_insert,
                motion_record=last_motion_record,
                in_point=this_in_point,
                out_point=this_chunk_outpoint,
                trigger='PreviousExtension',
                preload_task=preload_task
            )
            if not insert_success:
                msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                    'previous chunk end extension attempt failed, ' + \
                    'because insertion failed and could not be processed successfully, '
                ret_value = False
            if interval_start + n_frames_insert == timeline.end:
                request_dict['current_chunk_end_motion_record_id'] = \
                    last_motion_record_id
                request_dict['current_chunk_end_extendable'] = False \
                    if cross_chunk_outpoint == this_chunk_outpoint \
                    else True
                request_dict['current_chunk_end_interval_start'] = interval_start
                request_dict['current_chunk_end_in_point'] = this_in_point
                request_dict['current_chunk_end_out_point'] = cross_chunk_outpoint
                request_dict['current_chunk_end_payload'] = dict(
                    preload_task=preload_task,
                    in_point=this_in_point,
                    out_point=this_chunk_outpoint,
                )
        end_time = time.time()
        msg += f'took {end_time - start_time:.2f} seconds.'
        self.logger.debug(msg)
        return ret_value, timeline

    async def _handle_current_chunk_end(
            self,
            request_id: str,
            timeline: Timeline) -> Timeline:
        """Handle current Chunk end.

        Returns:
            Timeline:
                Processed motion timeline.
        """
        start_time = time.time()
        request_dict = self.request_space[request_id]
        sequence_number = request_dict['generated_sequence_number'] + 1
        loop = asyncio.get_running_loop()
        intervals_to_skip = list()
        final_interval = None
        last_memory_end_frame = timeline.start
        while True:
            blank_interval = await loop.run_in_executor(
                self.thread_pool_executor,
                timeline.get_next_blank,
                intervals_to_skip
            )
            if blank_interval is None:
                break
            elif blank_interval[1] == timeline.end:
                final_interval = blank_interval
                break
            elif blank_interval[1] > timeline.end:
                break
            # blank_interval[1] < timeline.end:
            else:
                intervals_to_skip.append(blank_interval)
        if final_interval is None:
            # last_chunk_end related values should have been
            # processed when motion matching succeeded
            msg = f'Request {request_id} BodyChunk {sequence_number} end ' + \
                'has been processed in higher priority matching, '
        else:
            interval_n_frames = final_interval[1] - final_interval[0]
            duration = interval_n_frames / self.__class__.FPS
            duration_extended = duration + request_dict['max_rear_extension_duration']
            # Match a speech random motion
            filter_pipeline = [
                self.avatar_filter,
                self.type_filter,
                self.chunk_end_memory_filter,
                self.random_memory_filter,
                self.cutoff_duration_filter,
            ]
            pipeline_input = dict(
                # All candidate motion records
                motion_records=self.cache.motion_records,
                # For avatar_filter
                avatar=request_dict['avatar'],
                # For type_filter
                motion_record_type=MotionRecordType.RANDOM,
                # For cutoff_duration_filter
                cutoff_duration_lowerbound=duration,
                cutoff_duration_upperbound=duration_extended,
                # For chunk_end_memory_filter and random_memory_filter
                user_id=request_dict['user_id'],
            )
            match_results = await filter_pipeline_retrieve(
                filter_pipeline=filter_pipeline,
                pipeline_input=pipeline_input,
                return_candidates=False
            )
            if len(match_results) > 0:
                motion_record_id = next(iter(match_results.keys()))
                motion_record = match_results[motion_record_id]
                earliest_cutoff_frame = \
                    await self.cutoff_duration_filter.get_earliest_cutoff_frame(
                        motion_record=motion_record,
                        cutoff_duration_lowerbound=duration,
                        cutoff_duration_upperbound=duration_extended
                    )
                cross_chunk_outpoint = earliest_cutoff_frame
                this_chunk_outpoint = final_interval[1] - final_interval[0]
                return_content = request_dict['return_content']
                preload_task = asyncio.create_task(
                    self.cache.get_motion_clip_by_id(motion_record_id)
                ) if return_content == 'motion_clip' else None
                insert_success, timeline = await self._try_to_insert(
                    timeline=timeline,
                    start_idx=final_interval[0],
                    end_idx=final_interval[1],
                    motion_record=motion_record,
                    in_point=0,
                    out_point=this_chunk_outpoint,
                    trigger='EndMatching',
                    preload_task=preload_task
                )
                if not insert_success:
                    msg = 'Current end matching motion insertion failed, ' +\
                        f'motion ID: {motion_record_id}, in-point: 0, ' +\
                        f'out-point: {this_chunk_outpoint}'
                    self.logger.error(msg)
                    msg = f'Request {request_id} BodyChunk {sequence_number} end ' + \
                        'could not be processed successfully ' +\
                        'due to insertion failure, ' +\
                        'will be handled at lower priority, '
                    await self.chunk_end_memory_filter.memory.remember(
                        user_id=request_dict['user_id'],
                        event_id=-1,
                        event_duration=(timeline.end - timeline.start) \
                            / self.__class__.FPS,
                    )
                else:
                    request_dict['current_chunk_end_motion_record_id'] = \
                        motion_record_id
                    request_dict['current_chunk_end_extendable'] = False \
                        if cross_chunk_outpoint == this_chunk_outpoint \
                        else True
                    request_dict['current_chunk_end_interval_start'] = \
                        final_interval[0]
                    request_dict['current_chunk_end_in_point'] = 0
                    request_dict['current_chunk_end_out_point'] = \
                        cross_chunk_outpoint
                    request_dict['current_chunk_end_payload'] = dict(
                        preload_task=preload_task,
                        in_point=0,
                        out_point=this_chunk_outpoint,
                    )
                    # Use -1 to placehold in memory before motion actually occurs
                    if final_interval[0] > last_memory_end_frame:
                        await self.chunk_end_memory_filter.memory.remember(
                            user_id=request_dict['user_id'],
                            event_id=-1,
                            event_duration=(final_interval[0] - last_memory_end_frame) \
                                / self.__class__.FPS,
                        )
                    await self.chunk_end_memory_filter.memory.remember(
                        user_id=request_dict['user_id'],
                        event_id=motion_record_id,
                        event_duration=this_chunk_outpoint / self.__class__.FPS,
                    )
                    last_memory_end_frame = final_interval[1]
                    msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                        'end processing completed, ' +\
                        'successfully matched a random motion, '
            else:
                for suffix in self.__class__.CHUNK_END_SUFFIXES:
                    request_dict[f'last{suffix}'] = None
                msg = f'Request {request_id} BodyChunk {sequence_number} end ' + \
                    'no cross-chunk random motion matched, ' +\
                    'will be handled at lower priority, '
        if last_memory_end_frame < timeline.end:
            await self.chunk_end_memory_filter.memory.remember(
                user_id=request_dict['user_id'],
                event_id=-1,
                event_duration=(timeline.end - last_memory_end_frame) \
                    / self.__class__.FPS,
            )
        end_time = time.time()
        msg += f'took {end_time - start_time:.2f} seconds.'
        self.logger.debug(msg)
        return timeline

    async def _fill_up_with_random_motion(
            self,
            request_id: str,
            timeline: Timeline) -> Timeline:
        """Fill up with random motion.

        Args:
            request_id (str):
                Request ID.
            timeline (Timeline):
                Motion timeline.

        Returns:
            Timeline:
                Filled motion timeline.
        """
        start_time = time.time()
        request_dict = self.request_space[request_id]
        sequence_number = request_dict['generated_sequence_number'] + 1
        loop = asyncio.get_running_loop()
        filter_pipeline = [
            self.avatar_filter,
            self.type_filter,
            self.chunk_end_memory_filter,
            self.random_memory_filter,
            self.cutoff_duration_filter,
            self.random_filter,
        ]
        pipeline_input_template = dict(
            # All candidate motion records
            motion_records=self.cache.motion_records,
            # For avatar_filter
            avatar=request_dict['avatar'],
            # For type_filter
            motion_record_type=MotionRecordType.RANDOM,
            # 用于cutoff_duration_filter
            cutoff_duration_lowerbound=0.0,
            cutoff_duration_upperbound=None,
            # 用于chunk_end_memory_filter和 random_memory_filter
            user_id=request_dict['user_id'],
        )
        last_memory_end_frame = timeline.start
        intervals_to_skip = list()
        while True:
            blank_interval = await loop.run_in_executor(
                self.thread_pool_executor,
                timeline.get_next_blank,
                intervals_to_skip
            )
            if blank_interval is None:
                break
            pipeline_input = pipeline_input_template.copy()
            blank_duration = (blank_interval[1] - blank_interval[0]) \
                / self.__class__.FPS
            pipeline_input['cutoff_duration_upperbound'] = blank_duration
            match_results = await filter_pipeline_retrieve(
                filter_pipeline=filter_pipeline,
                pipeline_input=pipeline_input,
                return_candidates=False
            )
            if len(match_results) > 0:
                motion_record_id = next(iter(match_results.keys()))
                motion_record = match_results[motion_record_id]
                cutoff_frame = \
                    await self.cutoff_duration_filter.get_earliest_cutoff_frame(
                        motion_record=motion_record,
                        cutoff_duration_lowerbound=0.0,
                        cutoff_duration_upperbound=blank_duration
                    )
                return_content = request_dict['return_content']
                preload_task = asyncio.create_task(
                    self.cache.get_motion_clip_by_id(motion_record_id)
                ) if return_content == 'motion_clip' else None
                out_point = cutoff_frame
                insert_success, timeline = await self._try_to_insert(
                    timeline=timeline,
                    start_idx=blank_interval[0],
                    end_idx=blank_interval[0] + out_point,
                    motion_record=motion_record,
                    in_point=0,
                    out_point=out_point,
                    trigger='Random',
                    preload_task=preload_task
                )
                if not insert_success:
                    msg = 'Random motion insertion failed ' +\
                        'during random motion filling, ' +\
                        f'motion ID: {motion_record_id}, in-point: 0, ' +\
                        f'out-point: {out_point}'
                    self.logger.error(msg)
                    intervals_to_skip.append(blank_interval)
                    continue
                # Placehold before motion actually occurs
                if blank_interval[0] > last_memory_end_frame:
                    await self.random_memory_filter.memory.remember(
                        user_id=request_dict['user_id'],
                        event_id=-1,
                        event_duration=(blank_interval[0] - last_memory_end_frame) \
                            / self.__class__.FPS,
                    )
                await self.random_memory_filter.memory.remember(
                    user_id=request_dict['user_id'],
                    event_id=motion_record_id,
                    event_duration=out_point / self.__class__.FPS,
                )
                last_memory_end_frame = blank_interval[0] + out_point
                if blank_interval[0] + out_point == timeline.end:
                    request_dict['current_chunk_end_motion_record_id'] = \
                        motion_record_id
                    request_dict['current_chunk_end_extendable'] = False
                    request_dict['current_chunk_end_interval_start'] = \
                        blank_interval[0]
                    request_dict['current_chunk_end_in_point'] = 0
                    request_dict['current_chunk_end_out_point'] = out_point
                    request_dict['current_chunk_end_payload'] = dict(
                        preload_task=preload_task,
                        in_point=0,
                        out_point=out_point,
                    )
            else:
                intervals_to_skip.append(blank_interval)
        if last_memory_end_frame < timeline.end:
            await self.random_memory_filter.memory.remember(
                user_id=request_dict['user_id'],
                event_id=-1,
                event_duration=(timeline.end - last_memory_end_frame) \
                    / self.__class__.FPS,
            )
        end_time = time.time()
        msg = f'Request {request_id} BodyChunk {sequence_number} ' + \
            f'random motion filling took {end_time - start_time:.2f} seconds.'
        self.logger.debug(msg)
        return timeline

    async def _fill_up_with_idle_long_motion(
            self,
            request_id: str,
            timeline: Timeline) -> Timeline:
        """Fill up with long idle motion.

        Args:
            request_id (str):
                Request ID.
            timeline (Timeline):
                Motion timeline.

        Returns:
            Timeline:
                Filled motion timeline.
        """
        start_time = time.time()
        request_dict = self.request_space[request_id]
        sequence_number = request_dict['generated_sequence_number'] + 1
        loop = asyncio.get_running_loop()
        filter_pipeline = [
            self.avatar_filter,
            self.type_filter,
            self.random_filter,
        ]
        pipeline_input = dict(
            # All candidate motion records
            motion_records=self.cache.motion_records,
            # For avatar_filter
            avatar=request_dict['avatar'],
            # For type_filter
            motion_record_type=MotionRecordType.IDLE_LONG,
        )
        intervals_to_skip = list()
        while True:
            blank_interval = await loop.run_in_executor(
                self.thread_pool_executor,
                timeline.get_next_blank,
                intervals_to_skip
            )
            if blank_interval is None:
                break
            match_results = await filter_pipeline_retrieve(
                filter_pipeline=filter_pipeline,
                pipeline_input=pipeline_input,
                return_candidates=False
            )
            if len(match_results) > 0:
                motion_record_id = next(iter(match_results.keys()))
                motion_record = match_results[motion_record_id]
                n_frames = motion_record.n_frames
                min_in_point = 0
                max_in_point = n_frames - self.__class__.FPS
                in_point = random.randint(min_in_point, max_in_point)
                out_point = n_frames \
                    if blank_interval[0] + n_frames - in_point <= blank_interval[1] \
                    else blank_interval[1] - blank_interval[0] + in_point
                return_content = request_dict['return_content']
                preload_task = asyncio.create_task(
                    self.cache.get_motion_clip_by_id(motion_record_id)
                ) if return_content == 'motion_clip' else None
                insert_success, timeline = await self._try_to_insert(
                    timeline=timeline,
                    start_idx=blank_interval[0],
                    end_idx=blank_interval[0] + out_point - in_point,
                    motion_record=motion_record,
                    in_point=in_point,
                    out_point=out_point,
                    trigger='IdleLong',
                    preload_task=preload_task
                )
                if not insert_success:
                    msg = 'Long idle motion insertion failed ' +\
                        'during long idle motion filling, ' +\
                        f'motion ID: {motion_record_id}, in-point: {in_point}, ' +\
                        f'out-point: {out_point}'
                    self.logger.error(msg)
                    raise RuntimeError(msg)
                if blank_interval[0] + out_point - in_point == timeline.end:
                    idle_long_extendable = request_dict['idle_long_extendable']
                    request_dict['current_chunk_end_motion_record_id'] = \
                        motion_record_id
                    request_dict['current_chunk_end_extendable'] = idle_long_extendable
                    request_dict['current_chunk_end_interval_start'] = \
                        blank_interval[0]
                    request_dict['current_chunk_end_in_point'] = in_point
                    request_dict['current_chunk_end_out_point'] = out_point
                    request_dict['current_chunk_end_payload'] = dict(
                        preload_task=preload_task,
                        in_point=in_point,
                        out_point=out_point,
                    )
            else:
                msg = 'No matching long idle motion found, ' +\
                    'please check logs and motion database.'
                self.logger.error(msg)
                raise RuntimeError(msg)
        end_time = time.time()
        msg = f'Request {request_id} BodyChunk {sequence_number} ' + \
            f'long idle motion filling took {end_time - start_time:.2f} seconds.'
        self.logger.debug(msg)
        return timeline

    async def _try_to_insert(
        self,
        timeline: Timeline,
        start_idx: int,
        end_idx: int,
        motion_record: MotionRecord,
        in_point: int,
        out_point: int,
        trigger: str | None = None,
        preload_task: asyncio.Task | None = None
    ) -> tuple[bool, Timeline]:
        """Try to insert a motion record at [start_idx, end_idx) position
        on timeline, may fail.

        Args:
            timeline (Timeline):
                Motion timeline.
            start_idx (int):
                Start frame of motion record on timeline.
            end_idx (int):
                End frame of motion record on timeline.
            motion_record (MotionRecord):
                Motion record to insert.
            in_point (int):
                Start frame of motion record.
            out_point (int):
                End frame of motion record.
            trigger (str | None, optional):
                Trigger word for motion record. If None, motion record
                has no trigger word. Defaults to None.
            preload_task (Task | None, optional):
                Preload task. If None, motion record has no preload task.
                Defaults to None.

        Returns:
            tuple[bool, Timeline]:
                Whether insertion was successful, and timeline after insertion.
        """
        timeline_backup = timeline.shallow_copy()
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                self.thread_pool_executor,
                timeline.insert,
                start_idx,
                end_idx,
                motion_record,
                in_point,
                out_point,
                trigger,
                preload_task
            )
        except Exception as e:
            msg = f'Motion insertion failed: {e}'
            self.logger.error(msg)
            timeline = timeline_backup
            return False, timeline
        return True, timeline

    async def _wait_for_motion_clips(self, timeline: Timeline) -> list[MotionClip]:
        """Wait for motion clips to load.

        Args:
            timeline (Timeline):
                Motion timeline.

        Returns:
            list[MotionClip]:
                Motion clips list.
        """
        timeline_list = timeline.to_list()
        while True:
            all_motion_clips_read = True
            for timeline_item in timeline_list:
                if 'motion_clip' in timeline_item['payload']:
                    continue
                preload_task: asyncio.Task | None = \
                    timeline_item['payload']['preload_task']
                if preload_task is None:
                    # No preload task, skip directly
                    continue
                elif preload_task.done():
                    motion_clip = preload_task.result()
                    timeline_item['payload']['motion_clip'] = motion_clip
                else:
                    # Preload task exists and not completed, continue waiting
                    all_motion_clips_read = False
            if all_motion_clips_read:
                break
            else:
                await asyncio.sleep(self.sleep_time)
        motion_clips = list()
        for timeline_item in timeline_list:
            src_motion_clip = timeline_item['payload']['motion_clip']
            in_point = timeline_item['payload']['in_point']
            out_point = timeline_item['payload']['out_point']
            dst_motion_clip = src_motion_clip.slice(
                in_point, out_point
            )
            motion_clips.append(dst_motion_clip)
        return motion_clips

    async def _smoothen_start(
            self,
            last_chunk_end_payload: dict[str, Any],
            motion_clips: list[MotionClip]) -> list[MotionClip]:
        """Smooth the first motion clip of timeline to match the motion clip
        at the end of previous BodyChunk.

        Args:
            last_chunk_end_payload (dict[str, Any]):
                Payload from previous BodyChunk end.
            motion_clips (list[MotionClip]):
                Motion clips list.

        Returns:
            list[MotionClip]:
                Smoothed motion clips list.
        """
        preload_task: asyncio.Task = last_chunk_end_payload['preload_task']
        while not preload_task.done():
            await asyncio.sleep(self.sleep_time)
        last_motion_clip: MotionClip = preload_task.result()
        out_point = last_chunk_end_payload['out_point']
        in_point = last_chunk_end_payload['in_point']
        last_chunk_end_motion_clip = last_motion_clip.slice(
            in_point, out_point
        )
        n_frames_last = last_chunk_end_motion_clip.n_frames
        timeline_start_motion_clip = motion_clips[0]
        n_frames_start = timeline_start_motion_clip.n_frames
        next_motion_clip_idx = 1
        # Parameter 15 is only good for currently implemented Interpolation class
        while n_frames_start < 15 and next_motion_clip_idx < len(motion_clips):
            next_motion_clip = motion_clips[next_motion_clip_idx]
            timeline_start_motion_clip = MotionClip.concat(
                [timeline_start_motion_clip, next_motion_clip]
            )
            n_frames_start = timeline_start_motion_clip.n_frames
            next_motion_clip_idx += 1
        if next_motion_clip_idx != 1:
            msg = 'The first motion clip is too short, ' +\
                f'concatenating first {next_motion_clip_idx} ' +\
                'motion clips for smoothing.'
            self.logger.warning(msg)
        # Increase priority of last_chunk_end_motion_clip
        # to ensure merge process only edits timeline_start_motion_clip
        last_chunk_end_motion_clip.set_cutoff_frames(
            [
                (0, 0, 0),
                (n_frames_last - 1, 9999, 0),
            ]
        )
        merge_clip = await self.motion_clip_merge.merge(
            [last_chunk_end_motion_clip, timeline_start_motion_clip]
        )
        smooth_motion_clip = merge_clip.slice(
            start_frame=n_frames_last,
            end_frame=n_frames_last + n_frames_start
        )
        ret_list = [smooth_motion_clip]
        if next_motion_clip_idx < len(motion_clips):
            ret_list.extend(motion_clips[next_motion_clip_idx:])
        return ret_list

    async def _update_last_chunk_end(self, request_id: str) -> None:
        """Update all last_chunk_end related values for a request.

        Copy current_chunk_end values to last_chunk_end and reset
        current_chunk_end values to None.

        Args:
            request_id (str):
                Request ID.
        """
        request_dict = self.request_space[request_id]
        for suffix in self.__class__.CHUNK_END_SUFFIXES:
            request_dict[f'last{suffix}'] = \
                request_dict[f'current{suffix}']
            request_dict[f'current{suffix}'] = None

    async def _read_log_file(self, start_pattern: str) -> str:
        """Read log entries related to request_id from log file.

        Args:
            start_pattern (str):
                Pattern to search for in log entries.

        Returns:
            str:
                Log entries containing the start pattern.

        Raises:
            FileNotFoundError:
                Raised when log file path is not found.
        """
        if self.log_path is None:
            msg = "Log file not found"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        log_lines = []
        line_count = 0
        with open(self.log_path, encoding='utf-8') as f:
            # Read from end of file
            f.seek(0, os.SEEK_END)  # Move to end of file
            position = f.tell()  # Record current position
            while position:
                f.seek(position - 1)  # Move to previous byte of current position
                char = f.read(1)  # Read one character
                # If newline encountered and log lines exist
                if char == '\n' and log_lines:
                    line = ''.join(reversed(log_lines))
                    if start_pattern in line or line_count >= 1000:
                        break
                    log_lines.clear()  # Clear current line
                    line_count += 1
                else:
                    log_lines.append(char)  # Continue building current line
                position = f.tell()  # Update current position
        return ''.join(reversed(log_lines))

    async def _maintain_check(self) -> None:
        """Check if cache needs update and if there are expired requests.

        Periodically checks cache version and request status, automatically
        updates cache and cleans up expired requests.
        """
        # Check Cache
        if self.cache_prepare_task is not None:
            if self.cache_prepare_task.done():
                msg = 'Cache update task completed, updating retrieval resources...'
                self.logger.info(msg)
                self.cache_prepare_task = None
                await self._build_filters()
                msg = 'Retrieval resources update completed'
                self.logger.info(msg)
                return
        cur_time = time.time()
        if cur_time - self.last_maintain_check_time > self.maintain_check_interval:
            self.last_maintain_check_time = cur_time
            # Check if there are expired requests in request_space
            expired_request_ids = []
            for request_id, request_dict in self.request_space.items():
                if cur_time - request_dict['last_request_time'] > \
                        self.request_expire_time:
                    expired_request_ids.append(request_id)
            if len(expired_request_ids) > 0:
                msg = f'Found {len(expired_request_ids)} expired requests, ' +\
                    f'deleted these requests: {expired_request_ids}'
                self.logger.warning(msg)
                for request_id in expired_request_ids:
                    self.request_space.pop(request_id)
            # Check if cache version matches meta_reader version
            cache_need_update = await self.cache.need_update()
            # Start cache update task in background
            if cache_need_update:
                msg = 'Cache layer needs update, ' +\
                    'starting cache update task...'
                self.logger.info(msg)
                self.cache_prepare_task = asyncio.create_task(
                    self.cache.prepare_next())
            cache_version = await self.cache.get_version()
            if cache_version != self.filters_version:
                msg = 'Retrieval resources version mismatch detected, ' +\
                    'updating retrieval resources...'
                self.logger.info(msg)
                await self._build_filters()
                msg = 'Retrieval resources update completed'
                self.logger.info(msg)

    def _build_readers(self) -> None:
        """Build meta_reader and motion_reader.

        Initialize metadata reader and motion data reader with
        provided configurations and logger settings.
        """
        meta_reader_cfg = self.meta_reader_cfg.copy()
        meta_reader_cfg['logger_cfg'] = self.logger_cfg
        self.meta_reader = build_meta_reader(meta_reader_cfg)
        motion_reader_cfg = self.motion_reader_cfg.copy()
        motion_reader_cfg['logger_cfg'] = self.logger_cfg
        motion_reader_cfg['thread_pool_executor'] = self.thread_pool_executor
        self.motion_reader = build_motion_reader(motion_reader_cfg)

    def _build_cache(self) -> LocalCache:
        """Build cache.

        Initialize local cache with provided configuration,
        metadata reader, and motion reader.

        Returns:
            LocalCache:
                Configured local cache instance.
        """
        cache_cfg = self.cache_cfg.copy()
        cache_cfg['logger_cfg'] = self.logger_cfg
        cache_cfg['thread_pool_executor'] = self.thread_pool_executor
        cache_cfg['meta_reader'] = self.meta_reader
        cache_cfg['motion_reader'] = self.motion_reader
        return build_cache(cache_cfg)

    async def _build_text_segmentation(self) -> None:
        """Build text segmentation.

        Initialize text segmentation component with speech keywords
        from cache and provided configuration.
        """
        speech_keyword_set = await self.cache.speech_keyword_mapping.keys()
        text_segmentation_cfg = self.text_segmentation_cfg.copy()
        text_segmentation_cfg['logger_cfg'] = self.logger_cfg
        init_white_list = text_segmentation_cfg.get('init_white_list', list())
        init_white_set = set(init_white_list)
        init_white_set.update(speech_keyword_set)
        text_segmentation_cfg['init_white_list'] = list(init_white_set)
        loop = asyncio.get_running_loop()
        self.text_segmentation = await loop.run_in_executor(
            self.thread_pool_executor,
            build_text_segmentation,
            text_segmentation_cfg
        )

    async def _build_motion_clip_merge(self) -> None:
        """Build motion clip merge.

        Initialize motion clip merge component with provided
        configuration and thread pool executor.
        """
        merge_cfg = self.merge_cfg.copy()
        merge_cfg['logger_cfg'] = self.logger_cfg
        merge_cfg['thread_pool_executor'] = self.thread_pool_executor
        self.motion_clip_merge = build_motion_clip_merge(merge_cfg)

    async def _build_filters(self) -> None:
        """Build filters.

        Initialize all filter components including avatar filter,
        type filter, keyword filters, memory filters, and other
        specialized filters for motion retrieval.
        """
        avatar_filter_cfg = dict(
            type='AvatarFilter',
            mapping=self.cache.avatar_mapping,
            logger_cfg=self.logger_cfg,
        )
        self.avatar_filter = build_filter(avatar_filter_cfg)
        type_filter_cfg = dict(
            type='TypeFilter',
            mapping=self.cache.type_mapping,
            logger_cfg=self.logger_cfg,
        )
        self.type_filter = build_filter(type_filter_cfg)
        motion_keyword_filter_cfg = dict(
            type='KeywordFilter',
            name='motion_keyword_filter',
            mapping=self.cache.motion_keyword_mapping,
            logger_cfg=self.logger_cfg,
        )
        self.motion_keyword_filter = build_filter(motion_keyword_filter_cfg)
        speech_keyword_filter_cfg = dict(
            type='KeywordFilter',
            name='speech_keyword_filter',
            mapping=self.cache.speech_keyword_mapping,
            logger_cfg=self.logger_cfg,
        )
        self.speech_keyword_filter = build_filter(speech_keyword_filter_cfg)
        keyword_align_filter_cfg = dict(
            type='KeywordAlignFilter',
            logger_cfg=self.logger_cfg,
        )
        self.keyword_align_filter = build_filter(keyword_align_filter_cfg)
        cutoff_duration_filter_cfg = dict(
            type='CutoffDurationFilter',
            logger_cfg=self.logger_cfg,
        )
        self.cutoff_duration_filter = build_filter(cutoff_duration_filter_cfg)
        random_filter_cfg = dict(
            type='RandomFilter',
            logger_cfg=self.logger_cfg,
        )
        self.random_filter = build_filter(random_filter_cfg)
        memory_cfg = self.memory_cfg.copy()
        memory_cfg['logger_cfg'] = self.logger_cfg
        motion_keyword_memory = build_memory(memory_cfg)
        motion_keyword_memory_filter_cfg = dict(
            type='MemoryFilter',
            name='motion_keyword_memory_filter',
            memory=motion_keyword_memory,
            logger_cfg=self.logger_cfg,
        )
        self.motion_keyword_memory_filter = build_filter(
            motion_keyword_memory_filter_cfg)
        speech_keyword_memory = build_memory(memory_cfg)
        speech_keyword_memory_filter_cfg = dict(
            type='MemoryFilter',
            name='speech_keyword_memory_filter',
            memory=speech_keyword_memory,
            logger_cfg=self.logger_cfg,
        )
        self.speech_keyword_memory_filter = build_filter(
            speech_keyword_memory_filter_cfg)
        chunk_end_memory = build_memory(memory_cfg)
        chunk_end_memory_filter_cfg = dict(
            type='MemoryFilter',
            name='chunk_end_memory_filter',
            memory=chunk_end_memory,
            logger_cfg=self.logger_cfg,
        )
        self.chunk_end_memory_filter = build_filter(
            chunk_end_memory_filter_cfg)
        random_memory = build_memory(memory_cfg)
        random_memory_filter_cfg = dict(
            type='MemoryFilter',
            name='random_memory_filter',
            memory=random_memory,
            logger_cfg=self.logger_cfg,
        )
        self.random_memory_filter = build_filter(random_memory_filter_cfg)
        self.filters_version = await self.cache.get_version()

