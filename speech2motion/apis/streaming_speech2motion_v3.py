import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel

from ..cache.builder import build_cache
from ..cache.local_cache import LocalCache
from ..data_structures.motion_clip import MotionClip
from ..data_structures.motion_record import MotionRecord, MotionRecordType
from ..data_structures.restpose import Restpose
from ..data_structures.timeline import Timeline
from ..filters.builder import (
    AvatarFilter,
    CutoffDurationFilter,
    KeywordAlignFilter,
    KeywordFilter,
    LabelFilter,
    MemoryFilter,
    RandomFilter,
    TypeFilter,
    build_filter,
)
from ..io.meta.builder import build_meta_reader
from ..io.motion.builder import build_motion_reader
from ..io.restpose.builder import build_restpose_reader
from ..merge.builder import (
    Blending,
    Concatenation,
    Interpolation,
    build_motion_clip_merge,
)
from ..retrieve.filter_pipeline import filter_pipeline_retrieve
from ..text_segmentation.builder import build_text_segmentation
from ..utils.super import Super
from ..variety.builder import build_memory

if TYPE_CHECKING:
    from ..io.meta.base_meta_reader import BaseMetaReader
    from ..io.motion.base_motion_reader import BaseMotionReader
    from ..io.restpose.base_restpose_reader import BaseRestposeReader
    from ..text_segmentation.jieba_text_segmentation import JiebaTextSegmentation


class StreamingSpeech2MotionV3ChunkStart(BaseModel):
    """V3 streaming motion generation start request.

    Used to initialize a streaming motion generation session, including user
    information, avatar selection, and application configuration.

    Args:
        request_id (str):
            Request ID for identifying an independent streaming motion
            generation request.
        user_id (str):
            User ID for distinguishing the requesting user and calling
            related memory.
        avatar (str):
            Avatar selected by user.
        app_name (Literal['babylon', 'python_backend'], optional):
            Target application name. Defaults to 'python_backend'.
        max_front_extension_duration (float, optional):
            Maximum additional duration that generated motion can extend
            before frame 0. Defaults to 0.0.
        max_rear_extension_duration (float, optional):
            Maximum additional duration that generated motion can extend
            after the last frame. Defaults to 0.0.
        memory_duration_override (float | None, optional):
            Memory duration in seconds. If None, uses the default
            `memory_duration` parameter. Defaults to None.
    """
    request_id: str
    user_id: str
    avatar: str
    app_name: Literal['babylon', 'python_backend'] = 'python_backend'
    max_front_extension_duration: float = 0.0
    max_rear_extension_duration: float = 0.0
    memory_duration_override: float | None = None

class StreamingSpeech2MotionV3ChunkBody(BaseModel):
    """V3 streaming motion generation body request.

    Used to process speech text and motion keywords, generating corresponding
    motion clips.

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
        label_expression (str | None, optional):
            Label expression for filtering motion records. If None, no
            filtering is applied. Defaults to None.
    """
    request_id: str
    duration: float
    speech_text: str
    sequence_number: int
    speech_time: list[tuple[int, float]] | None = None
    motion_keywords: list[tuple[int, str]] | None = None
    label_expression: str | None = None

class StreamingSpeech2MotionV3ChunkEnd(BaseModel):
    """V3 streaming motion generation end request.

    Used to end a streaming motion generation session, processing remaining
    extension motions.

    Args:
        request_id (str):
            Request ID for identifying an independent streaming motion
            generation request.
    """
    request_id: str

class StreamingSpeech2MotionV3(Super):
    """V3 streaming motion generator.

    This class provides streaming motion generation functionality for V3 API,
    supporting real-time motion generation based on speech input with
    keyword matching, memory management, and smooth transitions.
    """
    FPS: int = 30

    def __init__(
            self,
            meta_reader_cfg: dict[str, Any],
            motion_reader_cfg: dict[str, Any],
            restpose_reader_cfg: dict[str, Any],
            memory_cfg: dict,
            text_segmentation_cfg: dict[str, Any],
            cache_cfg: dict[str, Any],
            interpolation_cfg: dict[str, Any],
            blending_cfg: dict[str, Any],
            request_expire_time: int = 120,
            maintain_check_interval: int = 60,
            max_workers: int = 4,
            thread_pool_executor: ThreadPoolExecutor | None = None,
            sleep_time: float = 0.01,
            logger_cfg: None | dict[str, Any] = None):
        """Initialize the V3 streaming motion generator.

        Args:
            meta_reader_cfg (dict[str, Any]):
                Metadata reader configuration.
            motion_reader_cfg (dict[str, Any]):
                Motion data reader configuration.
            restpose_reader_cfg (dict[str, Any]):
                Restpose data reader configuration.
            memory_cfg (dict):
                Memory configuration.
            text_segmentation_cfg (dict[str, Any]):
                Text segmentation configuration.
            cache_cfg (dict[str, Any]):
                Cache configuration.
            interpolation_cfg (dict[str, Any]):
                Interpolation merge configuration.
            blending_cfg (dict[str, Any]):
                Blending merge configuration.
            request_expire_time (int, optional):
                Request expiration time in seconds. Defaults to 120.
                When a request has no new ChunkBody or ChunkEnd requests
                within request_expire_time, it will automatically expire.
            maintain_check_interval (int, optional):
                Maintenance task check interval, including motion library
                cache version check and expired request check.
                Defaults to 60.
            max_workers (int, optional):
                Maximum number of worker threads.
                Defaults to 4.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor for finding blank intervals and
                data reading. If None, creates new thread pool executor
                based on max_workers. Defaults to None.
            sleep_time (float, optional):
                Sleep time when waiting for data reading operations
                that are not completed. Defaults to 0.01 seconds.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration. Defaults to None.
        """
        Super.__init__(self, logger_cfg=logger_cfg)
        self.meta_reader_cfg = meta_reader_cfg
        self.motion_reader_cfg = motion_reader_cfg
        self.restpose_reader_cfg = restpose_reader_cfg
        self.cache_cfg = cache_cfg
        self.memory_cfg = memory_cfg
        self.text_segmentation_cfg = text_segmentation_cfg
        self.interpolation_cfg = interpolation_cfg
        self.blending_cfg = blending_cfg
        self.request_expire_time = request_expire_time
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
        self.restpose_reader: BaseRestposeReader | None = None
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
        self.label_filter: LabelFilter | None = None

        # Text segmentation
        self.text_segmentation: JiebaTextSegmentation | None = None

        # Motion merge
        self.interpolation: Interpolation | Concatenation | Blending | None = None
        self.blending: Blending | None = None

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

    async def handle_chunk_start(
            self,
            chunk_start: StreamingSpeech2MotionV3ChunkStart
            ) -> None:
        """Handle streaming motion generation start request.

        Initialize a new streaming motion generation session, setting up
        request state and configuration parameters.

        Args:
            chunk_start (StreamingSpeech2MotionV3ChunkStart):
                Streaming motion generation start request.

        Raises:
            RuntimeError:
                Raised when initialization fails.
        """
        self.logger.info(f'Received start signal for request {chunk_start.request_id}.')
        await self._maintain_check()
        request_dict = chunk_start.model_dump()
        request_dict['last_chunk_end_frame_idx'] = 0
        request_dict['received_sequence_number'] = -1
        request_dict['generated_sequence_number'] = -1
        request_dict['app_name'] = chunk_start.app_name
        request_dict['last_request_time'] = time.time()
        for timeline_name in ('keyword', 'base'):
            request_dict[f'last_{timeline_name}_timeline'] = None
            request_dict[f'current_{timeline_name}_timeline'] = None
        self.request_space[chunk_start.request_id] = request_dict

    async def handle_chunk_body(
            self,
            chunk_body: StreamingSpeech2MotionV3ChunkBody
            ) -> MotionClip | None:
        """Handle streaming motion generation body request.

        Generate corresponding motion clips based on speech text and motion
        keywords, supporting motion matching, merging, and conversion.

        Args:
            chunk_body (StreamingSpeech2MotionV3ChunkBody):
                Streaming motion generation body request.

        Returns:
            MotionClip | None:
                Returns MotionClip instance, or None if request frame count is 0.

        Raises:
            TimeoutError:
                Raised when waiting for previous matching results times out.
            RuntimeError:
                Raised when processing fails.
        """
        chunk_body_msg = f'Received request {chunk_body.request_id} for ' +\
            f'BodyChunk {chunk_body.sequence_number}.'
        self.logger.info(chunk_body_msg)
        msg = f'Speech duration is {chunk_body.duration} seconds, ' +\
            f'speech text: {chunk_body.speech_text}\n' +\
            f'speech time list: {chunk_body.speech_time}\n' +\
            f'motion keywords list: {chunk_body.motion_keywords}'
        self.logger.debug(msg)
        if hasattr(chunk_body, 'label_expression'):
            label_expression = chunk_body.label_expression
        else:
            label_expression = None
        body_duration = chunk_body.duration
        body_n_frames = int(body_duration * self.__class__.FPS)
        if body_n_frames <= 0:
            msg = f'Request {chunk_body.request_id} BodyChunk ' + \
                f'{chunk_body.sequence_number} length is 0, ' + \
                'skipping matching.'
            self.logger.warning(msg)
            return None
        request_dict = self.request_space[chunk_body.request_id]
        request_dict['received_sequence_number'] = max(
            request_dict['received_sequence_number'],
            chunk_body.sequence_number
        )
        request_time = time.time()
        request_dict['last_request_time'] = time.time()
        await self._maintain_check()
        while request_dict['generated_sequence_number'] < \
                chunk_body.sequence_number - 1:
            time_diff = time.time() - request_time
            if time_diff > self.request_expire_time:
                msg = f'Request {chunk_body.request_id} BodyChunk ' + \
                    f'{chunk_body.sequence_number} timed out waiting for ' + \
                    'previous matching results, terminating matching.'
                self.logger.error(msg)
                raise TimeoutError(msg)
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
            msg = f'Request {chunk_body.request_id} BodyChunk ' +\
                f'{chunk_body.sequence_number} has no speech time information, ' +\
                'predicting speech time based on uniform time distribution.'
            self.logger.warning(msg)
        else:
            speech_time = chunk_body.speech_time
        loop = asyncio.get_running_loop()
        # Prepare keyword motion timeline
        full_time_list = await loop.run_in_executor(
            self.thread_pool_executor,
            self._convert_speech_time,
            speech_time,
            chunk_body.speech_text,
            body_duration,
            body_start_time)
        original_end_idx = body_start_frame + body_n_frames
        keyword_timeline = Timeline(
            start_frame=body_start_frame,
            end_frame=original_end_idx,
            enable_extension=True,
            logger_cfg=self.logger_cfg,
        )
        last_keyword_timeline: Timeline | None = request_dict['last_keyword_timeline']
        last_chunk_end_frame_idx = request_dict['last_chunk_end_frame_idx']
        # Insert the confirmed keyword_timeline content from
        # the previous Chunk into the current timeline
        if last_keyword_timeline is not None:
            await self._copy_from_last_timeline(
                this_timeline=keyword_timeline,
                last_timeline=last_keyword_timeline,
                copy_start_frame=last_chunk_end_frame_idx
            )
        # Decide whether to allow extension after
        # the confirmed keyword_timeline from
        # the previous Chunk based on parameters
        keyword_timeline.enable_extension = extension_enabled
        keyword_first_item = keyword_timeline.get_first_item()
        if keyword_first_item is not None:
            base_timeline_start_frame = keyword_first_item['start']
        else:
            base_timeline_start_frame = body_start_frame
        # Prepare base motion timeline
        last_base_timeline: Timeline | None = request_dict['last_base_timeline']
        base_timeline = Timeline(
            start_frame=body_start_frame,
            end_frame=body_start_frame + body_n_frames,
            enable_extension=True,
            logger_cfg=self.logger_cfg,
        )
        # Insert the confirmed base_timeline content from
        # the previous Chunk into the current timeline
        if last_base_timeline is not None:
            await self._copy_from_last_timeline(
                this_timeline=base_timeline,
                last_timeline=last_base_timeline,
                copy_start_frame=base_timeline_start_frame
            )
        base_timeline.enable_extension = extension_enabled
        # Motion keyword matching
        if chunk_body.motion_keywords is not None:
            keyword_timeline = await self._match_motion_keywords(
                request_id=chunk_body.request_id,
                base_timeline=base_timeline,
                keyword_timeline=keyword_timeline,
                original_end_idx=original_end_idx,
                full_time_list=full_time_list,
                motion_keywords=chunk_body.motion_keywords,
                label_expression=label_expression
            )
        # Speech keyword matching
        keyword_timeline = await self._match_speech_keywords(
            request_id=chunk_body.request_id,
            base_timeline=base_timeline,
            keyword_timeline=keyword_timeline,
            original_end_idx=original_end_idx,
            full_time_list=full_time_list,
            speech_text=chunk_body.speech_text,
            label_expression=label_expression
        )
        # Select speech random motions that don't conflict with
        # keyword timeline and insert into timeline
        base_timeline = await self._extend_base_timeline(
            request_id=chunk_body.request_id,
            base_timeline=base_timeline,
            keyword_timeline=keyword_timeline,
            label_expression=label_expression
        )
        base_table_task = asyncio.wrap_future(
            loop.run_in_executor(
                self.thread_pool_executor,
                base_timeline.to_table,
                self.__class__.FPS,
                None,
                last_chunk_end_frame_idx if last_chunk_end_frame_idx > 0 else None,
                original_end_idx
            )
        )
        self.logger.info(
            f'Request {chunk_body.request_id} BodyChunk {chunk_body.sequence_number} ' +
            f'matching completed in {time.time() - start_time:.2f} seconds.'
        )
        waiting_for_read_start = time.time()
        base_mc_coroutine = self._wait_for_motion_clips(
            base_timeline,
        )
        keyword_mc_coroutine = self._wait_for_motion_clips(
            keyword_timeline,
        )
        base_mc_list, keyword_mc_list = await asyncio.gather(
            base_mc_coroutine,
            keyword_mc_coroutine
        )
        if len(keyword_mc_list) > 0:
            keyword_table_task = asyncio.wrap_future(
                loop.run_in_executor(
                    self.thread_pool_executor,
                    keyword_timeline.to_table,
                    self.__class__.FPS,
                    None,
                    last_chunk_end_frame_idx if last_chunk_end_frame_idx > 0 else None,
                    original_end_idx
                )
            )
        end_time = time.time()
        self.logger.debug(
            f'Request {chunk_body.request_id} ' +\
            f'BodyChunk {chunk_body.sequence_number} ' + \
            'finished loading ' +\
            f'{len(base_mc_list) + len(keyword_mc_list)} motion clips ' +\
            f'in {end_time - waiting_for_read_start:.2f} seconds.'
        )
        merge_start_time = time.time()
        # check if the first motion clip requires smoothing
        start_requires_smooth = False
        # If the previous Chunk's timeline exists and
        # ends exactly before the current Chunk starts,
        # check whether the motion is continuous
        if last_keyword_timeline is not None and \
                last_base_timeline is not None and \
                last_base_timeline.end == last_chunk_end_frame_idx:
            # Check from which timeline the previous Chunk ends
            last_keyword_item = last_keyword_timeline.get_last_item()
            if last_keyword_item is not None and \
                    last_keyword_item['end'] == body_start_frame:
                last_chunk_end_item = last_keyword_item
            else:
                last_chunk_end_item = last_base_timeline.get_last_item()
            last_id = last_chunk_end_item['payload'][
                'motion_record'].motion_record_id
            last_outpoint = last_chunk_end_item['payload']['out_point']
            # Determine which axis the motion at the start of this Chunk comes from
            keyword_first_item = keyword_timeline.get_first_item()
            if keyword_first_item is not None and \
                    keyword_first_item['start'] == body_start_frame:
                this_chunk_start_item = keyword_first_item
            else:
                this_chunk_start_item = base_timeline.get_first_item()
            this_id = this_chunk_start_item[
                'payload']['motion_record'].motion_record_id
            this_inpoint = this_chunk_start_item['payload']['in_point']
            if not (this_id == last_id and this_inpoint == last_outpoint):
                start_requires_smooth = True
        if start_requires_smooth:
            msg = f'Request {chunk_body.request_id} ' +\
                f'BodyChunk {chunk_body.sequence_number} ' + \
                'chunk start cannot smoothly connect with previous chunk end, ' + \
                'transition animation required.'
            self.logger.debug(msg)
            last_chunk_end_item = last_base_timeline.get_last_item()
            base_mc_list = await self._smoothen_start(
                last_chunk_end_payload=last_chunk_end_item['payload'],
                motion_clips=base_mc_list)
        # After updating smoothen_start with last_chunk_end_payload,
        # set generated_sequence_number
        request_dict['last_chunk_end_frame_idx'] = original_end_idx
        request_dict['last_keyword_timeline'] = keyword_timeline
        request_dict['last_base_timeline'] = base_timeline
        request_dict['generated_sequence_number'] = chunk_body.sequence_number
        base_table_str = await base_table_task
        msg = (
            f'Request {chunk_body.request_id} BodyChunk {chunk_body.sequence_number} ' +
            'matching result table conversion completed, ' +
            f'base timeline:\n{base_table_str}'
        )
        if len(keyword_mc_list) > 0:
            keyword_table_str = await keyword_table_task
            msg += f'\nkeyword timeline:\n{keyword_table_str}'
        self.logger.info(msg)
        # concat motions with the same motion_record_id before interpolation
        base_timeline_list = await loop.run_in_executor(
            self.thread_pool_executor,
            base_timeline.to_list,
        )
        base_motion_idx = 0
        while base_motion_idx < len(base_mc_list) - 1:
            this_motion_record_id = base_timeline_list[
                base_motion_idx]['payload']['motion_record'].motion_record_id
            this_out_point = base_timeline_list[
                base_motion_idx]['payload']['out_point']
            next_motion_record_id = base_timeline_list[
                base_motion_idx + 1]['payload']['motion_record'].motion_record_id
            next_in_point = base_timeline_list[
                base_motion_idx + 1]['payload']['in_point']
            if this_motion_record_id == next_motion_record_id and \
                    this_out_point == next_in_point:
                this_mc = base_mc_list.pop(base_motion_idx)
                next_mc = base_mc_list.pop(base_motion_idx)
                concat_mc = await loop.run_in_executor(
                    self.thread_pool_executor,
                    MotionClip.concat,
                    [this_mc, next_mc]
                )
                base_mc_list.insert(base_motion_idx, concat_mc)
            base_motion_idx += 1
        base_motion_clip = await self.interpolation.merge(base_mc_list)
        blend_motion_clip = base_motion_clip
        keyword_timeline_list = await loop.run_in_executor(
            self.thread_pool_executor,
            keyword_timeline.to_list,
        )
        for keyword_motion_idx, keyword_mc in enumerate(keyword_mc_list):
            motion_record = keyword_timeline_list[
                keyword_motion_idx]['payload']['motion_record']
            align_frame = keyword_timeline_list[
                keyword_motion_idx]['start'] - base_timeline.start
            startup_frame = motion_record.startup_frame
            recovery_frame = motion_record.recovery_frame
            blend_motion_clip = await self.blending.merge(
                [keyword_mc, blend_motion_clip],
                align_frame=align_frame,
                startup_frame=startup_frame,
                recovery_frame=recovery_frame
            )
        if last_chunk_end_frame_idx == 0:
            slice_start_idx = 0
            slice_end_idx = original_end_idx - base_timeline.start
        else:
            slice_start_idx = last_chunk_end_frame_idx - base_timeline.start
            slice_end_idx = original_end_idx - base_timeline.start
        blend_motion_clip = blend_motion_clip.slice(
            slice_start_idx,
            slice_end_idx
        )
        merge_end_time = time.time()
        self.logger.debug(
            f'Request {chunk_body.request_id} ' +\
            f'BodyChunk {chunk_body.sequence_number} ' + \
            'merge completed, ' + \
            f'took {merge_end_time - merge_start_time:.2f} seconds.'
        )
        app_name = request_dict['app_name']
        if app_name == 'python_backend':
            blend_motion_clip.app_name = app_name
            app_motion_clip = blend_motion_clip
        else:
            convert_start_time = time.time()
            app_motion_clip = await self._convert_motion_clip_app(
                blend_motion_clip,
                app_name
            )
            convert_end_time = time.time()
            self.logger.debug(
                f'Request {chunk_body.request_id} ' +\
                f'BodyChunk {chunk_body.sequence_number} ' + \
                f'conversion to {app_name} completed in ' + \
                f'{convert_end_time - convert_start_time:.2f} seconds.'
            )
        app_motion_clip.set_timeline_start_idx(base_timeline.start)
        return app_motion_clip

    async def handle_chunk_end(
            self,
            chunk_end: StreamingSpeech2MotionV3ChunkEnd
            ) -> MotionClip | None:
        """Handle streaming motion generation end request.

        Process remaining extension motions at session end and clean up
        request state.

        Args:
            chunk_end (StreamingSpeech2MotionV3ChunkEnd):
                Streaming motion generation end request.

        Returns:
            MotionClip | None:
                Returns None if there is no extension data needed based on
                previous requests. If extension is needed, returns MotionClip
                instance.

        Raises:
            RuntimeError:
                Raised when timeline state is inconsistent.
        """
        loop = asyncio.get_running_loop()
        request_dict = self.request_space[chunk_end.request_id]
        request_time = time.time()
        while request_dict['generated_sequence_number'] < \
                request_dict['received_sequence_number']:
            time_diff = time.time() - request_time
            if time_diff > self.request_expire_time:
                msg = f'Request {chunk_end.request_id} ' + \
                    'EndChunk ' + \
                    'timed out waiting for previous matching results, ' + \
                    'terminating matching.'
                self.logger.error(msg)
                raise TimeoutError(msg)
            await asyncio.sleep(self.sleep_time)
        last_keyword_timeline: Timeline | None = request_dict['last_keyword_timeline']
        last_base_timeline: Timeline | None = request_dict['last_base_timeline']
        last_chunk_end_frame_idx = request_dict['last_chunk_end_frame_idx']
        # Insert the confirmed keyword_timeline content from
        # the previous Chunk into the current timeline
        if last_keyword_timeline is None:
            self.request_space.pop(chunk_end.request_id)
            return None
        if last_base_timeline is None:
            msg = 'last_keyword_timeline is not None but ' +\
                'last_base_timeline is None when handling chunk end, ' +\
                f'request_id: {chunk_end.request_id}'
            self.logger.error(msg)
            raise RuntimeError(msg)
        # Last matching ends exactly before the current Chunk starts,
        # no data needed for ChunkEnd
        if last_base_timeline.end == last_chunk_end_frame_idx:
            self.request_space.pop(chunk_end.request_id)
            return None
        keyword_timeline = Timeline(
            start_frame=last_chunk_end_frame_idx,
            end_frame=last_keyword_timeline.end,
            enable_extension=True,
            logger_cfg=self.logger_cfg,
        )
        keyword_timeline = await self._copy_from_last_timeline(
            this_timeline=keyword_timeline,
            last_timeline=last_keyword_timeline,
            copy_start_frame=last_chunk_end_frame_idx
        )
        base_timeline = Timeline(
            start_frame=last_chunk_end_frame_idx,
            end_frame=last_base_timeline.end,
            enable_extension=True,
            logger_cfg=self.logger_cfg,
        )
        copy_start_frame = min(last_chunk_end_frame_idx, keyword_timeline.start)
        base_timeline = await self._copy_from_last_timeline(
            this_timeline=base_timeline,
            last_timeline=last_base_timeline,
            copy_start_frame=copy_start_frame
        )
        base_table_task = asyncio.wrap_future(
            loop.run_in_executor(
                self.thread_pool_executor,
                base_timeline.to_table,
                self.__class__.FPS,
                None,
                last_chunk_end_frame_idx,
                base_timeline.end
            )
        )
        merge_start_time = time.time()
        # Last handle_chunk_body has already completed motion_clip reading,
        # no waiting time spent here
        base_mc_list = await self._wait_for_motion_clips(
            timeline=base_timeline,
        )
        base_timeline_list = await loop.run_in_executor(
            self.thread_pool_executor,
            base_timeline.to_list,
        )
        keyword_mc_list = await self._wait_for_motion_clips(
            timeline=keyword_timeline,
        )
        keyword_timeline_list = await loop.run_in_executor(
            self.thread_pool_executor,
            keyword_timeline.to_list,
        )
        if len(keyword_mc_list) > 0:
            keyword_table_task = asyncio.wrap_future(
                loop.run_in_executor(
                    self.thread_pool_executor,
                    keyword_timeline.to_table,
                    self.__class__.FPS,
                    None,
                    last_chunk_end_frame_idx,
                    keyword_timeline.end
                )
            )
        # concat motions with the same motion_record_id before interpolation
        base_motion_idx = 0
        while base_motion_idx < len(base_mc_list) - 1:
            this_motion_record_id = base_timeline_list[
                base_motion_idx]['payload']['motion_record'].motion_record_id
            this_out_point = base_timeline_list[
                base_motion_idx]['payload']['out_point']
            next_motion_record_id = base_timeline_list[
                base_motion_idx + 1]['payload']['motion_record'].motion_record_id
            next_in_point = base_timeline_list[
                base_motion_idx + 1]['payload']['in_point']
            if this_motion_record_id == next_motion_record_id and \
                    this_out_point == next_in_point:
                this_mc = base_mc_list.pop(base_motion_idx)
                next_mc = base_mc_list.pop(base_motion_idx)
                concat_mc = await loop.run_in_executor(
                    self.thread_pool_executor,
                    MotionClip.concat,
                    [this_mc, next_mc]
                )
                base_mc_list.insert(base_motion_idx, concat_mc)
            base_motion_idx += 1
        base_motion_clip = await self.interpolation.merge(base_mc_list)
        blend_motion_clip = base_motion_clip
        for keyword_motion_idx, keyword_mc in enumerate(keyword_mc_list):
            motion_record = keyword_timeline_list[
                keyword_motion_idx]['payload']['motion_record']
            align_frame = keyword_timeline_list[
                keyword_motion_idx]['start'] - base_timeline.start
            startup_frame = motion_record.startup_frame
            recovery_frame = motion_record.recovery_frame
            blend_motion_clip = await self.blending.merge(
                [keyword_mc, blend_motion_clip],
                align_frame=align_frame,
                startup_frame=startup_frame,
                recovery_frame=recovery_frame
            )
        blend_motion_clip = await loop.run_in_executor(
            self.thread_pool_executor,
            blend_motion_clip.slice,
            last_chunk_end_frame_idx - base_timeline.start,
            blend_motion_clip.n_frames
        )
        merge_end_time = time.time()
        base_table_str = await base_table_task
        msg = (
            f'Request {chunk_end.request_id} ' +
            'EndChunk matching result table conversion completed, ' +
            f'base timeline:\n{base_table_str}'
        )
        if len(keyword_timeline_list) > 0:
            keyword_table_str = await keyword_table_task
            msg += f'\nkeyword timeline:\n{keyword_table_str}'
        self.logger.info(msg)
        self.logger.debug(
            f'Request {chunk_end.request_id} ' +\
            'EndChunk ' + \
            'merge completed, ' + \
            f'took {merge_end_time - merge_start_time:.2f} seconds.'
        )
        app_name = request_dict['app_name']
        if app_name == 'python_backend':
            blend_motion_clip.app_name = app_name
            app_motion_clip = blend_motion_clip
        else:
            convert_start_time = time.time()
            app_motion_clip = await self._convert_motion_clip_app(
                blend_motion_clip,
                app_name
            )
            convert_end_time = time.time()
            self.logger.debug(
                f'Request {chunk_end.request_id} ' +\
                'EndChunk ' + \
                f'conversion to {app_name} completed in ' + \
                f'{convert_end_time - convert_start_time:.2f} seconds.'
            )
        app_motion_clip.set_timeline_start_idx(last_chunk_end_frame_idx)
        self.request_space.pop(chunk_end.request_id)
        return app_motion_clip

    async def _copy_from_last_timeline(
            self,
            this_timeline: Timeline,
            last_timeline: Timeline,
            copy_start_frame: int,) -> Timeline:
        """Copy timeline items from last timeline to current timeline.

        Copies timeline items from the last timeline that extend beyond
        the copy_start_frame to the current timeline, updating trigger
        names to indicate extension.

        Args:
            this_timeline (Timeline):
                Current timeline to copy items to.
            last_timeline (Timeline):
                Last timeline to copy items from.
            copy_start_frame (int):
                Start frame from which to copy items.

        Returns:
            Timeline:
                Updated current timeline with copied items.
        """
        loop = asyncio.get_running_loop()
        items = await loop.run_in_executor(
            self.thread_pool_executor,
            last_timeline.to_list,
        )
        for item in items:
            end_idx = item['end']
            if end_idx <= copy_start_frame:
                continue
            payload = item['payload']
            # Add item to keyword_timeline for blending
            start_idx = item['start']
            in_point = payload['in_point']
            motion_record = payload['motion_record']
            out_point = payload['out_point']
            trigger = payload['trigger'].replace('Match', 'Extend')
            preload_task = payload['preload_task']
            await loop.run_in_executor(
                self.thread_pool_executor,
                this_timeline.insert,
                start_idx,
                end_idx,
                motion_record,
                in_point,
                out_point,
                trigger,
                preload_task
            )
        return this_timeline

    async def _extend_base_timeline(
            self,
            request_id: str,
            base_timeline: Timeline,
            keyword_timeline: Timeline,
            label_expression: str | None = None,) -> Timeline:
        """Prepare base motion timeline using random motions.

        Fills the base timeline with random motions to ensure complete
        timeline coverage.

        Args:
            request_id (str):
                Request ID.
            base_timeline (Timeline):
                Base motion timeline.
            keyword_timeline (Timeline):
                Keyword motion timeline.
            label_expression (str | None, optional):
                Label expression for filtering. If None, no filtering
                is applied. Defaults to None.

        Returns:
            Timeline:
                Filled motion timeline.

        Raises:
            RuntimeError:
                Raised when unable to extend base timeline or motion
                insertion fails.
        """
        start_time = time.time()
        request_dict = self.request_space[request_id]
        loop = asyncio.get_running_loop()
        sequence_number = request_dict['generated_sequence_number'] + 1
        # Check if base_timeline needs to be left-side extended
        # based on keyword_timeline first item
        keyword_first_item = await loop.run_in_executor(
            self.thread_pool_executor,
            keyword_timeline.get_first_item
        )
        if keyword_first_item is not None:
            keyword_item_start = keyword_first_item['start']
            base_first_item = await loop.run_in_executor(
                self.thread_pool_executor,
                base_timeline.get_first_item
            )
            if base_first_item is None:
                if keyword_item_start < base_timeline.start:
                    base_timeline = Timeline(
                        start_frame=keyword_item_start,
                        end_frame=base_timeline.end,
                        enable_extension=False,
                        logger_cfg=base_timeline.logger_cfg
                    )
            else:
                if keyword_item_start < base_timeline.start:
                    msg = 'Keyword motion starts earlier than base timeline, ' +\
                        'but there are already motions in base timeline, ' +\
                        'cannot extend base timeline.' +\
                        f'base timeline start: {base_timeline.start}, ' +\
                        f'keyword motion start: {keyword_item_start}'
                    self.logger.error(msg)
                    raise RuntimeError(msg)
        # Check if base_timeline needs to be right-side extended
        # based on keyword_timeline last item
        keyword_last_item = await loop.run_in_executor(
            self.thread_pool_executor,
            keyword_timeline.get_last_item
        )
        if keyword_last_item is not None:
            keyword_item_end = keyword_last_item['end']
            if keyword_item_end > base_timeline.end:
                base_timeline = Timeline(
                    start_frame=base_timeline.start,
                    end_frame=keyword_item_end,
                    enable_extension=False,
                    logger_cfg=base_timeline.logger_cfg
                )
        # Try to extend motions in base_timeline to ensure motion continuity
        last_base_timeline: Timeline | None = request_dict['last_base_timeline']
        if last_base_timeline is not None:
            base_last_item = await loop.run_in_executor(
                self.thread_pool_executor,
                last_base_timeline.get_last_item
            )
            if base_last_item is not None:
                last_motion_record = base_last_item['payload']['motion_record']
                last_outpoint = base_last_item['payload']['out_point']
                last_item_end = base_last_item['end']
                out_point_potential = min(
                    last_motion_record.n_frames - last_outpoint,
                    base_timeline.end - last_item_end)
                if last_outpoint < last_motion_record.n_frames and \
                        out_point_potential > 0:
                    out_point = last_outpoint + out_point_potential
                    in_point = last_outpoint
                    preload_task = base_last_item['payload']['preload_task']
                    motion_record_id = last_motion_record.motion_record_id
                    start_idx = last_item_end
                    end_idx = last_item_end + out_point - in_point
                    insert_success, base_timeline = await self._try_to_insert(
                        timeline=base_timeline,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        motion_record=last_motion_record,
                        in_point=in_point,
                        out_point=out_point,
                        trigger='RandomExtend',
                        preload_task=preload_task
                    )
                    if not insert_success:
                        msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                            'extend random motion failed, ' +\
                            f'motion record id: {motion_record_id}, ' +\
                            f'in-point: {in_point}, ' +\
                            f'out-point: {out_point}, ' +\
                            f'position: {start_idx}-{end_idx}'
                        self.logger.error(msg)
                        raise RuntimeError(msg)
        # TODO: Add cutoff filter to match a motion ends at base_timeline.end
        filter_pipeline = [
            self.avatar_filter,
            self.type_filter,
            self.random_filter,
        ]
        pipeline_input_template = dict(
            # All candidate motion records
            motion_records=self.cache.motion_records,
            # For avatar_filter
            avatar=request_dict['avatar'],
            # For type_filter
            motion_record_type=MotionRecordType.RANDOM,
        )
        if label_expression is not None:
            filter_pipeline.insert(2, self.label_filter)
            pipeline_input_template['label_expression'] = label_expression
        # Try to match a random motion
        intervals_to_skip = list()
        while True:
            blank_interval = await loop.run_in_executor(
                self.thread_pool_executor,
                base_timeline.get_next_blank,
                intervals_to_skip
            )
            if blank_interval is None:
                break
            pipeline_input = pipeline_input_template.copy()
            match_results = await filter_pipeline_retrieve(
                filter_pipeline=filter_pipeline,
                pipeline_input=pipeline_input,
                return_candidates=False
            )
            if len(match_results) > 0:
                motion_record_id = next(iter(match_results.keys()))
                motion_record = match_results[motion_record_id]
                preload_task = asyncio.create_task(
                    self.cache.get_motion_clip_by_id(motion_record_id)
                )
                out_point = min(
                    motion_record.n_frames,
                    blank_interval[1] - blank_interval[0])
                insert_success, base_timeline = await self._try_to_insert(
                    timeline=base_timeline,
                    start_idx=blank_interval[0],
                    end_idx=blank_interval[0] + out_point,
                    motion_record=motion_record,
                    in_point=0,
                    out_point=out_point,
                    trigger='RandomMatch',
                    preload_task=preload_task
                )
                if not insert_success:
                    msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                        'insert random motion failed, ' +\
                        f'motion record id: {motion_record_id}, ' +\
                        'in-point: 0, ' +\
                        f'out-point: {blank_interval[1]-blank_interval[0]}, ' +\
                        f'position: {blank_interval[0]}-{blank_interval[1]}'
                    self.logger.error(msg)
                    raise RuntimeError(msg)
                intervals_to_skip.append(
                    (blank_interval[0], blank_interval[0] + out_point))
        # If there are still blank intervals, use long idle motion to fill
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
        while True:
            blank_interval = await loop.run_in_executor(
                self.thread_pool_executor,
                base_timeline.get_next_blank,
            )
            if blank_interval is None:
                break
            pipeline_input = pipeline_input_template.copy()
            match_results = await filter_pipeline_retrieve(
                filter_pipeline=filter_pipeline,
                pipeline_input=pipeline_input,
                return_candidates=False
            )
            if len(match_results) > 0:
                motion_record_id = next(iter(match_results.keys()))
                motion_record = match_results[motion_record_id]
                preload_task = asyncio.create_task(
                    self.cache.get_motion_clip_by_id(motion_record_id)
                )
                out_point = motion_record.n_frames
                insert_success, base_timeline = await self._try_to_insert(
                    timeline=base_timeline,
                    start_idx=blank_interval[0],
                    end_idx=blank_interval[1],
                    motion_record=motion_record,
                    in_point=0,
                    out_point=blank_interval[1]-blank_interval[0],
                    trigger='IdleLongMatch',
                    preload_task=preload_task
                )
                if not insert_success:
                    msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                        'insert idle long motion failed, ' +\
                        f'motion record id: {motion_record_id}, ' +\
                        'in-point: 0, ' +\
                        f'out-point: {blank_interval[1]-blank_interval[0]}' +\
                        f'position: {blank_interval[0]}-{blank_interval[1]}'
                    self.logger.error(msg)
                    raise RuntimeError(msg)
            else:
                msg = f'No idle long motion found for avatar {request_dict["avatar"]}.'
                self.logger.error(msg)
                raise RuntimeError(msg)
        end_time = time.time()
        msg = f'Request {request_id} BodyChunk {sequence_number} ' + \
            f'extend base timeline took {end_time - start_time:.2f} seconds.'
        self.logger.debug(msg)
        return base_timeline

    async def _match_motion_keywords(
            self,
            request_id: str,
            base_timeline: Timeline,
            keyword_timeline: Timeline,
            original_end_idx: int,
            full_time_list: list[tuple[int, float]],
            motion_keywords: list[tuple[int, str]],
            label_expression: str | None = None,) -> Timeline:
        """Match motion keywords.

        Match corresponding motion clips on timeline based on motion keywords.

        Args:
            request_id (str):
                Request ID.
            base_timeline (Timeline):
                Base motion timeline.
            keyword_timeline (Timeline):
                Keyword motion timeline.
            original_end_idx (int):
                Original end frame index.
            full_time_list (list[tuple[int, float]]):
                Predicted speech time list.
            motion_keywords (list[tuple[int, str]]):
                Motion keywords list.
            label_expression (str | None, optional):
                Whether to limit motion range through label expression.
                If None, no limitation is applied. Defaults to None.

        Returns:
            Timeline:
                Processed motion timeline.
        """
        start_time = time.time()
        request_dict = self.request_space[request_id]
        loop = asyncio.get_running_loop()
        # Try to match motion keywords
        sequence_number = request_dict['generated_sequence_number'] + 1
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
        if label_expression is not None:
            filter_pipeline.insert(2, self.label_filter)
            pipeline_input_template['label_expression'] = label_expression
        intervals_to_skip = list()
        blank_interval = await loop.run_in_executor(
            self.thread_pool_executor,
            keyword_timeline.get_next_blank, intervals_to_skip)
        if blank_interval is None:
            msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                'no blank interval found during motion keyword matching.'
            self.logger.warning(msg)
        last_memory_end_frame = 0.0
        while blank_interval is not None:
            interval_start = blank_interval[0]
            pipeline_input = pipeline_input_template.copy()
            if interval_start == 0:
                pipeline_input['start_frame_lowerbound'] = \
                    0 - max_front_extension_n_frames
            else:
                pipeline_input['start_frame_lowerbound'] = interval_start
            interval_end = blank_interval[1]
            if interval_end == keyword_timeline.end:
                pipeline_input['end_frame_upperbound'] = \
                    keyword_timeline.end + max_rear_extension_n_frames
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
                    msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                        f'matching motion keyword {motion_keyword} ' +\
                        f'aligning at {align_frame} ' +\
                        f'exceeds interval range {interval_start}-{interval_end}, ' +\
                        'skip this interval.'
                    self.logger.debug(msg)
                    break
                pipeline_input['align_frame'] = align_frame
                pipeline_input['keyword'] = keyword_str
                match_results = await filter_pipeline_retrieve(
                    filter_pipeline=filter_pipeline,
                    pipeline_input=pipeline_input,
                    return_candidates=False
                )
                if len(match_results) > 0:
                    # TODO: Check if the blending interval of keyword motion
                    # conflicts with base_timeline
                    motion_record_id = next(iter(match_results.keys()))
                    motion_record = match_results[motion_record_id]
                    n_frames = motion_record.n_frames
                    keyword_frame = motion_record.motion_keyword.motion_keyword_frame
                    preload_task = asyncio.create_task(
                        self.cache.get_motion_clip_by_id(motion_record_id)
                    )
                    start_idx = align_frame - keyword_frame
                    # allow cross_chunk_outpoint to exceed
                    # the time range of current chunk
                    cross_chunk_outpoint = n_frames
                    end_idx = start_idx + cross_chunk_outpoint
                    insert_success, keyword_timeline = await self._try_to_insert(
                        timeline=keyword_timeline,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        motion_record=motion_record,
                        in_point=0,
                        out_point=cross_chunk_outpoint,
                        trigger=f'MotionKeywordMatch-{keyword_str}',
                        preload_task=preload_task
                    )
                    if not insert_success:
                        msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                            'insert motion keyword failed, ' +\
                            f'motion record id: {motion_record_id}, ' +\
                            'in-point: 0, ' +\
                            f'out-point: {cross_chunk_outpoint}' +\
                            f'position: {start_idx}-{end_idx}'
                        self.logger.error(msg)
                        interval_matched = False
                        break
                    # Record placeholder in memory
                    if start_idx > last_memory_end_frame:
                        place_holder_duration = (
                            start_idx - last_memory_end_frame) / self.__class__.FPS
                        await self.motion_keyword_memory_filter.memory.remember(
                            user_id=request_dict['user_id'],
                            event_id=-1,
                            event_duration=place_holder_duration,
                        )
                    # Add motion_record_id to memory
                    this_chunk_duration = (
                        min(end_idx, original_end_idx) - start_idx) / self.__class__.FPS
                    await self.motion_keyword_memory_filter.memory.remember(
                        user_id=request_dict['user_id'],
                        event_id=motion_record_id,
                        event_duration=this_chunk_duration,
                    )
                    last_memory_end_frame = min(end_idx, original_end_idx) - start_idx
                    interval_matched = True
                    break
            if not interval_matched:
                intervals_to_skip.append(blank_interval)
            blank_interval = await loop.run_in_executor(
                self.thread_pool_executor,
                keyword_timeline.get_next_blank,
                intervals_to_skip
            )
        if last_memory_end_frame < original_end_idx:
            place_holder_duration = (
                original_end_idx - last_memory_end_frame
            ) / self.__class__.FPS
            await self.motion_keyword_memory_filter.memory.remember(
                user_id=request_dict['user_id'],
                event_id=-1,
                event_duration=place_holder_duration,
            )
        end_time = time.time()
        self.logger.debug(
            f'Request {request_id} BodyChunk {sequence_number} ' +\
            'motion keyword matching completed, ' +
            f'took {end_time - start_time:.2f} seconds.'
        )
        return keyword_timeline

    async def _match_speech_keywords(
            self,
            request_id: str,
            base_timeline: Timeline,
            keyword_timeline: Timeline,
            original_end_idx: int,
            full_time_list: list[tuple[int, float]],
            speech_text: str,
            label_expression: str | None = None,) -> Timeline:
        """Match speech keywords.

        Match corresponding motion clips on timeline based on keywords
        in speech text.

        Args:
            request_id (str):
                Request ID.
            base_timeline (Timeline):
                Base motion timeline.
            keyword_timeline (Timeline):
                Keyword motion timeline.
            original_end_idx (int):
                Original end frame index.
            full_time_list (list[tuple[int, float]]):
                Predicted speech time list.
            speech_text (str):
                Speech text.
            label_expression (str | None, optional):
                Whether to limit motion range through label expression.
                If None, no limitation is applied. Defaults to None.

        Returns:
            Timeline:
                Processed motion timeline.
        """
        start_time = time.time()
        request_dict = self.request_space[request_id]
        loop = asyncio.get_running_loop()
        # Try to match speech keywords
        sequence_number = request_dict['generated_sequence_number'] + 1
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
        if label_expression is not None:
            filter_pipeline.insert(2, self.label_filter)
            pipeline_input_template['label_expression'] = label_expression
        text_segments = await loop.run_in_executor(
            self.thread_pool_executor,
            self.text_segmentation.cut_text,
            speech_text
        )
        intervals_to_skip = list()
        last_memory_end_frame = 0.0
        blank_interval = await loop.run_in_executor(
            self.thread_pool_executor,
            keyword_timeline.get_next_blank, intervals_to_skip)
        if blank_interval is None:
            msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                'no blank interval found during speech keyword matching.'
            self.logger.debug(msg)
        while blank_interval is not None:
            interval_start = blank_interval[0]
            pipeline_input = pipeline_input_template.copy()
            if interval_start == 0:
                pipeline_input['start_frame_lowerbound'] = \
                    0 - max_front_extension_n_frames
            else:
                pipeline_input['start_frame_lowerbound'] = interval_start
            interval_end = blank_interval[1]
            if interval_end == keyword_timeline.end:
                pipeline_input['end_frame_upperbound'] = \
                    keyword_timeline.end + max_rear_extension_n_frames
            else:
                pipeline_input['end_frame_upperbound'] = interval_end
            # Select text within the time range
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
                    char_time_tuple = full_time_list[char_idx]
                    msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                        'matching speech keyword at ' +\
                        f'char_idx={char_time_tuple[0]} ' +\
                        f'time={char_time_tuple[1]:.2f} seconds ' +\
                        f'char={text_segment["str"]}, exceeding ' +\
                        f'interval range {interval_start}-{interval_end}, ' +\
                        'skip this interval.'
                    self.logger.debug(msg)
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
                    preload_task = asyncio.create_task(
                        self.cache.get_motion_clip_by_id(motion_record_id)
                    )
                    start_idx = align_frame - keyword_frame
                    # To ensure the end position of the returned timeline
                    # matches the request body requirements
                    # this_chunk_outpoint may be earlier than the actual motion length
                    # remaining animation will be supplemented in the next BodyChunk
                    cross_chunk_outpoint = n_frames
                    end_idx = start_idx + cross_chunk_outpoint
                    insert_success, keyword_timeline = await self._try_to_insert(
                        timeline=keyword_timeline,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        motion_record=motion_record,
                        in_point=0,
                        out_point=cross_chunk_outpoint,
                        trigger=f'SpeechKeywordMatch-{text_segment["str"]}',
                        preload_task=preload_task
                    )
                    if not insert_success:
                        msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                            'insert speech keyword failed, ' +\
                            f'motion record id: {motion_record_id}, ' +\
                            'in-point: 0, ' +\
                            f'out-point: {cross_chunk_outpoint}' +\
                            f'position: {start_idx}-{end_idx}'
                        self.logger.error(msg)
                        interval_matched = False
                        break
                    # Record placeholder in memory
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
                    this_chunk_duration = (
                        min(end_idx, original_end_idx) - start_idx
                        ) / self.__class__.FPS
                    await self.speech_keyword_memory_filter.memory.remember(
                        user_id=request_dict['user_id'],
                        event_id=motion_record_id,
                        event_duration=this_chunk_duration,
                    )
                    last_memory_end_frame = min(end_idx, original_end_idx) - start_idx
                    interval_matched = True
                    break
            if not interval_matched:
                intervals_to_skip.append(blank_interval)
            blank_interval = await loop.run_in_executor(
                self.thread_pool_executor,
                keyword_timeline.get_next_blank,
                intervals_to_skip
            )
        if last_memory_end_frame < original_end_idx:
            place_holder_duration = (
                original_end_idx - last_memory_end_frame
            ) / self.__class__.FPS
            await self.motion_keyword_memory_filter.memory.remember(
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
        return keyword_timeline

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

        Raises:
            ValueError:
                Raised when predicted speech time list length does not match
                speech text length.
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
            preload_task (asyncio.Task | None, optional):
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
            msg = f'Insert motion failed: {e}'
            self.logger.error(msg)
            timeline = timeline_backup
            return False, timeline
        return True, timeline

    async def _wait_for_motion_clips(
            self,
            timeline: Timeline,
            slice_start_idx: int | None = None,
            slice_end_idx: int | None = None
            ) -> list[MotionClip]:
        """Wait for motion clips to load.

        Args:
            timeline (Timeline):
                Motion timeline.
            slice_start_idx (int | None, optional):
                Slice start frame. If None, uses timeline start frame.
                Defaults to None.
            slice_end_idx (int | None, optional):
                Slice end frame. If None, uses timeline end frame.
                Defaults to None.

        Returns:
            list[MotionClip]:
                Motion clips list.
        """
        loop = asyncio.get_running_loop()
        timeline_list = await loop.run_in_executor(
            self.thread_pool_executor,
            timeline.to_list,
            slice_start_idx,
            slice_end_idx
        )
        while True:
            all_motion_clips_read = True
            for timeline_item in timeline_list:
                if 'motion_clip' in timeline_item['payload']:
                    continue
                preload_task: asyncio.Task | None = \
                    timeline_item['payload']['preload_task']
                if preload_task is None:
                    # No preload task, skip
                    continue
                elif preload_task.done():
                    motion_clip = preload_task.result()
                    timeline_item['payload']['motion_clip'] = motion_clip
                else:
                    # Preload task exists and not done, continue waiting
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
        merge_clip = await self.interpolation.merge(
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

    async def _convert_motion_clip_app(
            self,
            src_motion_clip: MotionClip,
            dst_app_name: Literal['babylon']
            ) -> MotionClip:
        """Convert src_motion_clip to babylon application format.

        Args:
            src_motion_clip (MotionClip):
                Source motion_clip with app_name as None, representing
                motion library format, no need to check app_name.
            dst_app_name (Literal['babylon']):
                Target application name, currently only supports 'babylon'.

        Returns:
            MotionClip:
                Converted motion_clip with app_name as 'babylon'.

        Raises:
            ValueError:
                Raised when unsupported target application name is provided.
        """
        restpose_name = src_motion_clip.restpose_name
        restpose = await self.cache.get_restpose_by_name(restpose_name)
        loop = asyncio.get_running_loop()
        if dst_app_name == 'babylon':
            dst_joint_names, dst_rotmat, dst_root_world_position = \
                await loop.run_in_executor(
                    self.thread_pool_executor,
                    self._convert_to_babylon,
                    restpose,
                    src_motion_clip.joint_names,
                    src_motion_clip.joint_rotmat,
                    src_motion_clip.root_world_position
                )
            ret_motion_clip = src_motion_clip.clone()
            ret_motion_clip.app_name = 'babylon'
            ret_motion_clip.set_joint_rotmat(
                joint_rotmat=dst_rotmat,
                joint_names=dst_joint_names
            )
            ret_motion_clip.set_root_world_position(
                root_world_position=dst_root_world_position
            )
            return ret_motion_clip
        else:
            msg = f'Unsupported app_name: {dst_app_name}'
            self.logger.error(msg)
            raise ValueError(msg)

    def _convert_to_babylon(
            self,
            restpose: Restpose,
            src_joint_names: list[str],
            src_matrix_basis: np.ndarray,
            src_root_world_position: np.ndarray,
            ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Convert src_matrix_basis to babylon application format.

        Args:
            restpose (Restpose):
                Restpose data.
            src_joint_names (list[str]):
                Source joint names.
            src_matrix_basis (np.ndarray):
                Matrix basis in source world coordinate system, shape
                (n_frames, n_joints, 3, 3), where n_joints=len(src_joint_names).
            src_root_world_position (np.ndarray):
                Root bone position in source world coordinate system.

        Returns:
            tuple[list[str], np.ndarray, np.ndarray]:
                list[str]: Converted joint names list, can be different from
                    src_joint_names, but should not introduce too many
                    unnecessary bones.
                np.ndarray: Rotation matrices ready for application use, shape
                    (n_frames, n_joints, 3, 3).
                np.ndarray: Root bone position in application world coordinate
                    system ready for application use.
        """
        n_frames = src_matrix_basis.shape[0]
        n_joints = src_matrix_basis.shape[1]

        # Convert 3x3 rotation matrices to 4x4 motion basis matrices
        eye_4 = np.eye(4, dtype=src_matrix_basis.dtype)
        basis_matrices = np.tile(eye_4, (n_frames, n_joints, 1, 1))
        basis_matrices[:, :, :3, :3] = src_matrix_basis
        # Extract local_matrices corresponding to src_joint_names
        # from Restpose full data
        joint_indices = [
            restpose.get_joint_index(joint_name) for joint_name in src_joint_names]
        joint_indices = np.array(joint_indices, dtype=np.int32)
        local_matrices = restpose.local_matrices[joint_indices]
        # Create a 4D array to store results
        parent_space_transforms = np.zeros((n_frames, n_joints, 4, 4),
                                           dtype=src_matrix_basis.dtype)
        # Temporarily set indices without parent joints to 0
        # for batch creation of parent_local_mats
        parent_indices = [
            restpose.parent_indices[joint_idx] for joint_idx in joint_indices]
        parent_indices = np.array(parent_indices, dtype=np.int32)
        without_parent_indices = np.where(parent_indices < 0)[0]
        parent_indices[without_parent_indices] = 0
        parent_local_matrices = restpose.local_matrices[parent_indices]
        # Set positions without parent joints to
        # identity matrix to eliminate parent_inv influence
        parent_local_matrices[without_parent_indices] = np.eye(
            4, dtype=src_matrix_basis.dtype)
        parent_inv = np.linalg.inv(parent_local_matrices)
        parent_space_transforms = parent_inv @ local_matrices @ basis_matrices
        # Extract rotation matrices
        tgt_matrices = parent_space_transforms[:, :, :3, :3]
        tgt_transl = src_root_world_position * 100
        tgt_transl[:, [1, 2]] = tgt_transl[:, [2, 1]]
        tgt_transl[:, 2] = -tgt_transl[:, 2]
        return src_joint_names, tgt_matrices, tgt_transl

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

    async def _maintain_check(self) -> None:
        """Check if cache needs update and if there are expired requests.

        Periodically checks cache version and request status, automatically
        updates cache and cleans up expired requests.
        """
        # 检查Cache
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
        """Build meta_reader, motion_reader and restpose_reader.

        Create various data reader instances based on configuration.
        """
        meta_reader_cfg = self.meta_reader_cfg.copy()
        meta_reader_cfg['logger_cfg'] = self.logger_cfg
        self.meta_reader = build_meta_reader(meta_reader_cfg)
        motion_reader_cfg = self.motion_reader_cfg.copy()
        motion_reader_cfg['logger_cfg'] = self.logger_cfg
        motion_reader_cfg['thread_pool_executor'] = self.thread_pool_executor
        self.motion_reader = build_motion_reader(motion_reader_cfg)
        restpose_reader_cfg = self.restpose_reader_cfg.copy()
        restpose_reader_cfg['thread_pool_executor'] = self.thread_pool_executor
        restpose_reader_cfg['logger_cfg'] = self.logger_cfg
        self.restpose_reader = build_restpose_reader(restpose_reader_cfg)

    def _build_cache(self) -> LocalCache:
        """Build cache.

        Returns:
            LocalCache: Built local cache instance.
        """
        cache_cfg = self.cache_cfg.copy()
        cache_cfg['logger_cfg'] = self.logger_cfg
        cache_cfg['thread_pool_executor'] = self.thread_pool_executor
        cache_cfg['meta_reader'] = self.meta_reader
        cache_cfg['motion_reader'] = self.motion_reader
        cache_cfg['restpose_reader'] = self.restpose_reader
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

        Build interpolation merge and blending merge for motion clip
        merging processing.
        """
        interpolation_cfg = self.interpolation_cfg.copy()
        interpolation_cfg['logger_cfg'] = self.logger_cfg
        interpolation_cfg['thread_pool_executor'] = self.thread_pool_executor
        self.interpolation = build_motion_clip_merge(interpolation_cfg)
        blending_cfg = self.blending_cfg.copy()
        blending_cfg['logger_cfg'] = self.logger_cfg
        blending_cfg['thread_pool_executor'] = self.thread_pool_executor
        self.blending = build_motion_clip_merge(blending_cfg)

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
        label_filter_cfg = dict(
            type='LabelFilter',
            name='label_filter',
            mapping=self.cache.label_mapping,
            logger_cfg=self.logger_cfg,
        )
        self.label_filter = build_filter(label_filter_cfg)
        self.filters_version = await self.cache.get_version()

