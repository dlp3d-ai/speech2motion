import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel

from ..cache.builder import build_cache
from ..cache.local_cache import LocalCache
from ..data_structures.motion_clip import MotionClip
from ..data_structures.motion_record import MotionRecordType
from ..data_structures.restpose import Restpose
from ..data_structures.timeline import Timeline
from ..filters.builder import (
    LabelFilter,
    build_filter,
)
from ..io.restpose.builder import build_restpose_reader
from ..retrieve.filter_pipeline import filter_pipeline_retrieve
from ..utils.super import Super
from .streaming_speech2motion_v1 import (
    StreamingSpeech2MotionV1,
    StreamingSpeech2MotionV1ChunkBody,
    StreamingSpeech2MotionV1ChunkEnd,
    StreamingSpeech2MotionV1ChunkStart,
)

if TYPE_CHECKING:
    from ..io.restpose.base_restpose_reader import BaseRestposeReader


class StreamingSpeech2MotionV2ChunkStart(BaseModel):
    """V2 streaming motion generation start request.

    This class represents the initial request to start a streaming motion
    generation session. It contains all the necessary configuration
    parameters for the motion generation process.

    Args:
        request_id (str):
            Request ID for identifying an independent streaming motion
            generation request.
        user_id (str):
            User ID for distinguishing the requesting user and calling
            related memory.
        avatar (str):
            Avatar selected by user.
        app_name (Literal['babylon', 'python_backend']):
            Application name, defaults to 'python_backend'.
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
        return_content (Literal['motion_clip', 'log'):
            Return content type. Defaults to 'motion_clip'.
    """
    request_id: str
    user_id: str
    avatar: str
    app_name: Literal['babylon', 'python_backend'] = 'python_backend'
    max_front_extension_duration: float = 0.0
    max_rear_extension_duration: float = 0.0
    memory_duration_override: float | None = None
    first_body_fast_response_override: bool | None = None
    idle_long_extendable: bool = False
    return_content: Literal['motion_clip', 'log'] = 'motion_clip'

class StreamingSpeech2MotionV2ChunkBody(BaseModel):
    """V2 streaming motion generation body request.

    This class represents the body request containing speech text and
    motion generation parameters for a specific chunk of the streaming
    motion generation process.

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
            Whether to limit speech random motion scope through label
            expression. Defaults to None, indicating no limitation.
    """
    request_id: str
    duration: float
    speech_text: str
    sequence_number: int
    speech_time: list[tuple[int, float]] | None = None
    motion_keywords: list[tuple[int, str]] | None = None
    label_expression: str | None = None

class StreamingSpeech2MotionV2ChunkEnd(BaseModel):
    """V2 streaming motion generation end request.

    This class represents the end request that signals the completion
    of a streaming motion generation session.

    Args:
        request_id (str):
            Request ID for identifying an independent streaming motion
            generation request.
    """
    request_id: str

class StreamingSpeech2MotionV2(StreamingSpeech2MotionV1):
    """V2 streaming motion generator.

    This class extends the V1 streaming motion generator with additional
    features including restpose support, label filtering, and enhanced
    motion conversion capabilities for different applications.
    """

    def __init__(
            self,
            meta_reader_cfg: dict[str, Any],
            motion_reader_cfg: dict[str, Any],
            restpose_reader_cfg: dict[str, Any],
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
        """Initialize V2 streaming motion generator.

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
            merge_cfg (dict[str, Any]):
                Motion merger configuration.
            request_expire_time (int, optional):
                Request expiration time in seconds, defaults to 120.
                When a request has no new ChunkBody or ChunkEnd requests
                within request_expire_time, it will automatically expire.
            maintain_check_interval (int, optional):
                Maintenance task check interval, including motion library
                cache version check and expired request check.
                Defaults to 60.
            max_workers (int, optional):
                Maximum number of worker threads. Defaults to 4.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor. If None, creates a new thread pool
                executor based on max_workers. Defaults to None.
            first_body_fast_response (bool, optional):
                Whether to enable fast response for first Body request.
                Defaults to False.
            sleep_time (float, optional):
                Sleep time when waiting for data reading operations to
                complete, defaults to 0.01 seconds.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration. Defaults to None.
        """
        Super.__init__(self, logger_cfg=logger_cfg)
        self.restpose_reader_cfg = restpose_reader_cfg
        self.restpose_reader: BaseRestposeReader | None = None
        # Filter for retrieval
        self.label_filter: LabelFilter | None = None
        StreamingSpeech2MotionV1.__init__(
            self,
            meta_reader_cfg=meta_reader_cfg,
            motion_reader_cfg=motion_reader_cfg,
            cache_cfg=cache_cfg,
            memory_cfg=memory_cfg,
            text_segmentation_cfg=text_segmentation_cfg,
            merge_cfg=merge_cfg,
            request_expire_time=request_expire_time,
            maintain_check_interval=maintain_check_interval,
            max_workers=max_workers,
            thread_pool_executor=thread_pool_executor,
            first_body_fast_response=first_body_fast_response,
            sleep_time=sleep_time,
            logger_cfg=logger_cfg
        )

    async def startup(self) -> None:
        """Start up the API.

        Initializes all necessary components for the streaming motion
        generation service including readers, cache, and filters.
        """
        await StreamingSpeech2MotionV1.startup(self)

    async def handle_chunk_start(
            self,
            chunk_start: StreamingSpeech2MotionV2ChunkStart |
            StreamingSpeech2MotionV1ChunkStart
            ) -> None:
        """Handle streaming motion generation start request.

        Processes the initial request to start a streaming motion generation
        session. Converts V2 requests to V1 format and stores application
        name for later use.

        Args:
            chunk_start (
                    StreamingSpeech2MotionV2ChunkStart |
                    StreamingSpeech2MotionV1ChunkStart):
                Streaming motion generation start request.
        """
        if isinstance(chunk_start, StreamingSpeech2MotionV2ChunkStart):
            v1_chunk_start = StreamingSpeech2MotionV1ChunkStart(
                request_id=chunk_start.request_id,
                user_id=chunk_start.user_id,
                avatar=chunk_start.avatar,
                max_front_extension_duration=chunk_start.max_front_extension_duration,
                max_rear_extension_duration=chunk_start.max_rear_extension_duration,
                memory_duration_override=chunk_start.memory_duration_override,
                first_body_fast_response_override=chunk_start.first_body_fast_response_override,
                idle_long_extendable=chunk_start.idle_long_extendable,
                return_content=chunk_start.return_content
            )
            app_name = chunk_start.app_name
        else:
            v1_chunk_start = chunk_start
            app_name = 'python_backend'
        await StreamingSpeech2MotionV1.handle_chunk_start(
            self, chunk_start=v1_chunk_start
        )
        self.request_space[chunk_start.request_id]['app_name'] = app_name

    async def handle_chunk_body(
            self,
            chunk_body: StreamingSpeech2MotionV2ChunkBody |
            StreamingSpeech2MotionV1ChunkBody
            ) -> MotionClip | str:
        """Handle streaming motion generation body request.

        Processes a body request containing speech text and motion generation
        parameters. Performs motion matching, timeline processing, and returns
        the generated motion clip or log information.

        Args:
            chunk_body (
                    StreamingSpeech2MotionV2ChunkBody |
                    StreamingSpeech2MotionV1ChunkBody):
                Streaming motion generation body request.

        Returns:
            MotionClip | str:
                Return result. If return_content is 'motion_clip', returns
                MotionClip instance, otherwise returns log string.
        """
        chunk_body_msg = f'Received request {chunk_body.request_id} for ' +\
            f'BodyChunk {chunk_body.sequence_number}.'
        self.logger.info(chunk_body_msg)
        msg = f'Speech duration: {chunk_body.duration}s, ' +\
            f'Speech text: {chunk_body.speech_text}\n' +\
            f'Speech time list: {chunk_body.speech_time}\n' +\
            f'Motion keywords list: {chunk_body.motion_keywords}'
        self.logger.debug(msg)
        if hasattr(chunk_body, 'label_expression'):
            label_expression = chunk_body.label_expression
        else:
            label_expression = None
        body_duration = chunk_body.duration
        body_n_frames = int(body_duration * self.__class__.FPS)
        request_dict = self.request_space[chunk_body.request_id]
        request_dict['received_sequence_number'] = max(
            request_dict['received_sequence_number'],
            chunk_body.sequence_number
        )
        request_time = time.time()
        request_dict['last_request_time'] = request_time
        await self._maintain_check()
        while request_dict['generated_sequence_number'] < \
                chunk_body.sequence_number - 1:
            time_diff = time.time() - request_time
            if time_diff > self.request_expire_time:
                msg = f'Request {chunk_body.request_id} for ' + \
                    f'BodyChunk {chunk_body.sequence_number} ' + \
                    'timed out waiting for previous matching results, ' + \
                    'terminating matching.'
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
            msg = f'Request {chunk_body.request_id} for ' +\
                f'BodyChunk {chunk_body.sequence_number} ' +\
                'has no speech time information, ' +\
                'predicting speech time based on uniform distribution.'
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
                motion_keywords=motion_keywords,
                label_expression=label_expression
            )
            # Speech keyword matching
            timeline = await self._match_speech_keywords(
                request_id=chunk_body.request_id,
                timeline=timeline,
                full_time_list=full_time_list,
                speech_text=chunk_body.speech_text,
                label_expression=label_expression
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
                timeline=timeline,
                label_expression=label_expression
            )
        else:
            msg = f'Request {chunk_body.request_id} ' +\
                f'BodyChunk {chunk_body.sequence_number} ' + \
                'is the first Body request, enabling fast response, ' +\
                'skipping motion keyword matching and speech keyword matching.'
            self.logger.debug(msg)
            smooth_start = False
            last_chunk_end_payload = None
        # Fill with random motion
        timeline = await self._fill_up_with_random_motion(
            request_id=chunk_body.request_id,
            timeline=timeline,
            label_expression=label_expression
        )
        # Fill with long IDLE motion
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
            f'matching completed in {time.time() - start_time:.2f} seconds.'
        )
        if request_dict['return_content'] == 'motion_clip':
            waiting_for_read_start = time.time()
            motion_clips = await self._wait_for_motion_clips(timeline)
            end_time = time.time()
            self.logger.debug(
                f'Request {chunk_body.request_id} ' +\
                f'BodyChunk {chunk_body.sequence_number} ' + \
                f'finishes loading {len(motion_clips)} motion clips in ' + \
                f'{end_time - waiting_for_read_start:.2f} seconds.'
            )
            merge_start_time = time.time()
            # last_chunk_end_payload for the first BodyChunk request is None
            if not smooth_start and last_chunk_end_payload is not None:
                msg = f'Request {chunk_body.request_id} ' +\
                      f'BodyChunk {chunk_body.sequence_number} ' + \
                      'chunk start cannot smoothly connect ' +\
                      'with previous chunk end, ' + \
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
                'merging completed in ' + \
                f'{merge_end_time - merge_start_time:.2f} seconds.'
            )
            app_name = request_dict['app_name']
            if app_name == 'python_backend':
                motion_clip.app_name = app_name
            else:
                convert_start_time = time.time()
                motion_clip = await self._convert_motion_clip_app(
                    motion_clip,
                    app_name
                )
                convert_end_time = time.time()
                self.logger.debug(
                    f'Request {chunk_body.request_id} ' +\
                    f'BodyChunk {chunk_body.sequence_number} ' + \
                    f'conversion to {app_name} completed in ' + \
                    f'{convert_end_time - convert_start_time:.2f} seconds.'
                )
            motion_clip.set_timeline_start_idx(timeline.start)
            ret_value = motion_clip
        else:
            ret_value = await self._read_log_file(
                chunk_body_msg)
        return ret_value

    async def handle_chunk_end(
            self,
            chunk_end: StreamingSpeech2MotionV2ChunkEnd |
            StreamingSpeech2MotionV1ChunkEnd
            ) -> MotionClip | str | None:
        """Handle streaming motion generation end request.

        Processes the end request that signals completion of a streaming
        motion generation session. Handles any necessary motion extensions
        and converts the result to the appropriate application format.

        Args:
            chunk_end (
                    StreamingSpeech2MotionV2ChunkEnd |
                    StreamingSpeech2MotionV1ChunkEnd):
                Streaming motion generation end request.

        Returns:
            MotionClip | str | None:
                Returns None if there is no extension data needed based on
                previous requests. When extension is needed, returns MotionClip
                instance if return_content is 'motion_clip', otherwise returns
                log string.
        """
        return_content = self.request_space[chunk_end.request_id]['return_content']
        app_name = self.request_space[chunk_end.request_id]['app_name']
        if isinstance(chunk_end, StreamingSpeech2MotionV2ChunkEnd):
            v1_chunk_end = StreamingSpeech2MotionV1ChunkEnd(
                request_id=chunk_end.request_id
            )
        else:
            v1_chunk_end = chunk_end
        ret_value = await StreamingSpeech2MotionV1.handle_chunk_end(
            self, chunk_end=v1_chunk_end
        )
        # Note: StreamingSpeech2MotionV1.handle_chunk_end has already
        # removed request_id from request_space, cannot access request_space cache here
        if return_content == 'motion_clip' and\
                app_name != 'python_backend':
            if isinstance(ret_value, MotionClip):
                convert_start_time = time.time()
                ret_value = await self._convert_motion_clip_app(
                    ret_value,
                    app_name
                )
                convert_end_time = time.time()
                self.logger.debug(
                    f'Request {chunk_end.request_id} stream end request ' + \
                    f'conversion to {app_name} completed in ' + \
                    f'{convert_end_time - convert_start_time:.2f} seconds.'
                )
            else:
                self.logger.warning(
                    f'Request {chunk_end.request_id} stream end request ' + \
                    'MotionClip is empty'
                )
        return ret_value


    async def _handle_current_chunk_end(
            self,
            request_id: str,
            timeline: Timeline,
            label_expression: str | None = None,
            ) -> Timeline:
        """Handle current chunk end.

        Processes the end of the current chunk by matching random motions
        to fill any remaining blank intervals in the timeline.

        Args:
            request_id (str):
                Request ID.
            timeline (Timeline):
                Motion timeline.
            label_expression (str | None, optional):
                Label expression, defaults to None, indicating no limitation.

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
            # processed during successful motion matching
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
            if label_expression is not None:
                filter_pipeline.insert(2, self.label_filter)
                pipeline_input['label_expression'] = label_expression
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
                    msg = 'Failed to insert motion during ' +\
                        'current chunk end matching, ' +\
                        f'Motion ID: {motion_record_id}, In point: 0, ' +\
                        f'Out point: {this_chunk_outpoint}'
                    self.logger.error(msg)
                    msg = f'Request {request_id} BodyChunk {sequence_number} end ' + \
                        'failed to process due to insertion failure, ' +\
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
                    # Use -1 as placeholder in memory before actual motion occurs
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

    async def _match_motion_keywords(
            self,
            request_id: str,
            timeline: Timeline,
            full_time_list: list[tuple[int, float]],
            motion_keywords: list[tuple[int, str]],
            label_expression: str | None = None,) -> Timeline:
        """Match motion keywords.

        Matches motion keywords to appropriate time intervals in the timeline
        by finding suitable motion records and inserting them at the correct
        positions.

        Args:
            request_id (str):
                Request ID.
            timeline (Timeline):
                Motion timeline.
            full_time_list (list[tuple[int, float]]):
                Predicted speech time list.
            motion_keywords (list[tuple[int, str]]):
                Motion keywords list.
            label_expression (str | None, optional):
                Label expression, defaults to None, indicating no limitation.

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
        if label_expression is not None:
            filter_pipeline.insert(2, self.label_filter)
            pipeline_input_template['label_expression'] = label_expression
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
        if blank_interval is None:
            msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                'no blank intervals found during motion keyword matching stage.'
            self.logger.warning(msg)
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
                    msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                        f'matching motion keyword {motion_keyword} ' +\
                        f'at frame {align_frame} exceeds ' +\
                        f'interval range {interval_start}-{interval_end}, ' +\
                        'skipping this interval.'
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
                    motion_record_id = next(iter(match_results.keys()))
                    motion_record = match_results[motion_record_id]
                    n_frames = motion_record.n_frames
                    keyword_frame = motion_record.motion_keyword.motion_keyword_frame
                    return_content = request_dict['return_content']
                    preload_task = asyncio.create_task(
                        self.cache.get_motion_clip_by_id(motion_record_id)
                    ) if return_content == 'motion_clip' else None
                    start_idx = align_frame - keyword_frame
                    # To ensure the returned timeline end position
                    # matches the request body requirements
                    # this_chunk_outpoint may end earlier than the actual motion length
                    # remaining animation will be supplemented in the next BodyChunk
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
                            'failed to insert motion keyword, ' +\
                            f'Motion ID: {motion_record_id}, ' +\
                            'In point: 0, ' +\
                            f'Out point: {this_chunk_outpoint}'
                        self.logger.error(msg)
                        interval_matched = False
                        break
                    else:
                        # Record placeholder in memory,
                        # considering forward extension case
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
            'motion keyword matching completed in ' +
            f'{end_time - start_time:.2f} seconds.'
        )
        return timeline

    async def _match_speech_keywords(
            self,
            request_id: str,
            timeline: Timeline,
            full_time_list: list[tuple[int, float]],
            speech_text: str,
            label_expression: str | None = None,) -> Timeline:
        """Match speech keywords.

        Matches speech keywords extracted from text segments to appropriate
        time intervals in the timeline by finding suitable motion records
        and inserting them at the correct positions.

        Args:
            request_id (str):
                Request ID.
            timeline (Timeline):
                Motion timeline.
            full_time_list (list[tuple[int, float]]):
                Predicted speech time list.
            speech_text (str):
                Speech text.
            label_expression (str | None, optional):
                Whether to limit speech random motion scope through label
                expression. Defaults to None, indicating no limitation.

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
        if label_expression is not None:
            filter_pipeline.insert(2, self.label_filter)
            pipeline_input_template['label_expression'] = label_expression
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
        if blank_interval is None:
            msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                'no blank intervals found during speech keyword matching stage.'
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
                    char_time_tuple = full_time_list[char_idx]
                    msg = f'Request {request_id} BodyChunk {sequence_number} ' +\
                        f'matching speech keyword {text_segment["str"]} at position ' +\
                        f'{char_time_tuple[0]} ({char_time_tuple[1]:.2f}s) exceeds ' +\
                        f'interval range {interval_start}-{interval_end}, ' +\
                        'skipping this interval.'
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
                    return_content = request_dict['return_content']
                    preload_task = asyncio.create_task(
                        self.cache.get_motion_clip_by_id(motion_record_id)
                    ) if return_content == 'motion_clip' else None
                    start_idx = align_frame - keyword_frame
                    # To ensure the returned timeline end position matches
                    # the request body requirements
                    # this_chunk_outpoint may end earlier than the actual motion length
                    # remaining animation will be supplemented in the next BodyChunk
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
                            'failed to insert speech keyword, ' +\
                            f'Motion ID: {motion_record_id}, ' +\
                            'In point: 0, ' +\
                            f'Out point: {this_chunk_outpoint}'
                        self.logger.error(msg)
                        interval_matched = False
                        break
                    else:
                        # Record placeholder in memory,
                        # considering forward extension case
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
            'speech keyword matching completed in ' +
            f'{end_time - start_time:.2f} seconds.'
        )
        return timeline

    async def _fill_up_with_random_motion(
            self,
            request_id: str,
            timeline: Timeline,
            label_expression: str | None = None,
            ) -> Timeline:
        """Fill up with random motion.

        Fills remaining blank intervals in the timeline with random motions
        to ensure complete coverage of the requested duration.

        Args:
            request_id (str):
                Request ID.
            timeline (Timeline):
                Motion timeline.
            label_expression (str | None, optional):
                Label expression, defaults to None, indicating no limitation.

        Returns:
            Timeline:
                Timeline filled with random motions.
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
            # For cutoff_duration_filter
            cutoff_duration_lowerbound=0.0,
            cutoff_duration_upperbound=None,
            # For chunk_end_memory_filter and random_memory_filter
            user_id=request_dict['user_id'],
        )
        if label_expression is not None:
            filter_pipeline.insert(2, self.label_filter)
            pipeline_input_template['label_expression'] = label_expression
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
                    msg = 'Failed to insert random motion during ' +\
                        'random motion filling, ' +\
                        f'Motion ID: {motion_record_id}, In point: 0, ' +\
                        f'Out point: {out_point}'
                    self.logger.error(msg)
                    intervals_to_skip.append(blank_interval)
                    continue
                # Placeholder before actual motion occurs
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


    async def _convert_motion_clip_app(
            self,
            src_motion_clip: MotionClip,
            dst_app_name: Literal['babylon']
            ) -> MotionClip:
        """Convert motion clip to target application format.

        Converts a motion clip from the source format to the target
        application format, currently supporting conversion to Babylon format.

        Args:
            src_motion_clip (MotionClip):
                Source motion clip with app_name as None, representing
                motion library format, no need to check app_name.
            dst_app_name (Literal['babylon']):
                Target application name, currently only supports 'babylon'.

        Returns:
            MotionClip:
                Converted motion clip with app_name set to 'babylon'.
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
        """Convert matrix basis to Babylon application format.

        Converts rotation matrices and root positions from the source format
        to Babylon application format, including coordinate system transformations
        and joint hierarchy adjustments.

        Args:
            restpose (Restpose):
                Static pose data.
            src_joint_names (list[str]):
                Source joint names.
            src_matrix_basis (np.ndarray):
                Source matrix basis in world coordinate system with shape
                (n_frames, n_joints, 3, 3), where n_joints=len(src_joint_names).
            src_root_world_position (np.ndarray):
                Root bone position in source world coordinate system.

        Returns:
            tuple[list[str], np.ndarray, np.ndarray]:
                list[str]: Converted joint names list, which can be different
                    from src_joint_names, but should not introduce too many
                    unnecessary bones.
                np.ndarray: Rotation matrices ready for application use with shape
                    (n_frames, n_joints, 3, 3).
                np.ndarray: Root bone position in application world coordinate
                    system ready for use.
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

    def _build_readers(self) -> None:
        """Build meta_reader, motion_reader and restpose_reader.

        Initializes all necessary readers for metadata, motion data,
        and restpose data access.
        """
        StreamingSpeech2MotionV1._build_readers(self)
        restpose_reader_cfg = self.restpose_reader_cfg.copy()
        restpose_reader_cfg['thread_pool_executor'] = self.thread_pool_executor
        restpose_reader_cfg['logger_cfg'] = self.logger_cfg
        self.restpose_reader = build_restpose_reader(restpose_reader_cfg)

    def _build_cache(self) -> LocalCache:
        """Build cache.

        Creates and configures the local cache with all necessary readers
        and configuration parameters.

        Returns:
            LocalCache:
                Configured local cache instance.
        """
        cache_cfg = self.cache_cfg.copy()
        cache_cfg['logger_cfg'] = self.logger_cfg
        cache_cfg['thread_pool_executor'] = self.thread_pool_executor
        cache_cfg['meta_reader'] = self.meta_reader
        cache_cfg['motion_reader'] = self.motion_reader
        cache_cfg['restpose_reader'] = self.restpose_reader
        return build_cache(cache_cfg)

    async def _build_filters(self) -> None:
        """Build filters.

        Initializes all necessary filters for motion matching and processing,
        including the label filter specific to V2 functionality.
        """
        await StreamingSpeech2MotionV1._build_filters(self)
        label_filter_cfg = dict(
            type='LabelFilter',
            name='label_filter',
            mapping=self.cache.label_mapping,
            logger_cfg=self.logger_cfg,
        )
        self.label_filter = build_filter(label_filter_cfg)
