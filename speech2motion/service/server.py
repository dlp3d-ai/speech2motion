import asyncio
import io
import os
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import uvicorn
from fastapi import (
    APIRouter,
    FastAPI,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..apis.builder import build_api
from ..apis.streaming_speech2motion_v2 import (
    StreamingSpeech2MotionV1ChunkBody,
    StreamingSpeech2MotionV1ChunkEnd,
    StreamingSpeech2MotionV1ChunkStart,
    StreamingSpeech2MotionV2,
    StreamingSpeech2MotionV2ChunkBody,
    StreamingSpeech2MotionV2ChunkEnd,
    StreamingSpeech2MotionV2ChunkStart,
)
from ..apis.streaming_speech2motion_v3 import (
    StreamingSpeech2MotionV3,
    StreamingSpeech2MotionV3ChunkBody,
    StreamingSpeech2MotionV3ChunkEnd,
    StreamingSpeech2MotionV3ChunkStart,
)
from ..data_structures.motion_clip import MotionClip
from ..utils.io import export_npz
from ..utils.super import Super
from .exceptions import NoLogFileException, register_error_handlers

try:
    from ..io.protobuf import streaming_v3_pb2
    v3_pb2_imported = True
    v3_pb2_traceback_str = None
except ImportError:
    v3_pb2_imported = False
    v3_pb2_traceback_str = traceback.format_exc()


class LogResponse(BaseModel):
    """Log response model for API responses containing log messages."""
    log: str

class Speech2MotionV2ResponseChunkEnd(BaseModel):
    """Stream end response model for V2 API."""
    request_id: str

class Speech2MotionV3ResponseChunkEnd(BaseModel):
    """Stream end response model for V3 API."""
    request_id: str

class FastAPIServer(Super):
    """Backend server for handling HTTP requests and WebSocket connections.

    This class provides a FastAPI-based web server that supports streaming
    speech-to-motion API interfaces, including HTTP and WebSocket interfaces,
    supporting V1, V2, and V3 API versions.
    """

    def __init__(
        self,
        python_api_v2_cfg: dict | None = None,
        python_api_v3_cfg: dict | None = None,
        max_workers: int = 4,
        enable_cors: bool = False,
        host: str = '0.0.0.0',
        port: int = 80,
        startup_event_listener: None | list = None,
        shutdown_event_listener: None | list = None,
        logger_cfg: None | dict = None,
    ) -> None:
        """Initialize FastAPI server instance.

        Args:
            python_api_v2_cfg (dict | None, optional):
                Python API V2 configuration dictionary. Defaults to None.
            python_api_v3_cfg (dict | None, optional):
                Python API V3 configuration dictionary. Defaults to None.
            max_workers (int, optional):
                Maximum number of worker threads. Defaults to 4.
            enable_cors (bool, optional):
                Whether to enable CORS cross-origin support. Defaults to False.
            host (str, optional):
                Server listening host address. Defaults to '0.0.0.0'.
            port (int, optional):
                Server listening port number. Defaults to 80.
            startup_event_listener (None | list, optional):
                List of startup event listeners. Defaults to None.
            shutdown_event_listener (None | list, optional):
                List of shutdown event listeners. Defaults to None.
            logger_cfg (None | dict, optional):
                Logger configuration dictionary. Defaults to None.

        Raises:
            ValueError:
                Raised when both python_api_v2_cfg and python_api_v3_cfg are None.
        """
        Super.__init__(self, logger_cfg=logger_cfg)
        self.python_api_v2_cfg = python_api_v2_cfg
        self.python_api_v3_cfg = python_api_v3_cfg
        if self.python_api_v2_cfg is None and self.python_api_v3_cfg is None:
            msg = 'At least one of python_api_v2_cfg and python_api_v3_cfg is required.'
            self.logger.error(msg)
            raise ValueError(msg)
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.python_api_v2: StreamingSpeech2MotionV2 | None = None
        self.python_api_v3: StreamingSpeech2MotionV3 | None = None
        # for tailing the log file
        log_path = None
        for logger_handler in self.logger.handlers:
            if hasattr(logger_handler, "baseFilename"):
                log_path = logger_handler.baseFilename
                break
        self.log_path = log_path
        self.templates = Jinja2Templates(directory="templates")
        self.app = FastAPI()
        self.enable_cors = enable_cors
        if self.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=['*'],
                allow_credentials=True,
                allow_methods=['*'],
                allow_headers=['*'],
            )
        self.app.add_event_handler('startup', self._build_python_api)
        if startup_event_listener is not None:
            for listener in startup_event_listener:
                self.app.add_event_handler('startup', listener)
        if shutdown_event_listener is not None:
            for listener in shutdown_event_listener:
                self.app.add_event_handler('shutdown', listener)
        register_error_handlers(self.app)
        self.app.middleware('http')(self._print_request_id)
        self.asyncio_tasks = set()
        if not v3_pb2_imported:
            msg = 'Failed to import protobuf module, ' +\
                'PB-related interfaces cannot be used, error details:\n'
            msg += v3_pb2_traceback_str
            self.logger.warning(msg)

    async def _build_python_api(self):
        """Build Python API instances.

        Constructs V2 and V3 Python API instances based on configuration.
        """
        if self.python_api_v2_cfg is not None:
            python_api_cfg = self.python_api_v2_cfg.copy()
            python_api_cfg['thread_pool_executor'] = self.executor
            python_api_cfg['logger_cfg'] = self.logger_cfg
            self.python_api_v2 = await build_api(python_api_cfg)
        if self.python_api_v3_cfg is not None:
            python_api_cfg = self.python_api_v3_cfg.copy()
            python_api_cfg['thread_pool_executor'] = self.executor
            python_api_cfg['logger_cfg'] = self.logger_cfg
            self.python_api_v3 = await build_api(python_api_cfg)

    async def _print_request_id(self, request: Request, call_next):
        """Print request ID and request information.

        Args:
            request (Request):
                FastAPI request object.
            call_next:
                Next middleware or route handler function.

        Returns:
            Response:
                Processed response object.
        """
        # Print x-request-id for tracking request
        x_request_id = request.headers.get('x-request-id')
        start = time.time()
        response = await call_next(request)
        text = f'{request.method} {request.url.path} {response.status_code} ' +\
            f'cost: {time.time() - start:.3f}s'

        if x_request_id:
            text = f'x-request-id:{x_request_id} ' + text
        self.logger.debug(text)
        return response

    def _add_api_routes(self, router: APIRouter) -> None:
        """Add API routes to the router.

        Args:
            router (APIRouter):
                FastAPI router object.
        """
        # GET
        router.add_api_route(
            "/",
            self.root,
            methods=["GET"],
        )
        router.add_api_route(
            '/health',
            endpoint=self.health,
            status_code=200,
            methods=['GET'],
        )
        router.add_api_route(
            "/tail_log/{n_lines}",
            self.tail_log,
            methods=["GET"],
            status_code=200,
            response_model=str,
        )
        router.add_api_route(
            "/dowload_log_file",
            self.dowload_log_file,
            methods=["GET"],
            status_code=200,
        )
        # POST
        router.add_api_route(
            '/api/v1/streaming_speech2motion/chunk_start',
            endpoint=self.streaming_speech2motion_v1_chunk_start,
            status_code=200,
            methods=['POST'],
            responses={
                200: {
                    'content': {
                        'application/octet-stream': {}
                    }
                }
            }
        )
        router.add_api_route(
            '/api/v1/streaming_speech2motion/chunk_body',
            endpoint=self.streaming_speech2motion_v1_chunk_body,
            status_code=200,
            methods=['POST'],
            response_model=LogResponse,
            responses={
                200: {
                    'content': {
                        'application/octet-stream': {}
                    }
                }
            }
        )
        router.add_api_route(
            '/api/v1/streaming_speech2motion/chunk_end',
            endpoint=self.streaming_speech2motion_v1_chunk_end,
            status_code=200,
            methods=['POST'],
            response_model=LogResponse,
            responses={
                200: {
                    'content': {
                        'application/octet-stream': {}
                    }
                }
            }
        )
        # Websocket
        router.add_api_websocket_route(
            "/api/v2/streaming_speech2motion/ws",
            endpoint=self.streaming_speech2motion_v2_ws,
        )
        router.add_api_websocket_route(
            "/api/v3/streaming_speech2motion/ws",
            endpoint=self.streaming_speech2motion_v3_ws,
        )

    def run(self) -> None:
        """Run the FastAPI server.

        Starts the server based on configuration, listening on the specified
        host and port.
        """
        router = APIRouter()
        self._add_api_routes(router)
        self.app.include_router(router)
        uvicorn.run(self.app, host=self.host, port=self.port)

    def root(self) -> RedirectResponse:
        """Redirect to API documentation page.

        Returns:
            RedirectResponse:
                Response object that redirects to /docs.
        """
        return RedirectResponse(url="/docs")

    async def dowload_log_file(self) -> Response:
        """Download log file.

        Returns:
            Response:
                Response object containing log file content.

        Raises:
            NoLogFileException:
                Raised when no log file is found.
        """
        if self.log_path is None:
            msg = "No log file found."
            self.logger.error(msg)
            raise NoLogFileException(status_code=503, detail=msg)
        with open(self.log_path, "rb") as f:
            resp = Response(content=f.read(), media_type="application/octet-stream")
            base_name = os.path.basename(self.log_path)
            resp.headers["Content-Disposition"] = f"attachment; filename={base_name}"
            return resp

    async def tail_log(self, request: Request, n_lines: int) -> str:
        """Return the last n lines of the log file.

        Args:
            request (Request):
                FastAPI request object.
            n_lines (int):
                Maximum number of lines to return.

        Returns:
            str:
                Rendered HTML template content.

        Raises:
            NoLogFileException:
                Raised when no log file is found.
        """
        if self.log_path is None:
            msg = "No log file found."
            self.logger.error(msg)
            raise NoLogFileException(status_code=503, detail=msg)
        # read the last n_lines lines from the log file
        with open(self.log_path, encoding="utf-8") as f:
            lines = f.readlines()
            n_lines = min(int(n_lines), len(lines))
            log_content = "".join(lines[-n_lines:])
        # Render template and return
        return self.templates.TemplateResponse(
            "log_template.html", {"request": request, "log_content": log_content})

    async def streaming_speech2motion_v1_chunk_start(
        self,
        item: StreamingSpeech2MotionV1ChunkStart,
    ) -> Response:
        """Handle StreamingSpeech2MotionV1ChunkStart request.

        Args:
            item (StreamingSpeech2MotionV1ChunkStart):
                V1 streaming speech-to-motion start request.

        Returns:
            Response:
                HTTP 200 response.
        """
        await self.python_api.handle_chunk_start(item)
        return Response(status_code=200)

    async def streaming_speech2motion_v1_chunk_body(
        self,
        item: StreamingSpeech2MotionV1ChunkBody,
    ) -> LogResponse | Response:
        """Handle StreamingSpeech2MotionV1ChunkBody request.

        Args:
            item (StreamingSpeech2MotionV1ChunkBody):
                V1 streaming speech-to-motion chunk body request.

        Returns:
            LogResponse | Response:
                Log response or binary response containing motion data.
        """
        body_data = await self.python_api_v2.handle_chunk_body(item)
        # Return MotionClip as npz file
        if isinstance(body_data, dict):
            start = time.time()
            loop = asyncio.get_event_loop()
            npz_io = io.BytesIO()
            await loop.run_in_executor(
                self.executor,
                export_npz,
                body_data,
                npz_io
            )
            cost = time.time() - start
            msg = f'Time to export generated motion to npz bytes: {cost:.3f}s'
            self.logger.debug(msg)
            return Response(
                content=npz_io.getvalue(),
                media_type="application/octet-stream")
        elif isinstance(body_data, MotionClip):
            start = time.time()
            loop = asyncio.get_event_loop()
            npz_io = await loop.run_in_executor(
                self.executor,
                body_data.to_npz,
            )
            cost = time.time() - start
            msg = f'Time to export generated motion to npz bytes: {cost:.3f}s'
            self.logger.debug(msg)
            return Response(
                content=npz_io.getvalue(),
                media_type="application/octet-stream")
        # Return log string
        else:
            log_response = LogResponse(log=body_data)
            return log_response

    async def streaming_speech2motion_v1_chunk_end(
        self,
        item: StreamingSpeech2MotionV1ChunkEnd,
    ) -> LogResponse | Response:
        """Handle StreamingSpeech2MotionV1ChunkEnd request.

        Args:
            item (StreamingSpeech2MotionV1ChunkEnd):
                V1 streaming speech-to-motion chunk end request.

        Returns:
            LogResponse | Response:
                Log response or binary response containing motion data.
        """
        end_data = await self.python_api_v2.handle_chunk_end(item)
        # Return None
        if end_data is None:
            return Response(status_code=200)
        # Return MotionClip as npz file
        elif isinstance(end_data, dict):
            loop = asyncio.get_event_loop()
            npz_io = io.BytesIO()
            await loop.run_in_executor(
                self.executor,
                export_npz,
                end_data,
                npz_io
            )
            return Response(
                content=npz_io.getvalue(), media_type="application/octet-stream")
        elif isinstance(end_data, MotionClip):
            start = time.time()
            loop = asyncio.get_event_loop()
            npz_io = await loop.run_in_executor(
                self.executor,
                end_data.to_npz,
            )
            cost = time.time() - start
            msg = f'Time to export generated motion to npz bytes: {cost:.3f}s'
            self.logger.debug(msg)
            return Response(
                content=npz_io.getvalue(),
                media_type="application/octet-stream")
        # Return log string
        else:
            log_response = LogResponse(log=end_data)
            return log_response

    async def streaming_speech2motion_v2_ws(
        self,
        websocket: WebSocket,
    ):
        """Handle V2 WebSocket streaming speech-to-motion connection.

        Args:
            websocket (WebSocket):
                WebSocket connection object.
        """
        stream_ended = False
        await websocket.accept()
        try:
            loop = asyncio.get_event_loop()
            pb_bytes = await websocket.receive_bytes()
            pb_request = streaming_v3_pb2.Speech2MotionV3Request()
            await loop.run_in_executor(
                self.executor,
                pb_request.ParseFromString,
                pb_bytes
            )
            if pb_request.class_name != 'StreamingSpeech2MotionV2ChunkStart':
                msg = 'Expected StreamingSpeech2MotionV2ChunkStart, ' +\
                    f'but received class_name: {pb_request.class_name}'
                self.logger.error(msg)
                await websocket.close(code=1008, reason=msg)
                return
            request_id = pb_request.request_id
            if not pb_request.HasField('memory_duration_override_value'):
                memory_duration_override = None
            else:
                memory_duration_override = pb_request.memory_duration_override_value
            if not pb_request.HasField('first_body_fast_response_override_value'):
                first_body_fast_response_override = None
            else:
                first_body_fast_response_override = \
                    pb_request.first_body_fast_response_override_value
            if not pb_request.HasField('response_chunk_n_frames_value'):
                response_chunk_n_frames = None
            else:
                response_chunk_n_frames = \
                    pb_request.response_chunk_n_frames_value
            app_name = pb_request.app_name \
                if len(pb_request.app_name) > 0 else 'python_backend'
            item = StreamingSpeech2MotionV2ChunkStart(
                request_id=request_id,
                user_id=pb_request.user_id,
                avatar=pb_request.avatar,
                app_name=app_name,
                max_front_extension_duration=pb_request.max_front_extension_duration,
                max_rear_extension_duration=pb_request.max_rear_extension_duration,
                memory_duration_override=memory_duration_override,
                first_body_fast_response_override=first_body_fast_response_override,
                return_content=pb_request.return_content,
                idle_long_extendable=pb_request.idle_long_extendable
            )
            await self.python_api_v2.handle_chunk_start(item)
            first_chunk = True
            while True:
                pb_bytes = await websocket.receive_bytes()
                request_start_time = time.time()
                pb_request = streaming_v3_pb2.Speech2MotionV3Request()
                await loop.run_in_executor(
                    self.executor,
                    pb_request.ParseFromString,
                    pb_bytes
                )
                if pb_request.class_name == 'StreamingSpeech2MotionV2ChunkBody':
                    if len(pb_request.speech_time) > 0:
                        speech_time = []
                        speech_time.extend(
                            (pb_speech_time.char_index, pb_speech_time.start_time)
                            for pb_speech_time in pb_request.speech_time
                        )
                    else:
                        speech_time = None
                    if len(pb_request.motion_keywords) > 0:
                        motion_keywords = []
                        motion_keywords.extend(
                            (pb_motion_keyword.start_char_index,
                             pb_motion_keyword.keyword)
                            for pb_motion_keyword in pb_request.motion_keywords
                        )
                    else:
                        motion_keywords = None
                    if len(pb_request.label_expression) > 0:
                        label_expression = pb_request.label_expression
                    else:
                        label_expression = None
                    item = StreamingSpeech2MotionV2ChunkBody(
                        request_id=request_id,
                        duration=pb_request.duration,
                        speech_text=pb_request.speech_text,
                        sequence_number=pb_request.sequence_number,
                        speech_time=speech_time,
                        motion_keywords=motion_keywords,
                        label_expression=label_expression
                    )
                    await self._handle_chunk_and_respond_ws(
                            self.python_api_v2.handle_chunk_body,
                            item,
                            websocket,
                            first_chunk=first_chunk,
                            response_chunk_n_frames=response_chunk_n_frames
                        )
                    request_end_time = time.time()
                    time_diff = request_end_time - request_start_time
                    msg = 'Time to process StreamingSpeech2MotionV2ChunkBody: ' +\
                        f'{time_diff:.3f}s'
                    self.logger.debug(msg)
                    first_chunk = False
                elif pb_request.class_name == 'StreamingSpeech2MotionV2ChunkEnd':
                    item = StreamingSpeech2MotionV2ChunkEnd(
                        request_id=request_id
                    )
                    await self._handle_chunk_and_respond_ws(
                            self.python_api_v2.handle_chunk_end,
                            item,
                            websocket,
                            first_chunk=first_chunk,
                            response_chunk_n_frames=response_chunk_n_frames
                        )
                    request_end_time = time.time()
                    time_diff = request_end_time - request_start_time
                    msg = 'Time to process StreamingSpeech2MotionV2ChunkEnd: ' +\
                        f'{time_diff:.3f}s'
                    self.logger.debug(msg)
                    response_inst = Speech2MotionV2ResponseChunkEnd(
                        request_id=item.request_id)
                    class_name = 'Speech2MotionV2ResponseChunkEnd'
                    pb_response = streaming_v3_pb2.Speech2MotionV3Response()
                    pb_response.class_name = class_name
                    pb_response.request_id = response_inst.request_id
                    pb_response_bytes = await loop.run_in_executor(
                        self.executor,
                        pb_response.SerializeToString
                    )
                    await websocket.send_bytes(pb_response_bytes)
                    stream_ended = True
                else:
                    msg = 'Expected StreamingSpeech2MotionV2ChunkBody or ' +\
                        'StreamingSpeech2MotionV2ChunkEnd, ' +\
                        f'but received class_name: {pb_request.class_name}'
                    self.logger.error(msg)
                    await websocket.close(code=1008, reason=msg)
                    return
        except WebSocketDisconnect:
            msg = f"Connection with request ID {request_id} was disconnected by user."
            if stream_ended:
                self.logger.info(msg)
            else:
                msg = msg[:-1] + ", but streaming generation did not end normally."
                self.logger.warning(msg)

    async def streaming_speech2motion_v3_ws(
        self,
        websocket: WebSocket,
    ):
        """Handle V3 WebSocket streaming speech-to-motion connection.

        Args:
            websocket (WebSocket):
                WebSocket connection object.
        """
        stream_ended = False
        await websocket.accept()
        try:
            loop = asyncio.get_event_loop()
            pb_bytes = await websocket.receive_bytes()
            pb_request = streaming_v3_pb2.Speech2MotionV3Request()
            await loop.run_in_executor(
                self.executor,
                pb_request.ParseFromString,
                pb_bytes
            )
            if pb_request.class_name != 'StreamingSpeech2MotionV3ChunkStart':
                msg = 'Expected StreamingSpeech2MotionV3ChunkStart, ' +\
                    f'but received class_name: {pb_request.class_name}'
                self.logger.error(msg)
                await websocket.close(code=1008, reason=msg)
                return
            request_id = pb_request.request_id
            if not pb_request.HasField('memory_duration_override_value'):
                memory_duration_override = None
            else:
                memory_duration_override = pb_request.memory_duration_override_value
            if not pb_request.HasField('response_chunk_n_frames_value'):
                response_chunk_n_frames = None
            else:
                response_chunk_n_frames = \
                    pb_request.response_chunk_n_frames_value
            app_name = pb_request.app_name \
                if len(pb_request.app_name) > 0 else 'python_backend'
            item = StreamingSpeech2MotionV3ChunkStart(
                request_id=request_id,
                user_id=pb_request.user_id,
                avatar=pb_request.avatar,
                app_name=app_name,
                max_front_extension_duration=pb_request.max_front_extension_duration,
                max_rear_extension_duration=pb_request.max_rear_extension_duration,
                memory_duration_override=memory_duration_override,
            )
            await self.python_api_v3.handle_chunk_start(item)
            first_chunk = True
            while True:
                pb_bytes = await websocket.receive_bytes()
                request_start_time = time.time()
                pb_request = streaming_v3_pb2.Speech2MotionV3Request()
                await loop.run_in_executor(
                    self.executor,
                    pb_request.ParseFromString,
                    pb_bytes
                )
                if pb_request.class_name == 'StreamingSpeech2MotionV3ChunkBody':
                    if len(pb_request.speech_time) > 0:
                        speech_time = []
                        speech_time.extend(
                            (pb_speech_time.char_index, pb_speech_time.start_time)
                            for pb_speech_time in pb_request.speech_time
                        )
                    else:
                        speech_time = None
                    if len(pb_request.motion_keywords) > 0:
                        motion_keywords = []
                        motion_keywords.extend(
                            (pb_motion_keyword.start_char_index,
                             pb_motion_keyword.keyword)
                            for pb_motion_keyword in pb_request.motion_keywords
                        )
                    else:
                        motion_keywords = None
                    if len(pb_request.label_expression) > 0:
                        label_expression = pb_request.label_expression
                    else:
                        label_expression = None
                    item = StreamingSpeech2MotionV3ChunkBody(
                        request_id=request_id,
                        duration=pb_request.duration,
                        speech_text=pb_request.speech_text,
                        sequence_number=pb_request.sequence_number,
                        speech_time=speech_time,
                        motion_keywords=motion_keywords,
                        label_expression=label_expression
                    )
                    await self._handle_chunk_and_respond_ws(
                            self.python_api_v3.handle_chunk_body,
                            item,
                            websocket,
                            first_chunk=first_chunk,
                            response_chunk_n_frames=response_chunk_n_frames
                        )
                    request_end_time = time.time()
                    time_diff = request_end_time - request_start_time
                    msg = 'Time to process StreamingSpeech2MotionV3ChunkBody: ' +\
                        f'{time_diff:.3f}s'
                    self.logger.debug(msg)
                    first_chunk = False
                elif pb_request.class_name == 'StreamingSpeech2MotionV3ChunkEnd':
                    item = StreamingSpeech2MotionV3ChunkEnd(
                        request_id=request_id
                    )
                    await self._handle_chunk_and_respond_ws(
                            self.python_api_v3.handle_chunk_end,
                            item,
                            websocket,
                            first_chunk=first_chunk,
                            response_chunk_n_frames=response_chunk_n_frames
                        )
                    request_end_time = time.time()
                    time_diff = request_end_time - request_start_time
                    msg = 'Time to process StreamingSpeech2MotionV2ChunkEnd: ' +\
                        f'{time_diff:.3f}s'
                    self.logger.debug(msg)
                    response_inst = Speech2MotionV3ResponseChunkEnd(
                        request_id=item.request_id)
                    class_name = 'Speech2MotionV3ResponseChunkEnd'
                    pb_response = streaming_v3_pb2.Speech2MotionV3Response()
                    pb_response.class_name = class_name
                    pb_response.request_id = response_inst.request_id
                    pb_response_bytes = await loop.run_in_executor(
                        self.executor,
                        pb_response.SerializeToString
                    )
                    await websocket.send_bytes(pb_response_bytes)
                    stream_ended = True
                else:
                    msg = 'Expected StreamingSpeech2MotionV3ChunkBody or ' +\
                        'StreamingSpeech2MotionV3ChunkEnd, ' +\
                        f'but received class_name: {pb_request.class_name}'
                    self.logger.error(msg)
                    await websocket.close(code=1008, reason=msg)
                    return
        except WebSocketDisconnect:
            msg = f"Connection with request ID {request_id} was disconnected by user."
            if stream_ended:
                self.logger.info(msg)
            else:
                msg = msg[:-1] + ", but streaming generation did not end normally."
                self.logger.warning(msg)


    async def health(self) -> JSONResponse:
        """Health check endpoint.

        Returns:
            JSONResponse:
                JSON response containing 'OK'.
        """
        resp = JSONResponse(content='OK')
        return resp

    async def _handle_chunk_and_respond_ws(
            self,
            python_api_fn: Callable,
            item: StreamingSpeech2MotionV2ChunkBody |
                  StreamingSpeech2MotionV2ChunkEnd |
                  StreamingSpeech2MotionV3ChunkBody |
                  StreamingSpeech2MotionV3ChunkEnd,
            websocket: WebSocket,
            first_chunk: bool = True,
            response_chunk_n_frames: int | None = None,
    ) -> bool:
        """Handle chunk body/end and return response to websocket client.

        Args:
            python_api_fn (Callable):
                Python API function for handling chunk body/end asynchronously.
            item (StreamingSpeech2MotionV2ChunkBody |
                  StreamingSpeech2MotionV2ChunkEnd |
                  StreamingSpeech2MotionV3ChunkBody |
                  StreamingSpeech2MotionV3ChunkEnd):
                Chunk object to be processed.
            websocket (WebSocket):
                WebSocket connection object.
            first_chunk (bool, optional):
                Whether this is the first chunk. If True, joint_names and restpose_name
                will be returned, otherwise these data are omitted to save bandwidth.
                Defaults to True.
            response_chunk_n_frames (int | None, optional):
                Maximum number of frames for returned chunks. A single generated
                MotionClip
                will be sliced into multiple chunks not exceeding
                response_chunk_n_frames.
                If None, returns the entire MotionClip. Defaults to None.

        Returns:
            bool:
                Whether processing was successful.
        """
        ret_data = await python_api_fn(item)
        loop = asyncio.get_event_loop()
        if isinstance(item, StreamingSpeech2MotionV2ChunkBody) or \
                isinstance(item, StreamingSpeech2MotionV2ChunkEnd):
            version_str = 'V2'
        elif isinstance(item, StreamingSpeech2MotionV3ChunkBody) or \
                isinstance(item, StreamingSpeech2MotionV3ChunkEnd):
            version_str = 'V3'
        else:
            msg = f'Incorrect item parameter type: {type(item)}, please check.'
            self.logger.error(msg)
            msg = 'Internal server error.'
            await websocket.close(code=1008, reason=msg)
            return False
        if ret_data is None:
            return False
        elif isinstance(ret_data, MotionClip):
            start_time = time.time()
            # Send ChunkStart before first ChunkBody
            if first_chunk:
                pb_response = streaming_v3_pb2.Speech2MotionV3Response()
                pb_response.class_name = f'Speech2Motion{version_str}ResponseChunkStart'
                pb_response.request_id = item.request_id
                pb_response.joint_names.extend(ret_data.joint_names)
                pb_response.restpose_name = ret_data.restpose_name
                pb_response.dtype = str(ret_data.joint_rotmat.dtype)
                if ret_data.timeline_start_idx is None:
                    pb_response.timeline_start_idx_is_none = True
                else:
                    pb_response.timeline_start_idx_value = ret_data.timeline_start_idx
                if ret_data.blendshape_names is not None:
                    pb_response.blendshape_names.extend(ret_data.blendshape_names)
                pb_response_bytes = await loop.run_in_executor(
                    self.executor,
                    pb_response.SerializeToString
                )
                await websocket.send_bytes(pb_response_bytes)
            # slice MotionClip and send ChunkBody
            if response_chunk_n_frames is None:
                response_chunk_n_frames = ret_data.n_frames
            for start_idx in range(0, ret_data.n_frames, response_chunk_n_frames):
                end_idx = min(start_idx + response_chunk_n_frames, ret_data.n_frames)
                if start_idx == 0 and end_idx == ret_data.n_frames:
                    mc_chunk = ret_data
                else:
                    mc_chunk = ret_data.slice(start_idx, end_idx)
                pb_response = streaming_v3_pb2.Speech2MotionV3Response()
                pb_response.class_name = f'Speech2Motion{version_str}ResponseChunkBody'
                data_ndarray = await loop.run_in_executor(
                    self.executor,
                    self._convert_motion_clip_to_flat_bytes,
                    mc_chunk
                )
                pb_response.data = data_ndarray
                pb_response_bytes = await loop.run_in_executor(
                    self.executor,
                    pb_response.SerializeToString
                )
                await websocket.send_bytes(pb_response_bytes)
                if start_idx == 0:
                    cost = time.time() - start_time
                    msg = 'Time to export motion to first protobuf format and ' +\
                        f'send to client: {cost:.3f}s'
                    self.logger.debug(msg)
        elif isinstance(ret_data, str):
            response_inst = LogResponse(log=ret_data)
            class_name = 'LogResponse'
            pb_response = streaming_v3_pb2.Speech2MotionV3Response()
            pb_response.class_name = class_name
            pb_response.request_id = item.request_id
            pb_response.log = response_inst.log
            pb_response_bytes = await loop.run_in_executor(
                self.executor,
                pb_response.SerializeToString
            )
            await websocket.send_bytes(pb_response_bytes)
        else:
            msg = f'Incorrect return data type: {type(ret_data)}, ' +\
                'please check return data type.'
            self.logger.error(msg)
            msg = 'Internal server error, incorrect return data type.'
            await websocket.close(code=1008, reason=msg)
        return True

    def _convert_motion_clip_to_flat_bytes(self, motion_clip: MotionClip) -> bytes:
        """Convert numpy arrays in MotionClip to flattened byte data.

        Args:
            motion_clip (MotionClip):
                Motion clip object to be converted.

        Returns:
            bytes:
                Flattened byte data.
        """
        dtype = motion_clip.joint_rotmat.dtype
        n_frames = motion_clip.n_frames
        flat_joint_rotmat = motion_clip.joint_rotmat.reshape(n_frames, -1)
        flat_root_world_position = motion_clip.root_world_position.reshape(n_frames, -1)
        # n_frames * (cutoff_enabled, left_priority, right_priority)
        cutoff_marks = np.zeros((n_frames, 3), dtype=dtype)
        if motion_clip.cutoff_frames is not None:
            for cutoff_frame in motion_clip.cutoff_frames:
                frame_idx = cutoff_frame[0]
                left_priority = cutoff_frame[1]
                right_priority = cutoff_frame[2]
                cutoff_marks[frame_idx] = [1, left_priority, right_priority]
        if motion_clip.cutoff_ranges is not None:
            for cutoff_range in motion_clip.cutoff_ranges:
                start_frame_idx = cutoff_range[0]
                end_frame_idx = cutoff_range[1]
                left_priority = cutoff_range[2]
                right_priority = cutoff_range[3]
                cutoff_marks[start_frame_idx:end_frame_idx] = \
                    [1, left_priority, right_priority]
        ndarray_list = [
            flat_joint_rotmat,
            flat_root_world_position,
            cutoff_marks
        ]
        if motion_clip.blendshape_names is not None:
            flat_blendshape_values = motion_clip.blendshape_values.reshape(n_frames, -1)
            ndarray_list.append(flat_blendshape_values)
        flat_bytes = np.concatenate(ndarray_list, axis=1, dtype=dtype)
        return flat_bytes.tobytes()
