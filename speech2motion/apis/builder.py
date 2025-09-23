from .streaming_speech2motion_v1 import StreamingSpeech2MotionV1
from .streaming_speech2motion_v2 import StreamingSpeech2MotionV2
from .streaming_speech2motion_v3 import StreamingSpeech2MotionV3

_API = dict(
    StreamingSpeech2MotionV1=StreamingSpeech2MotionV1,
    StreamingSpeech2MotionV2=StreamingSpeech2MotionV2,
    StreamingSpeech2MotionV3=StreamingSpeech2MotionV3
)


async def build_api(cfg: dict) -> StreamingSpeech2MotionV1:
    """Build an API instance from a configuration dictionary.

    Args:
        cfg (dict):
            Configuration dictionary containing API type and parameters.
            Must include 'type' field specifying the API type.

    Returns:
        StreamingSpeech2MotionV1:
            API instance built according to configuration.

    Raises:
        TypeError:
            Raised when the specified API type is not in the supported
            type list.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop('type')
    if cls_name not in _API:
        msg = f'Unknown api type: {cls_name}'
        raise TypeError(msg)
    ret_inst = _API[cls_name](**cfg)
    await ret_inst.startup()
    return ret_inst
