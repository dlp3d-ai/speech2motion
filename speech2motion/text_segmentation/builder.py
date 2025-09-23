from .jieba_text_segmentation import JiebaTextSegmentation

_TEXT_SEGMENTATIONS = dict(
    JiebaTextSegmentation=JiebaTextSegmentation,
)


def build_text_segmentation(cfg: dict) -> JiebaTextSegmentation:
    """Build a text segmentation instance from configuration dictionary.

    Args:
        cfg (dict):
            Configuration dictionary containing segmentation type and parameters.
            Must include 'type' field specifying the segmentation type.

    Returns:
        JiebaTextSegmentation:
            Text segmentation instance built according to configuration.

    Raises:
        TypeError:
            Raised when the specified segmentation type is not in the supported
            type list.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop('type')
    if cls_name not in _TEXT_SEGMENTATIONS:
        msg = f'Unknown text segmentation type: {cls_name}'
        raise TypeError(msg)
    ret_inst = _TEXT_SEGMENTATIONS[cls_name](**cfg)
    return ret_inst

