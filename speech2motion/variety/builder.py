from .local_memory import LocalMemory

_MEMORY = dict(
    LocalMemory=LocalMemory)


def build_memory(cfg: dict) -> LocalMemory:
    """Build a memory instance from a configuration dictionary."""
    cfg = cfg.copy()
    cls_name = cfg.pop('type')
    if cls_name not in _MEMORY:
        msg = f'Unknown memory type: {cls_name}'
        raise TypeError(msg)
    ret_inst = _MEMORY[cls_name](**cfg)
    return ret_inst
