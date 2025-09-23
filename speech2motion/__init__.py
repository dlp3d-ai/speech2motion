from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("speech2motion")
except PackageNotFoundError:
    __version__ = "unknown version"
