from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from ..data_structures.motion_clip import MotionClip
from ..data_structures.motion_record import MotionRecord
from ..utils.super import Super


class BaseMerge(Super, ABC):
    """Base class for merging MotionClip objects.

    This abstract base class provides the interface for merging multiple
    MotionClip objects into a single MotionClip. It includes thread pool
    management for concurrent processing and prediction of merged frame counts.
    """

    def __init__(self,
                 max_workers: int = 1,
                 thread_pool_executor: ThreadPoolExecutor | None = None,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the base merge class.

        Args:
            max_workers (int, optional):
                Maximum number of worker threads for concurrent processing.
                Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                External thread pool executor to use. If None, creates a new
                one with max_workers. Defaults to None.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        ABC.__init__(self)
        Super.__init__(self, logger_cfg)
        self.max_workers = max_workers
        self.executor = thread_pool_executor \
            if thread_pool_executor is not None \
            else ThreadPoolExecutor(max_workers=max_workers)
        self.executor_external = True \
            if thread_pool_executor is not None \
            else False

    def __del__(self) -> None:
        """Destructor, cleanup thread pool executor.

        Automatically shuts down the thread pool executor if it was created
        internally (not provided externally).
        """
        if not self.executor_external:
            self.executor.shutdown(wait=True)

    @abstractmethod
    async def merge(self,
              motion_clips: list[MotionClip]) -> MotionClip:
        """Merge multiple MotionClip objects into a single MotionClip.

        Args:
            motion_clips (list[MotionClip]):
                List of MotionClip objects to be merged.

        Returns:
            MotionClip:
                The merged MotionClip object containing all input clips.
        """
        pass

    @abstractmethod
    async def predict_merge_n_frames(self,
                                     motion_records: list[MotionRecord]) -> int:
        """Predict the total number of frames after merging MotionRecord objects.

        Args:
            motion_records (list[MotionRecord]):
                List of MotionRecord objects to be merged.

        Returns:
            int:
                Total number of frames in the merged motion sequence.
        """
        pass
