import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..data_structures.motion_clip import MotionClip
from ..data_structures.motion_record import MotionRecord
from .base_merge import BaseMerge


class Concatenation(BaseMerge):
    """Motion clip concatenation merge class.

    This class concatenates multiple MotionClip objects sequentially,
    creating a single continuous motion sequence from multiple clips.
    """

    def __init__(self,
                 max_workers: int = 1,
                 thread_pool_executor: ThreadPoolExecutor | None = None,
                 logger_cfg: None | dict = None) -> None:
        """Initialize the motion clip concatenation merge class.

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
        super().__init__(max_workers, thread_pool_executor, logger_cfg)

    async def merge(self,
              motion_clips: list[MotionClip]) -> MotionClip:
        """Concatenate multiple MotionClip objects.

        Args:
            motion_clips (list[MotionClip]):
                List of MotionClip objects to be concatenated.

        Returns:
            MotionClip:
                The concatenated MotionClip object.
        """
        motion_clips_to_merge = motion_clips.copy()
        loop = asyncio.get_running_loop()
        while len(motion_clips_to_merge) > 1:
            coroutines = []
            for i in range(0, len(motion_clips_to_merge), 2):
                if i + 1 < len(motion_clips_to_merge):
                    coroutine = loop.run_in_executor(
                        self.executor,
                        self._merge_task,
                        motion_clips_to_merge[i],
                        motion_clips_to_merge[i + 1]
                    )
                    coroutines.append(coroutine)
            merge_results = await asyncio.gather(*coroutines)
            result_list = merge_results.copy()
            if len(motion_clips_to_merge) % 2 == 1:
                result_list.append(motion_clips_to_merge[-1])
            motion_clips_to_merge = result_list
        return motion_clips_to_merge[0]

    def _merge_task(self,
                    motion_clip_a: MotionClip,
                    motion_clip_b: MotionClip) -> MotionClip:
        """Concatenate two MotionClip objects.

        Args:
            motion_clip_a (MotionClip):
                First MotionClip to be concatenated.
            motion_clip_b (MotionClip):
                Second MotionClip to be concatenated.

        Returns:
            MotionClip:
                The concatenated MotionClip object.
        """
        merged_motion_clip = MotionClip.concat(
            [motion_clip_a, motion_clip_b]
        )
        return merged_motion_clip

    async def predict_merge_n_frames(self,
                                     motion_records: list[MotionRecord]) -> int:
        """Predict the total number of frames after concatenating MotionRecord objects.

        Args:
            motion_records (list[MotionRecord]):
                List of MotionRecord objects to be concatenated.

        Returns:
            int:
                Total number of frames in the concatenated motion sequence.
        """
        n_frames = 0
        for motion_record in motion_records:
            n_frames += motion_record.n_frames
        return n_frames
