from typing import Any

from ..data_structures.motion_record import MotionRecord
from ..filters.base_filter import BaseFilter


async def filter_pipeline_retrieve(
    filter_pipeline: list[BaseFilter],
    pipeline_input: dict[str, Any],
    return_candidates: bool,
) -> dict[int, MotionRecord]:
    """Retrieve MotionRecord objects using a pipeline of multiple filters.

    Args:
        filter_pipeline (list[BaseFilter]):
            List containing multiple filters for the pipeline.
        pipeline_input (dict[str, Any]):
            Pipeline input including motion record candidates and
            retrieval keyword parameters.
        return_candidates (bool):
            Whether to return all candidates.
            If False, the returned dictionary length will not exceed 1.

    Returns:
        dict[int, MotionRecord]:
            Dictionary containing retrieved MotionRecord objects,
            with MotionRecord id as key and MotionRecord object as value.
            Returns empty dictionary if no motion records meet the criteria.
    """
    pipeline_len = len(filter_pipeline)
    step_input = pipeline_input.copy()
    for filter_idx, filter in enumerate(filter_pipeline):
        if filter_idx == pipeline_len - 1 and not return_candidates:
            motion_record = await filter.select_one(**step_input)
            if motion_record is not None:
                step_output = {motion_record.motion_record_id: motion_record}
            else:
                step_output = dict()
        else:
            step_output = await filter.filter(**step_input)
        if len(step_output) == 0:
            return step_output
        step_input['motion_records'] = step_output
    return step_output
