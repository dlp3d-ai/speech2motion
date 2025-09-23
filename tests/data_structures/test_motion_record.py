import pytest

from speech2motion.data_structures.annotations.loopable import Loopable
from speech2motion.data_structures.annotations.motion_keyword import MotionKeyword
from speech2motion.data_structures.annotations.random import Random
from speech2motion.data_structures.annotations.speech_keyword import SpeechKeyword
from speech2motion.data_structures.motion_record import MotionRecord


@pytest.fixture
def motion_record():
    """Fixture to create a MotionRecord instance for testing."""
    return MotionRecord(
        motion_record_id=1,
        n_frames=10,
        startup_frame=2,
        recovery_frame=8,
        avatar_name="Avatar1",
        is_idle_long=False,
        labels=["label1", "label2"],
        fps=30.0
    )

def test_init(motion_record):
    """Test the initialization of MotionRecord."""
    assert motion_record.motion_record_id == 1
    assert motion_record.n_frames == 10
    assert motion_record.startup_frame == 2
    assert motion_record.recovery_frame == 8
    assert motion_record.avatar_name == "Avatar1"
    assert motion_record.is_idle_long is False
    assert motion_record.fps == 30.0
    assert motion_record.labels == ["label1", "label2"]

def test_set_loopable(motion_record):
    """Test setting loopable."""
    loopable = Loopable(loop_start_frame=1, loop_end_frame=5)
    motion_record.set_loopable(loopable)
    assert motion_record.loopable == loopable

def test_is_loopable(motion_record):
    """Test checking if motion is loopable."""
    assert not motion_record.is_loopable()
    motion_record.set_loopable(Loopable(loop_start_frame=1, loop_end_frame=5))
    assert motion_record.is_loopable()

def test_set_random(motion_record):
    """Test setting random."""
    random = Random(cutoff_frames=[1, 2], cutoff_ranges=[(1, 5)])
    motion_record.set_random(random)
    assert motion_record.random == random

def test_is_random(motion_record):
    """Test checking if motion is random."""
    assert not motion_record.is_random()
    motion_record.set_random(Random(cutoff_frames=[1, 2], cutoff_ranges=[(1, 5)]))
    assert motion_record.is_random()

def test_set_motion_keyword(motion_record):
    """Test setting motion keyword."""
    motion_keyword = MotionKeyword(
        motion_keywords_ch=["keyword1"], motion_keyword_frame=3)
    motion_record.set_motion_keyword(motion_keyword)
    assert motion_record.motion_keyword == motion_keyword

def test_is_motion_keyword(motion_record):
    """Test checking if motion has a motion keyword."""
    assert not motion_record.is_motion_keyword()
    motion_record.set_motion_keyword(
        MotionKeyword(motion_keywords_ch=["keyword1"], motion_keyword_frame=3))
    assert motion_record.is_motion_keyword()

def test_set_speech_keyword(motion_record):
    """Test setting speech keyword."""
    speech_keyword = SpeechKeyword(
        speech_keywords_ch=["speech1"], speech_keyword_frame=4)
    motion_record.set_speech_keyword(speech_keyword)
    assert motion_record.speech_keyword == speech_keyword

def test_is_speech_keyword(motion_record):
    """Test checking if motion has a speech keyword."""
    assert not motion_record.is_speech_keyword()
    motion_record.set_speech_keyword(
        SpeechKeyword(speech_keywords_ch=["speech1"], speech_keyword_frame=4))
    assert motion_record.is_speech_keyword()

def test_to_dict(motion_record):
    """Test converting MotionRecord to dictionary."""
    motion_record.set_loopable(Loopable(loop_start_frame=1, loop_end_frame=5))
    motion_record.set_random(Random(cutoff_frames=[1, 2], cutoff_ranges=[(1, 5)]))
    motion_record.set_motion_keyword(
        MotionKeyword(motion_keywords_ch=["keyword1"], motion_keyword_frame=3))
    motion_record.set_speech_keyword(
        SpeechKeyword(speech_keywords_ch=["speech1"], speech_keyword_frame=4))

    result = motion_record.to_dict()
    assert result['motion_record_id'] == motion_record.motion_record_id
    assert result['n_frames'] == motion_record.n_frames
    assert result['loopable'] is not None
    assert result['random'] is not None
    assert result['motion_keyword'] is not None
    assert result['speech_keyword'] is not None
    assert result['labels'] == ["label1", "label2"]

def test_from_dict():
    """Test creating MotionRecord from dictionary."""
    data = {
        "motion_record_id": 1,
        "n_frames": 10,
        "startup_frame": 2,
        "recovery_frame": 8,
        "avatar_name": "Avatar1",
        "is_idle_long": False,
        "fps": 30.0,
        "loopable": {"loop_start_frame": 1, "loop_end_frame": 5},
        "random": {"cutoff_frames": [1, 2], "cutoff_ranges": [(1, 5)]},
        "motion_keyword":
            {"motion_keywords_ch": ["keyword1"], "motion_keyword_frame": 3},
        "speech_keyword":
            {"speech_keywords_ch": ["speech1"], "speech_keyword_frame": 4}
    }

    motion_record = MotionRecord.from_dict(data)
    assert motion_record.motion_record_id == 1
    assert motion_record.n_frames == 10
    assert motion_record.loopable is not None
    assert motion_record.random is not None
    assert motion_record.motion_keyword is not None
    assert motion_record.speech_keyword is not None
    assert len(motion_record.labels) == 0
    data["labels"] = ["label1", "label2"]
    motion_record = MotionRecord.from_dict(data)
    assert len(motion_record.labels) == 2
    assert motion_record.labels == ["label1", "label2"]

def test_clone(motion_record):
    """Test cloning MotionRecord."""
    motion_record.set_loopable(Loopable(loop_start_frame=1, loop_end_frame=5))
    motion_record.set_random(Random(cutoff_frames=[1, 2], cutoff_ranges=[(1, 5)]))
    motion_record.set_motion_keyword(
        MotionKeyword(
            motion_keywords_ch=["keyword1"],
            motion_keyword_frame=3))
    motion_record.set_speech_keyword(
        SpeechKeyword(
            speech_keywords_ch=["speech1"],
            speech_keyword_frame=4))

    cloned_record = motion_record.clone()
    assert cloned_record.motion_record_id == motion_record.motion_record_id
    assert cloned_record.loopable is not None
    assert cloned_record.random is not None
    assert cloned_record.motion_keyword is not None
    assert cloned_record.speech_keyword is not None
    assert cloned_record.is_idle_long == motion_record.is_idle_long
    assert cloned_record.fps == motion_record.fps
    assert cloned_record.labels == motion_record.labels
